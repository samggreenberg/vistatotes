"""The ``vg_scale`` build's passes, exercised without the 100 GB of VG source.

``build_pile.py`` assembles this dataset in eight passes over the Visual Genome
source. Two of them are where this pile's expensive bugs have actually lived:

* :func:`apply_corrections` is the single point at which a human verdict's box
  crosses from normalised into pixel space. Getting that wrong is #3281 -- the
  region write normalises the already-normalised coordinate, divides it by ~500
  and parks the box on the frame origin, and *nothing downstream can see it*
  because the band is derived from the same corrupted box, so the cell name and
  its contents stay consistent with each other all the way into a published
  study.
* :func:`designate_cells` decides whether a rebuild keeps the images a human has
  already reviewed. Two separate regressions here orphaned reviews (49 of 360
  negatives; 99 of 360 boxed positives) while producing cells that looked
  perfect: right count, right vectors, right geometry.

Both used to be inline blocks in a 304-line function that needs the whole VG
tree to call, so neither had a test. They are ordinary functions now, and these
are the properties their comments claim.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PILE_DIR = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "pile"


@pytest.fixture(scope="module")
def vgs():
    """``pilebuild.loaders.vg_scale``, imported without ``build_pile``'s env setup.

    Importing ``build_pile`` runs ``pile_config.setup_env()``, which edits
    ``sys.meta_path`` and ``sys.path`` process-wide -- which is why
    ``test_pile_box_scan.py`` drives it through a subprocess. The passes under
    test deliberately depend on nothing but ``pile_config``'s constants, so they
    import directly and run in-process.
    """
    if str(_PILE_DIR) not in sys.path:
        sys.path.insert(0, str(_PILE_DIR))
    from pilebuild.loaders import vg_scale

    return vg_scale


@pytest.fixture(scope="module")
def pc():
    if str(_PILE_DIR) not in sys.path:
        sys.path.insert(0, str(_PILE_DIR))
    import pile_config

    return pile_config


def _verdict(present: bool, boxes: list[list[float]] | None = None) -> dict:
    return {"present": present, "boxes": boxes or []}


class TestApplyCorrections:
    """The one place a box changes space (#3281)."""

    def test_normalised_box_becomes_pixels_exactly_once(self, vgs):
        """A reviewer's box survives the pixel round trip bit for bit.

        This is the assertion #3281 would have failed: the correction arrives
        normalised, is scaled up here by the same (W, H) the region write later
        divides by, and must come back out as the reviewer drew it. Skipping the
        conversion leaves a box ~500x too small that still bands consistently.
        """
        drawn = [0.2, 0.4, 0.6, 0.8]
        labels = {7: {}}
        box_dims = {7: (640, 480)}
        vgs.apply_corrections(labels, {(7, "bus"): _verdict(True, [drawn])}, box_dims, set())

        stored = labels[7]["bus"][0]
        assert stored == [0.2 * 640, 0.4 * 480, 0.6 * 640, 0.8 * 480]

        W, H = box_dims[7]
        assert [stored[0] / W, stored[1] / H, stored[2] / W, stored[3] / H] == drawn

    def test_box_is_scaled_by_the_anchored_dims_not_the_vg_copy(self, vgs):
        """COCO-anchored images band in COCO's space, not the VG downscale.

        VG ships 500 px copies of COCO's 640 px originals. ``box_dims`` carries
        whichever space the image's *other* boxes were measured in, and the
        correction has to use that one or the reviewed box lands somewhere the
        VG-derived boxes are not.
        """
        labels = {7: {}}
        vgs.apply_corrections(labels, {(7, "bus"): _verdict(True, [[0.0, 0.0, 0.5, 0.5]])}, {7: (640, 480)}, set())
        assert labels[7]["bus"][0] == [0.0, 0.0, 320.0, 240.0]

    def test_present_without_a_box_is_unbanded_not_a_negative(self, vgs):
        """No size was measured, so no band can be claimed -- and it is not clean."""
        labels = {7: {"bus": [[1.0, 1.0, 2.0, 2.0]]}}
        unbanded = vgs.apply_corrections(labels, {(7, "bus"): _verdict(True)}, {7: (640, 480)}, set())

        assert unbanded == {(7, "bus")}
        assert "bus" not in labels[7]

    def test_absent_verdict_removes_the_class(self, vgs):
        labels = {7: {"bus": [[1.0, 1.0, 2.0, 2.0]], "dog": [[3.0, 3.0, 4.0, 4.0]]}}
        unbanded = vgs.apply_corrections(labels, {(7, "bus"): _verdict(False)}, {7: (640, 480)}, set())

        assert labels[7] == {"dog": [[3.0, 3.0, 4.0, 4.0]]}
        assert unbanded == set()

    def test_every_reviewed_image_becomes_exhaustive(self, vgs):
        """ "Nobody looked" and "somebody looked and saw nothing" must stay distinct."""
        exhaustive: set[int] = set()
        labels = {7: {}, 8: {}}
        vgs.apply_corrections(labels, {(7, "bus"): _verdict(False)}, {7: (640, 480), 8: (640, 480)}, exhaustive)
        assert exhaustive == {7}

    def test_verdict_for_an_unknown_image_is_ignored(self, vgs):
        labels: dict[int, dict] = {}
        assert vgs.apply_corrections(labels, {(99, "bus"): _verdict(True)}, {}, set()) == set()
        assert labels == {}


class TestAnchorToCoco:
    """COCO's exhaustive labels replace VG's, except where the copies disagree."""

    def _truth(self, boxes):
        return {100: {"bus": boxes}}

    def test_anchored_image_takes_cocos_labels_and_cocos_pixel_space(self, vgs, pc):
        labels = {7: {"dog": [[1.0, 1.0, 2.0, 2.0]]}}
        box_dims, exhaustive, n_anchored, n_reframed = vgs.anchor_to_coco(
            labels,
            dims={7: (500, 375)},
            coco_of={7: 100},
            truth=self._truth([[10.0, 10.0, 20.0, 20.0]]),
            coco_dims={100: (640, 480)},
            wanted={"bus", "dog"},
        )
        # REPLACED, not merged: VG's unverifiable `dog` is exactly what the
        # repair is removing.
        assert labels[7] == {"bus": [[10.0, 10.0, 20.0, 20.0]]}
        assert box_dims[7] == (640, 480)
        assert exhaustive == {7} and (n_anchored, n_reframed) == (1, 0)

    def test_reframed_copy_keeps_vgs_own_labels(self, vgs):
        """A transposed copy is a different framing; COCO's box does not describe it."""
        labels = {7: {"dog": [[1.0, 1.0, 2.0, 2.0]]}}
        box_dims, exhaustive, n_anchored, n_reframed = vgs.anchor_to_coco(
            labels,
            dims={7: (375, 500)},  # transposed against COCO's 640x480
            coco_of={7: 100},
            truth=self._truth([[10.0, 10.0, 20.0, 20.0]]),
            coco_dims={100: (640, 480)},
            wanted={"bus", "dog"},
        )
        assert labels[7] == {"dog": [[1.0, 1.0, 2.0, 2.0]]}
        assert box_dims[7] == (375, 500)
        assert exhaustive == set() and (n_anchored, n_reframed) == (0, 1)

    def test_classes_outside_c_are_dropped_from_cocos_labels(self, vgs):
        labels = {7: {}}
        vgs.anchor_to_coco(
            labels,
            dims={7: (640, 480)},
            coco_of={7: 100},
            truth={100: {"bus": [[1.0, 1.0, 2.0, 2.0]], "toaster": [[3.0, 3.0, 4.0, 4.0]]}},
            coco_dims={100: (640, 480)},
            wanted={"bus"},
        )
        assert labels[7] == {"bus": [[1.0, 1.0, 2.0, 2.0]]}

    def test_unanchored_images_keep_vg_dims(self, vgs):
        labels = {7: {}, 8: {}}
        box_dims, _, n_anchored, _ = vgs.anchor_to_coco(
            labels, dims={7: (500, 375), 8: (600, 400)}, coco_of={}, truth={}, coco_dims={}, wanted={"bus"}
        )
        assert box_dims == {7: (500, 375), 8: (600, 400)} and n_anchored == 0


class TestBandFor:
    """The banding rule itself, split out of :func:`band_candidates` for #3616.

    ``audit_band_drift.py`` re-bands the same images off COCO's annotation to
    measure how often VG's own boxes put one in too small a band. A second copy
    of the rule would answer that drift question with a drift of its own, so
    both callers go through this one.
    """

    def _square(self, frac: float) -> list[float]:
        """A box covering *frac* of a 100x100 frame."""
        side = (frac * 10000) ** 0.5
        return [0.0, 0.0, side, side]

    def test_each_band_claims_the_range_it_declares(self, vgs, pc):
        for band, (lo, hi) in pc.BOX_BANDS.items():
            assert vgs.band_for([self._square((lo + hi) / 2)], 100, 100) == band

    def test_scattered_instances_say_so_rather_than_banding(self, vgs):
        """The union describes the scatter, not an object anyone would drag."""
        assert vgs.band_for([[0.0, 0.0, 1.0, 1.0], [99.0, 99.0, 100.0, 100.0]], 100, 100) == vgs.SCATTERED

    def test_a_box_bigger_than_a_region_says_so_rather_than_banding(self, vgs, pc):
        """Above ``MAX_VOTED_AREA`` a box is not a region, it is the image."""
        assert vgs.band_for([self._square(pc.MAX_VOTED_AREA + 0.05)], 100, 100) == vgs.OVERSIZE

    def test_neither_refusal_can_be_mistaken_for_a_band(self, vgs, pc):
        """`band_candidates` filters on exactly this, and so does the audit."""
        assert vgs.SCATTERED not in pc.BOX_BANDS
        assert vgs.OVERSIZE not in pc.BOX_BANDS


class TestBandCandidates:
    def _labels_with_area_fraction(self, pc, frac: float):
        """One image whose single ``bus`` box covers *frac* of a 100x100 frame."""
        side = (frac * 10000) ** 0.5
        return {7: {"bus": [[0.0, 0.0, side, side]]}}, {7: (100, 100)}

    def test_image_lands_in_the_band_its_union_box_falls_in(self, vgs, pc):
        band, (lo, hi) = next(iter(pc.BOX_BANDS.items()))
        labels, box_dims = self._labels_with_area_fraction(pc, (lo + hi) / 2)
        supply, boxes_for, clean = vgs.band_candidates(labels, box_dims, set())

        assert supply["bus"][band] == [7]
        assert boxes_for[(7, pc.scale_cell("bus", band))] == labels[7]["bus"]
        assert clean == []

    def test_scattered_instances_are_excluded_from_every_band(self, vgs, pc):
        """The union box describes the scatter, not the object nobody would drag."""
        labels = {7: {"bus": [[0.0, 0.0, 1.0, 1.0], [99.0, 99.0, 100.0, 100.0]]}}
        supply, boxes_for, clean = vgs.band_candidates(labels, {7: (100, 100)}, set())

        assert all(not ids for bands in supply.values() for ids in bands.values())
        assert boxes_for == {} and clean == []

    def test_image_with_no_class_joins_the_clean_pool(self, vgs):
        supply, _, clean = vgs.band_candidates({7: {}}, {7: (100, 100)}, set())
        assert clean == [7]

    def test_an_unbanded_pair_keeps_the_image_out_of_the_clean_pool(self, vgs, pc):
        """A reviewer said the object IS there; it is not a true negative."""
        cls = pc.SCALE_CLASSES[0]
        _, _, clean = vgs.band_candidates({7: {}}, {7: (100, 100)}, {(7, cls)})
        assert clean == []


class TestVgNameTables:
    """#3605: a class built from one VG spelling turns its own instances into negatives."""

    def test_the_read_set_covers_every_declared_spelling(self, pc):
        """A spelling missing from the read is invisible, not merely unmatched."""
        wanted = pc.scale_vg_wanted()
        assert set(pc.SCALE_CLASSES) <= wanted
        for table in (pc.SCALE_VG_NAMES, pc.SCALE_VG_AMBIGUOUS):
            for names in table.values():
                assert set(names) <= wanted

    def test_both_tables_only_name_classes_in_c(self, pc):
        for table in (pc.SCALE_VG_NAMES, pc.SCALE_VG_AMBIGUOUS, {c: () for c in pc.SCALE_VG_NAMES_AUDITED}):
            assert set(table) <= set(pc.SCALE_CLASSES)

    def test_a_spelling_is_never_both_an_alias_and_ambiguous(self, pc):
        """The two tables mean opposite things; a name in both has no defined outcome."""
        alias = {n for names in pc.SCALE_VG_NAMES.values() for n in names}
        ambiguous = {n for names in pc.SCALE_VG_AMBIGUOUS.values() for n in names}
        assert not (alias & ambiguous)

    def test_bike_is_declared_ambiguous_for_bicycle(self, pc):
        """Not merged: 59.6% of VG `bike` boxes land on no COCO class (#3605)."""
        assert "bike" in pc.SCALE_VG_AMBIGUOUS["bicycle"]
        assert "bike" not in pc.SCALE_VG_NAMES.get("bicycle", ())

    def test_every_class_in_c_has_had_its_names_audited(self, pc):
        """The flag is only worth having if something checks it.

        A class added to *C* without an audit ships the #3605 defect silently:
        `bicycle` did, for the whole of #3156, with every structural check
        passing. The audit is four scripts and a few CPU-minutes.
        """
        missing = sorted(set(pc.SCALE_CLASSES) - set(pc.SCALE_VG_NAMES_AUDITED))
        assert not missing, (
            f"VG-name coverage unmeasured for {missing}. Run coco_folds.py and "
            "vg_name_families.py to find the candidates, name_evidence.py to adjudicate them, "
            "then add the class to SCALE_VG_NAMES_AUDITED whatever the verdict -- an audit that "
            "found nothing is the result the flag exists to record."
        )

    def test_a_class_with_nothing_measured_is_absent_rather_than_empty(self, pc):
        """`()` and "not looked at" would read the same; SCALE_VG_NAMES_AUDITED says which."""
        for table in (pc.SCALE_VG_NAMES, pc.SCALE_VG_AMBIGUOUS):
            assert all(names for names in table.values()), f"empty tuple in {table}"

    def test_every_spelling_is_written_the_way_the_read_matches_it(self, pc):
        """`vg_boxes_by_name` lowercases and strips; a stray capital folds nothing, silently."""
        for table in (pc.SCALE_VG_NAMES, pc.SCALE_VG_AMBIGUOUS):
            for names in table.values():
                for name in names:
                    assert name == name.strip().lower(), name

    def test_sign_is_not_listed_for_stop_sign(self, pc):
        """The largest fold-in column in C, and the one that must not be acted on.

        `sign` carries 46.6% of COCO's `stop sign` boxes and is a stop sign 7.9%
        of the time, so listing it would withhold 12.7 images from the *shared*
        pool -- a cost paid by all twelve classes -- per contaminated negative
        removed (#3618, #3635).
        """
        assert "sign" not in pc.SCALE_VG_AMBIGUOUS.get("stop sign", ())
        assert "sign" not in pc.SCALE_VG_NAMES.get("stop sign", ())


class TestCanonicalise:
    #: A class box and an alias box far enough apart that their union is more
    #: than BAND_MAX_INFLATION times either one -- i.e. the population of #3637.
    SCATTER = {7: {"clock": [[0.0, 0.0, 10.0, 10.0]], "clocks": [[90.0, 90.0, 100.0, 100.0]]}}
    DIMS = {7: (100, 100)}
    NAMES = {"clock": ("clock", "clocks")}

    def test_an_alternate_spelling_folds_onto_the_class_name(self, vgs):
        labels = {7: {"hydrant": [[0.0, 0.0, 1.0, 1.0]]}}
        folded, _ = vgs.canonicalise(labels, {"fire hydrant": ("fire hydrant", "hydrant")})

        assert labels[7] == {"fire hydrant": [[0.0, 0.0, 1.0, 1.0]]}
        assert folded == {"fire hydrant": 1}

    def test_boxes_under_both_spellings_are_kept_together(self, vgs):
        labels = {7: {"fire hydrant": [[0.0, 0.0, 1.0, 1.0]], "hydrant": [[2.0, 2.0, 3.0, 3.0]]}}
        vgs.canonicalise(labels, {"fire hydrant": ("fire hydrant", "hydrant")})

        assert len(labels[7]["fire hydrant"]) == 2

    def test_a_merge_that_folds_nothing_reports_zero(self, vgs):
        """Reported rather than silent: a mis-spelled entry looks exactly like this."""
        labels = {7: {"bus": [[0.0, 0.0, 1.0, 1.0]]}}
        assert vgs.canonicalise(labels, {"fire hydrant": ("hydrant",)})[0] == {"fire hydrant": 0}

    def test_an_empty_table_is_a_no_op(self, vgs):
        labels = {7: {"bus": [[0.0, 0.0, 1.0, 1.0]]}}
        assert vgs.canonicalise(labels, {}) == ({}, {})
        assert labels == {7: {"bus": [[0.0, 0.0, 1.0, 1.0]]}}

    def test_two_alias_spellings_on_one_image_are_one_merge(self, vgs):
        """Judged together, or the guard is asked about a union that never exists."""
        labels = {
            7: {
                "clock": [[0.0, 0.0, 10.0, 10.0]],
                "clocks": [[11.0, 0.0, 20.0, 10.0]],
                "clock face": [[90.0, 90.0, 100.0, 100.0]],
            }
        }
        _, contested = vgs.canonicalise(labels, {"clock": ("clocks", "clock face")}, {7: (100, 100)}, "guarded")

        assert contested == {"clock": 1}
        assert labels[7] == {"clock": [[0.0, 0.0, 10.0, 10.0]]}


class TestFoldModes:
    """#3637: what a fold does to an image the class had already banded."""

    SCATTER = TestCanonicalise.SCATTER
    DIMS = TestCanonicalise.DIMS
    NAMES = TestCanonicalise.NAMES

    def _labels(self):
        return {iid: {n: [list(b) for b in bs] for n, bs in by.items()} for iid, by in self.SCATTER.items()}

    def test_fold_merges_and_lets_the_guard_un_band_the_image(self, vgs, pc):
        labels = self._labels()
        _, contested = vgs.canonicalise(labels, self.NAMES, self.DIMS, "fold")

        assert contested == {"clock": 1}
        assert len(labels[7]["clock"]) == 2
        assert vgs.band_for(labels[7]["clock"], 100, 100) == vgs.SCATTERED

    def test_guarded_keeps_the_class_s_own_band(self, vgs, pc):
        labels = self._labels()
        folded, contested = vgs.canonicalise(labels, self.NAMES, self.DIMS, "guarded")

        assert contested == {"clock": 1}
        assert folded == {"clock": 0}
        assert labels[7] == {"clock": [[0.0, 0.0, 10.0, 10.0]]}
        assert vgs.band_for(labels[7]["clock"], 100, 100) in pc.BOX_BANDS

    def test_guarded_still_merges_when_the_union_stays_in_a_band(self, vgs):
        labels = {7: {"clock": [[0.0, 0.0, 10.0, 10.0]], "clocks": [[10.0, 0.0, 13.0, 10.0]]}}
        folded, contested = vgs.canonicalise(labels, self.NAMES, self.DIMS, "guarded")

        assert (folded, contested) == ({"clock": 1}, {"clock": 0})
        assert len(labels[7]["clock"]) == 2

    def test_additive_never_re_describes_an_image_the_class_already_sees(self, vgs):
        labels = {7: {"clock": [[0.0, 0.0, 10.0, 10.0]], "clocks": [[10.0, 0.0, 13.0, 10.0]]}}
        folded, _ = vgs.canonicalise(labels, self.NAMES, self.DIMS, "additive")

        assert folded == {"clock": 0}
        assert labels[7] == {"clock": [[0.0, 0.0, 10.0, 10.0]]}

    def test_every_mode_still_adds_an_image_the_class_cannot_see(self, vgs):
        """The repair is the point of the table, and no mode may cost it."""
        for mode in vgs.FOLD_MODES:
            labels = {7: {"clocks": [[0.0, 0.0, 10.0, 10.0]]}}
            folded, contested = vgs.canonicalise(labels, self.NAMES, self.DIMS, mode)

            assert (folded, contested) == ({"clock": 1}, {"clock": 0}), mode
            assert labels[7] == {"clock": [[0.0, 0.0, 10.0, 10.0]]}, mode

    def test_without_dims_the_count_is_zero_and_the_fold_is_unconditional(self, vgs):
        labels = self._labels()
        folded, contested = vgs.canonicalise(labels, self.NAMES, None, "fold")

        assert (folded, contested) == ({"clock": 1}, {"clock": 0})
        assert len(labels[7]["clock"]) == 2

    def test_guarded_without_dims_is_refused_rather_than_degraded_into_fold(self, vgs):
        """Silently folding under the name `guarded` is the failure this raises over."""
        with pytest.raises(ValueError, match="needs box_dims"):
            vgs.canonicalise(self._labels(), self.NAMES, None, "guarded")

    def test_additive_needs_no_dims_because_it_asks_no_question_about_size(self, vgs):
        labels = self._labels()
        folded, _ = vgs.canonicalise(labels, self.NAMES, None, "additive")

        assert folded == {"clock": 0}
        assert labels[7] == {"clock": [[0.0, 0.0, 10.0, 10.0]]}

    def test_an_unknown_mode_is_refused_rather_than_treated_as_the_default(self, vgs):
        with pytest.raises(ValueError, match="unknown mode"):
            vgs.canonicalise({}, self.NAMES, self.DIMS, "keep")

    def test_the_shipped_mode_is_one_of_them(self, vgs, pc):
        assert pc.SCALE_FOLD_MODE in vgs.FOLD_MODES


class TestLiftAmbiguous:
    AMBIG = {"bicycle": ("bike",)}

    def test_the_boxes_are_dropped_and_the_pair_returned(self, vgs):
        labels = {7: {"bike": [[0.0, 0.0, 1.0, 1.0]]}}
        assert vgs.lift_ambiguous(labels, self.AMBIG, set()) == {(7, "bicycle")}
        assert labels[7] == {}

    def test_the_lifted_pair_keeps_the_image_out_of_the_negative_pool(self, vgs):
        """The whole point: a `bike` image is not evidence that there is no bicycle."""
        labels = {7: {"bike": [[0.0, 0.0, 1.0, 1.0]]}}
        pairs = vgs.lift_ambiguous(labels, self.AMBIG, set())
        _, _, clean = vgs.band_candidates(labels, {7: (100, 100)}, pairs)

        assert clean == []

    def test_without_the_lift_that_image_would_be_a_negative(self, vgs):
        """The defect, stated as a test: an unmatched spelling reads as an empty image."""
        _, _, clean = vgs.band_candidates({7: {}}, {7: (100, 100)}, set())
        assert clean == [7]

    def test_an_ambiguous_box_never_becomes_a_positive(self, vgs, pc):
        labels = {7: {"bike": [[0.0, 0.0, 50.0, 50.0]]}}
        pairs = vgs.lift_ambiguous(labels, self.AMBIG, set())
        supply, boxes_for, _ = vgs.band_candidates(labels, {7: (100, 100)}, pairs)

        assert all(not ids for band in supply.values() for ids in band.values())
        assert boxes_for == {}

    def test_a_confirmed_class_on_the_same_image_is_not_discarded(self, vgs, pc):
        """`bicycle` is there under a box we trust; the ambiguous one only blurs its extent."""
        band, (lo, hi) = next(iter(pc.BOX_BANDS.items()))
        side = (((lo + hi) / 2) * 10000) ** 0.5
        labels = {7: {"bicycle": [[0.0, 0.0, side, side]], "bike": [[0.0, 0.0, 99.0, 99.0]]}}
        pairs = vgs.lift_ambiguous(labels, self.AMBIG, set())
        supply, _, _ = vgs.band_candidates(labels, {7: (100, 100)}, pairs)

        assert pairs == set()
        assert supply["bicycle"][band] == [7]

    def test_it_touches_only_the_ambiguous_name(self, vgs):
        labels = {7: {"bike": [[0.0, 0.0, 1.0, 1.0]], "bus": [[2.0, 2.0, 3.0, 3.0]]}}
        vgs.lift_ambiguous(labels, self.AMBIG, set())
        assert labels[7] == {"bus": [[2.0, 2.0, 3.0, 3.0]]}

    def test_an_exhaustively_labelled_image_is_not_suppressed(self, vgs):
        """COCO already answered; the image stays a usable negative."""
        labels = {7: {"bike": [[0.0, 0.0, 1.0, 1.0]]}}
        assert vgs.lift_ambiguous(labels, self.AMBIG, {7}) == set()
        assert labels[7] == {}

        _, _, clean = vgs.band_candidates(labels, {7: (100, 100)}, set())
        assert clean == [7]

    def test_the_boxes_go_either_way(self, vgs):
        """`band_candidates` bands by category name and has no cell for a `bike`."""
        labels = {7: {"bike": [[0.0, 0.0, 1.0, 1.0]]}}
        vgs.lift_ambiguous(labels, self.AMBIG, {7})
        assert "bike" not in labels[7]


class TestDesignateCells:
    """Selection must be stable under a changing pool, and keep reviewed images."""

    def _supply(self, pc, cls: str, band: str, ids: list[int]):
        supply = {c: {b: [] for b in pc.BOX_BANDS} for c in pc.SCALE_CLASSES}
        supply[cls][band] = list(ids)
        return supply

    def test_adding_a_candidate_moves_only_that_candidate(self, vgs, pc):
        """The whole reason selection ranks by hash instead of sampling.

        ``rng.sample`` is deterministic given the same list, but any edit to the
        pool reshuffles the entire draw -- and a rebuild then silently retires
        images a human already reviewed.
        """
        cls, band = pc.SCALE_CLASSES[0], next(iter(pc.BOX_BANDS))
        cell = pc.scale_cell(cls, band)
        pool = list(range(1000, 1000 + pc.SCALE_N_POS + 20))

        before = vgs.designate_cells(self._supply(pc, cls, band, pool), {}, {})[cell]
        after = vgs.designate_cells(self._supply(pc, cls, band, [7777, *pool]), {}, {})[cell]

        assert len(before) == len(after) == pc.SCALE_N_POS
        # At most one seat changes hands: the newcomer's, if it out-ranks the
        # last incumbent. A reshuffle would move nearly all of them.
        assert len(set(before) - set(after)) <= 1

    def test_reviewed_images_outrank_unreviewed_ones_for_a_seat(self, vgs, pc):
        """A correction can move an image to a cell that is already full.

        If it lands nowhere, the review quietly stops covering it -- which is
        what happened to 99 of 360 boxed positives the first time round.
        """
        cls, band = pc.SCALE_CLASSES[0], next(iter(pc.BOX_BANDS))
        cell = pc.scale_cell(cls, band)
        pool = list(range(1000, 1000 + pc.SCALE_N_POS + 50))
        # Whichever candidates the hash draw would have left out.
        unlucky = [i for i in pool if i not in vgs.designate_cells(self._supply(pc, cls, band, pool), {}, {})[cell]]
        assert unlucky, "test needs a pool larger than the cell"

        corrections = {(i, cls): {"present": True} for i in unlucky[:5]}
        chosen = vgs.designate_cells(self._supply(pc, cls, band, pool), corrections, {})[cell]

        assert set(unlucky[:5]) <= set(chosen)
        assert len(chosen) == pc.SCALE_N_POS

    def test_a_roster_pins_membership_across_a_rule_change(self, vgs, pc):
        """The roster is what a review was carried out against; it wins."""
        cls, band = pc.SCALE_CLASSES[0], next(iter(pc.BOX_BANDS))
        cell = pc.scale_cell(cls, band)
        pool = list(range(1000, 1000 + pc.SCALE_N_POS + 50))
        pinned = pool[-pc.SCALE_N_POS :]

        chosen = vgs.designate_cells(self._supply(pc, cls, band, pool), {}, {"cells": {cell: pinned}})[cell]
        assert set(chosen) == set(pinned)

    def test_roster_entries_that_are_no_longer_eligible_drop_out(self, vgs, pc):
        """A correction can move or remove an image; the shortfall backfills by rank."""
        cls, band = pc.SCALE_CLASSES[0], next(iter(pc.BOX_BANDS))
        cell = pc.scale_cell(cls, band)
        pool = list(range(1000, 1000 + pc.SCALE_N_POS + 10))

        chosen = vgs.designate_cells(
            self._supply(pc, cls, band, pool), {}, {"cells": {cell: [*pool[: pc.SCALE_N_POS - 1], 424242]}}
        )[cell]

        assert 424242 not in chosen
        assert len(chosen) == pc.SCALE_N_POS
        assert set(pool[: pc.SCALE_N_POS - 1]) <= set(chosen)

    def test_an_undersupplied_cell_is_reported_not_padded(self, vgs, pc, capsys):
        cls, band = pc.SCALE_CLASSES[0], next(iter(pc.BOX_BANDS))
        cell = pc.scale_cell(cls, band)
        chosen = vgs.designate_cells(self._supply(pc, cls, band, [1, 2, 3]), {}, {})[cell]

        assert chosen == [1, 2, 3] or set(chosen) == {1, 2, 3}
        assert f"UNDER-SUPPLIED {cell}" in capsys.readouterr().out


class TestDrawNegatives:
    def test_roster_negatives_are_kept_and_the_rest_backfilled(self, vgs, pc):
        clean = list(range(1, pc.SCALE_N_NEG + pc.SCALE_N_NEG_SPARE + 100))
        keep = clean[-10:]

        negatives, spares = vgs.draw_negatives(clean, {"negatives": keep}, set(clean), 1.0)

        assert set(keep) <= set(negatives)
        assert len(negatives) == pc.SCALE_N_NEG
        assert len(spares) == pc.SCALE_N_NEG_SPARE
        assert not set(negatives) & set(spares)

    def test_a_roster_entry_that_is_no_longer_clean_drops_out(self, vgs, pc):
        clean = list(range(1, pc.SCALE_N_NEG + pc.SCALE_N_NEG_SPARE + 100))

        negatives, spares = vgs.draw_negatives(clean, {"negatives": [424242]}, set(clean), 1.0)

        assert 424242 not in negatives and 424242 not in spares
        assert len(negatives) == pc.SCALE_N_NEG

    def test_a_short_clean_pool_yields_what_there_is(self, vgs, pc):
        negatives, spares = vgs.draw_negatives([1, 2, 3], {}, {1, 2, 3}, 1.0)
        assert sorted([*negatives, *spares]) == [1, 2, 3]

    def test_an_all_provable_pool_draws_only_coco_scored_images(self, vgs, pc):
        """`coco_fraction` 1.0 must exclude every image COCO never scored (#3670)."""
        clean = list(range(1, 3 * (pc.SCALE_N_NEG + pc.SCALE_N_NEG_SPARE)))
        exhaustive = set(clean[::2])

        negatives, spares = vgs.draw_negatives(clean, {}, exhaustive, 1.0)

        assert len(negatives) == pc.SCALE_N_NEG
        assert set(negatives) <= exhaustive
        assert set(spares) <= exhaustive

    def test_a_matched_pool_hits_the_requested_share(self, vgs, pc):
        """The point of `matched` is the RATIO, so the draw has to honour it."""
        clean = list(range(1, 4 * (pc.SCALE_N_NEG + pc.SCALE_N_NEG_SPARE)))
        exhaustive = set(clean[::2])

        negatives, _spares = vgs.draw_negatives(clean, {}, exhaustive, 0.6)

        assert len(negatives) == pc.SCALE_N_NEG
        assert abs(len(set(negatives) & exhaustive) / len(negatives) - 0.6) < 0.01

    def test_a_pinned_image_keeps_its_seat_within_its_own_stratum(self, vgs, pc):
        """A composition change must retire only what the new one cannot hold."""
        clean = list(range(1, 4 * (pc.SCALE_N_NEG + pc.SCALE_N_NEG_SPARE)))
        exhaustive = set(clean[::2])
        pinned_provable = sorted(exhaustive)[-10:]
        pinned_silent = sorted(set(clean) - exhaustive)[-10:]
        roster = {"negatives": pinned_provable + pinned_silent}

        negatives, _spares = vgs.draw_negatives(clean, roster, exhaustive, 1.0)

        assert set(pinned_provable) <= set(negatives)
        assert not set(pinned_silent) & set(negatives)


class TestRank:
    def test_rank_is_cell_local(self, vgs):
        """Two cells order the same image independently, so one cell's edit is local."""
        assert vgs.rank("bus@small", 7) != vgs.rank("bus@large", 7)

    def test_rank_is_stable_across_processes(self, vgs):
        assert vgs.rank("bus@small", 7) == vgs.rank("bus@small", 7)


@pytest.fixture(scope="module")
def audit():
    """``pilebuild.audit``, imported without ``build_pile``'s env setup."""
    if str(_PILE_DIR) not in sys.path:
        sys.path.insert(0, str(_PILE_DIR))
    from pilebuild import audit as mod

    return mod


@pytest.fixture(scope="module")
def coverage():
    """``check_review_coverage``, which defers ``setup_env`` into ``main``."""
    if str(_PILE_DIR) not in sys.path:
        sys.path.insert(0, str(_PILE_DIR))
    import check_review_coverage

    return check_review_coverage


def _pool_media(*, coco_scored: bool = True, designated: bool = True, stamped: bool = True) -> dict:
    """A shared negative as the loader writes one: no categories, scorable everywhere."""
    media = {
        "categories": [],
        "evaluable_categories": ["bus@small"] if designated else [],
        "labels_exhaustive": True,
    }
    if stamped:
        media["coco_scored"] = coco_scored
    return media


class TestNegativePoolProblems:
    """`--verify`'s two claims about the pool itself (#3670).

    Both are invisible to every other check: a pool of the wrong size drawn from
    the wrong half of VG loads, bands and scores exactly like a healthy one.
    """

    def _pool(self, pc, n: int, silent: int = 0) -> dict:
        medias = {i: _pool_media(coco_scored=i >= silent) for i in range(n)}
        # One positive, so the cell is not all-negative and the pool filter has
        # something to exclude.
        medias[-1] = {"categories": ["bus@small"], "evaluable_categories": ["bus@small"]}
        return medias

    def test_a_healthy_pool_is_silent(self, audit, pc):
        assert audit.negative_pool_problems("vg_scale", self._pool(pc, pc.SCALE_N_NEG)) == []

    def test_an_off_coco_negative_is_named_under_provable(self, audit, pc):
        problems = audit.negative_pool_problems("vg_scale", self._pool(pc, pc.SCALE_N_NEG, silent=7))

        assert any("not COCO-scored" in p and "7 of" in p for p in problems)

    def test_a_reviewed_image_does_not_count_as_provable(self, audit, pc):
        """`labels_exhaustive` is also set by a one-class review, so it is the wrong flag."""
        medias = self._pool(pc, pc.SCALE_N_NEG)
        for i in range(5):
            medias[i]["coco_scored"] = False
            medias[i]["labels_exhaustive"] = True  # a human looked -- at one class

        assert any("not COCO-scored" in p for p in audit.negative_pool_problems("vg_scale", medias))

    def test_a_cell_predating_the_stamp_is_told_to_rebuild(self, audit, pc):
        """An unstamped cell cannot answer the question; it must not pass by default."""
        medias = {i: _pool_media(stamped=False) for i in range(pc.SCALE_N_NEG)}

        problems = audit.negative_pool_problems("vg_scale", medias)

        assert any("built before the flag existed" in p for p in problems)

    def test_the_pre_3670_pool_size_is_caught(self, audit, pc):
        """The exact shape of a deferred rebuild: right cells, stale pool."""
        problems = audit.negative_pool_problems("vg_scale", self._pool(pc, 3900))

        assert any(f"3900 designated negatives, but the config says {pc.SCALE_N_NEG}" in p for p in problems)
        # The message has to quote the prevalence, because that is the number a
        # report would otherwise take from the constant and get wrong.
        assert any("2.50% prevalence" in p for p in problems)

    def test_spares_are_not_counted_as_pool(self, audit, pc):
        """300 spares in the denominator would fire the size check on a healthy pile."""
        medias = self._pool(pc, pc.SCALE_N_NEG)
        medias.update({10_000 + i: _pool_media(designated=False) for i in range(pc.SCALE_N_NEG_SPARE)})

        assert audit.negative_pool_problems("vg_scale", medias) == []

    def test_deep_is_exempt_from_the_composition_rule(self, audit, pc):
        """`vg_scale_deep` is pinned to the pre-#3670 construction on purpose (#3690)."""
        medias = {i: _pool_media(coco_scored=False) for i in range(pc.SCALE_DEEP_N_NEG)}

        assert audit.negative_pool_problems("vg_scale_deep", medias) == []


class TestCoverageRow:
    """Which retirements the coverage gate forgives, and which it must not (#3670)."""

    def test_an_intact_population_is_full_coverage(self, coverage):
        by_rule, by_fix, denom, kept, cov = coverage.coverage_row({1, 2, 3}, {1, 2, 3}, set(), lambda _i: True)

        assert (by_rule, by_fix, denom, kept, cov) == (0, 0, 3, 3, 1.0)

    def test_an_image_the_rule_cannot_hold_leaves_the_denominator(self, coverage):
        """The #3670 case: off-COCO reviews are about a stratum the pool dropped."""
        by_rule, _fix, denom, kept, cov = coverage.coverage_row({1, 2, 3, 4}, {1, 2}, set(), lambda i: i <= 2)

        assert (by_rule, denom, kept, cov) == (2, 2, 2, 1.0)

    def test_an_eligible_image_that_vanished_still_fails(self, coverage):
        """The gate's whole job. A reshuffle must not hide behind the rule."""
        _rule, _fix, denom, kept, cov = coverage.coverage_row({1, 2, 3, 4}, {1}, set(), lambda _i: True)

        assert (denom, kept, cov) == (4, 1, 0.25)

    def test_a_correction_is_forgiven_but_only_when_eligible(self, coverage):
        _rule, by_fix, denom, kept, cov = coverage.coverage_row({1, 2, 3}, {1, 2}, {3}, lambda _i: True)

        assert (by_fix, denom, kept, cov) == (1, 2, 2, 1.0)

    def test_an_ineligible_correction_is_counted_once(self, coverage):
        """Both exits apply; counting it twice would understate coverage."""
        by_rule, by_fix, denom, _kept, _cov = coverage.coverage_row({1, 2}, {1}, {2}, lambda i: i == 1)

        assert (by_rule, by_fix, denom) == (1, 0, 1)

    def test_matched_forgives_nothing(self, coverage):
        """Only `provable` narrows the frame; `matched` draws from both halves."""
        eligible = coverage.eligible_under("matched")

        assert all(eligible(i) for i in (1, 2, 424242))


class TestCorrectionsOutsideC:
    """A shared verdict file holds classes this build does not have (#3670)."""

    def test_a_verdict_for_a_class_outside_c_is_skipped(self, vgs):
        """`corrections.json` is shared; #3588's pass added thirteen classes to it.

        Without the skip the label is written anyway and `band_candidates` dies
        on `supply['car']` -- a shared file making the shipped twelve-class
        construction unbuildable, reported as a dict lookup three passes later.
        """
        labels = {7: {}}

        vgs.apply_corrections(labels, {(7, "car"): _verdict(True, [[0.0, 0.0, 0.5, 0.5]])}, {7: (640, 480)}, set())

        assert labels[7] == {}

    def test_a_verdict_outside_c_does_not_make_the_image_exhaustive(self, vgs):
        """The half that decides pool membership under #3670.

        Marking an image exhaustive claims absence is a fact for every class in
        C. A human who looked for a `car` established nothing about `bus`.
        """
        exhaustive: set[int] = set()

        vgs.apply_corrections({7: {}}, {(7, "car"): _verdict(False)}, {7: (640, 480)}, exhaustive)

        assert exhaustive == set()

    def test_a_verdict_inside_c_still_applies(self, vgs):
        exhaustive: set[int] = set()
        labels = {7: {"bus": [[1.0, 1.0, 2.0, 2.0]]}}

        vgs.apply_corrections(labels, {(7, "bus"): _verdict(False)}, {7: (640, 480)}, exhaustive)

        assert labels[7] == {} and exhaustive == {7}

    def test_a_widened_class_list_is_honoured(self, vgs):
        """#3588's expansion passes its own C rather than editing the module."""
        labels = {7: {}}

        vgs.apply_corrections(
            labels, {(7, "car"): _verdict(True, [[0.0, 0.0, 0.5, 0.5]])}, {7: (640, 480)}, set(), classes=("car",)
        )

        assert "car" in labels[7]


class TestContaminationIdentity:
    """`forward + reverse = 2` under pure contamination (#3702).

    #3670 read the two probe arms as independent routes and quoted their
    agreement as evidence. They are one route: contamination inflates the
    forward arm and depresses the reverse one by the same mechanism. The
    identity is what makes their SUM a contamination-free diagnostic, and the
    report's whole corrected argument rests on it, so it is asserted rather
    than left in a script nobody runs.
    """

    def _arms(self, c: float, sep: float, n: int = 200_000, fpr: float = 0.05):
        np = pytest.importorskip("numpy")
        rng = np.random.default_rng(0)
        provable = rng.normal(0.0, 1.0, n)
        hidden = int(round(c * n))
        # Same distribution as `provable` apart from the hidden positives: there
        # is deliberately NO provenance effect to find.
        silent = np.concatenate([rng.normal(0.0, 1.0, n - hidden), rng.normal(sep, 1.0, hidden)])
        t_f = float(np.quantile(provable, 1 - fpr))
        t_r = float(np.quantile(silent, 1 - fpr))
        return float((silent > t_f).mean()) / fpr, float((provable > t_r).mean()) / fpr

    @pytest.mark.parametrize("c", [0.005, 0.014, 0.025])
    @pytest.mark.parametrize("sep", [1.5, 2.6])
    def test_the_arms_sum_to_two_whatever_the_rate_and_tpr(self, c, sep):
        forward, reverse = self._arms(c, sep)

        assert abs(forward + reverse - 2.0) < 0.05

    def test_contamination_moves_both_arms_in_opposite_directions(self):
        """The claim that replaced 'the reverse arm is contamination-free'."""
        clean_f, clean_r = self._arms(0.0, 2.0)
        dirty_f, dirty_r = self._arms(0.025, 2.0)

        assert dirty_f > clean_f and dirty_r < clean_r

    def test_the_measured_sums_are_outside_what_contamination_explains(self):
        """2.35 is why #3670's asymmetry is real rather than dirt."""
        worst = max(sum(self._arms(c, sep)) for c in (0.005, 0.014, 0.025) for sep in (1.5, 2.0, 2.6))

        assert worst < 2.35
