"""Tests for the DocMarks corpus builder (``scripts/experiments/docmarks/``).

The builder assembles one instance-retrieval corpus out of four document
sources.  Everything network-touching lives behind ``fetch_*`` and is not
exercised here; everything that decides *what the corpus means* is pure and is
pinned below.

Three of these tests guard properties that a plausible-looking refactor would
quietly break, and whose breakage would not show up as a crash — only as
numbers that are wrong in a direction nobody notices:

* **tier nesting** — a study on ``docmarks_s`` and one on ``docmarks_l`` are
  only comparable if the small tier is a subset of the large one.  Sampling
  distractors with anything order-dependent silently breaks that.
* **contamination** — Tobacco800 and UCSF's Tobacco industry are the same
  underlying archive, so scoring Tobacco800 classes against UCSF tobacco pages
  counts correct retrievals as false positives.
* **synthetic box tightness** — a rotated paste's ground-truth box must come
  from the alpha bbox, not the paste rectangle, or every query crop carries a
  third of a page of blank paper.

The scripts are loose modules, not package members, so the directory goes on
``sys.path`` and they are imported by name.
"""

import importlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

_DOCMARKS = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "docmarks"


@pytest.fixture(scope="module", autouse=True)
def _on_path():
    sys.path.insert(0, str(_DOCMARKS))
    yield
    with pytest.MonkeyPatch.context():
        pass
    if str(_DOCMARKS) in sys.path:
        sys.path.remove(str(_DOCMARKS))


@pytest.fixture(scope="module")
def mods(_on_path):
    return {
        "cfg": importlib.import_module("docmarks_config"),
        "common": importlib.import_module("sources._common"),
        "spods": importlib.import_module("sources.spods"),
        "staver": importlib.import_module("sources.staver"),
        "tobacco800": importlib.import_module("sources.tobacco800"),
        "ucsf": importlib.import_module("sources.ucsf"),
        "artwork": importlib.import_module("sources.artwork"),
        "cluster": importlib.import_module("cluster_marks"),
        "build": importlib.import_module("build_corpus"),
        "synth": importlib.import_module("synth_compose"),
        "embed": importlib.import_module("embed_corpus"),
        "roster": importlib.import_module("roster"),
        "shortlist": importlib.import_module("shortlist"),
        "audit": importlib.import_module("audit_to_corrections"),
        "slate": importlib.import_module("make_audit_slate"),
        "siglip": importlib.import_module("siglip_audit"),
        "report": importlib.import_module("make_report"),
    }


def _page(mods, page_id, source, marks=(), path="x.png", w=1000, h=1400):
    Mark, Page = mods["common"].Mark, mods["common"].Page
    return Page(
        page_id=page_id,
        source=source,
        path=path,
        width=w,
        height=h,
        marks=[Mark(kind=k, box=b, class_id=c, provenance=p) for k, b, c, p in marks],
    )


# ---------------------------------------------------------------- primitives


class TestStableRank:
    def test_is_deterministic_and_in_range(self, mods):
        rank = mods["common"].stable_rank
        values = [rank(f"page/{i}", "salt") for i in range(200)]
        assert all(0.0 <= v < 1.0 for v in values)
        assert values == [rank(f"page/{i}", "salt") for i in range(200)]

    def test_salt_changes_the_ordering(self, mods):
        rank = mods["common"].stable_rank
        keys = [f"page/{i}" for i in range(50)]
        assert sorted(keys, key=lambda k: rank(k, "a")) != sorted(keys, key=lambda k: rank(k, "b"))


class TestMaskToBoxes:
    def test_finds_separated_components(self, mods):
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[10:40, 10:50] = 255
        mask[120:160, 130:170] = 255
        boxes = mods["common"].mask_to_boxes(mask, min_area_frac=0.0)
        assert len(boxes) == 2
        assert (10, 10, 40, 30) in boxes

    def test_drops_speckle_below_the_area_floor(self, mods):
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[10:40, 10:50] = 255
        mask[100, 100] = 255  # one pixel
        boxes = mods["common"].mask_to_boxes(mask, min_area_frac=0.001)
        assert len(boxes) == 1

    def test_empty_mask_is_no_boxes(self, mods):
        assert mods["common"].mask_to_boxes(np.zeros((50, 50), dtype=np.uint8)) == []

    def test_inverted_masks_are_detected_not_swallowed(self, mods):
        # SPODS ships 1-bit masks with the mark BLACK on white paper. Read as
        # "non-zero is foreground" this yields one page-sized box per page --
        # which does not crash, it silently produces 1,088 identical rectangles
        # that cluster into a single class and look like a working corpus. On the
        # real data that is exactly what happened: 2,176 marks, 1 class.
        mask = np.full((200, 200), 255, dtype=np.uint8)
        mask[10:40, 10:50] = 0  # the mark, dark on light
        boxes = mods["common"].mask_to_boxes(mask, min_area_frac=0.0)
        assert boxes == [(10, 10, 40, 30)]

    def test_normal_polarity_still_works(self, mods):
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[10:40, 10:50] = 255
        assert mods["common"].mask_to_boxes(mask, min_area_frac=0.0) == [(10, 10, 40, 30)]

    def test_polarity_can_be_forced(self, mods):
        # A genuinely dense mask (body text on a full page) can be forced rather
        # than left to the minority heuristic.
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[0:80, :] = 255  # 80% lit: "auto" would invert this
        auto = mods["common"].mask_to_boxes(mask, min_area_frac=0.0, polarity="auto")
        forced = mods["common"].mask_to_boxes(mask, min_area_frac=0.0, polarity="light")
        assert auto == [(0, 80, 100, 20)]
        assert forced == [(0, 0, 100, 80)]

    def test_an_unknown_polarity_is_refused(self, mods):
        with pytest.raises(ValueError, match="unknown polarity"):
            mods["common"].mask_to_boxes(np.zeros((10, 10), dtype=np.uint8), polarity="sideways")


#: A page the size of a real scan, so that the 0.0002 area floor is a real
#: number (260 px of ink) rather than something a toy fixture can clear by
#: accident.
_PAGE_W, _PAGE_H = 1000, 1300


def _stroke_stamp(origin=(120, 120), strokes=12):
    """A "stamp" mask built the way a real one decomposes: many thin strokes.

    Each stroke is 2 px wide and 24 px tall -- 48 filled px against a floor of
    ``0.0002 * 1000 * 1300`` = 260 px, so *every* fragment is individually well
    below it -- and consecutive strokes sit 10 px apart, further than the old
    fixed ``gap=6`` could bridge.
    """
    mask = np.zeros((_PAGE_H, _PAGE_W), dtype=np.uint8)
    ox, oy = origin
    for i in range(strokes):
        x = ox + i * 10
        mask[oy : oy + 24, x : x + 2] = 255
    return mask


class TestFragmentingStamps:
    """The geometry bug of issue #3361, pinned from both ends.

    A stamp's mask is not one component: it is a ring, the text inside it, and a
    broken arc where the ink did not take -- or, for a script stamp, one
    component per pen stroke. Every one of those is individually below the area
    floor. Running the floor *before* the merge therefore deleted eleven of the
    dozen as "speckle", left the merge nothing to reassemble, and promoted the
    one or two chunkiest survivors to classes of their own. That is how
    ``spods/stamp_00129_1`` came to be 38 instances of the word "New".

    The failure is silent in both directions, which is why both are pinned: too
    strict a floor yields *zero* boxes where a mark plainly is, and too small a
    gap yields *two*.
    """

    def test_a_stamp_of_sub_floor_strokes_is_one_box_not_zero_and_not_two(self, mods):
        mask = _stroke_stamp()
        gap = mods["common"].merge_gap_for_page(_PAGE_W, _PAGE_H)

        boxes = mods["common"].mask_to_boxes(mask, min_area_frac=0.0002, merge_gap=gap)

        assert len(boxes) == 1, f"expected one stamp, got {len(boxes)}: {boxes}"
        x, y, w, h = boxes[0]
        assert (x, y) == (120, 120)
        assert (w, h) == (112, 24)

    def test_filtering_before_merging_is_what_loses_it(self, mods):
        # The old order, spelled out: floor first (mask_to_boxes with no merge),
        # then merge. Every stroke is below the floor, so nothing survives to
        # merge and the stamp vanishes entirely.
        mask = _stroke_stamp()
        filtered_first = mods["common"].mask_to_boxes(mask, min_area_frac=0.0002, merge_gap=0)
        assert mods["common"].merge_overlapping(filtered_first, gap=6) == []

    def test_the_merged_group_is_judged_on_its_ink_not_its_box(self, mods):
        # A ring is mostly hole, so a merged group's box area wildly overstates
        # how much ink it carries. The floor must see the ink.
        comps = mods["common"].mask_components(_stroke_stamp())
        merged = mods["common"].merge_components(comps, gap=16)
        assert len(merged) == 1
        assert merged[0].ink == sum(c.ink for c in comps) == 12 * 48
        assert merged[0].ink < merged[0].box[2] * merged[0].box[3]

    def test_true_single_pixel_speckle_cannot_bridge_two_marks(self, mods):
        # The one filter that must still run *before* the merge. Two marks 40 px
        # apart, with a trail of single pixels between them: keep the trail and
        # the merge welds the pair into one mark.
        mask = np.zeros((300, 300), dtype=np.uint8)
        mask[100:140, 40:80] = 255
        mask[100:140, 160:200] = 255
        for x in range(85, 160, 10):
            mask[120, x] = 255
        boxes = mods["common"].mask_to_boxes(mask, min_area_frac=0.0, merge_gap=12)
        assert len(boxes) == 2


class TestMergeGap:
    def test_scales_with_the_page(self, mods):
        gap = mods["common"].merge_gap_for_page
        # A gap in pixels does not travel between a 950 px scan and a 2,500 px
        # one: the fragments of a broken stamp sit a fixed fraction of the stamp
        # apart, and the stamp is a fixed fraction of the page.
        assert gap(2500, 3300) > gap(950, 1300) > gap(200, 200)

    def test_never_falls_below_the_absolute_floor(self, mods):
        assert mods["common"].merge_gap_for_page(10, 10) == mods["common"].MERGE_GAP_MIN_PX


class TestMergeOverlapping:
    def test_merges_a_fragmented_stamp_into_one_mark(self, mods):
        # A rubber stamp's mask breaks into a ring plus its inner text; left
        # unmerged each fragment becomes its own "class" and the inventory is
        # nonsense.
        fragments = [(10, 10, 30, 5), (10, 16, 30, 5), (10, 22, 30, 5)]
        merged = mods["common"].merge_overlapping(fragments, gap=6)
        assert merged == [(10, 10, 30, 17)]

    def test_leaves_distant_marks_alone(self, mods):
        boxes = [(10, 10, 20, 20), (500, 500, 20, 20)]
        assert len(mods["common"].merge_overlapping(boxes, gap=6)) == 2


class TestRejectOversize:
    def test_a_page_sized_box_is_rejected_and_reported(self, mods):
        # spods/00975: a ruled table whose borders weld the whole grid into one
        # component, boxed at 45.9% of the page and captioned "the largest mark".
        table = (30, 40, 900, 700)
        stamp = (100, 100, 200, 200)
        kept, rejected = mods["common"].reject_oversize([stamp, table], 1000, 1400, 0.25)
        assert kept == [stamp]
        assert rejected == [table]

    def test_an_ordinary_mark_is_untouched(self, mods):
        kept, rejected = mods["common"].reject_oversize([(10, 10, 90, 90)], 1000, 1400, 0.25)
        assert (kept, rejected) == ([(10, 10, 90, 90)], [])


# ------------------------------------------------------------------- sources


class TestSpods:
    @pytest.mark.parametrize(
        "name,expected",
        [("image (417).png", 417), ("image (1).png", 1), ("IMAGE (23).PNG", 23), ("notes.txt", None)],
    )
    def test_page_number(self, mods, name, expected):
        assert mods["spods"].page_number(Path(name)) == expected

    @staticmethod
    def _write_gt(gt, page_no, masks):
        """``{kind: 2-D array}`` -> ``gt/<kind>/image (n).png``."""
        from PIL import Image

        for kind, arr in masks.items():
            (gt / kind).mkdir(parents=True, exist_ok=True)
            Image.fromarray(arr).save(gt / kind / f"image ({page_no}).png")

    @staticmethod
    def _box_mask(box, shape=(600, 500)):
        arr = np.zeros(shape, dtype=np.uint8)
        x, y, w, h = box
        arr[y : y + h, x : x + w] = 255
        return arr

    def test_marks_for_page_reads_every_category(self, mods, tmp_path):
        gt = tmp_path / "gt"
        self._write_gt(
            gt,
            7,
            {
                "logo": self._box_mask((20, 20, 60, 40)),
                "stamp": self._box_mask((200, 300, 80, 80)),
            },
        )

        found = mods["spods"].marks_for_page(gt, 7, width=500, height=600, min_area_frac=0.0)
        by_kind = {m.kind: m.box for m in found.marks}
        assert by_kind == {"logo": (20, 20, 60, 40), "stamp": (200, 300, 80, 80)}
        # Identity is never invented at parse time: SPODS does not ship it.
        assert all(m.class_id is None for m in found.marks)
        assert all(m.provenance == "gt" for m in found.marks)

    def test_a_stroke_built_stamp_survives_the_floor_as_one_mark(self, mods, tmp_path):
        # The end-to-end shape of issue #3361, through the real adapter: a stamp
        # whose components are each below the area floor must come out as one
        # box, not zero and not several.
        gt = tmp_path / "gt"
        self._write_gt(gt, 3, {"stamp": _stroke_stamp()})

        found = mods["spods"].marks_for_page(gt, 3, width=_PAGE_W, height=_PAGE_H, min_area_frac=0.0002)
        assert [m.kind for m in found.marks] == ["stamp"]
        assert found.marks[0].box == (120, 120, 112, 24)

    def test_the_text_mask_becomes_page_metadata_not_marks(self, mods, tmp_path):
        # SPODS' text mask is the page body. Emitted as marks it produced ~1.1
        # "marks" per page -- whichever headings and table rules had an underline
        # welding their glyphs into one component -- which were never query
        # classes but leaked into every consumer that read `page.marks` without a
        # kind filter, starting with the synthetic-background selector.
        gt = tmp_path / "gt"
        body = np.zeros((600, 500), dtype=np.uint8)
        body[100:130, 40:460] = 255  # an underlined heading: one fat component
        body[200:210, 40:200] = 255
        self._write_gt(gt, 9, {"logo": self._box_mask((20, 20, 60, 40)), "text": body})

        found = mods["spods"].marks_for_page(gt, 9, width=500, height=600, min_area_frac=0.0002)

        assert [m.kind for m in found.marks] == ["logo"]
        assert found.meta["text_components"] == 2
        assert found.meta["text_frac"] == pytest.approx((30 * 420 + 10 * 160) / (500 * 600), rel=1e-3)

    def test_a_page_scale_box_is_rejected_with_a_warning(self, mods, tmp_path):
        # spods/00975 boxed 45.9% of the page and the report captioned it "the
        # largest mark".
        gt = tmp_path / "gt"
        self._write_gt(gt, 11, {"stamp": self._box_mask((20, 20, 400, 350))})

        found = mods["spods"].marks_for_page(gt, 11, width=500, height=600, min_area_frac=0.0002, max_area_frac=0.25)
        assert found.marks == []
        assert len(found.warnings) == 1
        assert "46.7% of the page" in found.warnings[0]

    def test_find_tree_reports_an_unrecognised_layout(self, mods, tmp_path):
        (tmp_path / "something-else").mkdir()
        with pytest.raises(mods["common"].FetchError, match="layout not recognised"):
            mods["spods"].find_tree(tmp_path)


class TestStaver:
    def test_parse_info_normalises_keys_and_types(self, mods):
        parsed = mods["staver"].parse_info(
            "Number of stamps: 2\nSignature present : yes\nStamp Color: colored\nOverlap: no\n"
        )
        assert parsed["number_of_stamps"] == 2
        assert parsed["signature_present"] is True
        assert parsed["overlap"] is False
        assert parsed["stamp_color"] == "colored"

    def test_expected_stamp_count_accepts_the_known_spellings(self, mods):
        for key in ("number_of_stamps", "stamps", "num_stamps"):
            assert mods["staver"].expected_stamp_count({key: 3}) == 3
        assert mods["staver"].expected_stamp_count({"stamp_color": "black"}) is None


class TestTobacco800:
    GEDI = """<?xml version="1.0"?>
    <GEDI>
      <DL_DOCUMENT src="xyz.tif">
        <DL_PAGE gedi_type="DL_PAGE" src="xyz00001.tif" width="2544" height="3295">
          <DL_ZONE gedi_type="DLLogo" id="1" col="220" row="140" width="600" height="180"/>
          <DL_ZONE gedi_type="DLSignature" id="2" col="300" row="2400"
                   width="500" height="200" AuthorID="Horrigan"/>
          <DL_ZONE gedi_type="DLText" id="3" col="0" row="0" width="10" height="10"/>
        </DL_PAGE>
      </DL_DOCUMENT>
    </GEDI>"""

    def test_parses_logo_and_signature_zones_only(self, mods):
        parsed = mods["tobacco800"].parse_gedi(self.GEDI)
        marks = parsed["xyz00001"]
        assert {m.kind for m in marks} == {"logo", "signature"}

    def test_logo_box_uses_col_row_width_height(self, mods):
        logo = next(m for m in mods["tobacco800"].parse_gedi(self.GEDI)["xyz00001"] if m.kind == "logo")
        assert logo.box == (220, 140, 600, 180)

    def test_signature_identity_becomes_a_class_id(self, mods):
        sig = next(m for m in mods["tobacco800"].parse_gedi(self.GEDI)["xyz00001"] if m.kind == "signature")
        assert sig.class_id == "tobacco800/signature_horrigan"

    def test_a_bare_serial_id_is_not_treated_as_an_identity(self, mods):
        # `id` is a per-zone serial. Reading it as an identity would give every
        # single mark its own singleton class, and the corpus would look full of
        # classes while containing none.
        logo = next(m for m in mods["tobacco800"].parse_gedi(self.GEDI)["xyz00001"] if m.kind == "logo")
        assert logo.class_id is None


class TestUcsf:
    def test_build_query_quotes_multiword_values(self, mods):
        q = mods["ucsf"].build_query(industry="Fossil Fuel", author="LOR, LORILLARD")
        assert 'industry:"Fossil Fuel"' in q
        assert 'author:"LOR, LORILLARD"' in q
        assert "pages:1" in q

    def test_pdf_url_uses_the_split_character_scheme(self, mods):
        assert mods["ucsf"].pdf_url("ffbb0019").endswith("/f/f/b/b/ffbb0019/ffbb0019.pdf")

    def test_pdf_url_rejects_a_short_id(self, mods):
        with pytest.raises(ValueError, match="too short"):
            mods["ucsf"].pdf_url("ab")

    @pytest.mark.parametrize(
        "date,expected",
        [("1996 January 24", 1996), ("1965", 1965), ("2003 December 04", 2003), ("", None), (None, None)],
    )
    def test_year(self, mods, date, expected):
        assert mods["ucsf"].year(date) == expected

    def test_first_value_unwraps_solr_multivalued_fields(self, mods):
        assert mods["ucsf"].first_value({"collection": ["Lorillard Records", "MSA"]}, "collection") == (
            "Lorillard Records"
        )

    def test_an_author_is_a_candidate_pool_never_a_class(self, mods):
        # The metadata says the page is *from* Philip Morris; it has never
        # looked at the mark. Making it a class id would put two different
        # artworks in one class whenever a company redesigned its letterhead,
        # and split one artwork across two classes whenever subsidiaries shared
        # it -- both of them errors the eval exists to measure, written straight
        # into the labels.
        doc = {"id": "ffbb0019", "author": ["PHILIP MORRIS"], "documentdate": "1996 January 24"}
        page = mods["ucsf"].doc_to_page(doc, "/tmp/x.png", 1700, 2200, letterhead_author="PHILIP MORRIS")
        (mark,) = page.marks
        assert mark.class_id is None
        assert mark.provenance == "candidate"
        assert page.meta["letterhead_author"] == "PHILIP MORRIS"

    def test_candidate_carries_a_locatable_band(self, mods):
        page = mods["ucsf"].doc_to_page(
            {"id": "ffbb0019"}, "/tmp/x.png", 1700, 2200, letterhead_author="RJR", band_frac=0.2
        )
        # A mark nobody can see cannot be adjudicated, so the candidate gets a
        # coarse top-of-page strip to cluster on -- never a ground-truth box.
        assert page.marks[0].box == (0, 0, 1700, 440)

    def test_the_year_never_reaches_a_class_id(self, mods):
        doc = {"id": "ffbb0019", "documentdate": "1965 May 3"}
        page = mods["ucsf"].doc_to_page(doc, "/tmp/x.png", 100, 100, letterhead_author="RJR")
        assert page.meta["year"] == 1965
        # Era is a fact about the calendar, not about the mark. A class means
        # "this artwork" and nothing else.
        assert page.marks[0].class_id is None

    def test_a_page_with_no_author_carries_no_marks(self, mods):
        page = mods["ucsf"].doc_to_page({"id": "ffbb0019"}, "/tmp/x.png", 10, 10)
        assert page.marks == []


class TestArtworkVoc:
    def test_parses_pascal_voc_boxes(self, mods):
        xml = """<annotation><object><name>Nike</name>
                 <bndbox><xmin>10</xmin><ymin>20</ymin><xmax>110</xmax><ymax>70</ymax></bndbox>
                 </object></annotation>"""
        assert mods["artwork"].parse_voc(xml) == [("Nike", (10, 20, 100, 50))]

    def test_skips_degenerate_boxes(self, mods):
        xml = """<annotation><object><name>X</name>
                 <bndbox><xmin>10</xmin><ymin>20</ymin><xmax>10</xmax><ymax>70</ymax></bndbox>
                 </object></annotation>"""
        assert mods["artwork"].parse_voc(xml) == []


# ------------------------------------------------------------- contamination


class TestContamination:
    def test_tobacco800_may_not_use_ucsf_tobacco_as_distractors(self, mods):
        # Both are IIT-CDIP. A UCSF tobacco page is certain to carry more
        # instances of these same letterheads, so scoring against it counts
        # correct retrievals as false positives.
        assert not mods["cfg"].eligible_distractor("tobacco800", "ucsf", "Tobacco")

    def test_tobacco800_may_use_other_ucsf_industries(self, mods):
        assert mods["cfg"].eligible_distractor("tobacco800", "ucsf", "Opioids")

    def test_spods_may_use_ucsf_freely(self, mods):
        assert mods["cfg"].eligible_distractor("spods", "ucsf", "Tobacco")

    def test_no_source_is_its_own_distractor(self, mods):
        for source in ("spods", "staver", "tobacco800", "ucsf", "synth"):
            assert not mods["cfg"].eligible_distractor(source, source)


# ------------------------------------------------------------ class admission


class TestClassAdmission:
    def _corpus(self, mods, big=12, small=3):
        pages = []
        for i in range(big):
            pages.append(_page(mods, f"spods/{i:03d}", "spods", [("logo", (0, 0, 200, 120), "spods/a", "gt")]))
        for i in range(small):
            pages.append(_page(mods, f"spods/x{i:03d}", "spods", [("logo", (0, 0, 200, 120), "spods/b", "gt")]))
        return pages

    def test_survival_curve_counts_classes_per_threshold(self, mods):
        pages = self._corpus(mods)
        inv = mods["build"].class_inventory(pages)
        curve = mods["build"].survival_curve(inv, (2, 5, 10, 20))
        assert curve == {2: 2, 5: 1, 10: 1, 20: 0}

    def test_min_instances_rejects_the_thin_class(self, mods):
        pages = self._corpus(mods)
        inv = mods["build"].class_inventory(pages)
        admitted, rejected = mods["build"].admit_classes(pages, inv, min_instances=10, min_mark_px=32)
        assert set(admitted) == {"spods/a"}
        assert "instance(s) < min_instances" in rejected["spods/b"]

    def test_tiny_marks_are_rejected_with_a_reason(self, mods):
        pages = [
            _page(mods, f"spods/{i:03d}", "spods", [("logo", (0, 0, 12, 10), "spods/tiny", "gt")]) for i in range(20)
        ]
        inv = mods["build"].class_inventory(pages)
        admitted, rejected = mods["build"].admit_classes(pages, inv, min_instances=5, min_mark_px=32)
        assert admitted == {}
        assert "min_mark_px" in rejected["spods/tiny"]

    def test_signatures_are_never_queryable(self, mods):
        pages = [
            _page(mods, f"t/{i:03d}", "tobacco800", [("signature", (0, 0, 400, 200), "tobacco800/signature_x", "gt")])
            for i in range(30)
        ]
        inv = mods["build"].class_inventory(pages)
        admitted, rejected = mods["build"].admit_classes(pages, inv, min_instances=5, min_mark_px=32)
        assert admitted == {}
        assert "not queryable" in rejected["tobacco800/signature_x"]

    def test_band_classes_skip_the_mark_size_floor(self, mods):
        # A band's pixel size describes the top-of-page strip, not the mark, so
        # checking it against the 32px mark floor compares the wrong number
        # against the wrong threshold -- and reporting it as median_mark_px
        # would misdescribe the class.
        pages = [
            _page(mods, f"ucsf/{i:03d}", "ucsf", [("logo", (0, 0, 1700, 440), "ucsf/logo_a_0", "clustered_band")])
            for i in range(40)
        ]
        inv = mods["build"].class_inventory(pages)
        admitted, _ = mods["build"].admit_classes(pages, inv, min_instances=10, min_mark_px=32)
        assert admitted["ucsf/logo_a_0"]["located_by"] == "band"
        assert admitted["ucsf/logo_a_0"]["median_mark_px"] is None

    def test_unlocated_classes_are_rejected(self, mods):
        pages = [
            _page(mods, f"ucsf/{i:03d}", "ucsf", [("logo", (0, 0, 0, 0), "ucsf/nowhere", "candidate")])
            for i in range(40)
        ]
        inv = mods["build"].class_inventory(pages)
        admitted, rejected = mods["build"].admit_classes(pages, inv, min_instances=10, min_mark_px=32)
        assert admitted == {}
        assert "no located instances" in rejected["ucsf/nowhere"]

    def test_admitted_classes_record_their_eligible_distractors(self, mods):
        pages = self._corpus(mods)
        inv = mods["build"].class_inventory(pages)
        admitted, _ = mods["build"].admit_classes(pages, inv, min_instances=10, min_mark_px=32)
        eligible = admitted["spods/a"]["eligible_distractor_sources"]
        assert "spods" not in eligible
        assert "ucsf" in eligible


# ------------------------------------------------------------------- roster


class TestRoster:
    def _pages(self, mods, n_a=14, n_b=3):
        pages = [
            _page(mods, f"spods/a{i:03d}", "spods", [("logo", (0, 0, 200, 120), "spods/a", "clustered")])
            for i in range(n_a)
        ]
        pages += [
            _page(mods, f"spods/b{i:03d}", "spods", [("logo", (0, 0, 12, 10), "spods/b", "clustered")])
            for i in range(n_b)
        ]
        return pages

    def _admit(self, mods, pages, roster=None):
        inv = mods["build"].class_inventory(pages)
        return mods["build"].admit_classes(pages, inv, min_instances=10, min_mark_px=32, roster=roster)

    def test_without_a_roster_the_bars_decide(self, mods):
        admitted, rejected = self._admit(mods, self._pages(mods))
        assert set(admitted) == {"spods/a"}
        assert "spods/b" in rejected

    def test_a_roster_restricts_admission_to_its_own_classes(self, mods):
        roster = mods["roster"].Roster(name="t", classes=["spods/a"])
        admitted, rejected = self._admit(mods, self._pages(mods), roster)
        assert set(admitted) == {"spods/a"}
        assert rejected["spods/b"] == "not on the roster"

    def test_a_roster_class_overrides_the_bars_but_records_why(self, mods):
        # The human who picked it knows something the threshold does not; the
        # override is kept visible in the artifact rather than silently waived.
        roster = mods["roster"].Roster(name="t", classes=["spods/a", "spods/b"])
        admitted, _ = self._admit(mods, self._pages(mods), roster)
        assert set(admitted) == {"spods/a", "spods/b"}
        assert admitted["spods/a"]["caveats"] == []
        assert any("min_instances" in c for c in admitted["spods/b"]["caveats"])
        assert any("min_mark_px" in c for c in admitted["spods/b"]["caveats"])

    def test_classes_start_unverified(self, mods):
        roster = mods["roster"].Roster(name="t", classes=["spods/a"])
        admitted, _ = self._admit(mods, self._pages(mods), roster)
        # Until the membership pass runs, a class is a clustering proposal.
        assert admitted["spods/a"]["audit"]["membership_verified"] is False
        assert admitted["spods/a"]["on_roster"] is True

    def test_check_reports_drift_between_roster_and_corpus(self, mods):
        roster = mods["roster"].Roster(name="t", classes=["spods/a", "spods/gone"])
        present, missing = mods["roster"].check(roster, ["spods/a", "spods/b"])
        assert present == ["spods/a"]
        assert missing == ["spods/gone"]

    def test_roster_round_trips_and_deduplicates(self, mods, tmp_path):
        path = tmp_path / "roster.json"
        mods["roster"].save(mods["roster"].Roster("t", ["b", "a", "b"], notes="why"), path)
        back = mods["roster"].load(path)
        assert back.classes == ["a", "b"]
        assert back.notes == "why"

    def test_known_negatives_come_only_from_verified_sources(self, mods):
        meta = {
            "page_ids": ["spods/a000"],
            "eligible_distractor_sources": ["ucsf", "synth"],
        }
        pages_by_source = {
            "spods": ["spods/a000", "spods/x001"],
            "ucsf": ["ucsf/d1", "ucsf/d2"],
        }
        # SPODS contaminates SPODS by default -- but once SPODS has been
        # exhaustively checked for this class, its non-members become *known*
        # negatives: same scanner, same paper, verified clean, which is the
        # hardest and most useful negative there is.
        split = mods["roster"].eligible_pages(meta, pages_by_source, verified_negative_sources=["spods"])
        assert split["positive"] == ["spods/a000"]
        assert split["known_negative"] == ["spods/x001"]
        assert split["presumed_negative"] == ["ucsf/d1", "ucsf/d2"]

    def test_without_verification_same_source_pages_are_not_usable(self, mods):
        meta = {"page_ids": ["spods/a000"], "eligible_distractor_sources": ["ucsf"]}
        split = mods["roster"].eligible_pages(meta, {"spods": ["spods/a000", "spods/x001"], "ucsf": ["ucsf/d1"]})
        assert split["known_negative"] == []
        assert split["presumed_negative"] == ["ucsf/d1"]


class TestMembershipAudit:
    def _setup(self, mods):
        pages = [
            _page(mods, f"spods/{i:03d}", "spods", [("logo", (0, 0, 200, 120), "spods/a", "clustered")])
            for i in range(5)
        ]
        classes = {
            "spods/a": {
                "class_id": "spods/a",
                "n_instances": 5,
                "page_ids": [f"spods/{i:03d}" for i in range(5)],
                "audit": {"membership_verified": False, "rejected_page_ids": []},
            }
        }
        return pages, classes

    def test_ok_verifies_without_dropping_anything(self, mods):
        pages, classes = self._setup(mods)
        row = {"class_id": "spods/a", "page_ids": classes["spods/a"]["page_ids"], "verdict": "ok"}
        changes, problems = mods["audit"].apply_membership(pages, classes, [row])
        assert not problems
        assert classes["spods/a"]["n_instances"] == 5
        assert classes["spods/a"]["audit"]["membership_verified"] is True

    def test_rejected_indices_are_removed_from_the_class(self, mods):
        pages, classes = self._setup(mods)
        row = {"class_id": "spods/a", "page_ids": classes["spods/a"]["page_ids"], "verdict": "1, 3"}
        mods["audit"].apply_membership(pages, classes, [row])
        assert classes["spods/a"]["page_ids"] == ["spods/000", "spods/002", "spods/004"]
        assert classes["spods/a"]["audit"]["rejected_page_ids"] == ["spods/001", "spods/003"]

    def test_a_rejected_instance_keeps_its_box_and_page(self, mods):
        # It stops being a positive, but the page stays a *known* negative and
        # the mark is still a real mark a later roster might want.
        pages, classes = self._setup(mods)
        row = {"class_id": "spods/a", "page_ids": classes["spods/a"]["page_ids"], "verdict": "1"}
        mods["audit"].apply_membership(pages, classes, [row])
        dropped = next(p for p in pages if p.page_id == "spods/001")
        assert len(dropped.marks) == 1
        assert dropped.marks[0].class_id is None
        assert dropped.marks[0].box == (0, 0, 200, 120)

    def test_an_out_of_range_index_is_refused_not_silently_clamped(self, mods):
        pages, classes = self._setup(mods)
        row = {"class_id": "spods/a", "page_ids": classes["spods/a"]["page_ids"], "verdict": "9"}
        changes, problems = mods["audit"].apply_membership(pages, classes, [row])
        assert not changes
        assert "outside 0..4" in problems[0]
        assert classes["spods/a"]["n_instances"] == 5

    def test_a_malformed_verdict_is_refused(self, mods):
        pages, classes = self._setup(mods)
        row = {"class_id": "spods/a", "page_ids": classes["spods/a"]["page_ids"], "verdict": "maybe"}
        _changes, problems = mods["audit"].apply_membership(pages, classes, [row])
        assert "must be 'ok' or comma-separated indices" in problems[0]


class TestMergeSlateOrdering:
    """The slate's numbering is what the reviewer's answer refers to."""

    def test_seriation_puts_near_identical_classes_next_to_each_other(self, mods):
        # Three tight pairs, scrambled: 0~4, 1~3, 2~5.
        far = 0.9
        d = np.full((6, 6), far)
        for a, b in ((0, 4), (1, 3), (2, 5)):
            d[a, b] = d[b, a] = 0.01
        np.fill_diagonal(d, 0.0)
        order = mods["slate"].seriate(d)
        assert sorted(order) == list(range(6))
        adjacent = {frozenset(pair) for pair in zip(order, order[1:])}
        for pair in ((0, 4), (1, 3), (2, 5)):
            assert frozenset(pair) in adjacent, f"{pair} was split by the ordering"

    def test_seriation_is_deterministic_including_ties(self, mods):
        # An all-equal matrix is nothing but ties; a re-render must still
        # produce the same numbering or a half-finished merges.txt goes stale.
        d = np.full((8, 8), 0.5)
        np.fill_diagonal(d, 0.0)
        assert mods["slate"].seriate(d) == mods["slate"].seriate(d.copy())

    def test_seriation_of_an_empty_or_single_slate(self, mods):
        assert mods["slate"].seriate(np.zeros((0, 0))) == []
        assert mods["slate"].seriate(np.zeros((1, 1))) == [0]

    def test_near_pairs_are_the_closest_ones_nearest_first(self, mods):
        d = np.array([[0.0, 0.3, 0.1], [0.3, 0.0, 0.2], [0.1, 0.2, 0.0]])
        assert mods["slate"].near_pairs(d, 2) == [(0.1, 0, 2), (0.2, 1, 2)]


class TestMergeAnswerParsing:
    def test_a_group_is_a_line_of_indices(self, mods):
        groups, reviewed, problems = mods["audit"].parse_merge_groups("3 8 12\n", 20)
        assert not problems and reviewed is False
        assert [g["indices"] for g in groups] == [[3, 8, 12]]

    def test_commas_and_trailing_notes_are_accepted(self, mods):
        groups, _reviewed, problems = mods["audit"].parse_merge_groups("3, 8   # same elephant, blue and red\n", 20)
        assert not problems
        assert groups[0]["indices"] == [3, 8]
        assert groups[0]["note"] == "same elephant, blue and red"

    def test_overlapping_groups_are_unioned_not_refused(self, mods):
        # "3 8" and "8 12" are two observations of one equivalence class. A
        # reviewer writing the same truth twice is redundant, not contradictory.
        groups, _reviewed, problems = mods["audit"].parse_merge_groups("3 8\n8 12\n", 20)
        assert not problems
        assert [g["indices"] for g in groups] == [[3, 8, 12]]

    def test_comments_and_blank_lines_are_ignored(self, mods):
        groups, reviewed, problems = mods["audit"].parse_merge_groups("# a header\n\n   \n1 2\n", 5)
        assert not problems and [g["indices"] for g in groups] == [[1, 2]]

    def test_reviewed_all_is_a_line_of_its_own(self, mods):
        _groups, reviewed, problems = mods["audit"].parse_merge_groups("REVIEWED-ALL\n", 5)
        assert reviewed is True and not problems

    @pytest.mark.parametrize(
        "line, fragment",
        [
            ("99 1", "outside the slate"),
            ("foo 2", "not a slate index"),
            ("4", "fewer than two distinct classes"),
            ("3 3", "fewer than two distinct classes"),
        ],
    )
    def test_a_typo_is_refused_never_guessed_at(self, mods, line, fragment):
        # Every one of these silently reinterpreted would write a permanent
        # merge between classes nobody looked at.
        groups, _reviewed, problems = mods["audit"].parse_merge_groups(line + "\n", 6)
        assert groups == []
        assert fragment in problems[0]


class TestMergeVerdictCompilation:
    def _index(self, n=6, near=((0, 1, 0.01), (2, 3, 0.02), (4, 5, 0.03))):
        return {
            "classes": [{"index": i, "class_id": f"spods/c{i}", "n_instances": 5} for i in range(n)],
            "near_pairs": [
                {"rank": r, "left_index": a, "right_index": b, "distance": d} for r, (a, b, d) in enumerate(near)
            ],
        }

    def test_a_group_compiles_to_a_star_not_a_clique(self, mods):
        # Sameness is transitive and apply_confusable merges outright, so n-1
        # rows state the whole group; n(n-1)/2 would just restate it.
        groups, reviewed, _ = mods["audit"].parse_merge_groups("0 1 2\n", 6)
        rows, problems = mods["audit"].merge_verdicts(self._index(), groups, reviewed)
        assert not problems
        assert [(r["left_class_id"], r["right_class_id"]) for r in rows] == [
            ("spods/c0", "spods/c1"),
            ("spods/c0", "spods/c2"),
        ]

    def test_without_reviewed_all_only_merges_are_recorded(self, mods):
        groups, reviewed, _ = mods["audit"].parse_merge_groups("0 1\n", 6)
        rows, _problems = mods["audit"].merge_verdicts(self._index(), groups, reviewed)
        assert {r["verdict"] for r in rows} == {"same"}

    def test_reviewed_all_separates_the_appendix_pairs_and_only_those(self, mods):
        # The closed world covers what the reviewer was actually shown: the
        # near-pair sheets. Pairs that live only at the far end of the ranking
        # were never compared and stay unadjudicated.
        groups, reviewed, _ = mods["audit"].parse_merge_groups("REVIEWED-ALL\n", 6)
        rows, _problems = mods["audit"].merge_verdicts(self._index(), groups, reviewed)
        different = {(r["left_class_id"], r["right_class_id"]) for r in rows if r["verdict"] == "different"}
        assert different == {("spods/c0", "spods/c1"), ("spods/c2", "spods/c3"), ("spods/c4", "spods/c5")}
        assert ("spods/c0", "spods/c5") not in different

    def test_a_merged_near_pair_is_never_also_separated(self, mods):
        # save_adjudications refuses a pair ruled both ways outright, so this is
        # a correctness gate rather than tidiness: without it, merging an
        # appendix pair on a REVIEWED-ALL slate would abort the whole apply.
        groups, reviewed, _ = mods["audit"].parse_merge_groups("0 1\nREVIEWED-ALL\n", 6)
        rows, _problems = mods["audit"].merge_verdicts(self._index(), groups, reviewed)
        ruled = {(r["left_class_id"], r["right_class_id"]): r["verdict"] for r in rows}
        assert ruled[("spods/c0", "spods/c1")] == "same"
        assert list(ruled.values()).count("same") == 1
        assert ("spods/c1", "spods/c0") not in ruled

    def test_transitively_merged_classes_do_not_separate_each_other(self, mods):
        # 0~1 and 1~2 makes {0,1,2} one class, so an appendix pair (0,2) is
        # inside the group even though no line named that pair.
        index = self._index(near=((0, 2, 0.01),))
        groups, reviewed, _ = mods["audit"].parse_merge_groups("0 1\n1 2\nREVIEWED-ALL\n", 6)
        rows, _problems = mods["audit"].merge_verdicts(index, groups, reviewed)
        assert [r["verdict"] for r in rows] == ["same", "same"]

    def test_every_same_row_is_emitted_before_every_different_row(self, mods):
        # apply_confusable merges as it goes and follows the chain afterwards;
        # a separation pinned first would name a class about to stop existing.
        groups, reviewed, _ = mods["audit"].parse_merge_groups("0 1\nREVIEWED-ALL\n", 6)
        rows, _problems = mods["audit"].merge_verdicts(self._index(), groups, reviewed)
        verdicts = [r["verdict"] for r in rows]
        assert verdicts == sorted(verdicts, key=lambda v: v != "same")

    def test_pairs_that_name_the_same_two_post_merge_classes_are_stated_once(self, mods):
        # Once 0 and 1 are one class, the appendix pairs (0, 2) and (1, 2) are
        # one statement about one pair of classes. Emitting both prints a
        # contradiction-shaped log for a decision that was made once.
        index = self._index(near=((0, 1, 0.01), (0, 2, 0.02), (1, 2, 0.03)))
        groups, reviewed, _ = mods["audit"].parse_merge_groups("0 1\nREVIEWED-ALL\n", 6)
        rows, _problems = mods["audit"].merge_verdicts(index, groups, reviewed)
        different = [r for r in rows if r["verdict"] == "different"]
        assert len(different) == 1
        assert "rank 1" in different[0]["notes"], "the nearest of the redundant pairs is the one kept"

    def test_an_index_missing_from_the_slate_is_reported(self, mods):
        index = self._index()
        index["classes"] = index["classes"][:3]
        rows, problems = mods["audit"].merge_verdicts(index, [{"indices": [0, 5], "note": ""}], False)
        assert rows == []
        assert "not on the slate" in problems[0]


class TestMergeSlateEndToEnd:
    """The slate is an input format, not a second path through the ground truth."""

    def _corpus(self, mods):
        pages = [
            _page(mods, f"spods/{c}{i}", "spods", [("logo", (0, 0, 200, 120), f"spods/{c}", "clustered")])
            for c in "abcd"
            for i in range(3)
        ]
        classes = {
            f"spods/{c}": {
                "class_id": f"spods/{c}",
                "n_instances": 3,
                "page_ids": [f"spods/{c}{i}" for i in range(3)],
                "audit": {"membership_verified": False, "rejected_page_ids": []},
            }
            for c in "abcd"
        }
        return pages, classes

    def _index(self, near):
        return {
            "classes": [{"index": i, "class_id": f"spods/{c}", "n_instances": 3} for i, c in enumerate("abcd")],
            "near_pairs": [
                {"rank": r, "left_index": a, "right_index": b, "distance": 0.01 * (r + 1)}
                for r, (a, b) in enumerate(near)
            ],
        }

    def test_a_slate_answer_merges_and_separates_through_the_pairwise_applier(self, mods):
        pages, classes = self._corpus(mods)
        groups, reviewed, _ = mods["audit"].parse_merge_groups("0 1\nREVIEWED-ALL\n", 4)
        rows, _ = mods["audit"].merge_verdicts(self._index([(0, 1), (2, 3)]), groups, reviewed)
        _changes, problems, separations, merges = mods["audit"].apply_confusable(pages, classes, rows)
        assert not problems
        assert "spods/b" not in classes
        assert classes["spods/a"]["n_instances"] == 6
        assert len(merges) == 1 and len(separations) == 1
        assert classes["spods/c"]["distinct_from"] == ["spods/d"]

    def test_the_resulting_adjudications_never_rule_a_pair_both_ways(self, mods, tmp_path):
        # save_adjudications raises on a conflicting pair by design. Compiling a
        # slate that merges one of its own appendix pairs must not trip it.
        pages, classes = self._corpus(mods)
        groups, reviewed, _ = mods["audit"].parse_merge_groups("0 1\n2 3\nREVIEWED-ALL\n", 4)
        rows, _ = mods["audit"].merge_verdicts(self._index([(0, 1), (2, 3), (0, 2)]), groups, reviewed)
        _c, _p, separations, merges = mods["audit"].apply_confusable(pages, classes, rows)
        mods["cluster"].save_adjudications(merges, separations, tmp_path / "adjudications.json")
        same, diff = mods["cluster"].load_adjudications(tmp_path / "adjudications.json")
        assert not (set(map(frozenset, same)) & set(map(frozenset, diff)))
        assert len(same) == 2 and len(diff) == 1

    def test_separations_are_keyed_on_page_ids_so_they_survive_a_recluster(self, mods):
        pages, classes = self._corpus(mods)
        groups, reviewed, _ = mods["audit"].parse_merge_groups("REVIEWED-ALL\n", 4)
        rows, _ = mods["audit"].merge_verdicts(self._index([(0, 1)]), groups, reviewed)
        _c, _p, separations, _m = mods["audit"].apply_confusable(pages, classes, rows)
        assert separations[0]["left_page_id"] == "spods/a0"
        assert separations[0]["right_page_id"] == "spods/b0"


# -------------------------------------------------------------------- tiers


class TestTiers:
    def _pages(self, mods, n_distractors=500):
        pages = [
            _page(mods, f"spods/{i:03d}", "spods", [("logo", (0, 0, 200, 120), "spods/a", "gt")]) for i in range(20)
        ]
        pages += [_page(mods, f"ucsf/d{i:05d}", "ucsf") for i in range(n_distractors)]
        return pages

    def _assign(self, mods, pages, tiers, pinned=None):
        inv = mods["build"].class_inventory(pages)
        admitted, _ = mods["build"].admit_classes(pages, inv, min_instances=10, min_mark_px=32)
        return mods["build"].assign_tiers(
            pages, admitted, tiers=tiers, tier_order=("s", "m", "l"), salt="test-salt", pinned_cutoffs=pinned
        )

    @staticmethod
    def _members(tier_of, tier_order=("s", "m", "l")):
        """Cumulative membership per tier, since tiers nest."""
        out, running = {}, set()
        for tier in tier_order:
            running = running | {p for p, t in tier_of.items() if t == tier}
            out[tier] = set(running)
        return out

    def test_tiers_are_nested_and_hit_their_budgets(self, mods):
        tier_of, _ = self._assign(mods, self._pages(mods), {"s": 60, "m": 200, "l": 400})
        m = self._members(tier_of)
        assert m["s"] < m["m"] < m["l"]
        assert (len(m["s"]), len(m["m"]), len(m["l"])) == (60, 200, 400)

    def test_every_positive_page_is_in_the_smallest_tier(self, mods):
        tier_of, _ = self._assign(mods, self._pages(mods), {"s": 60, "m": 200, "l": 400})
        assert all(tier_of[f"spods/{i:03d}"] == "s" for i in range(20))

    def test_is_deterministic_for_a_fixed_page_set(self, mods):
        budgets = {"s": 60, "m": 200, "l": 400}
        first, _ = self._assign(mods, self._pages(mods, 500), budgets)
        second, _ = self._assign(mods, self._pages(mods, 500), budgets)
        assert first == second

    def test_pinned_cutoffs_survive_a_growing_source_pool(self, mods):
        # Budgets and cross-build stability genuinely conflict: you cannot hold
        # a page count fixed *and* hold membership fixed when the pool changes
        # size. Pinning the rank cutoffs buys stability and lets the count
        # drift, which is the trade a follow-up build wants so that its numbers
        # stay comparable to the earlier one's.
        budgets = {"s": 60, "m": 200, "l": 400}
        _, cutoffs = self._assign(mods, self._pages(mods, 500), budgets)
        small, _ = self._assign(mods, self._pages(mods, 500), budgets, pinned=cutoffs)
        grown, _ = self._assign(mods, self._pages(mods, 900), budgets, pinned=cutoffs)

        s_small = self._members(small)["s"]
        s_grown = self._members(grown)["s"]
        assert s_small <= s_grown, "pinning must never evict a page from a tier it was already in"

    def test_unpinned_growth_is_documented_as_a_new_corpus_version(self, mods):
        # The converse of the test above, pinned here so nobody "fixes" the
        # default into silent instability without noticing: without pinning, a
        # larger pool re-selects, and that is a new corpus, not the same one.
        budgets = {"s": 60, "m": 200, "l": 400}
        before, _ = self._assign(mods, self._pages(mods, 500), budgets)
        after, _ = self._assign(mods, self._pages(mods, 900), budgets)
        assert self._members(before)["s"] != self._members(after)["s"]

    def test_pages_past_the_largest_budget_are_excluded(self, mods):
        pages = self._pages(mods, 500)
        tier_of, _ = self._assign(mods, pages, {"s": 60, "m": 100, "l": 150})
        assert len(tier_of) == 150
        assert len(pages) > len(tier_of)


# --------------------------------------------------------------- clustering


class TestClustering:
    def test_phash_is_deterministic_and_sized_by_the_block(self, mods):
        from PIL import Image

        rng = np.random.default_rng(42)
        img = Image.fromarray(rng.integers(0, 255, (120, 90), dtype=np.uint8))
        a, b = mods["cluster"].phash(img), mods["cluster"].phash(img)
        assert a.shape == (mods["cluster"].PHASH_BLOCK ** 2,)
        assert np.array_equal(a, b)

    def test_the_hash_is_wide_enough_to_separate_two_ringed_stamps(self, mods):
        # 64 bits could not: a book stamp and an elephant stamp, both circular
        # with a heavy border, merged into one class of 32 and no threshold
        # split them. A stamp's ring is low-frequency and its interior is not,
        # so an 8x8 block encodes "is a round stamp" rather than which one.
        assert mods["cluster"].PHASH_BLOCK >= 16

    def test_the_radial_taper_damps_the_border_not_the_middle(self, mods):
        arr = np.full((64, 64), 10.0)
        arr[2, 2] = 250.0  # corner ink, where a border ring lives
        arr[32, 32] = 250.0  # centre ink, where the mark's identity lives
        out = mods["cluster"]._radial_taper(arr)
        centre_kept = (out[32, 32] - arr.mean()) / (250.0 - arr.mean())
        corner_kept = (out[2, 2] - arr.mean()) / (250.0 - arr.mean())
        assert centre_kept > 0.99
        assert corner_kept < 0.01

    def test_the_taper_leaves_a_flat_crop_flat(self, mods):
        # Fading toward the crop's own mean, not toward white: fading to white
        # would replace the border ring with a different strong edge.
        arr = np.full((32, 32), 7.5)
        assert np.allclose(mods["cluster"]._radial_taper(arr), 7.5)

    def test_phash_is_scale_invariant(self, mods):
        from PIL import Image

        rng = np.random.default_rng(7)
        base = Image.fromarray(rng.integers(0, 255, (200, 200), dtype=np.uint8)).filter(
            __import__("PIL.ImageFilter", fromlist=["GaussianBlur"]).GaussianBlur(3)
        )
        big, small = mods["cluster"].phash(base), mods["cluster"].phash(base.resize((80, 80)))
        assert (big != small).mean() < 0.15

    def test_single_linkage_chains_and_separates(self, mods):
        dist = np.array(
            [
                [0.0, 0.1, 0.9, 0.9],
                [0.1, 0.0, 0.15, 0.9],
                [0.9, 0.15, 0.0, 0.9],
                [0.9, 0.9, 0.9, 0.0],
            ]
        )
        labels = mods["cluster"].single_linkage(dist, threshold=0.2)
        assert labels[0] == labels[1] == labels[2]
        assert labels[3] != labels[0]

    def test_labels_are_dense_and_first_appearance_ordered(self, mods):
        dist = np.ones((4, 4)) - np.eye(4)
        labels = mods["cluster"].single_linkage(dist, threshold=0.0)
        assert labels == [0, 1, 2, 3]

    def test_aspect_gate_forces_apart_marks_of_different_shape(self, mods):
        MarkRef = mods["cluster"].MarkRef
        refs = [
            MarkRef(0, 0, "p/1", "logo", (0, 0, 100, 100)),  # square
            MarkRef(1, 0, "p/2", "logo", (0, 0, 400, 40)),  # wide banner
        ]
        desc = np.ones((2, 64), dtype=bool)  # identical hashes
        dist = mods["cluster"].distance_matrix(desc, refs, backend="phash")
        assert dist[0, 1] == 1.0

    def test_cannot_link_keeps_adjudicated_marks_apart(self, mods):
        # Two crops close enough that the threshold would merge them, which a
        # human has said are different marks. The whole point of recording that
        # is that it survives the clustering that would otherwise overrule it.
        dist = np.array([[0.0, 0.05], [0.05, 0.0]])
        assert mods["cluster"].single_linkage(dist, 0.2) == [0, 0]
        assert mods["cluster"].single_linkage(dist, 0.2, cannot_link=[(0, 1)]) == [0, 1]

    def test_a_separation_propagates_through_a_third_crop(self, mods):
        # a-b are separated; c is near both. Without propagation c would merge
        # with a and then with b, reuniting the pair through the back door.
        dist = np.array(
            [
                [0.0, 0.9, 0.05],
                [0.9, 0.0, 0.05],
                [0.05, 0.05, 0.0],
            ]
        )
        labels = mods["cluster"].single_linkage(dist, 0.2, cannot_link=[(0, 1)])
        assert labels[0] != labels[1]

    def test_pairs_resolve_by_page_id_not_row_index(self, mods):
        MarkRef = mods["cluster"].MarkRef
        refs = [
            MarkRef(0, 0, "spods/001", "logo", (0, 0, 10, 10)),
            MarkRef(1, 0, "spods/002", "logo", (0, 0, 10, 10)),
        ]
        assert mods["cluster"].resolve_pairs(refs, [("spods/001", "spods/002")]) == [(0, 1)]

    def test_a_pair_naming_a_dropped_page_is_skipped(self, mods):
        MarkRef = mods["cluster"].MarkRef
        refs = [MarkRef(0, 0, "spods/001", "logo", (0, 0, 10, 10))]
        # Pages come and go with tier budgets; a stale pair must not refuse the
        # build.
        assert mods["cluster"].resolve_pairs(refs, [("spods/001", "spods/999")]) == []

    def test_adjudications_round_trip_and_deduplicate(self, mods, tmp_path):
        path = tmp_path / "adjudications.json"
        mods["cluster"].save_adjudications(
            [{"left_page_id": "d", "right_page_id": "c"}],
            [{"left_page_id": "b", "right_page_id": "a"}, {"left_page_id": "a", "right_page_id": "b"}],
            path,
        )
        same, different = mods["cluster"].load_adjudications(path)
        # (a, b) and (b, a) are one decision, not two.
        assert same == [("c", "d")]
        assert different == [("a", "b")]

    def test_no_adjudication_file_means_no_constraints(self, mods, tmp_path):
        assert mods["cluster"].load_adjudications(tmp_path / "missing.json") == ([], [])

    def test_a_pair_ruled_both_ways_is_refused(self, mods, tmp_path):
        # Storing both would let whichever is applied last silently win, and the
        # loser is a human decision nobody would know had been discarded.
        with pytest.raises(ValueError, match="both same and different"):
            mods["cluster"].save_adjudications(
                [{"left_page_id": "a", "right_page_id": "b"}],
                [{"left_page_id": "a", "right_page_id": "b"}],
                tmp_path / "adjudications.json",
            )

    def test_must_link_joins_marks_the_threshold_would_split(self, mods):
        # The operating strategy: run strict so the partition over-splits, then
        # repair by hand. A merge has to beat the distance, or the repair does
        # not survive the next re-cluster.
        dist = np.array([[0.0, 0.9], [0.9, 0.0]])
        assert mods["cluster"].single_linkage(dist, 0.1) == [0, 1]
        assert mods["cluster"].single_linkage(dist, 0.1, must_link=[(0, 1)]) == [0, 0]

    def test_a_merge_and_a_separation_that_conflict_are_refused(self, mods):
        dist = np.array([[0.0, 0.9], [0.9, 0.0]])
        with pytest.raises(ValueError, match="both same and different"):
            mods["cluster"].single_linkage(dist, 0.1, must_link=[(0, 1)], cannot_link=[(0, 1)])

    def test_a_merge_carries_its_group_across_a_separation(self, mods):
        # a must-link b; c is separated from a. c must therefore stay apart from
        # b too, or the separation is honoured only against the row that
        # happened to be named in it.
        dist = np.array([[0.0, 0.9, 0.01], [0.9, 0.0, 0.01], [0.01, 0.01, 0.0]])
        labels = mods["cluster"].single_linkage(dist, 0.1, must_link=[(0, 1)], cannot_link=[(0, 2)])
        assert labels[0] == labels[1]
        assert labels[2] != labels[0]

    def test_merge_order_is_independent_of_row_order(self, mods):
        # Once constraints can block a merge, "which merge happened first"
        # decides the outcome, so merges are applied in distance order rather
        # than whatever order the loops produce.
        dist = np.array(
            [
                [0.0, 0.10, 0.15],
                [0.10, 0.0, 0.05],
                [0.15, 0.05, 0.0],
            ]
        )
        assert mods["cluster"].single_linkage(dist, 0.12, cannot_link=[(0, 2)]) == [0, 1, 1]

    def test_class_ids_are_anchored_to_a_page_not_a_counter(self, mods):
        MarkRef = mods["cluster"].MarkRef
        pages = [_page(mods, "spods/00042", "spods", [("logo", (0, 0, 10, 10), None, "gt")])]
        refs = [MarkRef(0, 0, "spods/00042", "logo", (0, 0, 10, 10))]
        classes = mods["cluster"].assign_class_ids(pages, refs, [0], source="spods")
        assert list(classes) == ["spods/logo_00042_0"]
        assert pages[0].marks[0].provenance == "clustered"


# ---------------------------------------------------------------- synthesis


class TestResplitRegistersItsPieces:
    """A ``split`` verdict must leave the pieces somewhere the corpus can see.

    ``resplit_classes`` relabels the marks and pops the parent.  Everything
    downstream -- the slate, the roster, the embedder, the report -- reads
    ``classes.json``, and the only other writer is ``build_corpus.py``, which
    rebuilds from the sources and would discard the split.  So a piece that is
    not registered here is not "deferred to the next sheet"; it is gone, with
    its marks pointing at an id nothing knows.
    """

    def _two_marks_one_class(self, mods, tmp_path):
        """Two visibly different stamps sharing one class id, on real pages."""
        from PIL import Image

        pages = []
        for i in range(6):
            arr = np.full((_PAGE_H, _PAGE_W), 255, dtype=np.uint8)
            if i < 4:
                arr[200:260, 200:400] = 0  # a solid bar
            else:
                for k in range(12):  # a comb, nothing like the bar
                    arr[200:260, 200 + k * 16 : 204 + k * 16] = 0
            path = tmp_path / f"p{i}.png"
            Image.fromarray(arr).save(path)
            pages.append(
                _page(
                    mods,
                    f"src/{i:05d}",
                    "src",
                    marks=[("stamp", (200, 200, 200, 60), "src/stamp_00000_0", "clustered")],
                    path=str(path),
                )
            )
        return pages

    def test_every_piece_lands_in_classes_and_keeps_all_instances(self, mods, tmp_path):
        pages = self._two_marks_one_class(mods, tmp_path)
        inventory = mods["build"].class_inventory(pages)
        classes, _ = mods["build"].admit_classes(pages, inventory, min_instances=1, min_mark_px=1)
        assert set(classes) == {"src/stamp_00000_0"}

        notes = mods["audit"].resplit_classes(
            pages,
            classes,
            ["src/stamp_00000_0"],
            backend="phash",
            threshold=0.20,
            corpus=tmp_path,
            min_mark_px=1,
        )

        # A piece takes its id from its own smallest page id, so the parent's id
        # comes back as the id of whichever piece kept page 00000 -- what must
        # not survive is the parent's *entry*, with all six instances on it.
        assert len(classes) >= 2, f"pieces were not registered: {notes}"
        assert classes["src/stamp_00000_0"]["n_instances"] == 4
        assert sum(m["n_instances"] for m in classes.values()) == 6
        assert sorted(m["n_instances"] for m in classes.values()) == [2, 4]
        # Every relabelled mark resolves to a registered class.
        for page in pages:
            for mark in page.marks:
                assert mark.class_id in classes

    def test_a_singleton_piece_is_registered_rather_than_sized_out(self, mods, tmp_path):
        # The reviewer already said this class holds more than one mark, so a
        # one-instance piece is a finding.  Dropping it here would hide it from
        # the roster, which is the only place that should decide it is too
        # small to search.
        pages = self._two_marks_one_class(mods, tmp_path)
        pages = pages[:4] + pages[4:5]  # 4 bars, 1 comb
        inventory = mods["build"].class_inventory(pages)
        classes, _ = mods["build"].admit_classes(pages, inventory, min_instances=1, min_mark_px=1)
        mods["audit"].resplit_classes(
            pages,
            classes,
            ["src/stamp_00000_0"],
            backend="phash",
            threshold=0.20,
            corpus=tmp_path,
            min_mark_px=1,
        )
        sizes = sorted(m["n_instances"] for m in classes.values())
        assert sizes == [1, 4], sizes


class TestSiglipClusterBackend:
    """The ``siglip`` cluster backend named symbols that do not exist.

    It imported ``SiglipEmbedder`` (the class is ``ImageSiglipEmbedder``) and
    then called ``embed_images`` (the in-memory entry point is
    ``embed_pil_image``), so ``--cluster-backend siglip`` raised ImportError on
    every path that reached it, from the corpus builder's first commit onward.
    Both names are checked here rather than the clustering itself, because the
    failure was never in the maths -- it was in code no test had executed.
    """

    def test_the_backend_names_the_embedder_that_exists(self, mods):
        import inspect

        src = inspect.getsource(mods["cluster"].describe_marks)
        assert "ImageSiglipEmbedder" in src
        assert "import SiglipEmbedder" not in src

    def test_the_embedder_exposes_the_method_the_backend_calls(self, mods):
        import inspect

        from vtscore.media.image.embedder_siglip import ImageSiglipEmbedder

        src = inspect.getsource(mods["cluster"].describe_marks)
        assert "embed_pil_image" in src
        assert hasattr(ImageSiglipEmbedder, "embed_pil_image")
        assert not hasattr(ImageSiglipEmbedder, "embed_images")

    def test_an_unknown_backend_still_says_so(self, mods):
        with pytest.raises(ValueError, match="unknown cluster backend"):
            mods["cluster"].describe_marks([], [], backend="nope")


class TestSynthesis:
    def _artwork(self, size=(120, 60)):
        from PIL import Image, ImageDraw

        img = Image.new("RGBA", size, (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.ellipse([4, 4, size[0] - 5, size[1] - 5], fill=(20, 30, 200, 255))
        return img

    def test_paste_box_is_tight_around_the_rotated_alpha(self, mods):
        from PIL import Image

        page = Image.new("RGBA", (800, 1000), (255, 255, 255, 255))
        box = mods["synth"].paste_mark(page, self._artwork(), target_px=200, rotation_deg=30.0, position=(0.5, 0.5))
        x, y, w, h = box
        # A 30-degree rotation expands the paste rectangle well beyond the mark.
        # The recorded box must follow the ink, not the rectangle.
        assert w < 200 and h < 200
        crop = np.array(page.crop((x, y, x + w, y + h)))
        assert crop[..., 3].max() > 0
        # And it must be tight: every edge row/column touches ink.
        assert crop[0, :, 3].max() > 0 and crop[-1, :, 3].max() > 0

    def test_paste_lands_inside_the_page(self, mods):
        from PIL import Image

        page = Image.new("RGBA", (600, 800), (255, 255, 255, 255))
        for pos in ((0.0, 0.0), (1.0, 1.0), (0.5, 0.5)):
            x, y, w, h = mods["synth"].paste_mark(page, self._artwork(), target_px=120, rotation_deg=0.0, position=pos)
            assert 0 <= x and 0 <= y and x + w <= page.width and y + h <= page.height

    def test_build_synthetic_pages_records_exact_ground_truth(self, mods, tmp_path):
        from PIL import Image

        bg = tmp_path / "bg.png"
        Image.new("RGB", (900, 1200), "white").save(bg)
        pool_dir = tmp_path / "pool"
        pool_dir.mkdir()
        for name in ("alpha", "beta"):
            self._artwork().save(pool_dir / f"{name}.png")
        pool = mods["artwork"].load_pool_dir(pool_dir)

        pages = mods["synth"].build_synthetic_pages(
            [bg], pool, tmp_path / "out", instances_per_class=3, size_px=(64, 128), rotation_deg=(-5, 5), seed=1
        )
        assert len(pages) == 6
        assert {m.class_id for p in pages for m in p.marks} == {"synth/alpha", "synth/beta"}
        assert all(m.provenance == "synthetic" for p in pages for m in p.marks)
        assert all(m.area() > 0 for p in pages for m in p.marks)

    def test_synthesis_is_reproducible_from_the_seed(self, mods, tmp_path):
        from PIL import Image

        bg = tmp_path / "bg.png"
        Image.new("RGB", (900, 1200), "white").save(bg)
        pool_dir = tmp_path / "pool"
        pool_dir.mkdir()
        self._artwork().save(pool_dir / "alpha.png")
        pool = mods["artwork"].load_pool_dir(pool_dir)

        def run(out):
            return [
                (m.class_id, m.box)
                for p in mods["synth"].build_synthetic_pages(
                    [bg], pool, out, instances_per_class=4, size_px=(64, 128), rotation_deg=(-5, 5), seed=99
                )
                for m in p.marks
            ]

        assert run(tmp_path / "a") == run(tmp_path / "b")

    def test_empty_pool_is_an_error_not_an_empty_corpus(self, mods, tmp_path):
        with pytest.raises(ValueError, match="artwork pool"):
            mods["synth"].build_synthetic_pages(
                [tmp_path / "bg.png"],
                {},
                tmp_path,
                instances_per_class=1,
                size_px=(64, 128),
                rotation_deg=(0, 0),
                seed=1,
            )


# ----------------------------------------------------------------- manifest


class TestManifest:
    def test_round_trips_marks_and_meta(self, mods, tmp_path):
        pages = [
            _page(mods, "spods/001", "spods", [("logo", (1, 2, 3, 4), "spods/a", "clustered")]),
            _page(mods, "ucsf/x#0", "ucsf"),
        ]
        pages[1].meta = {"industry": "Opioids", "decade": "1990s"}
        path = tmp_path / "corpus.jsonl"

        assert mods["common"].write_manifest(pages, path) == 2
        back = list(mods["common"].read_manifest(path))
        assert [p.page_id for p in back] == ["spods/001", "ucsf/x#0"]
        assert back[0].marks[0].box == (1, 2, 3, 4)
        assert back[0].marks[0].provenance == "clustered"
        assert back[1].meta["industry"] == "Opioids"

    def test_written_records_are_stable_json(self, mods, tmp_path):
        pages = [_page(mods, "spods/001", "spods", [("logo", (1, 2, 3, 4), "spods/a", "gt")])]
        a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
        mods["common"].write_manifest(pages, a)
        mods["common"].write_manifest(list(mods["common"].read_manifest(a)), b)
        assert a.read_text() == b.read_text()

    def test_manifest_is_valid_jsonl(self, mods, tmp_path):
        path = tmp_path / "c.jsonl"
        mods["common"].write_manifest([_page(mods, "spods/1", "spods")], path)
        for line in path.read_text().splitlines():
            json.loads(line)


# ------------------------------------------------------------------ report


class TestReport:
    def test_scale_section_bands_against_the_measured_floor(self, mods, tmp_path):
        from PIL import Image

        img = tmp_path / "p.png"
        Image.new("RGB", (1000, 1400), "white").save(img)
        # One sub-floor mark and three above it.
        pages = [
            _page(mods, "spods/001", "spods", [("logo", (0, 0, 20, 18), "spods/a", "gt")], path=str(img)),
            _page(mods, "spods/002", "spods", [("logo", (0, 0, 90, 70), "spods/a", "gt")], path=str(img)),
            _page(mods, "spods/003", "spods", [("logo", (0, 0, 200, 150), "spods/a", "gt")], path=str(img)),
            _page(mods, "spods/004", "spods", [("logo", (0, 0, 300, 200), "spods/a", "gt")], path=str(img)),
        ]
        html = mods["report"].section_scale(pages, {})
        assert "32px" in html
        # One of four marks is below the floor; the share must be reported, not
        # buried, because a class built from sub-floor instances measures the
        # floor rather than the method.
        assert "25% of labelled marks fall below" in html

    def test_scale_section_is_empty_without_labelled_marks(self, mods):
        assert mods["report"].section_scale([_page(mods, "u/1", "ucsf")], {}) == ""

    def test_overview_warns_when_nothing_is_verified(self, mods):
        classes = {"spods/a": {"n_instances": 3, "audit": {"membership_verified": False}}}
        html = mods["report"].section_overview([_page(mods, "spods/1", "spods")], classes, {})
        assert "Nothing here is verified yet" in html

    def test_overview_drops_the_warning_once_verified(self, mods):
        classes = {"spods/a": {"n_instances": 3, "audit": {"membership_verified": True}}}
        html = mods["report"].section_overview([_page(mods, "spods/1", "spods")], classes, {})
        assert "Nothing here is verified yet" not in html

    def test_tables_escape_their_content(self, mods):
        html = mods["report"]._table(["c"], [["<script>x</script>"]])
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_every_provenance_the_builder_emits_has_a_gloss(self, mods):
        # A provenance with no explanation in the report is a label the reader
        # cannot interpret, which defeats the point of tracking provenance.
        emitted = {"gt", "clustered", "clustered_band", "candidate", "synthetic"}
        assert emitted <= set(mods["report"]._PROVENANCE_MEANING)


class TestReportWholePageFigure:
    """The "whole pages, marks boxed" figure and the caption that measures it.

    The caption's px/% is the number a reader quotes for "how small is the
    target", so it has to describe a mark that is actually a *target*.  Sized
    over every mark instead, an underlined heading welded into one component by
    its own rule wins the title on a real SPODS page — and because such a mark
    carries no ``class_id``, highlighting by class id then reddened nothing at
    all, leaving the prose promising a colour the figure never drew.
    """

    @staticmethod
    def _blank(tmp_path, size=(1000, 1400)):
        from PIL import Image

        path = tmp_path / "p.png"
        Image.new("RGB", size, "white").save(path)
        return str(path)

    @staticmethod
    def _text(html):
        """*html* with the inlined image bytes removed.

        A base64 payload is made of characters an assertion like ``"403px" not
        in html`` can match by chance, so the captions are checked against the
        markup only.
        """
        return re.sub(r'src="data:[^"]*"', "", html)

    @staticmethod
    def _colour_bbox(im, colour):
        arr = np.asarray(im)
        hit = np.all(arr == np.array(colour, dtype=arr.dtype), axis=-1)
        ys, xs = np.nonzero(hit)
        if not len(xs):
            return None
        return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

    def test_caption_measures_a_labelled_mark_not_the_biggest_box(self, mods, tmp_path):
        page = _page(
            mods,
            "spods/00622",
            "spods",
            [
                ("logo", (50, 50, 120, 100), "spods/logo_a", "clustered"),
                ("text", (60, 400, 403, 40), None, "gt"),  # an underlined heading
            ],
            path=self._blank(tmp_path),
        )
        html = self._text(mods["report"].section_full_pages([page], 4, 11))
        assert "120px" in html
        assert "403px" not in html
        assert "<code>logo</code>" in html

    def test_caption_population_matches_the_scale_section(self, mods, tmp_path):
        # Both must band the same marks, or the histogram and the captions
        # describe different corpora.
        marks = [
            ("logo", (50, 50, 120, 100), "spods/logo_a", "clustered"),
            ("signature", (600, 1200, 300, 200), None, "gt"),
        ]
        page = _page(mods, "spods/00622", "spods", marks, path=self._blank(tmp_path))
        html = self._text(mods["report"].section_full_pages([page], 4, 11))
        sides = [m.longest_side() for m in page.marks if m.class_id and m.area() > 0]
        assert f"{max(sides)}px" in html
        assert "300px" not in html

    def test_exactly_one_box_is_red_even_when_a_class_repeats_on_the_page(self, mods, tmp_path):
        # Highlighting by class id reddens every instance of that class; the
        # caption describes one of them.
        page = _page(
            mods,
            "spods/00001",
            "spods",
            [
                ("logo", (100, 100, 120, 90), "spods/logo_a", "clustered"),
                ("logo", (400, 400, 300, 220), "spods/logo_a", "clustered"),
            ],
            path=self._blank(tmp_path),
        )
        biggest = max(page.marks, key=lambda m: m.area())
        im = mods["report"]._page_with_boxes(page, highlight=biggest)
        red = self._colour_bbox(im, mods["report"]._HIGHLIGHT_COLOUR)
        assert red == (400, 400, 700, 620)
        # ...and the other instance is still drawn, in its kind's colour.
        blue = self._colour_bbox(im, mods["report"]._KIND_COLOURS["logo"])
        assert blue is not None and blue[:2] == (100, 100)

    def test_every_figure_carries_a_red_box(self, mods, tmp_path):
        # The prose promises one; a page whose largest mark is unlabelled used
        # to produce a figure with no red pixel in it at all.
        page = _page(
            mods,
            "spods/00622",
            "spods",
            [
                ("logo", (50, 50, 120, 100), "spods/logo_a", "clustered"),
                ("text", (60, 400, 403, 40), None, "gt"),
            ],
            path=self._blank(tmp_path),
        )
        biggest = max((m for m in page.marks if m.class_id), key=lambda m: m.area())
        im = mods["report"]._page_with_boxes(page, highlight=biggest)
        assert self._colour_bbox(im, mods["report"]._HIGHLIGHT_COLOUR) is not None

    def test_kinds_are_drawn_in_distinguishable_colours(self, mods, tmp_path):
        # The single most useful thing this figure can say is that the box on a
        # handwritten signature is a deliberately non-queryable mark rather than
        # a mislabelled logo.  One shade of blue for everything withholds it.
        page = _page(
            mods,
            "spods/00622",
            "spods",
            [
                ("logo", (50, 50, 120, 100), None, "gt"),
                ("stamp", (300, 300, 160, 160), None, "gt"),
                ("signature", (600, 1200, 200, 90), None, "gt"),
                ("text", (60, 400, 403, 40), None, "gt"),
            ],
            path=self._blank(tmp_path),
        )
        im = mods["report"]._page_with_boxes(page)
        drawn = {c for _, c in im.getcolors(1 << 20)}
        seen = [mods["report"]._KIND_COLOURS[k] for k in ("logo", "stamp", "signature", "text")]
        assert all(c in drawn for c in seen)
        assert len(set(seen)) == len(seen)

    def test_legend_names_only_the_kinds_actually_drawn(self, mods, tmp_path):
        # A legend that promises a colour the figure never draws is the same
        # bug as prose that does, one line further down.
        page = _page(
            mods,
            "spods/00622",
            "spods",
            [("logo", (50, 50, 120, 100), "spods/logo_a", "clustered")],
            path=self._blank(tmp_path),
        )
        legend = mods["report"]._legend([page])
        assert "<code>logo</code>" in legend
        assert "signature" not in legend
        assert mods["report"]._rgb(mods["report"]._KIND_COLOURS["logo"]) in legend

    def test_zero_area_marks_are_neither_drawn_nor_advertised(self, mods, tmp_path):
        page = _page(
            mods,
            "spods/00622",
            "spods",
            [
                ("logo", (50, 50, 120, 100), "spods/logo_a", "clustered"),
                ("stamp", (10, 10, 0, 0), None, "gt"),
            ],
            path=self._blank(tmp_path),
        )
        assert mods["report"].kinds_drawn([page]) == ["logo"]
        im = mods["report"]._page_with_boxes(page)
        assert self._colour_bbox(im, mods["report"]._KIND_COLOURS["stamp"]) is None

    def test_every_kind_the_sources_emit_has_a_colour_and_a_gloss(self, mods):
        # Same contract as the provenance glosses: a kind with no swatch falls
        # back to grey and reads as "some other mark", which is the one thing
        # this figure is supposed to stop doing.
        # `LOCALISED_CONTEXT_CATEGORIES` since #3366: this assertion was written
        # against `CONTEXT_CATEGORIES` in #3364, and #3366 renamed the constant
        # in the same week -- both merged green on their own branches and dev
        # was left red, because neither ran against the other.  The name here is
        # deliberately the localised one: #3366 dropped `text` from the marks
        # entirely (it was the page body, a property of the page rather than a
        # thing on it), so a kind that is no longer emitted must not be required
        # to carry a swatch.
        emitted = set(mods["spods"].MARK_CATEGORIES) | set(mods["spods"].LOCALISED_CONTEXT_CATEGORIES)
        assert emitted <= set(mods["report"]._KIND_COLOURS)
        assert emitted <= set(mods["report"]._KIND_MEANING)


# ------------------------------------------------------------- embed cells


class TestEmbedCells:
    def test_a_tier_cell_is_cumulative_over_smaller_tiers(self, mods):
        assert mods["embed"].tiers_up_to("s") == {"s"}
        assert mods["embed"].tiers_up_to("m") == {"s", "m"}
        assert mods["embed"].tiers_up_to("l") == {"s", "m", "l"}

    def test_cell_names_match_the_pile_convention(self, mods):
        assert mods["embed"].cell_name("m", "sift_vlad") == "docmarks_m__sift_vlad.pkl"

    def test_load_medias_carries_boxes_as_regions(self, mods, tmp_path):
        from PIL import Image

        img = tmp_path / "p.png"
        Image.new("RGB", (400, 500), "white").save(img)
        page = _page(mods, "spods/001", "spods", [("logo", (10, 20, 60, 40), "spods/a", "gt")], path=str(img))
        medias = mods["embed"].load_medias([page], {}, "sift_vlad")
        (media,) = medias.values()
        assert media["categories"] == ["spods/a"]
        assert media["regions"] == [
            {"label": "spods/a", "x": 10, "y": 20, "width": 60, "height": 40, "provenance": "gt"}
        ]
        assert media["origin_name"] == "spods/001"

    def test_weak_boxless_marks_become_a_category_but_never_a_region(self, mods, tmp_path):
        from PIL import Image

        img = tmp_path / "p.png"
        Image.new("RGB", (400, 500), "white").save(img)
        page = _page(mods, "ucsf/x#0", "ucsf", [("logo", (0, 0, 0, 0), "ucsf/letterhead_rjr", "weak")], path=str(img))
        (media,) = mods["embed"].load_medias([page], {}, "siglip").values()
        # A zero-area region would be indistinguishable from a real box once it
        # is in the media dict, which is precisely the distinction the corpus
        # exists to keep visible.
        assert media["regions"] == []
        assert media["categories"] == ["ucsf/letterhead_rjr"]
        assert media["docmarks"]["provenances"] == ["weak"]

    def test_media_ids_are_stable_under_input_ordering(self, mods, tmp_path):
        from PIL import Image

        paths = []
        for name in ("a", "b", "c"):
            p = tmp_path / f"{name}.png"
            Image.new("RGB", (50, 50), "white").save(p)
            paths.append(p)
        pages = [_page(mods, f"spods/{n}", "spods", path=str(p)) for n, p in zip("abc", paths)]

        forward = mods["embed"].load_medias(pages, {}, "siglip")
        reverse = mods["embed"].load_medias(list(reversed(pages)), {}, "siglip")
        assert {i: m["origin_name"] for i, m in forward.items()} == {i: m["origin_name"] for i, m in reverse.items()}


class TestKaggleCredentialGate:
    """The gate in ``kaggle_download`` must not reject a credential that works.

    It exists to turn a mid-job 403 into an up-front error message, which is
    worth having -- but that makes a *false* BLOCKED the one failure mode worse
    than no gate at all: the run never starts, and the message sends you to
    look for a token you already have.  That is exactly what happened on the
    GRID in #3343.  Kaggle's "Create New Token" wrote ``~/.kaggle/access_token``
    and ``kagglesdk`` read it happily -- ``kaggle datasets download`` pulled the
    32.8 MB Tobacco800 archive from the same shell -- while the probe reported
    both Kaggle sources unreachable, because the gate looked only for
    ``kaggle.json``.

    So each accepted form is pinned here by name.  A future Kaggle rename is
    fine; silently narrowing the set is not.
    """

    @pytest.fixture
    def _no_env(self, monkeypatch):
        monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
        monkeypatch.delenv("KAGGLE_KEY", raising=False)

    @pytest.mark.parametrize("filename", ["kaggle.json", "access_token"])
    def test_credential_file_is_accepted(self, mods, monkeypatch, tmp_path, _no_env, filename):
        kaggle_dir = tmp_path / ".kaggle"
        kaggle_dir.mkdir()
        (kaggle_dir / filename).write_text("token-contents")
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))

        dest = tmp_path / "dest"
        calls = []
        monkeypatch.setattr(
            mods["common"].subprocess,
            "run",
            lambda cmd, **kw: calls.append(cmd) or _CompletedStub(),
        )
        mods["common"].kaggle_download("owner/name", dest)
        # It got as far as shelling out, which is all this gate governs.
        assert calls and calls[0][:3] == ["kaggle", "datasets", "download"]

    def test_env_pair_is_accepted(self, mods, monkeypatch, tmp_path):
        monkeypatch.setenv("KAGGLE_USERNAME", "someone")
        monkeypatch.setenv("KAGGLE_KEY", "somekey")
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))

        calls = []
        monkeypatch.setattr(
            mods["common"].subprocess,
            "run",
            lambda cmd, **kw: calls.append(cmd) or _CompletedStub(),
        )
        mods["common"].kaggle_download("owner/name", tmp_path / "dest")
        assert calls

    def test_no_credential_still_fails_fast(self, mods, monkeypatch, tmp_path, _no_env):
        (tmp_path / ".kaggle").mkdir()
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))

        def _explode(*a, **kw):  # pragma: no cover - the point is it is unreached
            raise AssertionError("shelled out to the CLI with no credential")

        monkeypatch.setattr(mods["common"].subprocess, "run", _explode)
        with pytest.raises(mods["common"].FetchError) as exc:
            mods["common"].kaggle_download("owner/name", tmp_path / "dest")
        # The message has to name every place a token is accepted, or it sends
        # the reader off to create a second one they do not need.
        assert "kaggle.json" in str(exc.value)
        assert "access_token" in str(exc.value)


class _CompletedStub:
    returncode = 0
    stdout = ""
    stderr = ""


class TestRealMirrorLayouts:
    """Layout facts measured on the real archives, not on the documented ones.

    #3343 pulled both Kaggle sources for the first time and each was parsed
    wrongly by a `--probe`-clean, 107-test-green builder.  Neither failure
    raised: one produced warnings and an empty source, the other produced a
    warning about data that was entirely present.  Fixtures could not have
    caught either, because a fixture is built from the layout the docs
    describe, and both bugs live in the gap between that and the mirror.

    So these pin the *mirror's* conventions by name.
    """

    @pytest.mark.parametrize(
        "gt_name,scan_stem",
        [
            ("stampDS-00001-px", "stampds-00001"),  # pixel-accurate masks
            ("stampDS-00001-gt", "stampds-00001"),  # binary maps
            ("stampDS-00001_px", "stampds-00001"),
            ("stampDS-00001", "stampds-00001"),  # no suffix: unchanged
        ],
    )
    def test_staver_gt_stem_maps_to_its_scan(self, mods, gt_name, scan_stem):
        """StaVer GT filenames carry a suffix the scan does not.

        The scan is ``scans/stampDS-00001.png``; the mask is
        ``ground-truth-pixel/stampDS-00001-px.png``.  Indexing masks by raw stem
        matched nothing on the real archive: 427 "no ground-truth mask"
        warnings and zero StaVer pages in the corpus, a source that looked
        skipped rather than broken.
        """
        assert mods["staver"].gt_stem_key(gt_name) == scan_stem

    def test_tobacco800_zoneless_page_is_not_reported_unmatched(self, mods):
        """A GEDI file with no zones is a negative, not a missing image.

        430 of Tobacco800's 1,290 pages have an empty ``DL_PAGE``.  They are
        kept on purpose as in-domain negatives, so counting them as GT that
        "had no matching image" describes absent data using data that is
        present -- and buries any real mismatch in the noise.
        """
        zoneless = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<GEDI xmlns="http://lamp.cfar.umd.edu/GEDI" version="1.0">'
            '<DL_DOCUMENT src="aao54e00_1.tif" NrOfPages="1" docTag="xml">'
            '<DL_PAGE gedi_type="DL_PAGE" src="aao54e00_1.tif" pageID="1" width="2592" height="3300">'
            "</DL_PAGE></DL_DOCUMENT></GEDI>"
        )
        parsed = mods["tobacco800"].parse_gedi(zoneless)
        # The key exists with an empty list -- that is what made `if marks:`
        # the wrong test, and it is the behaviour the fix depends on.
        assert parsed == {"aao54e00_1": []}


class TestTobacco800LogosAreClustered:
    """Tobacco800 ships identity for signatures and none for logos.

    The distinction is invisible at source level -- "Tobacco800 has GEDI ground
    truth" is true, and taking it as a fact about the whole source is what left
    the logos out of the clustering loop in #3343.  The result was a silent
    one: 432 logo marks with no class_id, so the only source with a published
    logo protocol contributed 1,290 distractor pages and zero eval classes,
    while its 130 signature classes were rejected as unqueryable.  An absent
    class raises nothing, and the survival curve counted the signature classes
    it would never admit, so even the printed numbers looked plausible.
    """

    def _pages(self, mods):
        signature = mods["common"].Mark(
            kind="signature", box=(10, 10, 40, 20), class_id="tobacco800/signature_rjr", provenance="gt"
        )
        logo = mods["common"].Mark(kind="logo", box=(10, 200, 80, 60), class_id=None, provenance="gt")
        return [
            mods["common"].Page(
                page_id=f"tobacco800/p{i}",
                source="tobacco800",
                path=f"/nonexistent/p{i}.tif",
                width=600,
                height=800,
                marks=[signature, logo],
                meta={},
            )
            for i in range(3)
        ]

    def test_collect_refs_takes_the_logos_and_leaves_the_signatures(self, mods):
        refs = mods["cluster"].collect_refs(self._pages(mods), kinds=("logo", "stamp"), source="tobacco800")
        # One per page, and every one a logo: signatures are out by kind, and an
        # already-identified mark is out by class_id.  This is what makes adding
        # the source to the loop safe rather than destructive.
        assert len(refs) == 3
        assert {r.kind for r in refs} == {"logo"}

    def test_builder_clusters_tobacco800(self, mods):
        """The source must be in the clustered set; a comment is not enough.

        This asserted the literal loop line until #3343 moved the list into
        `cfg.CLUSTERED_SOURCES` and it broke on a refactor that changed nothing
        it cared about. Pinning source *text* pins the spelling; pinning the
        value pins the behaviour, and only one of those is the thing at risk.
        """
        assert "tobacco800" in mods["cfg"].CLUSTERED_SOURCES


# -------------------------------------------------------------------- probe


class TestKaggleProbe:
    """``--probe`` must stay a metadata call.

    It used to reach Kaggle by downloading the bundle into ``raw/_probe_*``:
    ~2 GB of transfer, fetched a second time by the real build, never reclaimed
    — under a name and a runbook that both promised "seconds" (issue #3356).
    The cheapness is the whole feature, so it is pinned here.
    """

    @staticmethod
    def _with_creds(monkeypatch):
        monkeypatch.setenv("KAGGLE_USERNAME", "someone")
        monkeypatch.setenv("KAGGLE_KEY", "deadbeef")

    @staticmethod
    def _run(monkeypatch, mods, *, stdout="", stderr="", returncode=0, exc=None):
        """Stub ``subprocess.run`` and record the argv it was handed."""
        import subprocess

        seen = []

        def fake_run(cmd, **kwargs):
            seen.append(cmd)
            if exc is not None:
                raise exc
            if returncode:
                raise subprocess.CalledProcessError(returncode, cmd, output=stdout, stderr=stderr)
            return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr=stderr)

        monkeypatch.setattr(mods["common"].subprocess, "run", fake_run)
        return seen

    def test_lists_files_instead_of_downloading(self, monkeypatch, mods, tmp_path):
        self._with_creds(monkeypatch)
        monkeypatch.chdir(tmp_path)
        seen = self._run(monkeypatch, mods, stdout="name,size,creationDate\ngt.zip,44MB,2020-01-01\n")

        mods["common"].kaggle_probe("owner/name")

        (cmd,) = seen
        assert cmd[:4] == ["kaggle", "datasets", "files", "-d"]
        assert "download" not in cmd
        # Nothing may be staged anywhere: no destination is even passed.
        assert not any(str(tmp_path) in str(part) for part in cmd)
        assert list(tmp_path.iterdir()) == []

    def test_a_missing_token_is_reported_before_the_cli_runs(self, monkeypatch, mods, tmp_path):
        monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
        monkeypatch.delenv("KAGGLE_KEY", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))  # no ~/.kaggle/kaggle.json under here
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        seen = self._run(monkeypatch, mods)

        with pytest.raises(mods["common"].FetchError, match=r"kaggle\.json"):
            mods["common"].kaggle_probe("owner/name")
        assert seen == []

    def test_a_missing_cli_names_the_install(self, monkeypatch, mods):
        self._with_creds(monkeypatch)
        self._run(monkeypatch, mods, exc=FileNotFoundError("kaggle"))

        with pytest.raises(mods["common"].FetchError, match="pip install kaggle"):
            mods["common"].kaggle_probe("owner/name")

    def test_a_nonzero_exit_quotes_stderr(self, monkeypatch, mods):
        self._with_creds(monkeypatch)
        self._run(monkeypatch, mods, returncode=1, stderr="boom")

        with pytest.raises(mods["common"].FetchError, match="boom"):
            mods["common"].kaggle_probe("owner/name")

    @pytest.mark.parametrize(
        "stdout",
        [
            "403 - Forbidden\n",  # the CLI swallows API errors and still exits 0
            "404 - Not Found\n",
            "",
            "name,size,creationDate\n",  # a header with no rows is not a dataset
        ],
    )
    def test_exit_zero_is_not_taken_as_success(self, monkeypatch, mods, stdout):
        self._with_creds(monkeypatch)
        self._run(monkeypatch, mods, stdout=stdout)

        with pytest.raises(mods["common"].FetchError, match="owner/name"):
            mods["common"].kaggle_probe("owner/name")


class TestReclaimProbeDirs:
    def test_removes_stale_probe_dirs_and_reports_their_size(self, mods, tmp_path):
        stale = tmp_path / "_probe_staver" / "nested"
        stale.mkdir(parents=True)
        (stale / "big.bin").write_bytes(b"x" * 2048)
        keep = tmp_path / "staver"
        keep.mkdir()
        (keep / "real.bin").write_bytes(b"y" * 16)

        dirs, freed = mods["build"]._reclaim_probe_dirs(tmp_path)

        assert (dirs, freed) == (1, 2048)
        assert not (tmp_path / "_probe_staver").exists()
        assert (keep / "real.bin").exists()

    def test_a_missing_raw_root_is_not_an_error(self, mods, tmp_path):
        assert mods["build"]._reclaim_probe_dirs(tmp_path / "never-created") == (0, 0)


class TestKaggleProbeParsesRealCliOutput:
    """The probe must read what the CLI actually prints, not an idealised CSV.

    Kaggle CLI 2.2.4 emits a pagination preamble before the listing and uses
    CRLF inside it::

        Next Page Token = CfDJ8ImuQD4OY2pEnVW2WQ-kgndQdHqu9wY-...
        name,size,creationDate\r
        ground-truth-maps/.../stampDS-00001-gt.png,9151,2018-04-11 ...\r

    Reading row 0 as the header made every reachable dataset report unreachable.
    The failure is the expensive direction for this function: it exists so that
    a missing token is caught before a queue slot is burned, so a false BLOCKED
    costs precisely what the probe was written to save, and it does it while
    looking like a correctly-working guard.

    Captured verbatim from the GRID rather than imagined, which is the only
    reason the preamble is in here at all.
    """

    _REAL = (
        "Next Page Token = CfDJ8ImuQD4OY2pEnVW2WQ-kgnf0jam8ELDf3ktjyVw0Ztjp\n"
        "name,size,creationDate\r\n"
        "Tobacc800_Groundtruth_v2.0/Overview.txt,1634,2023-01-09 09:10:46.734000\r\n"
        "Tobacc800_Groundtruth_v2.0/aah97e00-page02_1.xml,502,2023-01-09 09:10:46.711000\r\n"
    )

    def _run(self, mods, monkeypatch, stdout):
        monkeypatch.setenv("KAGGLE_USERNAME", "someone")
        monkeypatch.setenv("KAGGLE_KEY", "somekey")

        class _Proc:
            returncode = 0
            stderr = ""

            def __init__(self, out):
                self.stdout = out

        monkeypatch.setattr(mods["common"].subprocess, "run", lambda *a, **k: _Proc(stdout))
        mods["common"].kaggle_probe("owner/name")

    def test_preamble_and_crlf_are_tolerated(self, mods, monkeypatch):
        self._run(mods, monkeypatch, self._REAL)  # must not raise

    def test_a_plain_csv_still_passes(self, mods, monkeypatch):
        self._run(mods, monkeypatch, "name,size\nfoo.png,10\n")

    def test_a_listing_with_no_rows_still_fails(self, mods, monkeypatch):
        with pytest.raises(mods["common"].FetchError):
            self._run(mods, monkeypatch, "Next Page Token = abc\nname,size,creationDate\r\n")

    def test_an_error_message_still_fails(self, mods, monkeypatch):
        with pytest.raises(mods["common"].FetchError):
            self._run(mods, monkeypatch, "403 - Forbidden - Permission denied\n")


class TestUcsfIndustryIsRecorded:
    """The Tobacco800 contamination rule needs an industry on the page.

    `industry` is INDEXED BUT NOT STORED in UCSF's Solr: `industry:Tobacco`
    filters correctly, and `fl=industry` returns nothing at all.  So
    `first_value(doc, "industry")` was always None, every page in the
    2026-08-31 build recorded `industry: null`, and
    `eligible_distractor("tobacco800", "ucsf", None)` could never fire.

    That is the single contamination rule that costs something.  Tobacco800 and
    UCSF's Tobacco industry are both IIT-CDIP, so an American Tobacco letterhead
    in a Tobacco800 class genuinely appears on UCSF Tobacco pages -- unlabelled
    positives, where retrieving one is CORRECT and the metric records a false
    positive.  The corpus looked fine: 117,028 pages, no warning, a null in a
    metadata field nobody reads.
    """

    def test_the_queried_industry_is_stamped_on_the_page(self, mods):
        page = mods["ucsf"].doc_to_page(
            {"id": "ffbb0002", "title": "x"}, "/nonexistent/p.png", 1240, 1680, industry="Tobacco"
        )
        assert page.meta["industry"] == "Tobacco"

    def test_a_tobacco_page_is_not_eligible_for_a_tobacco800_class(self, mods):
        assert mods["cfg"].eligible_distractor("tobacco800", "ucsf", "Tobacco") is False
        # ...and the non-tobacco industries still are: they are different
        # companies, not the same archive under another name.
        assert mods["cfg"].eligible_distractor("tobacco800", "ucsf", "Opioids") is True

    def test_a_null_industry_is_what_used_to_slip_through(self, mods):
        # Pinned as the regression itself: this returning True is precisely the
        # bug, and it is why the industry must be stamped at pull time.
        assert mods["cfg"].eligible_distractor("tobacco800", "ucsf", None) is True


class TestDistractorPullPlan:
    """Fill the budget, and spend Tobacco last.

    The six industries are wildly unequal -- measured live: Tobacco 9.4M and
    Opioids 4.07M against Fossil Fuel 311, Drug 1,064, Chemical 3,657 -- so
    `budget // 6` asks three of them for ~30k pages that do not exist.  At 200k
    the even split tops out at 105,031, which is why the 2026-08-31 build
    stopped at 119,806 pages looking like a job that had run to completion.
    """

    def test_the_budget_is_actually_filled(self, mods):
        plan = mods["build"]._plan_distractor_pull(200_000)
        assert sum(want for _, want in plan) == 200_000

    def test_the_production_budget_draws_no_tobacco_at_all(self, mods):
        """The 200k case, and the reason the ordering exists.

        Not a style preference: UCSF Tobacco is the same archive as Tobacco800,
        so every Tobacco page in the haystack is a page Tobacco800's classes
        cannot safely be scored against -- and worse, a place an unlabelled
        positive can hide, where a correct retrieval is recorded as a false
        positive.  Opioids alone has 4.07M single-page documents, so at 200k the
        budget is filled without touching Tobacco.
        """
        plan = dict(mods["build"]._plan_distractor_pull(200_000))
        assert plan.get("Tobacco", 0) == 0
        assert sum(plan.values()) == 200_000

    def test_tobacco_is_drawn_only_once_everything_else_is_exhausted(self, mods):
        cap = mods["build"].UCSF_INDUSTRY_CAPACITY
        others = sum(v for k, v in cap.items() if k != "Tobacco")
        plan = mods["build"]._plan_distractor_pull(others + 50_000)
        drawn = [industry for industry, want in plan if want > 0]
        assert drawn[-1] == "Tobacco"
        assert dict(plan)["Tobacco"] == 50_000
        # ...and only after every other industry was taken to capacity.
        for industry, want in plan:
            if industry != "Tobacco":
                assert want == cap[industry]

    def test_a_small_budget_never_reaches_tobacco(self, mods):
        plan = dict(mods["build"]._plan_distractor_pull(5_000))
        assert plan.get("Tobacco", 0) == 0

    def test_no_industry_is_asked_for_more_than_it_has(self, mods):
        plan = mods["build"]._plan_distractor_pull(200_000)
        cap = mods["build"].UCSF_INDUSTRY_CAPACITY
        assert all(want <= cap[industry] for industry, want in plan)


class TestFetchThrottle:
    """Concurrency is only defensible because it gives ground.

    UCSF is a shared public archive and README/GRID-RUNBOOK both warn against
    widening the pull.  The measurement that justifies 3 concurrent fetches --
    ~120k requests at ~3/s with zero 429/503/509, and 4,003 failures that were
    all per-document 403s -- shows we are *below* their limit, not where it is.
    So the pool has to detect the limit rather than assume it, and these pin the
    behaviour that makes that true.
    """

    def test_a_rate_limit_costs_a_worker_and_buys_delay(self, mods):
        t = mods["ucsf"]._Throttle(3)
        assert (t.workers, t.penalties) == (3, 0)
        t.penalise()
        assert t.workers == 2
        assert t.penalties == 1
        assert t._delay >= 1.0

    def test_sustained_pushback_converges_on_serial(self, mods):
        t = mods["ucsf"]._Throttle(3)
        for _ in range(10):
            t.penalise()
        # It gives up workers until it is doing exactly what it did before the
        # change -- one request at a time -- rather than hammering through.
        assert t.workers == 1
        assert t._delay <= 60.0

    def test_success_recovers_slowly(self, mods):
        t = mods["ucsf"]._Throttle(2)
        t.penalise()
        hot = t._delay
        for _ in range(5):
            t.reward()
        assert t._delay < hot
        # One 429 must not cost the rest of a 200k run, but recovery is gradual
        # rather than instant or the pool would oscillate against the limit.
        assert t._delay > 0

    def test_rate_limited_is_distinct_from_a_dead_document(self, mods):
        # The whole point of the separate class: a 403 is permanent and the
        # document must be skipped; a 429 means the document is still there and
        # we asked too fast.  Retrying a 403 wastes the run; skipping a 429
        # silently shrinks the corpus.
        assert issubclass(mods["common"].RateLimited, mods["common"].FetchError)
        assert not issubclass(mods["common"].FetchError, mods["common"].RateLimited)

    def test_default_pool_is_three(self, mods, monkeypatch):
        monkeypatch.delenv("VTS_DOCMARKS_FETCH_WORKERS", raising=False)
        assert mods["ucsf"]._fetch_workers() == 3
        monkeypatch.setenv("VTS_DOCMARKS_FETCH_WORKERS", "1")
        assert mods["ucsf"]._fetch_workers() == 1


class TestResumeSkipsRendering:
    """A resumed pull must not re-render pages it already has.

    GRID-RUNBOOK promises "Resume is free.  Downloads are atomic, rendered pages
    are skipped when present".  Two thirds of that was true.  The skip guarded
    the *save*::

        if not image_path.exists():
            image.save(image_path)

    while `render_pdf_pages` above it ran unconditionally -- so a resumed job
    re-rendered every page it already had, at 0.188 s each, and discarded the
    result.  At 200k pages that is ~10 h per restart, in a builder whose whole
    design premise is that restarting is cheap.  It mattered four times in two
    days: a promise that resume is cheap is what makes a long unattended run
    *correctable* rather than merely endurable.

    So the fast path is pinned as behaviour, not left as an optimisation
    somebody could reasonably tidy away.  The rendering call is made to explode;
    if the guard regresses, this test does too.
    """

    def _fake_png(self, path):
        from PIL import Image

        Image.new("RGB", (120, 160), "white").save(path)

    def test_all_pages_present_means_no_render_call(self, mods, tmp_path, monkeypatch):
        out = tmp_path / "images"
        out.mkdir()
        self._fake_png(out / "abc12345_0.png")

        import vtscore.datasets.pdf as pdfmod

        def _explode(*a, **kw):  # pragma: no cover - the point is it is unreached
            raise AssertionError("re-rendered a page that was already on disk")

        monkeypatch.setattr(pdfmod, "render_pdf_pages", _explode)

        got = mods["ucsf"]._render_to_disk(str(tmp_path / "nonexistent.pdf"), str(out), "abc12345", 150, 1)
        # Dimensions come off the PNG that is already there.
        assert got == [(0, str(out / "abc12345_0.png"), 120, 160)]

    def test_a_missing_page_still_renders(self, mods, tmp_path, monkeypatch):
        out = tmp_path / "images"
        out.mkdir()
        # Nothing on disk, so the fast path must NOT engage -- otherwise a fresh
        # pull would quietly produce no images at all.
        import vtscore.datasets.pdf as pdfmod

        called = []
        monkeypatch.setattr(pdfmod, "render_pdf_pages", lambda *a, **k: called.append(a) or [])
        mods["ucsf"]._render_to_disk(str(tmp_path / "x.pdf"), str(out), "def67890", 150, 1)
        assert called, "a page with no PNG on disk must be rendered"


class TestPerSourceClusteringKnobs:
    """One threshold and one instance bar for four sources served only one.

    `CLUSTER_THRESHOLD`'s own docstring ends "this number is a property of the
    data, and it does not travel".  It does not travel between SOURCES either,
    which a single global value quietly assumed.  Swept per source on the 200k
    build: SPODS percolates just above 0.10, StaVer is already 5.8% merged at
    0.02 and 22% by 0.10, and Tobacco800's usable classes PEAK at 0.18 with only
    14.7% merged -- three times what 0.10 gave it.

    The same applies to the instance bar.  A `>=10` that is right for SPODS's
    174 candidate classes empties StaVer, which has exactly one class that deep.
    """

    def test_each_source_gets_its_own_swept_threshold(self, mods):
        cfg = mods["cfg"]
        assert cfg.cluster_threshold_for("spods") == 0.10
        assert cfg.cluster_threshold_for("staver") == 0.04
        assert cfg.cluster_threshold_for("tobacco800") == 0.18

    def test_tobacco800_is_looser_than_spods_and_staver_is_tighter(self, mods):
        cfg = mods["cfg"]
        # The ordering is the finding; the exact numbers will move when the
        # descriptor or the sources change, and this should move with them.
        assert (
            cfg.cluster_threshold_for("staver")
            < cfg.cluster_threshold_for("spods")
            < cfg.cluster_threshold_for("tobacco800")
        )

    def test_an_unknown_source_falls_back_to_the_global(self, mods):
        assert mods["cfg"].cluster_threshold_for("synth") == mods["cfg"].CLUSTER_THRESHOLD

    def test_an_env_override_still_wins_for_every_source(self, mods, monkeypatch):
        monkeypatch.setenv("VTS_DOCMARKS_CLUSTER_THRESHOLD", "0.5")
        for src in ("spods", "staver", "tobacco800", "ucsf"):
            assert mods["cfg"].cluster_threshold_for(src) == mods["cfg"].CLUSTER_THRESHOLD

    def test_the_sparse_sources_get_a_lower_instance_bar(self, mods):
        cfg = mods["cfg"]
        assert cfg.min_instances_for("spods") == cfg.MIN_INSTANCES
        assert cfg.min_instances_for("staver") < cfg.MIN_INSTANCES
        assert cfg.min_instances_for("tobacco800") < cfg.MIN_INSTANCES


class TestAnchorPagesAreInEveryTier:
    """The known negatives must not be evicted by a distractor budget.

    README: a page from a source exhaustively checked for a class is "the
    hardest possible negative" -- same scanner, same paper, same era.  The
    2026-09-01 build dropped 129 of them over the tier budget to make room for
    UCSF pages, which spends the hardest negatives to buy the easiest.  There
    are only ~2,650 anchor pages; they fit in every tier including `s`.
    """

    def test_an_anchor_page_survives_a_budget_far_below_the_corpus(self, mods):
        pages = [_page(mods, f"spods/p{i}", "spods", path=f"/nonexistent/s{i}.png") for i in range(5)] + [
            _page(mods, f"ucsf/p{i}", "ucsf", path=f"/nonexistent/u{i}.png") for i in range(200)
        ]

        tiers, _cut = mods["build"].assign_tiers(pages, {}, tiers={"s": 10, "l": 100}, tier_order=("s", "l"), salt="t")
        anchors = [pid for pid in tiers if pid.startswith("spods/")]
        assert len(anchors) == 5, "an anchor page was dropped over budget"
        assert all(tiers[pid] == "s" for pid in anchors), "anchors must be in the smallest tier"

    def test_ucsf_distractors_are_still_budgeted(self, mods):
        pages = [_page(mods, f"ucsf/p{i}", "ucsf", path=f"/nonexistent/u{i}.png") for i in range(200)]
        tiers, _cut = mods["build"].assign_tiers(pages, {}, tiers={"s": 10, "l": 100}, tier_order=("s", "l"), salt="t")
        # The budget still binds on the population it is meant to bind on.
        assert sum(1 for t in tiers.values() if t == "s") == 10


class TestUcsfBandsAreNotClustered:
    """UCSF letterhead bands propose noise, not candidates.

    The band is a fixed-geometry crop -- the top 22% of every page -- so a
    perceptual hash of it describes page layout rather than the logo inside it,
    and two unrelated companies' letterheads at the same position hash alike.

    Swept on 3,000 marks there is no flat region at any threshold: largest
    component 12.4% at 0.02 (the lowest on the grid), 36% at 0.04, 81% at 0.10,
    99% at 0.22, with 85% of marks singletons throughout.  SPODS sits at 1.5%
    across the same range.  It is percolated from the start, so there is no
    threshold to pick -- which is why this is a source list and not a number.

    At 0.10 it produced one 12,706-instance "class"; being admitted-class pages,
    those pinned 13,874 pages into tier `s` against a 5,000 budget.
    """

    def test_ucsf_is_not_clustered(self, mods):
        assert "ucsf" not in mods["cfg"].CLUSTERED_SOURCES

    def test_the_anchor_sources_still_are(self, mods):
        assert set(mods["cfg"].CLUSTERED_SOURCES) == {"spods", "staver", "tobacco800"}

    def test_every_clustered_source_has_a_swept_threshold(self, mods):
        # A source we cluster without a swept number is one running on an
        # inherited default, which is how UCSF got a 12,706-instance class.
        cfg = mods["cfg"]
        for src in cfg.CLUSTERED_SOURCES:
            assert src in cfg.CLUSTER_THRESHOLD_BY_SOURCE

    def test_ucsf_pages_are_still_in_the_corpus(self, mods):
        # Dropping the CLASSES must not drop the PAGES: UCSF is 197k distractors
        # and that was always 92% of what it was for.
        assert "ucsf" not in mods["cfg"].ANCHOR_SOURCES
        assert mods["cfg"].eligible_distractor("spods", "ucsf", "Opioids") is True


class TestSiglipAuditVectors:
    """The audit's second opinion, over cached vectors rather than a card."""

    @staticmethod
    def _cache(tmp_path, items, vecs, embedder="siglip2_l"):
        import numpy as np

        out = tmp_path / "audit" / "siglip"
        out.mkdir(parents=True)
        np.savez_compressed(tmp_path / "audit" / "siglip" / "vectors.npz", vecs=np.asarray(vecs, dtype=np.float32))
        (out / "items.json").write_text(
            json.dumps({"embedder": embedder, "max_per_class": 24, "items": items}), encoding="utf-8"
        )
        return tmp_path

    def test_a_missing_cache_is_an_error_not_an_empty_answer(self, mods, tmp_path):
        with pytest.raises(SystemExit, match="--embed"):
            mods["siglip"].load_cache(tmp_path)

    def test_a_centroid_is_the_normalised_mean_of_its_instances(self, mods, tmp_path):
        items = [
            {"class_id": "a", "page_id": "p1", "kind": "instance"},
            {"class_id": "a", "page_id": "p2", "kind": "instance"},
            {"class_id": "b", "page_id": "p3", "kind": "instance"},
            # A query crop must not move the centroid it is going to be judged
            # against, or the check is comparing the crop with itself.
            {"class_id": "a", "page_id": "p9", "kind": "query"},
        ]
        vecs = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
        order, centroids = mods["siglip"].class_centroids(items, vecs)
        assert order == ["a", "b"]
        assert centroids[0] == pytest.approx([2**-0.5, 2**-0.5], abs=1e-6)
        assert np.linalg.norm(centroids[1]) == pytest.approx(1.0, abs=1e-6)

    def test_split_uses_average_linkage_so_one_crop_cannot_bridge_two_marks(self, mods):
        # Two tight groups plus a crop halfway between them.  Single linkage
        # chains all three together through the bridge -- which is the exact
        # defect this pass exists to detect, so it must not be the linkage.
        a, b = np.array([1.0, 0.0]), np.array([0.0, 1.0])
        bridge = (a + b) / np.linalg.norm(a + b)
        vecs = np.vstack([a, a, a, b, b, b, bridge]).astype(np.float32)
        labels = mods["siglip"].split_class(vecs, 0.4)
        assert len(set(labels.tolist())) >= 2
        assert labels[0] == labels[1] == labels[2]
        assert labels[3] == labels[4] == labels[5]
        assert labels[0] != labels[3]

    def test_one_tight_class_stays_one_group(self, mods):
        vecs = np.tile(np.array([1.0, 0.0], dtype=np.float32), (5, 1))
        assert len(set(mods["siglip"].split_class(vecs, 0.2).tolist())) == 1

    def test_the_query_check_reports_the_rank_of_the_class_own_centroid(self, mods):
        # The eval searches with the query crop, so the question is what that
        # crop retrieves -- not how far it is, which is the screen #3599
        # records failing.
        items = [
            {"class_id": "a", "page_id": "p1", "kind": "instance"},
            {"class_id": "b", "page_id": "p2", "kind": "instance"},
            {"class_id": "a", "page_id": "p9", "kind": "query"},
        ]
        vecs = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
        order, centroids = mods["siglip"].class_centroids(items, vecs)
        rows = mods["siglip"].query_check(items, vecs, order, centroids)
        assert len(rows) == 1
        assert rows[0]["class_id"] == "a"
        assert rows[0]["rank_of_own_class"] == 1
        assert rows[0]["nearest_class"] == "b"

    def test_pairs_are_ranked_by_centroid_distance_nearest_first(self, mods):
        order = ["a", "b", "c"]
        centroids = np.array([[1.0, 0.0], [0.99, 0.141], [0.0, 1.0]], dtype=np.float32)
        rows = mods["siglip"].pair_report(order, centroids, 3)
        assert [(r["left"], r["right"]) for r in rows][0] == ("a", "b")
        assert rows[0]["distance"] < rows[-1]["distance"]

    def test_the_split_sweep_records_every_threshold_not_a_verdict(self, mods):
        items = [{"class_id": "a", "page_id": f"p{i}", "kind": "instance"} for i in range(4)]
        # Two pairs 0.5 apart: a tight threshold separates them, a loose one
        # does not, and the file records both rather than choosing.
        far = [0.5, 3**0.5 / 2]
        vecs = np.array([[1.0, 0.0], [1.0, 0.0], far, far], dtype=np.float32)
        rows = mods["siglip"].split_report(items, vecs, (0.1, 0.9))
        assert rows[0]["sweep"]["0.10"] == [2, 2]
        assert rows[0]["sweep"]["0.90"] == [4]


class TestClassSamplingIsSpread:
    """Which instances of a class the audit actually looks at (#3610).

    Every pass that samples a class capped the sample by taking the head of the
    page-id list, so a split proposal for a class larger than the cap was a
    statement about its first ``max_per_class`` pages.  Page ids sort by source
    and number, so that is the scanner's order: the two classes #3610 was filed
    over are 27 and 30 instances against a cap of 24, and five of their marks
    live only in the tail.
    """

    def test_a_short_sequence_is_taken_whole(self, mods):
        assert mods["common"].spread(list(range(5)), 24) == list(range(5))

    def test_the_sample_always_reaches_the_tail(self, mods):
        # The regression that matters.  A `[::step]` stride computes
        # `step = 27 // 24 == 1` and hands back the first 24 -- the head sample
        # it was reached for to avoid.
        picked = mods["common"].spread(list(range(27)), 24)
        assert len(picked) == 24
        assert picked[0] == 0
        assert picked[-1] == 26

    def test_the_sample_is_spread_and_ordered_and_distinct(self, mods):
        picked = mods["common"].spread(list(range(30)), 24)
        assert len(set(picked)) == 24
        assert picked == sorted(picked)
        assert max(b - a for a, b in zip(picked, picked[1:])) <= 2

    def test_degenerate_limits(self, mods):
        assert mods["common"].spread([], 5) == []
        assert mods["common"].spread([1, 2, 3], 0) == []
        assert mods["common"].spread([1, 2, 3], 1) == [1]

    def test_collect_items_spreads_over_the_class_not_its_head(self, mods):
        pages = [
            _page(
                mods,
                f"spods/p{i:03d}",
                "spods",
                marks=(("stamp", (10, 10, 20, 20), "spods/c", ("clustered",)),),
            )
            for i in range(27)
        ]
        classes = {"spods/c": {"page_ids": [p.page_id for p in pages], "n_instances": 27}}
        items = mods["siglip"].collect_items(pages, classes, max_per_class=24)
        sampled = [it["page_id"] for it in items if it["kind"] == "instance"]
        assert len(sampled) == 24
        assert "spods/p026" in sampled

    def test_the_cluster_sheet_renders_exactly_what_the_proposal_covers(self, mods, tmp_path):
        # The sheet exists to adjudicate the proposal, and the two samples are
        # drawn by different code paths -- an untagged cell would invite a
        # verdict about a crop the clusterer never saw.
        from PIL import Image

        img = tmp_path / "p.png"
        Image.new("RGB", (200, 200), "white").save(img)
        pages = [
            _page(
                mods,
                f"spods/p{i:03d}",
                "spods",
                marks=(("stamp", (10, 10, 20, 20), "spods/c", ("clustered",)),),
                path=str(img),
            )
            for i in range(6)
        ]
        classes = {
            "spods/c": {
                "page_ids": [p.page_id for p in pages],
                "n_instances": 6,
                "provenance": ["clustered"],
            }
        }
        proposals = {"spods/c": {"spods/p000": 1, "spods/p005": 2}}
        verdicts = mods["slate"].task_cluster(pages, classes, tmp_path, proposals=proposals)
        assert verdicts[0]["proposed_groups"] == [1, 1]


class TestMixedClassScreen:
    """The screen for a mixed *class*, beside the screens for an odd *crop*.

    #3610: `staver/stamp_stampds-00156_0` holds five marks, and the query-crop
    rank scores it in the healthiest tier of all 59 classes -- correctly, since
    its query crop is a good instance of the 16-strong mark it was drawn from.
    A rank of a crop cannot see the other four marks; nothing did.
    """

    @staticmethod
    def _mixed(n_close=3, n_far=3):
        # Two tight groups a right angle apart: max within-class distance 1.0,
        # wider than any threshold in the sweep.
        near, far = [1.0, 0.0], [0.0, 1.0]
        items = [{"class_id": "a", "page_id": f"p{i}", "kind": "instance"} for i in range(n_close + n_far)]
        vecs = np.array([near] * n_close + [far] * n_far, dtype=np.float32)
        return items, vecs

    def test_a_class_wider_than_the_loosest_sweep_threshold_is_flagged(self, mods):
        items, vecs = self._mixed()
        (row,) = mods["siglip"].split_report(items, vecs, (0.1, 0.4))
        assert row["max_within"] == pytest.approx(1.0, abs=1e-3)
        assert row["mixed"] is True

    def test_a_tight_class_is_not(self, mods):
        items = [{"class_id": "a", "page_id": f"p{i}", "kind": "instance"} for i in range(3)]
        vecs = np.tile(np.array([1.0, 0.0], dtype=np.float32), (3, 1))
        (row,) = mods["siglip"].split_report(items, vecs, (0.1, 0.4))
        assert row["mixed"] is False

    def test_the_flag_defaults_to_the_loosest_threshold_in_the_sweep(self, mods):
        # Tied to the sweep on purpose: an independent number is one that can
        # drift away from the sweep, which is how CLUSTER_THRESHOLD's 0.16
        # outlived its decomposition (#3366).
        assert mods["cfg"].AUDIT_MIXED_MAX_WITHIN == max(mods["cfg"].AUDIT_SPLIT_SWEEP)

    def test_the_flag_records_the_threshold_it_was_taken_at(self, mods):
        items, vecs = self._mixed()
        (row,) = mods["siglip"].split_report(items, vecs, (0.4,), mixed_at=1.5)
        assert row["mixed"] is False
        assert row["mixed_at"] == 1.5

    def test_a_query_crop_can_rank_first_and_still_reach_only_part_of_its_class(self, mods):
        # The #3610 case in miniature: the crop is a good instance of the
        # majority mark, so the centroid rank is 0 and says nothing about the
        # instances that are a different mark.
        items = [
            {"class_id": "a", "page_id": "p0", "kind": "instance"},
            {"class_id": "a", "page_id": "p1", "kind": "instance"},
            {"class_id": "a", "page_id": "p2", "kind": "instance"},
            {"class_id": "b", "page_id": "p3", "kind": "instance"},
            {"class_id": "a", "page_id": "q", "kind": "query"},
        ]
        # `other` sits half-way to the query, so the cut-off the reach is
        # measured against falls between the two marks inside class "a".
        near, far, other = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.0, 3**0.5 / 2]
        vecs = np.array([near, near, far, other, near], dtype=np.float32)
        order, centroids = mods["siglip"].class_centroids(items, vecs)
        (row,) = mods["siglip"].query_check(items, vecs, order, centroids)
        assert row["rank_of_own_class"] == 0
        assert row["own_instances"] == 3
        assert row["own_instances_reached"] == 2
        assert row["nearest_other_class"] == "b"

    def test_a_representative_query_crop_reaches_its_whole_class(self, mods):
        items = [
            {"class_id": "a", "page_id": "p0", "kind": "instance"},
            {"class_id": "a", "page_id": "p1", "kind": "instance"},
            {"class_id": "b", "page_id": "p2", "kind": "instance"},
            {"class_id": "a", "page_id": "q", "kind": "query"},
        ]
        vecs = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        order, centroids = mods["siglip"].class_centroids(items, vecs)
        (row,) = mods["siglip"].query_check(items, vecs, order, centroids)
        assert row["own_instances_reached"] == row["own_instances"] == 2

    def test_the_query_crop_is_not_counted_as_one_of_its_own_instances(self, mods):
        items = [
            {"class_id": "a", "page_id": "p0", "kind": "instance"},
            {"class_id": "b", "page_id": "p1", "kind": "instance"},
            {"class_id": "a", "page_id": "q", "kind": "query"},
        ]
        vecs = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        order, centroids = mods["siglip"].class_centroids(items, vecs)
        (row,) = mods["siglip"].query_check(items, vecs, order, centroids)
        assert row["own_instances"] == 1


class TestSlateDescriptorChoice:
    """Which descriptor the slate is ordered by, and what it refuses."""

    @staticmethod
    def _items(class_ids):
        return [{"class_id": c, "page_id": f"p{i}", "kind": "instance"} for i, c in enumerate(class_ids)]

    def test_phash_needs_no_cache_and_is_the_default(self, mods, tmp_path):
        from PIL import Image

        images = [Image.new("RGB", (40, 40), c) for c in ("white", "black")]
        dist = mods["slate"].class_distances(["a", "b"], images, "phash", tmp_path)
        assert dist.shape == (2, 2)
        assert dist[0, 0] == 0.0
        assert mods["cfg"].SLATE_DESCRIPTOR == "phash"

    def test_a_semantic_descriptor_orders_by_centroid_not_by_exemplar(self, mods, tmp_path):
        from PIL import Image

        TestSiglipAuditVectors._cache(
            tmp_path,
            self._items(["a", "a", "b"]),
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        )
        # Identical exemplars: a phash ordering would call these distance 0.
        # The cache says the classes are orthogonal, and the cache is what a
        # semantic descriptor is being asked for.
        images = [Image.new("RGB", (40, 40), "white") for _ in range(2)]
        dist = mods["slate"].class_distances(["a", "b"], images, "siglip2_l", tmp_path)
        assert dist[0, 1] == pytest.approx(1.0, abs=1e-5)

    def test_a_cache_from_another_embedder_is_refused_not_used(self, mods, tmp_path):
        from PIL import Image

        TestSiglipAuditVectors._cache(tmp_path, self._items(["a", "b"]), [[1.0, 0.0], [0.0, 1.0]], embedder="siglip")
        images = [Image.new("RGB", (40, 40), "white") for _ in range(2)]
        with pytest.raises(SystemExit, match="siglip2_l"):
            mods["slate"].class_distances(["a", "b"], images, "siglip2_l", tmp_path)

    def test_a_class_missing_from_the_cache_is_refused_not_dropped(self, mods, tmp_path):
        from PIL import Image

        TestSiglipAuditVectors._cache(tmp_path, self._items(["a"]), [[1.0, 0.0]])
        images = [Image.new("RGB", (40, 40), "white") for _ in range(2)]
        with pytest.raises(SystemExit, match="no vectors"):
            mods["slate"].class_distances(["a", "b"], images, "siglip2_l", tmp_path)

    def test_phash_proposes_no_subgroups_about_its_own_output(self, mods, tmp_path):
        assert mods["slate"].subgroups(tmp_path, "phash", 0.2) == {}

    def test_subgroups_are_numbered_largest_first(self, mods, tmp_path):
        items = [
            {"class_id": "a", "page_id": "p0", "kind": "instance"},
            {"class_id": "a", "page_id": "p1", "kind": "instance"},
            {"class_id": "a", "page_id": "p2", "kind": "instance"},
            {"class_id": "a", "page_id": "p3", "kind": "instance"},
        ]
        TestSiglipAuditVectors._cache(tmp_path, items, [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        groups = mods["slate"].subgroups(tmp_path, "siglip2_l", 0.5)["a"]
        assert groups["p0"] == groups["p1"] == groups["p2"] == 1
        assert groups["p3"] == 2
