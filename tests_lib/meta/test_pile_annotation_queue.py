"""The annotation queue's selection rule, exercised without the pile (#3720).

`annotation_queue.py` decides which images the exhaustive pass owes a judgement
on. Everything expensive about that decision is in two places, and neither is
visible in the output it produces:

* **which source answered "did COCO score this image?"** -- the cell's own stamp
  or the COCO pairing. The pairing is a superset of what the build anchors, so
  reaching for it when the stamp is present would drop rows that need
  annotating, and a dropped row looks exactly like an image nobody owed;
* **an absent stamp against a false one.** They are the same value under
  `.get()` and opposite facts: no answer versus COCO answered and said no. The
  loader lost the deep sibling's cross-class rule to this shape (#3667), so it
  gets a test here rather than a comment.

The rest is arithmetic that has to hold for the counts to mean anything: one row
per image, a class counted once per image however many bands it spans.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_PILE_DIR = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "pile"


@pytest.fixture(scope="module")
def aq():
    """``annotation_queue``, which defers ``setup_env`` into ``main``."""
    if str(_PILE_DIR) not in sys.path:
        sys.path.insert(0, str(_PILE_DIR))
    import annotation_queue

    return annotation_queue


def _positive(cells: list[str], *, coco_scored: bool | None = False, iid: int = 1) -> dict:
    """A designated positive as the loader writes one.

    ``coco_scored=None`` spells a cell built before the stamp existed -- the key
    absent, not present and false.
    """
    media = {
        "id": iid,
        "categories": list(cells),
        "filename": f"{iid}.jpg",
        "origin_name": f"/vg/VG_100K/{iid}.jpg",
    }
    if coco_scored is not None:
        media["coco_scored"] = coco_scored
    return media


def _negative(*, coco_scored: bool | None = True, iid: int = 900) -> dict:
    """A shared negative: no categories, so the pass owes it nothing."""
    media = {"id": iid, "categories": [], "filename": f"{iid}.jpg", "origin_name": f"/vg/VG_100K/{iid}.jpg"}
    if coco_scored is not None:
        media["coco_scored"] = coco_scored
    return media


class TestClassOf:
    def test_a_banded_cell_yields_its_class(self, aq):
        assert aq.class_of("stop sign@large") == "stop sign"

    def test_a_bare_class_passes_through(self, aq):
        """``vg_scale_deep`` keys cells on the class alone, and reads the same."""
        assert aq.class_of("stop sign") == "stop sign"


class TestCocoAnswer:
    def test_the_stamp_is_read_when_present(self, aq):
        assert aq.coco_answer(1, _positive(["bus@small"], coco_scored=True), None) == (True, "stamp")

    def test_a_false_stamp_is_an_answer_not_a_gap(self, aq):
        """The regression this exists to stop.

        The image is stamped ``False`` -- COCO was asked and does not hold it --
        *and* it is in the COCO pairing, because the pairing is a superset of
        what the build anchors (it skips the aspect-drift filter). Re-asking the
        pairing would call it anchored and silently drop a row the pass owes.
        """
        answer = aq.coco_answer(1, _positive(["bus@small"], coco_scored=False), paired={1})

        assert answer == (False, "stamp")

    def test_an_unstamped_media_falls_back_to_the_pairing(self, aq):
        media = _positive(["bus@small"], coco_scored=None)

        assert aq.coco_answer(1, media, paired={1}) == (True, "pairing")
        assert aq.coco_answer(1, media, paired=set()) == (False, "pairing")

    def test_an_unstamped_media_with_no_pairing_refuses_to_guess(self, aq):
        """Defaulting to "not scored" would put the anchored half in the queue."""
        with pytest.raises(ValueError, match="no `coco_scored` stamp"):
            aq.coco_answer(1, _positive(["bus@small"], coco_scored=None), None)


class TestPairedImageIds:
    def test_only_images_carrying_a_coco_id_are_paired(self, aq, tmp_path):
        p = tmp_path / "image_data.json"
        p.write_text(json.dumps([{"image_id": 1, "coco_id": 42}, {"image_id": 2, "coco_id": None}, {"image_id": 3}]))

        assert aq.paired_image_ids(p) == {1}

    def test_a_missing_join_says_how_to_get_it(self, aq, tmp_path):
        with pytest.raises(SystemExit, match="coco_anchor.py --fetch"):
            aq.paired_image_ids(tmp_path / "absent.json")


class TestQueueRows:
    def test_only_off_coco_positives_are_owed(self, aq):
        medias = {
            1: _positive(["bus@small"], coco_scored=False, iid=1),
            2: _positive(["bus@small"], coco_scored=True, iid=2),
            900: _negative(iid=900),
        }

        assert [r["image_id"] for r in aq.queue_rows(medias)] == [1]

    def test_a_negative_is_never_asked_for_an_answer(self, aq):
        """A pool built before the stamp must not make the queue unemittable.

        Negatives are drawn from the COCO-scored half by rule (#3670) and the
        pass owes them nothing either way, so they are skipped *before* the
        source question is asked -- not answered by the pairing and discarded.
        """
        medias = {900: _negative(coco_scored=None, iid=900)}

        assert aq.queue_rows(medias, paired=None) == []

    def test_one_row_carries_every_cell_the_image_was_designated_in(self, aq):
        """One exhaustive judgement per image, not one per cell."""
        medias = {1: _positive(["bus@small", "car@large", "bus@medium"], iid=1)}

        (row,) = aq.queue_rows(medias)

        assert row["cells"] == ["bus@medium", "bus@small", "car@large"]
        assert row["classes"] == ["bus", "car"]

    def test_a_row_carries_what_a_reviewer_needs_to_open_it(self, aq):
        (row,) = aq.queue_rows({1: _positive(["bus@small"], iid=1)})

        assert row["path"] == "/vg/VG_100K/1.jpg"
        assert row["filename"] == "1.jpg"
        assert row["coco_source"] == "stamp"

    def test_the_order_is_a_function_of_the_seed_alone(self, aq):
        """The first batch is the pilot, so it must be reproducible and unbiased."""
        medias = {i: _positive([f"c{i % 5}@small"], iid=i) for i in range(200)}

        first = [r["image_id"] for r in aq.queue_rows(medias, seed=7)]
        again = [r["image_id"] for r in aq.queue_rows(medias, seed=7)]
        other = [r["image_id"] for r in aq.queue_rows(medias, seed=8)]

        assert first == again
        assert first != other
        assert sorted(first) == sorted(other) == list(range(200))

    def test_the_queue_is_not_in_id_order(self, aq):
        """An id-ordered queue makes the pilot a sample of VG's id space, not of C."""
        medias = {i: _positive(["bus@small"], iid=i) for i in range(200)}

        assert [r["image_id"] for r in aq.queue_rows(medias)] != list(range(200))


class TestBandColumns:
    def test_only_designated_bands_get_a_column(self, aq):
        rows = aq.queue_rows({1: _positive(["bus@small"], iid=1)})

        assert aq.band_columns(rows, ("small", "medium", "large")) == ("small",)

    def test_an_unbanded_cell_gets_no_band_columns(self, aq):
        """`vg_scale_deep` keys cells on the bare class, and three zero columns lie."""
        rows = aq.queue_rows({1: _positive(["bus"], iid=1)})

        assert aq.band_columns(rows, ("small", "medium", "large")) == ()

    def test_the_columns_keep_the_config_order(self, aq):
        """Not the order the cells happened to be counted in."""
        rows = aq.queue_rows({1: _positive(["bus@large"], iid=1), 2: _positive(["bus@small"], iid=2)})

        assert aq.band_columns(rows, ("small", "medium", "large")) == ("small", "large")


class TestRosterGaps:
    """The queue refuses to be a worklist for a set a rebuild can reshuffle (#3727)."""

    def test_a_roster_that_pins_the_queue_has_no_gap(self, aq):
        rows = aq.queue_rows({1: _positive(["bus@small", "car@large"], iid=1)})

        pinned, unpinned, examples = aq.roster_gaps(rows, {"cells": {"bus@small": [1], "car@large": [1]}})

        assert (pinned, unpinned, examples) == (2, 0, [])

    def test_gaps_are_counted_per_designation_not_per_image(self, aq):
        """An image pinned in two of its three cells is not a pinned image."""
        rows = aq.queue_rows({1: _positive(["bus@small", "car@large"], iid=1)})

        pinned, unpinned, examples = aq.roster_gaps(rows, {"cells": {"bus@small": [1]}})

        assert (pinned, unpinned) == (1, 1)
        assert examples == ["1 in car@large"]

    def test_an_empty_roster_pins_nothing(self, aq):
        rows = aq.queue_rows({1: _positive(["bus@small"], iid=1)})

        assert aq.roster_gaps(rows, {}) == (0, 1, ["1 in bus@small"])

    def test_string_ids_in_a_roster_still_match(self, aq):
        """JSON round-trips have made these strings before; a false gap is a false alarm."""
        rows = aq.queue_rows({1: _positive(["bus@small"], iid=1)})

        assert aq.roster_gaps(rows, {"cells": {"bus@small": ["1"]}}) == (1, 0, [])


class TestCounts:
    def test_a_class_is_counted_once_per_image_and_a_cell_once_per_designation(self, aq):
        """Two bands of one class are two cells and one image owing one judgement."""
        rows = aq.queue_rows({1: _positive(["bus@small", "bus@large"], iid=1)})

        per_class, per_cell, per_source = aq.counts(rows)

        assert per_class == {"bus": 1}
        assert per_cell == {"bus@small": 1, "bus@large": 1}
        assert per_source == {"stamp": 1}

    def test_the_sources_are_counted_separately(self, aq):
        """How much of the queue rests on the fallback is the queue's own caveat."""
        medias = {
            1: _positive(["bus@small"], coco_scored=False, iid=1),
            2: _positive(["bus@small"], coco_scored=None, iid=2),
        }

        _class, _cell, per_source = aq.counts(aq.queue_rows(medias, paired=set()))

        assert per_source == {"stamp": 1, "pairing": 1}
