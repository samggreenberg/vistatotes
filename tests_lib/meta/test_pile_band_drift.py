"""Band drift: a rebox that leaves its cell, and the audit that measures the rest (#3616).

A ``vg_scale`` image sits in ``class@band`` because of the box it arrived with.
A reviewer who redraws that box onto a *different, larger* instance of the same
class therefore moves the image to another cell and vacates the one it was
sampled to fill -- 6 of the first 13 redrawn boxes did.

The move is kept, because VG's annotation is not exhaustive and the sampled band
was the error. Two things had to change around that:

* ``verdicts_to_corrections.py`` now names every band-changing rebox instead of
  applying it silently, which needs a band derived from the *normalised* box the
  reviewer drew -- the coordinate space that has already cost this pile one
  published defect (#3281). :func:`_box_band` is tested here against the
  builder's own ``band_for`` rather than against hand-written edges.
* ``audit_band_drift.py`` measures how often VG's boxes alone would have made
  the same mistake, using the COCO-anchored half as a control. Its verdict
  vocabulary is what the report tabulates, so a verdict outside that vocabulary
  is a row that silently vanishes from the totals.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

_PILE_DIR = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "pile"


@pytest.fixture(scope="module")
def drift():
    """``audit_band_drift``, imported without ``pile_config.setup_env()``.

    The audit keeps its ``coco_anchor`` import inside ``main`` precisely so that
    importing the module does not rewrite the import machinery and ``os.environ`` for
    the whole test process; if that import ever moves back to module scope this
    fixture is what fails.
    """
    if str(_PILE_DIR) not in sys.path:
        sys.path.insert(0, str(_PILE_DIR))
    import audit_band_drift

    return audit_band_drift


@pytest.fixture(scope="module")
def pc():
    if str(_PILE_DIR) not in sys.path:
        sys.path.insert(0, str(_PILE_DIR))
    import pile_config

    return pile_config


class TestAuditState:
    """What one source's reading of an ``(image, class)`` pair comes out as."""

    def test_a_class_the_source_does_not_annotate_is_absent(self, drift):
        assert drift._state({"dog": [[0.0, 0.0, 5.0, 5.0]]}, "bus", (100, 100)) == drift.ABSENT

    def test_an_empty_box_list_is_absent_rather_than_an_error(self, drift):
        """COCO records "annotated and absent" as an empty entry, not a missing one."""
        assert drift._state({"bus": []}, "bus", (100, 100)) == drift.ABSENT

    def test_boxes_are_read_through_the_builders_own_banding(self, drift, pc):
        band, (lo, hi) = next(iter(pc.BOX_BANDS.items()))
        side = (((lo + hi) / 2) * 10000) ** 0.5
        assert drift._state({"bus": [[0.0, 0.0, side, side]]}, "bus", (100, 100)) == band


class TestAuditVerdict:
    """Which disagreements between VG and COCO count as the #3616 defect."""

    @pytest.fixture
    def order(self, pc):
        return list(pc.BOX_BANDS)

    def test_a_larger_coco_band_is_the_defect(self, drift, order):
        assert drift._verdict("small", "medium", order) == "up"
        assert drift._verdict("small", "large", order) == "up"

    def test_a_smaller_coco_band_is_reported_apart_from_it(self, drift, order):
        """An extent error in VG's box, not an instance VG failed to annotate."""
        assert drift._verdict("large", "small", order) == "down"

    def test_the_same_band_agrees(self, drift, order):
        assert drift._verdict("medium", "medium", order) == "agrees"

    def test_the_non_band_readings_pass_through_as_themselves(self, drift, order):
        for state in (drift.ABSENT, drift.SCATTERED, drift.OVERSIZE):
            assert drift._verdict("small", state, order) == state

    def test_every_verdict_it_can_return_is_one_the_report_prints(self, drift, order):
        """The table is built from ``VERDICTS``; anything outside it is a lost row."""
        states = [*order, drift.ABSENT, drift.SCATTERED, drift.OVERSIZE]
        produced = {drift._verdict(vg, coco, order) for vg in order for coco in states}
        assert produced <= set(drift.VERDICTS)

    def test_the_report_prints_nothing_it_cannot_produce(self, drift, order):
        """The other direction: a stale verdict name would print a column of zeros."""
        states = [*order, drift.ABSENT, drift.SCATTERED, drift.OVERSIZE]
        produced = {drift._verdict(vg, coco, order) for vg in order for coco in states}
        assert set(drift.VERDICTS) == produced


class TestAuditReport:
    """The two numbers a reader acts on: the measured rate, and its projection."""

    def _tally(self, pc, audited: list[str]) -> dict[str, dict[str, Counter]]:
        tally = {c: {b: Counter() for b in audited} for c in pc.SCALE_CLASSES}
        first, second = pc.SCALE_CLASSES[0], pc.SCALE_CLASSES[1]
        tally[first]["small"].update({"up": 3, "scattered": 1, "agrees": 16})
        tally[second]["small"].update({"up": 0, "agrees": 20})
        return tally

    def test_the_defect_rate_counts_up_and_scattered_together(self, drift, pc, capsys):
        """Both mean the image does not belong in the band VG put it in."""
        drift._report(self._tally(pc, ["small"]), ["small"], Counter())
        totals = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("ALL")]
        assert len(totals) == 1
        # 4 of 40 over both classes; the per-class rows are 4/20 and 0/20.
        assert totals[0].split()[2] == "40"
        assert totals[0].endswith("10.0%")

    def test_the_projection_applies_the_rate_to_unanchored_seats_only(self, drift, pc, tmp_path, monkeypatch, capsys):
        """An anchored seat already carries COCO's band, so it is not exposed."""
        roster = tmp_path / "roster.json"
        roster.write_text(json.dumps({"cells": {pc.scale_cell(pc.SCALE_CLASSES[0], "small"): list(range(100))}}))
        monkeypatch.setattr(pc, "ROSTER", roster)

        drift._project(["small"], set(range(40)), self._tally(pc, ["small"]))
        row = next(ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("small"))
        # 100 seats, 40 anchored, so 60 exposed; 10% of 60 is 6.
        assert row.split()[1:4] == ["100", "40", "60"]
        assert row.endswith("10.0% of 60 = 6 images")

    def test_a_missing_roster_is_said_rather_than_guessed_at(self, drift, pc, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(pc, "ROSTER", tmp_path / "absent.json")
        drift._project(["small"], {1}, self._tally(pc, ["small"]))
        out = capsys.readouterr().out
        assert "no roster" in out and "projected onto the designated cells" not in out


def _run(code: str, tmp_path: Path) -> subprocess.CompletedProcess:
    """Run *code* inside the pile directory, with the pile pointed at *tmp_path*.

    A subprocess because ``verdicts_to_corrections`` calls
    ``pile_config.setup_env()`` at import, which rewrites ``os.environ`` and
    ``sys.meta_path`` process-wide.
    """
    env = {
        **os.environ,
        "VTS_PILE": str(tmp_path),
        "VTSEARCH_DATA_DIR": str(tmp_path / "datadir"),
        "VTSEARCH_MODELS_DIR": str(tmp_path / "models"),
        "HF_HOME": str(tmp_path / "models"),
    }
    return subprocess.run(  # noqa: S603  # interpreter + test-controlled source
        [sys.executable, "-c", code],
        cwd=str(_PILE_DIR),
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )


class TestCorrectionBoxBand:
    """The band a *normalised* correction box implies, as the rebox report reads it."""

    def test_it_agrees_with_the_builder_at_every_band(self, tmp_path):
        """The report and the rebuild must name the same band, or the report lies.

        The two derive it from the same box in different coordinate spaces --
        normalised here, pixels there -- which is the exact confusion that put
        130 boxes on the frame origin in #3281. So this compares the two
        implementations rather than either one against a written-down edge.
        """
        code = """
import json
import verdicts_to_corrections as vtc
import pile_config as pc
from pilebuild.loaders.vg_scale import band_for

W = H = 640
out = {}
for band, (lo, hi) in pc.BOX_BANDS.items():
    side = ((lo + hi) / 2) ** 0.5
    out[band] = [vtc._box_band([0.0, 0.0, side, side]), band_for([[0.0, 0.0, side * W, side * H]], W, H)]
print(json.dumps(out))
"""
        proc = _run(code, tmp_path)
        assert proc.returncode == 0, proc.stderr
        got = json.loads(proc.stdout.strip().splitlines()[-1])
        assert got, "no bands were compared"
        for band, (from_normalised, from_pixels) in got.items():
            assert from_normalised == band == from_pixels

    def test_a_box_bigger_than_a_region_falls_in_no_band(self, tmp_path):
        """Above ``MAX_VOTED_AREA`` there is no band to move to, and none is invented."""
        code = """
import verdicts_to_corrections as vtc
print(repr(vtc._box_band([0.0, 0.0, 1.0, 1.0])))
"""
        proc = _run(code, tmp_path)
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip().splitlines()[-1] == "''"
