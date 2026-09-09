"""A definitional rejection must remove a small-band positive (#3614).

``verdicts_to_corrections.py`` guards the small band: a rejection there is read
as "not confirmed" rather than "absent", because boxed review confirms only
~2/3 of sub-patch positives and the failure tracks object *size*. That guard is
right for confirmability and wrong for identity. Three of the ten
``bicycle@small`` positives in the #3156 review are bicycle pictograms on road
signs -- not bicycles at any resolution -- and the guard swallowed the human
rejection and the adjudicated one alike, leaving them uncorrectable.

These tests pin both halves: the guard still holds for an ordinary small-band
rejection, and an adjudication carrying ``"reason": "definition"`` gets through
it. The script is run in a subprocess because ``pile_config.setup_env()`` edits
``os.environ`` and ``sys.meta_path`` at import.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

_PILE_DIR = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "pile"

_CLASS = "bicycle"
_SMALL = 2374765  # a bicycle pictogram on a road sign
_MEDIUM = 2352009  # an ordinary rejected positive, above the guard


def _slate(root: Path, rows: list[dict]) -> Path:
    d = root / "slates" / _CLASS
    d.mkdir(parents=True)
    cols = ["image_id", "class", "stratum", "cell", "text_score", "reference", "exhaustive", "n_boxes", "detector"]
    with (d / "manifest.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return root / "slates"


def _run(tmp_path: Path, adjudication: list[dict]) -> dict:
    rows = [
        {
            "image_id": _SMALL,
            "class": _CLASS,
            "stratum": "positive_boxed",
            "cell": f"{_CLASS}@small",
            "text_score": "0.0",
            "reference": "present",
            "exhaustive": "yes",
            "n_boxes": "1",
            "detector": _CLASS,
        },
        {
            "image_id": _MEDIUM,
            "class": _CLASS,
            "stratum": "positive_boxed",
            "cell": f"{_CLASS}@medium",
            "text_score": "0.0",
            "reference": "present",
            "exhaustive": "yes",
            "n_boxes": "1",
            "detector": _CLASS,
        },
    ]
    slates = _slate(tmp_path, rows)

    verdicts = [
        {
            "image_id": i,
            "class": _CLASS,
            "stratum": "positive_boxed",
            "human": "absent",
            "reference": "present",
            "exhaustive": "yes",
            "box": None,
            "text_score": 0.0,
            "export": "test",
        }
        for i in (_SMALL, _MEDIUM)
    ]
    vpath = tmp_path / "verdicts.json"
    vpath.write_text(json.dumps(verdicts))
    apath = tmp_path / "adjudication.json"
    apath.write_text(json.dumps(adjudication))
    out = tmp_path / "corrections.json"

    proc = subprocess.run(  # noqa: S603  # interpreter + test-controlled args
        [
            sys.executable,
            "verdicts_to_corrections.py",
            "--verdicts",
            str(vpath),
            "--adjudication",
            str(apath),
            "--slates",
            str(slates),
            "--triage",
            str(tmp_path / "absent.json"),
            "--sheets",
            str(tmp_path / "no_sheets"),
            "--out",
            str(out),
        ],
        cwd=_PILE_DIR,
        capture_output=True,
        text=True,
        check=False,
        # `pile_config.setup_env()` creates the pile dir at import, so without
        # this the test needs the scratch mount and fails everywhere else.
        env={**os.environ, "VTS_PILE": str(tmp_path / "pile")},
    )
    assert proc.returncode == 0, proc.stderr
    return {(r["image_id"], r["class"]): r for r in json.loads(out.read_text())}


def _adj(image_id: int, **extra) -> dict:
    return {
        "image_id": image_id,
        "class": _CLASS,
        "cell": f"{_CLASS}@small",
        "claude": "absent",
        "note": "bicycle pictogram on a road sign",
        **extra,
    }


def test_small_band_rejection_is_still_guarded(tmp_path):
    """Without a reason, the band guard stands: the positive is NOT removed."""
    got = _run(tmp_path, [_adj(_SMALL)])
    assert (_SMALL, _CLASS) not in got


def test_definitional_rejection_removes_a_small_band_positive(tmp_path):
    """``reason: definition`` names the object, so the guard must not apply."""
    got = _run(tmp_path, [_adj(_SMALL, reason="definition")])
    row = got[(_SMALL, _CLASS)]
    assert row["present"] is False
    assert row["boxes"] == []
    assert row["source"] == "human_reject+adjudicated_definition"
    assert "pictogram" in row["note"]


def test_ordinary_adjudicated_rejection_above_the_band_still_works(tmp_path):
    """The pre-existing path is untouched for a positive the guard never covered."""
    got = _run(
        tmp_path,
        [
            {
                "image_id": _MEDIUM,
                "class": _CLASS,
                "cell": f"{_CLASS}@medium",
                "claude": "absent",
                "note": "not a bicycle",
            }
        ],
    )
    assert got[(_MEDIUM, _CLASS)]["source"] == "human_reject+adjudicated"
