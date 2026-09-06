"""A `present` the class cannot hold must not spend a negative (#3676).

``verdicts_to_corrections.py`` turns a reviewer's `present` on a negative into
a correction: boxless it excludes the image from every cell of that class, boxed
it makes the image a positive. Both are the wrong answer when the object the
reviewer correctly saw is one the class's own construction would never have
admitted — a wristwatch for `clock`, a pop-up canopy for `umbrella`. #3666
adjudicated nine such finds and four were exactly that, so the class would have
lost four good negatives on a reading the build does not use.

The gate is the adjudication file the positive side already reads, with the same
two fields, because it is the same sentence pointed the other way:
``"claude": "absent"`` plus ``"reason": "definition"`` means *what the object is*
settles it. These tests pin all three states — refused with a note, applied when
no adjudication names it, and applied when an adjudication exists but is about
confirmability rather than identity — plus the boxed case, which is the more
damaging of the two and is not what the issue's title describes.

The script runs in a subprocess because ``pile_config.setup_env()`` edits
``os.environ`` and ``sys.meta_path`` at import; ``VTS_PILE`` is pointed at the
tmp dir so the run needs no scratch mount and the test is runnable anywhere.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

_PILE_DIR = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "pile"

_CLASS = "clock"
_WATCH = 2408671  # a wristwatch on a bystander's wrist
_REAL = 2327535  # an ordinary contaminated negative, nothing definitional about it


def _slate(root: Path) -> Path:
    """A negative-stratum manifest: no `cell`, because these are pool images."""
    d = root / "slates" / _CLASS
    d.mkdir(parents=True)
    cols = ["image_id", "class", "stratum", "cell", "text_score", "reference", "exhaustive", "n_boxes", "detector"]
    with (d / "manifest.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for iid in (_WATCH, _REAL):
            w.writerow(
                {
                    "image_id": iid,
                    "class": _CLASS,
                    "stratum": "random",
                    "cell": "",
                    "text_score": "0.0",
                    "reference": "",
                    "exhaustive": "no",
                    "n_boxes": "0",
                    "detector": _CLASS,
                }
            )
    return root / "slates"


def _run(tmp_path: Path, adjudication: list[dict], box: list[float] | None = None) -> tuple[dict, str]:
    slates = _slate(tmp_path)
    verdicts = [
        {
            "image_id": iid,
            "class": _CLASS,
            "stratum": "random",
            "human": "present",
            "reference": "",
            "exhaustive": "no",
            "box": box,
            "text_score": 0.0,
            "export": "test",
        }
        for iid in (_WATCH, _REAL)
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
        env={**os.environ, "VTS_PILE": str(tmp_path / "pile")},
    )
    assert proc.returncode == 0, proc.stderr
    return {(r["image_id"], r["class"]): r for r in json.loads(out.read_text())}, proc.stdout


def _adj(image_id: int, **extra) -> dict:
    row = {
        "image_id": image_id,
        "class": _CLASS,
        "claude": "absent",
        "note": "a wristwatch, and `watch` is not a name this class reads",
    }
    row.update(extra)
    return row


def test_an_inadmissible_present_writes_no_correction(tmp_path):
    """The class keeps the negative, and the refusal is named in the output."""
    rows, stdout = _run(tmp_path, [_adj(_WATCH, reason="definition")])
    assert (_WATCH, _CLASS) not in rows, "a `present` the class cannot hold must not exclude the image"
    assert (_REAL, _CLASS) in rows, "the ordinary contaminated negative is still corrected"
    assert "REFUSED" in stdout and str(_WATCH) in stdout, "a refused correction must not be silent"
    assert "wristwatch" in stdout, "the adjudicator's note is what tells the next reader why"


def test_a_boxed_present_is_refused_too(tmp_path):
    """The boxed case manufactures a positive, which is the worse of the two.

    A box on a wristwatch would make the image a `clock` positive in whatever
    band the box implies — an object the class does not believe in, entering the
    half of the benchmark it is scored against.
    """
    rows, stdout = _run(tmp_path, [_adj(_WATCH, reason="definition")], box=[0.4, 0.4, 0.5, 0.5])
    assert (_WATCH, _CLASS) not in rows
    assert rows[(_REAL, _CLASS)]["boxes"], "the ordinary find still carries its box"
    assert "yes" in stdout.split("boxed")[-1], "the report says the refused verdict carried a box"


def test_an_adjudication_about_confirmability_does_not_refuse(tmp_path):
    """`absent` without `reason: definition` is not a statement about identity.

    The positive side draws exactly this line — the small-band guard exists
    because "I cannot tell at 26 px" and "it is not one" are different claims —
    and the negative side must draw it in the same place, or an adjudicator's
    hedge silently becomes a ruling.
    """
    rows, _ = _run(tmp_path, [_adj(_WATCH)])
    assert (_WATCH, _CLASS) in rows, "only a DEFINITIONAL adjudication refuses the correction"


def test_no_adjudication_leaves_the_old_behaviour_untouched(tmp_path):
    """Nothing changes for the verdicts nobody has adjudicated."""
    rows, stdout = _run(tmp_path, [])
    assert (_WATCH, _CLASS) in rows and (_REAL, _CLASS) in rows
    assert "REFUSED" not in stdout


def test_the_emitter_and_the_gate_agree_on_their_two_fields(tmp_path):
    """`shipped_pool_error.py --adjudication-out` writes what this gate reads.

    The two live in different scripts and are joined only by a JSON shape, which
    is the kind of seam that drifts. This asserts the shape rather than trusting
    it: every row the emitter writes must be one the gate refuses.
    """
    out = tmp_path / "adj.json"
    proc = subprocess.run(  # noqa: S603  # interpreter + test-controlled args
        [
            sys.executable,
            "shipped_pool_error.py",
            "--verdicts",
            str(_PILE_DIR.parents[2] / "docs/experiments/2026-09-06-shipped-pool-3666/verdicts.csv"),
            "--adjudication-out",
            str(out),
        ],
        cwd=_PILE_DIR,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "VTS_PILE": str(tmp_path / "pile")},
    )
    assert proc.returncode == 0, proc.stderr
    rows = json.loads(out.read_text())
    assert rows, "the study adjudicated four finds as inadmissible; none reached the file"
    for r in rows:
        assert r["claude"] == "absent" and r["reason"] == "definition", r
        assert r["note"].strip(), "a refusal with no note is a ruling nobody can check"
        assert r["class"] and r["image_id"]
