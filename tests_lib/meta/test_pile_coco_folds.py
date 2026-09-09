"""Fold-out must answer with COCO's vocabulary, not with the caller's (#3640).

``coco_folds.py``'s **fold-out** column claims to say what COCO class sits under
a VG box of a given name, and that reading is what decides whether a VG spelling
is folded in as an alias or banished to
:data:`pile_config.SCALE_VG_AMBIGUOUS`. It used to load only the boxes of the
classes passed to ``--classes``, so a class nobody had happened to name carried
no boxes and every VG box over it fell through to ``(no COCO class)``.

The failure direction is the dangerous one: `bike` read **100%** "means nothing"
against a recorded 40.1%, and a 100% reading sends a perfectly good spelling to
the ambiguous table, costing the class half its positives -- the #3605 defect,
re-created by the tool built to detect it.

What the fix pins is an *invariance*: fold-out for a name must not depend on
which other classes the caller asked about. These tests run the real script in a
subprocess (``pile_config.setup_env()`` rewrites ``os.environ`` and
``sys.meta_path`` at import, which is not something to do to the test process)
over a synthetic COCO/VG overlap built so the two answers differ.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_PILE_DIR = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "pile"

#: One VG image per COCO image, same pixel dimensions, so a box transfers
#: exactly and IoU is 1.0 for a co-located pair.
_W, _H = 640, 480

#: Enough images that the default ``--min-count`` floor is irrelevant either way.
_N_IMAGES = 20


def _coco_annotations() -> dict:
    """Every image carries one `bicycle` box and one `bench` box.

    `bicycle` is the class VG's `bike` really denotes; `bench` is there so a
    second unasked-for class is present, which is what makes an "all classes"
    load distinguishable from "the one class that happens to matter".
    """
    images, annotations = [], []
    for i in range(_N_IMAGES):
        images.append({"id": 1000 + i, "width": _W, "height": _H})
        annotations.append(
            {"id": 10 * i + 1, "image_id": 1000 + i, "category_id": 2, "bbox": [100.0, 100.0, 200.0, 200.0]}
        )
        annotations.append(
            {"id": 10 * i + 2, "image_id": 1000 + i, "category_id": 15, "bbox": [400.0, 300.0, 100.0, 100.0]}
        )
    return {
        "categories": [
            {"id": 2, "name": "bicycle"},
            {"id": 15, "name": "bench"},
            {"id": 47, "name": "cup"},
        ],
        "images": images,
        "annotations": annotations,
    }


def _vg_objects() -> list[dict]:
    """VG names the same pixels `bike` (on the bicycle) and `bench` (on the bench)."""
    return [
        {
            "image_id": 2000 + i,
            "objects": [
                {"names": ["bike"], "x": 100, "y": 100, "w": 200, "h": 200},
                {"names": ["bench"], "x": 400, "y": 300, "w": 100, "h": 100},
            ],
        }
        for i in range(_N_IMAGES)
    ]


def _stage(root: Path) -> tuple[Path, dict[str, str]]:
    """Write a synthetic anchor + VG cache; return the anchor dir and its env."""
    anchor = root / "coco_anchor"
    anchor.mkdir(parents=True)
    (anchor / "instances_val2017.json").write_text(json.dumps(_coco_annotations()))
    # An absent train split is a supported state: the loader logs and skips it.
    (anchor / "image_data.json").write_text(
        json.dumps([{"image_id": 2000 + i, "coco_id": 1000 + i, "width": _W, "height": _H} for i in range(_N_IMAGES)])
    )

    demo = root / "demos"
    (demo / "visual_genome").mkdir(parents=True)
    (demo / "visual_genome" / "objects.json").write_text(json.dumps(_vg_objects()))

    env = {
        **os.environ,
        "VTS_PILE": str(root / "pile"),
        "VTS_DEMO_CACHE": str(demo),
        "VTSEARCH_DATA_DIR": str(root / "pile" / "datadir"),
        "VTSEARCH_MODELS_DIR": str(root / "pile" / "models"),
        "HF_HOME": str(root / "pile" / "models"),
    }
    return anchor, env


def _folds(root: Path, classes: str) -> dict:
    """Run the real script for ``--classes`` and return its JSON output."""
    anchor, env = _stage(root)
    out = root / "folds.json"
    result = subprocess.run(  # noqa: S603  # interpreter + test-controlled args
        [
            sys.executable,
            "coco_folds.py",
            "--classes",
            classes,
            "--anchor-dir",
            str(anchor),
            "--min-count",
            "1",
            "--out",
            str(out),
        ],
        cwd=str(_PILE_DIR),
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
    )
    assert result.returncode == 0, f"coco_folds failed:\n{result.stdout}\n{result.stderr}"
    return json.loads(out.read_text())


@pytest.fixture(scope="module")
def narrow(tmp_path_factory) -> dict:
    """`bike` alone: the caller never names the class its boxes actually sit on."""
    return _folds(tmp_path_factory.mktemp("narrow"), "bike")


@pytest.fixture(scope="module")
def wide(tmp_path_factory) -> dict:
    """`bike` with its target loaded, which is what used to be required."""
    return _folds(tmp_path_factory.mktemp("wide"), "bike,bicycle,bench")


class TestFoldOutSeesTheWholeVocabulary:
    def test_unnamed_target_class_is_found(self, narrow: dict) -> None:
        """The regression: `bicycle` under `bike` without `bicycle` in --classes."""
        assert narrow["fold_out"]["bike"].get("bicycle") == _N_IMAGES

    def test_no_coco_class_is_not_reported_for_a_box_that_has_one(self, narrow: dict) -> None:
        """A 100% `(no COCO class)` is the reading that costs a class its positives."""
        assert narrow["fold_out"]["bike"].get("(no COCO class)", 0) == 0

    def test_fold_out_does_not_depend_on_what_else_was_asked(self, narrow: dict, wide: dict) -> None:
        """The invariance. What a VG name denotes is a fact about VG and COCO."""
        assert narrow["fold_out"]["bike"] == wide["fold_out"]["bike"]
        assert narrow["vg_boxes"]["bike"] == wide["vg_boxes"]["bike"] == _N_IMAGES


class TestFoldInStaysScopedToTheRequestedClasses:
    """Loading the whole vocabulary must not widen the *other* two tables.

    Fold-in and the self-match canary answer a question about the classes the
    caller named; every counter behind them is keyed on that set, so a stray
    `bench` row would be a crash in ``name_coverage.py`` and a bogus row here.
    """

    def test_non_coco_name_has_no_coco_boxes(self, narrow: dict) -> None:
        assert narrow["coco_boxes"].get("bike", 0) == 0
        assert narrow["fold_in"]["bike"] == {}

    def test_only_requested_classes_are_counted(self, narrow: dict) -> None:
        assert set(narrow["fold_in"]) == {"bike"}
        assert set(narrow["coco_boxes"]) <= {"bike"}

    def test_a_requested_coco_class_still_folds_in(self, wide: dict) -> None:
        """`bench` is spelled the same in both vocabularies, so it self-matches."""
        assert wide["coco_boxes"]["bench"] == _N_IMAGES
        assert wide["fold_in"]["bench"].get("bench") == _N_IMAGES
        assert wide["fold_in"]["bicycle"].get("bike") == _N_IMAGES
