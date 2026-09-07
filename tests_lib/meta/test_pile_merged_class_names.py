"""A class defined as a UNION of COCO classes must be audited as that union (#3700).

`cup` is `cup` ∪ `wine glass` (:data:`pile_config.SCALE_CLASS_MERGES`), and until
this was fixed both name-audit scripts resolved "the class" to the COCO class of
the same name. So a stemware spelling was scored against COCO `cup` alone:
`wine glass` measured 38% repair precision and **2%** box agreement over 151 sole
images and landed in `neither` — a refusal produced entirely by the scorer
looking at the wrong half of the class. `mug`, whose object COCO really does call
a cup, scored 88% / 82% and passed, which is what kept the defect specific to the
merged class and silent everywhere else.

It cost the #3588 promotion six spellings that had to be carried into
`SCALE_VG_NAMES["cup"]` **by hand**, against the audit's own verdict. That is the
right answer reached the wrong way, and the next merged class would repeat it.

Both halves are planted here against a synthetic VG-COCO overlap whose answers
are known by construction, and both fail without the fix:

* `name_evidence.py` — a spelling whose object COCO files under the *partner*
  class must reach `alias`, not `neither`;
* `name_coverage.py` — the class's COCO box count must include the partner's
  boxes, or the coverage denominator is short and the spelling appears to
  recover nothing.

Like `test_pile_pooled_names.py` these run the real scripts in a subprocess:
`pile_config.setup_env()` rewrites `os.environ` and `sys.meta_path` at import,
which is not something to do to the test process.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_PILE_DIR = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "pile"

_W, _H = 640, 480

#: COCO's own ids for the two halves of the merged class, and an unrelated
#: object to put a box on when a name must NOT agree with either half.
_CAT_CUP, _CAT_WINE_GLASS, _CAT_BENCH = 47, 46, 15

_OBJECT_BOX = [100.0, 100.0, 200.0, 200.0]

#: `--min-boxes` defaults to 20: an alias claims every box under the name IS the
#: object, and five boxes cannot carry that claim. So the fixture plants enough
#: images to clear it, or the verdict falls to `ambiguous` for a reason that has
#: nothing to do with the merge.
_N_IMAGES = 22


def _obj(name: str, box: list[float]) -> dict:
    x, y, w, h = box
    return {"names": [name], "x": x, "y": y, "w": w, "h": h}


class _Overlap:
    """A synthetic VG-COCO overlap, built one image at a time.

    ``coco_category`` is what COCO files the object under — the whole point of
    these fixtures is that it is the merge PARTNER rather than the class itself.
    """

    def __init__(self) -> None:
        self.images: list[dict] = []
        self.annotations: list[dict] = []
        self.vg: list[dict] = []
        self.meta: list[dict] = []
        self._n = 0

    def add(self, vg_names: list[tuple[str, list[float]]], *, coco_category: int, coco_box: list[float]) -> None:
        i = self._n
        self._n += 1
        cid, vid = 1000 + i, 2000 + i
        self.vg.append({"image_id": vid, "objects": [_obj(n, b) for n, b in vg_names]})
        self.meta.append({"image_id": vid, "coco_id": cid, "width": _W, "height": _H})
        self.images.append({"id": cid, "width": _W, "height": _H})
        self.annotations.append(
            {"id": 10 * i + 1, "image_id": cid, "category_id": coco_category, "bbox": list(coco_box)}
        )

    def stage(self, root: Path) -> tuple[Path, dict[str, str]]:
        anchor = root / "coco_anchor"
        anchor.mkdir(parents=True)
        (anchor / "instances_val2017.json").write_text(
            json.dumps(
                {
                    "categories": [
                        {"id": _CAT_CUP, "name": "cup"},
                        {"id": _CAT_WINE_GLASS, "name": "wine glass"},
                        {"id": _CAT_BENCH, "name": "bench"},
                    ],
                    "images": self.images,
                    "annotations": self.annotations,
                }
            )
        )
        (anchor / "image_data.json").write_text(json.dumps(self.meta))
        demo = root / "demos"
        (demo / "visual_genome").mkdir(parents=True)
        (demo / "visual_genome" / "objects.json").write_text(json.dumps(self.vg))
        env = {
            **os.environ,
            "VTS_PILE": str(root / "pile"),
            "VTS_DEMO_CACHE": str(demo),
            "VTSEARCH_DATA_DIR": str(root / "pile" / "datadir"),
            "VTSEARCH_MODELS_DIR": str(root / "pile" / "models"),
            "HF_HOME": str(root / "pile" / "models"),
        }
        return anchor, env


def _run(script: str, root: Path, overlap: _Overlap, *args: str) -> dict:
    anchor, env = overlap.stage(root)
    out = root / "out.json"
    result = subprocess.run(  # noqa: S603  # interpreter + test-controlled args
        [sys.executable, script, "--anchor-dir", str(anchor), "--out", str(out), *args],
        cwd=str(_PILE_DIR),
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )
    assert result.returncode == 0, f"{script} failed:\n{result.stdout}\n{result.stderr}"
    return json.loads(out.read_text())


@pytest.fixture(scope="module")
def pc():
    if str(_PILE_DIR) not in sys.path:
        sys.path.insert(0, str(_PILE_DIR))
    import pile_config

    return pile_config


def _stemware_overlap() -> _Overlap:
    """Images where VG says `wine glass`, COCO says `wine glass`, and no `cup`.

    Exactly the state the real data is in: the spelling is right, the object is
    right, and the COCO class carrying it is the merge partner rather than the
    class's namesake.
    """
    overlap = _Overlap()
    for _ in range(_N_IMAGES):
        overlap.add(
            [("wine glass", _OBJECT_BOX)],
            coco_category=_CAT_WINE_GLASS,
            coco_box=_OBJECT_BOX,
        )
    return overlap


class TestTheMergeIsPartOfTheClass:
    def test_the_merge_is_declared_for_cup(self, pc):
        """Guards the guard: these tests say nothing if `cup` stops being a union."""
        assert pc.coco_classes_for("cup") == {"cup", "wine glass"}
        assert pc.coco_classes_for("bench") == {"bench"}, "an unmerged class must resolve to itself"

    def test_a_partner_class_spelling_reaches_alias(self, tmp_path, pc):
        """The regression. Scored against COCO `cup` alone this is `neither`.

        Precision and box agreement are both 1.0 here by construction, so the
        only thing that can produce a refusal is the scorer asking about the
        wrong half of the class.
        """
        cands = tmp_path / "cands.json"
        cands.write_text(json.dumps({"cup": ["wine glass"]}))
        evidence = _run("name_evidence.py", tmp_path, _stemware_overlap(), "--candidates", str(cands))

        row = evidence["names"]["cup"]["wine glass"]
        assert row["sole"] == _N_IMAGES, "every image has the name and not the class name"
        assert row["sole_present"] == _N_IMAGES, (
            "COCO annotates the object on every one of them -- as `wine glass`, "
            "which IS `cup` for this class"
        )
        assert row["boxes_on_class"] == row["boxes"] == _N_IMAGES
        assert row["verdict"] == "alias"

    def test_the_partner_counts_toward_the_base_rate_too(self, tmp_path, pc):
        """A precision is read against the class's base rate, so both must move.

        Fixing the numerator alone would leave the class looking absent on every
        image its partner carries, and every name would read as beating a base
        rate of zero.
        """
        cands = tmp_path / "cands.json"
        cands.write_text(json.dumps({"cup": ["wine glass"]}))
        evidence = _run("name_evidence.py", tmp_path, _stemware_overlap(), "--candidates", str(cands))

        assert evidence["base_rate"]["cup"] == pytest.approx(1.0), (
            "COCO annotates a wine glass on every overlap image, and a wine glass is a cup here"
        )

    def test_an_unmerged_class_is_unaffected(self, tmp_path, pc):
        """The fix must be inert for the other twenty-four classes.

        A `bench` spelling scored against a COCO `wine glass` stays refuted --
        `coco_classes_for` returns `{bench}`, so nothing widens.
        """
        overlap = _Overlap()
        for _ in range(_N_IMAGES):
            overlap.add([("park bench", _OBJECT_BOX)], coco_category=_CAT_WINE_GLASS, coco_box=_OBJECT_BOX)
        cands = tmp_path / "cands.json"
        cands.write_text(json.dumps({"bench": ["park bench"]}))
        evidence = _run("name_evidence.py", tmp_path, overlap, "--candidates", str(cands))

        row = evidence["names"]["bench"]["park bench"]
        assert row["sole_present"] == 0
        assert row["verdict"] == "neither"

    def test_coverage_counts_the_partners_boxes_in_the_denominator(self, tmp_path, pc):
        """`name_coverage.py`'s half of the same defect.

        The denominator is "the class's COCO boxes". For a union that is both
        halves; counting the namesake alone understates what the class has to
        cover AND hides what the spelling recovers.
        """
        proposal = tmp_path / "proposal.json"
        proposal.write_text(json.dumps({"alias": {"cup": ["wine glass"]}, "ambiguous": {}}))
        coverage = _run("name_coverage.py", tmp_path, _stemware_overlap(), "--propose", str(proposal))

        row = coverage["overlap"]["cup"]
        assert row["coco_boxes"] == _N_IMAGES, "every wine-glass box is a box this class must cover"
        assert row["own"] == 0, "the spelling `cup` is on none of these images"
        assert row["alias"] == _N_IMAGES, "and the alias table recovers all of them"
