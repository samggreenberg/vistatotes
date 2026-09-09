"""Pooling a group of VG names must not become a licence to fold one (#3636).

``name_evidence.py --pooled`` adjudicates a *group* of VG names as one
hypothesis and lets a member with too little evidence of its own inherit the
verdict. That is the only way to reach the 76 candidates #3618 left
`unmeasured`, and it is also one careless step from the mechanical head-noun
fold that same study refuted -- where `hot dog` (405 VG images, 0 of 181 really
a dog) rides into :data:`pile_config.SCALE_VG_NAMES` behind `puppy`.

Four properties keep the two apart, and each is planted here against a
synthetic COCO/VG overlap whose answers are known by construction:

* **a group is counted over images, not summed over names** -- an image
  carrying two members is one adjudicable image;
* **a group whose measured members disagree is not one hypothesis** and yields
  nothing (the homogeneity gate);
* **an individual measurement always wins** -- a name with a rate of its own
  never inherits, in either direction;
* **a member whose own boxes contradict its group cannot be folded**, only
  withheld, because folding is the claim that costs a mis-banded positive.

Like ``test_pile_coco_folds.py`` these run the real script in a subprocess:
``pile_config.setup_env()`` rewrites ``os.environ`` and ``sys.meta_path`` at
import, which is not something to do to the test process.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_PILE_DIR = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "pile"

#: One VG image per COCO image, same pixel dimensions, so a co-located box has
#: IoU 1.0 and box agreement is decided by where the fixture puts the box.
_W, _H = 640, 480

#: COCO ids for the two classes the fixtures use. `umbrella` is the class whose
#: colour family #3636 is really about; `bench` is an unrelated object to put a
#: box on when a name must NOT agree with the class.
_CAT_UMBRELLA, _CAT_BENCH = 28, 15

_UMBRELLA_BOX = [100.0, 100.0, 200.0, 200.0]
_BENCH_BOX = [400.0, 300.0, 100.0, 100.0]


def _obj(name: str, box: list[float]) -> dict:
    x, y, w, h = box
    return {"names": [name], "x": x, "y": y, "w": w, "h": h}


class _Overlap:
    """A synthetic VG-COCO overlap, built one image at a time.

    Every image gets a COCO record and a VG record with the same dimensions.
    ``present`` decides whether COCO annotates an umbrella on it, which is the
    ground truth the repair-precision question is asked against.
    """

    def __init__(self) -> None:
        self.images: list[dict] = []
        self.annotations: list[dict] = []
        self.vg: list[dict] = []
        self.meta: list[dict] = []
        self._n = 0

    def add(self, vg_names: list[tuple[str, list[float]]], *, present: bool, on_coco: bool = True) -> None:
        i = self._n
        self._n += 1
        cid, vid = 1000 + i, 2000 + i
        self.vg.append({"image_id": vid, "objects": [_obj(n, b) for n, b in vg_names]})
        if not on_coco:
            # The other half of VG: no coco_id, so it is supply and never truth.
            self.meta.append({"image_id": vid, "width": _W, "height": _H})
            return
        self.meta.append({"image_id": vid, "coco_id": cid, "width": _W, "height": _H})
        self.images.append({"id": cid, "width": _W, "height": _H})
        self.annotations.append(
            {
                "id": 10 * i + 1,
                "image_id": cid,
                "category_id": _CAT_UMBRELLA if present else _CAT_BENCH,
                "bbox": list(_UMBRELLA_BOX if present else _BENCH_BOX),
            }
        )

    def stage(self, root: Path) -> tuple[Path, dict[str, str]]:
        anchor = root / "coco_anchor"
        anchor.mkdir(parents=True)
        (anchor / "instances_val2017.json").write_text(
            json.dumps(
                {
                    "categories": [{"id": _CAT_UMBRELLA, "name": "umbrella"}, {"id": _CAT_BENCH, "name": "bench"}],
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


def _evidence(root: Path, overlap: _Overlap, candidates: dict[str, list[str]], *, extra: tuple[str, ...] = ()) -> dict:
    """Run the real script in pooled mode and return its JSON output."""
    anchor, env = overlap.stage(root)
    cands = root / "cands.json"
    cands.write_text(json.dumps(candidates))
    out = root / "evidence.json"
    result = subprocess.run(  # noqa: S603  # interpreter + test-controlled args
        [
            sys.executable,
            "name_evidence.py",
            "--candidates",
            str(cands),
            "--anchor-dir",
            str(anchor),
            "--pooled",
            "--out",
            str(out),
            *extra,
        ],
        cwd=str(_PILE_DIR),
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )
    assert result.returncode == 0, f"name_evidence failed:\n{result.stdout}\n{result.stderr}"
    return json.loads(out.read_text())


def _colour_family(
    *,
    thin_hits: int = 4,
    thin_total: int = 4,
    thin_box_on_class: bool = True,
    measured_hits: int = 30,
    measured_total: int = 30,
    dissenter_hits: int | None = None,
) -> _Overlap:
    """`blue umbrella` measurable, `green umbrella` too thin to be, plus dissent.

    The default shape is the one #3636 describes: a colour compound with plenty
    of evidence (`blue umbrella`) and one with four adjudicable images, one
    short of the floor (`green umbrella`). ``dissenter_hits`` adds a third
    measured member (`pink umbrella`) that disagrees, to fire the gate.
    """
    o = _Overlap()
    for k in range(measured_total):
        o.add([("blue umbrella", _UMBRELLA_BOX)], present=k < measured_hits)
    for k in range(thin_total):
        box = _UMBRELLA_BOX if thin_box_on_class else _BENCH_BOX
        o.add([("green umbrella", box)], present=k < thin_hits)
    if dissenter_hits is not None:
        for k in range(30):
            o.add([("pink umbrella", _UMBRELLA_BOX)], present=k < dissenter_hits)
    # Off-COCO supply, so the `off_coco_sole` column is non-zero and the group
    # has something to act on. These are never adjudicated.
    for _ in range(6):
        o.add([("green umbrella", _UMBRELLA_BOX)], present=False, on_coco=False)
    # Base rate needs images where the class name itself carries the class.
    for _ in range(10):
        o.add([("umbrella", _UMBRELLA_BOX)], present=True)
    return o


_CANDS = {"umbrella": ["blue umbrella", "green umbrella", "pink umbrella"]}


@pytest.fixture(scope="module")
def clean(tmp_path_factory) -> dict:
    """One measured member at 100%, one below the floor. The group should fold."""
    return _evidence(tmp_path_factory.mktemp("clean"), _colour_family(), _CANDS)


@pytest.fixture(scope="module")
def heterogeneous(tmp_path_factory) -> dict:
    """A second measured member at 0%: the colour family is not one hypothesis."""
    return _evidence(
        tmp_path_factory.mktemp("het"),
        _colour_family(dissenter_hits=0),
        _CANDS,
    )


@pytest.fixture(scope="module")
def refuted(tmp_path_factory) -> dict:
    """The measured member is mostly wrong, so the group clears no cut."""
    return _evidence(
        tmp_path_factory.mktemp("refuted"),
        _colour_family(measured_hits=3, thin_hits=1),
        _CANDS,
    )


@pytest.fixture(scope="module")
def box_dissent(tmp_path_factory) -> dict:
    """The thin member's own boxes are on a bench: present, but not the object.

    It must stay *below* the sole floor -- four adjudicable images -- or it
    acquires a verdict of its own and never reaches the inheritance path at
    all. The boxes it needs to be vetoed on come from stacking five of them on
    each of those four images, not from adding images.
    """
    o = _Overlap()
    for _ in range(30):
        o.add([("blue umbrella", _UMBRELLA_BOX)], present=True)
    for _ in range(4):
        o.add([("green umbrella", _BENCH_BOX)] * 5, present=True)
    for _ in range(6):
        o.add([("green umbrella", _UMBRELLA_BOX)], present=False, on_coco=False)
    for _ in range(10):
        o.add([("umbrella", _UMBRELLA_BOX)], present=True)
    return _evidence(tmp_path_factory.mktemp("boxes"), o, _CANDS)


def _group(payload: dict, cls: str, key: str) -> dict:
    return payload["groups"][cls][key]


def _name(payload: dict, cls: str, name: str) -> dict:
    return payload["names"][cls][name]


class TestTheGroupIsCountedOverImages:
    def test_an_image_with_two_members_is_one_adjudicable_image(self, tmp_path) -> None:
        """Summing the members would count it twice and halve the interval."""
        o = _Overlap()
        for _ in range(8):
            o.add([("blue umbrella", _UMBRELLA_BOX), ("green umbrella", _UMBRELLA_BOX)], present=True)
        for _ in range(10):
            o.add([("umbrella", _UMBRELLA_BOX)], present=True)
        payload = _evidence(tmp_path, o, _CANDS)
        grp = _group(payload, "umbrella", "colour")
        assert grp["sole"] == 8, "8 images carrying two members each is 8, not 16"
        assert grp["sole_present"] == 8
        # The per-name counts still see the image once apiece: 8 + 8 = 16.
        assert _name(payload, "umbrella", "blue umbrella")["sole"] == 8
        assert _name(payload, "umbrella", "green umbrella")["sole"] == 8

    def test_boxes_are_summed_because_two_boxes_are_two_objects(self, tmp_path) -> None:
        o = _Overlap()
        for _ in range(8):
            o.add([("blue umbrella", _UMBRELLA_BOX), ("green umbrella", _UMBRELLA_BOX)], present=True)
        for _ in range(10):
            o.add([("umbrella", _UMBRELLA_BOX)], present=True)
        grp = _group(_evidence(tmp_path, o, _CANDS), "umbrella", "colour")
        assert grp["boxes"] == 16


class TestAThinMemberInheritsFromAGroupThatClears:
    def test_the_group_is_adjudicated_and_folds(self, clean: dict) -> None:
        grp = _group(clean, "umbrella", "colour")
        assert grp["verdict"] == "alias"
        assert grp["dissent"] == []

    def test_the_thin_member_had_no_verdict_of_its_own(self, clean: dict) -> None:
        assert _name(clean, "umbrella", "green umbrella")["verdict"] == "unmeasured"

    def test_and_now_carries_the_group_s(self, clean: dict) -> None:
        row = _name(clean, "umbrella", "green umbrella")
        assert row["final"] == "alias"
        assert row["inherited_from"] == ["colour"]

    def test_the_proposal_carries_it(self, tmp_path) -> None:
        """What the tables actually receive, which is the point of the mode."""
        anchor, env = _colour_family().stage(tmp_path)
        cands = tmp_path / "c.json"
        cands.write_text(json.dumps(_CANDS))
        prop = tmp_path / "prop.json"
        result = subprocess.run(  # noqa: S603  # interpreter + test-controlled args
            [
                sys.executable,
                "name_evidence.py",
                "--candidates",
                str(cands),
                "--anchor-dir",
                str(anchor),
                "--pooled",
                "--propose-out",
                str(prop),
            ],
            cwd=str(_PILE_DIR),
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
        )
        assert result.returncode == 0, result.stderr
        assert "green umbrella" in json.loads(prop.read_text())["alias"]["umbrella"]


class TestAnIndividualMeasurementAlwaysWins:
    def test_a_measured_member_keeps_its_own_verdict(self, clean: dict) -> None:
        row = _name(clean, "umbrella", "blue umbrella")
        assert row["verdict"] == "alias"
        assert row["final"] == "alias"
        assert row["inherited_from"] == [], "a name with a rate of its own never inherits"

    def test_a_measured_member_is_not_promoted_by_its_group(self, tmp_path) -> None:
        """The direction that would silently reverse a shipped decision.

        `blue umbrella` is measurable and its own boxes are on a bench, so it is
        `ambiguous` alone. Its group folds; it must still not fold.
        """
        o = _Overlap()
        for k in range(30):
            # One box on the umbrella and two off it: 33% agreement, which is
            # above `--context-box` and below `--min-box`, so `blue umbrella`
            # is `ambiguous` when measured alone.
            o.add(
                [("blue umbrella", _UMBRELLA_BOX), ("blue umbrella", _BENCH_BOX), ("blue umbrella", _BENCH_BOX)],
                present=True,
            )
            if k < 20:
                # `pink umbrella` carries the group over the box cut on its own.
                o.add([("pink umbrella", _UMBRELLA_BOX)] * 4, present=True)
        for _ in range(4):
            o.add([("green umbrella", _UMBRELLA_BOX)], present=True)
        for _ in range(10):
            o.add([("umbrella", _UMBRELLA_BOX)], present=True)
        payload = _evidence(tmp_path, o, _CANDS)
        assert _group(payload, "umbrella", "colour")["verdict"] == "alias"
        blue = _name(payload, "umbrella", "blue umbrella")
        assert blue["verdict"] == "ambiguous"
        assert blue["final"] == "ambiguous"
        assert blue["inherited_from"] == []


class TestAGroupWhoseMembersDisagreeIsNotOneHypothesis:
    def test_the_gate_names_the_dissenters(self, heterogeneous: dict) -> None:
        """`blue umbrella` is 30 of 30 and `pink umbrella` 0 of 30.

        Both are named, and that is right rather than a quirk: the reference is
        the rate the members jointly imply, which lands between them, so a
        two-sided disagreement has two sides. What the gate has to get right is
        that the group is refused, not which member is "at fault".
        """
        grp = _group(heterogeneous, "umbrella", "colour")
        assert grp["verdict"] == "heterogeneous"
        assert set(grp["dissent"]) == {"blue umbrella", "pink umbrella"}

    def test_and_nothing_is_inherited(self, heterogeneous: dict) -> None:
        row = _name(heterogeneous, "umbrella", "green umbrella")
        assert row["final"] == "unmeasured"
        assert row["inherited_from"] == []

    def test_a_refuted_group_also_yields_nothing(self, refuted: dict) -> None:
        grp = _group(refuted, "umbrella", "colour")
        assert grp["verdict"] == "neither"
        assert _name(refuted, "umbrella", "green umbrella")["final"] == "unmeasured"

    def test_members_that_all_agree_exactly_are_not_dissent(self, tmp_path) -> None:
        """The p = 1 case, which fired the gate on a group with no disagreement.

        At 100% the Wilson upper end is analytically 1 and evaluates to
        0.9999999999999999, so `butter knife` (23 of 23) was recorded as
        dissenting from a pooled rate of exactly 1.0 -- and `spelling:butterknife`,
        two spellings of one word both perfect, came out `heterogeneous`.
        """
        payload = _evidence(
            tmp_path,
            _colour_family(dissenter_hits=30),  # blue 30/30, pink 30/30, green 4/4
            _CANDS,
        )
        grp = _group(payload, "umbrella", "colour")
        assert grp["sole_present"] == grp["sole"], "fixture check: every adjudicable image is a hit"
        assert grp["dissent"] == []
        assert grp["verdict"] == "alias"

    def test_the_reference_rate_is_member_weighted_not_image_weighted(self, tmp_path) -> None:
        """A member must be compared against a denominator it is part of.

        The group's own `sole` is a union over images, so a member sharing an
        image with another is counted once there and once in each member. Tested
        against that union, `paper` and `papers` BOTH dissented from a rate
        lying between them -- an impossible verdict that is an artefact of the
        mismatch, not a disagreement.
        """
        o = _Overlap()
        for k in range(40):
            # Every image carries both members, so the union is half the
            # member-weighted denominator.
            o.add(
                [("blue umbrella", _UMBRELLA_BOX), ("pink umbrella", _UMBRELLA_BOX)],
                present=k < 20,
            )
        for _ in range(10):
            o.add([("umbrella", _UMBRELLA_BOX)], present=True)
        payload = _evidence(tmp_path, o, _CANDS)
        grp = _group(payload, "umbrella", "colour")
        assert grp["sole"] == 40
        assert grp["member_sole"] == 80, "each member sees all 40 images"
        assert grp["member_rate"] == grp["sole_present"] / grp["sole"]
        assert grp["dissent"] == [], "two members with identical rates never disagree"

    def test_the_gate_is_adjusted_for_how_many_members_it_tests(self, clean: dict) -> None:
        """Un-adjusted, a ten-member group fires on nothing 40% of the time.

        The clean fixture's two measured members agree exactly, so this only
        pins that the adjustment is applied at all -- the field is reported.
        """
        assert clean["meta"]["homogeneity_alpha"] == 0.05


class TestFoldingIsVetoedByTheMemberSOwnBoxes:
    def test_a_member_whose_boxes_miss_the_class_is_withheld_not_folded(self, box_dissent: dict) -> None:
        """Present on the image, and not the object: the ambiguous table.

        This is the #3616 hazard -- a band is a claim about one object's size,
        so a folded name whose box frames something else injects a mis-banded
        positive. Withholding it costs a few pool images instead.
        """
        grp = _group(box_dissent, "umbrella", "colour")
        assert grp["verdict"] == "alias", "fixture check: the group itself clears the box cut"
        row = _name(box_dissent, "umbrella", "green umbrella")
        assert row["verdict"] == "unmeasured", "fixture check: it has no verdict of its own"
        assert row["boxes"] == 20 and row["boxes_on_class"] == 0
        assert row["final"] == "ambiguous"
        assert row["inherited_from"] == ["colour"]


class TestTheGroupingIsDeclaredNotInferred:
    def test_a_name_whose_head_is_not_the_class_s_is_not_a_colour_compound(self, tmp_path) -> None:
        """`black bag` is not a `backpack` colour compound, and `sign` proves why.

        The construction is `<colour> <the class's head noun>`. Matching on the
        modifier alone pulls in `black face` for `clock` and `black bag` for
        `backpack`, which is how a group stops measuring one thing.
        """
        o = _Overlap()
        for _ in range(30):
            o.add([("blue umbrella", _UMBRELLA_BOX)], present=True)
            o.add([("blue bench", _BENCH_BOX)], present=False)
        for _ in range(10):
            o.add([("umbrella", _UMBRELLA_BOX)], present=True)
        payload = _evidence(tmp_path, o, {"umbrella": ["blue umbrella", "green umbrella", "blue bench"]})
        assert "blue bench" not in _group(payload, "umbrella", "colour")["members"]

    def test_a_one_member_group_is_not_a_group(self, tmp_path) -> None:
        """It is the same measurement under another name, with the floor bypassed."""
        o = _Overlap()
        for _ in range(4):
            o.add([("green umbrella", _UMBRELLA_BOX)], present=True)
        for _ in range(10):
            o.add([("umbrella", _UMBRELLA_BOX)], present=True)
        payload = _evidence(tmp_path, o, {"umbrella": ["green umbrella"]})
        assert "umbrella" not in payload.get("groups", {})
        assert _name(payload, "umbrella", "green umbrella")["final"] == "unmeasured"

    def test_a_count_compound_can_be_evidence_but_never_a_band(self, tmp_path) -> None:
        """`two umbrellas` names a SET, and a band is a claim about one object."""
        o = _Overlap()
        for _ in range(30):
            o.add([("two umbrellas", _UMBRELLA_BOX)], present=True)
        for _ in range(4):
            o.add([("three umbrellas", _UMBRELLA_BOX)], present=True)
        for _ in range(10):
            o.add([("umbrella", _UMBRELLA_BOX)], present=True)
        payload = _evidence(tmp_path, o, {"umbrella": ["two umbrellas", "three umbrellas"]})
        grp = _group(payload, "umbrella", "count")
        assert grp["foldable"] is False
        assert grp["boxes_on_class"] == grp["boxes"], "every box is on the class, so only the cap stops the fold"
        assert grp["verdict"] == "ambiguous"
        assert _name(payload, "umbrella", "three umbrellas")["final"] == "ambiguous"


class TestPooledModeIsOptIn:
    def test_without_the_flag_nothing_is_grouped(self, tmp_path) -> None:
        """#3618's numbers must be reproducible from the same script."""
        anchor, env = _colour_family().stage(tmp_path)
        cands = tmp_path / "c.json"
        cands.write_text(json.dumps(_CANDS))
        out = tmp_path / "ev.json"
        result = subprocess.run(  # noqa: S603  # interpreter + test-controlled args
            [
                sys.executable,
                "name_evidence.py",
                "--candidates",
                str(cands),
                "--anchor-dir",
                str(anchor),
                "--out",
                str(out),
            ],
            cwd=str(_PILE_DIR),
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
        )
        assert result.returncode == 0, result.stderr
        payload = json.loads(out.read_text())
        assert payload["groups"] == {}
        assert payload["names"]["umbrella"]["green umbrella"]["final"] == "unmeasured"
