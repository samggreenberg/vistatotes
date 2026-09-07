"""One aspect-drift guard, read one way, at every call site (#3657).

Whether a COCO box may transfer to the VG copy of the same image is decided by
comparing the two aspect ratios against ``pile_config.MAX_ASPECT_DRIFT``. That
one constant was being read two ways:

* the loader (``pilebuild/loaders/vg_scale.py``) divided by COCO's ratio, i.e.
  took it as a **relative** drift;
* ``coco_folds.py``, ``name_evidence.py``, ``name_coverage.py`` and
  ``band_fold.py`` compared a bare difference of ratios, i.e. took it as an
  **absolute** one.

Neither is uniformly stricter: the two disagree in *both* directions depending
on the original's orientation, so the set of images the measurement called
adjudicable was not the set the build anchored, and the gap between them was
skewed by orientation. The population is small -- the loader reports 49 of
51,497 overlaps re-framed -- which is exactly why it wanted fixing before a
number rested on it.

These pin the resolution: the relative reading (the one that shipped the
dataset) is the only one, it lives in ``pile_config.aspect_transferable``, and
no call site may spell the comparison out again.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

_PILE_DIR = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "pile"


@pytest.fixture(scope="module")
def pc():
    """``pile_config``, which is constants only -- ``setup_env()`` is explicit."""
    if str(_PILE_DIR) not in sys.path:
        sys.path.insert(0, str(_PILE_DIR))
    import pile_config

    return pile_config


# --- the reading ---------------------------------------------------------

#: A landscape 4:3 original against a VG copy 6 px wider. The ratios differ by
#: 0.0125 absolute, which the old scripts rejected, and by 0.009375 relative,
#: which the loader accepted: the build anchored this image and the measurement
#: did not count it.
_LANDSCAPE = ((646, 480), (640, 480))

#: The exact mirror, at 3:4. The same two numbers swap roles -- 0.009375
#: absolute, 0.0125 relative -- so here it is the *scripts* that were looser:
#: they counted an image the build refused to anchor.
_PORTRAIT = ((486, 640), (480, 640))


def test_drift_is_relative_to_the_coco_ratio(pc):
    """The denominator is COCO's ratio, because COCO's is the framing the box came in."""
    assert pc.aspect_drift(*_LANDSCAPE) == pytest.approx(0.009375)
    assert pc.aspect_drift(*_PORTRAIT) == pytest.approx(0.012500)


def test_the_two_readings_disagreed_in_both_directions(pc):
    """Not a uniform margin: which spelling is stricter depends on orientation."""
    assert pc.MAX_ASPECT_DRIFT == pytest.approx(0.01), "the fixtures are sized to the shipped 0.01"

    for vg_wh, coco_wh in (_LANDSCAPE, _PORTRAIT):
        absolute = abs((vg_wh[0] / vg_wh[1]) - (coco_wh[0] / coco_wh[1]))
        relative = pc.aspect_drift(vg_wh, coco_wh)
        # Each fixture straddles the threshold, and does so the opposite way.
        assert (absolute > pc.MAX_ASPECT_DRIFT) != (relative > pc.MAX_ASPECT_DRIFT)

    # And the surviving reading is the loader's, on both.
    assert pc.aspect_transferable(*_LANDSCAPE) is True
    assert pc.aspect_transferable(*_PORTRAIT) is False


def test_identical_framings_transfer_and_transposed_ones_do_not(pc):
    """The two ends the guard exists for."""
    assert pc.aspect_transferable((640, 480), (640, 480)) is True
    # The re-framed copies the loader's comment names: VG 500x375 against COCO
    # 375x500 is a rotation, and COCO's box describes none of VG's pixels.
    assert pc.aspect_transferable((500, 375), (375, 500)) is False


def test_the_threshold_is_a_ceiling_not_a_floor(pc):
    """``> MAX_ASPECT_DRIFT`` rejected; a drift at the threshold is kept.

    Sampled just either side rather than exactly on it: no pair of dimensions
    puts the quotient exactly on 0.01 in binary, so an "exact boundary" fixture
    would be pinning a rounding accident rather than the comparison.
    """
    coco_wh = (640, 480)
    coco_ratio = coco_wh[0] / coco_wh[1]
    for fraction, transfers in ((0.999, True), (1.001, False)):
        vg_ratio = coco_ratio * (1 + pc.MAX_ASPECT_DRIFT * fraction)
        assert pc.aspect_transferable((vg_ratio, 1.0), coco_wh) is transfers


def test_it_reproduces_the_arithmetic_that_shipped_the_dataset(pc):
    """The loader's former expression, verbatim, over a spread of framings.

    The relative reading is not a coin toss between two defensible forms: it is
    the one the built pile was anchored under, so the predicate has to agree
    with it everywhere, not merely near the threshold.
    """
    coco_dims = [(640, 480), (480, 640), (500, 375), (375, 500), (1024, 1024), (1280, 720)]
    vg_dims = [(w + d, h) for w, h in coco_dims for d in (-40, -6, 0, 6, 40)]
    for wh in coco_dims:
        for vw, vh in vg_dims:
            shipped = abs((vw / vh) - (wh[0] / wh[1])) / (wh[0] / wh[1]) > pc.MAX_ASPECT_DRIFT
            assert pc.aspect_transferable((vw, vh), wh) is (not shipped), f"{(vw, vh)} vs {wh}"


# --- one implementation --------------------------------------------------


def _files_naming_the_constant() -> dict[str, list[int]]:
    """Every ``scripts/experiments/pile`` file that reads ``MAX_ASPECT_DRIFT``.

    Read from the AST rather than the text so a mention in a comment or a
    docstring -- of which the fix left several, deliberately -- does not count
    as a second implementation.
    """
    hits: dict[str, list[int]] = {}
    for path in sorted(_PILE_DIR.rglob("*.py")):
        if path.name == "pile_config.py":
            continue  # the definition, and the one function allowed to read it
        for node in ast.walk(ast.parse(path.read_text())):
            named = (isinstance(node, ast.Name) and node.id == "MAX_ASPECT_DRIFT") or (
                isinstance(node, ast.Attribute) and node.attr == "MAX_ASPECT_DRIFT"
            )
            if named:
                hits.setdefault(str(path.relative_to(_PILE_DIR)), []).append(node.lineno)
    return hits


def test_no_call_site_reads_the_constant_directly():
    """A second reading of the threshold is a second guard, however it is spelled.

    This is the check that keeps the fix from decaying: reconciling six call
    sites once is worth nothing if the seventh is written inline, and an inline
    comparison is indistinguishable at a glance from the correct one -- which is
    how the split lasted four call sites and two orientations.
    """
    hits = _files_naming_the_constant()
    assert not hits, (
        "these files read pile_config.MAX_ASPECT_DRIFT directly instead of asking "
        f"pile_config.aspect_transferable(vg_wh, coco_wh): {hits}. The constant is "
        "a *relative* drift; comparing a bare difference of ratios against it is "
        "the #3657 bug, and it disagrees with the build by orientation."
    )
