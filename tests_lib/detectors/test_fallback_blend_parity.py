"""The harness's fallback blend must feed the x-cal side what the app feeds it (#2936).

When calibration cannot form folds at all, the shipped threshold falls back to
the schedule blend — and the app's ``_fused_threshold`` substitutes
``NO_GOOD_THRESHOLD`` ("we never computed a cut, so admit nothing") for the
blend's x-cal input whenever ``folds.fallback is not None``, *whichever*
sentinel the fold rule returned.  The harness used to blend the sentinel itself,
which is ``0.5`` on the too-few / fewer-than-two-per-class paths.

The two only coincide at blend weight 0.  The production schedules ramp from
``lo=6`` labels, so the rare-class-starvation regime (7 good / 1 bad, reachable
under the faithful autopilot flow) carries real weight on the x-cal side and the
two sides disagree by ``weight * (2.0 - 0.5)`` — silently moving the recorded
operating point *and* the acquisition cut in exactly the cold-start steps the
calibration studies examine.
"""

from __future__ import annotations

import numpy as np
import pytest

from vtscore.detectors.training import _fused_threshold
from vtscore.eval.step_model import resolve_hidden_dim
from vtscore.eval.step_trainers import _train_and_calibrate
from vtscore.eval.voting_iterations import _blend_xcal_input, _safe_threshold_for_step
from vtscore.training.blend_schedules import BlendContext
from vtscore.training.thresholds import NO_GOOD_THRESHOLD, calculate_safe_threshold, calibration_folds

DIM = 8
SCHEDULE = "cap50"


def _unit(vec: np.ndarray) -> np.ndarray:
    return (vec / (np.linalg.norm(vec) + 1e-8)).astype(np.float32)


def _starved_votes(rng: np.random.Generator, n_good: int = 7, n_bad: int = 1):
    """A vote set past the schedule's ramp with one class still under 2 votes.

    ``n_good + n_bad > 6`` puts real weight on the x-cal side of the blend, while
    the single Bad vote is what makes ``compute_fold_orderings`` bail to its
    ``0.5`` sentinel instead of calibrating.
    """
    good = _unit(rng.standard_normal(DIM))
    bad = _unit(rng.standard_normal(DIM))
    clips: dict[int, dict] = {}
    good_votes: dict[int, None] = {}
    bad_votes: dict[int, None] = {}
    for i in range(n_good):
        cid = 100 + i
        clips[cid] = _media(cid, _unit(good + 0.05 * rng.standard_normal(DIM)), "cat0")
        good_votes[cid] = None
    for i in range(n_bad):
        cid = 200 + i
        clips[cid] = _media(cid, _unit(bad + 0.05 * rng.standard_normal(DIM)), "cat1")
        bad_votes[cid] = None
    # Unvoted media: the simulation set the blend fits its mixture on.
    for i in range(20):
        cid = 300 + i
        base = good if i % 2 == 0 else bad
        clips[cid] = _media(cid, _unit(base + 0.3 * rng.standard_normal(DIM)), "cat0" if i % 2 == 0 else "cat1")
    return clips, good_votes, bad_votes


def _media(cid: int, vec: np.ndarray, category: str) -> dict:
    return {
        "id": cid,
        "media_type": "image",
        "embedder": "siglip",
        "category": category,
        "embeddings": {"siglip": vec},
    }


def _calibrate_and_blend(clips, good_votes, bad_votes):
    """Train one starved step and return everything both sides of the mirror need."""
    sim_ids = sorted(cid for cid in clips if cid not in good_votes and cid not in bad_votes)
    X_all = np.array([clips[cid]["embeddings"]["siglip"] for cid in sim_ids], dtype=np.float32)

    step, threshold, n_labels, _timings, details = _train_and_calibrate(
        "app",
        good_votes,
        bad_votes,
        clips,
        "cat0",
        region_voting=False,
        input_dim=DIM,
        inclusion=0,
        calibrate_count=2,
        calibration_fraction=0.5,
    )
    ctx = BlendContext(n_labels=n_labels, n_good=len(good_votes), n_bad=len(bad_votes))
    blended, sim_scores, _ids, _hay, provenance, cut = _safe_threshold_for_step(
        threshold,
        step,
        details,
        False,
        None,
        X_all,
        ctx,
        sim_ids,
        0,
        schedule=SCHEDULE,
    )
    return threshold, details, ctx, blended, sim_scores, provenance, cut


class TestFallbackBlendParity:
    def test_starved_step_blends_the_app_sentinel(self):
        rng = np.random.default_rng(2936)
        clips, good_votes, bad_votes = _starved_votes(rng)
        threshold, details, ctx, blended, sim_scores, provenance, cut = _calibrate_and_blend(
            clips, good_votes, bad_votes
        )

        # The regime is the one the issue describes: folds bailed to the 0.5
        # sentinel, so there is no anchored cut and the blend is what ships.
        assert details["fold_fallback"] == 0.5
        assert threshold == 0.5
        assert cut is None
        assert provenance == "gmm_blend"

        expected = _fused_threshold(
            threshold,
            calibration_folds(
                [clips[cid]["embeddings"]["siglip"] for cid in [*good_votes, *bad_votes]],
                [1.0] * len(good_votes) + [0.0] * len(bad_votes),
                DIM,
                calibrate_count=2,
                calibration_fraction=0.5,
                hidden_dim=resolve_hidden_dim("mlp", len(good_votes) + len(bad_votes)),
                rng=np.random.RandomState(42),
            ),
            None,
            sim_scores,
            0,
            ctx,
            SCHEDULE,
        )
        assert blended == pytest.approx(expected), (
            "the harness's shipped-threshold arm blended a different x-cal input than the app"
        )

    def test_the_two_sentinels_would_have_differed(self):
        """Guard the guard: at this vote count the substitution is observable."""
        rng = np.random.default_rng(2936)
        clips, good_votes, bad_votes = _starved_votes(rng)
        _threshold, _details, ctx, blended, sim_scores, _prov, _cut = _calibrate_and_blend(clips, good_votes, bad_votes)

        as_sentinel = calculate_safe_threshold(0.5, sim_scores, ctx, schedule=SCHEDULE)
        as_no_good = calculate_safe_threshold(NO_GOOD_THRESHOLD, sim_scores, ctx, schedule=SCHEDULE)
        assert as_sentinel != pytest.approx(as_no_good), (
            "blend weight is 0 here, so this dataset cannot tell the two rules apart"
        )
        assert blended == pytest.approx(as_no_good)

    def test_real_folds_blend_the_computed_cut(self):
        """The substitution is confined to the fallback: real folds blend their cut."""
        rng = np.random.default_rng(2937)
        clips, good_votes, bad_votes = _starved_votes(rng, n_good=5, n_bad=5)
        threshold, details, _ctx, _blended, _scores, _prov, _cut = _calibrate_and_blend(clips, good_votes, bad_votes)

        assert details["fold_fallback"] is None
        assert _blend_xcal_input(threshold, details) == threshold


class TestBlendXcalInput:
    @pytest.mark.parametrize("sentinel", [0.5, NO_GOOD_THRESHOLD])
    def test_any_sentinel_becomes_no_good(self, sentinel):
        assert _blend_xcal_input(sentinel, {"fold_fallback": sentinel}) == NO_GOOD_THRESHOLD

    def test_real_calibration_passes_through(self):
        assert _blend_xcal_input(0.31, {"fold_fallback": None}) == 0.31

    def test_trainers_without_fold_provenance_pass_through(self):
        """The SVM arms carry no ``fold_fallback``; they blend their own value."""
        assert _blend_xcal_input(0.5, {}) == 0.5
