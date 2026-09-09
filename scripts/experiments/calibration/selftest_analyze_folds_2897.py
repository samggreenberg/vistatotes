"""Self-test for :mod:`analyze_folds_2897` on fabricated cells (no cluster data).

Plants a known answer and checks the analyzer recovers exactly it:

* a **saturating** benefit curve - regret falls from K=1 through K=4 and is flat
  past it - so the knee must come back as 4, not as the best-scoring K;
* a **linear** cost curve with the best-scoring K priced out: K=16 has the
  lowest regret of all but costs 8x production's calibration, over the
  ``COST_CEILING_X`` ceiling, so it must be excluded from the recommendation
  while still showing up as ``best_k_ignoring_cost``;
* a **null** binary arm, where no K beats K=2 by more than the margin, so the
  verdict must be "keep production" rather than the argmin of noise;
* the arithmetic comparison of the two decomposition terms, under the name that
  claims only arithmetic - #3116 established that they cannot carry H4's
  mechanism claim, because the reference defining the split is estimated from a
  calibration set that grows with K;
* H4's **direct** instrument, ``sd(threshold)`` across seeds, planted with a
  closed-form ``1/sqrt(K)`` shape so the averaging is checked exactly;
* the guard flag that records the reference moving with the arm;
* nothing leaking into the shallow windows, where the effect is planted at zero.

The whole point is that a sign error, an off-by-one in the knee scan, or a
ceiling applied in the wrong direction is caught here rather than after an
overnight GRID run.

Usage::

    python selftest_analyze_folds_2897.py     # exits non-zero on failure
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

CATEGORIES = ["cat_a", "cat_b", "cat_c", "cat_d"]
SEEDS = [0, 1]
MAX_T = 300
FOLD_COUNTS = [1, 2, 3, 4, 8, 16]

#: Regret *level* per fold count in the deep windows, region arm.  Falls through
#: K=4 then flattens, so the knee is 4 even though 16 scores a shade lower.
REGION_REGRET = {1: 0.140, 2: 0.120, 3: 0.106, 4: 0.100, 8: 0.0995, 16: 0.0990}
#: Binary arm: no real effect anywhere.  The verdict must keep production.
BINARY_REGRET = dict.fromkeys(FOLD_COUNTS, 0.120)

#: The deep regime starts one vote above the second checkpoint, so every window
#: the verdict reads (le_100 = (50, 100] upward) carries the planted effect and
#: every window below it carries none.
DEEP_FROM_VOTES = 51

#: Planted calibration wall clock, linear in K - the shape the real thing must
#: have, since the folds are independent repeats rather than a partition.  So
#: the cost ratio vs production is exactly K/2, and the 4x ceiling admits K up
#: to 8 while pricing out 16.
SECONDS_PER_FOLD = 0.4
OVERHEAD_S = 0.0
#: The non-calibration part of a step, so ``cal_share`` is a real fraction.
OTHER_STEP_S = 0.3

#: Planted seed-to-seed spread of the shipped threshold at K=1, shrinking as
#: ``1/sqrt(K)`` - the textbook shape for an estimator averaging K independent
#: draws.  With two seeds placed symmetrically about 0.5 the sample sd at fold
#: count K is exactly ``THRESHOLD_SPREAD / sqrt(2K)``, so
#: :func:`analyze_folds_2897.threshold_dispersion` has a closed-form answer to
#: be checked against rather than a "looks about right" bound.
#:
#: This is the instrument #3116 asks for: the decomposition terms cannot say
#: whether the cut got less noisy, and this can.  Planting it here means a
#: regression in the averaging (pooling across steps instead of across seeds,
#: say, which would pick up the trajectory's own drift) fails the selftest.
THRESHOLD_SPREAD = 0.08

#: Planted #3115 effect, region arm, deep windows, K>=3 - in *regret* units, as
#: a decomposition rather than one lump, so the selftest checks that the
#: analyzer's legs telescope rather than merely that a total came out.
#:
#: ``combine`` is the pooling-vs-averaging leg (``tmean`` vs the pooled control)
#: and ``space`` is the score-vs-quantile leg (``qmean`` vs ``tmean``); their sum
#: is the total the issue literally asks for.  Signs are chosen so the two legs
#: point the *same* way, because a selftest whose legs cancel cannot tell a
#: correct decomposition from one that dropped a term.
COMBINE_GAIN = -0.010
SPACE_GAIN = -0.006
#: The robust variants, which must be **identically zero** below three folds and
#: only then separate.  The shipped-path one is planted with the opposite sign so
#: a sign error in one contrast cannot be hidden by the other agreeing.
MEDIAN_GAIN_T = -0.004
MEDIAN_GAIN_Q = -0.002
MEDIAN_GAIN_ANCHORED = 0.003

#: Fraction of deep K>=3 steps on the region arm where one fold is too degenerate
#: to contribute a cut.  The contamination channel's exposure term: the analyzer
#: must report a rate near this, and near zero below K=3, or a reader cannot tell
#: a real robustness effect from one this grid never had the hazard to produce.
DROPPED_FOLD_RATE = 0.25


def _planted_threshold(seed: int, k: int) -> float:
    """The shipped threshold for one (seed, fold count) - see :data:`THRESHOLD_SPREAD`."""
    return 0.5 + (seed - 0.5) * THRESHOLD_SPREAD / np.sqrt(k)


def _planted_sd(k: int) -> float:
    """Sample sd of :func:`_planted_threshold` over :data:`SEEDS`, in closed form."""
    return THRESHOLD_SPREAD / np.sqrt(2.0 * k)


def _fabricate(results: Path, rng: np.random.Generator, counts: list[int] | None = None, region=None) -> None:
    counts = counts or FOLD_COUNTS
    region = region or REGION_REGRET
    cells = results / "cells"
    cells.mkdir(parents=True, exist_ok=True)
    idx = 0
    # The region arm must be a cell that *actually* region-votes: boxes from the
    # dataset AND a patch grid from the embedder.  `visual_genome_m x siglip`
    # has only the first and silently runs whole-image, which is the
    # mis-specification behind #2877/#2905 and #2897's own errata - planting it
    # as the region arm here would have made this selftest assert the bug.
    for dataset, embedder, style, levels in (
        ("visual_genome_m", "dinov3_patch", "max_patch", region),
        ("caltech101_m", "siglip", "whole_image", dict.fromkeys(counts, 0.120)),
    ):
        for cat in CATEGORIES:
            for seed in SEEDS:
                rows = []
                for t in range(2, MAX_T + 1):
                    n_votes = t
                    deep = n_votes >= DEEP_FROM_VOTES
                    base = {
                        "seed": seed,
                        "dataset": dataset,
                        "category": cat,
                        "strategy": "autopilot",
                        "trainer": "app",
                        "head": "linear",
                        "style": style,
                        "prevalence_arm": "natural",
                        "realized_prevalence": 0.05,
                        "t": t,
                        "n_good": n_votes // 2,
                        "n_bad": n_votes - n_votes // 2,
                        "phase": "hard",
                        "app_trained": 1,
                        "pool_variant": "max",
                        "oracle_threshold": 0.5,
                        "degenerate": 0,
                        "auroc": 0.9,
                        "average_precision": 0.5,
                        "embedder": embedder,
                        "train_seconds": OTHER_STEP_S / 3,
                        "pool_score_seconds": OTHER_STEP_S / 3,
                        "test_score_seconds": OTHER_STEP_S / 3,
                    }
                    # The base row (gmm_variant "") - the analyzer must ignore it.
                    rows.append({**base, "gmm_variant": "", "threshold": 0.5, "cost": 0.99, "regret": 0.99})
                    for k in counts:
                        # Shallow windows: every K identical, so a leak is visible.
                        regret = (levels[k] if deep else 0.130) + 0.001 * rng.standard_normal()
                        # More folds cut the sampling noise in the cut (rule
                        # inefficiency); the calibration->test shift is a
                        # property of the split, which K cannot move.
                        rule = (regret - 0.05) if deep else 0.08
                        # #3115's arms.  ``real`` is whether this cell carries a
                        # planted combine effect at all: only the region arm, only
                        # deep, only at K>=3 for the median variants.  Everything
                        # else must come back at exactly zero, which is what makes
                        # a leak visible instead of plausible.
                        # Only the region arm, only deep - and only at K>=2 for
                        # the combine legs, because at one fold averaging *is*
                        # pooling and the arms are the same rule.  Planting an
                        # effect at K=1 would fabricate a difference the real
                        # harness cannot produce, and would break the acceptance
                        # check that exists to catch exactly that.
                        real = deep and embedder == "dinov3_patch" and k >= 2
                        split = k >= 3
                        offsets = {
                            "xcal": 0.0,
                            "blend": 0.0,
                            "anchored": 0.0,
                            "tmean": COMBINE_GAIN if real else 0.0,
                            "qmean": (COMBINE_GAIN + SPACE_GAIN) if real else 0.0,
                            "tmedian": (COMBINE_GAIN + (MEDIAN_GAIN_T if split else 0.0)) if real else 0.0,
                            "qmedian": (
                                (COMBINE_GAIN + SPACE_GAIN + (MEDIAN_GAIN_Q if split else 0.0)) if real else 0.0
                            ),
                            "anchored_qmedian": (MEDIAN_GAIN_ANCHORED if (real and split) else 0.0),
                        }
                        # A degenerate fold is dropped by the combining arms and
                        # silently pooled by the control, so only the former can
                        # report one.  Deterministic in (cat, seed, t) so the run
                        # is reproducible and the rate is exactly as planted.
                        dropped = 1 if (real and split and (t % 4 == 0)) else 0
                        for arm, offset in offsets.items():
                            combining = arm not in ("xcal", "blend")
                            rows.append(
                                {
                                    **base,
                                    "gmm_variant": f"folds_k{k}_{arm}",
                                    # The threshold carries the same offset as
                                    # the regret, so the fixture is internally
                                    # coherent: an arm whose cost differs must
                                    # have cut somewhere else.  A fixture where
                                    # `moved_rate` is 0 while `d_regret` is not
                                    # would let a real such contradiction pass.
                                    # Both identities fall out: the medians match
                                    # their means below K=3, and every arm
                                    # matches the pooled cut at K=1.
                                    "threshold": _planted_threshold(seed, k) + offset,
                                    "cost": 0.2 + regret + offset,
                                    "regret": regret + offset + (0.0005 * rng.standard_normal() if combining else 0.0),
                                    "fpr": 0.05,
                                    "fnr": 0.12,
                                    "rule_inefficiency": rule,
                                    "calibration_shift": 0.05,
                                    # #3116's cross-fitted twin.  Planted flat,
                                    # like its naive sibling: this checks the
                                    # column reaches the verdict, and asserting
                                    # a *shape* here would be planting a
                                    # mechanism nobody has measured.
                                    "calibration_shift_honest": 0.02,
                                    "fold_count": k,
                                    "fold_seconds": OVERHEAD_S + SECONDS_PER_FOLD * k,
                                    "n_cal_scores": 20 * k,
                                    # NaN on the pooled arm and the blend by
                                    # design: neither ever reads a per-fold cut,
                                    # so neither has a fold it could drop.
                                    "n_folds_used": (k - dropped) if combining else float("nan"),
                                }
                            )
                pd.DataFrame(rows).to_csv(cells / f"task_{idx:04d}.csv", index=False)
                idx += 1


#: The A/B arm lives at K=8 and - because it also collected different votes -
#: lands a bigger win than the screen credited K=8 with.  The analyzer must
#: report that gap rather than quietly averaging the two together.
AB_K = 8
AB_REGION_REGRET = {2: 0.120, AB_K: 0.090}


def main() -> int:
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        results = Path(tmp) / "results"
        _fabricate(results, rng)
        # A full run that *lives* at K=8, laid out the way launch_folds_2897_ab.sh
        # writes it: <arm dir>/results/cells.
        ab_dir = Path(tmp) / "ab-k8"
        _fabricate(ab_dir / "results", rng, counts=[2, AB_K], region=AB_REGION_REGRET)

        os.environ["CALIB_EXP"] = tmp
        os.environ["CALIB_RESULTS"] = str(results)
        # `common.setup_env()` puts VTS_REPO on sys.path, and the analyzer needs
        # `experiment_config`'s region-voting predicate, which imports vtscore.
        # On the cluster the launcher sets this; point it at this checkout so the
        # selftest exercises the same import path rather than a special case.
        os.environ.setdefault("VTS_REPO", str(Path(__file__).resolve().parents[3]))
        sys.path.insert(0, str(Path(__file__).parent))

        import analyze_folds_2897 as az  # noqa: PLC0415

        rc = az.main([str(ab_dir)])
        assert rc == 0, f"analyze_folds_2897 returned {rc}"

        verd = json.loads((results / "summary.json").read_text())
        # The shipped rule (#3116), not the retired blend and not the raw cut.
        assert verd["arm_read"] == "anchored", verd
        region = verd["by_voting"]["region"]
        binary = verd["by_voting"]["binary"]

        # Region: a real, significant win from K=3 up, with K=16 priced out.
        assert region["h1_any_k_beats_baseline"] is True, region
        assert region["h1_ks_beating_baseline"] == [3, 4, 8, 16], region
        assert region["h2_ks_also_affordable"] == [3, 4, 8], region
        assert region["h3_recommended_k"] == 3, region
        assert region["h3_kept_production"] is False, region
        assert region["best_k_ignoring_cost"] == 16, region
        assert abs(region["cost_x_at_recommended"] - 1.5) < 1e-6, region
        assert region["h4_d_rule_below_d_shift"] is True, region
        assert abs(region["d_shift_at_best"]) < 1e-9, region
        # The honest twin reaches the verdict and is flat, as planted.
        assert abs(region["d_shift_honest_at_best"]) < 1e-9, region

        # --- #3116: the guard, and H4's direct instrument. ---
        # `n_cal_scores` is planted as 20*K, so the decomposition's reference
        # provably moves with the arm and the analyzer must say so.
        assert region["h4_reference_moves_with_k"] is True, region
        assert binary["h4_reference_moves_with_k"] is True, binary
        # sd(threshold) across seeds must come back at the planted closed form
        # for every K, not merely fall monotonically: an averaging bug that
        # pooled across steps rather than seeds would still be monotone here.
        sd_by_k = region["h4_sd_threshold_by_k"]
        assert sd_by_k, region
        for k_str, sd in sd_by_k.items():
            expected = _planted_sd(int(k_str))
            assert abs(sd - expected) < 1e-9, (k_str, sd, expected)
        # Best K is 16, baseline 2, and the spread shrinks as 1/sqrt(K).
        assert region["h4_sd_threshold_falls_at_best_k"] is True, region

        # Binary: nothing beats the margin, so production stands.
        assert binary["h1_any_k_beats_baseline"] is False, binary
        assert binary["h3_recommended_k"] == az.BASELINE_K, binary
        assert binary["h3_kept_production"] is True, binary

        # The knee scan must stop at saturation (4), not chase the argmin (8).
        knee = pd.read_csv(results / "agg" / "folds_knee.csv")
        deep_region = knee[
            (knee["voting"] == "region") & (knee["arm"] == "blend") & (knee["window"].isin(["le_200", "le_300"]))
        ]
        assert not deep_region.empty
        assert set(deep_region["knee_k"]) == {4}, deep_region
        assert set(deep_region["best_k"]) == {16}, deep_region

        # No planted effect below the deep threshold; the shallow windows must
        # report ~0 for every K, or the banding is not doing its job.
        paired = pd.read_csv(results / "agg" / "folds_paired_vs_k2.csv")
        shallow = paired[(paired["voting"] == "region") & (paired["window"].isin(["le_20", "le_50"]))]
        assert not shallow.empty and shallow["d_regret"].abs().max() < az.MARGIN, shallow

        # Cost side: linear in K, and reported as a share of the whole step.
        levels = pd.read_csv(results / "agg" / "folds_levels.csv")
        deep = levels[(levels["voting"] == "region") & (levels["arm"] == "blend") & (levels["window"] == "le_300")]
        secs = dict(zip(deep["k"], deep["fold_seconds"], strict=True))
        assert abs(secs[8] / secs[2] - 4.0) < 1e-6, secs
        assert abs(secs[16] / secs[2] - 8.0) < 1e-6, secs
        share = dict(zip(deep["k"], deep["cal_share"], strict=True))
        assert abs(share[2] - (0.8 / (0.8 + OTHER_STEP_S))) < 1e-6, share

        # A/B leg: the live arm's own delta vs the delta the screen credited the
        # same K with.  Planted to disagree, so a silently-passing "agrees"
        # would mean the comparison is not actually being made.
        ab = pd.read_csv(results / "agg" / "folds_ab_check.csv")
        region_ab = ab[(ab["voting"] == "region") & (ab["k"] == AB_K)]
        assert len(region_ab) == 1, ab
        row = region_ab.iloc[0]
        assert abs(row["live_d_regret"] - (-0.030)) < 0.002, row
        assert abs(row["screen_d_regret"] - (-0.0205)) < 0.002, row
        assert row["live_minus_screen"] < -az.MARGIN, row
        assert not row["screen_agrees"], row
        # The binary arm is planted null in both runs, so there the screen agrees.
        binary_ab = ab[(ab["voting"] == "binary") & (ab["k"] == AB_K)]
        assert len(binary_ab) == 1 and bool(binary_ab.iloc[0]["screen_agrees"]), ab

        # --- #3115: the combine axis, read at fixed K across rules. ---
        c = verd["combine_3115"]
        checks = c["control_checks"]
        # The acceptance check that does not depend on anything the run measures:
        # averaging one number is the identity, so at K=1 the score-space arm
        # reproduces the pooled cut exactly.  Planted true; a harness that
        # mis-slices the fold prefix breaks it.
        assert checks["k1_score_space_is_pooled"] is True, checks
        assert checks["k1_n_steps"] > 0, checks
        # Mean and median of at most two numbers coincide - the reason this
        # question has never been askable at production's calibrate_count=2.
        assert checks["median_collapses_below_k3"] is True, checks

        cregion = c["by_voting"]["region"]["contrasts"]
        cbinary = c["by_voting"]["binary"]["contrasts"]

        # The two legs come back at what was planted, and they telescope into the
        # total.  Checking the sum is what distinguishes a correct decomposition
        # from one that dropped or double-counted a term - a lone total cannot.
        assert abs(cregion["combine"]["d_regret"] - COMBINE_GAIN) < 5e-4, cregion
        assert abs(cregion["space"]["d_regret"] - SPACE_GAIN) < 5e-4, cregion
        assert abs(cregion["total"]["d_regret"] - (COMBINE_GAIN + SPACE_GAIN)) < 5e-4, cregion
        assert (
            abs(cregion["total"]["d_regret"] - (cregion["combine"]["d_regret"] + cregion["space"]["d_regret"])) < 5e-4
        ), cregion
        for name in ("combine", "space", "total"):
            assert cregion[name]["resolved"] is True, (name, cregion[name])
            assert cregion[name]["favours"] == "arm", (name, cregion[name])

        # The robust variants, planted with opposite signs on the two paths so a
        # sign error cannot be masked by the other contrast agreeing.
        assert abs(cregion["contamination_t"]["d_regret"] - MEDIAN_GAIN_T) < 5e-4, cregion
        assert abs(cregion["contamination_q"]["d_regret"] - MEDIAN_GAIN_Q) < 5e-4, cregion
        assert abs(cregion["shipped_contamination"]["d_regret"] - MEDIAN_GAIN_ANCHORED) < 5e-4, cregion
        assert cregion["shipped_contamination"]["favours"] == "ref", cregion

        # The headline, in the issue's own terms: pooling loses on this arm.
        assert c["by_voting"]["region"]["pooling_is_wrong"] is True, c["by_voting"]["region"]
        assert c["by_voting"]["region"]["averaging_is_wrong"] is False, c["by_voting"]["region"]

        # Binary is planted null: every contrast must land inside the margin, and
        # the verdict must not claim a side.  A selftest with only a positive arm
        # cannot catch a analyzer that reports an effect unconditionally.
        for name, row in cbinary.items():
            assert row["above_margin"] is False, (name, row)
        assert c["by_voting"]["binary"]["pooling_is_wrong"] is False, cbinary
        assert c["by_voting"]["binary"]["averaging_is_wrong"] is False, cbinary

        # Below K=3 the median contrasts are identically zero, and the table says
        # so rather than reporting noise around zero.
        contrasts = pd.read_csv(results / "agg" / "combine_contrasts.csv")
        collapsed = contrasts[contrasts["collapsed_by_construction"]]
        assert not collapsed.empty, contrasts["contrast"].unique()
        assert collapsed["moved_rate"].abs().max() == 0.0, collapsed

        # The contamination channel's exposure: one fold dropped on a quarter of
        # deep K>=3 steps, none below.  Without this a robustness result cannot
        # be told from a grid that never presented the hazard.
        degen = pd.read_csv(results / "agg" / "combine_degenerate.csv")
        deep_hi = degen[
            (degen["voting"] == "region") & (degen["k"] >= 3) & (degen["window"].isin(["le_200", "le_300"]))
        ]
        assert not deep_hi.empty, degen
        assert abs(deep_hi["any_dropped_rate"].mean() - DROPPED_FOLD_RATE) < 0.02, deep_hi
        low = degen[(degen["voting"] == "region") & (degen["k"] < 3)]
        assert not low.empty and low["any_dropped_rate"].max() == 0.0, low

        # The two axes must not contaminate each other: the fold-COUNT tables are
        # computed over one rule per arm, so #3115's challengers must be absent
        # from them entirely.
        assert set(levels["arm"]) <= set(az.COUNT_AXIS_ARMS), sorted(set(levels["arm"]))
        assert set(paired["arm"]) <= set(az.COUNT_AXIS_ARMS), sorted(set(paired["arm"]))

    print("selftest_analyze_folds_2897: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
