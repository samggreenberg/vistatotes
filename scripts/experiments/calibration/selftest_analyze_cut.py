"""Self-test for :mod:`analyze_cut` on fabricated cells (no cluster data needed).

The cut analyzer is read exactly once, on data that took an hour of cluster time
to produce, and it has to be right about a *sign*: whether a candidate rule beats
the incumbent, and which term in the derivation dominates. Both are easy to get
backwards. This plants known answers in a synthetic ``results`` tree and checks
they come back out:

* a candidate rule made cheaper inside the ramp window is found, with the right
  sign, size, and improved-cell fraction, and does not leak into the other window;
* the pairing unit is the **cell**, not the step (otherwise 29 autocorrelated
  steps per cell would inflate every p-value's confidence);
* the decomposition telescopes and names the term that was actually planted as
  dominant;
* a step whose oracle links are missing is **dropped from every term**, not
  averaged around: the terms and ``total`` must cover the same steps or the chain
  stops summing to the total, silently, in exactly the terms this study reads;
* a rule whose *blended* cost wins while its *raw cut* does not is reported as
  such, since that is the trap the plan pre-registers;
* the ship test is decided against the run's **own base row**, not against the
  ``pooled_mid`` reconstruction of it — the two are planted apart here, exactly
  as #2846 found them in the field, and the ship gate must follow the base row;
* fallback reasons are windowed, so an out-of-ramp fallback cannot be read as a
  ramp-window one;
* the #2881 tail sweep is classified, its cost-vs-alpha curve is recovered, and a
  swept level that beats production **cannot** become the ship candidate — the
  sweep varies a free parameter, so letting all seven levels at a 5 % bar would
  buy a winner out of noise.

Usage::

    python selftest_analyze_cut.py     # exits non-zero on failure
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

#: How much cheaper the planted winner is inside the ramp window.
RAMP_EFFECT = -0.04
#: Planted decomposition, in threshold units: the prior/loss term dominates.
PLANTED_TERMS = {"prior_loss": 0.10, "identification": 0.02, "misspecification": 0.01, "transfer": 0.005}
#: The same chain in cost units, anchored to a noise-free oracle cost, so the
#: analyzer's "which term dominates" verdict has a known right answer.
ORACLE_COST = 0.20
PLANTED_COST_TERMS = {"prior_loss": 0.10, "identification": 0.02, "misspecification": 0.01, "transfer": 0.005}
#: Raw-cut cost per chain variant, built backwards from the oracle.
_CHAIN_RAW_COST = {
    "pooled_sim_oracle": ORACLE_COST + PLANTED_COST_TERMS["transfer"],
    "pooled_supervised": ORACLE_COST + PLANTED_COST_TERMS["transfer"] + PLANTED_COST_TERMS["misspecification"],
}
_CHAIN_RAW_COST["pooled_priorfree"] = _CHAIN_RAW_COST["pooled_supervised"] + PLANTED_COST_TERMS["identification"]
_CHAIN_RAW_COST["pooled_cross"] = _CHAIN_RAW_COST["pooled_priorfree"] + PLANTED_COST_TERMS["prior_loss"]
CATEGORIES = ["cat_a", "cat_b", "cat_c"]
SEEDS = [0, 1, 2, 3]
ARMS = [("dinov3_patch", "max_patch"), ("siglip", "whole_image")]

WINNER = "pooled_priorfree"
INCUMBENT = "pooled_mid"

#: #2846's field condition, planted: on half the cells production no longer takes
#: the blended-GMM path that ``pooled_mid`` reconstructs, and the path it takes
#: instead is *cheaper* than the planted winner.  So the midpoint contrast says
#: ship and the base-row contrast says do not, and the analyzer has to follow the
#: base row.  Keyed on seed parity, which also gives the fidelity check the exact
#: 1:1 provenance partition the re-measure found.
ANCHORED_SEEDS = {2, 3}
PROD_EDGE = -0.10
#: A fallback planted **outside** the ramp window: the windowed reasons table
#: must not report it inside one.
FALLBACK_VARIANT = "pooled_gumbel_priorfree"
FALLBACK_REASON = "modes_swapped"
FALLBACK_STEPS = [t for t in range(2, 31) if t >= 21]
#: Steps where the oracle links are **missing**, planted inside the ramp window.
#: The field condition: oracle variants do not fall back, so a step whose oracle
#: cut is not finite emits no row at all and the analyzer's pivot fills NaN.
INCOMPLETE_STEPS = {8, 9}
#: The links that go absent on those steps - both oracle variants, which between
#: them feed three of the four terms.
INCOMPLETE_LINKS = ("pooled_supervised", "pooled_sim_oracle")
#: How far the *surviving* links are displaced on those steps.  Applied to both
#: ends of the ``prior_loss`` link so that term is unchanged and only ``total``
#: moves - which is exactly the imbalance a per-column ``mean()`` cannot see, and
#: what makes averaging around the hole (rather than dropping the step) show up
#: as a decomposition that no longer telescopes.
INCOMPLETE_DISPLACEMENT = 0.05

#: A rule present in the cells that the analyzer's ``SHIPPABLE`` allowlist does
#: not know about.  Planted deliberately: the allowlist gates the ship decision,
#: so an unlisted rule is omitted from the verdict while still showing up in the
#: window means, and that omission has to be loud.
UNKNOWN_VARIANT = "pooled_not_a_known_rule"

#: #2881's tail sweep, planted as a **knife-edge** curve whose minimum sits at a
#: level that is not the pre-registered one — and which is cheaper than both
#: production and the planted winner.  Two things have to come out of this:
#: the curve's argmin is found (0.30) and reported as *not* flat, and that argmin
#: is nonetheless **not** the ship candidate, because a sweep over one free
#: parameter does not get seven shots at the 5 % bar (``analyze_cut.SWEEP_ONLY``).
#: Per-cell ramp effect, relative to the incumbent midpoint.
TAIL_EFFECTS: dict[float, float] = {
    0.04: 0.01,
    0.08: 0.00,
    0.11: -0.01,
    0.158: -0.02,
    0.22: -0.03,
    0.30: -0.08,
    0.40: -0.01,
}
TAIL_BEST_ALPHA = 0.30
TAIL_PREREGISTERED_ALPHA = 0.158


def _tail_variant(alpha: float) -> str:
    return f"pooled_tail_a{round(alpha * 1000):03d}"


_TAIL_EFFECT_BY_VARIANT: dict[str, float] = {_tail_variant(a): e for a, e in TAIL_EFFECTS.items()}


def _ident(cat: str, seed: int, t: int, embedder: str, style: str) -> dict:
    n_votes = t
    return {
        "seed": seed,
        "dataset": "visual_genome_m",
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
        "embedder": embedder,
    }


def _fabricate(root: Path, rng: np.random.Generator) -> None:
    from vtscore.eval.voting_columns import CALIBRATION_COLUMNS, CUT_DIAGNOSTIC_COLUMNS
    from vtscore.eval.arms_safe_gmm import _SAFE_GMM_VARIANTS

    cells = root / "cells"
    cells.mkdir(parents=True, exist_ok=True)
    variants = [name for name, _f, _r in _SAFE_GMM_VARIANTS]
    # A rule the analyzer has never heard of, standing in for the next one added
    # to `_SAFE_GMM_VARIANTS` without a matching `SHIPPABLE` entry (#2881's
    # `tail_alpha` is the one actually coming).  It must not be able to slip
    # through unnoticed.
    variants.append(UNKNOWN_VARIANT)

    idx = 0
    for embedder, style in ARMS:
        for cat in CATEGORIES:
            for seed in SEEDS:
                rows, diag = [], []
                for t in range(2, 31):
                    ident = _ident(cat, seed, t, embedder, style)
                    in_ramp = 6 <= t <= 20
                    base_cost = 0.30 + 0.002 * rng.standard_normal()
                    oracle_cost = ORACLE_COST
                    oracle_threshold = 0.50
                    # Shared jitter: the realised offset and its closed-form
                    # prediction move together, so the identity stays exact while
                    # the correlation the analyzer reports is still computable.
                    jitter = 0.01 * rng.standard_normal()

                    # The base (production) row, and the variant that reconstructs
                    # it - which on the anchored cells it no longer does.
                    anchored = seed in ANCHORED_SEEDS
                    threshold = 0.55
                    incomplete = t in INCOMPLETE_STEPS
                    for variant in ["", *variants]:
                        if incomplete and variant in INCOMPLETE_LINKS:
                            continue
                        is_base = variant == ""
                        effect = RAMP_EFFECT if (variant == WINNER and in_ramp) else 0.0
                        if in_ramp and variant in _TAIL_EFFECT_BY_VARIANT:
                            effect = _TAIL_EFFECT_BY_VARIANT[variant]
                        if is_base and anchored:
                            effect = PROD_EDGE
                        # The blended and raw columns move together here except
                        # for the decoy, which wins only after blending.
                        decoy = variant == "pooled_gumbel_cross" and in_ramp
                        raw_cost = _CHAIN_RAW_COST.get(variant, base_cost) + (0.02 if decoy else 0.0)
                        if incomplete and variant in ("pooled_cross", "pooled_priorfree"):
                            raw_cost += INCOMPLETE_DISPLACEMENT
                        fell = variant == FALLBACK_VARIANT and t in FALLBACK_STEPS
                        row = dict(ident)
                        row.update(
                            pool_variant="max",
                            gmm_variant=variant,
                            threshold=threshold - (0.02 if (is_base and anchored) else 0.0),
                            threshold_provenance="fold_anchored[k0.3]" if anchored else "gmm_blend",
                            degenerate=0,
                            threshold_percentile=0.9,
                            xcal_threshold=0.52,
                            gmm_cut=oracle_threshold + (0.0 if variant == WINNER else 0.05),
                            blend_weight=0.5,
                            cut_fallback=1 if fell else 0,
                            cut_fail_reason=FALLBACK_REASON if fell else "",
                            raw_cut_cost=raw_cost,
                            raw_cut_fpr=0.1,
                            raw_cut_fnr=0.2,
                            # Half the winner's edge: enough to win on the blended
                            # column, small enough that no argmin is a tie (which
                            # a tie-break would then have to resolve arbitrarily).
                            cost=base_cost + effect + (RAMP_EFFECT / 2 if decoy else 0.0),
                            fpr=0.1,
                            fnr=0.2,
                            auroc=0.9,
                            average_precision=0.5,
                            oracle_threshold=oracle_threshold,
                            oracle_cost=oracle_cost,
                            oracle_fpr=0.05,
                            oracle_fnr=0.1,
                            regret=base_cost - oracle_cost,
                            cal_oracle_threshold=0.5,
                            cal_oracle_cost=oracle_cost,
                            rule_inefficiency=0.0,
                            calibration_shift=0.0,
                            n_pool_rows=100.0,
                            train_seconds=1.0,
                            xcal_seconds=1.0,
                            pool_score_seconds=1.0,
                            test_score_seconds=1.0,
                            backend="torch",
                            device="cpu",
                            elapsed_seconds=1.0,
                            seed_mode="text",
                            seed_query="a bus",
                        )
                        rows.append(row)

                    # The decomposition frame: cuts placed so each successive
                    # gap is exactly the planted term.
                    for geometry in ("pooled", "image"):
                        tau_test_oracle = oracle_threshold
                        tau_sim_oracle = tau_test_oracle + PLANTED_TERMS["transfer"]
                        tau_supervised = tau_sim_oracle + PLANTED_TERMS["misspecification"]
                        tau_priorfree = tau_supervised + PLANTED_TERMS["identification"]
                        tau_cross = tau_priorfree + PLANTED_TERMS["prior_loss"] + jitter
                        d = dict(ident)
                        d.update(
                            geometry=geometry,
                            sim_n=2000.0,
                            sim_prevalence=0.05,
                            fallback_median=0.4,
                            gmm_ok=1,
                            w_lo=0.95,
                            mu_lo=0.30,
                            var_lo=0.02,
                            w_hi=0.05,
                            mu_hi=0.80,
                            var_hi=0.01,
                            gmm_loglik=1.0,
                            gmm_logit_loglik=1.0,
                            # Equal-variance closed form for this fit, so the
                            # identity check has something exact to recover.
                            pred_offset_equal_var=PLANTED_TERMS["prior_loss"] + jitter,
                            evt_ok=1,
                            evt_fit_fail="ok",
                            evt_gumbel_is_low=1,
                            evt_w_gumbel=0.95,
                            evt_loc=-1.0,
                            evt_scale=0.5,
                            evt_mu=1.5,
                            evt_var=0.5,
                            evt_loglik=1.1 if geometry == "pooled" else 1.0,
                            evt_loglik_gain=0.1 if geometry == "pooled" else 0.0,
                            s_mu_neg=0.30,
                            s_var_neg=0.02,
                            s_mu_pos=0.80,
                            s_var_pos=0.01,
                            s_prevalence=0.05,
                            tau_mid=tau_cross - PLANTED_TERMS["prior_loss"] - jitter,
                            tau_cross=tau_cross,
                            tau_priorfree=tau_priorfree,
                            tau_rate=tau_priorfree,
                            tau_gumbel_cross=tau_cross,
                            tau_gumbel_priorfree=tau_priorfree,
                            tau_gumbel_rate=tau_priorfree,
                            tau_gumbel_any_cross=tau_cross,
                            tau_gumbel_any_priorfree=tau_priorfree,
                            tau_gumbel_any_rate=tau_priorfree,
                            tau_supervised=np.nan if incomplete else tau_supervised,
                            tau_sim_oracle=np.nan if incomplete else tau_sim_oracle,
                            tau_test_oracle=tau_test_oracle - (INCOMPLETE_DISPLACEMENT if incomplete else 0.0),
                            oracle_lo_sf_gauss=0.02,
                            oracle_lo_sf_evt=0.02,
                        )
                        # The tail sweep's cuts descend with alpha, as the rule's
                        # closed form requires; nothing here reads their values,
                        # but a frame missing them is not the frame a run emits.
                        d.update(
                            {f"tau_tail_a{round(a * 1000):03d}": tau_sim_oracle + 0.10 - 0.2 * a for a in TAIL_EFFECTS}
                        )
                        diag.append(d)

                pd.DataFrame(
                    rows, columns=pd.Index([*CALIBRATION_COLUMNS, "embedder", "seed_mode", "seed_query"])
                ).to_csv(cells / f"task_{idx:04d}.csv", index=False)
                pd.DataFrame(diag, columns=pd.Index([*CUT_DIAGNOSTIC_COLUMNS, "embedder"])).to_csv(
                    cells / f"task_{idx:04d}__cutdiag.csv", index=False
                )
                idx += 1


def _check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{(' - ' + detail) if detail else ''}")
    return ok


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "results"
        root.mkdir(parents=True)
        # analyze_cut reads CALIB_RESULTS at import time, via common.setup_env.
        os.environ["CALIB_RESULTS"] = str(root)
        os.environ.setdefault("CALIB_EXP", str(Path(tmp) / "exp"))

        import analyze_cut  # noqa: PLC0415

        _fabricate(root, np.random.default_rng(2836))
        rc = analyze_cut.main()
        if rc != 0:
            print("analyze_cut.main() returned non-zero")
            return 1

        import json  # noqa: PLC0415

        summary = json.loads((root / "summary_cut.json").read_text())
        contrasts = pd.read_csv(root / "agg" / "cut_contrasts.csv")
        decomp = pd.read_csv(root / "agg" / "cut_decomposition.csv")

        ok = True
        prod = contrasts[contrasts["arm"].str.contains("dinov3_patch/max_patch")]
        ramp = prod[(prod["window"] == "ramp_6_20") & (prod["variant"] == WINNER)]
        ok &= _check("winner recovered in the ramp window", len(ramp) == 1)
        if len(ramp) == 1:
            r = ramp.iloc[0]
            ok &= _check(
                "planted effect size",
                abs(r["mean_d_cost"] - RAMP_EFFECT) < 1e-6,
                f"{r['mean_d_cost']:.5f} vs {RAMP_EFFECT}",
            )
            ok &= _check("all cells improved", abs(r["frac_cells_improved"] - 1.0) < 1e-9)
            ok &= _check(
                "pairing unit is the cell",
                int(r["n_cells"]) == len(CATEGORIES) * len(SEEDS),
                f"n_cells={r['n_cells']}",
            )

        other = prod[(prod["window"] == "pure_gmm_2_5") & (prod["variant"] == WINNER)]
        ok &= _check(
            "effect does not leak into the sub-ramp window",
            len(other) == 1 and abs(other.iloc[0]["mean_d_cost"]) < 1e-9,
        )

        dec = summary["decisions"]
        ok &= _check("winner chosen", dec["best_by_cost"]["variant"] == WINNER, str(dec["best_by_cost"]["variant"]))
        ok &= _check("beats the incumbent", bool(dec["beats_midpoint"]))
        ok &= _check("closest to the oracle cut", dec["closest_to_oracle"] == WINNER, str(dec["closest_to_oracle"]))

        # --- The baseline that cannot go stale (#2846) ------------------------
        base = pd.read_csv(root / "agg" / "cut_contrasts_vs_base.csv")
        bprod = base[
            (base["arm"].str.contains("dinov3_patch/max_patch"))
            & (base["window"] == "ramp_6_20")
            & (base["variant"] == WINNER)
        ]
        # Half the cells are anchored (production PROD_EDGE cheaper than the
        # midpoint), half are not, so the winner's edge over production is the
        # average of RAMP_EFFECT and RAMP_EFFECT - PROD_EDGE.
        expected = RAMP_EFFECT - PROD_EDGE * len(ANCHORED_SEEDS) / len(SEEDS)
        ok &= _check("winner paired against the run's own base row", len(bprod) == 1)
        if len(bprod) == 1:
            ok &= _check(
                "base-row delta reflects the shipped threshold, not the midpoint",
                abs(float(bprod.iloc[0]["mean_d_cost"]) - expected) < 1e-6,
                f"{float(bprod.iloc[0]['mean_d_cost']):.5f} vs {expected}",
            )
            ok &= _check(
                "base row records which production path it was",
                "fold_anchored[k0.3]" in str(bprod.iloc[0]["base_provenance"]),
                str(bprod.iloc[0]["base_provenance"]),
            )
        ok &= _check("ship candidate is chosen vs production", dec["ship_candidate"] == WINNER)
        ok &= _check("does not beat what production actually ships", not dec["beats_production"])
        ok &= _check(
            "a stale-baseline win alone does not ship",
            dec["beats_midpoint"] and not dec["ship"],
        )
        ok &= _check("family headroom reported", dec["family_headroom"] is not None)
        ok &= _check("no headroom left on this axis", bool(dec["family_headroom_exhausted"]))

        # A rule the allowlist does not know about is named, not dropped in silence.
        ok &= _check(
            "an unclassified rule is reported in the verdict",
            dec["unclassified_variants"] == [UNKNOWN_VARIANT],
            str(dec["unclassified_variants"]),
        )
        ok &= _check(
            "the image geometry arm and the no-blend control are not flagged",
            not [v for v in dec["unclassified_variants"] if v.startswith("image_") or v == "xcal_only"],
        )

        # A failed fidelity check must say *why*: harness bug or shipped incumbent.
        sanity = summary["sanity"]
        prov = sanity.get("by_provenance", {})
        ok &= _check(
            "fidelity failure is broken down by production path",
            not sanity["ok"]
            and prov.get("gmm_blend", {}).get("n_mismatched") == 0
            and prov.get("fold_anchored[k0.3]", {}).get("n_mismatched", 0) > 0,
            str(prov),
        )

        # Both tail models named, so neither can be read as the other.
        ok &= _check(
            "tail stability is reported per tail model",
            "tail_alpha_stable" not in dec and {"tail_alpha_stable_gauss", "tail_alpha_stable_evt"} <= set(dec),
        )

        # --- #2881's tail sweep -----------------------------------------------
        ok &= _check(
            "the tail sweep is classified, not flagged as unknown",
            not [v for v in dec["unclassified_variants"] if v.startswith("pooled_tail_")],
            str(dec["unclassified_variants"]),
        )
        curve = summary["tail_alpha_curve"]
        ok &= _check(
            "every swept alpha appears on the curve",
            curve.get("n_levels") == len(TAIL_EFFECTS),
            f"{curve.get('n_levels')} vs {len(TAIL_EFFECTS)}",
        )
        ok &= _check(
            "the planted curve minimum is found",
            curve.get("best_alpha") == TAIL_BEST_ALPHA,
            str(curve.get("best_alpha")),
        )
        curve_rows = {round(float(r["alpha"]), 3): r for r in curve.get("curve", [])}
        # Base row is PROD_EDGE cheaper on half the cells, so the contrast against
        # production is the planted effect plus half of that edge.
        expected_prereg = TAIL_EFFECTS[TAIL_PREREGISTERED_ALPHA] - PROD_EDGE * len(ANCHORED_SEEDS) / len(SEEDS)
        got = curve_rows.get(TAIL_PREREGISTERED_ALPHA, {}).get("mean_d_cost")
        ok &= _check(
            "the pre-registered level is paired against production",
            got is not None and abs(float(got) - expected_prereg) < 1e-6,
            f"{got} vs {expected_prereg}",
        )
        ok &= _check(
            "only the pre-registered level is ship-eligible",
            [a for a, r in curve_rows.items() if r["ship_eligible"]] == [TAIL_PREREGISTERED_ALPHA],
            str([a for a, r in curve_rows.items() if r["ship_eligible"]]),
        )
        # The trap: the swept minimum is the cheapest rule in the whole table, and
        # must still not be what the run proposes to ship.
        cheapest = base[
            (base["arm"].str.contains("dinov3_patch/max_patch")) & (base["window"] == "ramp_6_20")
        ].sort_values("mean_d_cost")
        ok &= _check(
            "the swept minimum really is the cheapest rule measured",
            str(cheapest.iloc[0]["variant"]) == _tail_variant(TAIL_BEST_ALPHA),
            str(cheapest.iloc[0]["variant"]),
        )
        ok &= _check(
            "a sweep-only level cannot become the ship candidate",
            dec["ship_candidate"] != _tail_variant(TAIL_BEST_ALPHA),
            str(dec["ship_candidate"]),
        )
        ok &= _check(
            "sweep-only levels are named in the verdict",
            _tail_variant(TAIL_BEST_ALPHA) in dec["sweep_only_variants"]
            and _tail_variant(TAIL_PREREGISTERED_ALPHA) not in dec["sweep_only_variants"],
        )
        ok &= _check(
            "a knife-edge curve is not reported as flat",
            dec["tail_curve_is_flat"] is False,
            str(dec["tail_curve_is_flat"]),
        )

        # Fallback reasons are windowed: the planted fallback sits outside the ramp.
        reasons = pd.read_csv(root / "agg" / "cut_fallback_reasons.csv")
        planted = reasons[
            (reasons["arm"].str.contains("dinov3_patch/max_patch")) & (reasons["gmm_variant"] == FALLBACK_VARIANT)
        ]
        n_expected = len(FALLBACK_STEPS) * len(CATEGORIES) * len(SEEDS)
        all_steps = planted[planted["window"] == "all_steps"]
        ok &= _check(
            "fallbacks counted over all steps",
            len(all_steps) == 1 and int(all_steps.iloc[0]["n_steps"]) == n_expected,
            f"{None if all_steps.empty else int(all_steps.iloc[0]['n_steps'])} vs {n_expected}",
        )
        ok &= _check(
            "an out-of-ramp fallback is not reported inside the ramp",
            planted[planted["window"] == "ramp_6_20"].empty,
        )

        # The decoy wins on the blended column but not on the raw cut; the
        # analyzer must expose both so it cannot be shipped on the wrong one.
        decoy = prod[(prod["window"] == "ramp_6_20") & (prod["variant"] == "pooled_gumbel_cross")]
        ok &= _check(
            "decoy's raw cut is worse than its blended cost",
            len(decoy) == 1 and decoy.iloc[0]["mean_d_raw_cut_cost"] > 0 > decoy.iloc[0]["mean_d_cost"],
        )

        # Steps with a missing chain link must be dropped whole, from the terms
        # *and* from `total`.  Averaging around them leaves every term looking
        # sane on its own while the chain no longer sums to the total - which is
        # unfindable from the table unless the residual and the drop count are
        # both reported.  `INCOMPLETE_DISPLACEMENT` is planted so that a
        # per-column mean would miss the total by a visible amount.
        n_incomplete_expected = len(CATEGORIES) * len(SEEDS) * len(INCOMPLETE_STEPS)
        n_ramp_steps = len([t for t in range(2, 31) if 6 <= t <= 20])
        n_complete_expected = len(CATEGORIES) * len(SEEDS) * (n_ramp_steps - len(INCOMPLETE_STEPS))

        pooled = decomp[(decomp["window"] == "ramp_6_20") & (decomp["geometry"] == "pooled")]
        ok &= _check("decomposition telescopes", bool((pooled["residual"].abs() < 1e-9).all()))
        ok &= _check(
            "incomplete steps dropped from the threshold-unit chain",
            bool((pooled["n"] == n_complete_expected).all())
            and bool((pooled["n_incomplete"] == n_incomplete_expected).all()),
            f"n={list(pooled['n'])} n_incomplete={list(pooled['n_incomplete'])}",
        )

        costs = pd.read_csv(root / "agg" / "cut_cost_decomposition.csv")
        cost_ramp = costs[costs["window"] == "ramp_6_20"]
        ok &= _check(
            "cost decomposition telescopes",
            bool((cost_ramp["cost_residual"].abs() < 1e-9).all()),
            f"max |residual| = {cost_ramp['cost_residual'].abs().max():.6f}",
        )
        ok &= _check(
            "incomplete steps dropped from the cost chain",
            bool((cost_ramp["n"] == n_complete_expected).all())
            and bool((cost_ramp["n_incomplete"] == n_incomplete_expected).all()),
            f"n={list(cost_ramp['n'])} n_incomplete={list(cost_ramp['n_incomplete'])}",
        )
        for term, planted_cost in PLANTED_COST_TERMS.items():
            got = float(cost_ramp[f"cost_{term}"].iloc[0])
            ok &= _check(f"cost term {term}", abs(got - planted_cost) < 1e-9, f"{got:.5f} vs {planted_cost}")
        ok &= _check(
            "summary reports the decomposition's own integrity",
            abs(float(dec["error_terms_residual"])) < 1e-9
            and int(dec["error_terms_n_incomplete"]) == n_incomplete_expected,
            f"residual={dec.get('error_terms_residual')} n_incomplete={dec.get('error_terms_n_incomplete')}",
        )
        for term, planted in PLANTED_TERMS.items():
            got = float(pooled[f"term_{term}"].iloc[0])
            # prior_loss carries the shared jitter, which averages out over cells.
            tol = 2e-3 if term == "prior_loss" else 1e-9
            ok &= _check(f"term {term}", abs(got - planted) < tol, f"{got:.5f} vs {planted}")
        ok &= _check(
            "dominant term named",
            dec["dominant_error_term"] == "prior_loss",
            str(dec["dominant_error_term"]),
        )

        offs = summary["offsets"]["identity"]
        pooled_off = [o for o in offs if o["geometry"] == "pooled" and "max_patch" in o["arm"]]
        ok &= _check(
            "offset identity recovered",
            bool(pooled_off) and abs(pooled_off[0]["mean_abs_residual"]) < 1e-9,
        )
        ok &= _check(
            "offset correlation computable",
            bool(pooled_off) and abs(pooled_off[0]["corr"] - 1.0) < 1e-6,
            "" if not pooled_off else f"corr={pooled_off[0]['corr']}",
        )

        evt = pd.read_csv(root / "agg" / "cut_evt_evidence.csv")
        ok &= _check(
            "EVT gain is geometry-specific",
            bool(
                (evt[evt["geometry"] == "pooled"]["evt_loglik_gain"] > 0).all()
                and (evt[evt["geometry"] == "image"]["evt_loglik_gain"].abs() < 1e-9).all()
            ),
        )
        return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
