"""Self-test for :mod:`analyze_transfer` on fabricated cells (no cluster data).

The transfer analyzer's whole job is to tell three things apart that all look
like "the last term is big": a genuine bias, finite-sample variance, and an
optimistic reference point.  Getting that wrong produces a clean table pointing
the next study in the wrong direction, which is the failure this line has already
paid for twice.  So the answers are planted here and checked to come back out:

* a term that is **symmetric** about zero is called a variance, and a term that
  is **one-signed** is not - and the verdict is relative to the siblings measured
  the same way, so a run where every term is symmetric does not produce four
  "variances";
* the **bracket** recovers a planted reference optimism, and ``optimism_share``
  is that optimism over the *naive* transfer, not over the honest one;
* the **learning curve** recovers a planted ``a + b/m`` - both the slope and,
  more importantly, the **intercept**, which is the study's third and independent
  estimate of the reference point;
* the curve is fitted on the axis the data actually supports: a run where cost
  tracks **positives** and not sample size is reported as ``n_pos``, because
  "what does it scale with" is the question #2883 asks;
* an estimator planted **cheaper than the empirical minimiser on test** is found
  and named, since that is the falsification of ``family_headroom_exhausted``;
* the label-free ``bagfit_*`` arms are reported but are **never** promoted - they
  are in ``SWEEP_ONLY``, and a remedy cannot win in the run that diagnoses the
  disease;
* the **reference sanity check fires** when the diagnostic columns are misaligned
  with the rows they are read beside, because a silent join error here is
  indistinguishable from a finding.

Usage::

    python selftest_analyze_transfer.py     # exits non-zero on failure
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

#: Planted reference optimism, in cost units: how much the sample-minimum
#: reference undershoots the honest one.
PLANTED_OPTIMISM = 0.030
#: Planted honest transfer - what is left after the reference is fixed.
PLANTED_HONEST_TRANSFER = 0.010
#: So the naive transfer, which is what the decomposition reports today, is:
PLANTED_NAIVE_TRANSFER = PLANTED_HONEST_TRANSFER + PLANTED_OPTIMISM
#: ...and the planted share of today's number that is reference artefact:
PLANTED_SHARE = PLANTED_OPTIMISM / PLANTED_NAIVE_TRANSFER  # 0.75

#: Planted learning curve ``cost = a + b / n_pos``.  ``a`` is the honest floor.
CURVE_A = 0.20
CURVE_B = 2.0

#: How much cheaper the planted variance-reduced estimator is than the ERM.
SMOOTH_EFFECT = -0.006

CATEGORIES = ["cat_a", "cat_b", "cat_c", "cat_d"]
SEEDS = [0, 1, 2, 3, 4, 5]
STEPS = list(range(2, 31))
ARMS = [("dinov3_patch", "max_patch"), ("siglip", "whole_image")]
SIM_N = 2000.0
#: Per-category prevalence, deliberately spread so ``n_pos`` and ``m`` are NOT
#: proportional across cells - otherwise the two candidate x-axes would be the
#: same axis and the "which one fits better" test could not fail.
#: Chosen so the smallest level (5 %) still leaves >= 1 positive: the analyzer
#: computes n_pos as sim_n * prevalence * frac with no floor, so a fabrication
#: that clamps would be planting a curve the analyzer cannot see.
PREVALENCE = {"cat_a": 0.01, "cat_b": 0.03, "cat_c": 0.10, "cat_d": 0.30}

LEVELS = [
    (0.05, "pooled_sim_oracle_f050"),
    (0.10, "pooled_sim_oracle_f100"),
    (0.25, "pooled_sim_oracle_f250"),
    (0.50, "pooled_sim_oracle_f500"),
    (1.00, "pooled_sim_oracle"),
]

#: Threshold-unit terms.  ``transfer`` is planted **symmetric** (mean ~0, large
#: spread) and the other three one-signed, which is the shape the corrected #3187
#: table shows and the shape H1 has to be able to detect.
ONE_SIGNED = {"prior_loss": 0.015, "identification": -0.018, "misspecification": 0.006}
TRANSFER_SPREAD = 0.017


def _ident(cat, seed, t, embedder, style):
    n_good = 1 + t // 3
    return {
        "seed": seed,
        "dataset": "vg_selftest",
        "category": cat,
        "strategy": "autopilot",
        "trainer": "app",
        "head": "linear_svm",
        "style": style,
        "embedder": embedder,
        "prevalence_arm": "",
        "realized_prevalence": PREVALENCE[cat],
        "t": t,
        "n_good": n_good,
        "n_bad": t - n_good,
        "phase": "good",
        "app_trained": 1,
        "acq_threshold": 0.5,
        "acq_pool_percentile": 0.5,
        "report_pool_percentile": 0.5,
    }


def _fabricate(root: Path, rng: np.random.Generator, *, break_join: bool = False) -> None:
    cells = root / "cells"
    cells.mkdir(parents=True, exist_ok=True)
    idx = 0
    for embedder, style in ARMS:
        for cat in CATEGORIES:
            for seed in SEEDS:
                rows, diag = [], []
                prev = PREVALENCE[cat]
                for t in STEPS:
                    ident = _ident(cat, seed, t, embedder, style)
                    n_pos_full = SIM_N * prev
                    # The curve: cost falls like 1/n_pos toward CURVE_A.
                    lvl_cost = {}
                    for frac, variant in LEVELS:
                        n_pos = n_pos_full * frac
                        lvl_cost[variant] = CURVE_A + CURVE_B / n_pos + rng.normal(0, 0.002)
                    erm_cost = lvl_cost["pooled_sim_oracle"]
                    # References, planted relative to the ERM's test cost.
                    honest_ref = erm_cost - PLANTED_HONEST_TRANSFER
                    naive_ref = honest_ref - PLANTED_OPTIMISM

                    d = dict(ident)
                    d.update(
                        {
                            "geometry": "pooled",
                            "sim_n": SIM_N,
                            "sim_prevalence": prev,
                            "sim_n_pos": n_pos_full,
                            "test_n": SIM_N + 1.0,
                            "test_n_pos": n_pos_full,
                            "fallback_median": 0.5,
                            "cost_test_oracle_naive": naive_ref,
                            "cost_test_oracle_honest": honest_ref,
                            "tau_test_oracle": 0.5,
                            "tau_test_oracle_honest": 0.5,
                        }
                    )
                    # Threshold-unit chain: three one-signed terms and one
                    # symmetric one, built so the differences come out exactly.
                    tau = {"tau_test_oracle": 0.5}
                    tau["tau_sim_oracle"] = 0.5 + rng.normal(0.0, TRANSFER_SPREAD)
                    tau["tau_supervised"] = tau["tau_sim_oracle"] + ONE_SIGNED["misspecification"]
                    tau["tau_priorfree"] = tau["tau_supervised"] + ONE_SIGNED["identification"]
                    tau["tau_cross"] = tau["tau_priorfree"] + ONE_SIGNED["prior_loss"]
                    d.update(tau)
                    if break_join:
                        # Same step, a *different* step's reference: the shape of
                        # a mis-keyed join, which must not pass silently.
                        d["cost_test_oracle_naive"] = naive_ref + 0.05
                    diag.append(d)

                    def _row(variant, raw_cost):
                        r = dict(ident)
                        r.update(
                            {
                                "pool_variant": "max",
                                "gmm_variant": variant,
                                "schedule": "",
                                "threshold": 0.5,
                                "threshold_provenance": "fold_anchored[2/2]",
                                "cost": raw_cost,
                                "fpr": 0.05,
                                "fnr": 0.05,
                                "oracle_cost": naive_ref,
                                "oracle_threshold": 0.5,
                                "raw_cut_cost": raw_cost,
                                "raw_cut_fpr": 0.05,
                                "raw_cut_fnr": 0.05,
                                "cut_fallback": 0,
                                "cut_fallback_kind": "interior",
                                "cut_fail_reason": "",
                                "auroc": 0.9,
                                "average_precision": 0.5,
                            }
                        )
                        return r

                    for _frac, variant in LEVELS:
                        rows.append(_row(variant, lvl_cost[variant]))
                    # The falsification arm: cheaper than the ERM on test.
                    rows.append(_row("pooled_sim_oracle_smooth", erm_cost + SMOOTH_EFFECT))
                    # ...and one that is not, so "beaten by" is a real filter.
                    rows.append(_row("pooled_sim_oracle_bag", erm_cost + 0.004))
                    # Label-free arms, planted to look like winners.
                    rows.append(_row("pooled_mid", erm_cost + 0.05))
                    rows.append(_row("pooled_priorfree", erm_cost + 0.04))
                    rows.append(_row("pooled_bagfit_mid", erm_cost + 0.01))
                    rows.append(_row("pooled_bagfit_priorfree", erm_cost + 0.005))
                pd.DataFrame(rows).to_csv(cells / f"task_{idx:04d}.csv", index=False)
                pd.DataFrame(diag).to_csv(cells / f"task_{idx:04d}__cutdiag.csv", index=False)
                idx += 1


def _check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}")
    return ok


def _run(root: Path):
    os.environ["CALIB_EXP"] = str(root.parent)
    os.environ["CALIB_RESULTS"] = str(root)
    for mod in ("common", "analyze_cut", "analyze_transfer"):
        sys.modules.pop(mod, None)
    import analyze_transfer

    rc = analyze_transfer.main()
    agg = root / "agg"
    tables = {p.stem.replace("transfer_", ""): pd.read_csv(p) for p in agg.glob("transfer_*.csv")}
    import json

    summary = json.loads((root / "summary_transfer.json").read_text())
    return rc, tables, summary


def main() -> int:
    rng = np.random.default_rng(2883)
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "results"
        _fabricate(root, rng)
        rc, tables, summary = _run(root)
        ok &= _check("analyzer exits 0", rc == 0)
        dec = summary["decisions"]

        # --- the reference sanity check passes on aligned data ---------------
        ok &= _check(
            "reference sanity passes when aligned",
            summary["reference_sanity"]["ok"] is True,
            f"max_abs_diff={summary['reference_sanity']['max_abs_diff']}",
        )

        # --- H1: symmetric term called a variance, one-signed ones not ------
        bov = tables["bias_or_variance"]
        prod = bov[(bov.window == "ramp_6_20") & bov.arm.str.contains("max_patch")]
        tr = prod[prod.term == "transfer"].iloc[0]
        ok &= _check("H1 transfer is symmetric", tr.symmetry < 0.10, f"symmetry={tr.symmetry:.3f}")
        others = prod[prod.term != "transfer"]
        ok &= _check(
            "H1 one-signed siblings are not",
            bool((others.symmetry > 0.3).all()),
            f"min sibling symmetry={others.symmetry.min():.3f}",
        )
        ok &= _check("H1 verdict is 'variance'", dec["h1_transfer_is_variance"] is True)

        # --- H2: the bracket recovers the planted optimism ------------------
        br = tables["bracket"]
        b = br[(br.window == "ramp_6_20") & br.arm.str.contains("max_patch")].iloc[0]
        ok &= _check(
            "H2 optimism recovered",
            abs(b.optimism - PLANTED_OPTIMISM) < 0.002,
            f"{b.optimism:.4f} vs planted {PLANTED_OPTIMISM}",
        )
        ok &= _check(
            "H2 honest transfer recovered",
            abs(b.transfer_honest - PLANTED_HONEST_TRANSFER) < 0.002,
            f"{b.transfer_honest:.4f} vs planted {PLANTED_HONEST_TRANSFER}",
        )
        ok &= _check(
            "H2 share is over the NAIVE transfer",
            abs(b.optimism_share - PLANTED_SHARE) < 0.05,
            f"{b.optimism_share:.3f} vs planted {PLANTED_SHARE:.3f}",
        )
        ok &= _check("H2 majority-is-reference verdict", dec["h2_majority_is_reference"] is True)

        # --- H3: the curve, its intercept, and the right axis ---------------
        ok &= _check(
            "H3 intercept recovers the planted floor",
            abs(dec["h3_n_pos_intercept"] - CURVE_A) < 0.01,
            f"{dec['h3_n_pos_intercept']:.4f} vs planted {CURVE_A}",
        )
        ok &= _check(
            "H3 slope recovers the planted b",
            abs(dec["h3_n_pos_slope"] - CURVE_B) < 0.3,
            f"{dec['h3_n_pos_slope']:.3f} vs planted {CURVE_B}",
        )
        ok &= _check(
            "H3 the two axes are NOT separated by fit quality",
            abs(dec["h3_n_pos_median_r2"] - dec["h3_m_median_r2"]) < 1e-3,
            f"r2(n_pos)={dec['h3_n_pos_median_r2']:.4f} r2(m)={dec['h3_m_median_r2']:.4f}",
        )
        ok &= _check(
            "H3 picks the positives axis",
            dec["h3_better_axis"] == "n_pos",
            f"rho(n_pos)={dec['h3_n_pos_slope_prevalence_rho']:+.3f} rho(m)={dec['h3_m_slope_prevalence_rho']:+.3f}",
        )

        # --- H4: the bound is refuted, by the right arm ---------------------
        beaten = dec["h4_erm_beaten_by"]
        ok &= _check("H4 finds the cheaper estimator", "pooled_sim_oracle_smooth" in beaten, str(beaten))
        ok &= _check("H4 does not name the dearer one", "pooled_sim_oracle_bag" not in beaten, str(beaten))
        ok &= _check("H4 verdict: not a bound", dec["h4_sim_oracle_is_not_a_bound"] is True)

        # --- the label-free arms are reported but cannot be promoted --------
        vr = tables["variance_reduction"]
        lf = vr[vr.label_free]
        ok &= _check("label-free arms are measured", len(lf) > 0, f"{len(lf)} rows")
        from analyze_cut import SHIP_ELIGIBLE, SWEEP_ONLY

        ok &= _check(
            "bagfit arms are sweep-only",
            all(v in SWEEP_ONLY and v not in SHIP_ELIGIBLE for v in ("pooled_bagfit_mid", "pooled_bagfit_priorfree")),
        )

    # --- and the join check actually fires when the join is broken ----------
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "results"
        _fabricate(root, np.random.default_rng(7), break_join=True)
        _rc, _tables, summary = _run(root)
        ok &= _check(
            "reference sanity FAILS on a broken join",
            summary["reference_sanity"]["ok"] is False,
            f"max_abs_diff={summary['reference_sanity']['max_abs_diff']}",
        )

    print("\n" + ("ALL CHECKS PASSED" if ok else "SELFTEST FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
