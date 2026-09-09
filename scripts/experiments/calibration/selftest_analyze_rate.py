"""Self-test for :mod:`analyze_rate` on fabricated cells (no cluster data).

Plants a **known kappa optimum** and checks the analyzer finds it.  The headline
number of the #2861 run is an argmin over a swept grid, and an argmin is exactly
the kind of quantity a sign error, a mis-parsed arm name, or a bad pairing key
will move without ever crashing - so it gets a planted-answer test.

What is planted:

* a fold-anchored curve in log-kappa that is **exactly flat between 0.3 and
  0.5** and rises outside it, in **both** environments, and only in the deep
  windows.  A flat bottom is planted on purpose: the real question the
  analyzer answers is "which kappas are tied with the best", so a curve with a
  single sharp minimum would test the easy half of the machinery only;
* a monotone-worsening label-anchored curve (the #2860 "dishonest anchors"
  shape), so family separation is exercised;
* two environments whose fit populations differ 5x, so the kappa-vs-gamma table
  must report the same ``best_kappa`` with different ``gamma_at_deep``;
* a flat region around the optimum wide enough that the plateau must contain
  more than the argmin alone, and a far-away kappa that must be excluded.

Usage::

    python selftest_analyze_rate.py     # exits non-zero on failure
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

KAPPAS = [0.01, 0.03, 0.1, 0.3, 0.5, 1.0, 2.0, 3.0]
#: The planted flat bottom: these two kappas are exactly tied and everything
#: else is strictly worse.
FLAT = (0.3, 0.5)
#: (dataset, embedder, style, n_medias) - the 5x span the gamma test needs.
ENVS = [
    ("visual_genome_m", "dinov3_patch", "max_patch", 4000),
    ("caltech101_m", "siglip", "whole_image", 800),
]
CATEGORIES = ["cat_a", "cat_b", "cat_c", "cat_d"]
SEEDS = [0, 1]
MAX_T = 300
#: Curvature of the planted rise outside the flat bottom, in log10(kappa).
CURVATURE = 0.05
NOISE = 0.004


def _prov(name: str, t: int) -> str:
    """Provenance strings shaped like the harness's own, incl. a fold fallback.

    Every 50th step one fold degenerates, so the analyzer's fold-arm fallback
    classifier has something to find; reading fold arms with the label family's
    "unanchored" rule would report 0% here and the test would catch it.
    """
    if name.startswith("fold_anchored_"):
        return "fold_anchored[1/2]" if t % 50 == 0 else "fold_anchored[2/2]"
    if name.startswith("anchored_w"):
        return "unanchored:inverted_means" if t % 100 == 0 else "anchored"
    return name


def _fold_edge(kappa: float) -> float:
    lo, hi = (math.log10(k) for k in FLAT)
    x = math.log10(kappa)
    over = max(0.0, x - hi) if x > hi else max(0.0, lo - x)
    return -0.08 + CURVATURE * over**2


def _label_edge(kappa: float) -> float:
    """Monotone in kappa: helps a little when light, hurts a lot when heavy."""
    return -0.05 + 0.06 * (math.log10(kappa) - math.log10(KAPPAS[0]))


def _fabricate(results: Path, rng: np.random.Generator) -> None:
    cells = results / "cells"
    cells.mkdir(parents=True, exist_ok=True)
    info: dict = {"datasets": {}, "failed": []}
    idx = 0
    for ds, emb, style, n_medias in ENVS:
        info["datasets"].setdefault(ds, {})[emb] = {
            "n_medias": n_medias,
            "selected_categories": CATEGORIES,
        }
        for cat in CATEGORIES:
            for seed in SEEDS:
                rows = []
                walk = 0.5
                cell_bias = NOISE * rng.standard_normal()
                for t in range(2, MAX_T + 1):
                    n_votes = t
                    # The analyzer's "deep" set is every window whose upper
                    # edge is >= 100, i.e. le_100 / le_200 / le_300 = votes
                    # 51-300 (the same definition the #2860 report used).
                    deep = n_votes > 50
                    base = {
                        "seed": seed,
                        "dataset": ds,
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
                        "embedder": emb,
                        "n_pool_rows": 1.0,
                        "schedule": "",
                    }
                    xcal_regret = 0.30 + 0.002 * rng.standard_normal()
                    walk += 0.05 * rng.standard_normal()

                    def emit(name: str, edge: float, jitter: float, thr: float = walk) -> None:
                        regret = xcal_regret + edge + NOISE * rng.standard_normal()
                        rows.append(
                            {
                                **base,
                                "gmm_variant": name,
                                "threshold": thr + jitter * rng.standard_normal(),
                                "cost": 0.2 + regret,
                                "regret": regret,
                                "fpr": 0.05,
                                "fnr": 0.12 + max(edge, -0.1),
                                "threshold_provenance": _prov(name, t),
                            }
                        )

                    emit("xcal_only", 0.0, 0.05)
                    emit("pooled_mid", 0.005, 0.05)
                    # Counterfactual schedule rows: the shipped blend, planted
                    # halfway between x-cal and the fusion arms, so the
                    # "vs shipped schedule" table has a known answer too.
                    for sched, edge in (("slow_cap50", -0.04), ("cap50", -0.04), ("pure_gmm", -0.03)):
                        rows.append(
                            {
                                **base,
                                "gmm_variant": "",
                                "schedule": sched,
                                "threshold": walk,
                                "cost": 0.2 + xcal_regret + (edge if deep else 0.0),
                                "regret": xcal_regret + (edge if deep else 0.0),
                                "fpr": 0.05,
                                "fnr": 0.12,
                                "threshold_provenance": "gmm_blend",
                            }
                        )
                    emit("rank_transfer", -0.02 if deep else 0.0, 0.02)
                    for kappa in KAPPAS:
                        f = (_fold_edge(kappa) + cell_bias) if deep else 0.0
                        lbl = (_label_edge(kappa) + cell_bias) if deep else 0.0
                        emit(f"fold_anchored_w{kappa:g}_rate_qmean", f, 0.005)
                        emit(f"fold_anchored_w{kappa:g}_mid_qmean", f + 0.004, 0.005)
                        emit(f"anchored_w{kappa:g}_rate", lbl, 0.002)
                        emit(f"anchored_w{kappa:g}_mid", lbl + 0.004, 0.002)
                pd.DataFrame(rows).to_csv(cells / f"task_{idx:04d}.csv", index=False)
                idx += 1
    (results / "prepare_info.json").write_text(json.dumps(info, indent=2))


def main() -> int:
    rng = np.random.default_rng(7)
    with tempfile.TemporaryDirectory() as tmp:
        results = Path(tmp) / "results"
        _fabricate(results, rng)

        os.environ["CALIB_EXP"] = tmp
        os.environ["CALIB_RESULTS"] = str(results)
        sys.path.insert(0, str(Path(__file__).parent))

        import analyze_rate  # noqa: PLC0415

        rc = analyze_rate.main()
        assert rc == 0, f"analyze_rate returned {rc}"

        # --- the argmin, pooled and per environment ------------------------
        def plateau_of(row) -> set[float]:
            return {float(x) for x in str(row["plateau"]).split(",")}

        plats = pd.read_csv(results / "agg" / "rate_plateau.csv")
        pooled = plats[
            (plats["scope"] == "pooled_deep") & (plats["family"] == "fold_anchored") & (plats["rule"] == "rate")
        ]
        assert len(pooled) == 1, pooled
        assert pooled["best_kappa"].iloc[0] in FLAT, pooled.to_dict("records")

        per_env = plats[
            (plats["scope"] == "per_env_deep") & (plats["family"] == "fold_anchored") & (plats["rule"] == "rate")
        ]
        assert len(per_env) == len(ENVS), per_env
        assert set(per_env["best_kappa"]) <= set(FLAT), per_env.to_dict("records")

        # The plateau must recover the whole planted flat bottom and nothing
        # from the far ends of the grid.
        plateau = plateau_of(pooled.iloc[0])
        assert set(FLAT) <= plateau, plateau
        assert not plateau & {0.01, 0.03, 2.0, 3.0}, plateau
        for _, row in per_env.iterrows():
            assert set(FLAT) <= plateau_of(row), (row["env"], row["plateau"])

        # --- family separation ---------------------------------------------
        lab = plats[
            (plats["scope"] == "pooled_deep") & (plats["family"] == "label_anchored") & (plats["rule"] == "rate")
        ]
        assert lab["best_kappa"].iloc[0] == min(KAPPAS), lab.to_dict("records")

        # --- the curve itself, and that shallow windows stay null -----------
        curve = pd.read_csv(results / "agg" / "rate_curve_pooled_deep.csv")
        fr = curve[(curve["family"] == "fold_anchored") & (curve["rule"] == "rate")].set_index("kappa")
        assert abs(fr.loc[FLAT[0], "d_regret"] - _fold_edge(FLAT[0])) < 0.01, fr
        assert fr["d_regret"].idxmin() in FLAT, fr
        assert fr.loc[3.0, "d_regret"] > fr.loc[FLAT[1], "d_regret"] + 0.01, fr
        allw = pd.read_csv(results / "agg" / "rate_curve.csv")
        shallow = allw[(allw["window"] == "le_20") & (allw["family"] == "fold_anchored")]
        assert not shallow.empty and shallow["d_regret"].abs().max() < 0.01, shallow

        # --- kappa* fixed while gamma* moves with N -------------------------
        gam = pd.read_csv(results / "agg" / "rate_gamma_test.csv")
        gam = gam[gam["rule"] == "rate"]
        assert set(gam["best_kappa"]) <= set(FLAT), gam.to_dict("records")
        assert gam["n_fit"].nunique() == len(ENVS), gam.to_dict("records")
        assert gam["gamma_at_deep"].max() / gam["gamma_at_deep"].min() > 3, gam.to_dict("records")

        # --- vs the shipped blend, with the right control per voting mode ---
        ship = pd.read_csv(results / "agg" / "rate_vs_shipped_schedule.csv")
        assert set(ship["control"]) == {"sched:slow_cap50", "sched:cap50"}, set(ship["control"])
        best = ship[ship["arm"] == f"fold_anchored_w{FLAT[0]:g}_rate_qmean"]
        assert len(best) == len(ENVS), best.to_dict("records")
        # planted: fusion -0.08 vs xcal, shipped blend -0.04 vs xcal
        assert (best["d_regret"] - (-0.04)).abs().max() < 0.01, best.to_dict("records")

        # --- provenance: both families classified by their own convention --
        prov = pd.read_csv(results / "agg" / "rate_provenance.csv")
        fold = prov[prov["family"] == "fold_anchored"]["fallback_rate"]
        lab = prov[prov["family"] == "label_anchored"]["fallback_rate"]
        # planted: 1 step in 50 for fold arms, 1 in 100 for label arms
        assert (fold - 0.02).abs().max() < 0.005, prov.to_dict("records")
        assert (lab - 0.01).abs().max() < 0.005, prov.to_dict("records")

        assert (results / "REPORT_rate.md").exists()

    print("selftest_analyze_rate: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
