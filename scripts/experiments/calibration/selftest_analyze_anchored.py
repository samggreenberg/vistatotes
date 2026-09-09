"""Self-test for :mod:`analyze_anchored` on fabricated cells (no cluster data).

Plants a known deep-regime ordering - ``fold_anchored`` best, ``anchored`` next,
``rank_transfer`` a small win, all null below 50 votes - plus a stability gap
and an in-budget FNR, then checks the analyzer's mechanical H1-H4 verdicts
recover exactly that: right winner, right attribution (fold beats
label-anchored), right sign in every window, H3 from the threshold jitter, H4
from the FNR ceiling.  A sign error here would otherwise surface only after an
overnight GRID run.

Usage::

    python selftest_analyze_anchored.py     # exits non-zero on failure
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

CATEGORIES = ["cat_a", "cat_b", "cat_c"]
SEEDS = [0, 1]
MAX_T = 300
#: Deep-regime (votes >= 51, i.e. windows le_100/le_200/le_300) regret edge vs
#: xcal_only, per arm.  fold_anchored planted best; below 51 votes all null.
DEEP_EDGE = {
    "anchored_w10_mid": -0.02,
    "fold_anchored_w10_mid_qmean": -0.04,
    "rank_transfer": -0.01,
    "pooled_mid": 0.0,
    "xcal_only": 0.0,
}
#: Step-to-step threshold jitter scale per arm (H3: the winner must be calmer).
JITTER = {
    "anchored_w10_mid": 0.01,
    "fold_anchored_w10_mid_qmean": 0.005,
    "rank_transfer": 0.01,
    "pooled_mid": 0.03,
    "xcal_only": 0.05,
}


def _fabricate(results: Path, rng: np.random.Generator) -> None:
    cells = results / "cells"
    cells.mkdir(parents=True, exist_ok=True)
    idx = 0
    for cat in CATEGORIES:
        for seed in SEEDS:
            rows = []
            walk = {name: 0.5 for name in DEEP_EDGE}
            for t in range(2, MAX_T + 1):
                n_votes = t
                base_regret = 0.10 + 0.002 * rng.standard_normal()
                base = {
                    "seed": seed,
                    "dataset": "visual_genome_m",
                    "category": cat,
                    "strategy": "autopilot",
                    "trainer": "app",
                    "head": "linear",
                    "style": "max_patch",
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
                    "embedder": "dinov3_patch",
                }
                # Base blended row (gmm_variant "") - the analyzer must ignore it.
                rows.append(
                    {
                        **base,
                        "gmm_variant": "",
                        "threshold": 0.5,
                        "cost": 0.99,
                        "regret": 0.99,
                        "fpr": 0.5,
                        "fnr": 0.5,
                        "threshold_provenance": "gmm_blend",
                    }
                )
                for name, edge in DEEP_EDGE.items():
                    effect = edge if n_votes >= 51 else 0.0
                    walk[name] += JITTER[name] * rng.standard_normal()
                    regret = base_regret + effect
                    rows.append(
                        {
                            **base,
                            "gmm_variant": name,
                            "threshold": walk[name],
                            "cost": 0.2 + regret,
                            "regret": regret,
                            "fpr": 0.05,
                            "fnr": 0.12 + effect,  # well inside the 0.25 budget
                            "threshold_provenance": "anchored" if name.startswith("anchored") else name,
                        }
                    )
            pd.DataFrame(rows).to_csv(cells / f"task_{idx:04d}.csv", index=False)
            idx += 1


def main() -> int:
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        results = Path(tmp) / "results"
        _fabricate(results, rng)

        os.environ["CALIB_EXP"] = tmp
        os.environ["CALIB_RESULTS"] = str(results)
        sys.path.insert(0, str(Path(__file__).parent))

        import analyze_anchored  # noqa: PLC0415

        rc = analyze_anchored.main()
        assert rc == 0, f"analyze_anchored returned {rc}"

        verdicts = json.loads((results / "summary.json").read_text())
        assert verdicts["h1_supported"] is True, verdicts
        assert verdicts["h1_best_arm"] == "fold_anchored_w10_mid_qmean", verdicts
        att = verdicts["h1_attribution"]
        assert att["fold_beats_label_anchored"] is True, att
        assert abs(att["best_fold_anchored_d_regret"] - (-0.04)) < 0.005, att
        assert abs(att["rank_transfer_d_regret"] - (-0.01)) < 0.005, att
        assert verdicts["h2_supported"] is True, verdicts  # also beats the shipped blend
        assert verdicts["h3_supported"] is True, verdicts  # calmer than xcal_only
        assert verdicts["h4_supported"] is True, verdicts  # FNR inside the budget

        # The shallow windows must not leak the planted deep effect.
        contrasts = pd.read_csv(results / "agg" / "anchored_paired_contrasts.csv")
        shallow = contrasts[
            (contrasts["window"].isin(["le_20", "le_50"]))
            & (contrasts["control"] == "xcal_only")
            & (contrasts["variant"] == "fold_anchored_w10_mid_qmean")
        ]
        assert not shallow.empty and shallow["d_regret"].abs().max() < 0.005, shallow

    print("selftest_analyze_anchored: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
