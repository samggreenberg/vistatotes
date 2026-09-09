"""Self-test for :mod:`analyze_ab` on fabricated cells (no cluster data needed).

The A/B analyzer is the piece the #2799 ship decision reads, and it only ever
sees real data once - at the end of an overnight run.  This plants a known
effect (safe-ON cheaper by a fixed amount inside the ramp window, identical
outside it) in two synthetic ``results/cells`` trees and checks the analyzer
recovers it: the right sign and size of Δ, the right pairing unit (cells, not
steps), and windows that do not leak into each other.

Usage::

    python selftest_analyze_ab.py     # exits non-zero on failure
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

#: The planted ON−OFF cost effect inside the ramp window (ON is better).
RAMP_EFFECT = -0.05
#: Cells: 3 categories x 4 seeds on one arm.
CATEGORIES = ["cat_a", "cat_b", "cat_c"]
SEEDS = [0, 1, 2, 3]


def _fabricate(root: Path, *, safe_on: bool, rng: np.random.Generator, ramp_effect: float = RAMP_EFFECT) -> None:
    """Write one run's ``cells/task_*.csv`` with a planted ramp-window effect."""
    cells = root / "cells"
    cells.mkdir(parents=True, exist_ok=True)
    idx = 0
    for cat in CATEGORIES:
        for seed in SEEDS:
            rows = []
            for t in range(2, 31):
                n_votes = t
                base_cost = 0.30 + 0.002 * rng.standard_normal()
                effect = ramp_effect if (safe_on and 6 <= n_votes <= 20) else 0.0
                row = {
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
                    "gmm_variant": "",
                    "threshold": 0.5,
                    "cost": base_cost + effect,
                    "fpr": 0.02,
                    "fnr": 0.4 + effect,
                    "regret": 0.1 + effect,
                    "auroc": 0.9,
                    "average_precision": 0.5,
                    "degenerate": 0,
                    "embedder": "dinov3_patch",
                }
                rows.append(row)
                if safe_on:
                    # The ON run also emits #2799 variant rows; the A/B analyzer
                    # must ignore them (they are not what the run operated at).
                    for variant in ("xcal_only", "pooled_cross"):
                        rows.append({**row, "gmm_variant": variant, "cost": 0.99})
            pd.DataFrame(rows).to_csv(cells / f"task_{idx:04d}.csv", index=False)
            idx += 1


def main() -> int:
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        on_dir, off_dir = tmpdir / "on" / "results", tmpdir / "off" / "results"
        _fabricate(on_dir, safe_on=True, rng=rng)
        _fabricate(off_dir, safe_on=False, rng=rng)

        os.environ["CALIB_EXP"] = str(tmpdir / "on")
        os.environ["CALIB_RESULTS"] = str(on_dir)
        os.environ["CALIB_AB_ON"] = str(on_dir)
        os.environ["CALIB_AB_OFF"] = str(off_dir)
        sys.path.insert(0, str(Path(__file__).parent))

        import analyze_ab  # noqa: PLC0415

        rc = analyze_ab.main()
        assert rc == 0, f"analyze_ab returned {rc}"

        tbl = pd.read_csv(on_dir / "agg" / "ab_window_by_arm.csv")
        assert set(tbl["scope"]) == {"app_visible", "all_steps"}, set(tbl["scope"])
        # Fabricated rows are all app_trained=1, so both scopes must agree.
        for scope in ("app_visible", "all_steps"):
            s = tbl[(tbl["scope"] == scope) & (tbl["metric"] == "cost") & (tbl["window"] == "ramp_6_20")]
            assert abs(float(s["delta_on_minus_off"].iloc[0]) - RAMP_EFFECT) < 0.005, s
        cost = tbl[(tbl["metric"] == "cost") & (tbl["scope"] == "app_visible")].set_index("window")

        # Variant rows must be excluded: a mean cost near 0.99 would mean the
        # analyzer read the ON run's counterfactual rows instead of its real ones.
        assert cost.loc["ramp_6_20", "safe_on"] < 0.5, cost.loc["ramp_6_20"]
        # The planted effect, recovered inside the window and absent outside it.
        assert abs(cost.loc["ramp_6_20", "delta_on_minus_off"] - RAMP_EFFECT) < 0.005, cost.loc["ramp_6_20"]
        assert abs(cost.loc["pure_gmm_2_5", "delta_on_minus_off"]) < 0.005, cost.loc["pure_gmm_2_5"]
        assert abs(cost.loc["post_ramp_21_plus", "delta_on_minus_off"]) < 0.005, cost.loc["post_ramp_21_plus"]
        # Cells, not steps, are the paired unit.
        assert int(cost.loc["ramp_6_20", "n_cells"]) == len(CATEGORIES) * len(SEEDS)
        # A real, one-sided effect must register as significant and as a sweep.
        assert cost.loc["ramp_6_20", "p_wilcoxon"] < 0.05
        assert cost.loc["ramp_6_20", "win_rate_on"] == 1.0
        # ... and the verdict must read "force it on".
        summary = (on_dir / "summary_ab.json").read_text()
        assert '"force_on_for_all_users": true' in summary, summary

        # Sign check: flip the planted effect and the verdict must flip too.
        harmful = tmpdir / "on2" / "results"
        _fabricate(harmful, safe_on=True, rng=np.random.default_rng(1), ramp_effect=-RAMP_EFFECT)
        os.environ["CALIB_AB_ON"] = str(harmful)
        os.environ["CALIB_AB_OUT"] = str(harmful)
        assert analyze_ab.main() == 0
        assert '"force_on_for_all_users": false' in (harmful / "summary_ab.json").read_text()

    print("selftest_analyze_ab: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
