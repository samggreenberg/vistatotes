"""Planted-answer self-test for ``analyze_spikes.py`` (issue #2847).

Fabricates four arms of cells whose spike structure is *known by construction*
and asserts the analyzer recovers it.  The point is to catch the class of error
that reads as good news - a filter that keeps zero rows, a cold-start hump
counted as a mid-run spike, an oracle jump miscounted as a threshold failure -
before an overnight run rather than after.

Run: ``python selftest_analyze_spikes.py``
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import common

common.setup_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import analyze_spikes as A  # noqa: E402

#: Planted deep-spike counts per arm, in trajectories out of N_CAT*N_SEED.
PLANT = {"A_mlp_xcal": 6, "B_mlp_fused": 3, "C_lin_xcal": 5, "D_lin_fused": 0}
N_CAT, N_SEED, N_STEP = 3, 4, 100


def _cell(arm: str, cat: str, seed: int, *, deep: bool, oracle_spike: bool, cold_hump: bool) -> pd.DataFrame:
    t = np.arange(1, N_STEP + 1)
    base = 0.30 * np.exp(-t / 25.0) + 0.06
    oracle = 0.6 * base
    cost = base.copy()
    if cold_hump:  # a big early bump - must NOT be counted (t < WARM_T)
        cost[:8] += 0.55
        oracle[:8] += 0.50
    if deep:  # a mid-run threshold blip: cost leaps, oracle does not
        cost[60] = 0.68
    if oracle_spike:  # ranking collapse - both move, so it is not a threshold spike
        cost[40] = 0.62
        oracle[40] = 0.58
    return pd.DataFrame(
        {
            "seed": seed,
            "dataset": "coco_val",
            "embedder": "siglip2",
            "category": cat,
            "strategy": "autopilot",
            "trainer": "app",
            "head": "linear" if "lin" in arm else "mlp",
            "style": "whole_image",
            "prevalence_arm": "",
            "realized_prevalence": 0.037,
            "t": t,
            "n_good": np.minimum(t // 8 + 1, 12),
            "n_bad": t,
            "phase": "hard",
            "app_trained": 1,
            "pool_variant": "max",
            "gmm_variant": "",
            "schedule": "",
            "threshold": 0.2,
            "threshold_provenance": "fold_anchored[2/2]",
            "degenerate": 0,
            "xcal_threshold": 0.12,
            "gmm_cut": "",
            "cost": cost,
            "fpr": cost * 0.3,
            "fnr": cost * 0.7,
            "auroc": 0.9,
            "average_precision": 0.5,
            "oracle_threshold": 0.2,
            "oracle_cost": oracle,
            "oracle_fpr": oracle * 0.3,
            "oracle_fnr": oracle * 0.7,
            "regret": cost - oracle,
        }
    )


def _decoy(df: pd.DataFrame) -> pd.DataFrame:
    """A variant row that must be filtered out: absurd costs, different tag."""
    d = df.copy()
    d["gmm_variant"] = "image_mid"
    d["cost"] = 0.95
    d["oracle_cost"] = 0.05
    return d


def build(root: Path) -> None:
    for arm, n_deep in PLANT.items():
        cells = root / arm / "cells"
        cells.mkdir(parents=True, exist_ok=True)
        idx, planted = 0, 0
        for ci in range(N_CAT):
            for seed in range(N_SEED):
                deep = planted < n_deep
                planted += int(deep)
                df = _cell(
                    arm,
                    f"cat{ci}",
                    seed,
                    deep=deep,
                    # One trajectory per arm gets a ranking collapse instead -
                    # it must not be counted as a threshold spike.
                    oracle_spike=(ci == 0 and seed == 0),
                    cold_hump=True,
                )
                pd.concat([df, _decoy(df)], ignore_index=True).to_csv(cells / f"task_{idx:04d}.csv", index=False)
                # A sidecar the loader must skip, and a zero-byte cell it must
                # count rather than silently swallow.
                (cells / f"task_{idx:04d}__sweep.csv").write_text("junk\n1\n")
                idx += 1
        (cells / f"task_{idx:04d}.csv").touch()  # zero-byte
        # Header-only cell: the "100 votes, zero positives" outcome.  Must be
        # counted as such, not as a trajectory and not as a read failure.
        _cell(arm, "cat0", 0, deep=False, oracle_spike=False, cold_hump=False).head(0).to_csv(
            cells / f"task_{idx + 1:04d}.csv", index=False
        )


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="spikeselftest-"))
    try:
        build(root)
        df, prov = A.load_all(root)
        fails: list[str] = []

        # The decoy rows must be gone, the sidecars never read.
        exp_rows = len(PLANT) * N_CAT * N_SEED * N_STEP
        if len(df) != exp_rows:
            fails.append(f"base rows {len(df)} != {exp_rows} (variant/sidecar filter wrong)")
        if not all(p["zero_byte"] for p in prov.values()):
            fails.append("zero-byte cell not reported in provenance")
        if not all(len(p["no_positive_found"]) == 1 for p in prov.values()):
            fails.append(f"header-only cell miscounted: {[p['no_positive_found'] for p in prov.values()]}")
        if any(p["unreadable"] for p in prov.values()):
            fails.append("header-only or zero-byte cell wrongly counted as unreadable")

        traj = A.trajectory_stats(df)
        for arm, want in PLANT.items():
            got = int(traj[traj["arm"] == arm]["has_deep"].sum())
            if got != want:
                fails.append(f"{arm}: deep-spike trajectories {got} != planted {want}")

        # The cold hump is bigger than every planted spike; if WARM_T were not
        # applied, every trajectory would flag.
        if int(traj["has_deep"].sum()) == len(traj):
            fails.append("every trajectory flagged - cold-start window not excluded")
        # The ranking collapse moves cost AND oracle together, so its excess is
        # small and it must not be a deep spike.
        if traj["max_jump_oracle"].max() <= 0:
            fails.append("oracle jump never detected - the ranking control is dead")

        summary = A.build_summary(df, traj, prov)
        if not summary["control_reproduces_phenomenon"]:
            fails.append("control arm planted with 6 spikes reported as not reproducing")
        d_rate = summary["per_arm"]["D_lin_fused"]["deep_spike_trajectory_rate"]
        a_rate = summary["per_arm"]["A_mlp_xcal"]["deep_spike_trajectory_rate"]
        if not (d_rate < a_rate):
            fails.append(f"arm ordering lost: D={d_rate} not below A={a_rate}")

        inc = A.mcnemar_incidence(traj, "A_mlp_xcal", "D_lin_fused")
        if inc["only_a"] != PLANT["A_mlp_xcal"] or inc["only_b"] != 0:
            fails.append(f"discordant pairs wrong: {inc}")

        # Sign convention: the paired delta is arm minus control, so an arm with
        # fewer/smaller spikes must come out NEGATIVE.  A flipped sign here is
        # exactly the error that reads as "production is worse".
        pv = A.paired_vs(traj, "max_excess_warm", "A_mlp_xcal", "D_lin_fused")
        if not (pv["median_delta"] <= 0 and pv["n_pairs"] == N_CAT * N_SEED):
            fails.append(f"paired sign/pairing wrong: {pv}")

        if fails:
            print("SELFTEST FAILED:")
            for f in fails:
                print("  -", f)
            return 1
        print(f"selftest OK: {len(df)} base rows, planted spike counts recovered for all {len(PLANT)} arms")
        return 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
