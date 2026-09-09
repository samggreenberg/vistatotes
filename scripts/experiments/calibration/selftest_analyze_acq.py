"""Planted-answer self-test for ``analyze_acq.py``.

Fabricates arms whose answer is known by construction and asserts the analyzer
recovers it — in particular the three things that would otherwise read as good
news:

* an arm whose acquisition cut never moved must be reported as having measured
  nothing, not as "the lever does nothing";
* a falsification arm that fails to falsify must withhold the verdict;
* the ship rule must reject an arm that buys positives at the cost of a
  regression, and the cost criterion must read the **CI**, not the p-value.

A second planted grid covers the per-mode split added for #2877's pile re-run.
Its answer is a **disagreement**: the same arm ships under binary voting and
regresses under region voting, on cells that are otherwise identical.  Pooled,
the two cancel into a comfortable-looking null - which is the failure the split
exists to prevent, so the test asserts the pooled verdict and the region verdict
differ rather than merely that the split is present.

Run: ``python selftest_analyze_acq.py``
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
import analyze_acq as A  # noqa: E402

N_CAT, N_SEED, N_STEP = 4, 6, 100

#: (positives at t=100, final cost, acq percentile, deep-spike?) per arm.
#: `acq_m2` is the planted winner: more positives, cost unchanged, no new spikes.
#: `acq_m3` buys positives but regresses cost -> must be REJECTED.
#: `acq_m4` never moved its cut -> must be flagged as having measured nothing.
PLANT = {
    "prod": (4, 0.140, 0.885, False),
    "acq_m1": (6, 0.139, 0.910, False),
    "acq_m2": (9, 0.138, 0.940, False),
    "acq_m3": (11, 0.190, 0.960, True),
    "acq_m4": (4, 0.140, 0.885, False),  # lever stuck
    "acq_p2": (2, 0.145, 0.840, False),  # falsifier: fewer positives
    "rank_pin": (9, 0.138, 0.959, False),
}


def _cell(arm, cat, seed, rng):
    pos, cost_end, acq_pct, deep = PLANT[arm]
    t = np.arange(1, N_STEP + 1)
    cost = 0.30 * np.exp(-t / 20.0) + cost_end + rng.normal(0, 0.004, N_STEP)
    oracle = 0.6 * cost
    if deep:  # a mid-run threshold blip on a healthy ranking
        cost[70] = 0.62
        oracle[70] = 0.05
    n_good = np.clip((t * pos / N_STEP).astype(int), 0, None)
    return pd.DataFrame(
        {
            "seed": seed,
            "dataset": "coco_val",
            "embedder": "siglip2",
            "category": cat,
            "strategy": "autopilot",
            "trainer": "app",
            "head": "linear",
            "style": "whole_image",
            "prevalence_arm": "",
            "realized_prevalence": 0.037,
            "t": t,
            "n_good": n_good,
            "n_bad": t - n_good,
            "phase": "hard",
            "app_trained": 1,
            "acq_threshold": 0.2 if arm == "prod" else 0.25,
            "acq_pool_percentile": acq_pct,
            "report_pool_percentile": 0.885,
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


def build(root: Path):
    for arm in PLANT:
        cells = root / arm / "cells"
        cells.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(abs(hash(arm)) % 2**31)
        i = 0
        for ci in range(N_CAT):
            for seed in range(N_SEED):
                _cell(arm, f"cat{ci}", seed, rng).to_csv(cells / f"task_{i:04d}.csv", index=False)
                i += 1


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="acqselftest-"))
    try:
        build(root)
        df, prov = A.load_all(root)
        traj = A.trajectory_stats(df)
        s = A.build_summary(traj, prov)
        fails = []

        if len(df) != len(PLANT) * N_CAT * N_SEED * N_STEP:
            fails.append(f"row count wrong: {len(df)}")

        # 1. Lever verification.
        if s["lever_verification"]["acq_m4"]["moved"]:
            fails.append("acq_m4's cut never moved but was not flagged")
        if not s["lever_verification"]["acq_m2"]["moved"]:
            fails.append("acq_m2 moved its cut but was flagged as stuck")

        # 2. Falsifier.
        if not s["falsifier_behaved"]:
            fails.append("acq_p2 was planted with FEWER positives but did not register as falsifying")

        # 3. Ship rule.
        adopt = set(s["adopt"])
        if "acq_m2" not in adopt:
            fails.append(f"planted winner acq_m2 not adopted (ship={s['ship_rule'].get('acq_m2')})")
        if "acq_m3" in adopt:
            fails.append("acq_m3 regresses cost (+0.05) but was adopted")
        if "acq_m4" in adopt:
            fails.append("acq_m4's lever never moved but was adopted")

        # 4. The cost criterion must be an interval, not a point.
        c = s["contrasts_vs_control"]["acq_m3"]["final_cost"]
        if not (c["ci95_lo"] < c["ci95_hi"]) or c["ci95_hi"] <= 0:
            fails.append(f"cost CI degenerate or wrong side for a planted regression: {c}")

        # 5. Direction sanity: positives must rise monotonically m1 -> m2.
        pa = s["per_arm"]
        if not pa["acq_m1"]["median_positives_100"] < pa["acq_m2"]["median_positives_100"]:
            fails.append("planted positive ordering lost")

        # 6. Withheld verdict propagates to the report.
        s_bad = dict(s)
        s_bad["falsifier_behaved"] = False
        out = Path(tempfile.mkdtemp(prefix="acqrep-"))
        rep = A.write_report(s_bad, [], out).read_text()
        if "VERDICT WITHHELD" not in rep:
            fails.append("report does not withhold the verdict when the falsifier fails")
        if "acq_m4" not in rep:
            fails.append("report omits the stuck-lever arm")
        shutil.rmtree(out, ignore_errors=True)

        if fails:
            print("SELFTEST FAILED:")
            for f in fails:
                print("  -", f)
            return 1
        print(f"selftest OK: {len(df)} rows, {len(PLANT)} arms; planted winner={s['adopt']}")
        return 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


# --- scenario 2: a grid holding both voting modes ---------------------------
#: ``(positives@100, final cost, acq percentile)`` per arm, per mode.  The plant
#: is a DISAGREEMENT, and specifically one sized so that POOLING HIDES IT.
#: Every negative-k arm buys positives in both modes.  In binary they are free
#: (deltas within +/-0.001 of `prod`); in region they cost a ramp of +0.002 /
#: +0.008 / +0.016 / +0.024, so region rejects everything past k=-1.  Averaged
#: over an equal number of cells in each mode those deltas halve, and arms that
#: regress in region come back inside the +0.01 tolerance and are ADOPTED --
#: a regression in half the grid, reported as a pass.  That is the failure the
#: split exists to prevent, so the assertion is that the pooled verdict and the
#: region verdict DIFFER; a plant where they agree tests the plumbing, not the
#: point.
MODE_PLANT = {
    "binary": {
        "prod": (4, 0.140, 0.885),
        "acq_m1": (6, 0.140, 0.910),
        "acq_m2": (9, 0.139, 0.940),
        "acq_m3": (11, 0.139, 0.960),
        "acq_m4": (12, 0.141, 0.970),
        "acq_p2": (2, 0.145, 0.840),
        "rank_pin": (9, 0.139, 0.959),
    },
    "region": {
        "prod": (8, 0.300, 0.885),
        "acq_m1": (10, 0.302, 0.910),
        "acq_m2": (13, 0.308, 0.940),
        "acq_m3": (15, 0.316, 0.960),
        "acq_m4": (16, 0.324, 0.970),
        "acq_p2": (5, 0.305, 0.840),
        "rank_pin": (13, 0.312, 0.959),
    },
}
#: The two styles one patch cell emits, and the mode each is.  Both come out of
#: ONE task off one loaded pickle, which is what makes the difference between
#: them attributable to the geometry -- so the fixture writes them into one
#: file, exactly as `run_cells.py` does.
MODE_STYLES = {"whole_image": "binary", "max_patch": "region"}


def _mode_cell(arm, cat, seed, rng):
    frames = []
    for style, mode in MODE_STYLES.items():
        pos, cost_end, acq_pct = MODE_PLANT[mode][arm]
        t = np.arange(1, N_STEP + 1)
        cost = 0.30 * np.exp(-t / 20.0) + cost_end + rng.normal(0, 0.004, N_STEP)
        oracle = 0.6 * cost
        n_good = np.clip((t * pos / N_STEP).astype(int), 0, None)
        frames.append(
            pd.DataFrame(
                {
                    "seed": seed,
                    "dataset": "vg_scale_any",
                    "category": cat,
                    "strategy": "autopilot",
                    "trainer": "app",
                    "head": "linear_svm",
                    "style": style,
                    "prevalence_arm": "",
                    "realized_prevalence": 0.071,
                    "t": t,
                    "n_good": n_good,
                    "n_bad": t - n_good,
                    "phase": "hard",
                    "app_trained": 1,
                    "acq_threshold": 0.2 if arm == "prod" else 0.25,
                    "acq_pool_percentile": acq_pct,
                    "report_pool_percentile": 0.885,
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
                    # A paired arm: the opening runs in SigLIP space and every
                    # piece of learning in DINOv3's.  `embedder` names the pair
                    # because that is what decides the mode.
                    "embedder": "siglip+dinov3_patch",
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def build_modes(base: Path):
    """One half (`reg`), one arm dir per arm, cells holding BOTH styles."""
    for arm in MODE_PLANT["binary"]:
        cells = base / "reg" / arm / "results" / "cells"
        cells.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(abs(hash(("mode", arm))) % 2**31)
        i = 0
        for ci in range(N_CAT):
            for seed in range(N_SEED):
                _mode_cell(arm, f"cat{ci}", seed, rng).to_csv(cells / f"task_{i:04d}.csv", index=False)
                i += 1


def main_modes() -> int:
    base = Path(tempfile.mkdtemp(prefix="acqmodes-"))
    try:
        build_modes(base)
        df, prov = A.load_halves(base, ["bin", "reg"])
        traj = A.trajectory_stats(df)
        s = A.build_summary(traj, prov)
        fails = []

        # 0. Both styles survived as separate trajectories.  Without `style` in
        # the grouping key they collapse into one row per (category, seed) and
        # every endpoint below is a mixture of the two modes.
        if len(traj) != len(MODE_PLANT["binary"]) * N_CAT * N_SEED * len(MODE_STYLES):
            fails.append(f"trajectory count wrong: {len(traj)} - did `style` drop out of the key?")

        by_mode = s.get("by_mode") or {}
        if set(by_mode) != {"binary", "region"}:
            fails.append(f"expected both modes, got {sorted(by_mode)}")
        else:
            # 1. The plant: ships under binary, regresses under region.
            if "acq_m3" not in by_mode["binary"]["adopt"]:
                fails.append(
                    f"acq_m3 is free under binary but was not adopted there "
                    f"(ship={by_mode['binary']['ship_rule'].get('acq_m3')})"
                )
            if "acq_m3" in by_mode["region"]["adopt"]:
                fails.append("acq_m3 regresses cost (+0.05) under region voting but was adopted there")

            # 2. The point of the split: the pooled verdict is NOT the region
            # one, so a report that printed only the pooled table would ship an
            # arm that regresses in half its own grid.
            if set(s["adopt"]) == set(by_mode["region"]["adopt"]):
                fails.append("pooled verdict equals the region verdict - the plant no longer tests the split")
            if not s.get("pooled_is_descriptive"):
                fails.append("a two-mode summary did not mark its pooled verdict as descriptive")

        # 3. The DiD must recover the planted +0.05, within one embedder.
        did = (s.get("mode_did") or {}).get("contrasts", {}).get("final_cost", {}).get("acq_m3")
        if not did:
            fails.append("no difference-in-differences on final_cost for acq_m3")
        else:
            if not (0.012 < did["did"] < 0.022):
                fails.append(f"DiD did not recover the planted +0.017 cost gap: {did['did']:+.4f}")
            if did["ci95_lo"] <= 0:
                fails.append(f"DiD CI includes 0 for a planted mode split: {did}")

        # 4. Sizing is an output, not an assumption.
        for mode, sub in by_mode.items():
            if "n_for_target" not in (sub.get("sizing") or {}):
                fails.append(f"{mode}: no sizing readout on the decision endpoint")

        # 5. The report has to SAY all of that.
        out = Path(tempfile.mkdtemp(prefix="acqmodesrep-"))
        rep = A.write_report(s, [], out).read_text()
        for want in ("Voting mode: binary", "Voting mode: region", "descriptive only", "difference-in-differences"):
            if want not in rep:
                fails.append(f"report omits {want!r}")
        shutil.rmtree(out, ignore_errors=True)

        if fails:
            print("MODE SELFTEST FAILED:")
            for f in fails:
                print("  -", f)
            return 1
        print(
            f"mode selftest OK: {len(traj)} trajectories; "
            f"binary adopts {by_mode['binary']['adopt']}, region adopts {by_mode['region']['adopt']}, "
            f"pooled adopts {s['adopt']}"
        )
        return 0
    finally:
        shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main() or main_modes())
