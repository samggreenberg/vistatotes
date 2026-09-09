"""When does a cell acquire its FIRST deep spike?

`frontier_3547.py` reports identical deep-spike incidence at t=100 and t=400 for
all seven arms. That is either the finding H2 predicts (late spikes were
exhaustion, and this pile has none) or a masking bug. Distinguished by asking
the raw trajectories directly: if no cell's FIRST spike lands after t=100, the
two horizons agree because nothing happens between them.

Pass `--base` to point this at the H2 control (the shallow pile on the same
commit), whose arms are a subset -- missing arms are skipped, not fatal.
`--csv` writes the per-cell first-spike times the report's figures read, so the
picture and the table cannot drift apart.
"""

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _cells_io import load_arm  # noqa: E402

WARM_T, DEEP_COST, DEEP_EXCESS = 20, 0.25, 0.20
ALL_ARMS = ["prod", "acq_m1", "acq_m3", "acq_m4", "acq_m5", "acq_m6", "acq_p2"]
ARM_K = {"prod": 0, "acq_m1": -1, "acq_m3": -3, "acq_m4": -4, "acq_m5": -5, "acq_m6": -6, "acq_p2": 2}

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--base", type=pathlib.Path, default=pathlib.Path("/expscratch/sgreenberg/acq-3547/bin"))
p.add_argument("--arms", default=",".join(ALL_ARMS), help="comma-separated; missing ones are skipped")
p.add_argument("--split-t", type=int, default=100, help="horizon the early/late counts straddle")
p.add_argument("--csv", type=pathlib.Path, default=None, help="directory for the per-cell CSV")
a = p.parse_args()

print("base %s   split at t=%d" % (a.base, a.split_t))
print(
    "%-8s %6s %8s %9s %9s   %s"
    % ("arm", "cells", "any", "first<=%d" % a.split_t, "first>%d" % a.split_t, "first-spike t (quartiles)")
)
rows = []
for arm in a.arms.split(","):
    results = a.base / arm / "results"
    if not results.exists():
        print("%-8s %6s   (no results dir -- skipped)" % (arm, "-"))
        continue
    raw, _ = load_arm(results)
    raw = raw[(raw["embedder"] == "siglip") & (raw["style"] == "whole_image")]
    firsts, n = [], 0
    for (cat, seed), g in raw.groupby(["category", "seed"], dropna=False):
        n += 1
        g = g.sort_values("t")
        t = g["t"].to_numpy()
        c = g["cost"].to_numpy(dtype=float)
        o = g["oracle_cost"].to_numpy(dtype=float)
        d = (t >= WARM_T) & (c >= DEEP_COST) & ((c - o) >= DEEP_EXCESS)
        first = float(t[np.argmax(d)]) if d.any() else np.nan
        firsts.append(first)
        rows.append({"arm": arm, "k": ARM_K.get(arm), "category": cat, "seed": seed, "first_t": first})
    f = np.array(firsts, dtype=float)
    has = np.isfinite(f)
    early = int(np.sum(f[has] <= a.split_t))
    late = int(np.sum(f[has] > a.split_t))
    q = np.percentile(f[has], [25, 50, 75]) if has.any() else [np.nan] * 3
    print(
        "%-8s %6d %8d %9d %9d   %s"
        % (arm, n, has.sum(), early, late, "%.0f / %.0f / %.0f" % tuple(q) if has.any() else "-")
    )

if a.csv:
    import pandas as pd

    a.csv.mkdir(parents=True, exist_ok=True)
    dest = a.csv / "spike_timing_3547_cells.csv"
    pd.DataFrame(rows).to_csv(dest, index=False)
    print("wrote %s (%d cells)" % (dest, len(rows)))
