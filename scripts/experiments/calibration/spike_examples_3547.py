"""Literal deep-spike cells, printed as trajectory rows around the spike.

An incidence rate says how often the guardrail fired; it does not say what
firing looked like, and a report that never shows one is asking to be trusted
about a thing nobody has seen. This prints the actual rows: the step the cell
crossed the rule, what the fitted threshold was costing, and what the oracle
would have cost on the same step.

Usage::

    python spike_examples_3547.py --base <study>/bin --arm acq_m3 [--after 100]
"""

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _cells_io import load_arm  # noqa: E402

WARM_T, DEEP_COST, DEEP_EXCESS = 20, 0.25, 0.20

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--base", type=pathlib.Path, required=True)
p.add_argument("--arm", required=True)
p.add_argument("--after", type=int, default=None, help="only cells whose FIRST spike lands after this t")
p.add_argument("--n", type=int, default=3, help="how many cells to show")
p.add_argument("--context", type=int, default=2, help="rows either side of the first spike")
a = p.parse_args()

raw, _ = load_arm(a.base / a.arm / "results")
raw = raw[(raw["embedder"] == "siglip") & (raw["style"] == "whole_image")]

hits = []
for (cat, seed), g in raw.groupby(["category", "seed"], dropna=False):
    g = g.sort_values("t")
    t = g["t"].to_numpy()
    c = g["cost"].to_numpy(dtype=float)
    o = g["oracle_cost"].to_numpy(dtype=float)
    d = (t >= WARM_T) & (c >= DEEP_COST) & ((c - o) >= DEEP_EXCESS)
    if not d.any():
        continue
    i = int(np.argmax(d))
    if a.after is not None and t[i] <= a.after:
        continue
    hits.append((float(c[i] - o[i]), cat, seed, i, g))

hits.sort(reverse=True, key=lambda r: r[0])
print(
    "arm %s under %s -- %d cell(s) match%s"
    % (a.arm, a.base, len(hits), "" if a.after is None else " (first spike after t=%d)" % a.after)
)
for excess, cat, seed, i, g in hits[: a.n]:
    print("\n=== %s / seed %s -- first spike at t=%d, excess %.2f" % (cat, seed, int(g["t"].to_numpy()[i]), excess))
    print("   %6s %8s %8s %8s %8s" % ("t", "cost", "oracle", "excess", "n_good"))
    lo, hi = max(0, i - a.context), min(len(g), i + a.context + 1)
    for j in range(lo, hi):
        r = g.iloc[j]
        mark = "  <-- spike" if j == i else ""
        print(
            "   %6d %8.2f %8.2f %8.2f %8d%s"
            % (
                int(r["t"]),
                float(r["cost"]),
                float(r["oracle_cost"]),
                float(r["cost"]) - float(r["oracle_cost"]),
                int(r["n_good"]),
                mark,
            )
        )
