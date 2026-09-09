"""Is a 400-step trajectory a strict extension of the 100-step one?

If yes, #3547 needs ONE wave and reads both horizons off it, perfectly paired
within a cell.  #3319 ran two.  Compared on the arms both waves share.
"""

import csv
import glob
import os

SHALLOW = "/expscratch/sgreenberg/acq-3319/bin"
DEEP = "/expscratch/sgreenberg/acq-3319-deep/bin"
KEYCOLS = ("seed", "category", "gmm_variant", "pool_variant", "style", "embedder")
CMP = ("n_good", "n_bad", "cost", "threshold", "acq_threshold", "acq_pool_percentile", "average_precision", "auroc")


def rows_at(base, arm, t):
    d = os.path.join(base, arm, "results", "cells")
    out = {}
    for f in sorted(glob.glob(d + "/task_*.csv")):
        if "__" in os.path.basename(f):
            continue
        with open(f) as fh:
            for r in csv.DictReader(fh):
                if int(r["t"]) == t:
                    out[tuple(r.get(k, "") for k in KEYCOLS)] = r
    return out


for arm in ["prod", "acq_m1", "acq_m3", "acq_m4"]:
    a = rows_at(SHALLOW, arm, 100)
    b = rows_at(DEEP, arm, 100)
    shared = set(a) & set(b)
    if not shared:
        print("%-8s no shared (seed,category) keys: shallow=%d deep=%d" % (arm, len(a), len(b)))
        continue
    diffs = {c: 0 for c in CMP}
    for k in shared:
        for c in CMP:
            if a[k].get(c) != b[k].get(c):
                diffs[c] += 1
    bad = {c: n for c, n in diffs.items() if n}
    print("%-8s shared=%4d  %s" % (arm, len(shared), "IDENTICAL at t=100" if not bad else "DIFFER: %s" % bad))
