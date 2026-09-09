#!/usr/bin/env python3
"""The union rate has to come from the RANDOM stratum alone.

Half the 200 sampled images are the boundary stratum, chosen by text rank to be
suspicious. Comparing a rate over all 200 against COCO's 12.7% -- which was
computed uniformly over the whole frame -- is apples to oranges, and the report
did exactly that.
"""

import csv
import json
from pathlib import Path

BANK = Path("/expscratch/sgreenberg/classes-3588")
man = {int(r["image_id"]): r["stratum"] for r in csv.DictReader((BANK / "slates/Table_Objects/manifest.csv").open())}
POL = json.loads((BANK / "polarity.json").read_text())
OLD = set(POL["old"]["detectors"])

found, per = {}, {}
for p in sorted((BANK / "negbank").glob("*.json")):
    d = json.loads(p.read_text())
    name, labels = d.get("name"), d["labelset"].get("labels", [])
    if not name or len(labels) < 200 or name in per:
        continue
    per[name] = 1
    old = name in OLD
    for lb in labels:
        iid = int(Path(lb["origin_name"]).stem)
        present = (lb.get("label") != "good") if old else (lb.get("label") == "good")
        if present:
            found.setdefault(iid, []).append(name)

for st in ("random", "boundary"):
    ids = [i for i, s in man.items() if s == st]
    k = sum(1 for i in ids if i in found)
    print(
        f"{st:<10} {k:3d} / {len(ids):3d} = {100 * k / len(ids):5.1f}%"
        + ("   <- THE ESTIMATE" if st == "random" else "   (ranked, biased by design)")
    )
k = sum(1 for i in man if i in found)
print(f"{'all 200':<10} {k:3d} / {len(man):3d} = {100 * k / len(man):5.1f}%   (not an estimate of anything)")

n = sum(1 for i, s in man.items() if s == "random")
k = sum(1 for i, s in man.items() if s == "random" and i in found)
p_hat = k / n
se = (p_hat * (1 - p_hat) / n) ** 0.5
print(f"\nunion contamination of the shared pool: {100 * p_hat:.0f}% +/- {100 * 1.96 * se:.0f} (95% CI, n={n})")
print(
    "COCO's prediction over the same frame:   12.7%  -> "
    + ("inside the interval" if abs(p_hat - 0.127) < 1.96 * se else "OUTSIDE the interval")
)
