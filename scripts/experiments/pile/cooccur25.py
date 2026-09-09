#!/usr/bin/env python3
"""Group all TWENTY-FIVE classes for one negative pass, by measured co-occurrence.

The thirteen candidates were reviewed and the shipped twelve never were. If the
negative pass is going to look at an image anyway, adding the twelve costs
almost nothing -- provided they fall into the same scene groups. This checks.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts/experiments/pile")
sys.path.insert(0, "scripts/experiments/calibration")
import pile_config as pc  # noqa: E402
from _cells_io import load_medias  # noqa: E402
from coco_anchor import coco_truth, ensure_sources  # noqa: E402

ALL = list(pc.SCALE_CANDIDATES_3588) + list(pc.SCALE_CLASSES_ORIGINAL)
m = load_medias(pc.EMBEDDINGS / "vg_scale__siglip.pkl")
anchor = Path(pc.PILE / "coco_anchor")
image_data, instances = ensure_sources(anchor, False)
truth = coco_truth(instances, set(ALL))
meta = json.loads(Path(image_data).read_text())
coco_of = {int(x["image_id"]): int(x["coco_id"]) for x in meta if x.get("coco_id")}

sets = []
for i in m:
    cid = coco_of.get(i)
    if cid is None or not m[i].get("labels_exhaustive"):
        continue
    p = {c for c, b in truth.get(cid, {}).items() if b}
    if p:
        sets.append(p)
n = {c: sum(1 for s in sets if c in s) for c in ALL}
tot = len(sets)
print(f"{tot} COCO-scored images holding at least one of the 25\n")

TABLE = ["bowl", "cup", "bottle", "vase", "fork", "spoon", "sink", "chair", "cell phone"]
STREET = ["car", "truck", "bench", "fire hydrant"]

print("For each SHIPPED class: which existing group does it sit with?")
print(f"{'shipped':<12}{'n':>6}{'P(any table)':>14}{'P(any street)':>15}   verdict")
print("-" * 62)
for c in pc.SCALE_CLASSES_ORIGINAL:
    have = [s for s in sets if c in s]
    if not have:
        print(f"{c:<12}{0:>6}")
        continue
    pt = sum(1 for s in have if s & set(TABLE)) / len(have)
    ps = sum(1 for s in have if s & set(STREET)) / len(have)
    v = "TABLE" if pt > ps * 1.5 else ("STREET" if ps > pt * 1.5 else "either / its own")
    print(f"{c:<12}{len(have):>6}{100 * pt:>13.0f}%{100 * ps:>14.0f}%   {v}")

print("\nAnd the reverse -- how often a shipped class is the ONLY thing present:")
alone = {c: sum(1 for s in sets if s == {c}) for c in pc.SCALE_CLASSES_ORIGINAL}
for c, k in sorted(alone.items(), key=lambda x: -x[1])[:12]:
    print(f"  {c:<12}{k:>5} images alone  ({100 * k / max(n[c], 1):.0f}% of its images)")
