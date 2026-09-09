#!/usr/bin/env python3
"""Do COCO's annotators box vehicle seating as `chair`?

The class pass ruled that a car seat is not a Chair -- partly on principle and
partly on labelability, since you would be squinting through a windscreen. A
train interior is the opposite: the seats are plainly in frame. So ask the
annotators.
"""

import sys
from pathlib import Path

sys.path.insert(0, "scripts/experiments/pile")
import pile_config as pc  # noqa: E402
from coco_anchor import coco_truth, ensure_sources  # noqa: E402

VEHICLES = ("train", "bus", "airplane", "boat", "truck", "car")
WANT = {"chair", "bench", "couch", *VEHICLES}
anchor = Path(pc.PILE / "coco_anchor")
image_data, instances = ensure_sources(anchor, False)
truth = coco_truth(instances, WANT)

n_img = len(truth)
n_chair = sum(1 for v in truth.values() if v.get("chair"))
base = n_chair / n_img
print(f"{n_img} COCO images; {n_chair} hold a chair -- base rate {100 * base:.1f}%\n")
print(f"{'vehicle':<10}{'images':>9}{'also a chair':>14}{'rate':>8}{'lift':>8}   reading")
print("-" * 62)
for v in VEHICLES:
    imgs = [t for t in truth.values() if t.get(v)]
    if not imgs:
        continue
    k = sum(1 for t in imgs if t.get("chair"))
    r = k / len(imgs)
    lift = r / base
    read = (
        "annotators DO box seating there"
        if lift > 1.5
        else "no more than anywhere"
        if lift > 0.6
        else "annotators do NOT"
    )
    print(f"{v:<10}{len(imgs):>9}{k:>14}{100 * r:>7.1f}%{lift:>7.2f}x   {read}")

# The interior case specifically: a train image with MANY chairs is an interior.
tr = [t for t in truth.values() if t.get("train")]
many = [t for t in tr if len(t.get("chair") or []) >= 4]
print(f"\ntrain images with 4+ chair boxes (i.e. an interior): {len(many)} of {len(tr)}")
bs = [t for t in truth.values() if t.get("bus")]
print(
    f"bus   images with 4+ chair boxes:                    "
    f"{sum(1 for t in bs if len(t.get('chair') or []) >= 4)} of {len(bs)}"
)
