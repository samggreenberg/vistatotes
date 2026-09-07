#!/usr/bin/env python3
"""If we kept only negatives COCO can PROVE, how many would each class have?

The pool's negatives rest on VG silence for 56% of their number, and #3588
measured that silence wrong 0.0-7.1% of the time. Meanwhile #3667 found 3,247
images excluded from every class's evaluation for holding a different class,
about 1,850 of them COCO-exhaustive.

Both halves of that are answerable exactly on the COCO-anchored subset. So:
what does a provably-correct negative set look like, with no human in it?
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts/experiments/pile")
sys.path.insert(0, "scripts/experiments/calibration")
import pile_config as pc  # noqa: E402
from _cells_io import load_medias  # noqa: E402
from coco_anchor import coco_truth, ensure_sources  # noqa: E402

SHIPPED, CAND = list(pc.SCALE_CLASSES_ORIGINAL), list(pc.SCALE_CANDIDATES_3588)
m = load_medias(pc.EMBEDDINGS / "vg_scale__siglip.pkl")
pool = {i for i in m if not m[i].get("categories")}

anchor = Path(pc.PILE / "coco_anchor")
image_data, instances = ensure_sources(anchor, False)
truth = coco_truth(instances, set(SHIPPED) | set(CAND))
meta = json.loads(Path(image_data).read_text())
coco_of = {int(x["image_id"]): int(x["coco_id"]) for x in meta if x.get("coco_id")}

scored = [i for i in m if m[i].get("labels_exhaustive") and coco_of.get(i) is not None]
print(f"pile {len(m)}, COCO-scored {len(scored)} ({100 * len(scored) / len(m):.1f}%)")
print(
    f"current negatives per class: {len(pool)}, of which "
    f"{sum(1 for i in pool if i in set(scored))} are provable "
    f"and the rest rest on VG silence\n"
)


def has(i, c):
    return bool(truth.get(coco_of[i], {}).get(c))


print(f"{'class':<12}{'provable negs':>15}{'from pool':>11}{'from other-class':>18}{'vs 4200':>10}")
print("-" * 66)
for c in SHIPPED + ["--"] + CAND:
    if c == "--":
        print("-" * 66)
        continue
    negs = [i for i in scored if not has(i, c)]
    frm_pool = sum(1 for i in negs if i in pool)
    print(f"{c:<12}{len(negs):>15}{frm_pool:>11}{len(negs) - frm_pool:>18}{100 * len(negs) / len(pool) - 100:>+9.0f}%")
