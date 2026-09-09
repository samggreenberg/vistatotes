#!/usr/bin/env python3
"""How should the negative pass be grouped? Ask the images.

Two classes belong in one pass when a single look covers both: they live in the
same kind of scene. COCO's exhaustive annotation over the whole pile gives that
directly as co-occurrence.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts/experiments/pile")
sys.path.insert(0, "scripts/experiments/calibration")
import pile_config as pc  # noqa: E402
from _cells_io import load_medias  # noqa: E402
from coco_anchor import coco_truth, ensure_sources  # noqa: E402

C = list(pc.SCALE_CANDIDATES_3588)
m = load_medias(pc.EMBEDDINGS / "vg_scale__siglip.pkl")
anchor = Path(pc.PILE / "coco_anchor")
image_data, instances = ensure_sources(anchor, False)
truth = coco_truth(instances, set(C))
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
print(f"{len(sets)} COCO-scored images holding at least one candidate\n")

n = {c: sum(1 for s in sets if c in s) for c in C}
print("P(B present | A present), row A -- how often a second look would be wasted")
w = 5
print(" " * 13 + "".join(f"{c[:w]:>7}" for c in C))
for a in C:
    row = ""
    for b in C:
        if a == b:
            row += f"{'-':>7}"
        else:
            j = sum(1 for s in sets if a in s and b in s)
            row += f"{100 * j / max(n[a], 1):>6.0f}%"
    print(f"{a:<13}{row}")

print("\nStrongest pairs (symmetric lift over independence):")
tot = len(sets)
pairs = []
for i, a in enumerate(C):
    for b in C[i + 1 :]:
        j = sum(1 for s in sets if a in s and b in s)
        if j < 15:
            continue
        exp = n[a] * n[b] / tot
        pairs.append((j / exp, j, a, b))
for lift, j, a, b in sorted(pairs, reverse=True)[:12]:
    print(f"  {a:<12} + {b:<12} {j:4d} images, {lift:4.1f}x independence")
