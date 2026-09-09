#!/usr/bin/env python3
"""What #3667 changes, computed from the shipped cell without rebuilding it.

`evaluable_categories` is derivable from data already in the pickle: an image's
own cells, whether it is a shared negative, whether COCO annotated it, and which
classes it holds. So the effect can be priced before spending the GPU hours --
and before changing a file other studies are reading right now.
"""

import sys
from collections import Counter

sys.path.insert(0, "scripts/experiments/pile")
sys.path.insert(0, "scripts/experiments/calibration")
import pile_config as pc  # noqa: E402
from _cells_io import load_medias  # noqa: E402

m = load_medias(pc.EMBEDDINGS / "vg_scale__siglip.pkl")
CELLS = [pc.scale_cell(c, b) for c in pc.SCALE_CLASSES for b in pc.BOX_BANDS]
pool = {i for i in m if not m[i].get("categories")}

before = Counter()
after = Counter()
for iid, d in m.items():
    cats = list(d.get("categories") or [])
    exh = bool(d.get("labels_exhaustive"))
    held = {c.split("@")[0] for c in cats}
    if not cats:
        # A media with no categories is a shared negative OR a SPARE, and only
        # the first is evaluable anywhere. The spares (`SCALE_N_NEG_SPARE`, 300
        # when this was written and 1,000 since the #3588 promotion) are drawn
        # into the pickle and designated into no cell, so that retiring a
        # contaminated negative is a relabel rather than a re-embedding pass.
        # Counting them put 4,300 in the `before` column of a cell that holds
        # 4,000, and carried a 300-image error into the whole table -- found when
        # the rebuilt cell was measured against it (#3667). Read the constant,
        # never the number: the error scales with it.
        if not d.get("evaluable_categories"):
            continue
        for cell in CELLS:
            before[cell] += 1
            after[cell] += 1
        continue
    for cell in cats:
        before[cell] += 1
        after[cell] += 1
    if exh:
        for c in pc.SCALE_CLASSES:
            if c not in held:
                for b in pc.BOX_BANDS:
                    after[pc.scale_cell(c, b)] += 1

print(f"{'cell':<22}{'evaluable before':>18}{'after':>9}{'gain':>9}")
print("-" * 60)
gains = []
for c in pc.SCALE_CLASSES:
    for b in pc.BOX_BANDS:
        cell = pc.scale_cell(c, b)
        g = 100 * (after[cell] - before[cell]) / before[cell]
        gains.append(g)
        if b == "medium":
            print(f"{cell:<22}{before[cell]:>18}{after[cell]:>9}{g:>8.1f}%")
print(f"\nmean gain across all {len(gains)} cells: {sum(gains) / len(gains):.1f}%")
print(
    f"negatives per cell: {pc.SCALE_N_NEG} designated -> "
    f"{pc.SCALE_N_NEG + (after[CELLS[0]] - before[CELLS[0]])} evaluable"
)
n_pos = pc.SCALE_N_POS
print(
    f"prevalence: {100 * n_pos / (n_pos + pc.SCALE_N_NEG):.2f}% -> "
    f"{100 * n_pos / (n_pos + pc.SCALE_N_NEG + (after[CELLS[0]] - before[CELLS[0]])):.2f}%"
)
print(
    "\nThis is a PRICE, and it reads `categories` because that is all the pickle\n"
    "carries. `_evaluable` reads the LABELS, and an image can hold a class without\n"
    "being designated a positive for it -- so the built cell gets fewer cross-class\n"
    "negatives than this promises. `cross_class_negatives_rebuilt.py` measures both."
)
