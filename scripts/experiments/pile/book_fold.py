#!/usr/bin/env python3
"""What is actually inside COCO's `book`? Fold-in for the printed-matter words."""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "scripts/experiments/pile")
import pile_config as pc  # noqa: E402
from coco_anchor import coco_truth, ensure_sources  # noqa: E402
from pilebuild.loaders.vg_scale import read_vg_labels, vg_source  # noqa: E402

WORDS = (
    "book",
    "books",
    "magazine",
    "magazines",
    "newspaper",
    "newspapers",
    "paper",
    "papers",
    "novel",
    "textbook",
    "notebook",
    "comic",
    "journal",
    "bible",
    "dictionary",
    "menu",
    "catalog",
    "brochure",
    "pamphlet",
    "manual",
    "album",
    "binder",
    "folder",
    "document",
    "letter",
    "card",
    "poster",
    "flyer",
    "leaflet",
    "print",
)

paths, records, dims = vg_source()
labels = read_vg_labels(records, paths, dims, set(WORDS))
anchor = Path(pc.PILE / "coco_anchor")
image_data, instances = ensure_sources(anchor, False)
truth = coco_truth(instances, {"book"})
meta = json.loads(Path(image_data).read_text())
coco_of = {int(x["image_id"]): int(x["coco_id"]) for x in meta if x.get("coco_id")}


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    i = (x1 - x0) * (y1 - y0)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


on_book, total = Counter(), Counter()
for iid, byname in labels.items():
    cid = coco_of.get(iid)
    if cid is None:
        continue
    bks = truth.get(cid, {}).get("book") or []
    for name, boxes in byname.items():
        for bx in boxes:
            total[name] += 1
            if any(iou(bx, cb) >= 0.5 for cb in bks):
                on_book[name] += 1

print(f"{'VG name':<14}{'boxes on a COCO images':>24}{'land on a book box':>21}{'rate':>8}")
print("-" * 68)
for w in WORDS:
    if total[w] < 4:
        continue
    print(f"{w:<14}{total[w]:>24}{on_book[w]:>21}{100 * on_book[w] / total[w]:>7.0f}%")
