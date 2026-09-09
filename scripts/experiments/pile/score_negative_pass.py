#!/usr/bin/env python3
"""Score a finished negative-pass group against COCO.

FIXES A BUG IN THE MANIFEST. `make_negative_slate.py` wrote

    "reference": "present" if medias[i].get("labels_exhaustive") else ""

which records only WHETHER an image is scored, not what the answer is -- every
scored row claimed "clean". The real reference is per-group and comes from COCO:
an image is clean for a group when COCO, which annotates all eighty of its
classes on any image it annotates, lists none of that group's members.
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "scripts/experiments/pile")
sys.path.insert(0, "scripts/experiments/calibration")
import pile_config as pc  # noqa: E402
from _cells_io import load_medias  # noqa: E402
from coco_anchor import coco_truth, ensure_sources  # noqa: E402

DETS = Path("/expscratch/sgreenberg/classes-3588/negbank")
GROUPS = {
    "Vehicles": ("car", "truck", "bus", "bicycle"),
    "Outdoor Objects": ("bird", "kite", "boat", "dog"),
    "Bench": ("bench",),
}
ALL = list(pc.SCALE_CANDIDATES_3588) + list(pc.SCALE_CLASSES_ORIGINAL)

m = load_medias(pc.EMBEDDINGS / "vg_scale__siglip.pkl")
anchor = Path(pc.PILE / "coco_anchor")
image_data, instances = ensure_sources(anchor, False)
truth = coco_truth(instances, set(ALL))
meta = json.loads(Path(image_data).read_text())
coco_of = {int(x["image_id"]): int(x["coco_id"]) for x in meta if x.get("coco_id")}


def coco_present(iid):
    cid = coco_of.get(iid)
    if cid is None or not m.get(iid, {}).get("labels_exhaustive"):
        return None
    return {c for c, b in truth.get(cid, {}).items() if b}


for name, members in GROUPS.items():
    slug = name.lower().replace(" ", "_")
    labels = json.loads((DETS / f"{slug}.json").read_text())["labelset"]["labels"]
    said_clean, said_dirty, scored, agree = 0, 0, 0, 0
    misses, false_alarms = [], []
    for lb in labels:
        iid = int(Path(lb["origin_name"]).stem)
        human_clean = lb.get("label") == "good"
        said_clean += human_clean
        said_dirty += not human_clean
        p = coco_present(iid)
        if p is None:
            continue
        scored += 1
        ref_clean = not (p & set(members))
        if ref_clean == human_clean:
            agree += 1
        elif human_clean:
            misses.append((iid, sorted(p & set(members))))
        else:
            false_alarms.append(iid)
    n = len(labels)
    print(f"\n=== {name} ({', '.join(members)}) ===")
    print(f"  {n} labelled: {said_dirty} NOT clean ({100 * said_dirty / n:.1f}%), {said_clean} clean")
    if scored:
        print(f"  scored against COCO: {scored} rows, agreement {100 * agree / scored:.1f}%")
        print(f"    COCO saw one and the reviewer did not : {len(misses)}")
        print(f"    reviewer saw one and COCO did not     : {len(false_alarms)}")
        if misses:
            print("    missed:", Counter(c for _i, cs in misses for c in cs).most_common())
    found = [int(Path(lb["origin_name"]).stem) for lb in labels if lb.get("label") != "good"]
    print(f"  images the reviewer FOUND one in: {found}")
