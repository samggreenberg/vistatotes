#!/usr/bin/env python3
"""Can the COCO-anchored half alone supply 300 positives per class per band?

That is the question #3668 turns on. If yes, the whole benchmark can be made
provable -- positives and negatives both -- and the unprovable half of the pool
is simply dropped. If no, both halves must stay and the non-COCO pool's error
rate becomes load-bearing.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts/experiments/pile")
import pile_config as pc  # noqa: E402
import coco_anchor  # noqa: E402
from coco_anchor import coco_truth, ensure_sources  # noqa: E402
from make_class_slate import canonicalise  # noqa: E402
from pilebuild.loaders.vg_scale import (  # noqa: E402
    anchor_to_coco,
    band_candidates,
    read_vg_labels,
    vg_source,
)

NEED = 300
GROUPS = {
    "shipped": (list(pc.SCALE_CLASSES_ORIGINAL), pc.SCALE_VG_NAMES),
    "candidate": (list(pc.SCALE_CANDIDATES_3588), pc.SCALE_VG_NAMES),
}

paths, records, dims = vg_source()
anchor = Path(pc.PILE / "coco_anchor")
image_data, instances = ensure_sources(anchor, False)
with image_data.open() as fh:
    meta = json.load(fh)
coco_of = {int(m["image_id"]): int(m["coco_id"]) for m in meta if m.get("coco_id")}

for label, (classes, table) in GROUPS.items():
    vg_names = {c: table.get(c, (c,)) for c in classes}
    wanted = {n for v in vg_names.values() for n in v}
    labels = read_vg_labels(records, paths, dims, wanted)
    canonicalise(labels, dict(vg_names))
    truth = coco_truth(instances, set(classes))
    box_dims, exhaustive, *_ = anchor_to_coco(labels, dims, coco_of, truth, coco_anchor.COCO_DIMS, set(classes))
    supply, _boxes, _clean = band_candidates(labels, box_dims, set(), classes=classes)

    print(f"\n=== {label} ===")
    print(f"{'class':<13}" + "".join(f"{b + ' all/exact':>18}" for b in pc.BOX_BANDS))
    short = []
    for c in classes:
        cells = []
        for b in pc.BOX_BANDS:
            ids = supply[c][b]
            ex = sum(1 for i in ids if i in exhaustive)
            cells.append(f"{len(ids)}/{ex}")
            if ex < NEED:
                short.append(f"{c}@{b}={ex}")
        print(f"{c:<13}" + "".join(f"{x:>18}" for x in cells))
    print(f"  cells short of {NEED} on the COCO half: {len(short)}")
    if short:
        print("   ", ", ".join(short[:12]))
