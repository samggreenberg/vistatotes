#!/usr/bin/env python3
"""One dataset, six detectors. The negatives are the same 200 images every time;
only the question changes, and a question is a detector, not a dataset."""

import csv
import json
import shutil
import time
import uuid
from pathlib import Path

D = Path("/exp/sgreenberg/projects/VTSearch/data")
S = Path("/expscratch/sgreenberg/classes-3588/slates")
DATASET = "negative pass 200"

GROUPS = {
    "Table Objects": ("bowl", "cup", "bottle", "vase", "fork", "spoon", "sink", "knife", "chair"),
    "Handheld Objects": ("cell phone", "book", "umbrella", "backpack"),
    "Vehicles": ("car", "truck", "bus", "bicycle"),
    "Street Objects": ("fire hydrant", "stop sign", "clock", "bench"),
    "Outdoor Objects": ("bird", "kite", "boat", "dog"),
}
assert sum(len(v) for v in GROUPS.values()) == 25, sum(len(v) for v in GROUPS.values())

# --- keep ONE dataset, drop the rest ----------------------------------------
dsr = json.loads((D / "dataset_registry.json").read_text())
keep = next(x for x in dsr if x["name"] in ("none of the table 12", "negative pass 200"))
keep["name"] = DATASET
json.dump([keep], (D / "dataset_registry.json").open("w"), indent=1)
print(f"dataset: kept 1 of {len(dsr)}, renamed to {DATASET!r} ({keep.get('file_type_counts')})")

# --- one empty detector per group -------------------------------------------
src_det = json.loads(next((D / "detectors").glob("*.json")).read_text())
for f in (D / "detectors").glob("*.json"):
    f.unlink()

entries = []
for name in GROUPS:
    slug = name.lower().replace(" ", "_")
    det = dict(src_det)
    det["name"] = name
    det["labelset"] = {}
    det["created_at"] = time.time()
    (D / "detectors" / f"{slug}.json").write_text(json.dumps(det, indent=1))
    entries.append(
        {
            "id": uuid.uuid4().hex,
            "name": name,
            "path": f"{slug}.json",
            "media_type": "image",
            "embedder_type": "semantic",
            "created_at": det["created_at"],
        }
    )
json.dump(entries, (D / "detector_registry.json").open("w"), indent=1)
print(f"detectors: {len(entries)} -> {[e['name'] for e in entries]}")

# --- manifests, one per group, over the same rows ---------------------------
base = list(csv.DictReader((S / "Table_Objects" / "manifest.csv").open()))
for name, members in GROUPS.items():
    out = S / name.replace(" ", "_")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in base:
        r = dict(r)
        r["class"], r["detector"] = name, name
        rows.append(r)
    with (out / "manifest.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"  {name:<18} {len(rows):>4} rows, {len(members)} classes: {', '.join(members)}")

for old in ("none_of_the_table_12", "none_of_the_street_7", "none_of_the_outdoors_6"):
    shutil.rmtree(S / old, ignore_errors=True)
print("\nold three-group slate dirs removed")
