#!/usr/bin/env python3
"""Does the screening's recall depend on object size? (#3720)

The owner's question from the outset was whether a detector can be trusted "when
large" even if not when small, since a few-pixel object is where a model is
weakest and `vg_scale` is banded by size on purpose. Nothing measured so far
answers it: the slates are all above-cut, and the anchored pilot was pooled
across bands.

An image is banded by its **largest** instance of the class, because that is the
one that decides whether the class is findable at all -- an image holding a large
dog and a small dog is an easy image, and calling it `small` would blame the
detector for a miss it did not make.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "scripts/experiments/pile")
import pile_config as pc  # noqa: E402


def band_of(area: float) -> str | None:
    for name, (lo, hi) in pc.BOX_BANDS.items():
        if lo <= area < hi:
            return name
    return None


def main() -> int:
    pc.setup_env()
    sys.path.insert(0, "scripts/experiments/pile/pilebuild")
    import coco_anchor as ca  # noqa: PLC0415

    image_data, instances = ca.ensure_sources(pc.PILE / "coco_anchor", fetch=False)
    truth = ca.coco_truth(instances, set(pc.SCALE_CLASSES))
    with image_data.open() as fh:
        coco_of = {int(m["image_id"]): int(m["coco_id"]) for m in json.load(fh) if m.get("coco_id")}

    dets = {}
    for line in Path("/expscratch/sgreenberg/vlm-3720/owl_pilot.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if "dets" in r:
                dets[int(r["image_id"])] = r["dets"]

    manifest = json.loads(Path("/expscratch/sgreenberg/vlm-3720/slates/slates.json").read_text())

    # (band -> [found, total]) pooled, and per class
    pooled: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    perclass: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for vg_id, dd in dets.items():
        cid = coco_of.get(vg_id)
        if cid is None or cid not in truth:
            continue
        W, H = ca.COCO_DIMS.get(cid, (0, 0))
        if not W or not H:
            continue
        for cls, boxes in truth[cid].items():
            if not boxes:
                continue
            areas = [((b[2] - b[0]) * (b[3] - b[1])) / (W * H) for b in boxes]
            band = band_of(max(areas))
            if band is None:
                continue
            cut = manifest.get(cls, {}).get("cut", 0.1)
            found = any(d["cls"] == cls and d["score"] >= cut for d in dd)
            pooled[band][1] += 1
            pooled[band][0] += int(found)
            perclass[cls][band][1] += 1
            perclass[cls][band][0] += int(found)

    print("OWLv2 recall at each class's own shipped cut, by the band of the largest instance\n")
    print(f"{'band':<10}{'positives':>11}{'found':>8}{'recall':>9}")
    print("-" * 38)
    for band in ("small", "medium", "large"):
        f, n = pooled[band]
        print(f"{band:<10}{n:>11,}{f:>8,}{(f / n if n else 0):>9.2f}")
    tot_f = sum(v[0] for v in pooled.values())
    tot_n = sum(v[1] for v in pooled.values())
    print("-" * 38)
    print(f"{'ALL':<10}{tot_n:>11,}{tot_f:>8,}{(tot_f / tot_n if tot_n else 0):>9.2f}")

    print(f"\n{'class':<13}{'small':>16}{'medium':>16}{'large':>16}")
    print("-" * 61)
    for cls in pc.SCALE_CLASSES:
        cells = []
        for band in ("small", "medium", "large"):
            f, n = perclass[cls][band]
            cells.append(f"{f}/{n} {f / n:.2f}" if n else "     -")
        print(f"{cls:<13}{cells[0]:>16}{cells[1]:>16}{cells[2]:>16}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
