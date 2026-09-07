#!/usr/bin/env python3
"""What does the GLOBAL ambiguous exclusion cost the shared negative pool? (#3655)

`lift_ambiguous` suppresses an `(image, class)` pair, and `band_candidates` then
keeps that image out of **every** class's bands *and* out of the shared clean
pool -- so one ambiguous spelling for one class costs the image to all 25. #3655
observes that `evaluable_categories` could make it cost exactly one, and measured
2,200-3,941 pool images per class when there were twelve classes and ~82
ambiguous names.

There are now **25 classes and 160 ambiguous names**, so that figure is stale in
the direction that matters. This re-measures it, because the number decides
whether a 400-per-class review slate is drawn from the right population: an image
wrongly outside the pool can never be sampled into the pass, and no amount of
reviewing finds it.

Method: run the loader's own passes twice over the real source -- once as shipped,
once with `SCALE_VG_AMBIGUOUS` emptied -- and diff the clean pool. Attribution is
per suppressing class, so the cost can be read as "images class X's spellings
withheld from the other 24".
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "scripts/experiments/pile")
sys.path.insert(0, "scripts/experiments/calibration")

import pile_config as pc  # noqa: E402
from pilebuild.corrections import load_corrections  # noqa: E402
from pilebuild.loaders.vg_scale import (  # noqa: E402
    anchor_to_coco,
    apply_corrections,
    band_candidates,
    canonicalise,
    lift_ambiguous,
    read_vg_labels,
)
from pilebuild.vgsource import vg_image_paths, vg_source  # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("ambig_cost.json")


def main() -> None:
    import coco_anchor as ca

    wanted = set(pc.SCALE_CLASSES)
    paths = vg_image_paths()
    _, records, dims = vg_source()
    image_data, instances = ca.ensure_sources(pc.PILE / "coco_anchor", fetch=False)
    truth = ca.coco_truth(instances, wanted)
    with image_data.open() as fh:
        coco_of = {int(m["image_id"]): int(m["coco_id"]) for m in json.load(fh) if m.get("coco_id")}
    corrections = load_corrections()

    def clean_pool(ambiguous: dict) -> tuple[set[int], set[tuple[int, str]]]:
        labels = read_vg_labels(records, paths, dims, pc.scale_vg_wanted())
        box_dims, exhaustive, _a, _r = anchor_to_coco(labels, dims, coco_of, truth, ca.COCO_DIMS, wanted)
        canonicalise(labels, pc.SCALE_VG_NAMES, box_dims, pc.SCALE_FOLD_MODE)
        unbanded = apply_corrections(labels, corrections, box_dims, exhaustive)
        suppressed = lift_ambiguous(labels, ambiguous, exhaustive)
        unbanded |= suppressed
        _s, _b, clean = band_candidates(labels, box_dims, unbanded)
        return set(clean), suppressed

    shipped, suppressed = clean_pool(pc.SCALE_VG_AMBIGUOUS)
    none_, _ = clean_pool({})

    lost = none_ - shipped
    by_class = Counter(c for iid, c in suppressed if iid in lost)
    report = {
        "classes": len(pc.SCALE_CLASSES),
        "ambiguous_names": sum(len(v) for v in pc.SCALE_VG_AMBIGUOUS.values()),
        "clean_with_exclusion": len(shipped),
        "clean_without_exclusion": len(none_),
        "images_withheld_from_the_pool": len(lost),
        "share_of_pool_lost": round(len(lost) / len(none_), 4) if none_ else None,
        # Each of these is an image the suppressing class had a reason to
        # withhold and the OTHER 24 did not.
        "withheld_by_class": dict(by_class.most_common()),
    }
    OUT.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "withheld_by_class"}, indent=1))
    print("\nwithheld by the suppressing class (cost borne by the other 24):")
    for c, n in by_class.most_common():
        print(f"  {c:14s} {n:6d}")


if __name__ == "__main__":
    main()
