#!/usr/bin/env python3
"""Why did the 25-class rebuild retire reviewed negatives? (verify's 60.8%)

`check_review_coverage.py` forgives exactly one reason for an image leaving the
pool: a correction removed it. It knows nothing about a CLASS LIST change -- and
an image that was a sound negative under twelve classes is correctly disqualified
under 25 the moment it turns out to hold a `cup`. That is progress, not lost
coverage, but the gate cannot tell the two apart and fails the build.

So attribute every retirement before deciding whether the gate or the build is
wrong. Three causes are possible and they mean different things:

* **holds a new class** -- legitimate, and the expansion working as intended;
* **suppressed by an ambiguous spelling** -- legitimate but #3655's global
  exclusion, which costs every class the image for one class's ambiguity;
* **neither** -- unexplained, and the only kind that should worry anyone.
"""

from __future__ import annotations

import json
import sys
from collections import Counter

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

ORIGINAL_12 = {
    "clock",
    "bird",
    "boat",
    "umbrella",
    "kite",
    "book",
    "dog",
    "backpack",
    "knife",
    "bicycle",
    "bus",
    "stop sign",
}
NEW_13 = set(pc.SCALE_CLASSES) - ORIGINAL_12


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

    labels = read_vg_labels(records, paths, dims, pc.scale_vg_wanted())
    box_dims, exhaustive, _a, _r = anchor_to_coco(labels, dims, coco_of, truth, ca.COCO_DIMS, wanted)
    canonicalise(labels, pc.SCALE_VG_NAMES, box_dims, pc.SCALE_FOLD_MODE)
    unbanded = apply_corrections(labels, corrections, box_dims, exhaustive)
    suppressed = lift_ambiguous(labels, pc.SCALE_VG_AMBIGUOUS, exhaustive)

    # (a) the pool as built, and (b) the counterfactual where the ambiguous
    # SUPPRESSION is dropped but the same names are still cleaned out of labels.
    # Passing an empty table instead would leave `books` in labels and crash
    # band_candidates, which has no cell for it -- the bug in the first attempt.
    _s, _b, clean_shipped = band_candidates(labels, box_dims, unbanded | suppressed)
    _s2, _b2, clean_no_suppress = band_candidates(labels, box_dims, unbanded)
    clean_shipped, clean_no_suppress = set(clean_shipped), set(clean_no_suppress)

    print(f"classes {len(pc.SCALE_CLASSES)}, ambiguous names {sum(len(v) for v in pc.SCALE_VG_AMBIGUOUS.values())}")
    print(
        f"#3655 -- clean pool with global suppression {len(clean_shipped)}, "
        f"without {len(clean_no_suppress)}, "
        f"withheld {len(clean_no_suppress - clean_shipped)} "
        f"({100 * len(clean_no_suppress - clean_shipped) / max(1, len(clean_no_suppress)):.1f}%)"
    )

    base = pc.PILE.parent / "vgscale-3156"
    rows = json.loads((base / "verdicts_20260820b.json").read_text())
    reviewed_neg = {int(r["image_id"]) for r in rows if r.get("human") == "absent"}
    retired = sorted(reviewed_neg - clean_shipped)
    print(f"\nreviewed negatives {len(reviewed_neg)}; retired from the clean pool {len(retired)}")

    why = Counter()
    detail = Counter()
    supp_by_img = {}
    for iid, cls in suppressed:
        supp_by_img.setdefault(iid, []).append(cls)
    for iid in retired:
        held = set(labels.get(iid, {}))
        new = held & NEW_13
        if new:
            why["holds a NEW class"] += 1
            for c in new:
                detail[c] += 1
        elif held & ORIGINAL_12:
            why["holds an original class"] += 1
        elif iid in supp_by_img:
            why["ambiguous suppression (#3655)"] += 1
        elif any(k[0] == iid for k in corrections):
            why["a correction removed it"] += 1
        else:
            why["UNEXPLAINED"] += 1
    for k, v in why.most_common():
        print(f"  {k:32s} {v:5d}  ({100 * v / max(1, len(retired)):.0f}%)")
    print("  new classes responsible:", dict(detail.most_common(8)))


if __name__ == "__main__":
    main()
