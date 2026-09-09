#!/usr/bin/env python3
"""Stratified sample of COCO-answered images, with truth, for scoring a VLM.

The anchored half is free ground truth: COCO annotated all eighty of its classes
on these images at once, so one image is a labelled example for all 25 of ours --
a positive for the classes it holds and a *negative* for every class it does not.
That is what makes precision and recall both measurable from one pass.

Sampled per class so the rare classes (`fire hydrant` at 1.49% prevalence) carry
enough positives to say anything about, rather than falling out of a uniform draw.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "scripts/experiments/pile")
import pile_config as pc  # noqa: E402


def main() -> int:
    pc.setup_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=12)
    ap.add_argument("--seed", type=int, default=3720)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cell", default=str(pc.EMBEDDINGS / "vg_scale__siglip.pkl"))
    args = ap.parse_args()

    sys.path.insert(0, "scripts/experiments/calibration")
    from _cells_io import load_medias  # noqa: PLC0415

    from pilebuild.audit import coco_held_by  # noqa: PLC0415

    medias = load_medias(Path(args.cell))
    held = coco_held_by()
    C = list(pc.SCALE_CLASSES)
    cset = set(C)

    # Only images we can actually open, and that COCO answered for.
    usable = {
        iid: m for iid, m in medias.items() if iid in held and m.get("origin_name") and Path(m["origin_name"]).exists()
    }
    by_class: dict[str, list[int]] = defaultdict(list)
    for iid in usable:
        for c in held[iid]:
            if c in cset:
                by_class[c].append(iid)

    rng = random.Random(args.seed)
    chosen: dict[int, None] = {}
    for c in C:
        pool = sorted(by_class.get(c, []))
        for iid in rng.sample(pool, min(args.per_class, len(pool))):
            chosen.setdefault(iid, None)

    rows = []
    for iid in sorted(chosen):
        truth = sorted(c for c in held[iid] if c in cset)
        rows.append(
            {
                "image_id": iid,
                "path": usable[iid]["origin_name"],
                "truth": truth,
            }
        )
    Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in rows))

    pos = Counter(c for r in rows for c in r["truth"])
    print(f"[sample] {len(rows)} images -> {args.out}")
    print(f"[sample] {len(rows) * len(C)} image-class pairs, {sum(pos.values())} of them positive")
    thin = [c for c in C if pos[c] < args.per_class]
    if thin:
        print(f"[sample] fewer than {args.per_class} positives for: {', '.join(thin)}")
    print("[sample] positives per class:", dict(pos.most_common()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
