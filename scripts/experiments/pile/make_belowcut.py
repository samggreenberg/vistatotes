#!/usr/bin/env python3
"""Sampled slates from BELOW each class's cut, to measure what screening lost (#3768).

Every finished slate measures precision and none of them can measure recall,
because a slate contains only what cleared the cut. This draws the complement:
queue images the detector scored *below* the class's cut, sampled at random.

Three things make it a different question from the slate's, and the build has to
respect all three:

* **no box.** There is no detection to confirm, so the question becomes "is there
  a `<class>` anywhere in this image?" -- a wider question than "is this box a
  `<class>`?", and pooling the two would silently mix them (#3612). Separate
  dataset, separate detector, separate name.
* **excluded by SCORE only.** An image kept out of a slate for being a known A,
  or for having been answered already, was not the screening's decision and
  says nothing about its recall.
* **random, not ranked.** A ranked sample would re-introduce exactly the
  selection bias that made the earlier 1,223-verdict comparison untrustworthy.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

FINISHED = ["bicycle", "boat", "dog", "fire hydrant", "sink", "stop sign"]
SAMPLE = 100
SEED = 3768


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dets", default="/expscratch/sgreenberg/vlm-3720/owl_queue.jsonl")
    ap.add_argument("--slates", default="/expscratch/sgreenberg/vlm-3720/slates")
    ap.add_argument("--queue", default="/expscratch/sgreenberg/vgscale-3156/annotation_queue.jsonl")
    ap.add_argument("--out", default="/expscratch/sgreenberg/vlm-3720/belowcut")
    ap.add_argument("--n", type=int, default=SAMPLE)
    args = ap.parse_args()

    manifest = json.loads((Path(args.slates) / "slates.json").read_text())
    known: dict[str, set[int]] = defaultdict(set)
    paths: dict[int, str] = {}
    for line in Path(args.queue).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            paths[int(r["image_id"])] = r["path"]
            for c in r["classes"]:
                known[c].add(int(r["image_id"]))

    # images already answered anywhere, so the sample never re-asks
    answered: dict[str, set[int]] = defaultdict(set)
    hr = Path("scripts/experiments/pile/human_record")
    for f in list(hr.glob("LABELSETS__slate__*.json")) + list(hr.glob("LABELSETS__pass25__*.json")):
        doc = json.loads(f.read_text())
        cls = doc.get("class", "")
        for side in ("good", "bad"):
            for row in doc.get(side, []):
                try:
                    answered[cls].add(int(str(row["filename"]).split(".")[0]))
                except (ValueError, KeyError, TypeError):
                    continue

    dets = [json.loads(x) for x in Path(args.dets).read_text().splitlines() if x.strip()]
    dets = [r for r in dets if "dets" in r]

    rng = random.Random(SEED)
    out = Path(args.out)
    summary = {}
    print(f"{'class':<13}{'cut':>6}{'below':>8}{'eligible':>10}{'sampled':>9}")
    print("-" * 47)
    for cls in FINISHED:
        cut = manifest[cls]["cut"]
        below = []
        for r in dets:
            iid = int(r["image_id"])
            top = max((d["score"] for d in r["dets"] if d["cls"] == cls), default=0.0)
            if top >= cut:
                continue  # cleared the cut: that is the slate
            if iid in known[cls] or iid in answered[cls]:
                continue  # not the screening's decision
            below.append(iid)
        pick = sorted(rng.sample(sorted(below), min(args.n, len(below))))
        d = out / cls.replace(" ", "_") / "images"
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
        for iid in pick:
            (d / f"{iid}.jpg").symlink_to(paths[iid])
        summary[cls] = {"cut": cut, "below": len(below), "sampled": len(pick), "ids": pick}
        print(f"{cls:<13}{cut:>6.2f}{len(below):>8,}{len(below):>10,}{len(pick):>9}")
    (out / "belowcut.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(f"\nwrote {out}/belowcut.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
