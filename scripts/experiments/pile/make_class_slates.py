#!/usr/bin/env python3
"""Per-class review slates of over-flagged, pre-boxed candidates (#3720).

**One dataset per class, not one shared set.** The pass asks "which of *C* is
present?" per image, but a reviewer holds one class in mind at a time, and a
class's slate need not contain the images already *designated* for it -- those
are the known As, and the question there was already answered by the
designation. So each slate is that class's candidates minus its own known As.

**Over-flagged on purpose.** The cut per class is the lowest that still recalls
~95% of COCO-confirmed positives on the labelled 292, not the cut that maximises
F1. That is the reviewer's own economics: a false flag costs one fast Bad click,
while a missed positive is invisible forever, because an image nobody is shown
is an image nobody can correct.

**Pre-boxed.** Every candidate carries the detector's box burned in with a
magnified inset, so a verdict is a single click rather than a drawn rectangle.
Bands are a downstream question and the owner has ruled them second-order to
class correctness, so a loose box costs a stratum, not a label.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

TARGET_RECALL = 0.95
CUTS = [0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]


def per_class_cuts(pilot: Path, classes: list[str]) -> dict[str, float]:
    """Lowest cut per class that still recalls TARGET_RECALL on the labelled set.

    Per class rather than one global number because precision varies by nearly
    5x across these classes at a fixed cut (`car` 0.81, `bench` 0.17), so a
    single cut either drowns the reviewer in the easy classes or loses the hard
    ones.
    """
    rows = [json.loads(x) for x in pilot.read_text().splitlines() if x.strip()]
    rows = [r for r in rows if "dets" in r]
    out: dict[str, float] = {}
    for c in classes:
        best = CUTS[0]
        for t in sorted(CUTS, reverse=True):
            tp = fn = 0
            for r in rows:
                truth, pred = c in set(r["truth"]), any(d["cls"] == c and d["score"] >= t for d in r["dets"])
                if truth and pred:
                    tp += 1
                elif truth:
                    fn += 1
            rec = tp / (tp + fn) if tp + fn else 1.0
            if rec >= TARGET_RECALL:
                best = t
                break
        out[c] = best
    return out


def _render(job: tuple[str, str, list[float]]) -> str | None:
    src, dest, box = job
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from make_positive_slate import draw_with_inset  # noqa: PLC0415

    try:
        draw_with_inset(Path(src), tuple(box), Path(dest))
    except Exception as exc:  # noqa: BLE001 - report, never sink the batch
        return f"{src}: {exc}"
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dets", required=True, help="owl_queue.jsonl")
    ap.add_argument("--pilot", required=True, help="owl_pilot.jsonl (labelled, for the cuts)")
    ap.add_argument("--queue", required=True, help="annotation_queue.jsonl (for the known As)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--answered", default="", help="banked_labels.json: pairs already judged")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, "scripts/experiments/pile")
    import pile_config as pc  # noqa: PLC0415

    classes = list(pc.SCALE_CLASSES)

    # Images the reviewer has already ruled on for a class, under the earlier
    # one-dataset pass. A verdict is a verdict whatever slate it was cast in, so
    # re-asking is duplicated labour -- the precise complaint that reshaped this
    # pass. Keyed by class, so a bicycle verdict only excuses the bicycle slate.
    answered: dict[str, set[int]] = defaultdict(set)
    if args.answered and Path(args.answered).exists():
        NAME2CLS = {"bicycle incl trikes not motorcycles": "bicycle", "bench not chairs": "bench"}
        banked = json.loads(Path(args.answered).read_text())
        for det_name, body in banked.items():
            cls = NAME2CLS.get(det_name)
            if not cls:
                continue
            for side in ("good", "bad"):
                for row in body.get("labels-detail", {}).get(side, []):
                    try:
                        answered[cls].add(int(str(row["filename"]).split(".")[0]))
                    except (ValueError, KeyError, TypeError):
                        continue

    known: dict[str, set[int]] = defaultdict(set)
    for line in Path(args.queue).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            for c in r["classes"]:
                known[c].add(int(r["image_id"]))

    cuts = per_class_cuts(Path(args.pilot), classes)
    dets = [json.loads(x) for x in Path(args.dets).read_text().splitlines() if x.strip()]
    dets = [r for r in dets if "dets" in r]

    slates: dict[str, list[dict]] = defaultdict(list)
    for r in dets:
        iid = int(r["image_id"])
        best: dict[str, dict] = {}
        for d in r["dets"]:
            c = d["cls"]
            if d["score"] < cuts[c] or iid in known[c] or iid in answered[c]:
                continue
            if c not in best or d["score"] > best[c]["score"]:
                best[c] = d
        for c, d in best.items():
            slates[c].append({"image_id": iid, "path": r["path"], "score": d["score"], "box": d["box"]})

    total = sum(len(v) for v in slates.values())
    print(f"{'class':<13}{'cut':>6}{'candidates':>12}{'known As':>10}{'answered':>10}")
    print("-" * 51)
    for c in classes:
        print(f"{c:<13}{cuts[c]:>6.2f}{len(slates[c]):>12,}{len(known[c]):>10}{len(answered[c]):>10}")
    print("-" * 51)
    print(f"{'TOTAL':<13}{'':>6}{total:>12,}")
    print(
        f"\n{total:,} pre-boxed candidates vs 81,363 image-views for the exhaustive sweep ({100 * total / 81363:.1f}%)"
    )
    if args.dry_run:
        return 0

    out = Path(args.out)
    jobs, manifest = [], {}
    for c in classes:
        d = out / c.replace(" ", "_") / "images"
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
        rows = sorted(slates[c], key=lambda r: -r["score"])
        for r in rows:
            dest = d / f"{r['image_id']}.jpg"
            jobs.append((r["path"], str(dest), r["box"]))
        manifest[c] = {"cut": cuts[c], "n": len(rows), "rows": rows}
    (out / "slates.json").write_text(json.dumps(manifest, indent=1) + "\n")

    print(f"\nrendering {len(jobs):,} boxed images with {args.workers} workers ...")
    errs = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, err in enumerate(ex.map(_render, jobs, chunksize=32), 1):
            if err:
                errs += 1
                if errs <= 5:
                    print("  ", err)
            if i % 2000 == 0:
                print(f"  {i:,}/{len(jobs):,}", flush=True)
    print(f"done; {errs} failed renders")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
