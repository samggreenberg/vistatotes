#!/usr/bin/env python3
"""Re-derive per-class cuts from a large anchored sample, and validate them (#3720).

The shipped cuts came from 292 anchored images, which gave `stop sign` 24
positives to choose a 95%-recall threshold from -- and the reviewer's own
verdicts later showed that cut was worth 135 wasted clicks. This re-derives every
cut from 3,595 anchored images (`stop sign`: 270 positives) and then does the
thing that makes it more than a guess: for the six classes finished end to end,
it checks the new cut against the optimum the human verdicts actually revealed.

A re-cut is only proposed for the nineteen unfinished classes. The six finished
ones are left alone -- their slates are already judged, and moving a cut under a
completed review would silently change what the verdicts mean.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

BIG = Path("/expscratch/sgreenberg/vlm-3720/owl_big.jsonl")
SLATES = Path("/expscratch/sgreenberg/vlm-3720/slates/slates.json")
HR = Path("scripts/experiments/pile/human_record")
CUTS = [0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
TARGET = 0.95


def derive(rows, classes):
    out = {}
    for c in classes:
        best = CUTS[0]
        for t in sorted(CUTS, reverse=True):
            tp = fn = 0
            for r in rows:
                truth = c in set(r["truth"])
                pred = any(d["cls"] == c and d["score"] >= t for d in r["dets"])
                if truth and pred:
                    tp += 1
                elif truth:
                    fn += 1
            if (tp / (tp + fn) if tp + fn else 1.0) >= TARGET:
                best = t
                break
        out[c] = best
    return out


def real_optimum(cls, manifest):
    f = HR / f"LABELSETS__slate__{cls.replace(' ', '_')}.json"
    if not f.exists():
        return None
    doc = json.loads(f.read_text())
    verdict = {}
    for side, val in (("good", True), ("bad", False)):
        for r in doc.get(side, []):
            try:
                verdict[int(str(r["filename"]).split(".")[0])] = val
            except (ValueError, KeyError, TypeError):
                continue
    scored = {int(r["image_id"]): r["score"] for r in manifest[cls]["rows"]}
    pairs = [(scored[i], v) for i, v in verdict.items() if i in scored]
    npos = sum(1 for _, v in pairs if v)
    if not npos:
        return None
    best = manifest[cls]["cut"]
    for t in CUTS:
        if t < manifest[cls]["cut"]:
            continue
        found = sum(1 for s, v in pairs if v and s >= t)
        if found / npos >= TARGET:
            best = t
    return best


def main() -> int:
    import sys

    sys.path.insert(0, "scripts/experiments/pile")
    import pile_config as pc  # noqa: PLC0415

    classes = list(pc.SCALE_CLASSES)
    rows = [json.loads(x) for x in BIG.read_text().splitlines() if x.strip()]
    rows = [r for r in rows if "dets" in r]
    npos = defaultdict(int)
    for r in rows:
        for c in r["truth"]:
            npos[c] += 1
    print(f"{len(rows)} anchored images, {sum(npos.values())} positives\n")

    manifest = json.loads(SLATES.read_text())
    new = derive(rows, classes)

    print("VALIDATION -- the six classes with human verdicts")
    print(f"{'class':<13}{'n_anch':>8}{'shipped':>9}{'big-sample':>12}{'human says':>12}{'':>4}")
    print("-" * 58)
    hits = tot = 0
    for c in classes:
        opt = real_optimum(c, manifest)
        if opt is None:
            continue
        tot += 1
        ok = new[c] == opt
        hits += ok
        print(
            f"{c:<13}{npos[c]:>8}{manifest[c]['cut']:>9.2f}{new[c]:>12.2f}{opt:>12.2f}{'  ok' if ok else '  MISS':>6}"
        )
    print("-" * 58)
    print(f"the larger sample reproduces the human optimum on {hits} of {tot}\n")

    print("PROPOSED for the nineteen unfinished classes")
    print(f"{'class':<13}{'n_anch':>8}{'shipped':>9}{'proposed':>10}{'slate now':>11}")
    print("-" * 52)
    for c in classes:
        if real_optimum(c, manifest) is not None:
            continue
        n = manifest[c]["n"]
        mark = "" if new[c] == manifest[c]["cut"] else "   <-"
        print(f"{c:<13}{npos[c]:>8}{manifest[c]['cut']:>9.2f}{new[c]:>10.2f}{n:>11,}{mark}")
    Path("/expscratch/sgreenberg/vlm-3720/recut.json").write_text(
        json.dumps({"target": TARGET, "cuts": new}, indent=1) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
