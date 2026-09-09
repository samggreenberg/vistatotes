#!/usr/bin/env python3
"""What cut SHOULD each class have had, judged by the reviewer's own verdicts?

The shipped cuts were chosen on the COCO-anchored half. Six classes are now
finished end to end, so for those the real answer is known: every candidate has
a score and a human verdict, and the trade between clicks saved and positives
lost can be read off directly instead of transferred from another stratum.

The recall here is recall *within the slate* -- positives below the shipped cut
were never shown and cannot be counted. So this says what a HIGHER cut would
have cost, which is exactly the live question for the nineteen classes still to
do; it cannot say what the shipped cut already lost. That is #3768's job.
"""

from __future__ import annotations

import json
from pathlib import Path

HR = Path("scripts/experiments/pile/human_record")
SLATES = Path("/expscratch/sgreenberg/vlm-3720/slates/slates.json")
CUTS = [0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]


def main() -> int:
    manifest = json.loads(SLATES.read_text())
    total_saved = total_lost = 0
    for f in sorted(HR.glob("LABELSETS__slate__*.json")):
        doc = json.loads(f.read_text())
        cls = doc["class"]
        verdict = {}
        for side, val in (("good", True), ("bad", False)):
            for r in doc.get(side, []):
                try:
                    verdict[int(str(r["filename"]).split(".")[0])] = val
                except (ValueError, KeyError, TypeError):
                    continue
        scored = {int(r["image_id"]): r["score"] for r in manifest[cls]["rows"]}
        pairs = [(scored[i], v) for i, v in verdict.items() if i in scored]
        if not pairs:
            continue
        npos = sum(1 for _, v in pairs if v)
        shipped = manifest[cls]["cut"]
        print(
            f"\n=== {cls}  (shipped cut {shipped}, {len(pairs)} judged, {npos} positive, "
            f"precision {npos / len(pairs):.2f})"
        )
        print(f"{'cut':>6}{'clicks':>8}{'found':>7}{'missed':>8}{'prec':>7}{'kept':>7}")
        best = None
        for t in CUTS:
            if t < shipped:
                continue
            keep = [(s, v) for s, v in pairs if s >= t]
            found = sum(1 for _, v in keep if v)
            print(
                f"{t:>6.2f}{len(keep):>8}{found:>7}{npos - found:>8}"
                f"{(found / len(keep) if keep else 0):>7.2f}{(found / npos if npos else 0):>7.2f}"
            )
            # the highest cut that still keeps 95% of the positives it was shown
            if npos and found / npos >= 0.95:
                best = (t, len(pairs) - len(keep), npos - found)
        if best:
            t, saved, lost = best
            total_saved += saved
            total_lost += lost
            print(
                f"  -> cut {t} keeps 95% of the positives and saves {saved} clicks "
                f"({100 * saved / len(pairs):.0f}%), losing {lost}"
            )
    print(
        f"\nacross the six finished classes: {total_saved} clicks saved, "
        f"{total_lost} positives lost, had the cuts been set this way"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
