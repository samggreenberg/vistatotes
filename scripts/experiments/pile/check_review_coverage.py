"""Assert that the dataset still contains the images its review was performed on.

Human review is the most expensive input in this pipeline and the only one that
cannot be regenerated, yet nothing about a rebuilt cell reveals whether it still
covers that review. Every structural check keeps passing while coverage
collapses: the cells are full, prevalence is exact, boxes agree with their
bands, patch grids are present. A dataset reviewed at 20% is indistinguishable
from one reviewed completely.

That is not hypothetical -- three rebuilds retired 577 of 743 reviewed images
here before anyone looked (`scripts/experiments/lessons/`). So coverage gets an
assertion of its own, run *before* a rebuild is trusted rather than after it is
noticed.

An image legitimately leaves the pool for two reasons, and neither is held
against coverage:

* **a correction removed it** -- a contaminated negative the review itself
  promoted is *supposed* to disappear;
* **the class list grew past it** -- an image holding `cup` was a sound negative
  when C had twelve classes and cannot be one now that it has 25. The build
  records these in the roster as ``disqualified``, because by the time this
  script runs the reason is no longer recoverable: the image is simply absent.
  Expanding C disqualified 271 reviewed negatives at once (#3588), and without
  this the gate reads a correct rebuild as a 60.8% coverage collapse.

What is still held against coverage is an image that left for neither reason --
a reshuffle, which is the failure this exists to catch.

Usage::

    python check_review_coverage.py                 # report, exit 1 if below threshold
    python check_review_coverage.py --min 0.85
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import pile_config as pc

pc.setup_env()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    base = pc.PILE.parent / "vgscale-3156"
    ap.add_argument("--cell", default=str(pc.EMBEDDINGS / "vg_scale__siglip.pkl"))
    ap.add_argument("--verdicts", default=str(base / "verdicts_20260820b.json"))
    ap.add_argument("--sheets", default=str(base / "sheets_neg"))
    ap.add_argument("--corrections", default=str(pc.PILE / "corrections.json"))
    ap.add_argument("--min", type=float, default=0.85, help="fail below this coverage")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "calibration"))
    from _cells_io import load_medias  # noqa: PLC0415

    medias = load_medias(Path(args.cell))
    negatives = {i for i, m in medias.items() if not m.get("categories")}

    # Disqualified by a class-list change: recorded by the build, in the roster.
    corrected = set()
    if pc.ROSTER.exists():
        corrected |= {int(i) for i in json.loads(pc.ROSTER.read_text()).get("disqualified", [])}
    if Path(args.corrections).exists():
        for c in json.loads(Path(args.corrections).read_text()):
            # A correction that fixes or excludes a negative is *meant* to
            # remove it from the pool; it is progress, not lost coverage.
            corrected.add(int(c["image_id"]))

    verdicts = json.loads(Path(args.verdicts).read_text())
    rows = []

    neg_ids = {v["image_id"] for v in verdicts if v["stratum"] in ("boundary", "random")}
    rows.append(("reviewed negatives", neg_ids, negatives))

    tri: set[int] = set()
    for p in glob.glob(str(Path(args.sheets) / "*" / "index.json")):
        for r in json.loads(Path(p).read_text()):
            tri.add(r["image_id"])
    if tri:
        rows.append(("triaged negatives", tri, negatives))

    pos_pairs = {(v["image_id"], v["class"]) for v in verdicts if v["stratum"] == "positive_boxed"}
    pos_ok = {
        (i, c) for i, c in pos_pairs if any(x.startswith(c + "@") for x in (medias.get(i, {}).get("categories") or []))
    }

    worst = 1.0
    print(f"{'population':<22}{'reviewed':>9}{'still in':>9}{'by fix':>8}{'coverage':>10}")
    print("-" * 58)
    for label, ids, present in rows:
        kept = len(ids & present)
        by_fix = len({i for i in ids - present if i in corrected})
        denom = len(ids) - by_fix
        cov = kept / denom if denom else 1.0
        worst = min(worst, cov)
        print(f"{label:<22}{len(ids):>9}{kept:>9}{by_fix:>8}{cov:>10.1%}")
    cov_pos = len(pos_ok) / len(pos_pairs) if pos_pairs else 1.0
    worst = min(worst, cov_pos)
    print(f"{'reviewed positives':<22}{len(pos_pairs):>9}{len(pos_ok):>9}{'-':>8}{cov_pos:>10.1%}")
    print()
    if worst < args.min:
        print(f"FAIL: coverage {worst:.1%} is below --min {args.min:.0%}.")
        print("The rebuild retired images that were reviewed. Check the roster before trusting this cell.")
        return 1
    print(f"OK: every reviewed population is at or above {args.min:.0%}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
