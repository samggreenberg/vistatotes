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

**A declared composition change is the second such exit, and it is much
larger** (#3670). When the pool is drawn from the COCO-scored half alone, an
off-COCO image is no longer *eligible* to be in it -- so 558 of the 743 reviewed
negatives here left the pool by a rule written down in `pile_config`, not by a
draw that reshuffled. Holding those against coverage reports 22% and reads as a
catastrophe; ignoring them silently would let a real reshuffle hide behind the
same rule. So they are excluded from the denominator and **printed as their own
column**, and a reviewed image the composition *can* hold still counts against
coverage if it went missing. The distinction is the check:

* left because the rule cannot hold it -> ``by rule``, not a failure;
* left for any other reason -> counted, and the gate fires.

Eligibility is read from the COCO pairing (``coco_anchor/image_data.json``) --
the same fact the build stratifies on, and deliberately not ``labels_exhaustive``,
which a one-class review also sets. It is a slight *superset* of what the build
admits, because it does not re-apply the aspect-drift filter that drops 49 of
51,497 pairs. That errs toward calling an image eligible, i.e. toward a *larger*
denominator and a *lower* coverage, which is the safe direction for a gate.

Usage::

    python check_review_coverage.py                 # report, exit 1 if below threshold
    python check_review_coverage.py --min 0.85
    python check_review_coverage.py --composition matched
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections.abc import Callable
from pathlib import Path

import pile_config as pc

#: Below this many eligible images a coverage figure is not worth quoting. The
#: `by rule` column can legitimately excuse most of a review, and a gate that
#: reports 100% off a handful of survivors is worse than one that says it cannot
#: tell -- it looks like evidence.
MIN_DENOM = 50


def eligible_under(composition: str) -> Callable[[int], bool]:
    """``image_id -> can this composition hold it as a designated negative?``

    Only ``provable`` restricts the frame; ``matched`` draws from both halves in
    a ratio, so every clean image stays eligible and no reviewed image is ever
    excused by the rule.

    The COCO pairing is read from ``coco_anchor/image_data.json`` -- the same
    file the loader reads -- rather than from the cell, because the images this
    has to classify are the ones that are NOT in the cell any more. Reading
    ``labels_exhaustive`` off the pickle can only ever describe the survivors,
    which is precisely the population the question is not about.
    """
    if composition != "provable":
        return lambda _iid: True
    image_data = pc.PILE / "coco_anchor" / "image_data.json"
    if not image_data.exists():
        raise SystemExit(f"missing {image_data}; run coco_anchor.py --fetch, or pass --composition matched")
    paired = {int(m["image_id"]) for m in json.loads(image_data.read_text()) if m.get("coco_id")}
    return lambda iid: iid in paired


def coverage_row(
    ids: set[int], present: set[int], corrected: set[int], eligible: Callable[[int], bool]
) -> tuple[int, int, int, int, float]:
    """``(by_rule, by_fix, denominator, kept, coverage)`` for one reviewed population.

    Three ways out of the pool, and only one of them is a failure:

    * **by rule** -- the declared composition cannot hold this image at all, so
      it was never a candidate for the new pool. Excluded from the denominator.
    * **by fix** -- a correction removed it. That is the review working, and it
      has never counted against coverage.
    * anything else -- the draw lost an image it could have kept. This is the
      only exit the gate exists to catch, and it is what the denominator holds.

    Order matters: `by rule` is applied first, so an image that is both
    ineligible and corrected is counted once, as ineligible. Counting it twice
    would inflate the denominator and understate coverage.
    """
    can_hold = {i for i in ids if eligible(i)}
    by_rule = len(ids) - len(can_hold)
    kept = len(can_hold & present)
    by_fix = len({i for i in can_hold - present if i in corrected})
    denom = len(can_hold) - by_fix
    return by_rule, by_fix, denom, kept, (kept / denom if denom else 1.0)


def main() -> int:
    # Called here rather than at import, unlike this directory's other scripts:
    # `setup_env` rewrites the import machinery process-wide, and the arithmetic
    # above is worth a unit test that does not pay for that.
    pc.setup_env()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    base = pc.PILE.parent / "vgscale-3156"
    ap.add_argument("--cell", default=str(pc.EMBEDDINGS / "vg_scale__siglip.pkl"))
    ap.add_argument("--verdicts", default=str(base / "verdicts_20260820b.json"))
    ap.add_argument("--sheets", default=str(base / "sheets_neg"))
    ap.add_argument("--corrections", default=str(pc.PILE / "corrections.json"))
    ap.add_argument("--min", type=float, default=0.85, help="fail below this coverage")
    ap.add_argument(
        "--composition",
        default=pc.SCALE_NEG_COMPOSITION,
        choices=("provable", "matched"),
        help="which pool composition to judge eligibility against (default: the live config)",
    )
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

    eligible = eligible_under(args.composition)

    worst, thin, ruled_out = 1.0, [], 0
    print(
        f"{'population':<22}{'reviewed':>9}{'by rule':>9}{'by fix':>8}{'eligible':>10}{'still in':>10}{'coverage':>10}"
    )
    print("-" * 78)
    for label, ids, present in rows:
        by_rule, by_fix, denom, kept, cov = coverage_row(ids, present, corrected, eligible)
        ruled_out += by_rule
        worst = min(worst, cov)
        if denom < MIN_DENOM:
            thin.append((label, denom))
        print(f"{label:<22}{len(ids):>9}{by_rule:>9}{by_fix:>8}{denom:>10}{kept:>10}{cov:>10.1%}")
    # Positives are unaffected: the composition governs the negative pool alone,
    # and a positive is designated by its own box in its own band.
    cov_pos = len(pos_ok) / len(pos_pairs) if pos_pairs else 1.0
    worst = min(worst, cov_pos)
    print(
        f"{'reviewed positives':<22}{len(pos_pairs):>9}{'-':>9}{'-':>8}"
        f"{len(pos_pairs):>10}{len(pos_ok):>10}{cov_pos:>10.1%}"
    )
    print()
    if ruled_out:
        # Loud, and above the verdict. The `by rule` column is what turns a 22%
        # catastrophe into a 100% pass, so the reader is told how much of the
        # review it excused before being told the answer.
        print(
            f"composition={args.composition} rules out {ruled_out} reviewed images: they are off-COCO, "
            "and this composition draws only from the COCO-scored half."
        )
        print("Those judgements are not wrong -- they are about a stratum the pool no longer holds.")
    for label, denom in thin:
        # A rule that excuses almost everything leaves a gate that cannot fail.
        # Say so rather than printing a confident 100% off eleven images.
        print(f"NOTE: '{label}' is judged on {denom} images -- too few for this gate to mean much.")
    if worst < args.min:
        print(f"FAIL: coverage {worst:.1%} is below --min {args.min:.0%}.")
        print("The rebuild retired images the composition CAN hold. Check the roster before trusting this cell.")
        return 1
    print(f"OK: every reviewed population is at or above {args.min:.0%}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
