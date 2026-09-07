#!/usr/bin/env python3
"""Would a rebuild produce THIS cell's `evaluable_categories`? (#3678, #3697)

`--verify` asks whether a built cell is internally consistent and whether it
matches two declared constants. It does not ask the question this script asks:
**would running the selection again produce what the pickle holds?** Those come
apart in the direction that hurts -- a cell can load perfectly, carry the right
media count and the right vectors, and encode a rule that was superseded three
merges ago, with nothing anywhere saying so (#3678).

Answering that in general needs the VG source. Answering it for
``evaluable_categories`` does not, and that is the field where the rules actually
churn: #3667 rewrote it, #3697 rewrote it again. Everything the rule reads is
already in the pickle --

* ``categories`` -- the cells the image is a positive for;
* ``evaluable_categories`` -- what the build decided, i.e. the thing under test;
* ``labels_exhaustive`` -- someone or something answered for this image;
* ``coco_scored`` -- COCO answered for all eighty classes at once (#3670);

-- plus ``corrections.json``, which says per ``(image, class)`` what a human
established. So the replay costs seconds, reads no pixels, and can be run
against a cell other studies are using right now.

**What it cannot see.** Only the images the pickle contains. If a rule change
would designate a *different* set of images, this reports nothing about the ones
that are not there -- that is the selection replay #3678 also wants, and it needs
the source. Read a clean report here as "the rule agrees on the images we have",
never as "the cell is current".

Usage::

    python replay_evaluable.py                          # the shipped siglip cell
    python replay_evaluable.py --cell <path.pkl> --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "calibration"))

import pile_config as pc  # noqa: E402

pc.setup_env()

from _cells_io import load_medias  # noqa: E402
from pilebuild.corrections import load_corrections  # noqa: E402
from pilebuild.loaders.vg_scale import _evaluable  # noqa: E402


def log(msg: str) -> None:
    print(f"[replay] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", default=str(pc.EMBEDDINGS / "vg_scale__siglip.pkl"))
    ap.add_argument("--json", default="", help="write the per-cell delta as JSON")
    ap.add_argument(
        "--deep",
        action="store_true",
        help="the cell keys on the bare class (`vg_scale_deep`) rather than `class@band`",
    )
    args = ap.parse_args()

    medias = load_medias(Path(args.cell))
    cells = (
        list(pc.SCALE_CLASSES)
        if args.deep
        else [pc.scale_cell(c, b) for c in pc.SCALE_CLASSES for b in pc.BOX_BANDS]
    )
    log(f"{Path(args.cell).name}: {len(medias)} medias, {len(cells)} cells")

    corrections = load_corrections()
    in_c = set(pc.SCALE_CLASSES)
    reviewed_absent = {k for k, v in corrections.items() if k[1] in in_c and not v.get("present")}
    reviewed_present = {k for k, v in corrections.items() if k[1] in in_c and v.get("present")}
    log(f"{len(corrections)} verdicts on file: {len(reviewed_absent)} absent, {len(reviewed_present)} present")

    # The pickle stores what a class HOLDS only as `categories`, which is the
    # cells it was designated for. That is what the rule reads too, so the
    # replay reconstructs `labels` from it rather than from the VG source: a
    # class appears iff the image is a positive for one of its cells.
    neg_set = {i for i, d in medias.items() if not d.get("categories") and d.get("evaluable_categories")}
    coco_scored = {i for i, d in medias.items() if d.get("coco_scored")}
    exhaustive = {i for i, d in medias.items() if d.get("labels_exhaustive")}
    log(
        f"{len(neg_set)} shared negatives; {len(coco_scored)} COCO-scored, "
        f"{len(exhaustive - coco_scored)} exhaustive by REVIEW alone"
    )

    added: Counter[str] = Counter()
    dropped: Counter[str] = Counter()
    changed_medias = 0
    for iid, d in medias.items():
        cats = list(d.get("categories") or [])
        held = {c.split("@", 1)[0] for c in cats}
        labels = {iid: dict.fromkeys(held, [[0.0, 0.0, 1.0, 1.0]])}
        want = set(_evaluable(iid, cats, cells, neg_set, labels, coco_scored, reviewed_absent, reviewed_present))
        have = set(d.get("evaluable_categories") or [])
        if want == have:
            continue
        changed_medias += 1
        for cell in want - have:
            added[cell] += 1
        for cell in have - want:
            dropped[cell] += 1

    total_add, total_drop = sum(added.values()), sum(dropped.values())
    print("\n" + "=" * 78)
    print("REPLAY: what the CURRENT rule would write, against what the cell HOLDS")
    print("=" * 78)
    print(f"medias whose evaluable set would change: {changed_medias} of {len(medias)}")
    print(f"  cell memberships ADDED by a replay:   {total_add}")
    print(f"  cell memberships DROPPED by a replay: {total_drop}")
    if not changed_medias:
        print("\nAGREES -- on the images this cell contains. Selection is not checked here.")
    else:
        print("\nSTALE -- this cell encodes a rule the current code would not write.")
        worst = (dropped + added).most_common(12)
        if worst:
            print(f"\n{'cell':<24}{'added':>8}{'dropped':>9}")
            for cell, _n in worst:
                print(f"{cell:<24}{added[cell]:>8}{dropped[cell]:>9}")

    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {
                    "cell": str(args.cell),
                    "medias": len(medias),
                    "changed_medias": changed_medias,
                    "added_total": total_add,
                    "dropped_total": total_drop,
                    "review_only_exhaustive": len(exhaustive - coco_scored),
                    "added": dict(added),
                    "dropped": dict(dropped),
                },
                indent=1,
            )
            + "\n"
        )
        log(f"wrote {args.json}")
    return 1 if changed_medias else 0


if __name__ == "__main__":
    raise SystemExit(main())
