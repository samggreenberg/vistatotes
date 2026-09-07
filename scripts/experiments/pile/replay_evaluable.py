#!/usr/bin/env python3
"""Would a rebuild produce THIS cell's `evaluable_categories`? (#3678, #3697)

`--verify` asks whether a built cell is internally consistent and whether it
matches two declared constants. It does not ask the question this script asks:
**would running the selection again produce what the pickle holds?** Those come
apart in the direction that hurts -- a cell can load perfectly, carry the right
media count and the right vectors, and encode a rule that was superseded three
merges ago, with nothing anywhere saying so (#3678).

**A pickle cannot answer this about itself, and finding that out is half the
point.** The first cut of this script reconstructed "what the image holds" from
``categories`` and compared the rule's output against the stored field, on the
theory that everything the rule reads is already in the media dict. It is not.
``categories`` is what the image was **designated** for -- 100 images per cell --
and an image can hold a `car` perfectly well without ever being drawn into a
`car` cell. So the reconstruction under-counts what is held, the rule admits
cells it should not, and the replay reports thousands of spurious ADDITIONS
against a change that can only ever remove them. That is a property of the
schema, not a bug in the idea: **`evaluable_categories` is not self-checking,
because the pickle never records what an image holds.**

So this reads the source. It runs the loader's own front half -- the same passes
in the same order, called rather than restated -- to recover ``labels``, then
asks :func:`~pilebuild.loaders.vg_scale._evaluable` what it would write today and
diffs that against what the cell holds. A minute of CPU, no pixels, no GPU, and
it can be run against a cell other studies are reading right now.

**What it still cannot see: selection.** It compares the rule on the images the
cell contains. If a change would designate a *different* set, the images that are
not there are invisible to it. Read a clean report as "the rule agrees on the
images we have", never as "the cell is current".

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

import coco_anchor as ca  # noqa: E402
from _cells_io import load_medias  # noqa: E402
from pilebuild.corrections import load_corrections  # noqa: E402
from pilebuild.loaders.vg_scale import (  # noqa: E402
    _evaluable,
    anchor_to_coco,
    apply_corrections,
    canonicalise,
    lift_ambiguous,
    read_vg_labels,
)
from pilebuild.vgsource import vg_image_paths, vg_source  # noqa: E402


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
        list(pc.SCALE_CLASSES) if args.deep else [pc.scale_cell(c, b) for c in pc.SCALE_CLASSES for b in pc.BOX_BANDS]
    )
    log(f"{Path(args.cell).name}: {len(medias)} medias, {len(cells)} cells")

    corrections = load_corrections()
    in_c = set(pc.SCALE_CLASSES)
    reviewed_absent = {k for k, v in corrections.items() if k[1] in in_c and not v.get("present")}
    reviewed_present = {k for k, v in corrections.items() if k[1] in in_c and v.get("present")}
    log(f"{len(corrections)} verdicts on file: {len(reviewed_absent)} absent, {len(reviewed_present)} present")

    # The loader's own front half, in the loader's own order, so a rule change in
    # the build cannot drift away from this check. `labels` is the thing the
    # pickle cannot give us -- see the module docstring.
    wanted = set(pc.SCALE_CLASSES)
    paths = vg_image_paths()
    _, records, dims = vg_source()
    image_data, instances = ca.ensure_sources(pc.PILE / "coco_anchor", fetch=False)
    truth = ca.coco_truth(instances, wanted)
    with image_data.open() as fh:
        coco_of = {int(m["image_id"]): int(m["coco_id"]) for m in json.load(fh) if m.get("coco_id")}

    labels = read_vg_labels(records, paths, dims, pc.scale_vg_wanted())
    box_dims, exhaustive, _na, _nr = anchor_to_coco(labels, dims, coco_of, truth, ca.COCO_DIMS, wanted)
    coco_scored = set(exhaustive)
    canonicalise(labels, pc.SCALE_VG_NAMES, box_dims, pc.SCALE_FOLD_MODE)
    apply_corrections(labels, corrections, box_dims, exhaustive)
    lift_ambiguous(labels, pc.SCALE_VG_AMBIGUOUS, exhaustive)

    neg_set = {i for i, d in medias.items() if not d.get("categories") and d.get("evaluable_categories")}
    stamped = {i for i, d in medias.items() if d.get("coco_scored")}
    log(
        f"{len(neg_set)} shared negatives; {len(coco_scored & set(medias))} COCO-scored by replay "
        f"({len(stamped)} stamped in the cell); "
        f"{len((exhaustive - coco_scored) & set(medias))} exhaustive by REVIEW alone"
    )

    added: Counter[str] = Counter()
    dropped: Counter[str] = Counter()
    changed_medias = 0
    for iid, d in medias.items():
        cats = list(d.get("categories") or [])
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
