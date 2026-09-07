#!/usr/bin/env python3
"""Would a rebuild DESIGNATE this cell's images? (#3678)

``replay_evaluable.py`` answers the same question for one derived field. This
answers it for membership itself -- which images are positives, which are the
shared negative pool -- and it has to ask twice, because the honest answer
depends on something the naive check hides.

**The roster is the reason a rebuild reproduces the past.** ``designate_cells``
pins the membership a review was carried out against, backfilling only what is no
longer eligible; that is deliberate, and #3667's rebuild is what it is for (three
earlier rebuilds retired 577 of 743 reviewed images between them). But it also
means a rebuild against superseded rules quietly reproduces the old selection and
reports nothing -- which is precisely #3678's complaint. A replay that reads the
live roster therefore cannot detect staleness: it is comparing the cell against
the file the build wrote from.

So both are reported, and they answer different questions:

* **pinned** -- replay with the roster the pile carries. This must match the cell
  exactly. A difference is an alarm: the cell and its own roster disagree, which
  means something outside the build has moved (a correction landed, a ruling
  merged, a file was edited by hand). Exit code 1.
* **unpinned** -- replay with no roster at all, i.e. what the current rules would
  choose from scratch. Divergence here is **expected and not a failure**: it is
  the measure of how much of the cell is held in place by the pin rather than
  chosen by today's rules. Quoting it is the point; failing on it would be
  telling the roster off for doing its job.

The second number is the one #3678 asks for and nothing computes today. Read it
as "if the review did not have to be preserved, this is how different the
dataset would be" -- a standing estimate of how far the pile has drifted from the
rules that describe it.

Costs a couple of minutes of CPU: it runs the loader's own front half plus both
designations, reads no pixels and needs no GPU.

Usage::

    python replay_selection.py
    python replay_selection.py --cell <path.pkl> --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "calibration"))

import pile_config as pc  # noqa: E402

pc.setup_env()

import coco_anchor as ca  # noqa: E402
from _cells_io import load_medias  # noqa: E402
from pilebuild.corrections import load_corrections  # noqa: E402
from pilebuild.loaders.vg_scale import (  # noqa: E402
    anchor_to_coco,
    apply_corrections,
    band_candidates,
    canonicalise,
    designate_cells,
    draw_negatives,
    lift_ambiguous,
    read_vg_labels,
)
from pilebuild.vgsource import vg_image_paths, vg_source  # noqa: E402


def log(msg: str) -> None:
    print(f"[selection] {msg}", flush=True)


def _cell_membership(medias: dict[int, dict]) -> tuple[dict[str, set[int]], set[int]]:
    """What the built cell actually holds: positives per cell, and the pool.

    A media with no categories is a shared negative **or** a spare, and only the
    first carries `evaluable_categories` -- the same distinction
    `cross_class_negatives_effect.py` had to make, and getting it wrong put 300
    extra images in a count of 4,000 (#3667).
    """
    positives: dict[str, set[int]] = {}
    pool: set[int] = set()
    for iid, d in medias.items():
        cats = d.get("categories") or []
        if not cats:
            if d.get("evaluable_categories"):
                pool.add(iid)
            continue
        for cell in cats:
            positives.setdefault(cell, set()).add(iid)
    return positives, pool


def _compare(
    label: str, chosen: dict[str, list[int]], negatives: list[int], built_pos: dict[str, set[int]], built_pool: set[int]
) -> dict:
    """One replay against the built cell, as counts plus the worst cells."""
    per_cell = {}
    add_total = drop_total = 0
    for cell in sorted(set(chosen) | set(built_pos)):
        want, have = set(chosen.get(cell, [])), built_pos.get(cell, set())
        a, d = len(want - have), len(have - want)
        add_total += a
        drop_total += d
        if a or d:
            per_cell[cell] = {"added": a, "dropped": d}
    want_neg, have_neg = set(negatives), built_pool
    neg_add, neg_drop = len(want_neg - have_neg), len(have_neg - want_neg)

    print(f"\n--- {label} ---")
    print(f"positives: {add_total} would be added, {drop_total} would be dropped, over {len(per_cell)} cells")
    print(f"negatives: {neg_add} would be added, {neg_drop} would be dropped (pool of {len(want_neg)})")
    if per_cell:
        worst = sorted(per_cell.items(), key=lambda kv: -(kv[1]["added"] + kv[1]["dropped"]))[:8]
        print(f"  {'cell':<24}{'added':>8}{'dropped':>9}")
        for cell, row in worst:
            print(f"  {cell:<24}{row['added']:>8}{row['dropped']:>9}")
    return {
        "positives_added": add_total,
        "positives_dropped": drop_total,
        "cells_changed": len(per_cell),
        "negatives_added": neg_add,
        "negatives_dropped": neg_drop,
        "per_cell": per_cell,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", default=str(pc.EMBEDDINGS / "vg_scale__siglip.pkl"))
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    medias = load_medias(Path(args.cell))
    built_pos, built_pool = _cell_membership(medias)
    log(f"{Path(args.cell).name}: {len(medias)} medias, {len(built_pos)} cells, {len(built_pool)} in the pool")

    wanted = set(pc.SCALE_CLASSES)
    corrections = load_corrections()
    paths = vg_image_paths()
    _, records, dims = vg_source()
    image_data, instances = ca.ensure_sources(pc.PILE / "coco_anchor", fetch=False)
    truth = ca.coco_truth(instances, wanted)
    with image_data.open() as fh:
        coco_of = {int(m["image_id"]): int(m["coco_id"]) for m in json.load(fh) if m.get("coco_id")}

    labels = read_vg_labels(records, paths, dims, pc.scale_vg_wanted())
    box_dims, exhaustive, *_ = anchor_to_coco(labels, dims, coco_of, truth, ca.COCO_DIMS, wanted)
    coco_scored = set(exhaustive)
    canonicalise(labels, pc.SCALE_VG_NAMES, box_dims, pc.SCALE_FOLD_MODE)
    unbanded = apply_corrections(labels, corrections, box_dims, exhaustive)
    unbanded |= lift_ambiguous(labels, pc.SCALE_VG_AMBIGUOUS, exhaustive)
    supply, _boxes, clean = band_candidates(labels, box_dims, unbanded)
    clean.sort()

    roster = json.loads(pc.ROSTER.read_text()) if pc.ROSTER.exists() else {}
    log(f"roster pins {len(roster.get('cells', {}))} cells, {len(roster.get('negatives', []))} negatives")

    report = {}
    for label, r in (("pinned (the roster the pile carries)", roster), ("unpinned (what the rules would choose)", {})):
        chosen = designate_cells(supply, corrections, r)
        pos_ids = {i for ids in chosen.values() for i in ids}
        pos_frac = len(pos_ids & coco_scored) / len(pos_ids) if pos_ids else 1.0
        frac = 1.0 if pc.SCALE_NEG_COMPOSITION == "provable" else pos_frac
        negatives, _spares = draw_negatives(clean, r, coco_scored, frac)
        report["pinned" if r else "unpinned"] = _compare(label, chosen, negatives, built_pos, built_pool)

    pinned = report["pinned"]
    stale = pinned["positives_added"] or pinned["positives_dropped"] or pinned["negatives_added"]
    print("\n" + "=" * 78)
    if stale:
        print("ALARM: the cell disagrees with its OWN roster -- something outside the build moved.")
    else:
        print("OK: the cell is exactly what a rebuild from its own roster would designate.")
    u = report["unpinned"]
    held = u["positives_dropped"]
    print(
        f"Held in place by the pin: {held} positives and {u['negatives_dropped']} negatives "
        f"would be chosen differently by today's rules with no roster."
    )
    print("That divergence is the roster doing its job, not a failure -- see the module docstring.")
    print("=" * 78)

    if args.json:
        Path(args.json).write_text(json.dumps({"cell": str(args.cell), **report}, indent=1) + "\n")
        log(f"wrote {args.json}")
    return 1 if stale else 0


if __name__ == "__main__":
    raise SystemExit(main())
