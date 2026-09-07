#!/usr/bin/env python3
"""How many negatives can `vg_scale` draw, and how many of them can COCO PROVE?

#3670 asks for 1% prevalence (9,900 negatives against today's 3,900). The images
exist; the question this answers is what they are made of, because a negative
drawn from the COCO-anchored half is **provable** -- COCO annotates all eighty of
its classes on any image it touches, so "holds none of C" is a fact there -- while
one drawn off-COCO rests on VG's silence, measured wrong 0.3-2.8% per class
(#3635, #3666).

It reproduces the front half of ``vg_scale.load`` by CALLING the loader's own
passes rather than restating them, so a rule change in the build cannot drift
away from this measurement. That is the same reason those passes were factored
out in the first place (#3156).

Reports, for each candidate composition:

* **all-provable** -- every negative COCO-scored. Zero VG-silence error, but the
  negatives are then 100% COCO-sourced while the positives are ~57%, so image
  PROVENANCE becomes a free signal (VG draws on COCO and YFCC100M, which look
  different). Whether that is a real hazard or a theoretical one is what
  ``provenance_probe.py`` measures.
* **matched** -- the split #3670 recommends, negatives matching the positives'
  own COCO share. No provenance signal, but only that share is provable.
* **today** -- the 3,900-image pool as built, for reference.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts/experiments/pile")
sys.path.insert(0, "scripts/experiments/calibration")

import pile_config as pc  # noqa: E402
from pilebuild.corrections import load_corrections  # noqa: E402
from pilebuild.loaders.vg_scale import (  # noqa: E402
    anchor_to_coco,
    apply_corrections,
    band_candidates,
    canonicalise,
    designate_cells,
    lift_ambiguous,
    read_vg_labels,
)
from pilebuild.vgsource import vg_image_paths, vg_source  # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("negpool_supply.json")
#: Which class list defines "clean". The build uses SCALE_CLASSES (12); pass
#: `all25` to price the pool #3588's expansion would need, where an image must
#: hold none of the twelve AND none of the thirteen candidates to be a negative.
SCOPE = sys.argv[2] if len(sys.argv) > 2 else "shipped"


def main() -> None:
    import coco_anchor as ca

    # `all25` has to widen SCALE_CLASSES itself, not just the read set. The
    # loader keys `supply` on SCALE_CLASSES, so a label for a class outside it
    # raises KeyError -- which is the sharpest available demonstration that the
    # class list is not merely a filter but the schema the build is written
    # against. Simulating the expansion means setting it, exactly as #3588's
    # config change would.
    if SCOPE == "all25":
        pc.SCALE_CLASSES = tuple(list(pc.SCALE_CLASSES) + list(pc.SCALE_CANDIDATES_3588))
    wanted = set(pc.SCALE_CLASSES)
    wanted_vg = pc.scale_vg_wanted() | wanted
    print(f"scope={SCOPE}: {len(wanted)} classes define a clean image")

    paths = vg_image_paths()
    _, records, dims = vg_source()
    image_data, instances = ca.ensure_sources(pc.PILE / "coco_anchor", fetch=False)
    truth = ca.coco_truth(instances, wanted)
    with image_data.open() as fh:
        coco_of = {int(m["image_id"]): int(m["coco_id"]) for m in json.load(fh) if m.get("coco_id")}

    corrections = load_corrections()
    labels = read_vg_labels(records, paths, dims, wanted_vg)
    box_dims, exhaustive, _n_anchored, _n_reframed = anchor_to_coco(labels, dims, coco_of, truth, ca.COCO_DIMS, wanted)
    canonicalise(labels, pc.SCALE_VG_NAMES, box_dims, pc.SCALE_FOLD_MODE)
    unbanded = apply_corrections(labels, corrections, box_dims, exhaustive)
    unbanded |= lift_ambiguous(labels, pc.SCALE_VG_AMBIGUOUS, exhaustive)

    supply, _boxes_for, clean = band_candidates(labels, box_dims, unbanded)

    roster = json.loads(pc.ROSTER.read_text()) if pc.ROSTER.exists() else {}
    chosen = designate_cells(supply, corrections, roster)

    # --- the two strata -------------------------------------------------------
    clean_set = set(clean)
    provable = clean_set & exhaustive
    silent = clean_set - exhaustive

    # A positive's provenance is the same property, asked of the images actually
    # designated -- not of the class supply, which is much larger.
    pos_ids = {i for ids in chosen.values() for i in ids}
    pos_coco = len(pos_ids & exhaustive)
    pos_frac = pos_coco / len(pos_ids) if pos_ids else 0.0

    # --- today's pool, for reference -----------------------------------------
    today = [i for i in roster.get("negatives", []) if i in clean_set]
    today_provable = len(set(today) & exhaustive)

    report = {
        "clean_total": len(clean_set),
        "provable_available": len(provable),
        "silent_available": len(silent),
        "positives_designated": len(pos_ids),
        "positives_coco_anchored": pos_coco,
        "positives_coco_fraction": round(pos_frac, 4),
        "today": {
            "n": len(today),
            "provable": today_provable,
            "provable_fraction": round(today_provable / len(today), 4) if today else None,
        },
        "compositions": {},
    }

    for n_neg in (pc.SCALE_N_NEG, 9900):
        allp = {
            "n": n_neg,
            "provable": min(n_neg, len(provable)),
            "silent": max(0, n_neg - len(provable)),
            "feasible": len(provable) >= n_neg,
            "coco_fraction": 1.0,
        }
        want_coco = round(n_neg * pos_frac)
        matched = {
            "n": n_neg,
            "provable": want_coco,
            "silent": n_neg - want_coco,
            "feasible": len(provable) >= want_coco and len(silent) >= n_neg - want_coco,
            "coco_fraction": round(pos_frac, 4),
        }
        # What the composition does to the contamination floor. Only the
        # VG-silence half carries it, so the floor scales with that half's share.
        for name, comp in (("all_provable", allp), ("matched", matched)):
            comp["silence_share"] = round(comp["silent"] / comp["n"], 4)
            if today:
                base = 1 - today_provable / len(today)
                comp["floor_vs_today"] = round(comp["silence_share"] / base, 3) if base else None
            report["compositions"].setdefault(str(n_neg), {})[name] = comp

    OUT.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
