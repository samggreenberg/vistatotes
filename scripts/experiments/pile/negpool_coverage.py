#!/usr/bin/env python3
"""What #3670's pool does to the negative review, computed without a rebuild.

The composition change retires images. That is the point -- a pool drawn from
the COCO-scored half cannot hold an off-COCO image -- but the shared pile's
review is the most expensive input this dataset has, and "how much of it
survives" is not a question anyone should answer by rebuilding and looking.

So this **redraws the pool** rather than reading one off disk. Every pass it
uses is the loader's own (:mod:`pilebuild.loaders.vg_scale`), and
:func:`~pilebuild.loaders.vg_scale.draw_negatives` is hash-ranked and
roster-pinned, so the draw it produces is the draw a build would produce --
byte for byte, with no pixels read and no GPU. That matters twice over here:
#3670's own build was overwritten by a parallel study's rebuild an hour after it
ran, and the rebuild that will finally land it is deferred behind #3588's class
expansion. Neither should cost the answer.

For each composition it reports the table :mod:`check_review_coverage` prints,
using that module's own arithmetic rather than a second copy of it:

* **today** -- the pool as the roster has it, for reference.
* **provable** -- every negative COCO-scored.
* **matched** -- the positives' own COCO share.

Read the `by rule` column as the price and the `coverage` column as the risk.
A composition that retires most of the review but keeps everything it is
*allowed* to keep has spent the review, not lost it; one that drops images it
could have held has a draw problem, and no composition argument excuses it.

Usage::

    python negpool_coverage.py [out.json]
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts/experiments/pile")

import pile_config as pc  # noqa: E402

pc.setup_env()

from check_review_coverage import coverage_row  # noqa: E402
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

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("negpool_coverage.json")


def reviewed_populations(base: Path) -> dict[str, set[int]]:
    """The negative populations `check_review_coverage` judges, by name."""
    verdicts = json.loads((base / "verdicts_20260820b.json").read_text())
    out = {"reviewed negatives": {v["image_id"] for v in verdicts if v["stratum"] in ("boundary", "random")}}
    triaged: set[int] = set()
    for p in glob.glob(str(base / "sheets_neg" / "*" / "index.json")):
        for r in json.loads(Path(p).read_text()):
            triaged.add(r["image_id"])
    if triaged:
        out["triaged negatives"] = triaged
    return out


def main() -> None:
    import coco_anchor as ca  # noqa: PLC0415

    wanted = set(pc.SCALE_CLASSES)
    paths = vg_image_paths()
    _, records, dims = vg_source()
    image_data, instances = ca.ensure_sources(pc.PILE / "coco_anchor", fetch=False)
    truth = ca.coco_truth(instances, wanted)
    with image_data.open() as fh:
        coco_of = {int(m["image_id"]): int(m["coco_id"]) for m in json.load(fh) if m.get("coco_id")}

    corrections = load_corrections()
    labels = read_vg_labels(records, paths, dims, pc.scale_vg_wanted())
    box_dims, exhaustive, _n_anchored, _n_reframed = anchor_to_coco(labels, dims, coco_of, truth, ca.COCO_DIMS, wanted)
    canonicalise(labels, pc.SCALE_VG_NAMES, box_dims, pc.SCALE_FOLD_MODE)
    unbanded = apply_corrections(labels, corrections, box_dims, exhaustive)
    unbanded |= lift_ambiguous(labels, pc.SCALE_VG_AMBIGUOUS, exhaustive)
    supply, _boxes_for, clean = band_candidates(labels, box_dims, unbanded)
    clean.sort()

    roster = json.loads(pc.ROSTER.read_text()) if pc.ROSTER.exists() else {}
    chosen = designate_cells(supply, corrections, roster)
    pos_ids = {i for ids in chosen.values() for i in ids}
    pos_frac = len(pos_ids & exhaustive) / len(pos_ids) if pos_ids else 1.0

    populations = reviewed_populations(pc.PILE.parent / "vgscale-3156")
    # Every image a correction touched, matching `check_review_coverage`: a
    # correction that removes a negative is the review working, not coverage lost.
    corrected = {int(c["image_id"]) for c in json.loads((pc.PILE / "corrections.json").read_text())}
    # Eligibility under `provable` is asked of the DRAW, not of the pickle: these
    # are the images that would leave, so there is no cell to read it from.
    provable_eligible = exhaustive.__contains__

    report: dict[str, dict] = {
        "positives_coco_fraction": round(pos_frac, 4),
        "clean_total": len(clean),
        "compositions": {},
    }

    arms = {
        # The roster's own pool, whatever composition built it.
        "today": (list(roster.get("negatives", [])), None),
        "provable": (None, 1.0),
        "matched": (None, pos_frac),
    }
    for name, (fixed, frac) in arms.items():
        if fixed is None:
            pool, _spares = draw_negatives(clean, roster, exhaustive, frac)
        else:
            pool = [i for i in fixed if i in set(clean)]
        present = set(pool)
        eligible = provable_eligible if name == "provable" else (lambda _i: True)
        rows = {}
        for label, ids in populations.items():
            by_rule, by_fix, denom, kept, cov = coverage_row(ids, present, corrected, eligible)
            rows[label] = {
                "reviewed": len(ids),
                "by_rule": by_rule,
                "by_fix": by_fix,
                "eligible": denom,
                "still_in": kept,
                "coverage": round(cov, 4),
            }
        report["compositions"][name] = {
            "n": len(pool),
            "provable": len(present & exhaustive),
            "provable_fraction": round(len(present & exhaustive) / len(pool), 4) if pool else None,
            "populations": rows,
        }

    OUT.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
