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

It also reports the **realised** prevalence, which is not the designed one and
has not been since #3667. A cell scores its shared negatives *plus* every other
class's COCO-scored positives, so the pool a detector actually faces is larger
than ``SCALE_N_NEG`` and the prevalence lower than ``SCALE_PREVALENCE`` -- and by
a different amount for each class, because how many positives the other eleven
have is a property of the class list. Asking for "1% prevalence" and setting
``SCALE_N_NEG`` to 9,900 are therefore two different requests (#3681).

For each composition it reports the table :mod:`check_review_coverage` prints,
using that module's own arithmetic rather than a second copy of it:

* **today** -- the pool as the roster has it, for reference.
* **provable** -- every negative COCO-scored.
* **matched** -- the positives' own COCO share.

Read the `by rule` column as the price and the `coverage` column as the risk.
A composition that retires most of the review but keeps everything it is
*allowed* to keep has spent the review, not lost it; one that drops images it
could have held has a draw problem, and no composition argument excuses it.

Point ``VTS_SCALE_ROSTER`` at the roster the change starts FROM. The pile's
live roster is whatever the last build wrote, which may be another study's; the
pins are what decide how much of the review a composition can keep, so reading
the wrong one silently answers a different question.

Usage::

    VTS_SCALE_ROSTER=<archived pre-change roster> python negpool_coverage.py [out.json]
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
    _evaluable,
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
    # Captured before `apply_corrections` widens `exhaustive`, exactly as the
    # loader does it. Using the widened set here would stratify on a different
    # fact from the build and quietly answer a different question.
    coco_scored = set(exhaustive)
    canonicalise(labels, pc.SCALE_VG_NAMES, box_dims, pc.SCALE_FOLD_MODE)
    unbanded = apply_corrections(labels, corrections, box_dims, exhaustive)
    unbanded |= lift_ambiguous(labels, pc.SCALE_VG_AMBIGUOUS, exhaustive)
    supply, _boxes_for, clean = band_candidates(labels, box_dims, unbanded)
    clean.sort()

    roster = json.loads(pc.ROSTER.read_text()) if pc.ROSTER.exists() else {}
    chosen = designate_cells(supply, corrections, roster)
    pos_ids = {i for ids in chosen.values() for i in ids}
    pos_frac = len(pos_ids & coco_scored) / len(pos_ids) if pos_ids else 1.0

    populations = reviewed_populations(pc.PILE.parent / "vgscale-3156")
    # Every image a correction touched, matching `check_review_coverage`: a
    # correction that removes a negative is the review working, not coverage lost.
    corrected = {int(c["image_id"]) for c in json.loads((pc.PILE / "corrections.json").read_text())}
    # Eligibility under `provable` is asked of the DRAW, not of the pickle: these
    # are the images that would leave, so there is no cell to read it from.
    provable_eligible = coco_scored.__contains__

    report: dict[str, object] = {
        "positives_coco_fraction": round(pos_frac, 4),
        "clean_total": len(clean),
        "compositions": {},
    }
    compositions: dict[str, dict] = report["compositions"]  # type: ignore[assignment]

    arms = {
        # The roster's own pool, whatever composition built it.
        "today": (list(roster.get("negatives", [])), None),
        "provable": (None, 1.0),
        "matched": (None, pos_frac),
    }
    for name, (fixed, frac) in arms.items():
        if fixed is None:
            pool, _spares = draw_negatives(clean, roster, coco_scored, frac)
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
        compositions[name] = {
            "n": len(pool),
            "provable": len(present & coco_scored),
            "provable_fraction": round(len(present & coco_scored) / len(pool), 4) if pool else None,
            "populations": rows,
        }

    # --- realised prevalence, per cell ---------------------------------------
    # Read off `_evaluable` itself rather than a formula: the cross-class rule is
    # the thing being counted, and a second implementation of it here is exactly
    # how the deep sibling got a rule nobody had checked (#3667).
    cells = [pc.scale_cell(c, b) for c in pc.SCALE_CLASSES for b in pc.BOX_BANDS]
    # Same per-(image, class) split `vg_scale.load` makes: a one-class review
    # answers for that class alone (#3697). Counted here because this script
    # reads `_evaluable` itself rather than restating the rule.
    reviewed_absent = {k for k, v in corrections.items() if k[1] in set(pc.SCALE_CLASSES) and not v.get("present")}
    reviewed_present = {k for k, v in corrections.items() if k[1] in set(pc.SCALE_CLASSES) and v.get("present")}
    provable_pool, _sp = draw_negatives(clean, roster, coco_scored, 1.0)
    neg_set = set(provable_pool)
    positive_in: dict[int, list[str]] = {}
    for cell, ids in chosen.items():
        for iid in ids:
            positive_in.setdefault(iid, []).append(cell)

    per_cell: dict[str, int] = dict.fromkeys(cells, 0)
    for iid in set(positive_in) | neg_set:
        ev = _evaluable(
            iid,
            sorted(positive_in.get(iid, [])),
            cells,
            neg_set,
            labels,
            coco_scored,
            reviewed_absent,
            reviewed_present,
        )
        for cell in ev:
            if cell in per_cell and cell not in positive_in.get(iid, []):
                per_cell[cell] += 1

    realised = {}
    for cell, n_neg in per_cell.items():
        pi = pc.SCALE_N_POS / (pc.SCALE_N_POS + n_neg)
        realised[cell] = {"evaluable_negatives": n_neg, "prevalence": round(pi, 5)}
    pis = [v["prevalence"] for v in realised.values()]
    report["realised"] = {
        "designed_cell_prevalence": round(pc.SCALE_N_POS / (pc.SCALE_N_POS + pc.SCALE_N_NEG), 5),
        "mean": round(sum(pis) / len(pis), 5),
        "min": round(min(pis), 5),
        "max": round(max(pis), 5),
        "per_cell": realised,
    }

    # --- how far `exhaustive` reaches past `coco_scored` ----------------------
    # #3667's cross-class rule reads `exhaustive`, which a one-class review also
    # sets -- so an image can serve as a negative for eleven classes on the
    # strength of a human who looked at the twelfth. #3670 removed that from the
    # POOL draw; this counts what it still buys elsewhere.
    promoted = exhaustive - coco_scored
    report["over_promoted"] = {
        "coco_scored": len(coco_scored),
        "exhaustive": len(exhaustive),
        "promoted_by_review": len(promoted),
        "promoted_positives": len(promoted & set(positive_in)),
        "share_of_designated_positives": (
            round(len(promoted & set(positive_in)) / len(positive_in), 4) if positive_in else None
        ),
    }

    OUT.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "realised"}, indent=1))
    print(json.dumps({k: v for k, v in report["realised"].items() if k != "per_cell"}, indent=1))
    print(json.dumps(report["over_promoted"], indent=1))


if __name__ == "__main__":
    main()
