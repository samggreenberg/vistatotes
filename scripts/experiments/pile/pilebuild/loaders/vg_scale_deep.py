"""``vg_scale_deep``: ``vg_scale_any``'s construction, sized for a long session.

#3319 ran the first 400-click wave on ``vg_scale_any`` and found that the
aggressive arms harvested **82-85%** of the ~150 positives their sim half held,
against 15% for the control.  That is not a truncation but a one-sided
*confound*: the arms stop being compared over the same opportunity precisely
where the interesting part of the trajectory is.  It cost #3319 half its
question -- "does the optimum get **deeper** at depth?" cannot be answered by
arms that are ceiling-limited over their last quarter (#3547).

**What is actually scarce.**  ``vg_scale_any`` is not thin because VG is thin.
It collapses ``class@band`` *after* designation, so it inherits ``vg_scale``'s
per-band draw and is capped by the thinnest band anywhere in the class list --
``bus@small`` has 138 candidates.  Band-free designation off the same
COCO-anchored labels takes the binding class from 414 to **1006**
(``stop sign``; ``measure_supply.py``), which is what makes 400 clicks
measurable.  So this is one selection change, not a new dataset: the labels, the
COCO repair, the human corrections and the exclusion semantics are ``vg_scale``'s
passes, called unchanged.

**Prevalence is held, not inherited.**  The quantity this family of studies is
trying to locate is ``k* = -log2((1-pi)/pi)``, so pi is not a free parameter of
the rebuild.  ``pile_config.SCALE_DEEP_N_NEG`` is *derived* from the positive
count against :data:`pile_config.SCALE_PREVALENCE`; 900 positives against a
negative pool left at 3900 would read as "a deeper haystack" while moving the
answer by a full bit.  At 900 x 11700 the prevalence is 7.143%, the same 7.14%
``vg_scale`` designs for and the same number ``-3.71`` is computed from.

That equality is between two **designed** counts, and the assertion below
compares them -- which is why it passed unchanged through #3667's rebuild while
the number a harness actually sees moved.  Scoring happens over the *evaluable
pool*, and #3667 grew it ~45% without adding a positive: realised prevalence is
**5.09%** here against **4.99%** on ``vg_scale_any``.  The premise holds, and
holds for the right reason -- the two moved together, 0.03 bits apart, because
the mechanism adding the negatives is the same on both -- but both are now
about half a bit from ``SCALE_PREVALENCE``.  See #3681.

**Why not just rebuild ``vg_scale_any`` deeper.**  Five studies (#3115, #3196,
#3287, #3290, #3318, #3319) ran on that cell, and ``pile_config`` already warns
that a rebuild silently changes what it is.  A deeper cell under the same name
would make every one of those numbers unreproducible for a gain none of them
needs.  This is a sibling, and the two are built from the same source.
"""

from __future__ import annotations

import json

import pile_config as pc

from pilebuild.corrections import load_corrections
from pilebuild.env import log
from pilebuild.geometry import region_geometry_problems
from pilebuild.loaders.vg_scale import (
    _emit_medias,
    anchor_to_coco,
    apply_corrections,
    band_candidates,
    canonicalise,
    lift_ambiguous,
    rank,
    read_vg_labels,
)
from pilebuild.vgsource import vg_image_paths, vg_objects_json, vg_source


def collapse_bands(
    supply: dict[str, dict[str, list[int]]],
    boxes_for: dict[tuple[int, str], list[list[float]]],
) -> tuple[dict[str, list[int]], dict[tuple[int, str], list[list[float]]]]:
    """Band-keyed supply and boxes, re-keyed on the bare class.

    An image lands in exactly one band per class, so the union is a disjoint
    concatenation and no image can be drawn twice for one class -- asserted
    rather than assumed, because a duplicate positive would inflate a cell's
    prevalence silently.
    """
    flat: dict[str, list[int]] = {}
    flat_boxes: dict[tuple[int, str], list[list[float]]] = {}
    for c in pc.SCALE_CLASSES:
        seen: list[int] = []
        for band in pc.BOX_BANDS:
            seen.extend(supply[c][band])
        if len(set(seen)) != len(seen):
            raise SystemExit(f"vg_scale_deep: {c} lands in two bands for one image; bands are not disjoint")
        flat[c] = seen
        for iid in seen:
            for band in pc.BOX_BANDS:
                boxes = boxes_for.get((iid, pc.scale_cell(c, band)))
                if boxes is not None:
                    flat_boxes[(iid, c)] = boxes
                    break
    return flat, flat_boxes


def designate_deep(
    supply: dict[str, list[int]],
    corrections: dict[tuple[int, str], dict],
    roster: dict,
) -> dict[str, list[int]]:
    """Choose each class's ``SCALE_DEEP_N_POS`` positives, band-free.

    The ordering is :func:`pilebuild.loaders.vg_scale.designate_cells`'s, for
    the reason that function gives: a reviewed image outranks an unreviewed one
    so a rebuild never orphans a human verdict, and everything else is ranked by
    a hash of ``(cell, image_id)`` so adding or removing one candidate changes
    only that candidate's membership.

    Under-supply is fatal here rather than logged.  ``vg_scale`` can afford to
    note a thin band because the band is the thing it is measuring; a cell that
    cannot fill in *this* dataset is a cell at a different prevalence, and equal
    prevalence across the twelve is what the deep comparison rests on.
    """
    chosen: dict[str, list[int]] = {}
    thin = []
    for c in pc.SCALE_CLASSES:
        pool = sorted(supply[c])
        if len(pool) < pc.SCALE_DEEP_N_POS:
            thin.append(f"{c} ({len(pool)})")
            continue
        eligible = set(pool)
        pinned = [i for i in roster.get("cells", {}).get(c, []) if i in eligible]
        reviewed = {i for i in eligible if (i, c) in corrections}
        order = (
            [i for i in pinned if i in reviewed]
            + sorted(reviewed - set(pinned), key=lambda i: rank(c, i))
            + [i for i in pinned if i not in reviewed]
            + sorted(eligible - reviewed - set(pinned), key=lambda i: rank(c, i))
        )
        chosen[c] = order[: pc.SCALE_DEEP_N_POS]
    if thin:
        raise SystemExit(
            f"vg_scale_deep: {len(thin)} class(es) cannot fill {pc.SCALE_DEEP_N_POS} positives: "
            f"{', '.join(thin)}. Lower VTS_SCALE_DEEP_N_POS or drop the class -- do NOT "
            "short-fill, which would give the twelve cells unequal prevalence."
        )
    return chosen


def draw_negatives(clean: list[int], roster: dict) -> tuple[list[int], list[int]]:
    """The shared negative pool and its spares, at this dataset's sizes."""
    want = pc.SCALE_DEEP_N_NEG + pc.SCALE_DEEP_N_NEG_SPARE
    clean_set = set(clean)
    drawn = [i for i in roster.get("negatives", []) + roster.get("spares", []) if i in clean_set]
    if len(drawn) < want:
        extra = sorted(clean_set - set(drawn), key=lambda i: rank("__negatives__", i))
        drawn += extra[: want - len(drawn)]
    drawn = drawn[:want]
    if len(drawn) < want:
        raise SystemExit(f"vg_scale_deep: only {len(drawn)} clean images for a {want}-image negative pool")
    return drawn[: pc.SCALE_DEEP_N_NEG], drawn[pc.SCALE_DEEP_N_NEG :]


def load(dataset: str, medias: dict[int, dict], embedder_name: str) -> None:
    """``vg_scale``'s eight passes, with a band-free designation in the middle."""
    import coco_anchor as ca  # noqa: PLC0415

    wanted = set(pc.SCALE_CLASSES)
    cells = list(pc.SCALE_CLASSES)

    paths = vg_image_paths()
    _, records, dims = vg_source()

    image_data, instances = ca.ensure_sources(pc.PILE / "coco_anchor", fetch=False)
    truth = ca.coco_truth(instances, wanted)
    with image_data.open() as fh:
        coco_of = {int(m["image_id"]): int(m["coco_id"]) for m in json.load(fh) if m.get("coco_id")}

    corrections = load_corrections()
    log(f"  {len(coco_of)} VG images carry a coco_id; {len(corrections)} human verdicts on file")

    # Read wider than the class list, fold the measured spellings in, and
    # withhold the ambiguous ones -- `vg_scale`'s passes, called unchanged, so
    # the sibling cannot carry a defect the shallow cell has been repaired of
    # (#3605). Leaving them out would put a VG `bike` image in this pool as a
    # `bicycle` negative while `vg_scale` excludes it, which is precisely the
    # "only depth changed" premise breaking.
    labels = read_vg_labels(records, paths, dims, pc.scale_vg_wanted())
    # The re-banding ledger is `vg_scale.load`'s to print (#3659): deep shares
    # this pass and the shallow cell's labels, so the two builds' rows would be
    # identical and the second copy would only invite a reader to diff them.
    box_dims, exhaustive, n_anchored, n_reframed, _reband = anchor_to_coco(
        labels, dims, coco_of, truth, ca.COCO_DIMS, wanted
    )
    # Before corrections widen it -- see `vg_scale.load`. Deep's own pool is not
    # stratified on it (#3690 pins deep to the pre-#3670 construction), but the
    # medias carry the flag so a reader of either dataset can ask the same
    # question of both.
    coco_scored = set(exhaustive)
    # Per (image, class), what a human established -- the same distinction
    # `vg_scale.load` draws, and deep needs it for the same reason: it shares
    # `_emit_medias` and therefore `_evaluable`, so a one-class review would
    # otherwise promote an image into cross-class negatives for the other
    # twenty-four here too (#3697).
    reviewed_absent = {k for k, v in corrections.items() if k[1] in wanted and not v.get("present")}
    reviewed_present = {k for k, v in corrections.items() if k[1] in wanted and v.get("present")}
    # After the anchor and with the pixel space, exactly as `vg_scale` runs it:
    # the fold's treatment of a scattered union is part of "what a positive is",
    # so a sibling that folded differently would break the only-depth-changed
    # premise as surely as a missing spelling would (#3637).
    canonicalise(labels, pc.SCALE_VG_NAMES, box_dims, pc.SCALE_FOLD_MODE)
    unbanded = apply_corrections(labels, corrections, box_dims, exhaustive)
    unbanded |= lift_ambiguous(labels, pc.SCALE_VG_AMBIGUOUS, exhaustive)
    log(
        f"  labels: {len(labels)} VG images, {n_anchored} repaired from COCO, "
        f"{len(exhaustive)} with a verified pair, {n_reframed} skipped as re-framed copies"
    )

    banded, banded_boxes, clean = band_candidates(labels, box_dims, unbanded)
    supply, boxes_for = collapse_bands(banded, banded_boxes)
    log("  band-free supply: " + ", ".join(f"{c}={len(supply[c])}" for c in pc.SCALE_CLASSES))

    roster = {}
    if pc.DEEP_ROSTER.exists():
        roster = json.loads(pc.DEEP_ROSTER.read_text())
        log(f"  roster: {pc.DEEP_ROSTER.name} pins {len(roster.get('cells', {}))} cells")

    chosen = designate_deep(supply, corrections, roster)
    clean.sort()
    negatives, spares = draw_negatives(clean, roster)
    pc.DEEP_ROSTER.write_text(json.dumps({"cells": chosen, "negatives": negatives, "spares": spares}, indent=1) + "\n")

    prevalence = pc.SCALE_DEEP_N_POS / (pc.SCALE_DEEP_N_POS + pc.SCALE_DEEP_N_NEG)
    log(
        f"  {sum(len(v) for v in chosen.values())} positives over {len(cells)} cells, "
        f"{len(negatives)} shared negatives + {len(spares)} spares (from {len(clean)} clean images)"
    )
    log(f"  prevalence {prevalence:.5f} vs the pinned {pc.SCALE_DEEP_PREVALENCE:.5f} (#3690)")
    # The rebuild's whole premise is that only DEPTH changed. A drift here is
    # the failure mode the derived SCALE_DEEP_N_NEG exists to prevent, so it is
    # asserted at build time rather than left to a reader of the manifest.
    #
    # The target is `SCALE_DEEP_PREVALENCE`, not `SCALE_PREVALENCE`. Those were
    # the same constant until #3670 took `vg_scale` to 1% and deep stayed at the
    # 7.14% its horizon comparison was measured against; against the live
    # `vg_scale` number this assertion now fires on a correct build, which is a
    # rebuild that aborts rather than a dataset that drifts -- but it would have
    # aborted at the end of the GPU hours, and the message would have named the
    # wrong culprit.
    if abs(prevalence - pc.SCALE_DEEP_PREVALENCE) > 1e-4:
        raise SystemExit(
            f"vg_scale_deep: prevalence {prevalence:.5f} != the pinned {pc.SCALE_DEEP_PREVALENCE:.5f}; "
            "the deep cell would not be comparable to the runs it exists to extend"
        )

    # `labels` is not optional here, whatever the signature's default says: it
    # is how #3667 knows which classes an image HOLDS, and omitting it reads as
    # "holds nothing", which turns the cross-class rule into "scorable
    # everywhere" -- including the image's own class. This call passed it for
    # the first time in #3667's rebuild; before that the deep cell got the
    # exclusion semantics of a dataset with no labels at all.
    _emit_medias(
        medias,
        paths,
        chosen,
        negatives,
        spares,
        boxes_for,
        box_dims,
        exhaustive,
        cells,
        embedder_name,
        labels,
        coco_scored,
        reviewed_absent,
        reviewed_present,
    )
    for d in medias.values():
        d["origin"] = {
            "importer": "vg_scale_deep",
            "params": {"embedder": embedder_name, "labels": "coco", "n_pos": pc.SCALE_DEEP_N_POS},
        }

    n_excluded = sum(1 for d in medias.values() if not d["categories"] and not d["evaluable_categories"])
    if not n_excluded:
        raise SystemExit("vg_scale_deep: no excluded-everywhere medias survived - the exclusion semantics were lost")

    bad = region_geometry_problems(medias)
    if bad:
        raise SystemExit("vg_scale_deep: " + "; ".join(bad))


def check(dataset: str) -> str:
    """What a ``vg_scale_deep`` rebuild reads -- the same sources as ``vg_scale``."""
    objects_json = vg_objects_json()
    if not objects_json.exists():
        raise SystemExit(f"vg_scale_deep: missing {objects_json}")
    anchor = pc.PILE / "coco_anchor"
    missing = [n for n in ("image_data.json", "instances_train2017.json") if not (anchor / n).exists()]
    if missing:
        raise SystemExit(f"vg_scale_deep: missing {', '.join(str(anchor / m) for m in missing)}")
    return (
        f"{objects_json.name} + coco_anchor/, {pc.SCALE_DEEP_N_POS} positives x "
        f"{len(pc.SCALE_CLASSES)} classes vs {pc.SCALE_DEEP_N_NEG} negatives"
    )
