"""``vg_scale``: one pickle holding every ``(class, band)`` cell of #3156.

The construction, in one paragraph: one image pool and one class list
(:data:`pile_config.SCALE_CLASSES`); for a class *c* and band *B* an image is a
**positive** when its compact union box for *c* falls in *B*, and a **negative**
when it does not hold *c* at all. It is **excluded** from *c*'s cells only when
it holds *c* at some other size -- scoring it as a negative would penalise a
detector for finding a real bus, which is what #3156 is about. Exclusion is
carried per media as ``evaluable_categories`` and honoured by
``vtscore.eval.labels.evaluable_pool``.

That "does not hold *c*" is a claim about the image, and it is only free on the
COCO-annotated half, where all eighty classes are answered at once; off COCO it
would be VG's silence. So :func:`_evaluable` gates the cross-class half of the
rule on ``labels_exhaustive`` (#3667), and until that issue an image was
evaluable **only in its own cells** -- an image holding a book and no bus was
neither a bus positive nor a bus negative, and 41.9% of the pile fell out of
every class's evaluation.

Cells are *designated* rather than inferred: exactly ``SCALE_N_POS`` positives
and one shared pool of ``SCALE_N_NEG`` negatives each, so **within a class the
three bands are scored against identical negatives** -- a small-vs-large
difference is a paired contrast on one class rather than two datasets of
different difficulty. Across classes they are no longer identical, which is the
trade #3667 made deliberately: the shared pool is still shared, but each class
also gets the positives of the other eleven as negatives, and how many of those
there are depends on the class. :data:`pile_config.SCALE_PREVALENCE` is
therefore the **designed** prevalence, not the realised one -- see
``docs/experiments/2026-09-06-cross-class-negatives-3667/REPORT.md``.

**The labels are COCO's, and the pool is the half of VG that can carry them.**
VG's own annotation is not exhaustive and measurably fails this construction --
see :func:`anchor_to_coco` and ``coco_anchor.py``.

**On the other half, VG's vocabulary is the construction's weak point.** VG names
objects in free text and the read matches an object's primary name only, so a
class is built from one spelling out of several. What that costs is not supply:
an instance annotated `bike` while the class is `bicycle` is not a missing
positive, it is a *negative*, because on the non-COCO half VG's silence is the
only evidence of absence (#3605). :func:`canonicalise` folds in the spellings
measured to be the same object, and :func:`lift_ambiguous` withholds the ones
that may not be -- from the bands and from the negative pool alike.

The build is eight passes, and they are named rather than inlined because two of
them are where this dataset's expensive bugs have lived: :func:`apply_corrections`
is the single point at which a box crosses from normalised into pixel space
(#3281), and :func:`designate_cells` is what decides whether a rebuild keeps the
images a human already reviewed. Each takes what it reads and returns what it
produces, so both can be exercised without the 100 GB of VG source the loader
otherwise needs. See ``tests_lib/meta/test_pile_vg_scale.py``.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict

import pile_config as pc

from pilebuild.corrections import correction_boxes_px, load_corrections
from pilebuild.env import log
from pilebuild.geometry import region_geometry_problems
from pilebuild.vgsource import vg_boxes_by_name, vg_image_paths, vg_objects_json, vg_source


def read_vg_labels(
    records: list, paths: dict, dims: dict[int, tuple[int, int]], wanted: set[str]
) -> dict[int, dict[str, list[list[float]]]]:
    """``{image_id: {class: [box_px]}}`` for every usable image in the VG source.

    An image with none of the classes still gets an entry (an empty dict): it is
    a candidate *negative*, and dropping it here would shrink the shared pool to
    the images that already hold something.
    """
    labels: dict[int, dict[str, list[list[float]]]] = {}
    for rec in records:
        iid = int(rec["image_id"])
        if iid not in paths or dims.get(iid) is None:
            continue
        labels[iid] = vg_boxes_by_name(rec, wanted)
    return labels


#: How :func:`canonicalise` reconciles an alias box with a class box that is
#: already on the image. The three are the readings of #3637. All three behave
#: identically where the class has no box of its own, which is the repair the
#: name table exists for; they differ only in what an alias box may do to an
#: image the class already sees:
#:
#: * ``fold`` -- always merge. The scatter guard then judges the union, so an
#:   image the class already banded can leave every band on the strength of a
#:   second spelling.
#: * ``guarded`` -- merge, but keep the class's own boxes on the images where
#:   the merge would push a cleanly-banded one out of every band. Differs from
#:   ``fold`` on those images alone.
#: * ``additive`` -- merge only where the class has no box of its own, so a fold
#:   can add an image but never re-describe one. Differs from ``fold`` wherever
#:   both spellings appear, whether or not a band changes.
FOLD_MODES = ("fold", "guarded", "additive")


def canonicalise(
    labels: dict[int, dict[str, list[list[float]]]],
    vg_names: dict[str, tuple[str, ...]],
    box_dims: dict[int, tuple[int, int]] | None = None,
    mode: str = "fold",
) -> tuple[dict[str, int], dict[str, int]]:
    """Fold each class's alternate VG spellings onto the class name, in place.

    ``vg_boxes_by_name`` matches VG's PRIMARY name only, so `hydrant` and
    `fire hydrant` arrive as two categories of one object. Merging after the read
    -- rather than aliasing during it -- keeps the merge visible and reversible,
    and keeps it out of the shared reader every other VG build uses.

    Returns ``(folded, contested)``, the two numbers the build reports per class:

    * ``folded`` -- boxes actually merged onto the class name. A merge that
      folds nothing has either been mis-spelled or is not needed, and both are
      worth seeing.
    * ``contested`` -- images where the class had a box of its own, that box put
      the image in a band, and the merged union does not land in one. Under
      ``fold`` that is the count of images the fold **un-bands**; under the other
      two modes it is the count it rescues. Either way it is the price of the
      table, and #3637 is the issue that it went unmeasured for a release.

    *box_dims* is what makes ``contested`` measurable, since a band is a share of
    image area. Without it the count is zero and ``guarded`` -- whose whole
    decision *is* the count -- is refused rather than silently degraded into
    ``fold``; ``additive`` does not consult it and is unaffected.
    **Run this after :func:`anchor_to_coco`**, which is where
    ``box_dims`` comes from and which makes the count exact: on an anchored image
    COCO's labels replace VG's wholesale, so a fold there is discarded and
    counting it would report a price the build never pays.
    """
    if mode not in FOLD_MODES:
        raise ValueError(f"canonicalise: unknown mode {mode!r} (want one of {FOLD_MODES})")
    if mode == "guarded" and box_dims is None:
        raise ValueError("canonicalise: mode 'guarded' needs box_dims; a band is a share of image area")
    reverse = {n: cls for cls, names in vg_names.items() for n in names if n != cls}
    folded: dict[str, int] = {c: 0 for c in vg_names}
    contested: dict[str, int] = {c: 0 for c in vg_names}
    for iid, by_name in labels.items():
        # Every alias of one class is collected before any of them is judged: two
        # spellings on one image are one merge, and deciding them one at a time
        # would ask the scatter guard about a union that never exists.
        hits: dict[str, list[list[float]]] = {}
        for vg_name in [n for n in by_name if n in reverse]:
            hits.setdefault(reverse[vg_name], []).extend(by_name.pop(vg_name))
        for cls, boxes in hits.items():
            own = by_name.get(cls)
            if own and box_dims is not None:
                W, H = box_dims[iid]
                if band_for(own, W, H) in pc.BOX_BANDS and band_for(own + boxes, W, H) not in pc.BOX_BANDS:
                    contested[cls] += 1
                    if mode == "guarded":
                        continue
            if own and mode == "additive":
                continue
            by_name.setdefault(cls, []).extend(boxes)
            folded[cls] += len(boxes)
    return folded, contested


def lift_ambiguous(
    labels: dict[int, dict[str, list[list[float]]]],
    vg_names: dict[str, tuple[str, ...]],
    exhaustive: set[int],
) -> set[tuple[int, str]]:
    """Take ambiguous spellings out of *labels*, and return the pairs they suppress.

    A name in :data:`pile_config.SCALE_VG_AMBIGUOUS` may denote its class or
    something else -- VG's `bike` sits on a COCO `bicycle` 40% of the time, and
    59.6% of the time on no COCO class at all, because much of it is motorcycles
    (#3605). It is therefore evidence in neither direction: too weak to band as a
    positive, and far too strong to leave in the shared negative pool, where it
    would score a detector wrong for finding the bicycle that is really there.

    So the boxes are dropped, and the ``(image, class)`` pair joins the
    ``unbanded`` set :func:`band_candidates` already keeps out of both. Three
    things stop a pair being suppressed, and they are the whole judgment in this
    pass:

    * **The image is exhaustively labelled.** COCO annotates C exhaustively and a
      reviewer has looked; either answers the question the spelling leaves open,
      so the spelling is ignored and the image stays a usable negative. This is
      why the pass runs after :func:`anchor_to_coco` and
      :func:`apply_corrections` rather than straight off the read -- on the ~48%
      of VG that is COCO-sourced the defect does not exist, and suppressing there
      would throw away good negatives to fix nothing.
    * **The class is already established on the image.** It is confirmed present
      by a box we trust, and an ambiguous box can only make its extent less
      certain -- a smaller question than this one, and not worth discarding a
      confirmed positive over.
    * The name is the class name itself, which is not ambiguous by definition.

    The boxes are dropped either way: :func:`band_candidates` bands by category
    name and has no cell to put a `bike` in.

    **That last sentence is why a table entry must never name another class in
    C.** The drop is unconditional and global -- it happens before the three
    exemptions are even consulted -- so an entry naming a class deletes that
    class's own boxes from every image in the corpus. `truck` listed as
    ambiguous for `car` took `truck` from 3,386 band-free positives to zero in
    all three bands, and the run said nothing: the fold ledger still printed
    `truck+273/23`, because the fold had already run. The guard is in the suite
    (``test_no_listed_spelling_is_itself_a_class_in_c``) rather than here,
    because this function receives the table as an argument and is used by tests
    with deliberately small ones (#3588).
    """
    reverse = {n: cls for cls, names in vg_names.items() for n in names if n != cls}
    pairs: set[tuple[int, str]] = set()
    for iid, by_name in labels.items():
        for vg_name in [n for n in by_name if n in reverse]:
            cls = reverse[vg_name]
            by_name.pop(vg_name)
            if cls not in by_name and iid not in exhaustive:
                pairs.add((iid, cls))
    return pairs


def anchor_to_coco(
    labels: dict[int, dict[str, list[list[float]]]],
    dims: dict[int, tuple[int, int]],
    coco_of: dict[int, int],
    truth: dict[int, dict[str, list[list[float]]]],
    coco_dims: dict[int, tuple[int, int]],
    wanted: set[str],
) -> tuple[dict[int, tuple[int, int]], set[int], int, int]:
    """Replace VG's labels with COCO's wherever COCO annotates the same image.

    Returns ``(box_dims, exhaustive, n_anchored, n_reframed)`` and edits
    *labels* in place.

    VG's own annotation is not exhaustive: measured against COCO its recall over
    C is 0.61, and 1.35% of the images it treats as negatives actually hold the
    object -- ~54 hidden positives against 100 labelled ones per cell, and 4.1%
    for ``backpack`` puts more real backpacks among the negatives than among the
    positives (``coco_anchor.py``, issue #3156).

    48% of VG's images ARE COCO images, and COCO annotates C exhaustively, so on
    that half the repair is free and total. The other half has no reference and
    is what the human slates are for (``make_audit_slate.py``); its verdicts
    arrive through the corrections file. The pool stays the whole of VG either
    way -- restricting it to the COCO half would make this a COCO subset with
    extra steps, losing VG's non-COCO diversity for nothing.

    ``box_dims`` is the pixel space each image's boxes live in, and is the whole
    reason this pass returns anything. VG ships DOWNSCALED copies of the COCO
    originals -- 500 px wide against COCO's 640, on 95% of the overlap -- so a
    COCO box normalised by the VG file's dimensions lands in the wrong place and
    with the wrong extent. Every box is normalised by the dimensions of the
    image its coordinates were measured on.
    """
    box_dims: dict[int, tuple[int, int]] = dict(dims)
    exhaustive: set[int] = set()
    n_anchored = 0
    n_reframed = 0
    for iid in labels:
        cid = coco_of.get(iid)
        ref = truth.get(cid) if cid is not None else None
        if ref is None:
            continue
        wh = coco_dims.get(cid)
        if wh is None:
            continue
        # A normalised box only transfers between two copies of an image if they
        # frame the same thing. 49 of the 51,497 overlaps disagree on aspect
        # ratio -- some are transposed (VG 500x375 against COCO 375x500), i.e. a
        # rotated or re-cropped copy -- and there COCO's box does not describe
        # VG's pixels at all. Those keep VG's own labels and stay unanchored
        # rather than importing a box that points at the wrong part of a
        # different framing.
        vw, vh = dims[iid]
        if abs((vw / vh) - (wh[0] / wh[1])) / (wh[0] / wh[1]) > pc.MAX_ASPECT_DRIFT:
            n_reframed += 1
            continue
        box_dims[iid] = wh
        # COCO's annotation REPLACES VG's for this image rather than merging
        # with it: the two disagree in both directions, and only one of them is
        # exhaustive. Keeping VG's extra boxes would reintroduce exactly the
        # unverifiable labels this is repairing.
        labels[iid] = {name: bs for name, bs in ref.items() if name in wanted}
        exhaustive.add(iid)
        n_anchored += 1
    return box_dims, exhaustive, n_anchored, n_reframed


def apply_corrections(
    labels: dict[int, dict[str, list[list[float]]]],
    corrections: dict[tuple[int, str], dict],
    box_dims: dict[int, tuple[int, int]],
    exhaustive: set[int],
    classes: tuple[str, ...] | None = None,
) -> set[tuple[int, str]]:
    """Fold human verdicts into *labels*, and return the pairs that cannot be banded.

    **This is the only place a correction's box changes space**, and it is the
    reason the pass has a name. The box, when there is one, is NORMALISED, and
    everything in *labels* is in pixels -- so it is converted here, once, against
    the same ``(W, H)`` the region write later divides by. Merging it unconverted
    is #3281: the region write then normalises a normalised coordinate, which
    divides it by ~500 and parks the box on the frame origin. Nothing downstream
    could see that, because the BAND is derived from the same corrupted box, so
    the cell name and its boxes stayed consistent with each other all the way
    into the study.

    A pair a reviewer ruled "present" without drawing a box cannot be banded: a
    band is a claim about size, and no size was measured. It leaves every cell of
    that class instead -- neither a positive nor a negative -- which is precisely
    what the third value is for.

    **A verdict on a class outside *C* is skipped**, and that is not a nicety.
    `corrections.json` is shared by every build of this family, so it holds rows
    for classes a given build does not have: #3588's negative pass added
    thirteen, and from that moment a twelve-class build died in
    :func:`band_candidates` with ``KeyError: 'car'`` -- a shared file making the
    shipped construction unbuildable, reported as a dict lookup three passes
    later. Skipping is also the only *correct* reading: a verdict about `car`
    cannot move a `bus` cell, and folding it in would write a class the supply
    table has no column for.

    The skip covers ``exhaustive`` too, which is the half that would have
    survived a narrower fix. Marking an image exhaustive means "absence is a
    fact here for every class in *C*"; a human who looked for a `car` did not
    establish that, and under #3670 that flag is what admits an image to the
    provable negative pool.
    """
    in_c = set(pc.SCALE_CLASSES if classes is None else classes)
    unbanded: set[tuple[int, str]] = set()
    for (iid, name), verdict in corrections.items():
        if iid not in labels or name not in in_c:
            continue
        if verdict.get("present"):
            boxes = correction_boxes_px(verdict, *box_dims[iid])
            if boxes:
                labels[iid][name] = boxes
            else:
                labels[iid].pop(name, None)
                unbanded.add((iid, name))
        else:
            labels[iid].pop(name, None)
        exhaustive.add(iid)  # someone looked at this pair
    return unbanded


#: What :func:`band_for` returns for boxes that fall in no band. Named because
#: they are two different facts about an image and an audit has to tell them
#: apart: ``SCATTERED`` means several instances too far apart to be one region,
#: ``OVERSIZE`` means one box bigger than :data:`pile_config.MAX_VOTED_AREA` --
#: not a region, the image.
SCATTERED = "scattered"
OVERSIZE = "oversize"


def band_for(boxes: list[list[float]], W: int, H: int) -> str:
    """The band one class's boxes put an image in, or why they put it in none.

    Returns a key of :data:`pile_config.BOX_BANDS`, or :data:`SCATTERED` /
    :data:`OVERSIZE`. Split out of :func:`band_candidates` so that anything
    asking "what band would these boxes imply?" -- the builder, and
    ``audit_band_drift.py``, which re-bands the same images off a second
    annotation source -- asks it of one implementation. A second copy of this
    rule would answer a drift question with its own drift.

    *boxes* must be non-empty and in the pixel space of ``(W, H)``.
    """
    area = float(W * H)
    ux0 = min(b[0] for b in boxes)
    uy0 = min(b[1] for b in boxes)
    ux1 = max(b[2] for b in boxes)
    uy1 = max(b[3] for b in boxes)
    union = max(0.0, ux1 - ux0) * max(0.0, uy1 - uy0) / area
    largest = max((b[2] - b[0]) * (b[3] - b[1]) for b in boxes) / area
    # Scattered instances in *this* image: the union box describes the scatter
    # rather than the object, so the image is excluded from every band of this
    # class rather than banded by a box no user would drag.
    if union > largest * pc.BAND_MAX_INFLATION:
        return SCATTERED
    for band, (lo, hi) in pc.BOX_BANDS.items():
        if lo <= union < hi:
            return band
    return OVERSIZE


def band_candidates(
    labels: dict[int, dict[str, list[list[float]]]],
    box_dims: dict[int, tuple[int, int]],
    unbanded: set[tuple[int, str]],
    classes: tuple[str, ...] | None = None,
) -> tuple[dict[str, dict[str, list[int]]], dict[tuple[int, str], list[list[float]]], list[int]]:
    """Sort every image into ``(class, band)`` supply, or into the clean pool.

    Returns ``(supply, boxes_for, clean)``. An image with no instance of any
    class in C joins ``clean`` -- unless its ``(image, class)`` pair is in
    *unbanded*, which makes it neither a positive nor a true negative. Two things
    put a pair there: a reviewer who said one *is* present without drawing it (no
    size was measured, and a band is a claim about size), and an ambiguous VG
    spelling that may or may not be the class (:func:`lift_ambiguous`).

    *classes* defaults to :data:`pile_config.SCALE_CLASSES`, i.e. the built
    dataset. It is a parameter so a class being *considered* for C can be banded
    by this exact rule -- the scatter filter and the band edges included --
    rather than by a second implementation in the slate builder that would be
    free to drift from it (#3588). The default keeps every existing caller
    byte-identical.
    """
    classes = tuple(classes) if classes is not None else pc.SCALE_CLASSES
    supply: dict[str, dict[str, list[int]]] = {c: {b: [] for b in pc.BOX_BANDS} for c in classes}
    boxes_for: dict[tuple[int, str], list[list[float]]] = {}
    clean: list[int] = []

    for iid, by_name in labels.items():
        W, H = box_dims[iid]
        if not by_name:
            # Only a true negative for every class in C may join the shared pool.
            if not any((iid, c) in unbanded for c in classes):
                clean.append(iid)
            continue
        for name, bs in by_name.items():
            band = band_for(bs, W, H)
            if band not in pc.BOX_BANDS:  # scattered, or bigger than a region
                continue
            supply[name][band].append(iid)
            boxes_for[(iid, pc.scale_cell(name, band))] = bs
    return supply, boxes_for, clean


def rank(cell: str, iid: int) -> str:
    """A cell-local ordering key that does not move when the pool changes.

    Selection must be stable under a changing candidate list, not merely
    deterministic. ``rng.sample`` is deterministic given the same list, but any
    edit to the pool -- a label fix, an image excluded as a re-framed copy --
    reshuffles the entire draw, and a rebuild then silently retires images a
    human already reviewed (49 of 360 in one such rebuild). Ranking each
    candidate by a hash of ``(cell, image_id)`` instead means adding or removing
    one image changes only that image's membership.
    """
    return hashlib.sha1(f"{cell}:{iid}".encode()).hexdigest()  # noqa: S324 - not security


def designate_cells(
    supply: dict[str, dict[str, list[int]]],
    corrections: dict[tuple[int, str], dict],
    roster: dict,
) -> dict[str, list[int]]:
    """Choose each cell's ``SCALE_N_POS`` positives, preserving reviewed membership.

    A roster pins the membership a review was actually carried out against.
    Without it, switching selection rules retires images a human has already
    judged -- the hash draw and the earlier random draw share only ~228 of 3,900
    negatives, so the entire negative review would have been orphaned. Entries
    that are no longer eligible (a correction moved or removed them) drop out and
    the shortfall is backfilled by :func:`rank`, so the roster adapts without
    ever reshuffling what it can keep.
    """
    chosen: dict[str, list[int]] = {}
    for c in pc.SCALE_CLASSES:
        for band in pc.BOX_BANDS:
            pool = sorted(supply[c][band])
            cell = pc.scale_cell(c, band)
            if len(pool) < pc.SCALE_N_POS:
                # Say so rather than quietly building a smaller cell: unequal
                # prevalence between bands is the defect this construction
                # exists to remove.
                log(f"  UNDER-SUPPLIED {cell}: {len(pool)} positives (wanted {pc.SCALE_N_POS})")
            eligible = set(pool)
            pinned = [i for i in roster.get("cells", {}).get(cell, []) if i in eligible]
            # A correction can move an image to another band -- that is the
            # point of re-drawing a box. If the destination cell is already full
            # of images nobody has looked at, the reviewed one lands nowhere and
            # the review quietly stops covering it (99 of 360 boxed positives,
            # first time round). Reviewed images therefore outrank unreviewed
            # ones for a seat, wherever their box now puts them.
            reviewed = {i for i in eligible if (i, c) in corrections}
            order = (
                [i for i in pinned if i in reviewed]
                + sorted(reviewed - set(pinned), key=lambda i: rank(cell, i))
                + [i for i in pinned if i not in reviewed]
                + sorted(eligible - reviewed - set(pinned), key=lambda i: rank(cell, i))
            )
            chosen[cell] = order[: pc.SCALE_N_POS]
    return chosen


def draw_negatives(
    clean: list[int], roster: dict, coco_scored: set[int], coco_fraction: float
) -> tuple[list[int], list[int]]:
    """The shared negative pool and its spares, drawn from the clean images.

    Spares are drawn beyond the designated pool on purpose: a human verdict can
    retire a contaminated negative later, and re-designating from spares costs a
    relabel rather than a re-embed of every cell.

    The draw is **stratified by provenance** (#3670). *coco_fraction* is the share
    of the pool that must come from *coco_scored* -- images COCO annotated, where
    "holds none of C" is a fact rather than VG's silence. The caller derives it
    from :data:`pile_config.SCALE_NEG_COMPOSITION`: 1.0 for an all-provable pool,
    the positives' own COCO share for a provenance-matched one.

    ``coco_scored`` is deliberately **not** the ``exhaustive`` set the rest of
    the build carries. That one also holds every image a human looked at, and a
    human who looked for a `bus` established nothing about the other eleven
    classes -- so admitting those here would put images in an "all-provable"
    pool whose absence claim is still VG's silence for most of *C*. The
    distinction costs nothing: 34,071 clean images carry a COCO pairing against
    the 9,900 the pool needs.

    Stratifying rather than filtering is what keeps the roster honest. A pinned
    image that is still clean keeps its seat **within its own stratum**, so a
    composition change retires only the images the new composition cannot hold
    instead of reshuffling the whole draw -- the failure that orphaned 49 of 360
    reviewed images the last time a selection rule changed.
    """
    clean_set = set(clean)
    want_prov = round(pc.SCALE_N_NEG * coco_fraction)
    strata = [
        (clean_set & coco_scored, want_prov),
        (clean_set - coco_scored, pc.SCALE_N_NEG - want_prov),
    ]
    pinned = [i for i in roster.get("negatives", []) + roster.get("spares", []) if i in clean_set]

    def draw(pool: set[int], want: int, taken: set[int]) -> list[int]:
        """*want* images from *pool*, roster pins first, then hash-ranked."""
        avail = pool - taken
        out = [i for i in pinned if i in avail][:want]
        if len(out) < want:
            rest = sorted(avail - set(out), key=lambda i: rank("__negatives__", i))
            out += rest[: want - len(out)]
        return out

    negatives: list[int] = []
    for pool, want in strata:
        negatives += draw(pool, want, set(negatives))

    # Spares come from the same strata in the same proportion, so promoting one
    # later cannot change what the pool is made of.
    spares: list[int] = []
    for pool, want in strata:
        share = round(pc.SCALE_N_NEG_SPARE * want / pc.SCALE_N_NEG) if pc.SCALE_N_NEG else 0
        spares += draw(pool, share, set(negatives) | set(spares))
    # A short pool yields what there is rather than silently rebalancing: the
    # builder reports an under-supplied pool, it does not paper over one.
    short = pc.SCALE_N_NEG + pc.SCALE_N_NEG_SPARE - len(negatives) - len(spares)
    if short > 0:
        spares += draw(clean_set, short, set(negatives) | set(spares))
    return negatives, spares


def _cells_by_class(cells: list[str]) -> dict[str, set[str]]:
    """``{class: {cell, ...}}`` for a cell list, whatever its keying.

    ``vg_scale`` cells are ``class@band`` and ``vg_scale_deep`` cells are the
    bare class, so the split is on the suffix separator :func:`pile_config.scale_cell`
    writes. A class name never contains it -- `stop sign` has a space, not an
    `@` -- so the head of the split is the class in both spellings.
    """
    out: dict[str, set[str]] = defaultdict(set)
    for cell in cells:
        out[cell.split("@", 1)[0]].add(cell)
    return dict(out)


def _evaluable(
    iid: int,
    cats: list[str],
    cells: list[str],
    neg_set: set[int],
    labels: dict[int, dict[str, list[list[float]]]],
    exhaustive: set[int],
) -> list[str]:
    """Which cells this image can be SCORED in -- positive or negative.

    ``categories`` says what an image *is* a positive for; this says where it is
    allowed to be judged at all. A shared negative is judged everywhere. A
    positive was, until #3667, judged **only in its own cells** -- so an image
    holding a book and no bus was neither a bus positive nor a bus negative, and
    41.9% of the pile fell out of every class's evaluation.

    The exclusion is right for the same class at another size and wrong for
    every other class, so it is now applied per class: a class this image does
    not hold contributes all of its cells, and the image scores there as the
    negative it is.

    Gated on ``exhaustive``. On a COCO-annotated image "holds no bus" is a fact
    about all eighty classes. Off COCO it is VG's silence, which #3588 measured
    wrong 0.5-2.5% of the time, and importing that into the negatives is a
    separate decision -- ``SCALE_CROSS_CLASS_NEGATIVES`` turns the whole thing
    off rather than pretending the two halves are alike.

    **The cells a class owns are read off ``cells``, never spelled.** This
    function serves two datasets that name their cells differently --
    ``vg_scale`` keys on ``class@band`` and ``vg_scale_deep`` on the bare class
    -- and the first cut of #3667 spelled ``scale_cell(c, band)`` inline. On the
    deep sibling that wrote 36 band-suffixed names that are not cells of that
    dataset into every COCO-exhaustive positive, including the image's **own**
    class at other bands: inert, because nothing matches them, but it is the
    #3156 guarantee stated backwards in a shipped pickle. Deriving the map from
    the caller's own cell list is what makes the rule dataset-agnostic.
    """
    if not cats:
        return list(cells) if iid in neg_set else []
    out = set(cats)
    if pc.SCALE_CROSS_CLASS_NEGATIVES and iid in exhaustive:
        held = set(labels.get(iid, {}))
        for c, owned in _cells_by_class(cells).items():
            if c not in held:
                out |= owned
    return sorted(out)


def _emit_medias(
    medias: dict[int, dict],
    paths: dict,
    chosen: dict[str, list[int]],
    negatives: list[int],
    spares: list[int],
    boxes_for: dict[tuple[int, str], list[list[float]]],
    box_dims: dict[int, tuple[int, int]],
    exhaustive: set[int],
    cells: list[str],
    embedder_name: str,
    labels: dict[int, dict[str, list[list[float]]]],
    coco_scored: set[int],
) -> None:
    """Read the pixels and write one media dict per designated image.

    ``labels`` is REQUIRED, and was an optional ``None`` for exactly one commit.
    An absent label read is not an image that holds nothing -- it is no answer
    at all -- and the two were spellable as the same value, which is how the
    deep sibling silently got #3667's rule with an empty world (:func:`_evaluable`).
    A missing measurement must not be expressible as a measurement of zero; the
    ``vg_box_*`` picker lost a day to the same shape (#3299).
    """
    from PIL import Image  # noqa: PLC0415

    # media id -> the cells it is a positive for. Negatives get every cell.
    positive_in: dict[int, list[str]] = defaultdict(list)
    for cell, ids in chosen.items():
        for iid in ids:
            positive_in[iid].append(cell)
    neg_set = set(negatives)

    for iid in sorted(set(positive_in) | set(negatives) | set(spares)):
        path = paths[iid]
        try:
            with Image.open(path) as im:
                vw, vh = im.size
            data = path.read_bytes()
        except Exception:  # noqa: BLE001 - a corrupt file just drops out
            continue
        if vw <= 0 or vh <= 0:
            continue
        # Normalised region boxes are resolution-independent, so they must be
        # divided by the size of the image the coordinates came from -- which is
        # the COCO original for a repaired image, not the VG copy carrying the
        # pixels.
        W, H = box_dims[iid]
        cats = sorted(positive_in.get(iid, []))
        regions = [
            {"box": [b[0] / W, b[1] / H, b[2] / W, b[3] / H], "label": cell}
            for cell in cats
            for b in boxes_for.get((iid, cell), [])
        ]
        medias[iid] = {
            "id": iid,
            "media_type": "image",
            "embedder": embedder_name,
            "duration": 0,
            "file_size": 0,
            "md5": "",
            "embeddings": {},
            "media_bytes": data,
            "media_string": None,
            "filename": path.name,
            "category": cats[0] if cats else "",
            "categories": cats,
            # A designated cell membership, not a closed world: a positive is
            # scorable only in the cells it was drawn for, and the shared
            # negatives are scorable everywhere.
            "evaluable_categories": _evaluable(iid, cats, cells, neg_set, labels, exhaustive),
            # Whether this image's labels rest on an exhaustive reference (COCO,
            # or a human who looked). False means VG's silence is the only
            # evidence of absence -- which is what the review slates target.
            "labels_exhaustive": iid in exhaustive,
            # The strict half of the flag above, and the one #3670's composition
            # is defined on. `labels_exhaustive` is also set by a human looking
            # at ONE class; this says COCO answered for all eighty at once, which
            # is what makes a negative provable rather than merely reviewed.
            "coco_scored": iid in coco_scored,
            "regions": regions,
            "origin": {"importer": "vg_scale", "params": {"embedder": embedder_name, "labels": "coco"}},
            "origin_name": str(path),
        }


def load(dataset: str, medias: dict[int, dict], embedder_name: str) -> None:
    """Run the eight passes over the VG source and write the designated medias."""
    import coco_anchor as ca  # noqa: PLC0415

    wanted = set(pc.SCALE_CLASSES)
    # The READ has to be wider than the class list: a VG spelling absent from it
    # is invisible downstream, because an image holding only that spelling then
    # looks like an image holding nothing -- i.e. like a negative (#3605).
    wanted_vg = pc.scale_vg_wanted()
    cells = [pc.scale_cell(c, b) for c in pc.SCALE_CLASSES for b in pc.BOX_BANDS]

    paths = vg_image_paths()
    _, records, dims = vg_source()

    image_data, instances = ca.ensure_sources(pc.PILE / "coco_anchor", fetch=False)
    truth = ca.coco_truth(instances, wanted)
    with image_data.open() as fh:
        coco_of = {int(m["image_id"]): int(m["coco_id"]) for m in json.load(fh) if m.get("coco_id")}

    corrections = load_corrections()
    log(f"  {len(coco_of)} VG images carry a coco_id; {len(corrections)} human verdicts on file")

    labels = read_vg_labels(records, paths, dims, wanted_vg)
    box_dims, exhaustive, n_anchored, n_reframed = anchor_to_coco(labels, dims, coco_of, truth, ca.COCO_DIMS, wanted)
    # Taken HERE, before `apply_corrections` folds every reviewed image into
    # `exhaustive`. The two are different claims and #3670 turns on the
    # difference: `exhaustive` says "someone or something answered for this
    # image", `coco_scored` says "all eighty classes were answered at once".
    # Only the second makes "holds none of C" a fact.
    coco_scored = set(exhaustive)
    # The fold runs AFTER the anchor, not before it. On an anchored image COCO's
    # labels replace VG's wholesale, so folding there was always discarded --
    # the order is a no-op on what gets built (verified cell-by-cell in #3637)
    # and is what lets `contested` be counted against the real pixel space, and
    # only on the images that actually pay it.
    folded, contested = canonicalise(labels, pc.SCALE_VG_NAMES, box_dims, pc.SCALE_FOLD_MODE)
    unbanded = apply_corrections(labels, corrections, box_dims, exhaustive)
    suppressed = lift_ambiguous(labels, pc.SCALE_VG_AMBIGUOUS, exhaustive)
    unbanded |= suppressed
    log(
        f"  labels: {len(labels)} VG images, {n_anchored} repaired from COCO, "
        f"{len(exhaustive)} with a verified pair, {n_reframed} skipped as re-framed copies"
    )
    if folded:
        # Both halves of the ledger on one line. A class whose fold contests more
        # images than it repairs is being made worse by its own name table, and
        # `clock` was exactly that (+18 banded, -34 un-banded) for a release
        # before anyone printed the second number (#3637).
        verb = "un-bands" if pc.SCALE_FOLD_MODE == "fold" else "rescues"
        log(
            f"  merged VG spellings (mode={pc.SCALE_FOLD_MODE}) as `class+boxes folded/images it {verb}` -- "
            "a merged union that scatters, or outgrows a region, leaves every band: "
            + ", ".join(f"{c}+{n}/{contested[c]}" for c, n in sorted(folded.items()))
        )
    if suppressed:
        by_class: dict[str, int] = defaultdict(int)
        for _iid, c in suppressed:
            by_class[c] += 1
        log(
            "  ambiguous spellings withheld from both bands and the pool: "
            + ", ".join(f"{c}={n}" for c, n in sorted(by_class.items()))
        )
    unaudited = [c for c in pc.SCALE_CLASSES if c not in pc.SCALE_VG_NAMES_AUDITED]
    if unaudited:
        # Not a failure: the dataset this builds is the one #3156 published, and
        # blocking a rebuild on unmeasured classes would strand it. But it is the
        # one moment the fix is cheap, so it says so rather than passing quietly.
        log(
            f"  VG-NAME COVERAGE UNMEASURED for {len(unaudited)} of {len(pc.SCALE_CLASSES)} classes: "
            + ", ".join(unaudited)
        )
        log("    run `coco_folds.py --classes <c>` and record the result in pile_config.SCALE_VG_NAMES_AUDITED (#3605)")

    supply, boxes_for, clean = band_candidates(labels, box_dims, unbanded)

    roster = {}
    if pc.ROSTER.exists():
        roster = json.loads(pc.ROSTER.read_text())
        log(f"  roster: {pc.ROSTER.name} pins {len(roster.get('cells', {}))} cells")

    chosen = designate_cells(supply, corrections, roster)
    clean.sort()
    # The target provenance mix. `provable` asks for an all-COCO-scored pool;
    # `matched` asks for the positives' OWN share, measured on the images
    # actually designated rather than on the much larger class supply.
    pos_ids = {i for ids in chosen.values() for i in ids}
    pos_frac = len(pos_ids & coco_scored) / len(pos_ids) if pos_ids else 1.0
    if pc.SCALE_NEG_COMPOSITION == "provable":
        coco_fraction = 1.0
    elif pc.SCALE_NEG_COMPOSITION == "matched":
        coco_fraction = pos_frac
    else:
        raise SystemExit(f"unknown SCALE_NEG_COMPOSITION {pc.SCALE_NEG_COMPOSITION!r}")
    negatives, spares = draw_negatives(clean, roster, coco_scored, coco_fraction)
    n_prov = len(set(negatives) & coco_scored)
    log(
        f"  negative pool composition={pc.SCALE_NEG_COMPOSITION}: {n_prov} provable "
        f"({n_prov / len(negatives):.1%} of {len(negatives)}), "
        f"positives are {pos_frac:.1%} COCO-scored"
    )
    pc.ROSTER.write_text(json.dumps({"cells": chosen, "negatives": negatives, "spares": spares}, indent=1) + "\n")
    log(
        f"  {sum(len(v) for v in chosen.values())} positives over {len(cells)} cells, "
        f"{len(negatives)} shared negatives + {len(spares)} spares (from {len(clean)} clean images)"
    )

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
    )

    # Refuse to embed a pickle whose boxes are impossible. `--verify` runs the
    # same check, but only after the GPU hours are spent and the cell is on
    # disk; #3281 got as far as three published studies that way.
    bad = region_geometry_problems(medias)
    if bad:
        raise SystemExit("vg_scale: " + "; ".join(bad))


def check(dataset: str) -> str:
    """What a ``vg_scale`` rebuild reads, without parsing the multi-GB source."""
    objects_json = vg_objects_json()
    if not objects_json.exists():
        raise SystemExit(f"{dataset}: missing {objects_json}")
    n_cells = len(pc.SCALE_CLASSES) * len(pc.BOX_BANDS)
    roster = "roster present" if pc.ROSTER.exists() else "NO ROSTER (membership would be redrawn)"
    return f"VG source present, {n_cells} cells, {roster}"
