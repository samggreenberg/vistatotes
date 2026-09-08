"""The read-only modes: ``--verify``, ``--rebuildable``, ``--bands``, ``--list``.

``--verify`` and ``--rebuildable`` are complements, and the two answer different
questions. ``--verify`` asks whether the cells on disk are usable;
``--rebuildable`` asks whether they could be produced again. A cell that loads
says nothing about whether it can be rebuilt -- the paths share no code -- which
is how the ``vg_box_*`` rebuild sat broken for eleven days behind a pile that
verified clean (#3297).
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import pile_config as pc

from pilebuild.env import cells_io, experiment_config, log
from pilebuild.geometry import region_geometry_problems, scale_label_digest
from pilebuild.loaders import loader_for


def negative_pool_problems(ds: str, medias: dict) -> list[str]:
    """What one dataset's shared negative pool is MADE OF, and how big it is (#3670).

    Neither is implied by anything else ``--verify`` checks. A pool of the wrong
    size, drawn from the wrong half of VG, still loads: the cells are full, the
    vectors are there, the boxes agree with their bands, and the prevalence
    *of the pool it actually holds* is exact. What breaks is the relation
    between the pickle and the constants every reader quotes -- and that relation
    was only ever true by construction, so a construction change breaks it in
    silence. This is #3299's shape twice over: the cell was fine, what it was
    built FROM was not.

    Two separate claims, so two separate messages:

    * **composition** -- under ``provable`` every designated negative is
      ``coco_scored``, so "holds none of C" is COCO's answer rather than VG's
      silence. A rebuild that quietly drew off-COCO images passes every other
      check here. It reads ``coco_scored`` and not ``labels_exhaustive``: the
      latter is also set by a human who looked at ONE class, which establishes
      nothing about the other eleven, and a cell predating the stamp is told to
      rebuild rather than passed on the weaker flag.
    * **size** -- the pool has as many images as :data:`pile_config.SCALE_N_NEG`
      says, so ``SCALE_PREVALENCE`` describes this pickle. #3670 changed that
      constant while the shared pile still held the old pool; without this check
      the only symptom is that every k\\* a report quotes is computed from a
      prevalence the data does not have.

    Spares are excluded from both: they are drawn from the same strata but
    designated into no cell, which is exactly what an empty
    ``evaluable_categories`` says. Counting them would put the size check 300
    images off and make it fire on a healthy pile.
    """
    problems: list[str] = []
    # A designated negative is scorable everywhere; a spare is scorable nowhere.
    # `categories` cannot tell them apart -- both are empty.
    pool = [m for m in medias.values() if not m.get("categories") and m.get("evaluable_categories")]
    if not pool:
        return problems
    unstamped = sum(1 for m in pool if "coco_scored" not in m)
    n_silent = sum(1 for m in pool if not m.get("coco_scored"))
    # `vg_scale_deep` draws its own pool and is deliberately NOT provable
    # (#3690): it is pinned to the pre-#3670 construction so the #3319/#3547
    # horizon comparison keeps one prevalence from end to end.
    if pc.SCALE_NEG_COMPOSITION == "provable" and n_silent and ds != "vg_scale_deep":
        if unstamped == len(pool):
            problems.append(
                f"{ds}: composition=provable, but no negative carries a `coco_scored` stamp -- "
                "this cell was built before the flag existed, so the claim cannot be checked; rebuild it"
            )
        else:
            problems.append(
                f"{ds}: composition=provable, but {n_silent} of {len(pool)} designated negatives "
                "are not COCO-scored -- their absence claim is VG's silence"
            )
    # Positives per cell differ by construction: `vg_scale` designates one band,
    # `vg_scale_any` collapses all three, `vg_scale_deep` is its own depth.
    # Quoting the realised prevalence is the whole point of the message, so it is
    # read per dataset rather than assumed.
    want, n_pos = {
        "vg_scale": (pc.SCALE_N_NEG, pc.SCALE_N_POS),
        "vg_scale_any": (pc.SCALE_N_NEG, 3 * pc.SCALE_N_POS),
        "vg_scale_deep": (pc.SCALE_DEEP_N_NEG, pc.SCALE_DEEP_N_POS),
    }.get(ds, (pc.SCALE_N_NEG, pc.SCALE_N_POS))
    if len(pool) != want:
        problems.append(
            f"{ds}: {len(pool)} designated negatives, but the config says {want} -- this cell "
            f"predates the current construction, so a cell of it sits at "
            f"{n_pos / (n_pos + len(pool)):.2%} prevalence and not the {n_pos / (n_pos + want):.2%} "
            "the config implies"
        )
    return problems


def boxes_imply_band(boxes: list[list[float]], lo: float, hi: float) -> bool:
    """Does this cell's stored geometry imply the band its name claims?

    Two readings are accepted, because since #3726 a cell carries **every**
    instance of its class while the band comes from the one the reviewer
    designated -- which `apply_corrections` puts at the head of the list. The
    union was the only reading before that, and 37 of 7,500 boxes failed this
    check on the first rebuild afterwards: the data was right and the check was
    still asserting the pre-#3726 invariant.

    **Accepting either does not weaken what this exists to catch.** It is a
    coordinate-space check (#3281): a box normalised twice is ~500x too small
    and sits on the frame origin, so it lands in no band under either reading.
    What it stops asserting is *which* instance the band was taken from -- a
    fact the media dict does not record. Recording it is the right fix; until
    then, inferring it here would be guessing.
    """
    if not boxes:
        return True
    union = (max(b[2] for b in boxes) - min(b[0] for b in boxes)) * (
        max(b[3] for b in boxes) - min(b[1] for b in boxes)
    )
    designated = (boxes[0][2] - boxes[0][0]) * (boxes[0][3] - boxes[0][1])
    return lo <= union < hi or lo <= designated < hi


def coco_held_by() -> dict[int, list[str]]:
    """``VG image id -> the classes of C COCO annotates it with``, empty for none.

    Keyed on VG's ids because that is what a cell carries, and *absent* rather
    than empty for an image COCO never scored: "annotated and holds none" and
    "never annotated" are the two facts this whole pool rests on distinguishing,
    so they must not be the same value here either.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import coco_anchor as ca  # noqa: PLC0415

    try:
        image_data, instances = ca.ensure_sources(pc.PILE / "coco_anchor", fetch=False)
    except SystemExit:
        # Missing sources must not abort the whole of --verify; the pool check
        # reports the gap itself, and says the claim went untested.
        log("NOTE: coco_anchor sources are not staged, so the pool cannot be checked against COCO")
        return {}
    truth = ca.coco_truth(instances, set(pc.SCALE_CLASSES))
    with image_data.open() as fh:
        coco_of = {int(m["image_id"]): int(m["coco_id"]) for m in json.load(fh) if m.get("coco_id")}
    return {i: sorted(c for c, boxes in truth[cid].items() if boxes) for i, cid in coco_of.items() if cid in truth}


def pool_coco_counts(medias: dict, held_by: dict[int, list[str]]) -> tuple[int, int, list[tuple[int, list[str]]]]:
    """``(designated negatives, how many COCO can answer for, the dirty ones)``.

    Split out from the message below so the clean case still has numbers to
    print: "the pool is clean" and "the check could not run" are the same empty
    list of problems, and a measurement that cannot be told from its own absence
    is the failure this directory keeps re-learning (#3667, #3299).

    Spares are excluded exactly as in :func:`negative_pool_problems` -- they are
    designated into no cell, so they make no claim -- and so are positives,
    which hold their class by definition.

    **Only images the build itself anchored are checked**, which is what the
    ``coco_scored`` stamp says. A ``coco_id`` in `image_data.json` is a *wider*
    claim than the build acts on: `anchor_to_coco` refuses a pairing whose
    aspect ratio drifts by more than `MAX_ASPECT_DRIFT`, precisely because a
    drifted pair is evidence the two files are not the same picture. Reading the
    raw join instead reported three dirty negatives in `vg_scale_deep` on the
    first run of this check, and all three were drift rejects -- drift 0.025,
    0.119 and 0.552 against a 0.01 limit, the last pairing a 298x500 portrait
    with a 450x338 landscape. Those images entered the pool on VG's silence,
    which is what deep's composition allows, and COCO was never asked about
    them. Asserting COCO's answer for a pairing the build rejected does not test
    the pool's claim; it invents a different one.
    """
    pool = [(i, m) for i, m in medias.items() if not m.get("categories") and m.get("evaluable_categories")]
    checked = [i for i, m in pool if m.get("coco_scored") and i in held_by]
    dirty = sorted((i, held_by[i]) for i in checked if held_by[i])
    return len(pool), len(checked), dirty


def provable_pool_problems(ds: str, medias: dict, held_by: dict[int, list[str]]) -> list[str]:
    """Does the pool hold none of *C*, by COCO's own annotation? (#3701)

    :func:`negative_pool_problems` tests the pool's **provenance** -- every
    designated negative carries a ``coco_scored`` stamp -- and a true stamp is
    not the claim. An image can be COCO-scored, hold a `truck`, and still be
    designated a negative for every class, if a pass upstream of the draw
    removed the label: an ambiguous-table entry naming another class in *C* pops
    that class's boxes, and on an anchored image the ``exhaustive`` exemption
    suppresses the compensating ``unbanded`` pair, so ``band_candidates`` files
    a COCO-confirmed truck as clean (#3588). A config guard now blocks that one
    route; this tests the *property*, which holds however it was violated.

    **Named, not repaired, and deliberately not dropped.** An image that reaches
    the pool holding a class of *C* means an upstream pass is wrong; removing it
    leaves the pool clean and the cause invisible. The ids are what turn "the
    pool is dirty" into a diagnosis -- in the #3588 case they said `truck` at
    once.
    """
    n_pool, n_checked, dirty = pool_coco_counts(medias, held_by)
    if not n_pool:
        return []
    if not n_checked:
        # Under `provable` every negative is COCO-scored by construction, so
        # nothing to check means the join is missing -- not that the pool is
        # fine. `vg_scale_deep` is pinned to the pre-#3670 draw (#3690) and may
        # legitimately have little COCO can answer for.
        if pc.SCALE_NEG_COMPOSITION == "provable" and ds != "vg_scale_deep":
            return [
                f"{ds}: composition=provable, but COCO could answer for none of the {n_pool} designated "
                "negatives -- the pool's claim went untested. Either `coco_anchor/image_data.json` is not "
                "staged, or this cell predates the `coco_scored` stamp; rebuild it"
            ]
        return []
    if not dirty:
        return []
    shown = ", ".join(f"{i} ({'/'.join(cs)})" for i, cs in dirty[:5])
    return [
        f"{ds}: {len(dirty)} of {n_checked} designated negatives hold a class of C by COCO's own "
        f"annotation -- e.g. {shown}{', ...' if len(dirty) > 5 else ''}. A pass upstream of the draw "
        "put them there; find it rather than dropping them"
    ]


def verify() -> int:
    """Load every present cell and check it is usable. Returns an exit code."""
    io = cells_io()
    problems: list[str] = []
    rows = []
    counts_by_dataset: dict[str, dict[str, int]] = defaultdict(dict)
    for ds, emb in pc.cells():
        path = pc.cell_path(ds, emb)
        if not path.exists():
            rows.append((ds, emb, "MISSING", "", "", ""))
            continue
        medias = io.load_medias(path)
        n = len(medias)
        counts_by_dataset[ds][emb] = n
        n_patch = sum(1 for m in medias.values() if m.get("patch_grid") is not None)
        first = next(iter(medias.values()), None)
        dim = ""
        if first is not None:
            from vtscore.embedding.media_vectors import media_embedding  # noqa: PLC0415

            vec = media_embedding(first)
            dim = str(len(vec)) if vec is not None else "NO-VECTOR"
        want_region = pc.region_capable(ds, emb)
        state = "ok"
        if n == 0:
            state = "EMPTY"
            problems.append(f"{ds} x {emb}: 0 medias")
        elif dim in ("", "NO-VECTOR"):
            state = "NO-VECTOR"
            problems.append(f"{ds} x {emb}: medias carry no embedding")
        elif want_region and n_patch < n:
            state = "PATCH-GAP"
            problems.append(f"{ds} x {emb}: region-capable but patch_grid on only {n_patch}/{n}")
        elif not pc.is_patch_embedder(emb) and n_patch:
            state = "UNEXPECTED-PATCH"
            problems.append(f"{ds} x {emb}: single-vector embedder carries patch grids")
        rows.append((ds, emb, state, str(n), f"{n_patch}/{n}", dim))

    # A banded cell's NAME asserts the size of its boxes, so the stored box has
    # to agree with it. That is what catches a coordinate-space mistake: VG
    # ships 500 px copies of COCO's 640 px originals, and normalising a COCO box
    # by the VG file's dimensions leaves every box shifted and mis-scaled while
    # every other check still passes -- the medias load, the vectors are there,
    # the patch grids are there, and the boxes are quietly pointing at the wrong
    # pixels. Recomputing the band from the box is cheap and would have caught
    # it at build time instead of via a human noticing a box drawn on snow.
    for ds, emb in pc.cells():
        if pc.DATASETS.get(ds, {}).get("kind") != "vg_scale":
            continue
        path = pc.cell_path(ds, emb)
        if not path.exists():
            continue
        medias = io.load_medias(path)
        bad = 0
        checked = 0
        for m in medias.values():
            for cell in m.get("categories") or []:
                boxes = [r["box"] for r in (m.get("regions") or []) if r.get("label") == cell]
                if not boxes:
                    continue
                lo, hi = pc.BOX_BANDS[cell.rsplit("@", 1)[1]]
                checked += 1
                if not boxes_imply_band(boxes, lo, hi):
                    bad += 1
        if checked and bad:
            problems.append(
                f"{ds} x {emb}: {bad}/{checked} region boxes fall outside the band their cell "
                f"name claims, by the union AND by the designated box -- boxes and bands were "
                f"measured in different pixel spaces"
            )
        # The band check above compares a box against its own label, so it is
        # blind to a box corrupted BEFORE banding -- the band moves with it and
        # the two stay consistent (#3281). This one compares the box against the
        # frame, which nothing can drag along with it.
        problems += [f"{ds} x {emb}: {g}" for g in region_geometry_problems(medias)]

        break  # one embedder is enough; the boxes are identical across cells

    # One cell per vg_scale-family dataset: the pool is the same set of images in
    # every embedder's copy, so a second one would only repeat the finding.
    held_by: dict[int, list[str]] | None = None
    for ds in pc.DATASETS:
        if not str(pc.DATASETS[ds].get("kind", "")).startswith("vg_scale"):
            continue
        path = next((p for p in (pc.cell_path(ds, e) for _d, e in pc.cells() if _d == ds) if p.exists()), None)
        if path is not None:
            pool_medias = io.load_medias(path)
            problems += negative_pool_problems(ds, pool_medias)
            # Read once for every dataset: `coco_truth` parses 490 MB of COCO
            # annotation, and the pool is the same population in each of them.
            if held_by is None:
                held_by = coco_held_by()
            problems += provable_pool_problems(ds, pool_medias, held_by)
            n_pool, n_checked, dirty = pool_coco_counts(pool_medias, held_by)
            log(f"{ds}: pool vs COCO -- {n_checked}/{n_pool} negatives answerable, {len(dirty)} hold a class of C")

    # A derived cell that no longer matches its parent. `vg_scale_any` is a
    # relabel of the built `vg_scale` pickle and shares its vectors, so it
    # survives a parent rebuild looking perfect while carrying the parent's
    # PREVIOUS labels, boxes and bands -- which is how a box repair ships to one
    # study and not the other.
    live_digest: dict[Path, str] = {}  # one parent serves every derived cell built from it
    for ds, emb in pc.cells():
        if pc.DATASETS.get(ds, {}).get("kind") != "vg_scale_any":
            continue
        path, parent = pc.cell_path(ds, emb), pc.cell_path("vg_scale", emb)
        if not path.exists() or not parent.exists():
            continue
        medias = io.load_medias(path)
        problems += [f"{ds} x {emb}: {g}" for g in region_geometry_problems(medias)]
        first = next(iter(medias.values()), None)
        stamped = ((first or {}).get("origin") or {}).get("params", {}).get("parent_labels")
        if parent not in live_digest:
            live_digest[parent] = scale_label_digest(io.load_medias(parent))
        live = live_digest[parent]
        if stamped is None:
            problems.append(f"{ds} x {emb}: no parent_labels stamp -- built before the staleness check, rebuild it")
        elif stamped != live:
            problems.append(
                f"{ds} x {emb}: derived from a {parent.name} that has since changed "
                f"({stamped[:12]} != {live[:12]}) -- rebuild it, --force on the parent alone leaves it stale"
            )

    # A dataset's cells must all cover the same medias, or cross-embedder
    # comparisons silently compare different populations. This is not
    # hypothetical: a datadir missing its demo-source symlink sent the loader
    # off to re-download the dataset, and it embedded a truncated 1662-media
    # subset of a 4193-media dataset into a cell that otherwise looked healthy.
    for ds, per_emb in counts_by_dataset.items():
        if len(set(per_emb.values())) > 1:
            majority = max(set(per_emb.values()), key=list(per_emb.values()).count)
            odd = {e: n for e, n in per_emb.items() if n != majority}
            problems.append(
                f"{ds}: cells disagree on media count (most are {majority}); "
                f"rebuild {', '.join(f'{e} ({n})' for e, n in sorted(odd.items()))}"
            )

    log(f"{'dataset':18s} {'embedder':14s} {'state':16s} {'medias':>7s} {'patch':>12s} {'dim':>6s}")
    for ds, emb, state, n, patch, dim in rows:
        log(f"{ds:18s} {emb:14s} {state:16s} {n:>7s} {patch:>12s} {dim:>6s}")

    # Coverage is not implied by anything above: a cell can be structurally
    # perfect and no longer contain the images a human reviewed. Reported here
    # so a rebuild cannot be declared healthy without it being looked at.
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import check_review_coverage  # noqa: PLC0415

        if pc.cell_path("vg_scale", "siglip").exists():
            log("")
            log("review coverage:")
            sys.argv = ["check_review_coverage"]
            if check_review_coverage.main() != 0:
                problems.append("vg_scale: the rebuild retired images that had been reviewed")
    except Exception as exc:  # noqa: BLE001 - an absent review is not a build failure
        log(f"  (review-coverage check skipped: {exc})")

    # The human record is the one input a rebuild cannot regenerate, and it
    # lives on the same purgeable mount as the cells (#3729). Checked here
    # because this is where a pile is declared healthy, and an uncommitted
    # verdict is the kind of loss nobody notices until the mount is cleared.
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import verdict_store  # noqa: PLC0415

        log("")
        log("human record:")
        if verdict_store.do_check(strict=False) != 0:
            problems.append("the human record on disk is not in the repository -- run `verdict_store.py export`")
    except Exception as exc:  # noqa: BLE001 - a missing store is not a build failure
        log(f"  (human-record check skipped: {exc})")

    if problems:
        log("")
        for p in problems:
            log(f"PROBLEM: {p}")
        return 1
    log("all present cells verified")
    return 0


def rebuildable(datasets: list[str] | None = None) -> int:
    """Exercise every dataset's *selection* step without embedding anything.

    The pile documents itself as purgeable -- ``pile_config``: "every cell must
    be rebuildable from sources that are **not** on scratch". Nothing checked
    that. ``--verify`` loads the built cells, and a cell that loads says
    nothing about whether it can be rebuilt: the two paths share no code, so a
    rebuild path can rot for months behind a pile that verifies clean. It did
    (#3297): a scan-format change on 2026-08-17 outran the scan file the
    ``vg_box_*`` cells were selected from, and the break surfaced only when
    somebody asked for a rebuild eleven days later.

    So this is the canary that would have caught it the same day, for a few
    seconds. It runs the part of each build that reads sources and decides
    *what goes in the cell*, and skips the part that costs GPU-hours.

    Each dataset kind answers for itself, through the ``check()`` of the very
    module whose ``load()`` builds it (:mod:`pilebuild.loaders`). That identity
    is the point rather than tidiness: when this check spelled its own source
    paths it named ``COCO_IMAGES`` while the builder opened ``val2017.zip``, and
    reported ``coco_val`` REBUILD-BROKEN against a staging area that was present
    and fine (#3299). A canary that names a different path than the build is not
    a canary.

    A banded dataset goes one question further, via
    :func:`pilebuild.loaders.vg_band._vocab_drift`: not only "would a rebuild
    run?" but "would a rebuild produce *this*?". A repair that restores the
    former while quietly changing the latter is the expensive kind, and it is
    invisible from the built cell.

    Deliberately does not parse the multi-GB sources (VG's ``objects.json``,
    the COCO zip). A canary nobody runs is worth nothing, and the way to make
    it run is to keep it cheap enough to sit in front of every build.
    """
    for bad in [d for d in (datasets or []) if d not in pc.DATASETS]:
        raise SystemExit(f"unknown dataset {bad!r}; known: {sorted(pc.DATASETS)}")
    wanted = list(datasets) if datasets else list(pc.DATASETS)

    problems: list[str] = []

    for ds in wanted:
        try:
            ok = loader_for(ds, pc.DATASETS[ds].get("kind")).check(ds)
            log(f"  {ds:18s} ok       {ok}")
        except SystemExit as exc:  # the loaders' own way of reporting a bad source
            log(f"  {ds:18s} BROKEN   {exc}")
            problems.append(f"{ds}: {exc}")

    if problems:
        log(f"{len(problems)} dataset(s) CANNOT be rebuilt from their sources:")
        for p in problems:
            log(f"REBUILD-BROKEN: {p}")
        return 1
    log("every dataset's selection step runs against its current sources")
    return 0


def report_bands() -> int:
    """Report voted-box scale-band populations for each boxed dataset.

    The bands are anchored to the patch embedder's geometry: ``sub_patch`` is
    "smaller than one DINOv3 patch", i.e. below what the patch grid can resolve
    at all. That anchoring is the band's whole meaning, so a thin ``sub_patch``
    is a fact about the data, not a threshold to tune — widening the edge would
    inflate the count with objects that *are* resolvable.

    Reads the smallest available cell for each dataset: scale stats need only
    ``regions``, which every cell carries, so there is no reason to page in the
    multi-GB patch cell.
    """
    io = cells_io()
    cfg = experiment_config()
    from vtscore.eval.labels import category_scale_stats  # noqa: PLC0415

    boxed = [ds for ds, info in pc.DATASETS.items() if info.get("boxed")]
    if not boxed:
        log("no boxed datasets in the pile; nothing to stratify")
        return 0

    for ds in boxed:
        present = [(pc.cell_path(ds, e).stat().st_size, e) for e in pc.EMBEDDERS if pc.cell_path(ds, e).exists()]
        if not present:
            log(f"{ds}: no cells present")
            continue
        _, emb = min(present)
        medias = io.load_medias(pc.cell_path(ds, emb))

        counts: dict[str, int] = defaultdict(int)
        for m in medias.values():
            for c in m.get("categories") or [m.get("category")]:
                if c:
                    counts[c] += 1

        selected, report = cfg.select_categories_by_scale(medias, dict(counts))
        log("")
        log(f"=== {ds}: {len(medias)} medias, {len(counts)} categories (via {emb}) ===")
        dropped = report.get("dropped_above_max_voted_area") or []
        log(f"  dropped above max_voted_area={report.get('max_voted_area')}: {len(dropped)}")
        for name, info in (report.get("bands") or {}).items():
            lo, hi = info["range"]
            flag = "  ** UNDER-POPULATED **" if info["under_populated"] else ""
            log(
                f"  {name:14s} [{lo * 100:5.2f}%, {hi * 100:6.2f}%): "
                f"{len(info['selected'])}/{info['target']} of {info['n_candidates']} candidates{flag}"
            )
            log(f"      {info['selected']}")

        # When a band is starved, say whether the min-count filter is even the
        # binding constraint. Measured on the first run it was not: the
        # sub_patch pool held 5 categories (VG) and 1 (COCO) at every
        # min_count from 5 to 30, so lowering it recovers nothing.
        starved = [n for n, i in (report.get("bands") or {}).items() if i["under_populated"]]
        if starved:
            stats = {c: s for c in counts if (s := category_scale_stats(medias, c)) is not None}
            for name in starved:
                lo, hi = report["bands"][name]["range"]
                pools = {
                    mc: sum(1 for c, s in stats.items() if counts[c] >= mc and lo <= s["voted_area"] < hi)
                    for mc in (5, 10, 20, 30)
                }
                spread = "same at every min_count" if len(set(pools.values())) == 1 else str(pools)
                log(f"  {name}: candidate pool by min category count -> {spread}")
        log(f"  -> selected {len(selected)} categories")
    return 0


def list_cells() -> None:
    log(f"pile: {pc.PILE}")
    for ds, emb in pc.cells():
        path = pc.cell_path(ds, emb)
        # An `on_request` cell that is absent is absent BY DESIGN, and reading
        # it as MISSING is how a foot-gun gets "fixed" by building it.
        on_request = pc.DATASETS.get(ds, {}).get("on_request")
        mark = "present" if path.exists() else ("on-request" if on_request else "MISSING")
        size = f"{path.stat().st_size / 1e6:8.0f} MB" if path.exists() else " " * 11
        region = " region-voting" if pc.region_capable(ds, emb) else ""
        log(f"  {ds:18s} x {emb:14s} {mark:8s} {size}{region}")
