"""The read-only modes: ``--verify``, ``--rebuildable``, ``--bands``, ``--list``.

``--verify`` and ``--rebuildable`` are complements, and the two answer different
questions. ``--verify`` asks whether the cells on disk are usable;
``--rebuildable`` asks whether they could be produced again. A cell that loads
says nothing about whether it can be rebuilt -- the paths share no code -- which
is how the ``vg_box_*`` rebuild sat broken for eleven days behind a pile that
verified clean (#3297).
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import pile_config as pc

from pilebuild.env import cells_io, experiment_config, log
from pilebuild.geometry import region_geometry_problems, scale_label_digest
from pilebuild.loaders import loader_for


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
                area = (max(b[2] for b in boxes) - min(b[0] for b in boxes)) * (
                    max(b[3] for b in boxes) - min(b[1] for b in boxes)
                )
                lo, hi = pc.BOX_BANDS[cell.rsplit("@", 1)[1]]
                checked += 1
                if not (lo <= area < hi):
                    bad += 1
        if checked and bad:
            problems.append(
                f"{ds} x {emb}: {bad}/{checked} region boxes fall outside the band their cell "
                f"name claims -- boxes and bands were measured in different pixel spaces"
            )
        # The band check above compares a box against its own label, so it is
        # blind to a box corrupted BEFORE banding -- the band moves with it and
        # the two stay consistent (#3281). This one compares the box against the
        # frame, which nothing can drag along with it.
        problems += [f"{ds} x {emb}: {g}" for g in region_geometry_problems(medias)]

        # What the negatives are MADE OF is implied by nothing above. A pool of
        # the right size, with valid boxes, vectors and prevalence, can still
        # rest on VG's silence rather than on COCO's answer -- and the whole
        # point of the `provable` composition is that it does not (#3670). A
        # rebuild that quietly drew off-COCO images would pass every other check
        # in this function, which is the same shape as #3299: the cell was fine,
        # what it was built FROM was not.
        pool = [m for m in medias.values() if not m.get("categories")]
        n_prov = sum(1 for m in pool if m.get("labels_exhaustive"))
        if pool and pc.SCALE_NEG_COMPOSITION == "provable" and n_prov < len(pool):
            problems.append(
                f"{ds} x {emb}: composition=provable, but {len(pool) - n_prov} of {len(pool)} "
                "shared negatives carry no exhaustive reference -- they rest on VG's silence"
            )
        break  # one embedder is enough; the boxes are identical across cells

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
