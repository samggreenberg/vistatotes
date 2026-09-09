"""Build (and verify) the shared pre-embedded pile of ``(dataset, embedder)`` cells.

One cell = one ``<dataset>__<embedder>.pkl`` of media dicts carrying vectors
(and ``patch_grid`` for patch embedders) but no pixels. Studies point
``VTSEARCH_DATA_DIR`` at the pile and load cells in place, so an embedder runs
once per pair ever rather than once per study.

Idempotent: a cell that already exists is skipped unless ``--force``. That makes
this safe to re-run after a partial SLURM job, and makes it the rebuild path if
scratch is ever purged.

Usage::

    python build_pile.py --list                      # what exists / what's missing
    python build_pile.py                             # build every missing cell
    python build_pile.py --datasets coco_val         # just COCO's cells
    python build_pile.py --embedders siglip2,siglip2_l
    python build_pile.py --verify                    # load every cell, check geometry
    python build_pile.py --rebuildable               # can every cell still be REBUILT?
    python build_pile.py --manifest                  # (re)write MANIFEST.{json,md}
    python build_pile.py --provenance                # which device built each cell
    python build_pile.py --backfill-provenance       # fingerprint the pre-#3160 cells

This module is the CLI and the per-cell build loop; everything it does inside a
build lives in :mod:`pilebuild`, split by the question each part answers -- one
loader module per ``DATASETS[ds]["kind"]``, and separate modules for provenance,
geometry, the manifest and the read-only audit modes. The names below are
re-exported so that ``import build_pile`` stays the entry point every sibling
script and test already uses.

``--rebuildable`` is the complement to ``--verify``, and the two answer
different questions. ``--verify`` asks whether the cells on disk are usable;
``--rebuildable`` asks whether they could be produced again. A cell that loads
says nothing about whether it can be rebuilt -- the paths share no code -- which
is how the ``vg_box_*`` rebuild sat broken for eleven days behind a pile that
verified clean (#3297). Run it before trusting the pile to be purgeable.

``--verify`` is the guard the region-voting studies needed: it asserts that
every cell whose ``(dataset, embedder)`` pair claims region capability actually
carries ``patch_grid`` on its medias, and that no cell silently holds zero. It
also checks the *geometry*: boxes against the band their cell name claims, and
boxes against the frame -- the second because the first cannot see a box
corrupted before banding, since the band is derived from that same box and moves
with it (#3281).
"""

from __future__ import annotations

import argparse
import os
import time
from contextlib import contextmanager

import pile_config as pc

pc.setup_env()

from pilebuild.audit import list_cells, rebuildable, report_bands, verify  # noqa: E402
from pilebuild.boxscan import band_categories, load_box_scan_categories  # noqa: E402
from pilebuild.env import assert_vtscore_is_this_checkout, cells_io, log  # noqa: E402
from pilebuild.geometry import region_geometry_problems, scale_label_digest  # noqa: E402
from pilebuild.loaders import loader_for  # noqa: E402
from pilebuild.manifest import write_manifest  # noqa: E402
from pilebuild.provenance import (  # noqa: E402
    cell_fingerprint,
    effective_embed_batch_size,
    write_provenance,
)
from pilebuild.provenance_report import provenance_report  # noqa: E402
from pilebuild.vgsource import vg_image_paths  # noqa: E402

#: Names sibling scripts and tests import off this module. Spelled out so that
#: moving a definition into ``pilebuild/`` cannot quietly break an importer that
#: this repo's test suite does not cover (``precision/build_arm.py``,
#: ``fastproc/build_arm.py``, and the five ``pile/make_*.py`` sheet builders).
__all__ = [
    "assert_vtscore_is_this_checkout",
    "band_categories",
    "build_cell",
    "cell_fingerprint",
    "effective_embed_batch_size",
    "list_cells",
    "load_box_scan_categories",
    "main",
    "provenance_report",
    "rebuildable",
    "region_geometry_problems",
    "report_bands",
    "scale_label_digest",
    "verify",
    "vg_image_paths",
    "write_manifest",
    "write_provenance",
]

#: Pre-split spellings, kept because out-of-tree callers and older notebooks use
#: them. ``_band_categories`` in particular is what ``tests_lib`` calls today.
_band_categories = band_categories
_vg_image_paths = vg_image_paths
_cells_io = cells_io


@contextmanager
def _embed_batch_size(embedder: str):
    """Apply this embedder's batch size for the duration of the embed pass.

    The app reads ``VTSEARCH_EMBED_BATCH_SIZE`` per bulk call, so one build
    process can run each embedder at its own size rather than every model at
    the shipped default of 32. An explicit env var wins: someone who set one
    is tuning for the card in front of them, and the table cannot know that.

    Yields the size the pass will actually run at, for the provenance sidecar
    (#3683). It has to be read here rather than at write time: this is the only
    window in which the env var is set, and the number is not recoverable from
    the table afterwards once an explicit override is in play.
    """
    want = pc.embed_batch_size(embedder)
    if want is None or os.environ.get("VTSEARCH_EMBED_BATCH_SIZE", "").strip():
        yield effective_embed_batch_size(embedder)
        return
    os.environ["VTSEARCH_EMBED_BATCH_SIZE"] = str(want)
    log(f"  embed batch size {want}")
    try:
        yield effective_embed_batch_size(embedder)
    finally:
        os.environ.pop("VTSEARCH_EMBED_BATCH_SIZE", None)


def build_cell(dataset: str, embedder: str, force: bool = False) -> dict:
    """Build one cell, returning a summary record."""
    out = pc.cell_path(dataset, embedder)
    if out.exists() and not force:
        log(f"skip {dataset} x {embedder} (exists: {out.name})")
        return {"dataset": dataset, "embedder": embedder, "status": "exists"}

    if pc.EMBEDDERS.get(embedder, {}).get("gated") and not os.environ.get("HF_TOKEN"):
        log(f"SKIP {dataset} x {embedder}: HF_TOKEN unset (weights are licence-gated)")
        return {"dataset": dataset, "embedder": embedder, "status": "skipped_gated"}

    kind = pc.DATASETS[dataset]["kind"]
    log(f"=== build {dataset} x {embedder} ({kind}) ===")
    t0 = time.time()

    medias: dict[int, dict] = {}
    loader_for(dataset, kind).load(dataset, medias, embedder)
    log(f"  loaded {len(medias)} medias in {time.time() - t0:.0f}s")

    from vtscore.datasets.stages.embedding import embed_missing  # noqa: PLC0415

    t1 = time.time()
    with _embed_batch_size(embedder) as batch_size:
        embed_missing(medias, embedder)
    embed_s = time.time() - t1

    n_patch = sum(1 for m in medias.values() if m.get("patch_grid") is not None)
    nbytes = cells_io().dump_medias(medias, out)
    total_s = time.time() - t0
    log(
        f"  wrote {out.name}: {nbytes / 1e6:.0f} MB, {len(medias)} medias, "
        f"patch grids {n_patch}/{len(medias)}, embed {embed_s:.0f}s, total {total_s:.0f}s"
    )
    summary = {
        "dataset": dataset,
        "embedder": embedder,
        "status": "built",
        "n_medias": len(medias),
        "n_patch_grids": n_patch,
        "megabytes": round(nbytes / 1e6, 1),
        "embed_seconds": round(embed_s, 1),
        "wall_seconds": round(total_s, 1),
    }
    write_provenance(dataset, embedder, summary, medias, embed_batch_size=batch_size)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", help="comma-separated subset (default: all)")
    ap.add_argument("--embedders", help="comma-separated subset (default: all)")
    ap.add_argument("--force", action="store_true", help="rebuild cells that already exist")
    ap.add_argument("--list", action="store_true", help="show cell status and exit")
    ap.add_argument("--verify", action="store_true", help="load every cell and check geometry")
    ap.add_argument(
        "--rebuildable",
        action="store_true",
        help="run every dataset's selection step against its sources, embedding nothing",
    )
    ap.add_argument("--bands", action="store_true", help="report voted-box scale bands for boxed datasets")
    ap.add_argument("--manifest", action="store_true", help="(re)write the manifest and exit")
    ap.add_argument("--provenance", action="store_true", help="show which device built each cell")
    ap.add_argument(
        "--backfill-provenance",
        action="store_true",
        help="stamp a sidecar (fingerprint only, device unknown) on cells built before #3160",
    )
    args = ap.parse_args()

    pc.EMBEDDINGS.mkdir(parents=True, exist_ok=True)
    assert_vtscore_is_this_checkout()

    if args.list:
        list_cells()
        return 0
    if args.verify:
        return verify()
    if args.rebuildable:
        return rebuildable(args.datasets.split(",") if args.datasets else None)
    if args.bands:
        return report_bands()
    if args.manifest:
        write_manifest()
        return 0
    if args.provenance or args.backfill_provenance:
        return provenance_report(backfill=args.backfill_provenance)

    # `on_request` datasets are built only when NAMED. The pile's default sweep
    # is "everything the shared studies need"; a cell sized for one study is a
    # cost the sweep should not silently take on (see `vg_scale_deep`).
    datasets = (
        args.datasets.split(",")
        if args.datasets
        else [d for d, spec in pc.DATASETS.items() if not spec.get("on_request")]
    )
    embedders = args.embedders.split(",") if args.embedders else list(pc.EMBEDDERS)
    for bad in [d for d in datasets if d not in pc.DATASETS]:
        raise SystemExit(f"unknown dataset {bad!r}; known: {sorted(pc.DATASETS)}")
    for bad in [e for e in embedders if e not in pc.EMBEDDERS]:
        raise SystemExit(f"unknown embedder {bad!r}; known: {sorted(pc.EMBEDDERS)}")

    # A derived dataset joins the run whenever its parent is in it. `vg_scale`
    # rebuilt without `vg_scale_any` leaves the derived cell holding the
    # parent's previous labels, boxes and bands -- with the right media count
    # and the right vectors, so nothing looks wrong (#3281 shipped that way).
    # Pulling it in costs a relabel and no embedding pass, and `--force` is what
    # makes it actually happen: the derived cell already exists.
    derived = [
        d
        for d, spec in pc.DATASETS.items()
        if spec.get("kind") == "vg_scale_any" and d not in datasets and "vg_scale" in datasets
    ]
    if derived:
        log(f"including {', '.join(derived)}: derived from vg_scale, and stale the moment it is rebuilt")
        datasets += derived

    summaries = []
    for ds in datasets:
        for emb in embedders:
            summaries.append(build_cell(ds, emb, force=args.force))

    built = [s for s in summaries if s["status"] == "built"]
    log(f"done: {len(built)} built, {len(summaries) - len(built)} skipped")
    write_manifest()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
