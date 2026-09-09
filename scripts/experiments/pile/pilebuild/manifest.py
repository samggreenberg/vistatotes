"""``MANIFEST.json`` + ``MANIFEST.md``: what the pile holds and how to rebuild it."""

from __future__ import annotations

import json

import pile_config as pc

from pilebuild.env import cells_io, log


def _manifest_provenance(dataset: str, embedder: str) -> dict:
    """The provenance fields the manifest carries per cell, or nulls if unknown."""
    path = pc.provenance_path(dataset, embedder)
    if not path.exists():
        return {"gpu_name": None, "built_by": None, "repo": None, "commit": None, "vectors_sha256": None}
    rec = json.loads(path.read_text())
    dev = rec.get("device", {})
    # `code` since #3693; before that the commit lived under `device` and no
    # sidecar recorded the checkout at all. Read both, so a pile holding cells
    # from either era reports what each one actually knows.
    code = rec.get("code", {})
    return {
        "gpu_name": dev.get("gpu_name"),
        "built_by": dev.get("hostname"),
        "repo": code.get("repo"),
        "commit": code.get("commit") or dev.get("commit"),
        "vectors_sha256": rec.get("fingerprint", {}).get("vectors_sha256"),
    }


def write_manifest() -> None:
    """Write MANIFEST.json + MANIFEST.md describing the pile and how to rebuild it."""
    io = cells_io()
    entries = []
    for ds, emb in pc.cells():
        path = pc.cell_path(ds, emb)
        if not path.exists():
            entries.append({"dataset": ds, "embedder": emb, "present": False})
            continue
        medias = io.load_medias(path)
        n = len(medias)
        entries.append(
            {
                "dataset": ds,
                "embedder": emb,
                "present": True,
                "file": path.name,
                "megabytes": round(path.stat().st_size / 1e6, 1),
                "n_medias": n,
                "n_patch_grids": sum(1 for m in medias.values() if m.get("patch_grid") is not None),
                "region_capable": pc.region_capable(ds, emb),
                # Which machine built it (#3160). None for cells that predate the
                # sidecar; a null here is a fact about the pile, not a gap to hide.
                **_manifest_provenance(ds, emb),
            }
        )

    doc = {
        "pile": str(pc.PILE),
        "sources": {
            "demo_cache": str(pc.DEMO_CACHE),
            "coco_root": str(pc.COCO_ROOT),
        },
        "datasets": pc.DATASETS,
        "embedders": pc.EMBEDDERS,
        "cells": entries,
    }
    (pc.PILE / "MANIFEST.json").write_text(json.dumps(doc, indent=2) + "\n")

    present = [e for e in entries if e["present"]]
    total_mb = sum(e["megabytes"] for e in present)
    lines = [
        "# Pre-embedded pile",
        "",
        f"`{pc.PILE}` — {len(present)}/{len(entries)} cells, {total_mb / 1000:.1f} GB of embeddings.",
        "",
        "Point a study at it with:",
        "",
        "```bash",
        f'export VTSEARCH_DATA_DIR="{pc.DATADIR}"',
        f'export VTSEARCH_MODELS_DIR="{pc.MODELS}"',
        "```",
        "",
        "## Cells",
        "",
        "| dataset | embedder | medias | patch grids | region-voting | size |",
        "|---|---|---:|---:|:--:|---:|",
    ]
    for e in entries:
        if not e["present"]:
            lines.append(f"| `{e['dataset']}` | `{e['embedder']}` | — | — | — | *missing* |")
            continue
        region = "**yes**" if e["region_capable"] else "no"
        lines.append(
            f"| `{e['dataset']}` | `{e['embedder']}` | {e['n_medias']} | "
            f"{e['n_patch_grids']} | {region} | {e['megabytes']:.0f} MB |"
        )
    lines += [
        "",
        "**Region voting needs both halves**: ground-truth boxes (dataset) *and* a patch",
        "grid (embedder). A boxed dataset on a single-vector embedder silently runs as",
        "binary voting — the failure behind #2877, #2897 and #2905. `build_pile.py --verify`",
        "asserts the geometry rather than trusting the arm table.",
        "",
        "## Rebuilding",
        "",
        "Scratch is treated as purgeable. Every cell rebuilds from staged, non-scratch",
        "sources, so the pile is disposable:",
        "",
        "```bash",
        "python build_pile.py            # rebuild whatever is missing (idempotent)",
        "python build_pile.py --verify   # check geometry after a rebuild",
        "```",
        "",
        f"Sources: demo datasets from `{pc.DEMO_CACHE}`, COCO from `{pc.COCO_ROOT}`.",
        "",
    ]
    (pc.PILE / "MANIFEST.md").write_text("\n".join(lines))
    log(f"wrote MANIFEST.json + MANIFEST.md ({len(present)}/{len(entries)} cells)")
