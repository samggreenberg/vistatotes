"""The definitive Max-Patch voting run (one SLURM-array task per cell).

Each array task handles exactly one ``(dataset, embedder, category, seed)``
cell and runs *all* styles for that embedder inside it (they share the one
loaded pickle), so the MaxHAC / MaxPatch / whole-image trajectories are paired
on identical data, identical sim/test splits, and the identical startup
exemplar.  The cell for this task is ``array_cells(...)[SLURM_ARRAY_TASK_ID]``
- a stable enumeration derived from ``prepare_info.json``.

The startup sort mirrors the "train a new detector from an example" flow: the
cell's exemplar (a cropped positive, pre-embedded by ``prepare_data.py``) is
scored against the dataset **per style** - whole-image cosine, max cosine over
HAC region nodes, or max cosine over raw patches - and the Autopilot seed
phase votes down that ranking.

Results land in ``results/cells/task_<idx>.csv``; the summariser concatenates
whatever tasks completed.

Run directly with ``--index N`` for a single cell, or via SLURM with
``$SLURM_ARRAY_TASK_ID``.
"""

from __future__ import annotations

import argparse
import json
import os

import common

common.setup_env()

import numpy as np  # noqa: E402

import experiment_config as cfg  # noqa: E402


def _categories_by_dataset(prepare_info: dict) -> dict[str, dict[str, list[str]]]:
    out: dict[str, dict[str, list[str]]] = {}
    for ds, per_emb in prepare_info.get("datasets", {}).items():
        out[ds] = {emb: entry.get("selected_categories", []) for emb, entry in per_emb.items()}
    return out


def _seed_query_text(ds: str, cat: str) -> str:
    """The text a user would type to find *cat* in *ds*, or "" if none is known.

    Two tables, because there are two kinds of dataset.  ``EXPERIMENT_QUERIES``
    covers fixtures that exist only inside this experiment (``vg_scale``);
    ``vtscore.eval.config.EVAL_DATASETS`` covers the real demo datasets the app
    ships (``visual_genome_m``, ``caltech101_m``).  The experiment table wins so
    a fixture can override, but neither is required -- an unknown dataset simply
    has no query, and the autopilot seeds from known-goods instead.
    """
    try:
        local = cfg.EXPERIMENT_QUERIES.get(ds) or {}
    except AttributeError:
        local = {}
    if cat in local:
        return local[cat]

    from vtscore.eval.config import EVAL_DATASETS  # noqa: PLC0415

    info = EVAL_DATASETS.get(ds)
    if not info:
        return ""
    for query in info["queries"]:
        if query.target_category == cat:
            return query.text
    return ""


def _text_seed_scores(ds: str, emb: str, cat: str, medias: dict) -> "dict[int, float] | None":
    """The app's text sort: cosine from the typed query to every media.

    This is what a real user starts from -- they type "boat" and vote down the
    ranking -- and it is what ``seed_scores`` has always been documented to hold
    (``al_strategies``, ``EVAL.md``, ``voting_iterations`` all say "similarity to
    the typed query").

    Returns ``None`` when no query is defined for the cell, or when the embedder
    has no text tower -- DINOv3 does not, so ``embed_text`` is the base class's
    ``return None`` and ``embed_text_query`` yields nothing.  ``None`` is the
    signal for the autopilot to seed from *three random known-good examples*
    instead, the app's other real start ("3 random examples pulled from the
    Good").  Both are things a user does; ranking by cosine to a cropped box is
    not, which is why this no longer seeds from crops.
    """
    from vtscore.embedding.helpers import embed_text_query  # noqa: PLC0415
    from vtscore.embedding.media_vectors import media_embedding  # noqa: PLC0415

    text = _seed_query_text(ds, cat)
    if not text:
        return None
    qvec = embed_text_query(text, "image", enrich=cfg.SEED_ENRICH, embedder_name=emb)
    if qvec is None:
        return None

    def _unit(vec):
        v = np.asarray(vec, dtype=np.float32)
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-12 else v

    ids = list(medias.keys())
    if not ids:
        return None
    matrix = np.stack([_unit(media_embedding(medias[c])) for c in ids])
    cos = matrix @ _unit(qvec)
    return {ids[k]: float(cos[k]) for k in range(len(ids))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Max-Patch: one cell (dataset,embedder,category,seed).")
    parser.add_argument("--index", type=int, default=None, help="Cell index; defaults to $SLURM_ARRAY_TASK_ID.")
    parser.add_argument("--outdir", default=str(common.RESULTS / "cells"))
    parser.add_argument(
        "--print-cells", action="store_true", help="Print the total cell count and exit (for array sizing)."
    )
    args = parser.parse_args(argv)

    prepare_info = json.loads((common.RESULTS / "prepare_info.json").read_text())
    cells = cfg.array_cells(_categories_by_dataset(prepare_info))

    if args.print_cells:
        print(len(cells))
        return 0

    idx = args.index if args.index is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    if idx >= len(cells):
        common.log(f"index {idx} >= {len(cells)} cells; nothing to do")
        return 0
    cell = cells[idx]
    ds, emb, cat, seed = cell["dataset"], cell["embedder"], cell["category"], cell["seed"]
    styles = cfg.styles_for_embedder(emb)
    common.log(f"cell {idx}/{len(cells)}: dataset={ds} embedder={emb} category={cat} seed={seed} styles={styles}")

    import pandas as pd

    from vtscore.eval.voting_columns import VOTING_COLUMNS
    from vtscore.eval.voting_iterations import simulate_voting_iterations

    from vtscore.config import EMBEDDINGS_DIR  # isort: skip

    from _cells_io import load_medias  # noqa: PLC0415

    pkl = EMBEDDINGS_DIR / cfg.pickle_name(ds, emb)
    medias: dict[int, dict] = load_medias(pkl)
    common.log(f"loaded {len(medias)} medias from {pkl}")

    seed_scores = _text_seed_scores(ds, emb, cat, medias)
    seed_mode = "text" if seed_scores is not None else "known_good"
    seed_query = _seed_query_text(ds, cat) if seed_scores is not None else ""
    common.log(f"seed: mode={seed_mode} query={seed_query!r}")

    all_rows: list[dict] = []
    for style in styles:
        # The exemplar startup sort, computed in this style's own geometry.
        rows = simulate_voting_iterations(
            medias,
            target_category=cat,
            seed=seed,
            dataset_name=ds,
            inclusion=cfg.INCLUSION,
            sim_fraction=cfg.SIM_FRACTION,
            safe_thresholds=cfg.SAFE_THRESHOLDS,
            calibrate_count=cfg.CALIBRATE_COUNT,
            calibration_fraction=cfg.CALIBRATION_FRACTION,
            region_voting=cfg.REGION_VOTING,
            max_steps=cfg.MAX_STEPS,
            seed_scores=seed_scores,
            trainer="app",
            style=style,
        )
        for r in rows:
            r["embedder"] = emb
            r["seed_mode"] = seed_mode
            r["seed_query"] = seed_query
        all_rows.extend(rows)
        common.log(f"  style={style}: {len(rows)} rows")

    outdir = common.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"task_{idx:04d}.csv"
    columns = [*VOTING_COLUMNS, "embedder", "seed_mode", "seed_query"]
    pd.DataFrame(all_rows, columns=pd.Index(columns)).to_csv(out, index=False)
    common.log(f"wrote {len(all_rows)} rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
