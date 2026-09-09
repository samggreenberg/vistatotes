#!/usr/bin/env python3
"""Are #3667's new negatives actually HARDER, or just more numerous?

#3667's argument was not about supply. It was that the two sides of every
`vg_scale` contrast were drawn from different distributions -- positives are
images containing a labelled object, negatives are images containing none of
twelve common classes -- so a detector could score well by learning *"is this a
scene with stuff in it"* rather than *"is there a bus"*. The images added as
negatives are the cluttered, realistic ones the benchmark had none of.

That claim is testable with no training at all, using the free text sort the app
gives a user for typing the class name. The measurement is deliberately
prevalence-free:

- **AUC** ranks positives against negatives and does not move when you add
  negatives that are exactly as hard as the ones already there. A drop is the
  claim being true.
- **AP** is reported beside it because it is what a ship decision reads -- and
  it falls mechanically when prevalence falls, so it cannot be read as
  difficulty on its own.
- **AUC against the added negatives ALONE**, versus AUC against the old shared
  pool alone, is the direct statement: same positives, same query, two negative
  sets, one number each.

Scoring is unit-normalised cosine against the text vector, which is exactly what
`patch_styles`'s whole-image `exemplar_sims` does, so this is the geometry the
harness ranks in rather than a separate one invented here.

Usage::

    python cross_class_negatives_difficulty.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "calibration"))

import pile_config as pc  # noqa: E402

pc.setup_env()

import numpy as np  # noqa: E402

from _cells_io import load_medias  # noqa: E402

BEFORE_DIR = Path("/expscratch/sgreenberg/archive/pre-3667-vg_scale")


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n else v


def media_vec(d: dict) -> np.ndarray:
    emb = d.get("embeddings") or {}
    for key in ("image", "clip", "default"):
        if key in emb:
            return np.asarray(emb[key], dtype=np.float32)
    # One embedding per media in a pile cell; take whatever it is called.
    return np.asarray(next(iter(emb.values())), dtype=np.float32)


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney), ties counted as half."""
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.concatenate([pos, neg])
    ranks = order.argsort().argsort().astype(np.float64)
    # average ranks for ties
    _, inv, counts = np.unique(order, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg)))


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(-scores)
    y = labels[order]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    n_pos = int(y.sum())
    return float((prec * y).sum() / n_pos) if n_pos else float("nan")


def main() -> int:
    ap_ = argparse.ArgumentParser(description=__doc__)
    ap_.add_argument("--embedder", default="siglip")
    ap_.add_argument("--before-dir", type=Path, default=BEFORE_DIR)
    ap_.add_argument("--after-dir", type=Path, default=None)
    ap_.add_argument("--json", type=Path, default=None)
    args = ap_.parse_args()

    after_dir = args.after_dir or pc.EMBEDDINGS
    name = f"vg_scale__{args.embedder}.pkl"
    before = load_medias(args.before_dir / name)
    after = load_medias(after_dir / name)

    from vtscore.embedding.helpers import embed_text_query  # noqa: PLC0415

    cells = [pc.scale_cell(c, b) for c in pc.SCALE_CLASSES for b in pc.BOX_BANDS]
    ids = sorted(after)
    idx = {i: k for k, i in enumerate(ids)}
    matrix = np.stack([unit(media_vec(after[i])) for i in ids])

    qvec = {}
    for c in pc.SCALE_CLASSES:
        v = embed_text_query(c, "image", enrich=False, embedder_name=args.embedder)
        if v is None:
            print(f"{args.embedder} has no text tower; nothing to measure", file=sys.stderr)
            return 2
        qvec[c] = unit(np.asarray(v, dtype=np.float32))

    ev_b: dict[str, set[int]] = defaultdict(set)
    ev_a: dict[str, set[int]] = defaultdict(set)
    pos_of: dict[str, set[int]] = defaultdict(set)
    for i in ids:
        for cell in before.get(i, {}).get("evaluable_categories") or []:
            ev_b[cell].add(i)
        for cell in after[i].get("evaluable_categories") or []:
            ev_a[cell].add(i)
        for cell in after[i].get("categories") or []:
            pos_of[cell].add(i)

    rows = []
    print(f"{'cell':<20}{'AUC old':>9}{'AUC new':>9}{'d':>8}{'AUC add':>9}{'AP old':>8}{'AP new':>8}")
    print("-" * 71)
    for cell in cells:
        c = cell.split("@", 1)[0]
        sims = matrix @ qvec[c]
        p = np.array([idx[i] for i in sorted(pos_of[cell])])
        nb = np.array([idx[i] for i in sorted(ev_b[cell] - pos_of[cell])])
        na = np.array([idx[i] for i in sorted(ev_a[cell] - pos_of[cell])])
        added = np.array([idx[i] for i in sorted((ev_a[cell] - ev_b[cell]) - pos_of[cell])])
        if not len(p) or not len(nb) or not len(na):
            continue
        a_b, a_a = auc(sims[p], sims[nb]), auc(sims[p], sims[na])
        a_add = auc(sims[p], sims[added]) if len(added) else float("nan")
        ap_b = average_precision(
            np.concatenate([sims[p], sims[nb]]), np.concatenate([np.ones(len(p)), np.zeros(len(nb))])
        )
        ap_a = average_precision(
            np.concatenate([sims[p], sims[na]]), np.concatenate([np.ones(len(p)), np.zeros(len(na))])
        )
        rows.append(
            {
                "cell": cell,
                "n_pos": len(p),
                "n_neg_old": len(nb),
                "n_neg_new": len(na),
                "n_added": len(added),
                "auc_old": a_b,
                "auc_new": a_a,
                "auc_added_only": a_add,
                "ap_old": ap_b,
                "ap_new": ap_a,
            }
        )
        if cell.endswith("@medium"):
            print(f"{cell:<20}{a_b:>9.3f}{a_a:>9.3f}{a_a - a_b:>+8.3f}{a_add:>9.3f}{ap_b:>8.3f}{ap_a:>8.3f}")

    d_auc = np.array([r["auc_new"] - r["auc_old"] for r in rows])
    d_add = np.array([r["auc_added_only"] - r["auc_old"] for r in rows])
    se = float(d_auc.std(ddof=1) / np.sqrt(len(d_auc)))
    se_add = float(d_add.std(ddof=1) / np.sqrt(len(d_add)))
    print(f"\n{len(rows)} cells")
    print(f"paired dAUC (old pool -> new pool):   {d_auc.mean():+.3f} +- {se:.3f}")
    print(f"paired dAUC (old pool -> ADDED only): {d_add.mean():+.3f} +- {se_add:.3f}")
    print(f"mean AP {np.mean([r['ap_old'] for r in rows]):.3f} -> {np.mean([r['ap_new'] for r in rows]):.3f}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "embedder": args.embedder,
                    "n_cells": len(rows),
                    "d_auc_mean": float(d_auc.mean()),
                    "d_auc_se": se,
                    "d_auc_added_mean": float(d_add.mean()),
                    "d_auc_added_se": se_add,
                    "cells": rows,
                },
                indent=1,
            )
            + "\n"
        )
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
