#!/usr/bin/env python3
"""Was the old `vg_scale` contrast partly "is this a scene with stuff in it"?

That is #3667's central claim, and neither of the other two scripts can test it.
`cross_class_negatives_effect.py` counts images. `cross_class_negatives_difficulty.py`
ranks by a TEXT query, and a text query cannot learn a shortcut -- it can only
say whether the added negatives are semantically nearer the class.

A *trained* head can learn one, so this trains the head the benchmark trains:

1. Fit a linear head on the cell exactly as the old benchmark posed it --
   positives against the **old shared pool only**, which is all the pre-#3667
   cell could offer.
2. Score two held-out sets with it: unseen **old-pool** negatives, and the
   **added** negatives (#3667's cross-class images, never seen in training).
3. Read the false-positive rate on each, at a threshold pinned to 5% on the
   held-out old pool.

If the old contrast were substantially "a scene with stuff in it", the added
negatives -- every one of which contains a labelled object -- would sit on the
positive side of that head, and their FPR would be far above 5%. If it were
already "is there a bus", they sit with the rest of the negatives and the FPR
is about 5%. The ratio between the two is the size of the shortcut, in the only
units that matter: negatives the shipped benchmark would have scored wrong.

Folded 5 ways over the positives and the old pool together, so every number is
out-of-sample and the old-pool FPR is not the training set's own.

Usage::

    python cross_class_negatives_shortcut.py --json out.json
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
SEED = 0


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n else v


def media_vec(d: dict) -> np.ndarray:
    emb = d.get("embeddings") or {}
    return np.asarray(next(iter(emb.values())), dtype=np.float32)


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.concatenate([pos, neg])
    ranks = order.argsort().argsort().astype(np.float64)
    _, inv, counts = np.unique(order, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embedder", default="siglip")
    ap.add_argument("--before-dir", type=Path, default=BEFORE_DIR)
    ap.add_argument("--after-dir", type=Path, default=None)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--fpr", type=float, default=0.05, help="old-pool FPR the threshold is pinned to")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415

    after_dir = args.after_dir or pc.EMBEDDINGS
    name = f"vg_scale__{args.embedder}.pkl"
    before = load_medias(args.before_dir / name)
    after = load_medias(after_dir / name)

    ids = sorted(set(before) & set(after))
    idx = {i: k for k, i in enumerate(ids)}
    X = np.stack([unit(media_vec(after[i])) for i in ids])

    cells = [pc.scale_cell(c, b) for c in pc.SCALE_CLASSES for b in pc.BOX_BANDS]
    ev_b: dict[str, set[int]] = defaultdict(set)
    ev_a: dict[str, set[int]] = defaultdict(set)
    pos_of: dict[str, set[int]] = defaultdict(set)
    for i in ids:
        for cell in before[i].get("evaluable_categories") or []:
            ev_b[cell].add(i)
        for cell in after[i].get("evaluable_categories") or []:
            ev_a[cell].add(i)
        for cell in after[i].get("categories") or []:
            pos_of[cell].add(i)

    rng = np.random.RandomState(SEED)
    rows = []
    print(f"{'cell':<20}{'AUC old':>9}{'AUC added':>11}{'FPR old':>9}{'FPR added':>11}{'ratio':>8}")
    print("-" * 68)
    for cell in cells:
        p = np.array([idx[i] for i in sorted(pos_of[cell]) if i in idx])
        nb = np.array([idx[i] for i in sorted(ev_b[cell] - pos_of[cell]) if i in idx])
        add = np.array([idx[i] for i in sorted((ev_a[cell] - ev_b[cell]) - pos_of[cell]) if i in idx])
        if len(p) < 10 or len(nb) < 100 or not len(add):
            continue

        sp, sn, sa = np.zeros(len(p)), np.zeros(len(nb)), np.zeros((args.folds, len(add)))
        fp, fn = rng.permutation(len(p)) % args.folds, rng.permutation(len(nb)) % args.folds
        for k in range(args.folds):
            tr = np.concatenate([X[p[fp != k]], X[nb[fn != k]]])
            y = np.concatenate([np.ones((fp != k).sum()), np.zeros((fn != k).sum())])
            clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced").fit(tr, y)
            w = clf.coef_[0]
            b = float(clf.intercept_[0])
            sp[fp == k] = X[p[fp == k]] @ w + b
            sn[fn == k] = X[nb[fn == k]] @ w + b
            # The added negatives are never in training, so every fold scores
            # all of them; averaging the folds is the out-of-sample analogue of
            # the single held-out score the other two sets get.
            sa[k] = X[add] @ w + b
        sa_m = sa.mean(axis=0)

        thr = float(np.quantile(sn, 1 - args.fpr))
        fpr_old = float((sn > thr).mean())
        fpr_add = float((sa_m > thr).mean())
        a_old, a_add = auc(sp, sn), auc(sp, sa_m)
        rows.append(
            {
                "cell": cell,
                "n_pos": len(p),
                "n_old": len(nb),
                "n_added": len(add),
                "auc_old": a_old,
                "auc_added": a_add,
                "fpr_old": fpr_old,
                "fpr_added": fpr_add,
                "ratio": fpr_add / fpr_old if fpr_old else float("nan"),
            }
        )
        if cell.endswith("@medium"):
            print(f"{cell:<20}{a_old:>9.3f}{a_add:>11.3f}{fpr_old:>9.3f}{fpr_add:>11.3f}{rows[-1]['ratio']:>8.2f}")

    d_auc = np.array([r["auc_added"] - r["auc_old"] for r in rows])
    ratio = np.array([r["ratio"] for r in rows])
    se = float(d_auc.std(ddof=1) / np.sqrt(len(d_auc)))
    se_r = float(ratio.std(ddof=1) / np.sqrt(len(ratio)))
    print(f"\n{len(rows)} cells, {args.folds}-fold, threshold pinned to {args.fpr:.0%} FPR on the old pool")
    print(f"paired dAUC (added − old-pool):  {d_auc.mean():+.3f} ± {se:.3f}")
    print(f"FPR ratio (added ÷ old-pool):    {ratio.mean():.2f} ± {se_r:.2f}")
    print(
        f"mean FPR: old {np.mean([r['fpr_old'] for r in rows]):.3f} → added {np.mean([r['fpr_added'] for r in rows]):.3f}"
    )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "embedder": args.embedder,
                    "folds": args.folds,
                    "fpr_target": args.fpr,
                    "n_cells": len(rows),
                    "d_auc_mean": float(d_auc.mean()),
                    "d_auc_se": se,
                    "ratio_mean": float(ratio.mean()),
                    "ratio_se": se_r,
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
