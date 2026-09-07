#!/usr/bin/env python3
"""Is a VG image's PROVENANCE readable off its embedding?

#3670 declines to draw every negative from the COCO-anchored half on the grounds
that the positives are only ~57% COCO-sourced, so an all-COCO negative pool would
let a detector separate positives from negatives on provenance rather than on
content -- VG draws its images from COCO and from YFCC100M, and those look
different.

That is an argument, not a measurement, and the whole composition decision turns
on it: if provenance is not linearly readable then the all-provable pool is free
and the negatives stop resting on VG's silence entirely.

So ask the vectors. Fit a linear probe to predict ``labels_exhaustive`` (i.e.
COCO-sourced) from the embedding, on balanced classes, and report cross-validated
AUC per embedder. Chance is 0.5.

Read the number as a bound on the shortcut, not as a verdict on the pool: a high
AUC says the signal EXISTS to be learned, and a low one says an all-provable pool
cannot be gamed this way even in principle.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts/experiments/pile")
sys.path.insert(0, "scripts/experiments/calibration")

import pile_config as pc  # noqa: E402
from _cells_io import load_medias  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_score  # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("provenance_probe.json")
SEED = 0


def vec(media: dict, embedder: str) -> np.ndarray | None:
    """The whole-image vector, without `or`-chaining numpy arrays.

    A truthiness test on an ndarray raises, so each candidate key is checked for
    presence explicitly. Patch cells carry a grid as well; this probe wants the
    single whole-image vector, which is what a text sort ranks against.
    """
    emb = media.get("embeddings")
    if not isinstance(emb, dict) or not emb:
        return None
    for key in (embedder, "whole_image"):
        if key in emb and emb[key] is not None:
            v = np.asarray(emb[key], dtype=np.float32).ravel()
            return v if v.size else None
    for value in emb.values():
        if value is None:
            continue
        v = np.asarray(value, dtype=np.float32).ravel()
        if v.size:
            return v
    return None


def main() -> None:
    rng = np.random.default_rng(SEED)
    report: dict[str, dict] = {}

    for embedder in ("siglip", "siglip2_l", "clip", "clip_l", "dinov3_patch"):
        pkl = pc.EMBEDDINGS / f"vg_scale__{embedder}.pkl"
        if not pkl.exists():
            report[embedder] = {"error": "cell missing"}
            continue
        m = load_medias(pkl)
        sample = next(iter(m.values()), {})
        print(f"{embedder}: keys={sorted((sample.get('embeddings') or {}).keys())}")

        X, y = [], []
        for iid, media in m.items():
            v = vec(media, embedder)
            if v is None:
                continue
            X.append(v)
            y.append(1 if media.get("labels_exhaustive") else 0)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int8)

        # Balance, so AUC is not read off a skewed prior.
        pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        k = min(len(pos), len(neg))
        if k < 50:
            report[embedder] = {"error": f"too few in one class ({len(pos)}/{len(neg)})"}
            continue
        keep = np.concatenate([rng.choice(pos, k, replace=False), rng.choice(neg, k, replace=False)])
        Xb, yb = X[keep], y[keep]
        # L2-normalise: these are cosine spaces, and an unnormalised probe can
        # read vector NORM, which is a different property from direction.
        Xb = Xb / np.clip(np.linalg.norm(Xb, axis=1, keepdims=True), 1e-8, None)

        clf = LogisticRegression(max_iter=2000, C=1.0)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        auc = cross_val_score(clf, Xb, yb, cv=cv, scoring="roc_auc")
        report[embedder] = {
            "n_balanced": int(2 * k),
            "n_coco": int(len(pos)),
            "n_off_coco": int(len(neg)),
            "auc_mean": round(float(auc.mean()), 4),
            "auc_sd": round(float(auc.std(ddof=1)), 4),
            "auc_folds": [round(float(a), 4) for a in auc],
        }
        print(f"{embedder:14s} AUC {auc.mean():.4f} +/- {auc.std(ddof=1):.4f}  (n={2 * k})")

    OUT.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
