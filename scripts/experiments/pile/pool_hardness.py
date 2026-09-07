#!/usr/bin/env python3
"""Is the shared negative pool doing the same work for every class? (#3680)

#3667's rebuild measured, in aggregate, that admitting other classes' positives
as negatives makes a class's contrast harder. Two classes went the other way:
`knife` at 0.15 and `book` at 0.49 found the *newly admitted* negatives **easier**
than the pool they already had, where `umbrella` and `dog` found them 3.4x and
2.5x harder.

The hypothesis #3680 offers for that is about what the pool is made of. It is
"images holding none of *C*", which on VG is largely rooms, desks, counters and
kitchens -- the exact contexts a knife or a book lives in. So for indoor
tabletop classes the pool was **already** the hard-negative set, and what #3667
hands them is buses, boats and kites, which are trivially separable. If that is
right, one shared pool is doing two different jobs, and its value is not uniform
across the class list.

This tests it directly, on one cell, with no training and no GPU beyond a single
text-tower call per class. For each class the medias split three ways:

* **positives** -- the images designated for any of its bands;
* **pool** -- the shared negatives, images holding none of *C*;
* **cross-class** -- images that are some *other* class's positive and are
  evaluable in this class's cells (#3667's admission).

Ranked by the class's own text query, `AUC(positives, pool)` and
`AUC(positives, cross-class)` say which stratum is harder **for that class**.
The difference is the quantity #3680 is about: negative where the pool is the
harder set (the indoor prediction), positive where the cross-class images are.

Deliberately prevalence-free. AUC does not move when a stratum merely gets
bigger, which is what separates "these negatives are harder" from "there are
more of them" -- the same reason `cross_class_negatives_difficulty.py` reads AUC
rather than AP.

**One shared pool, but not one shared question.** Read a large spread across
classes as evidence that the construction is choosing a different difficulty for
each class without saying so, and see the recommendation the report draws from it.

Usage::

    python pool_hardness.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "calibration"))

import numpy as np
import pile_config as pc

pc.setup_env()


def log(msg: str) -> None:
    print(f"[hardness] {msg}", flush=True)


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Rank AUC of *pos* over *neg*, ties counted at half."""
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    r = ranks[: len(pos)].sum()
    return float((r - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", default=str(pc.EMBEDDINGS / "vg_scale__siglip.pkl"))
    ap.add_argument("--embedder", default="siglip")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    from experiment_config import seed_query_text  # noqa: PLC0415
    from vtscore.embedding import embed_text_query  # noqa: PLC0415
    from vtscore.embedding.media_vectors import media_embedding  # noqa: PLC0415

    with Path(args.cell).open("rb") as fh:
        medias = pickle.load(fh)
    if isinstance(medias, dict) and "medias" in medias:
        medias = medias["medias"]
    log(f"{Path(args.cell).name}: {len(medias)} medias")

    ids = sorted(medias)
    idx = {i: n for n, i in enumerate(ids)}
    vecs = np.stack([media_embedding(medias[i], args.embedder) for i in ids]).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12

    rows = []
    for cls in pc.SCALE_CLASSES:
        own = {pc.scale_cell(cls, b) for b in pc.BOX_BANDS}
        pos, pool, cross = [], [], []
        for i in ids:
            d = medias[i]
            cats = set(d.get("categories") or [])
            ev = set(d.get("evaluable_categories") or [])
            if cats & own:
                pos.append(idx[i])
            elif not cats:
                if ev & own:
                    pool.append(idx[i])
            elif ev & own:
                cross.append(idx[i])
        text = seed_query_text("vg_scale", pc.scale_cell(cls, "medium"))
        if not text:
            log(f"  {cls}: no seed query, skipped")
            continue
        q = np.asarray(embed_text_query(text, "image", embedder_name=args.embedder), dtype=np.float32)
        q /= np.linalg.norm(q) + 1e-12
        s = vecs @ q
        a_pool = auc(s[pos], s[pool])
        a_cross = auc(s[pos], s[cross])
        rows.append(
            {
                "class": cls,
                "query": text,
                "n_pos": len(pos),
                "n_pool": len(pool),
                "n_cross": len(cross),
                "auc_vs_pool": a_pool,
                "auc_vs_cross": a_cross,
                # Negative => the POOL is the harder stratum for this class.
                "delta": a_cross - a_pool,
            }
        )

    rows.sort(key=lambda r: r["delta"])
    print("\n" + "=" * 92)
    print("WHICH NEGATIVES ARE HARDER, PER CLASS -- ranked by the class's own text query")
    print("`delta` = AUC(vs cross-class) - AUC(vs pool). NEGATIVE means the shared POOL is harder.")
    print("=" * 92)
    print(f"{'class':<14}{'n_pos':>7}{'n_pool':>8}{'n_cross':>9}{'AUC vs pool':>13}{'AUC vs cross':>14}{'delta':>9}")
    for r in rows:
        print(
            f"{r['class']:<14}{r['n_pos']:>7}{r['n_pool']:>8}{r['n_cross']:>9}"
            f"{r['auc_vs_pool']:>13.3f}{r['auc_vs_cross']:>14.3f}{r['delta']:>+9.3f}"
        )
    d = np.array([r["delta"] for r in rows])
    print(f"\nspread: {d.min():+.3f} to {d.max():+.3f}, mean {d.mean():+.3f}")
    print(f"classes where the POOL is the harder set: {int((d < 0).sum())} of {len(d)}")

    if args.json:
        Path(args.json).write_text(json.dumps({"cell": str(args.cell), "rows": rows}, indent=1) + "\n")
        log(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
