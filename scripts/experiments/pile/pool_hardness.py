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
The difference is the quantity #3680 is about. AUC falls as a stratum gets
harder -- a perfectly separable negative set scores 1.0 -- so `delta` is
POSITIVE where the pool is the harder set (the indoor prediction) and NEGATIVE
where the cross-class images are.

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


def win_rates(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """Per-positive win rate against *neg*, ties at half.

    ``win_rates(p, n).mean()`` is exactly ``auc(p, n)``.  The vector form is
    what makes an interval possible: both AUCs a class reports are means over
    the SAME positives, so their difference is itself a per-positive quantity
    and its standard error follows directly, with the pairing intact.  A
    difference of two opaque scalars would have neither.
    """
    if not len(pos) or not len(neg):
        return np.full(len(pos), np.nan)
    order = np.sort(neg)
    lo = np.searchsorted(order, pos, side="left")
    hi = np.searchsorted(order, pos, side="right")
    return (lo + (hi - lo) / 2.0) / len(neg)


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
        medias = pickle.load(fh)  # noqa: S301 - our own artefact
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
        w_pool = win_rates(s[pos], s[pool])
        w_cross = win_rates(s[pos], s[cross])
        a_pool = float(w_pool.mean())
        a_cross = float(w_cross.mean())
        assert abs(a_pool - auc(s[pos], s[pool])) < 1e-9, "win_rates disagrees with auc()"
        # Paired per-positive difference.  This is the POSITIVE-side sampling
        # error only; at n_pos=300 against strata of 3,216+ it is the term that
        # dominates, but the interval is not a full resample of the negatives.
        per_pos = w_cross - w_pool
        delta = float(per_pos.mean())
        se = float(per_pos.std(ddof=1) / np.sqrt(len(per_pos))) if len(per_pos) > 1 else float("nan")
        rows.append(
            {
                "class": cls,
                "query": text,
                "n_pos": len(pos),
                "n_pool": len(pool),
                "n_cross": len(cross),
                "auc_vs_pool": a_pool,
                "auc_vs_cross": a_cross,
                # Positive => the POOL is the harder stratum for this class:
                # a_pool is the LOWER AUC, so the pool separates less well.
                "delta": delta,
                "se": se,
                "ci_lo": delta - 1.96 * se,
                "ci_hi": delta + 1.96 * se,
            }
        )

    rows.sort(key=lambda r: -r["delta"])
    print("\n" + "=" * 107)
    print("WHICH NEGATIVES ARE HARDER, PER CLASS -- ranked by the class's own text query")
    print("`delta` = AUC(vs cross-class) - AUC(vs pool). POSITIVE means the shared POOL is harder.")
    print("95% CI on the paired per-positive difference; `*` marks an interval excluding 0.")
    print("=" * 107)
    print(
        f"{'class':<14}{'n_pos':>7}{'n_pool':>8}{'n_cross':>9}"
        f"{'AUC vs pool':>13}{'AUC vs cross':>14}{'delta':>9}{'95% CI':>19}{'':>3}"
    )
    for r in rows:
        ci = f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]"
        sig = "*" if (r["ci_lo"] > 0 or r["ci_hi"] < 0) else ""
        print(
            f"{r['class']:<14}{r['n_pos']:>7}{r['n_pool']:>8}{r['n_cross']:>9}"
            f"{r['auc_vs_pool']:>13.3f}{r['auc_vs_cross']:>14.3f}{r['delta']:>+9.3f}{ci:>19}{sig:>3}"
        )
    d = np.array([r["delta"] for r in rows])
    pool_sig = [r["class"] for r in rows if r["ci_lo"] > 0]
    cross_sig = [r["class"] for r in rows if r["ci_hi"] < 0]
    print(f"\nspread: {d.min():+.3f} to {d.max():+.3f}, mean {d.mean():+.3f}")
    print(f"POOL harder by point estimate:  {int((d > 0).sum())} of {len(d)}")
    print(f"  ...with a CI excluding 0:     {len(pool_sig)} -- {', '.join(pool_sig) or '(none)'}")
    print(f"CROSS-CLASS harder, CI excl 0:  {len(cross_sig)} of {len(d)}")

    if args.json:
        Path(args.json).write_text(json.dumps({"cell": str(args.cell), "rows": rows}, indent=1) + "\n")
        log(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
