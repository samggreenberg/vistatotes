#!/usr/bin/env python3
"""Does the scene qualifier in three queries drive #3680's pool-harder result?

Only three of the twenty-five typed queries name a scene: `a boat on the water`,
`a kite in the sky` and `a car on the street`. Two of those are in the five
classes that find the shared pool harder, and the pool is exactly "images
holding none of C" -- so a query naming water or sky targets the stratum the
pool retains. That is a confound the main result cannot rule out on its own.

This re-ranks those three classes with the scene term stripped and nothing else
changed. If `boat` and `kite` keep a positive delta, the qualifier is not what
put them there.

Lives beside `pool_hardness.py` rather than in the report directory. It needs the
GRID, the pile and this repo's sibling modules, where a report directory holds
only a `figures.py` that rebuilds from `measurements/` alone -- and deptry scans
`docs/`, so a sibling import from there reads as an undeclared dependency and
blocks the suite for every branch.

Statistic is `pool_hardness.py`'s own -- `win_rates` and `auc` are imported, not
reimplemented.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "calibration"))

import numpy as np  # noqa: E402
import pile_config as pc  # noqa: E402
import pool_hardness as ph  # noqa: E402  (runs pc.setup_env() on import)

BARE = {"boat": "a boat", "kite": "a kite", "car": "a car"}
OUT = HERE.parents[2] / "docs/experiments/2026-09-07-pool-hardness-3680" / "measurements" / "query_ablation.json"


def main() -> int:
    from experiment_config import seed_query_text  # noqa: PLC0415
    from vtscore.embedding import embed_text_query  # noqa: PLC0415
    from vtscore.embedding.media_vectors import media_embedding  # noqa: PLC0415

    cell = pc.EMBEDDINGS / "vg_scale__siglip.pkl"
    with cell.open("rb") as fh:
        medias = pickle.load(fh)  # noqa: S301 - our own artefact
    if isinstance(medias, dict) and "medias" in medias:
        medias = medias["medias"]
    ids = sorted(medias)
    idx = {i: n for n, i in enumerate(ids)}
    vecs = np.stack([media_embedding(medias[i], "siglip") for i in ids]).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12

    rows = []
    for cls, bare in BARE.items():
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
        shipped = seed_query_text("vg_scale", pc.scale_cell(cls, "medium"))
        out = {"class": cls, "shipped_query": shipped, "bare_query": bare}
        for tag, text in (("shipped", shipped), ("bare", bare)):
            q = np.asarray(embed_text_query(text, "image", embedder_name="siglip"), dtype=np.float32)
            q /= np.linalg.norm(q) + 1e-12
            s = vecs @ q
            wp = ph.win_rates(s[pos], s[pool])
            wc = ph.win_rates(s[pos], s[cross])
            d = wc - wp
            delta = float(d.mean())
            se = float(d.std(ddof=1) / np.sqrt(len(d)))
            out[tag] = {
                "auc_vs_pool": float(wp.mean()),
                "auc_vs_cross": float(wc.mean()),
                "delta": delta,
                "se": se,
                "ci_lo": delta - 1.96 * se,
                "ci_hi": delta + 1.96 * se,
            }
        out["shift"] = out["bare"]["delta"] - out["shipped"]["delta"]
        rows.append(out)
        print(
            f"{cls:6s} shipped {out['shipped']['delta']:+.3f}  "
            f"bare {out['bare']['delta']:+.3f}  shift {out['shift']:+.3f}"
        )

    OUT.write_text(json.dumps({"cell": str(cell), "rows": rows}, indent=1) + "\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
