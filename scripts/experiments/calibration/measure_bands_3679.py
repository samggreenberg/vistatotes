#!/usr/bin/env python3
"""Emit every number #3679's report quotes, as one JSON, from both runs' cells.

Lives beside `analyze_scale.py`, not in the report directory: it imports that
module, and deptry scans `docs/`, so a sibling import from there reads as an
undeclared dependency and blocks the suite for every branch.

Run on the GRID (needs 96G -- 8.7M rows):

    python measure_bands.py > /dev/null

Writes ``measurements/band_effect.json`` beside this script. Statistic is
``analyze_scale.py``'s own: same endpoint rule (last row at or before t=150),
same pairing within (class, seed), same mean/SE -- imported rather than
reimplemented so the report cannot drift from the analyser.
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

CALIB = Path(__file__).resolve().parent
sys.path.insert(0, str(CALIB))
os.chdir(CALIB)
import analyze_scale as A  # noqa: E402

HERE = CALIB.parents[2] / "docs/experiments/2026-09-07-band-remeasure-3679"
AT = 150
TWELVE = {
    "backpack",
    "bicycle",
    "bird",
    "boat",
    "book",
    "bus",
    "clock",
    "dog",
    "kite",
    "knife",
    "stop sign",
    "umbrella",
}

RUNS = {
    "remeasure": "/expscratch/sgreenberg/scale-3679",
    "baseline": "/expscratch/sgreenberg/scale-3156-map",
}


def endpoints(exp: str) -> dict:
    rows, nfiles, dropped = A.load_rows(Path(exp) / "results" / "cells")
    last: dict = {}
    for r in rows:
        try:
            t = int(r["t"])
        except (KeyError, ValueError):
            continue
        if t > AT:
            continue
        k = (r.get("embedder", ""), r["category"], r["seed"], r.get("style", ""))
        p = last.get(k)
        if p is None or int(p["t"]) < t:
            last[k] = r
    out = {}
    for (emb, cat, seed, style), r in last.items():
        b = A.band_of(cat)
        if b not in A.BANDS:
            continue
        try:
            out[(f"{emb}/{style}" if style else emb, A.class_of(cat), seed, b)] = float(r["cost"])
        except (KeyError, ValueError, TypeError):
            continue
    return {"cells": out, "n_files": nfiles, "n_rows": len(rows), "dropped": dropped}


def summarise(cells: dict, classes: set[str] | None) -> dict:
    per_band = defaultdict(list)
    for (arm, cls, seed, b), c in cells.items():
        if classes is not None and cls not in classes:
            continue
        per_band[(arm, b)].append(c)
    diffs = defaultdict(list)
    for arm, cls, seed, b in cells:
        if b != "small" or (classes is not None and cls not in classes):
            continue
        lg = cells.get((arm, cls, seed, "large"))
        if lg is not None:
            diffs[arm].append(cells[(arm, cls, seed, "small")] - lg)
    arms = sorted({a for a, _ in per_band})
    return {
        arm: {
            "by_band": {b: dict(zip(("mean", "se"), A.mean_se(per_band[(arm, b)]))) for b in A.BANDS},
            "n_cells": max(len(per_band[(arm, b)]) for b in A.BANDS),
            "small_minus_large": dict(zip(("mean", "se"), A.mean_se(diffs[arm]))),
            "n_pairs": len(diffs[arm]),
        }
        for arm in arms
    }


def main() -> None:
    out = {"at_step": AT, "twelve": sorted(TWELVE), "runs": {}}
    for name, exp in RUNS.items():
        e = endpoints(exp)
        shape = json.loads((Path(exp) / "results" / "grid_shape.json").read_text())
        out["runs"][name] = {
            "exp": exp,
            "n_cell_files": e["n_files"],
            "n_rows": e["n_rows"],
            "grid_shape": {k: shape[k] for k in ("n_cells", "n_seeds", "embedders") if k in shape},
            "all_classes": summarise(e["cells"], None),
            "twelve_only": summarise(e["cells"], TWELVE),
        }
    (HERE / "measurements").mkdir(parents=True, exist_ok=True)
    p = HERE / "measurements" / "band_effect.json"
    p.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
