"""Analyse the scale study: does cost rise as the target shrinks? (#3156)

The contrast is the **band**, and the design makes it a paired one: the same
twelve classes appear at all three sizes, against identical negatives. So every
comparison here is within `(class, seed, embedder)` and differs only in band --
which is exactly what the published `vg_box_*` sets could not do, since their
vocabularies are disjoint and a small-vs-large gap there confounds size with
class identity.

"Identical negatives" is now a within-class statement, and prevalence is no
longer 0.0250. #3667 gave each class the other eleven's COCO-exhaustive
positives as negatives, so the three bands of ONE class still share a negative
set (asserted per media by `cross_class_negatives_rebuilt.py`) while two
different classes no longer do, and the realised prevalence fell to **0.0172**.
The paired design this file rests on is intact; the level is not.

**Read the band effect measured here as a LOWER BOUND.** The negatives #3667
admitted are 2.5x harder at `@small` and no harder at `@large`, so the old
construction inflated exactly one end of this file's own axis (#3679).

Reported paired, with a standard error, to two significant digits. A difference
smaller than twice its SE is called unresolvable rather than dressed up: "not
resolvable at three seeds" is a finding, and a more useful one than a decimal
the sample cannot support.

Encoders are a **blocking factor**, not a contrast: the question is whether the
band effect survives all three, so each is reported separately and they are
never pooled into one number.

Usage::

    python analyze_scale.py --exp /expscratch/$USER/scale-3156
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path

from _cells_io import main_frame_files

BANDS = ("small", "medium", "large")


def freshness_report(cells_dir: Path, expected: int) -> tuple[int, list[str]]:
    """How many cells exist, and how many share the newest run's generation.

    File existence is not evidence of a result: a task that dies leaves its
    PREVIOUS output in place, so a directory can hold a full set of cells from
    two different runs and look complete. Cells more than a few hours older than
    the newest are reported as suspect rather than silently averaged in.
    """
    import time

    files = main_frame_files(cells_dir)
    if not files:
        return 0, []
    newest = max(f.stat().st_mtime for f in files)
    stale = [f.name for f in files if newest - f.stat().st_mtime > 6 * 3600]
    if not expected:
        print(f"cells present: {len(files)} (no grid_shape.json, so the expected count is unknown)")
    else:
        print(f"cells present: {len(files)} of {expected} expected")
    print(f"newest cell written: {time.strftime('%Y-%m-%d %H:%M', time.localtime(newest))}")
    if stale:
        print(f"WARNING: {len(stale)} cells are >6h older than the newest — from an earlier run?")
        print(f"         e.g. {', '.join(stale[:5])}")
    if expected and len(files) != expected:
        print(f"WARNING: {expected - len(files)} cells missing; results below are a SUBSET")
    return len(files), stale


def load_rows(cells_dir: Path) -> list[dict]:
    import csv

    rows = []
    dropped = 0
    files = main_frame_files(cells_dir)
    for f in files:
        if f.stat().st_size == 0:
            dropped += 1
            continue
        try:
            with f.open(newline="") as fh:
                rows.extend(list(csv.DictReader(fh)))
        except Exception:
            dropped += 1
    return rows, len(files), dropped


def band_of(category: str) -> str:
    return category.rsplit("@", 1)[1] if "@" in category else ""


def class_of(category: str) -> str:
    return category.rsplit("@", 1)[0]


def mean_se(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(xs) / n
    if n < 2:
        return m, float("nan")
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var / n)


def fmt(v: float, se: float | None = None) -> str:
    if v != v:
        return "n/a"
    if se is None or se != se:
        return f"{v:.2f}"
    return f"{v:.2f} ± {se:.2f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp", default=f"/expscratch/{os.environ.get('USER', 'sgreenberg')}/scale-3156")
    ap.add_argument("--at-step", type=int, default=150, help="votes spent at which to read the headline")
    # Read from the run, not baked in here. A literal belongs to whichever grid
    # was current when it was typed: 324 was the three-seed grid's, and against
    # the 3600-cell run that replaced it this printed "3600 of 324 expected" and
    # "WARNING: -3276 cells missing", which is a complete grid reported as a
    # subset. `launch_scale.sh` records the shape it launched.
    ap.add_argument("--expect", type=int, default=0, help="expected cell count; 0 = read results/grid_shape.json")
    ap.add_argument(
        "--extra-cells",
        default="",
        help="another cells dir to merge in, e.g. the same encoder with geometry OFF -- "
        "the control that separates region voting from encoder quality",
    )
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cells = Path(args.exp) / "results" / "cells"
    expect = args.expect
    if not expect:
        shape = Path(args.exp) / "results" / "grid_shape.json"
        try:
            expect = int(json.loads(shape.read_text())["n_cells"])
        except (OSError, ValueError, KeyError, TypeError):
            expect = 0
    freshness_report(cells, expect)
    rows, n_files, dropped = load_rows(cells)
    if args.extra_cells:
        extra, n_extra, d_extra = load_rows(Path(args.extra_cells))
        # Relabel so the control cannot be confused with the live arm: same
        # encoder, same cells, same seeds, geometry off.
        for r in extra:
            r["style"] = r.get("style", "") + "(control)"
        rows += extra
        n_files += n_extra
        dropped += d_extra
        print(f"merged {len(extra)} control rows from {args.extra_cells}")
    print(f"loaded {len(rows)} rows from {n_files} cell files ({dropped} unreadable)")

    # The endpoint of each trajectory: the last row at or before --at-step for
    # each (embedder, category, seed, style).
    last: dict[tuple, dict] = {}
    for r in rows:
        try:
            t = int(r["t"])
        except (KeyError, ValueError):
            continue
        if t > args.at_step:
            continue
        key = (r.get("embedder", ""), r["category"], r["seed"], r.get("style", ""))
        prev = last.get(key)
        if prev is None or int(prev["t"]) < t:
            last[key] = r

    # Encoder is the blocking factor and style is its arm (whole_image for the
    # single-vector encoders, max_patch for the patch one), so the two are kept
    # together as one label rather than pooled.
    per_band: dict[tuple[str, str], list[float]] = defaultdict(list)
    by_key: dict[tuple, float] = {}
    styles = set()
    for (emb, cat, seed, style), r in last.items():
        try:
            cost = float(r["cost"])
        except (KeyError, ValueError, TypeError):
            continue
        b = band_of(cat)
        if b not in BANDS:
            continue
        arm = f"{emb}/{style}" if style else emb
        styles.add(arm)
        per_band[(arm, b)].append(cost)
        by_key[(arm, class_of(cat), seed, b)] = cost

    print(f"arms present: {sorted(styles)}")
    print()
    print(f"=== cost at t={args.at_step}, by band (lower is better) ===")
    hdr = f"{'arm':<26}" + "".join(f"{b:>16}" for b in BANDS) + f"{'n':>6}"
    print(hdr)
    print("-" * len(hdr))
    for style in sorted(styles):
        line = f"{style:<26}"
        n = 0
        for b in BANDS:
            xs = per_band[(style, b)]
            n = max(n, len(xs))
            m, se = mean_se(xs)
            line += f"{fmt(m, se):>16}"
        print(line + f"{n:>6}")

    print()
    print("=== paired small - large, within (class, seed) ===")
    print(f"{'arm':<26}{'mean diff':>16}{'n pairs':>9}{'resolvable?':>14}")
    print("-" * 65)
    for style in sorted(styles):
        diffs = []
        for (st, cls, seed, b), v in by_key.items():
            if st != style or b != "small":
                continue
            other = by_key.get((st, cls, seed, "large"))
            if other is not None:
                diffs.append(v - other)
        m, se = mean_se(diffs)
        verdict = "yes" if (se == se and abs(m) > 2 * se) else "NOT RESOLVABLE"
        print(f"{style:<26}{fmt(m, se):>16}{len(diffs):>9}{verdict:>14}")

    ctrl = [a for a in styles if "(control)" in a]
    if ctrl:
        print()
        print("=== region voting vs the SAME encoder with geometry off, per band ===")
        print(f"{'band':<10}{'max_patch':>14}{'whole_image':>14}{'paired diff':>18}{'n':>5}")
        print("-" * 62)
        # Identify the region arm by its STYLE, not by its embedder's name: the
        # region arm is `siglip+dinov3_patch/max_patch` since #3276, and a
        # `startswith("dinov3")` test silently finds nothing and prints an empty
        # control table rather than failing.
        live = next((a for a in styles if a.endswith("/max_patch") and "(control)" not in a), None)
        base = ctrl[0]
        for b in BANDS:
            diffs = []
            for (st, cls, seed, bb), v in by_key.items():
                if st != live or bb != b:
                    continue
                other = by_key.get((base, cls, seed, b))
                if other is not None:
                    diffs.append(v - other)
            m, se = mean_se(diffs)
            lm, _ = mean_se(per_band[(live, b)])
            cm, _ = mean_se(per_band[(base, b)])
            print(f"{b:<10}{fmt(lm):>14}{fmt(cm):>14}{fmt(m, se):>18}{len(diffs):>5}")
        print("Negative = region voting costs less. Same encoder and cells on both")
        print("sides, so this isolates the geometry from the encoder.")

    print()
    print("Cost is the harness's operating-point cost; every comparison above is")
    print("paired on (class, seed) and differs only in band. A difference smaller")
    print("than twice its standard error is not resolvable at this seed count.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
