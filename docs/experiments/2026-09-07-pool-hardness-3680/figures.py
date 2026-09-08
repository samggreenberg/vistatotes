#!/usr/bin/env python3
"""Figures for #3680, from the JSON `pool_hardness.py` wrote.

    python figures.py            # regenerate every figure into figures/

The input lives beside this script under `measurements/`, so the figures rebuild
without the GRID, the pile, or a GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
MEAS = HERE / "measurements"
FIGS = HERE / "figures"
DPI = 130

POOL_HARDER = "#2b7bba"  # delta > 0
CROSS_HARDER = "#e8a33d"  # delta < 0
NULL = "#9aa0a6"


def load() -> list[dict]:
    rows = json.loads((MEAS / "pool_hardness.json").read_text())["rows"]
    return sorted(rows, key=lambda r: -r["delta"])


def colour(r: dict) -> str:
    if r["ci_lo"] > 0:
        return POOL_HARDER
    if r["ci_hi"] < 0:
        return CROSS_HARDER
    return NULL


def fig_delta(rows: list[dict]) -> None:
    """Ranked per-class delta with its 95% interval."""
    fig, ax = plt.subplots(figsize=(7.6, 7.2))
    ys = range(len(rows))
    for y, r in zip(ys, rows):
        c = colour(r)
        ax.plot([r["ci_lo"], r["ci_hi"]], [y, y], color=c, lw=2.2, solid_capstyle="round")
        ax.plot([r["delta"]], [y], "o", color=c, ms=5.5, zorder=3)
    ax.axvline(0, color="#444", lw=1.0, ls="--", zorder=1)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([r["class"] for r in rows], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("delta  =  AUC(vs cross-class)  −  AUC(vs pool)", fontsize=10)
    ax.set_title(
        "Which negatives are harder, per class\n"
        "right of zero: the shared POOL is harder   •   left: the CROSS-CLASS admission is",
        fontsize=10.5,
    )
    ax.text(0.012, 0.6, "pool harder", color=POOL_HARDER, fontsize=9, fontweight="bold")
    ax.text(-0.062, 0.6, "cross-class harder", color=CROSS_HARDER, fontsize=9, fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIGS / "delta-per-class.png", dpi=DPI)
    plt.close(fig)


def fig_scatter(rows: list[dict]) -> None:
    """The two strata against each other; distance from the diagonal IS delta."""
    fig, ax = plt.subplots(figsize=(6.6, 6.4))
    lo = min(min(r["auc_vs_pool"], r["auc_vs_cross"]) for r in rows) - 0.02
    hi = max(max(r["auc_vs_pool"], r["auc_vs_cross"]) for r in rows) + 0.02
    ax.plot([lo, hi], [lo, hi], color="#444", lw=1.0, ls="--", zorder=1, label="equally hard")
    # Labels collide in the crowded 0.87-0.93 band, so alternate the offset by
    # rank rather than letting three class names land on the same pixels.
    for i, r in enumerate(rows):
        ax.plot([r["auc_vs_pool"]], [r["auc_vs_cross"]], "o", color=colour(r), ms=6, zorder=3)
        dx, dy = ((6, 4), (6, -9), (-8, 6), (-8, -10))[i % 4]
        ax.annotate(
            r["class"],
            (r["auc_vs_pool"], r["auc_vs_cross"]),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=7.5,
            color="#333",
        )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("AUC vs the shared pool   (lower = harder)", fontsize=10)
    ax.set_ylabel("AUC vs the cross-class admission   (lower = harder)", fontsize=10)
    ax.set_title(
        "Above the line: the pool is the harder stratum\nBelow: #3667's cross-class images are",
        fontsize=10.5,
    )
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIGS / "pool-vs-cross-auc.png", dpi=DPI)
    plt.close(fig)


def main() -> None:
    FIGS.mkdir(exist_ok=True)
    rows = load()
    fig_delta(rows)
    fig_scatter(rows)
    print(f"wrote {len(list(FIGS.glob('*.png')))} figures into {FIGS}")


if __name__ == "__main__":
    main()
