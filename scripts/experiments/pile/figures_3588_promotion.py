#!/usr/bin/env python3
"""Figures for the #3588 promotion: what the name audit bought, and what it cost.

Three panels, each answering one question the report makes a claim about:

1. **coverage** -- of a class's COCO boxes, the share some VG spelling of ours
   lands on, before and after the alias table. This is the audit's whole case,
   and it is the panel that shows how badly a plain class name can miss.
   (`coverage.json`, from `name_coverage.py`.)
2. **supply** -- per-class band-free supply as `measure_supply.py` reported it
   before the repair and after, i.e. what the alias table is worth in positives.
   A class with no alias row must sit exactly on the diagonal, which is what
   makes this a control rather than an illustration.
   (`supply25.json` and `supply25_final.json`.)
3. **repairs** -- what the thirteen recover on the non-COCO half, where VG's
   silence is the only evidence of absence, drawn against what each class could
   already see. (`coverage.json`.)

Read from JSON rather than recomputed, so a figure can be regenerated without
redoing an analysis -- the same reason `figures_3670.py` is shaped this way.

Usage::

    python figures_3588_promotion.py <results-dir> [figdir]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
FIGDIR = Path(sys.argv[2]) if len(sys.argv) > 2 else RESULTS / "figures"

#: The thirteen #3588 added, in the order that issue ranked them.
C13 = (
    "truck",
    "car",
    "fork",
    "spoon",
    "cup",
    "bowl",
    "bottle",
    "vase",
    "bench",
    "chair",
    "sink",
    "cell phone",
    "fire hydrant",
)

INK = "#1b1b1f"
GRID = "#d9d9de"
OWN = "#8a8a93"
GAINED = "#2f6f9f"
SHIPPED = "#c1683c"


def _frame(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=9)
    ax.set_axisbelow(True)


def fig_coverage(cov: dict, path: Path) -> None:
    """The gain, not the level: `fire hydrant` at 44.7% is the same class as at 73.8%."""
    ov = cov["overlap"]
    rows = sorted(((ov[c]["own"] / ov[c]["coco_boxes"], ov[c]["alias"] / ov[c]["coco_boxes"], c) for c in C13))
    fig, ax = plt.subplots(figsize=(6.6, 4.4), dpi=130)
    ys = range(len(rows))
    ax.barh(list(ys), [100 * o for o, _, _ in rows], color=OWN, height=0.6, label="the class name alone")
    ax.barh(
        list(ys),
        [100 * (a - o) for o, a, _ in rows],
        left=[100 * o for o, _, _ in rows],
        color=GAINED,
        height=0.6,
        label="recovered by the alias table",
    )
    for i, (o, a, _c) in enumerate(rows):
        if a - o > 0.02:
            ax.text(100 * a + 0.8, i, f"+{100 * (a - o):.0f}", va="center", fontsize=8, color=GAINED, weight="bold")
    ax.set_yticks(list(ys))
    ax.set_yticklabels([f"`{c}`" for _, _, c in rows])
    ax.set_xlabel("share of COCO's boxes for the class that some VG spelling of ours lands on (%)", fontsize=9)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    _frame(ax)
    ax.set_title(
        "One spelling carries a third of `fire hydrant`, and half of `cell phone`",
        fontsize=10.5,
        color=INK,
        loc="left",
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_supply(before: dict, after: dict, path: Path) -> None:
    """The control is a class with no alias row: it must move by exactly nothing.

    Plotted as the delta rather than as before-against-after, because the claim
    is about a measurement that could not respond to its input, and a scatter on
    a log diagonal makes "moved by 1,175" and "moved by 0" look similar.
    """
    b, a = before["supply"], after["supply"]
    rows = sorted(((a[c]["union"] - b[c]["union"], c) for c in a if c in b), reverse=True)
    fig, ax = plt.subplots(figsize=(6.6, 5.6), dpi=130)
    ys = range(len(rows))
    ax.barh(
        list(ys),
        [d for d, _ in rows],
        color=[GAINED if c in C13 else SHIPPED for _, c in rows],
        height=0.62,
    )
    for i, (d, c) in enumerate(rows):
        pct = 100 * d / max(b[c]["union"], 1)
        if d > 0:
            label = f"  +{d:,}  ({pct:.0f}%)"
        elif d < 0:
            # A fold can also un-band: the merged union scatters or outgrows a
            # region and leaves every band (#3637, where `clock` nets -16).
            label = f"  \u2212{-d:,}  (the scatter guard)"
        else:
            label = "  0 -- no alias row, so nothing could move"
        ax.text(max(d, 0) + 12, i, label, va="center", fontsize=8, color=INK if d else OWN)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([f"`{c}`" for _, c in rows], fontsize=8.5)
    ax.set_xlabel("band-free positives the alias table adds, invisible to the old measurement", fontsize=9)
    ax.set_xlim(min(0, min(d for d, _ in rows) * 4), max(d for d, _ in rows) * 1.42)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.invert_yaxis()
    _frame(ax)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=GAINED),
        plt.Rectangle((0, 0), 1, 1, color=SHIPPED),
    ]
    ax.legend(handles, ["the #3588 thirteen", "the original twelve"], frameon=False, fontsize=8.5, loc="lower right")
    ax.set_title(
        "A measurement that could not move when its input changed",
        fontsize=10.5,
        color=INK,
        loc="left",
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_repairs(cov: dict, path: Path) -> None:
    """The half COCO cannot score is where a missing spelling becomes a NEGATIVE."""
    nc = cov["non_coco"]
    rows = sorted(((nc[c]["repaired"] / max(nc[c]["own"], 1), nc[c]["repaired"], nc[c]["own"], c) for c in C13))
    fig, ax = plt.subplots(figsize=(6.6, 4.4), dpi=130)
    ys = range(len(rows))
    ax.barh(list(ys), [100 * r for r, _, _, _ in rows], color=GAINED, height=0.6)
    for i, (r, rep, own, _c) in enumerate(rows):
        ax.text(100 * r + 3, i, f"+{rep:,} on {own:,}", va="center", fontsize=8, color=INK)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([f"`{c}`" for _, _, _, c in rows])
    ax.set_xlabel("images the alias table repairs, as a share of what the class could already see (%)", fontsize=9)
    ax.set_xlim(0, 275)
    ax.grid(axis="x", color=GRID, lw=0.6)
    _frame(ax)
    ax.set_title(
        "On the non-COCO half a missing spelling is not a lost positive, it is a NEGATIVE",
        fontsize=10.5,
        color=INK,
        loc="left",
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    cov = json.loads((RESULTS / "coverage.json").read_text())
    before = json.loads((RESULTS / "supply25.json").read_text())
    after = json.loads((RESULTS / "supply25_final.json").read_text())
    fig_coverage(cov, FIGDIR / "coverage.png")
    fig_supply(before, after, FIGDIR / "supply.png")
    fig_repairs(cov, FIGDIR / "repairs.png")
    print(f"wrote 3 figures to {FIGDIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
