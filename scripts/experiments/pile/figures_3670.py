#!/usr/bin/env python3
"""Figures for #3670: what the pool is made of, what it costs, and what it buys.

Four panels, each answering one question the report makes a claim about:

1. **supply** -- is an all-provable pool of 9,900 even available, and with how
   much headroom? (`negpool_supply.json`)
2. **distortion** -- the two compositions' distortions on #3667's FPR-inflation
   scale, with the intervals drawn, because the whole point is that they
   *overlap* and the decision does not rest on which bar is taller.
   (`provenance_shortcut.json`, plus #3666's measured contamination rate.)
3. **review** -- what each composition does to the negative review: how much it
   rules ineligible, and how much of what it *can* hold it keeps.
   (`negpool_coverage.json`)
4. **prevalence** -- designed against realised, before and after, with k\\* on
   the right axis. (`negpool_coverage.json`)

Run after `negpool_supply.py`, `provenance_shortcut.py` and
`negpool_coverage.py`; it reads their JSON and writes PNGs, so a figure can be
regenerated without redoing an analysis.

Usage::

    python figures_3670.py <results-dir> [figdir]
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
FIGDIR = Path(sys.argv[2]) if len(sys.argv) > 2 else RESULTS / "figures"

#: #3666's measured pool error for the shipped twelve: pooled rate and its 95%
#: interval, and #3635's per-class predicted range. Constants rather than a file
#: read because they come from a different study, and a figure that silently
#: re-derived them could drift from the report that cites them.
POOL_ERROR = 0.0140
POOL_ERROR_CI = (0.0068, 0.0286)
POOL_ERROR_PER_CLASS = (0.0028, 0.0287)
#: The threshold #3667's scale is pinned at, and the TPR a head reaches there.
#: A contaminated negative is a real positive, so it is found at the TPR rather
#: than at the false-positive rate: ratio = 1 + c * (TPR/FPR - 1).
TARGET_FPR = 0.05
TPR_AT_THRESHOLD = 0.70

INK = "#1b1b1f"
GRID = "#d9d9de"
PROVABLE = "#2f6f9f"
MIXED = "#c1683c"
MUTED = "#8a8a93"


def contamination_ratio(c: float) -> float:
    """FPR inflation a contamination rate *c* buys, on #3667's scale."""
    return 1.0 + c * (TPR_AT_THRESHOLD / TARGET_FPR - 1.0)


def _frame(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=9)
    ax.set_axisbelow(True)


def fig_supply(supply: dict, path: Path) -> None:
    """Headroom, not feasibility: 'it fits' is a weaker claim than 'it fits 3x'."""
    fig, ax = plt.subplots(figsize=(6.4, 3.2), dpi=130)
    avail = supply["provable_available"]
    silent = supply["silent_available"]
    ax.barh(["COCO-scored", "off-COCO"], [avail, silent], color=[PROVABLE, MIXED], height=0.55)
    for x, label in ((3900, "old pool\n3,900"), (9900, "#3670\n9,900")):
        ax.axvline(x, color=INK, lw=1.1, ls="--")
        ax.text(x, 1.75, label, ha="center", va="bottom", fontsize=8, color=INK)
    for i, v in enumerate((avail, silent)):
        ax.text(v - avail * 0.02, i, f"{v:,}", ha="right", va="center", color="white", fontsize=9, weight="bold")
    ax.set_xlabel("clean images available (hold none of C)", fontsize=9, color=INK)
    ax.set_ylim(-0.6, 2.1)
    ax.grid(axis="x", color=GRID, lw=0.6)
    _frame(ax)
    ax.set_title("An all-provable 9,900 draws on 34,071 images", fontsize=10.5, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_distortion(shortcut: dict, path: Path) -> None:
    """Both distortions with their intervals, because the overlap IS the finding."""
    fig, ax = plt.subplots(figsize=(6.8, 3.4), dpi=130)

    rows = []
    for emb, r in sorted(shortcut.items()):
        if "ratio_reverse_mean" not in r or r["ratio_reverse_mean"] is None:
            continue
        # The reverse arm reads the shortcut with the sign flipped: fitted
        # against the silent half, a provable negative fires LESS often, so the
        # shortcut is 1/ratio. It is the ESTIMATE, because contamination cannot
        # move it -- the residual route below is only ever as good as the
        # contamination rate it subtracts.
        rows.append((f"provenance, {emb}\n(reverse arm)", 1.0 / r["ratio_reverse_mean"], None))
        rows.append(
            (
                f"provenance, {emb}\n(forward minus predicted contamination)",
                r["ratio_mean"] / contamination_ratio(POOL_ERROR),
                None,
            )
        )
    rows.append(
        (
            "contamination\n(#3666 pooled 1.40%)",
            contamination_ratio(POOL_ERROR),
            tuple(contamination_ratio(c) for c in POOL_ERROR_CI),
        )
    )

    ys = range(len(rows))
    # Solid where the estimate stands on its own, hatched where it is only as
    # good as the contamination rate it subtracts. Two readings of one quantity
    # drawn identically would invite averaging them, and they are not equals.
    colors, hatches = [], []
    for lbl, _v, _ci in rows:
        colors.append(PROVABLE if "provenance" in lbl else MIXED)
        hatches.append("///" if "forward minus" in lbl else "")
    bars = ax.barh(list(ys), [v for _l, v, _c in rows], color=colors, height=0.6)
    for bar, hatch in zip(bars, hatches, strict=True):
        bar.set_hatch(hatch)
        bar.set_edgecolor("white")
    for y, (_lbl, v, ci) in zip(ys, rows, strict=True):
        right = v
        if ci:
            ax.plot(ci, [y, y], color=INK, lw=1.4)
            ax.plot([ci[0], ci[0]], [y - 0.12, y + 0.12], color=INK, lw=1.4)
            ax.plot([ci[1], ci[1]], [y - 0.12, y + 0.12], color=INK, lw=1.4)
            right = max(right, ci[1])
        # Past the interval, never on top of it.
        ax.text(right + 0.018, y, f"{v:.2f}x", va="center", fontsize=9, color=INK)
    ax.axvline(1.0, color=INK, lw=1.0)
    ax.text(1.0, len(rows) - 0.35, " no distortion", fontsize=8, color=MUTED, va="bottom")
    ax.text(
        1.30,
        -0.75,
        "solid = reverse arm (independent of the contamination rate)   hatched = forward minus predicted contamination",
        fontsize=7.5,
        color=MUTED,
        ha="center",
    )
    # The effect that justified the last rebuild, for scale.
    ax.axvline(1.88, color=MUTED, lw=1.0, ls=":")
    ax.text(1.88, len(rows) - 0.35, " #3667: 1.88x", fontsize=8, color=MUTED, va="bottom", ha="left")
    ax.set_yticks(list(ys))
    ax.set_yticklabels([lbl for lbl, _v, _c in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0.95, 2.0)
    ax.set_xlabel("FPR inflation vs the stratum the head was fitted against", fontsize=9, color=INK)
    ax.grid(axis="x", color=GRID, lw=0.6)
    _frame(ax)
    ax.set_title("The two distortions overlap; the bar chart is not the argument", fontsize=10.5, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_review(coverage: dict, path: Path) -> None:
    """The price. Two bars per composition, and they say different things."""
    comps = ["today", "matched", "provable"]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2), dpi=130)
    for ax, pop in zip(axes, ("reviewed negatives", "triaged negatives"), strict=True):
        kept, by_fix, by_rule = [], [], []
        for c in comps:
            row = coverage["compositions"][c]["populations"][pop]
            kept.append(row["still_in"])
            by_fix.append(row["by_fix"])
            by_rule.append(row["by_rule"])
        lost = [
            coverage["compositions"][c]["populations"][pop]["eligible"] - k for c, k in zip(comps, kept, strict=True)
        ]
        ax.bar(comps, kept, color=PROVABLE, label="still in the pool")
        ax.bar(comps, lost, bottom=kept, color="#b23b3b", label="eligible but lost")
        ax.bar(
            comps,
            by_fix,
            bottom=[k + i for k, i in zip(kept, lost, strict=True)],
            color=MUTED,
            label="removed by a correction",
        )
        ax.bar(
            comps,
            by_rule,
            bottom=[k + i + f for k, i, f in zip(kept, lost, by_fix, strict=True)],
            color="#dcdce2",
            label="ineligible by rule",
        )
        top = max(coverage["compositions"][c]["populations"][pop]["reviewed"] for c in comps)
        for i, c in enumerate(comps):
            row = coverage["compositions"][c]["populations"][pop]
            ax.text(i, row["reviewed"] + top * 0.03, f"{row['coverage']:.0%}", ha="center", fontsize=9, color=INK)
        # Headroom for the percentage labels, so they never meet the title.
        ax.set_ylim(0, top * 1.18)
        ax.set_title(pop, fontsize=9.5, color=INK, pad=22)
        ax.grid(axis="y", color=GRID, lw=0.6)
        _frame(ax)
    axes[0].set_ylabel("human judgements", fontsize=9, color=INK)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(
        "`provable` spends two thirds of the review and loses none of the rest "
        "(% = coverage of what the rule can hold)",
        fontsize=10,
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_prevalence(coverage: dict, path: Path) -> None:
    """Designed vs realised, with k* on the right -- the axis studies actually read."""
    stages = [
        ("#3156 as built\n3,900 shared", 0.0250, 0.0250),
        ("after #3667\ncross-class negatives", 0.0250, 0.01722),
        ("#3670\n9,900 provable", 0.0100, coverage["realised"]["mean"]),
    ]
    fig, ax = plt.subplots(figsize=(6.8, 3.4), dpi=130)
    xs = range(len(stages))
    ax.plot(list(xs), [d for _n, d, _r in stages], "o--", color=MUTED, lw=1.4, label="designed")
    ax.plot(list(xs), [r for _n, _d, r in stages], "o-", color=PROVABLE, lw=2.0, label="realised")
    for x, (_name, d, r) in zip(xs, stages, strict=True):
        ax.annotate(f"{d:.2%}", (x, d), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, color=MUTED)
        ax.annotate(f"{r:.2%}", (x, r), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=9, color=INK)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([n for n, _d, _r in stages], fontsize=8)
    ax.set_ylabel("prevalence of one band cell", fontsize=9, color=INK)
    ax.set_ylim(0.005, 0.030)
    ax.grid(axis="y", color=GRID, lw=0.6)
    _frame(ax)

    kax = ax.twinx()
    kax.set_ylim(*(-math.log2((1 - p) / p) for p in ax.get_ylim()))
    kax.set_ylabel("k* = -log2((1-pi)/pi)", fontsize=9, color=MUTED)
    kax.tick_params(colors=MUTED, labelsize=8)
    kax.spines[["top", "left"]].set_visible(False)
    kax.spines[["right", "bottom"]].set_color(GRID)

    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.set_title(
        "The ask was 1%; the pool delivers 0.85%, because #3667 already adds ~1,900 negatives",
        fontsize=10,
        color=INK,
        loc="left",
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    supply = json.loads((RESULTS / "negpool_supply.json").read_text())
    shortcut = json.loads((RESULTS / "provenance_shortcut.json").read_text())
    coverage = json.loads((RESULTS / "negpool_coverage.json").read_text())

    fig_supply(supply, FIGDIR / "supply.png")
    fig_distortion(shortcut, FIGDIR / "distortion.png")
    fig_review(coverage, FIGDIR / "review-coverage.png")
    fig_prevalence(coverage, FIGDIR / "prevalence.png")
    print(f"wrote 4 figures to {FIGDIR}")


if __name__ == "__main__":
    main()
