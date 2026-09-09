#!/usr/bin/env python3
"""Figures for the #3667 rebuild, from the JSON the two measurement scripts wrote.

    python figures.py            # regenerate every figure into figures/

Both inputs live beside this script under `measurements/`, so the figures can be
rebuilt without the GRID, the pile, or a GPU.
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

BEFORE = "#9aa0a6"
PRICED = "#e8a33d"
ACTUAL = "#2b7bba"


def band_of(cell: str) -> str:
    return cell.split("@", 1)[1] if "@" in cell else ""


def klass(cell: str) -> str:
    return cell.split("@", 1)[0]


def fig_evaluable(rebuilt: dict) -> None:
    """Per-cell evaluable count: what it was, what it was priced at, what it is."""
    rows = rebuilt["cells"]
    classes = sorted(
        {klass(r["cell"]) for r in rows}, key=lambda c: -sum(r["actual"] for r in rows if klass(r["cell"]) == c)
    )
    bands = ["small", "medium", "large"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharey=True)
    for ax, band in zip(axes, bands, strict=True):
        by = {klass(r["cell"]): r for r in rows if band_of(r["cell"]) == band}
        xs = range(len(classes))
        ax.bar([x - 0.27 for x in xs], [by[c]["before"] for c in classes], 0.27, color=BEFORE, label="before")
        ax.bar([x for x in xs], [by[c]["priced"] for c in classes], 0.27, color=PRICED, label="priced (#3667)")
        ax.bar([x + 0.27 for x in xs], [by[c]["actual"] for c in classes], 0.27, color=ACTUAL, label="rebuilt")
        ax.set_title(f"@{band}")
        ax.set_xticks(list(xs))
        ax.set_xticklabels(classes, rotation=60, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("images evaluable in the cell")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle(
        "Evaluable images per cell — the price was computed from designated categories, "
        "the rebuild from the label read",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(FIGS / "evaluable-per-cell.png", dpi=DPI)
    plt.close(fig)


def fig_shortfall(rebuilt: dict) -> None:
    """Priced minus actual, per cell: the co-occurrence the price could not see."""
    rows = sorted(rebuilt["cells"], key=lambda r: -r["shortfall"])
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.bar(range(len(rows)), [r["shortfall"] for r in rows], color=PRICED)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r["cell"] for r in rows], rotation=75, ha="right", fontsize=7)
    ax.set_ylabel("priced − rebuilt (images)")
    ax.grid(axis="y", alpha=0.25)
    ax.set_title(
        "Every cell got fewer negatives than the price promised, and by a class-dependent amount:\n"
        "an image can HOLD a class without being DESIGNATED a positive for it",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(FIGS / "price-shortfall.png", dpi=DPI)
    plt.close(fig)


def fig_difficulty(diff: dict) -> None:
    """Is the free text sort worse on the new pool, and worse still on the added half?"""
    rows = diff["cells"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.0))

    lo = min(min(r["auc_old"] for r in rows), min(r["auc_added_only"] for r in rows)) - 0.02
    hi = max(max(r["auc_old"] for r in rows), max(r["auc_new"] for r in rows)) + 0.02
    ax1.plot([lo, hi], [lo, hi], color="#bbb", lw=1, zorder=0)
    ax1.scatter([r["auc_old"] for r in rows], [r["auc_new"] for r in rows], s=26, color=ACTUAL, label="whole new pool")
    ax1.scatter(
        [r["auc_old"] for r in rows],
        [r["auc_added_only"] for r in rows],
        s=26,
        color="#c0392b",
        marker="^",
        label="added negatives only",
    )
    ax1.set_xlabel("AUC against the old shared pool")
    ax1.set_ylabel("AUC against the new negatives")
    ax1.set_xlim(lo, hi)
    ax1.set_ylim(lo, hi)
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(alpha=0.25)
    ax1.set_title("Free text sort, 36 cells. Below the diagonal = harder", fontsize=10)

    order = sorted(rows, key=lambda r: r["auc_added_only"] - r["auc_old"])
    ax2.barh(
        range(len(order)),
        [r["auc_added_only"] - r["auc_old"] for r in order],
        color=["#c0392b" if r["auc_added_only"] < r["auc_old"] else "#2e8b57" for r in order],
    )
    ax2.set_yticks(range(len(order)))
    ax2.set_yticklabels([r["cell"] for r in order], fontsize=6)
    ax2.axvline(0, color="#444", lw=1)
    ax2.set_xlabel("ΔAUC: added negatives − old pool")
    ax2.grid(axis="x", alpha=0.25)
    ax2.set_title("Per cell, paired on the same positives and the same query", fontsize=10)

    fig.suptitle(
        "AUC is prevalence-free: it moves only if the negatives are genuinely harder",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(FIGS / "negative-difficulty.png", dpi=DPI)
    plt.close(fig)


def fig_prevalence(rebuilt: dict) -> None:
    """What the pile looks like from one cell's point of view, before and after."""
    rows = rebuilt["cells"]
    n = len(rows)
    pos = sum(r["positives"] for r in rows) / n
    neg_b = sum(r["before"] - r["positives"] for r in rows) / n
    neg_a = sum(r["actual"] - r["positives"] for r in rows) / n
    dropped_b = rebuilt["n_medias"] - pos - neg_b
    dropped_a = rebuilt["n_medias"] - pos - neg_a

    fig, ax = plt.subplots(figsize=(8.0, 3.2))
    for i, (neg, dropped) in enumerate([(neg_b, dropped_b), (neg_a, dropped_a)]):
        ax.barh(i, pos, color="#2e8b57", label="positives" if not i else None)
        ax.barh(i, neg, left=pos, color=ACTUAL, label="negatives" if not i else None)
        ax.barh(i, dropped, left=pos + neg, color="#d9d9d9", label="not scored at all" if not i else None)
        ax.text(pos + neg + dropped / 2, i, f"{dropped:,.0f}", va="center", ha="center", fontsize=9, color="#555")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["before", "rebuilt"])
    ax.set_xlabel(f"images, averaged over {n} cells (pile = {rebuilt['n_medias']:,})")
    ax.legend(fontsize=8, ncol=3, loc="lower right")
    ax.set_title(
        f"prevalence {rebuilt['prevalence_before_pct']:.2f}% → {rebuilt['prevalence_after_pct']:.2f}%", fontsize=10
    )
    fig.tight_layout()
    fig.savefig(FIGS / "pile-composition.png", dpi=DPI)
    plt.close(fig)


def fig_shortcut(short: dict, diff: dict) -> None:
    """The headline: how much of the old contrast was scene clutter?

    Two probes, one difference. A text query has no shortcut available to it, so
    what it loses on the added negatives is how much nearer they sit to the
    class *semantically*. A trained head can learn a shortcut, so what IT loses
    is that plus the shortcut. The gap between the two bars is the shortcut.
    """
    s_by = {r["cell"]: r for r in short["cells"]}
    d_by = {r["cell"]: r for r in diff["cells"]}
    both = [c for c in s_by if c in d_by]

    fig, (ax1, ax0, ax2) = plt.subplots(1, 3, figsize=(15.5, 4.8), gridspec_kw={"width_ratios": [1, 1, 1.9]})

    text_d = [d_by[c]["auc_added_only"] - d_by[c]["auc_old"] for c in both]
    head_d = [s_by[c]["auc_added"] - s_by[c]["auc_old"] for c in both]
    parts = ax1.violinplot([text_d, head_d], showmeans=True, widths=0.7)
    for pc_, col in zip(parts["bodies"], ["#7f8fa6", "#c0392b"], strict=True):
        pc_.set_facecolor(col)
        pc_.set_alpha(0.55)
    ax1.axhline(0, color="#444", lw=1)
    ax1.set_xticks([1, 2])
    ax1.set_xticklabels(
        [
            f"text query\n(cannot learn one)\n{diff['d_auc_added_mean']:+.3f} ± {diff['d_auc_added_se']:.3f}",
            f"trained head\n(can)\n{short['d_auc_mean']:+.3f} ± {short['d_auc_se']:.3f}",
        ],
        fontsize=8,
    )
    ax1.set_ylabel("ΔAUC on the added negatives, vs the old pool")
    ax1.grid(axis="y", alpha=0.25)
    ax1.set_title(f"Paired over {len(both)} cells", fontsize=10)

    # The shortcut runs along the band -- which is the axis this dataset exists
    # to measure (#3156), so the bias was largest exactly where its own question
    # is decided.
    bands = ["small", "medium", "large"]
    mean, err = [], []
    for band in bands:
        cs = [c for c in both if c.endswith("@" + band)]
        v = [s_by[c]["ratio"] for c in cs]
        mean.append(sum(v) / len(v))
        err.append((sum((x - mean[-1]) ** 2 for x in v) / (len(v) - 1)) ** 0.5 / len(v) ** 0.5)
    ax0.errorbar(range(3), mean, yerr=err, marker="o", color="#c0392b", capsize=4, lw=2)
    ax0.axhline(1.0, color="#444", lw=1.2, ls="--")
    ax0.set_xticks(range(3))
    ax0.set_xticklabels([f"@{b}\n(n=12)" for b in bands], fontsize=8)
    ax0.set_ylim(0.8, max(m + e for m, e in zip(mean, err, strict=True)) + 0.3)
    ax0.set_ylabel("FPR ratio")
    ax0.grid(axis="y", alpha=0.25)
    ax0.set_title("...and it runs along the band", fontsize=10)

    order = sorted(both, key=lambda c: -s_by[c]["ratio"])
    ax2.bar(
        range(len(order)),
        [s_by[c]["ratio"] for c in order],
        color=["#c0392b" if s_by[c]["ratio"] > 1 else "#2e8b57" for c in order],
    )
    ax2.axhline(1.0, color="#444", lw=1.2, ls="--")
    ax2.set_xticks(range(len(order)))
    ax2.set_xticklabels(order, rotation=75, ha="right", fontsize=6)
    ax2.set_ylabel("false positives on the added negatives\n÷ on the old pool")
    ax2.grid(axis="y", alpha=0.25)
    ax2.set_title(
        f"At a threshold pinned to 5% FPR on the old pool. Mean {short['ratio_mean']:.2f} ± {short['ratio_se']:.2f}",
        fontsize=10,
    )

    fig.suptitle(
        "The old benchmark scored a bus detector against nothing that held a book. "
        "It counted about half the false positives it should have.",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(FIGS / "scene-clutter-shortcut.png", dpi=DPI)
    plt.close(fig)


def main() -> None:
    FIGS.mkdir(exist_ok=True)
    rebuilt = json.loads((MEAS / "rebuilt-siglip.json").read_text())
    fig_evaluable(rebuilt)
    fig_shortfall(rebuilt)
    fig_prevalence(rebuilt)
    diff_p = MEAS / "difficulty-siglip.json"
    short_p = MEAS / "shortcut-siglip.json"
    if diff_p.exists():
        fig_difficulty(json.loads(diff_p.read_text()))
    if diff_p.exists() and short_p.exists():
        fig_shortcut(json.loads(short_p.read_text()), json.loads(diff_p.read_text()))
    print(f"wrote figures into {FIGS}")


if __name__ == "__main__":
    main()
