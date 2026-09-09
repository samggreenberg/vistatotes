#!/usr/bin/env python3
"""Figures for #3679, from the JSON `measure_bands_3679.py` wrote.

    python figures.py            # regenerate every figure into figures/

The input lives beside this script under `measurements/`, so the figures rebuild
without the GRID, the pile, or a GPU. The measurement script itself lives in
`scripts/experiments/calibration/`, because it imports `analyze_scale` and
deptry scans `docs/` (#3747).
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

BASE = "#9aa0a6"
NEW = "#2b7bba"
BANDC = {"small": "#e8a33d", "medium": "#9aa0a6", "large": "#2b7bba"}

SHORT = {
    "clip/whole_image": "clip",
    "clip_l/whole_image": "clip_l",
    "siglip/whole_image": "siglip",
    "siglip2_l/whole_image": "siglip2_l",
    "siglip+dinov3_patch/max_patch": "siglip+dinov3\n(region)",
}


def load() -> dict:
    return json.loads((MEAS / "band_effect.json").read_text())


def fig_gap(d: dict) -> None:
    """The band effect, both scopes, baseline against re-measure."""
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2), sharey=True, constrained_layout=True)
    for ax, scope, title in zip(
        axes,
        ("all_classes", "twelve_only"),
        (
            "All classes in each run (25 vs 12)\nthe better-powered comparison",
            "The twelve shared classes only\ncomposition controlled, and underpowered",
        ),
    ):
        arms = sorted(d["runs"]["baseline"][scope], key=lambda a: SHORT[a])
        for y, arm in enumerate(arms):
            b = d["runs"]["baseline"][scope][arm]["small_minus_large"]
            n = d["runs"]["remeasure"][scope][arm]["small_minus_large"]
            sed = (b["se"] ** 2 + n["se"] ** 2) ** 0.5
            ax.plot(
                [b["mean"] - 1.96 * b["se"], b["mean"] + 1.96 * b["se"]],
                [y + 0.15] * 2,
                color=BASE,
                lw=2.2,
                solid_capstyle="round",
            )
            ax.plot([b["mean"]], [y + 0.15], "o", color=BASE, ms=6, zorder=3)
            ax.plot(
                [n["mean"] - 1.96 * n["se"], n["mean"] + 1.96 * n["se"]],
                [y - 0.15] * 2,
                color=NEW,
                lw=2.2,
                solid_capstyle="round",
            )
            ax.plot([n["mean"]], [y - 0.15], "o", color=NEW, ms=6, zorder=3)
            # x in AXES fraction, y in data -- keeps the marker on the plot
            # whatever the data range is (it was landing at x=0.012 before).
            if abs(n["mean"] - b["mean"]) > 2 * sed:
                ax.text(
                    0.965, y, "*", transform=ax.get_yaxis_transform(), fontsize=15, color=NEW, va="center", ha="center"
                )
        ax.set_yticks(range(len(arms)))
        ax.set_yticklabels([SHORT[a] for a in arms], fontsize=9)
        ax.set_ylim(len(arms) - 0.5, -0.5)
        ax.set_xlabel("paired  cost(small) - cost(large)   at t=150", fontsize=9.5)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", alpha=0.25)
        ax.set_axisbelow(True)
    axes[0].plot([], [], "o-", color=BASE, lw=2.2, label="baseline, 20 seeds")
    axes[0].plot([], [], "o-", color=NEW, lw=2.2, label="re-measure, 5 seeds")
    axes[0].legend(fontsize=8.5, loc="upper left", framealpha=0.95)
    fig.suptitle(
        "#3679 predicted the gap would WIDEN. It narrowed in all five arms.\n"
        "* marks a change larger than twice its standard error",
        fontsize=11,
    )
    fig.savefig(FIGS / "band-gap.png", dpi=DPI)
    plt.close(fig)


def fig_levels(d: dict) -> None:
    """Why it narrowed: cost rose most where detection was easiest."""
    arms = sorted(d["runs"]["baseline"]["all_classes"], key=lambda a: SHORT[a])
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    w = 0.26
    for i, band in enumerate(("small", "medium", "large")):
        xs, rises = [], []
        for j, arm in enumerate(arms):
            b = d["runs"]["baseline"]["all_classes"][arm]["by_band"][band]["mean"]
            n = d["runs"]["remeasure"]["all_classes"][arm]["by_band"][band]["mean"]
            xs.append(j + (i - 1) * w)
            rises.append(100.0 * (n - b) / b)
        ax.bar(xs, rises, width=w, color=BANDC[band], label=band)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([SHORT[a] for a in arms], fontsize=9)
    ax.set_ylabel("cost increase, re-measure vs baseline  (%)", fontsize=10)
    ax.set_title(
        "The gap closed from the easy end: cost rose most at `large`,\n"
        "which is where #3667's negatives had least to bite before",
        fontsize=10.5,
    )
    ax.legend(fontsize=9, title="band", title_fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIGS / "cost-levels.png", dpi=DPI)
    plt.close(fig)


def main() -> None:
    FIGS.mkdir(exist_ok=True)
    d = load()
    fig_gap(d)
    fig_levels(d)
    print(f"wrote {len(list(FIGS.glob('*.png')))} figures into {FIGS}")


if __name__ == "__main__":
    main()
