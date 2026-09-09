#!/usr/bin/env python3
"""Figures for #3521, generated from the sweep's own rows and ``summary.json``.

Three, because the study makes three claims.

1. **branch_cost.png** — what the fork is actually worth: the coverage step's
   seconds against ``n``, restored versus rebuilt, on a log axis. This is the
   mechanism figure. Everything else in the study is a consequence of the gap it
   shows, and it is also where the shipped comment's then-"minutes-long" claim
   got checked against a measurement, and corrected (#3595).
2. **bar_error_by_branch.png** — the fraction of the progress bar each arm
   budgets to the wrong step, from the **within-leg holdout**, which is the only
   split in which all three arms meet all the branches. Combinations an arm was
   never scored on are drawn as an explicit *no runs* tick rather than left as a
   gap, because a missing bar and a zero-height bar are the same picture and
   mean opposite things.
3. **observed_vs_predicted.png** — predicted against observed seconds per step,
   log-log with the identity line. A point far below it is a step a profile
   thinks is free.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

#: One colour per arm, stable across every figure so a reader learns them once.
_ARMS = ("old", "new", "shipped")
_ARM_COLOURS = {"old": "#c2532f", "new": "#2f6fc2", "shipped": "#8a8a8a"}
#: One colour per branch of the coverage fork.
_BRANCH_COLOURS = {"restored": "#2f8f6f", "rebuilt": "#c2532f", "deferred": "#8a8a8a"}


def _rows(exp: Path, leg: str) -> list[dict]:
    path = exp / leg / "rows.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def fig_branch_cost(exp: Path, figdir: Path) -> None:
    """The coverage step, restored vs rebuilt, against n."""
    points: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for leg in ("old", "new"):
        for row in _rows(exp, leg):
            if row["task"] == "dataset_open" and row["step"] == "coverage" and row.get("branch"):
                points[(row["media_type"], row["branch"])].append((float(row["n"]), float(row["seconds"])))
    if not points:
        return
    medias = sorted({m for m, _ in points})
    fig, axes = plt.subplots(1, len(medias), figsize=(4.6 * len(medias), 4.2), squeeze=False, sharey=True)
    for ax, media in zip(axes[0], medias):
        for branch in ("restored", "rebuilt"):
            pts = sorted(points.get((media, branch), []))
            if not pts:
                continue
            ax.plot(
                [n for n, _ in pts],
                [s for _, s in pts],
                "o-",
                color=_BRANCH_COLOURS[branch],
                label=branch,
                markersize=5,
            )
        ax.set_yscale("log")
        ax.set_xlabel("media items in the dataset (n)")
        ax.set_title(media, fontsize=10)
        ax.grid(alpha=0.3, which="both")
    axes[0][0].set_ylabel("coverage step, seconds (log)")
    axes[0][0].legend(fontsize=8, title="atlas branch")
    fig.suptitle(
        "The fork the sweep could not see: the coverage step restored from the pickle's cache\n"
        "against the same step rebuilt from scratch, same datasets, same node",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(figdir / "branch_cost.png", dpi=130)
    plt.close(fig)


def fig_bar_error(summary: dict, figdir: Path) -> None:
    """Bar error per arm per branch, from the within-leg holdout."""
    grouped: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for key, vals in summary.get("by_arm_branch_self", {}).items():
        arm, task, branch = key.split("|", 2)
        if vals["bar_error"] is not None:
            grouped[task][branch][arm] = vals["bar_error"]
    tasks = sorted(grouped)
    if not tasks:
        return
    fig, axes = plt.subplots(1, len(tasks), figsize=(3.6 * len(tasks), 4.4), squeeze=False, sharey=True)
    for ax, task in zip(axes[0], tasks):
        branches = sorted(grouped[task])
        width = 0.26
        for offset, arm in enumerate(_ARMS):
            for i, branch in enumerate(branches):
                x = i + (offset - 1) * width
                if arm in grouped[task][branch]:
                    value = grouped[task][branch][arm]
                    ax.bar(x, value, width=width, color=_ARM_COLOURS[arm])
                    # Labelled, because a well-paced bar scores 0.00 and a
                    # zero-height bar is the same picture as no bar at all.
                    ax.text(x, value + 0.03, f"{value:.2f}", ha="center", fontsize=6.5, color=_ARM_COLOURS[arm])
                else:
                    # No held-out runs for this arm on this branch — below the
                    # axis, so it can never be read as a small measurement.
                    ax.plot([x], [-0.045], marker="x", ms=5, color=_ARM_COLOURS[arm], mew=1.5)
                    ax.text(x, -0.10, "n/a", ha="center", fontsize=6, color=_ARM_COLOURS[arm])
        ax.set_xticks(range(len(branches)))
        ax.set_xticklabels([b.replace("=", "\n") for b in branches], fontsize=7)
        ax.set_title(task, fontsize=10)
        ax.set_ylim(-0.14, 1.05)
        ax.axhline(0, color="#666", lw=0.8)
        ax.grid(axis="y", alpha=0.3)
    axes[0][0].set_ylabel("fraction of the bar budgeted to the wrong step")
    # Proxy handles: an arm whose first branch has no bar would otherwise be
    # missing from the legend, which is the one place a reader checks that all
    # three arms are present.
    axes[0][0].legend(
        handles=[matplotlib.patches.Patch(color=_ARM_COLOURS[a], label=a) for a in _ARMS],
        fontsize=8,
        title="profile",
    )
    fig.suptitle(
        "Within-leg holdout: how much of the progress bar each profile puts in the wrong step.\n"
        "Lower is better; 1.0 is every second budgeted elsewhere. A bar labelled 0.00 is a well-paced bar;\n"
        "an × below the axis is a combination with no held-out runs at all.",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(figdir / "bar_error_by_branch.png", dpi=130)
    plt.close(fig)


def fig_observed_vs_predicted(summary: dict, figdir: Path) -> None:
    """Predicted against observed seconds, per step, per arm."""
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    drawn = False
    for arm in _ARMS:
        xs, ys = [], []
        for rec in summary["self_records"]:
            if rec["arm"] != arm:
                continue
            for step, err in rec["steps"].items():
                observed = rec.get("observed", {}).get(step)
                if observed:
                    xs.append(observed)
                    ys.append(max(1e-3, observed * (1 + err)))
        if xs:
            ax.scatter(xs, ys, s=16, alpha=0.55, label=arm, color=_ARM_COLOURS[arm])
            drawn = True
    if drawn:
        lo, hi = 0.03, 400.0
        ax.plot([lo, hi], [lo, hi], color="#333", lw=1, ls="--", label="exact")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("observed seconds")
        ax.set_ylabel("predicted seconds")
        ax.set_title(
            "Per-step prediction, within-leg holdout.\nDistance from the dashed line is the error; the axis is log, so a\ndecade off the line is a 10x mis-budget.",
            fontsize=9,
        )
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")
        fig.tight_layout()
        fig.savefig(figdir / "observed_vs_predicted.png", dpi=130)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp", required=True)
    args = ap.parse_args()
    exp = Path(args.exp)
    summary = json.loads((exp / "summary.json").read_text())
    figdir = exp / "figures"
    figdir.mkdir(exist_ok=True)

    fig_branch_cost(exp, figdir)
    fig_bar_error(summary, figdir)
    fig_observed_vs_predicted(summary, figdir)
    print(f"wrote figures to {figdir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
