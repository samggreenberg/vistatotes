#!/usr/bin/env python3
"""Figures for the shipped-pool-error study (#3666).

Imported by ``shipped_pool_error.py --figures``; every number it draws is
handed in by the caller, so the figures and the tables cannot drift. The
photograph panels are cropped from the VG source on the grid, which is why this
is a separate module -- the measurement runs anywhere the committed CSV goes,
the figures need the pixels.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

VG = Path("/exp/scale26/datasets/external/vtsearch-demos/visual_genome")

#: One colour per verdict of the admissibility column, used in both the forest
#: plot and the photograph borders so the two read as one argument.
COL = {"yes": "#ab3a2a", "no": "#2f7d55", "unverifiable": "#8a8f98"}


def _image(iid: int):
    from PIL import Image

    for sub in ("VG_100K", "VG_100K_2"):
        p = VG / sub / f"{iid}.jpg"
        if p.exists():
            return Image.open(p).convert("RGB")
    raise FileNotFoundError(iid)


def measured_vs_predicted(table: dict, out: Path) -> None:
    """Every class's uniform-stratum rate, its interval, and #3635's prediction.

    Ordered by the prediction rather than the measurement: the question is
    whether an extrapolation from the COCO half survives contact with a human on
    the other half, and sorting by the measurement would make any agreement look
    like a trend.
    """
    rows = sorted(table.items(), key=lambda kv: -kv[1]["predicted_3635"])
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    for y, (_c, v) in enumerate(rows):
        lo, hi = 100 * v["ci"][0], 100 * v["ci"][1]
        rate = 100 * v["random_hits"] / v["random_n"]
        solo = v["kind"] == "per-class"
        ax.plot([lo, hi], [y, y], color="#454d5a" if solo else "#aab0ba", lw=2.2, zorder=2)
        ax.scatter([rate], [y], s=74, color="#2a4a8c" if solo else "#8a97ad", zorder=3, edgecolor="white", lw=1.1)
        if v["random_admissible"] != v["random_hits"]:
            adm = 100 * v["random_admissible"] / v["random_n"]
            ax.scatter([adm], [y], s=74, marker="D", color="#2f7d55", zorder=4, edgecolor="white", lw=1.1)
            ax.annotate(
                "",
                xy=(adm, y),
                xytext=(rate, y),
                arrowprops=dict(arrowstyle="->", color="#2f7d55", lw=1.4, shrinkA=6, shrinkB=6),
            )
        ax.scatter([100 * v["predicted_3635"]], [y], s=64, marker="|", color="#a8742a", lw=2.4, zorder=5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{c}  ({v['kind']})" for c, v in rows], fontsize=9.4)
    ax.invert_yaxis()
    ax.set_xlabel("share of uniformly drawn shared-pool images holding the class  (%)")
    ax.set_title(
        "Shipped-class pool error, measured for the first time — and what a ruling moves",
        fontsize=11.5,
        pad=13,
    )
    ax.grid(axis="x", alpha=0.25, zorder=0)
    ax.set_xlim(-0.4, 12)
    handles = [
        plt.Line2D([], [], marker="o", ls="", color="#2a4a8c", label="as the reviewer read it (asked alone)", ms=8),
        plt.Line2D([], [], marker="o", ls="", color="#8a97ad", label="attributed out of a group pass", ms=8),
        plt.Line2D([], [], marker="D", ls="", color="#2f7d55", label="only finds the class's own names admit", ms=7),
        plt.Line2D([], [], marker="|", ls="", color="#a8742a", label="#3635's COCO extrapolation", ms=11, mew=2.4),
    ]
    ax.legend(handles=handles, fontsize=8.8, loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(out / "measured-vs-predicted.png", dpi=170)
    plt.close(fig)
    print("wrote measured-vs-predicted.png")


def what_the_finds_are(adjudication: dict, out: Path, ids: list[int], name: str, title: str) -> None:
    """The literal errors behind the rate, cropped to the object in question.

    A rate of 2% over 100 draws is two photographs. Printing them is not a
    courtesy -- it is the only way a reader can tell an annotation error from a
    definition nobody wrote down, and for this study they are almost all the
    second.
    """
    cols = 3
    rows = math.ceil(len(ids) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.1 * cols, 4.1 * rows))
    for ax, iid in zip(axes.ravel(), ids, strict=False):
        a = adjudication[iid]
        im = _image(iid)
        w, h = im.size
        x0, y0, x1, y1 = a.get("crop", (0, 0, 1, 1))
        ax.imshow(im.crop((int(w * x0), int(h * y0), int(w * x1), int(h * y1))))
        ax.set_xticks([])
        ax.set_yticks([])
        col = COL[a["admits"]]
        for s in ax.spines.values():
            s.set_color(col)
            s.set_linewidth(2.6)
        what = a["what"]
        wrapped = what if len(what) < 46 else what[: what.rfind(" ", 0, 46)] + "\n" + what[what.rfind(" ", 0, 46) + 1 :]
        ax.set_title(
            f"{a['cls'] or 'not one of the twelve'} · {iid}\n{wrapped}",
            fontsize=8.6,
            color="#15181e",
            pad=6,
        )
    for ax in axes.ravel()[len(ids) :]:
        ax.axis("off")
    handles = [
        plt.Line2D([], [], color=COL["yes"], lw=3, label="the class's own names admit it — real pool error"),
        plt.Line2D([], [], color=COL["no"], lw=3, label="neither vocabulary admits it, measured — the pool is right"),
        plt.Line2D([], [], color=COL["unverifiable"], lw=3, label="the pixels do not settle it"),
    ]
    fig.legend(handles=handles, fontsize=9, loc="lower center", frameon=False, ncol=1, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(title, fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0.075, 1, 0.965), h_pad=4.2)
    fig.savefig(out / name, dpi=150)
    plt.close(fig)
    print(f"wrote {name}")


def sample_size_vs_ruling(table: dict, out: Path) -> None:
    """What another 1,000 draws buy, against what one sentence in a guide buys.

    Both axes are in points of the same rate, which is the only way the two are
    comparable -- and the comparison is the study's recommendation.
    """
    ns = list(range(50, 3001, 10))
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    for p, col in ((0.01, "#2a4a8c"), (0.03, "#8a97ad")):
        half = [100 * 1.96 * math.sqrt(p * (1 - p) / n) for n in ns]
        ax.plot(ns, half, color=col, lw=2.1, label=f"95% CI half-width at p = {100 * p:.0f}%")
    swing = 100 * (table["clock"]["random_hits"] - table["clock"]["random_admissible"]) / table["clock"]["random_n"]
    ax.axhline(swing, color="#ab3a2a", lw=2, ls="--")
    ax.text(
        1500,
        swing + 0.08,
        f"what ONE ruling moved `clock`: {swing:.1f} pp\n(is a wristwatch, a screen widget or a\ndeparture board a clock?)",
        color="#ab3a2a",
        fontsize=9.2,
    )
    ax.axvline(100, color="#454d5a", lw=1.4, ls=":")
    ax.text(112, 3.32, "the two vertical marks:\n70 and 100 draws/class", fontsize=8.8, color="#454d5a")
    ax.axvline(70, color="#a8742a", lw=1.4, ls=":")
    ax.text(
        112,
        2.55,
        "what #3666 priced: 70 draws/class\n(840 judgements). The pass already\nspent 100/class, i.e. 2,400.",
        fontsize=8.8,
        color="#a8742a",
    )
    ax.set_xlabel("uniform draws per class")
    ax.set_ylabel("precision bought, in points of the rate")
    ax.set_title("At 1%, the definition moves the estimate further than the sample does", fontsize=11.5, pad=12)
    ax.grid(alpha=0.25)
    ax.set_ylim(0, 4.0)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "sample-size-vs-ruling.png", dpi=170)
    plt.close(fig)
    print("wrote sample-size-vs-ruling.png")


def build(table: dict, adjudication: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    measured_vs_predicted(table, out)
    solo = [i for i, a in adjudication.items() if a["cls"] and "+" not in a["cls"] and a.get("crop")]
    per_class = [i for i in solo if adjudication[i]["cls"] in ("clock", "book", "backpack", "umbrella", "stop sign")]
    grouped = [i for i, a in adjudication.items() if a.get("crop") and i not in per_class]
    what_the_finds_are(
        adjudication,
        out,
        sorted(per_class, key=lambda i: (adjudication[i]["cls"], i)),
        "finds-asked-alone.png",
        "Every find in the five classes asked as their own question",
    )
    what_the_finds_are(
        adjudication,
        out,
        sorted(grouped, key=lambda i: (adjudication[i]["cls"] or "zz", i)),
        "finds-attributed.png",
        "The nine group-pass finds COCO could not attribute, settled by eye",
    )
    sample_size_vs_ruling(table, out)
