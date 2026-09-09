"""Figures for the VG-name coverage study (#3618).

Reads the JSONs committed beside this script in `measurements/`, so a re-plot needs nothing
from the cluster:

    python figures.py                      # -> figures/*.png
    python figures.py --data other-run/ --out /tmp/figs

`evidence.json` is ``name_evidence.py --out``; `name-coverage.json` is
``name_coverage.py --out`` scored against the tables pile_config actually ships.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent

#: One colour per verdict, and they are ordered the way the decision is made:
#: fold it, withhold it, withhold a whole scene, or leave it alone.
COLOURS = {
    "alias": "#1b7837",
    "ambiguous": "#7fbf7b",
    "context": "#9970ab",
    "neither": "#b8b8b8",
    "unmeasured": "#e0e0e0",
}


def load(data: Path) -> tuple[dict, dict]:
    return (
        json.loads((data / "evidence.json").read_text()),
        json.loads((data / "name-coverage.json").read_text()),
    )


def rows(ev: dict) -> list[dict]:
    """Every candidate name with a readable rate, flattened over classes."""
    out = []
    for cls, names in ev["names"].items():
        for name, d in names.items():
            if not d["sole"]:
                continue
            out.append(
                {
                    "cls": cls,
                    "name": name,
                    "precision": d["sole_present"] / d["sole"],
                    "box": d["boxes_on_class"] / d["boxes"] if d["boxes"] else 0.0,
                    "off": d["off_coco_sole"],
                    "sole": d["sole"],
                    "boxes": d["boxes"],
                    "verdict": d["verdict"],
                }
            )
    return out


def fig_plane(ev: dict, out: Path) -> None:
    """The decision plane: what the class is, against whether this box is it."""
    meta = ev["meta"]
    fig, ax = plt.subplots(figsize=(9, 6.2))
    pts = [r for r in rows(ev) if r["sole"] >= meta["min_sole"]]
    for verdict in ("neither", "context", "ambiguous", "alias"):
        sel = [r for r in pts if r["verdict"] == verdict]
        # Two draws, because a name below the box floor was never judged on the
        # vertical axis: it is plotted where it falls, hollow, so the reader can
        # see that its height was not what decided it.
        solid = [r for r in sel if r["boxes"] >= meta["min_boxes"]]
        hollow = [r for r in sel if r["boxes"] < meta["min_boxes"]]
        ax.scatter(
            [r["precision"] for r in solid],
            [r["box"] for r in solid],
            s=[12 + 5.5 * (r["off"] ** 0.5) for r in solid],
            color=COLOURS[verdict],
            edgecolors="white",
            linewidths=0.6,
            alpha=0.85,
            label=f"{verdict} ({len(sel)})",
            zorder=3,
        )
        ax.scatter(
            [r["precision"] for r in hollow],
            [r["box"] for r in hollow],
            s=[12 + 5.5 * (r["off"] ** 0.5) for r in hollow],
            facecolors="none",
            edgecolors=COLOURS[verdict],
            linewidths=1.1,
            alpha=0.9,
            zorder=3,
        )

    ax.axvline(meta["min_precision"], color="#333", ls="--", lw=1.1, zorder=2)
    ax.axhline(meta["min_box"], color="#333", ls=":", lw=1.1, zorder=2)
    ax.axhline(meta["context_box"], color="#333", ls=":", lw=0.8, zorder=2)
    ax.text(meta["min_precision"] + 0.008, 1.02, "act above this", fontsize=8, color="#333")
    ax.text(0.995, meta["min_box"] + 0.012, "fold above this", fontsize=8, color="#333", ha="right")

    # A handful of names carry the whole argument; label those and nothing else.
    NUDGE = {("clock", "watch"): (7, -12), ("stop sign", "sign"): (-30, 4), ("bird", "beak"): (9, -3)}
    label = {
        ("stop sign", "sign"),
        ("dog", "hot dog"),
        ("bicycle", "bike"),
        ("clock", "watch"),
        ("clock", "clock face"),
        ("book", "books"),
        ("book", "magazine"),
        ("bird", "beak"),
        ("stop sign", "stop"),
        ("kite", "parasail"),
        ("bird", "duck"),
        ("bird", "chicken"),
    }
    for r in pts:
        if (r["cls"], r["name"]) in label:
            ax.annotate(
                f"{r['name']}",
                (r["precision"], r["box"]),
                textcoords="offset points",
                xytext=NUDGE.get((r["cls"], r["name"]), (7, 5)),
                fontsize=8.5,
                color="#222",
                zorder=4,
            )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.03, 1.06)
    ax.set_xlabel("repair precision — COCO says the class is present when this name is its only evidence")
    ax.set_ylabel("box agreement — this name's boxes land on a COCO box of the class")
    ax.set_title("Two questions, two tables: what a VG name is worth to `vg_scale`", fontsize=12)
    ax.legend(loc="upper left", frameon=False, fontsize=9, title="verdict", title_fontsize=9)
    ax.grid(alpha=0.25, zorder=0)
    fig.text(
        0.5,
        0.005,
        "marker area ∝ non-COCO images the name would act on.  Hollow: fewer than "
        f"{meta['min_boxes']} boxes, so the vertical axis was not read and the name fell to the safe table.",
        ha="center",
        fontsize=7.5,
        color="#555",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(out / "evidence-plane.png", dpi=160)
    plt.close(fig)


def fig_price(ev: dict, out: Path, top: int = 20, min_lift: float = 3.0) -> None:
    """The price of acting, for the names with the most contaminated negatives.

    Ranked by what a name would *repair* -- off-COCO images it acts on, times its
    precision -- among names whose precision is at least ``min_lift`` times the
    class's base rate. Without that filter the chart is just `man`, `wall` and
    `table`: a word on a third of VG "repairs" hundreds of images at 2% precision
    and is nobody's candidate. Lift separates them cleanly -- scene words sit
    near 1, every name in either table is above 4.
    """
    cut = 1 / ev["meta"]["min_precision"]
    base = ev["base_rate"]
    pts = sorted(
        (
            r
            for r in rows(ev)
            if r["sole"] >= ev["meta"]["min_sole"]
            and r["precision"] >= min_lift * base[r["cls"]]
            and r["precision"] > 0
        ),
        key=lambda r: -(r["off"] * r["precision"]),
    )[:top]
    pts.sort(key=lambda r: 1 / r["precision"])
    cap = max(cut * 1.6, min(20.0, max(1 / r["precision"] for r in pts) * 1.02))
    fig, ax = plt.subplots(figsize=(9.2, 0.36 * len(pts) + 1.7))
    ys = list(range(len(pts)))
    ax.barh(
        ys,
        [min(1 / r["precision"], cap) for r in pts],
        color=[COLOURS[r["verdict"]] for r in pts],
        height=0.72,
    )
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r['name']}  ({r['cls']})" for r in pts], fontsize=8.5)
    ax.axvline(cut, color="#333", ls="--", lw=1.2)
    ax.text(cut + 0.12, len(pts) - 0.4, "cut: 3 withheld per repair", fontsize=8.5, color="#333")
    for y, r in zip(ys, pts, strict=True):
        price = 1 / r["precision"]
        ax.text(
            min(price, cap) + 0.12,
            y,
            f"{r['off'] * r['precision']:.0f} repaired" + ("" if price <= cap else f"  (price {price:.0f})"),
            va="center",
            fontsize=8,
            color="#444",
        )
    ax.set_xlim(0, cap * 1.28)
    ax.set_xlabel(
        "images withheld from the shared negative pool per contaminated negative removed  (1 / precision)",
        labelpad=8,
    )
    ax.set_title(
        f"The {top} candidate names with the most contaminated negatives behind them",
        fontsize=11.5,
    )
    ax.text(
        0.5,
        -0.14,
        f"names whose precision is at least {min_lift:.0f}x their class's base rate; "
        "the count beside each bar is what it would repair on the non-COCO half",
        transform=ax.transAxes,
        ha="center",
        fontsize=8,
        color="#555",
    )
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOURS[v]) for v in ("alias", "ambiguous", "context", "neither")]
    ax.legend(handles, ("alias", "ambiguous", "context", "neither"), loc="lower right", frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(out / "price-per-repair.png", dpi=160)
    plt.close(fig)


def fig_repair(cov: dict, out: Path) -> None:
    """What the shipped tables do to each class, on the half that needed it."""
    classes = sorted(cov["non_coco"], key=lambda c: -cov["non_coco"][c]["repaired"])
    own = [cov["non_coco"][c]["own"] for c in classes]
    rep = [cov["non_coco"][c]["repaired"] for c in classes]
    held = [cov["non_coco"][c]["withheld"] for c in classes]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.8), gridspec_kw={"width_ratios": [1.3, 1]})
    ys = list(range(len(classes)))
    ax.barh(ys, own, color="#cfcfcf", height=0.7, label="images the class already sees")
    ax.barh(ys, rep, left=own, color=COLOURS["alias"], height=0.7, label="repaired — were negatives")
    ax.barh(
        ys,
        held,
        left=[o + r for o, r in zip(own, rep, strict=True)],
        color=COLOURS["ambiguous"],
        height=0.7,
        label="withheld from the pool",
    )
    ax.set_yticks(ys)
    ax.set_yticklabels(classes, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(f"non-COCO VG images  (of {cov['meta']['non_coco_images']})")
    ax.set_title("What the tables move, per class", fontsize=11.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False, fontsize=8.5)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    for y, (o, r) in enumerate(zip(own, rep, strict=True)):
        if r:
            ax.text(o + r + 22, y, f"+{100.0 * r / o:.0f}%", va="center", fontsize=8, color="#1b7837")

    # The band ledger: folding merges boxes, so an image the class already saw can
    # leave every band on the scatter guard. Net is what the positives supply gains.
    gained = [cov["bands"][c]["repaired_banded"] for c in classes]
    lost = [-cov["bands"][c]["lost"] for c in classes]
    ax2.barh(ys, gained, color=COLOURS["alias"], height=0.7, label="new banded positives")
    ax2.barh(ys, lost, color="#c2683a", height=0.7, label="banded images lost to the scatter guard")
    ax2.axvline(0, color="#333", lw=0.9)
    ax2.set_yticks(ys)
    ax2.set_yticklabels([])
    ax2.invert_yaxis()
    ax2.set_xlabel("banded images")
    ax2.set_title("…and the band ledger it pays out of", fontsize=11.5)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=1, frameon=False, fontsize=8.5)
    ax2.grid(axis="x", alpha=0.25)
    ax2.set_axisbelow(True)
    for y, (g, lo) in enumerate(zip(gained, lost, strict=True)):
        net = g + lo
        ax2.text(
            max(g, 4) + 4,
            y,
            f"net {net:+d}",
            va="center",
            fontsize=8,
            color="#1b7837" if net >= 0 else "#a0431f",
        )
    fig.tight_layout()
    fig.savefig(out / "repair-by-class.png", dpi=160)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(HERE / "measurements"))
    ap.add_argument("--out", default=str(HERE / "figures"))
    args = ap.parse_args()

    ev, cov = load(Path(args.data))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig_plane(ev, out)
    fig_price(ev, out)
    fig_repair(cov, out)
    print(f"wrote 3 figures to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
