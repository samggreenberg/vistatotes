"""Figures for the fold-scatter study (#3637).

Reads the JSON committed beside this script in `measurements/`, plus the band
ledger from the study that raised the question, so nothing is re-derived here
and nothing is copied:

    python figures.py                      # -> figures/*.png
    python figures.py --data other-run/ --out /tmp/figs

`band-fold.json` is ``scripts/experiments/pile/band_fold.py --out``.
`name-coverage.json` is #3618's, and is the only source of the non-COCO half's
band ledger -- restating those three numbers here would be a second home for
them, and a second home drifts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
COVERAGE = HERE.parent / "2026-09-04-vg-name-coverage" / "measurements" / "name-coverage.json"

#: One colour per mode, ordered the way the decision reads: merge and let the
#: guard judge, rescue the class's own band, or never re-describe at all.
MODE_COLOUR = {"fold": "#1b7837", "guarded": "#7fbf7b", "additive": "#b8b8b8"}
#: One colour per verdict COCO can return about a contested image.
VERDICT_COLOUR = {
    "same band": "#1b7837",
    "another band": "#7fbf7b",
    "scattered": "#762a83",
    "oversize": "#9970ab",
    "not there": "#b8b8b8",
}


def pct(a: int, b: int) -> float:
    return 100.0 * a / b if b else 0.0


def verdict(data: dict, out: Path) -> None:
    """The decision: which mode agrees with an exhaustive reference."""
    t = data["truth"]
    modes = ["fold", "guarded", "additive"]
    scopes = [("all", "every banded image"), ("unband", "the fold un-bands it"), ("move", "the fold moves it")]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={"width_ratios": [1.55, 1]})
    ax = axes[0]
    width = 0.26
    for i, m in enumerate(modes):
        xs = [j + (i - 1) * width for j in range(len(scopes))]
        ys = [pct(t["agreement"][m][s], t["seen"][s]) for s, _ in scopes]
        bars = ax.bar(xs, ys, width, label=m, color=MODE_COLOUR[m])
        for b, y in zip(bars, ys, strict=True):
            ax.text(b.get_x() + b.get_width() / 2, y + 1.5, f"{y:.0f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(scopes)))
    ax.set_xticklabels([f"{lab}\nn = {t['seen'][s]:,}" for s, lab in scopes], fontsize=9)
    ax.set_ylabel("agrees with COCO's band (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("(a) each mode against an exhaustive reference", fontsize=10)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    says = t["truth_says"]["unband"]
    total = sum(says.values())
    order = sorted(says, key=lambda k: -says[k])
    labels = {"scattered": "scattered", "oversize": "oversize"}
    colours = [VERDICT_COLOUR.get(labels.get(k, "another band"), "#7fbf7b") for k in order]
    ys = [pct(says[k], total) for k in order]
    bars = ax.barh(range(len(order)), ys, color=colours)
    for b, k in zip(bars, order, strict=True):
        ax.text(b.get_width() + 1.5, b.get_y() + b.get_height() / 2, f"{says[k]}", va="center", fontsize=8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("share of the un-banded images (%)")
    ax.set_xlim(0, 100)
    ax.set_title(f"(b) what COCO says about those {total}", fontsize=10)
    ax.grid(axis="x", alpha=0.25)

    fig.suptitle("A scattered fold is the right outcome: the class really is scattered", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "verdict.png", dpi=130)
    plt.close(fig)


def two_halves(data: dict, cov: dict, out: Path) -> None:
    """The same un-banding, on both halves of the dataset, at very different rates."""
    a = data["truth"]["anchor_unbands"]
    n_coco = a["banded_by_vg"]
    coco = [
        ("same band", n_coco - a["unbanded_by_coco"] - a["moved_by_coco"] - a["absent_in_coco"]),
        ("another band", a["moved_by_coco"]),
        ("scattered", a["unbanded_by_coco"]),
        ("not there", a["absent_in_coco"]),
    ]
    b = cov["bands"]
    n_off = sum(v["banded_now"] for v in b.values())
    off = [
        ("same band", sum(v["same"] for v in b.values())),
        ("another band", sum(v["moved"] for v in b.values())),
        ("scattered", sum(v["lost"] for v in b.values())),
        ("not there", 0),
    ]

    fig, ax = plt.subplots(figsize=(10.0, 3.4))
    rows = [
        (f"VG$\\cap$COCO: the ANCHOR re-reads the image\nn = {n_coco:,}", coco),
        (f"non-COCO: the FOLD adds a spelling\nn = {n_off:,}", off),
    ]
    for y, (_lab, parts) in enumerate(rows):
        left = 0.0
        total = sum(v for _, v in parts)
        # A segment narrower than its own caption is called out above the bar on
        # a leader instead, and consecutive call-outs are stepped apart -- three
        # captions over a 2.3%-wide tail otherwise print on top of each other,
        # which is how a figure ends up less legible than the table it replaced.
        step = 0
        # The last row's call-outs point DOWN; pointing them up would put a
        # caption inside the bar above it, which is the collision again.
        side = 1 if y == len(rows) - 1 else -1
        for name, v in parts:
            if not v:
                continue
            share = pct(v, total)
            ax.barh(y, share, left=left, color=VERDICT_COLOUR[name], edgecolor="white", label=name if y == 0 else None)
            txt = f"{v:,} ({share:.1f}%)"
            if share > 12:
                ax.text(left + share / 2, y, txt, ha="center", va="center", fontsize=8)
            else:
                ax.annotate(
                    txt,
                    xy=(left + share / 2, y + 0.34 * side),
                    xytext=(left + share / 2 - 6 * (step + 1), y + side * (0.52 + 0.20 * step)),
                    ha="right",
                    fontsize=7.5,
                    arrowprops={"arrowstyle": "-", "lw": 0.6, "color": "#666666"},
                )
                step += 1
            left += share
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([lab for lab, _ in rows], fontsize=9)
    ax.set_ylim(2.15, -1.15)
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of the images the class's own spelling banded (%)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.62), ncol=4, frameon=False, fontsize=9)
    ax.set_title("Un-banding a cleanly-banded image is not new, and the fold is the smaller half of it", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "two-halves.png", dpi=130)
    plt.close(fig)


def supply(data: dict, out: Path) -> None:
    """The ledger is real and it never binds: no cell comes near its 100."""
    s = data["supply"]
    cells = [(c, band) for c in s["fold"]["supply"] for band in s["fold"]["supply"][c]]
    cells.sort(key=lambda cb: s["fold"]["supply"][cb[0]][cb[1]])
    names = [f"{c}@{b}" for c, b in cells]
    vals = [s["fold"]["supply"][c][b] for c, b in cells]

    fig, ax = plt.subplots(figsize=(7.5, 8.5))
    ax.barh(range(len(names)), vals, color="#1b7837")
    for i, (c, b) in enumerate(cells):
        d = s["guarded"]["supply"][c][b] - s["fold"]["supply"][c][b]
        if d:
            ax.text(vals[i] * 1.06, i, f"+{d}", va="center", fontsize=7, color="#555555")
    n_pos = 100
    ax.axvline(n_pos, color="#b2182b", lw=1.4)
    ax.text(n_pos * 1.04, -0.9, f"SCALE_N_POS = {n_pos}", color="#b2182b", fontsize=9, va="center")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_ylim(len(names) - 0.4, -1.4)
    ax.set_xscale("log")
    ax.set_xlabel("banded positives available (log scale)")
    ax.set_title(
        f"Every cell designates {n_pos} from these pools.\n"
        f"Grey is what `guarded` would add back; the scarcest cell, "
        f"{names[0]}, has {vals[0] - n_pos} to spare.",
        fontsize=10,
    )
    ax.grid(axis="x", alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(out / "supply.png", dpi=130)
    plt.close(fig)


def inflation(data: dict, out: Path) -> None:
    """The guard's own threshold, and whether the truth agrees with it."""
    infl = data["truth"]["inflation"]
    xs = sorted(float(k) for k in infl)
    n = [infl[_k(x)]["contested"] for x in xs]
    agree = [pct(infl[_k(x)]["truth_scatters"], infl[_k(x)]["contested"]) for x in xs]

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.plot(xs, agree, "o-", color="#1b7837", label="COCO agrees they scatter (%)")
    ax.set_ylim(0, 100)
    ax.set_ylabel("COCO agrees they scatter (%)", color="#1b7837")
    ax.set_xlabel("BAND_MAX_INFLATION")
    ax2 = ax.twinx()
    ax2.plot(xs, n, "s--", color="#762a83", label="images un-banded")
    ax2.set_ylabel("images the fold un-bands", color="#762a83")
    ax2.set_ylim(0, max(n) * 1.25)
    ax.axvline(data["meta"]["band_max_inflation"], color="#b2182b", lw=1.2, ls=":")
    ax.text(data["meta"]["band_max_inflation"] + 0.06, 8, "shipped", color="#b2182b", fontsize=9)
    ax.set_title("Loosening the guard buys back images it was right about", fontsize=10)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "inflation.png", dpi=130)
    plt.close(fig)


def _k(x: float) -> str:
    return str(x)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(HERE / "measurements"))
    ap.add_argument("--coverage", default=str(COVERAGE), help="#3618's name-coverage.json, for the non-COCO ledger")
    ap.add_argument("--out", default=str(HERE / "figures"))
    args = ap.parse_args()

    data = json.loads((Path(args.data) / "band-fold.json").read_text())
    cov = json.loads(Path(args.coverage).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    verdict(data, out)
    two_halves(data, cov, out)
    supply(data, out)
    inflation(data, out)
    print(f"wrote {out}/*.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
