"""Figures for the `stop sign` pool study (#3635).

Reads the JSONs committed beside this script in `measurements/`, so a re-plot needs
nothing from the cluster:

    python figures.py                      # -> figures/*.png
    python figures.py --data other-run/ --out /tmp/figs

* `supply.json`        -- ``measure_supply.py --out``
* `contam-sign.json`   -- ``pool_contamination.py --propose prop-sign.json --out``
* `contam-family.json` -- the same over all 155 names of the `sign` head-noun family
* `hard-stopsign.json` / `hard-bicycle.json` -- ``withheld_difficulty.py --out``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent

#: `stop sign` is the class on trial, `backpack` is the class that makes the
#: verdict: it is worse on every pool measure and nobody has proposed dropping it.
SUBJECT = "#c2452d"
FOIL = "#1b5e9c"
PLAIN = "#b8b8b8"


def bar_colour(cls: str) -> str:
    return SUBJECT if cls == "stop sign" else (FOIL if cls == "backpack" else PLAIN)


def fig_contamination(contam: dict, out: Path) -> None:
    """Every class's pool false-negative rate, with the Wilson interval.

    Sorted worst-first, because the only question the figure has to answer is
    where `stop sign` falls in the order.
    """
    rows = sorted(contam["classes"].items(), key=lambda kv: -kv[1]["global"]["rate"])
    names = [c for c, _ in rows]
    rate = [100 * d["global"]["rate"] for _, d in rows]
    lo = [100 * (d["global"]["rate"] - d["global"]["lo"]) for _, d in rows]
    hi = [100 * (d["global"]["hi"] - d["global"]["rate"]) for _, d in rows]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    y = range(len(names))
    ax.barh(list(y), rate, xerr=[lo, hi], color=[bar_colour(c) for c in names], error_kw={"lw": 1, "ecolor": "#444"})
    ax.set_yticks(list(y))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("share of pool-eligible images that actually hold the class (%)")
    n_neg = contam["meta"]["scale_n_neg"]
    n_pos = contam["meta"]["scale_n_pos"]
    ax.set_title(f"vg_scale negative-pool contamination\n(right axis: expected false negatives per {n_neg}-image pool)")
    sec = ax.secondary_xaxis("top", functions=(lambda v: v * n_neg / 100.0, lambda v: v * 100.0 / n_neg))
    sec.set_xlabel(f"expected false negatives per cell (against {n_pos} positives)")
    ax.grid(axis="x", lw=0.4, alpha=0.5)
    fig.tight_layout()
    fig.savefig(out / "contamination-by-class.png", dpi=130)
    plt.close(fig)


def fig_supply(supply: dict, out: Path) -> None:
    """Per-band supply against the number a cell actually needs."""
    n_pos = supply["meta"]["n_pos"]
    classes = list(supply["supply"])
    bands = ["small", "medium", "large"]
    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    w = 0.26
    for k, b in enumerate(bands):
        xs = [i + (k - 1) * w for i in range(len(classes))]
        ys = [supply["supply"][c]["bands"][b] for c in classes]
        ax.bar(xs, ys, width=w, label=b, color=["#8ab4d8", "#4a7fb5", "#1b3f66"][k])
    ax.axhline(n_pos, color=SUBJECT, lw=1.4, ls="--", label=f"SCALE_N_POS = {n_pos} (what a cell needs)")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=35, ha="right")
    ax.set_ylabel("images available")
    ax.set_yscale("log")
    ax.set_title("vg_scale positive supply per (class, band), after anchor_to_coco")
    # Headroom before the legend is placed, so it never sits on the tallest bar
    # (`bus@large` at 2,170) -- a legend covering the datum is how a figure that
    # makes its point still fails to show it.
    ax.set_ylim(top=ax.get_ylim()[1] * 3.2)
    ax.legend(fontsize=8, loc="upper center", ncol=4)
    ax.grid(axis="y", lw=0.4, alpha=0.5)
    fig.tight_layout()
    fig.savefig(out / "supply-vs-design.png", dpi=130)
    plt.close(fig)


def fig_tradeoff(sign: dict, family: dict, out: Path) -> None:
    """What listing `sign` buys, and what each exclusion rule charges for it."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
    ss = sign["classes"]["stop sign"]
    ff = family["classes"]["stop sign"]

    ax = axes[0]
    labels = ["shipped\n(no sign)", "+ `sign`", "+ all 155\n`* sign` names"]
    vals = [100 * ss["global"]["rate"], 100 * ss["proposal"]["rate_after"], 100 * ff["proposal"]["rate_after"]]
    ax.bar(labels, vals, color=[PLAIN, SUBJECT, SUBJECT])
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.2f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("stop sign pool contamination (%)")
    ax.set_title("what the name buys")
    ax.grid(axis="y", lw=0.4, alpha=0.5)

    ax = axes[1]
    meta = sign["meta"]
    rules = ["global rule\n(all 12 classes pay)", "per-class rule\n(only stop sign pays)"]
    before = [meta["pool_global_shipped"], ss["pool_per_class_shipped"]]
    after = [meta["pool_global_proposed"], ss["pool_per_class_shipped"] - ss["proposal"]["withheld_per_class"]]
    x = range(2)
    ax.bar([i - 0.18 for i in x], before, width=0.36, label="shipped", color=PLAIN)
    ax.bar([i + 0.18 for i in x], after, width=0.36, label="+ `sign`", color=SUBJECT)
    ax.axhline(4200, color="#111", lw=1.4, ls=":", label="images the pool actually draws (3900 + 300)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(rules, fontsize=8)
    ax.set_ylabel("pool-eligible images")
    ax.set_title("what it costs")
    ax.set_ylim(top=max(before) * 1.42)
    ax.legend(fontsize=7.5, loc="upper center", ncol=1)
    ax.grid(axis="y", lw=0.4, alpha=0.5)

    fig.tight_layout()
    fig.savefig(out / "sign-tradeoff.png", dpi=130)
    plt.close(fig)


def fig_hard(stopsign: dict, bicycle: dict, out: Path) -> None:
    """Where the withheld images sit when the pool is ranked by the class query."""
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for d, colour, style in ((stopsign, SUBJECT, "-"), (bicycle, FOIL, "--")):
        ks = [r["k"] for r in d["prefixes"]]
        lift = [r["lift"] for r in d["prefixes"]]
        ax.plot(ks, lift, style, color=colour, marker="o", ms=4, label=f"`{'`, `'.join(d['names'])}` -> {d['class']}")
    ax.axhline(1.0, color="#111", lw=1.0, ls=":", label="indiscriminate (withheld share = base rate)")
    ax.set_xscale("log")
    ax.set_xlabel("top-k of the drawn negative pool, ranked by the class's text query")
    ax.set_ylabel("withheld share / base rate")
    ax.set_title("Are the withheld images the pool's HARD negatives?")
    ax.legend(fontsize=8)
    ax.grid(lw=0.4, alpha=0.5)
    fig.tight_layout()
    fig.savefig(out / "withheld-difficulty.png", dpi=130)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(HERE / "measurements"))
    ap.add_argument("--out", default=str(HERE / "figures"))
    args = ap.parse_args()
    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sign = json.loads((data / "contam-sign.json").read_text())
    family = json.loads((data / "contam-family.json").read_text())
    fig_contamination(sign, out)
    fig_supply(json.loads((data / "supply.json").read_text()), out)
    fig_tradeoff(sign, family, out)
    fig_hard(
        json.loads((data / "hard-stopsign.json").read_text()),
        json.loads((data / "hard-bicycle.json").read_text()),
        out,
    )
    print(f"wrote figures to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
