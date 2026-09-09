"""Figures for the pooled name-adjudication study (#3636).

Reads the JSONs committed beside this script in `measurements/`, so a re-plot needs
nothing from the cluster:

    python figures.py                      # -> figures/*.png
    python figures.py --data other-run/ --out /tmp/figs

`evidence-pooled.json` is ``name_evidence.py --pooled --out``; the two
`coverage-*.json` are ``name_coverage.py --out``, one against the tables #3618
shipped and one against the tables this study proposes.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent

#: Ordered the way a group's fate is decided: it is not one hypothesis at all;
#: it is one and the answer is no; it is one and the answer is a table.
VERDICT_COLOURS = {
    "alias": "#1b7837",
    "ambiguous": "#7fbf7b",
    "context": "#9970ab",
    "neither": "#b8b8b8",
    "heterogeneous": "#d6604d",
    "thin": "#e8e8e8",
}
VERDICT_ORDER = ["alias", "ambiguous", "context", "heterogeneous", "neither", "thin"]

CLASSES = [
    "clock", "bird", "boat", "umbrella", "kite", "book",
    "dog", "backpack", "knife", "bicycle", "bus", "stop sign",
]  # fmt: skip


def load(data: Path) -> tuple[dict, dict, dict]:
    return (
        json.loads((data / "evidence-pooled.json").read_text()),
        json.loads((data / "coverage-shipped.json").read_text()),
        json.loads((data / "coverage-pooled.json").read_text()),
    )


def wilson(hits: int, n: int, z: float) -> tuple[float, float]:
    """The same interval the gate uses, so the figure and the verdict agree."""
    if n <= 0:
        return (0.0, 1.0)
    p = hits / n
    z2 = z * z
    centre = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def groups(ev: dict) -> list[dict]:
    """Every declared group, flattened over classes, with what it granted."""
    out = []
    for cls, gs in ev["groups"].items():
        for key, g in gs.items():
            granted = [n for n in g["members"] if key in ev["names"][cls][n].get("inherited_from", [])]
            out.append({"cls": cls, "key": key, "granted": granted, **g})
    return out


def fig_gate(ev: dict, out: Path) -> None:
    """What happens to a group, and how little of it is the precision cut.

    The reading that matters is the size of the `heterogeneous` band: those are
    groups that never reached a cut at all, because their own members disagree.
    """
    gs = groups(ev)
    kinds = {"construction": [], "declared kind": []}
    for g in gs:
        base = g["key"].split(":", 1)[0]
        kinds[
            "construction" if base in {"colour", "size", "typing", "count", "spelling", "plural"} else "declared kind"
        ].append(g)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
    for ax, (label, rows) in zip(axes, kinds.items(), strict=True):
        counts = {v: sum(1 for g in rows if g["verdict"] == v) for v in VERDICT_ORDER}
        gave = {v: sum(len(g["granted"]) for g in rows if g["verdict"] == v) for v in VERDICT_ORDER}
        ys = [v for v in VERDICT_ORDER if counts[v]]
        ax.barh(
            range(len(ys)),
            [counts[v] for v in ys],
            color=[VERDICT_COLOURS[v] for v in ys],
            edgecolor="white",
        )
        ax.set_yticks(range(len(ys)))
        ax.set_yticklabels(ys)
        ax.invert_yaxis()
        for i, v in enumerate(ys):
            note = f"  {counts[v]}" + (f"  ->  {gave[v]} names" if gave[v] else "")
            ax.text(counts[v], i, note, va="center", fontsize=9, color="#333")
        ax.set_xlim(0, max(counts.values()) * 1.45)
        ax.set_xlabel("groups")
        ax.set_title(f"{label}  ({len(rows)} groups)", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "What a pooled group is worth, by how it was declared\n"
        "`heterogeneous` = the group's own measured members disagree, so it was never one hypothesis",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out / "group-verdicts.png", dpi=160)
    plt.close(fig)


def fig_spread(ev: dict, out: Path, alpha: float = 0.05) -> None:
    """Why the hand-declared kinds fail: their members do not agree.

    One row per measured member, its Wilson interval at the gate's own
    Bonferroni-adjusted width, against the group's member-weighted rate. A
    member whose bar misses the dashed line is a dissenter, and one dissenter
    is enough.
    """
    picks = [
        ("bird", "species"),
        ("boat", "vessel"),
        ("umbrella", "colour"),
        ("bus", "subtype"),
    ]
    gs = {(g["cls"], g["key"]): g for g in groups(ev)}
    fig, axes = plt.subplots(1, len(picks), figsize=(14, 4.6))
    for ax, key in zip(axes, picks, strict=True):
        g = gs[key]
        z = NormalDist().inv_cdf(1.0 - alpha / (2.0 * max(len(g["measured"]), 1)))
        rows = sorted(
            g["measured"],
            key=lambda n: ev["names"][key[0]][n]["sole_present"] / ev["names"][key[0]][n]["sole"],
        )
        for i, n in enumerate(rows):
            d = ev["names"][key[0]][n]
            lo, hi = wilson(d["sole_present"], d["sole"], z)
            p = d["sole_present"] / d["sole"]
            dissents = n in g["dissent"]
            ax.plot([lo, hi], [i, i], color="#d6604d" if dissents else "#666", lw=2, solid_capstyle="butt")
            ax.plot([p], [i], "o", ms=4, color="#d6604d" if dissents else "#222")
        ax.axvline(g["member_rate"], ls="--", color="#1b7837", lw=1.4)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(
            [f"{n} ({ev['names'][key[0]][n]['sole']})" for n in rows],
            fontsize=7.5,
        )
        ax.set_xlim(-0.02, 1.02)
        ax.set_xlabel("repair precision")
        ax.set_title(
            f"{key[0]} / {key[1]}\n{g['verdict'].upper()}"
            + (f"  ({len(g['dissent'])} dissent)" if g["dissent"] else ""),
            fontsize=10,
        )
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "A group is pooled only if its own measured members agree\n"
        "bar = Wilson interval at the gate's Bonferroni width; dashed = the group's member-weighted rate; "
        "the count after each name is its adjudicable images",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out / "member-spread.png", dpi=160)
    plt.close(fig)


def fig_ledger(shipped: dict, pooled: dict, out: Path) -> None:
    """What the pooled tables add, per class, on the non-COCO half.

    `repaired` is images that stop being negatives for their own class;
    `banded` is the subset that survives the scatter guard and becomes a
    positive. The second is the one a rebuild actually gains.
    """
    rep = [pooled["non_coco"][c]["repaired"] - shipped["non_coco"][c]["repaired"] for c in CLASSES]
    band = [pooled["bands"][c]["repaired_banded"] - shipped["bands"][c]["repaired_banded"] for c in CLASSES]
    lost = [pooled["bands"][c]["lost"] - shipped["bands"][c]["lost"] for c in CLASSES]
    keep = [i for i, c in enumerate(CLASSES) if rep[i] or band[i] or lost[i]]

    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    x = range(len(keep))
    w = 0.27
    ax.bar([i - w for i in x], [rep[i] for i in keep], w, label="repaired", color="#7fbf7b")
    ax.bar(list(x), [band[i] for i in keep], w, label="of which banded", color="#1b7837")
    ax.bar([i + w for i in x], [-lost[i] for i in keep], w, label="un-banded by the scatter guard", color="#d6604d")
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels([CLASSES[i] for i in keep])
    ax.set_ylabel("non-COCO images, pooled minus shipped")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(
        f"What pooling adds to #3618's tables: +{sum(rep)} repaired, +{sum(band)} banded, -{sum(lost)} un-banded\n"
        "the seven classes not drawn gained nothing; a rebuild gains the net of the last two bars",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out / "pooled-ledger.png", dpi=160)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(HERE / "measurements"))
    ap.add_argument("--out", default=str(HERE / "figures"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ev, shipped, pooled = load(Path(args.data))
    fig_gate(ev, out)
    fig_spread(ev, out)
    fig_ledger(shipped, pooled, out)
    print(f"wrote {out}/group-verdicts.png, member-spread.png, pooled-ledger.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
