#!/usr/bin/env python3
"""What #3667's rebuild actually changed, measured against the cell it replaced.

`cross_class_negatives_effect.py` PRICED the change from the shipped pickle
before spending the GPU hours. This is the other end: the rebuilt cell against a
copy of the one it replaced, which is the only thing that can say whether the
price was right.

It answers three questions, in the order they can invalidate each other:

1. **Did anything but the labels move?** A relabel that also moved the vectors
   is not a relabel, and every study holding a result off the old cell would be
   comparing against a different dataset without being told. The provenance
   sidecar carries a `vectors_sha256`, so this is an equality, not an estimate.
2. **Do the invariants hold in the built data?** The unit tests pin
   `_evaluable`; they cannot say that 7,747 real images came out obeying it.
   Three properties are checked on every media, not sampled.
3. **What did it buy, and does that match the price?** The prediction was
   computed from each image's DESIGNATED categories, which is not the same set
   as the classes it HOLDS -- so the two numbers were never going to agree, and
   the gap between them is a property of the pile worth naming.

Usage::

    python cross_class_negatives_rebuilt.py                    # vg_scale, siglip
    python cross_class_negatives_rebuilt.py --dataset vg_scale_deep --embedder siglip
    python cross_class_negatives_rebuilt.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "calibration"))

import pile_config as pc  # noqa: E402

from _cells_io import load_medias  # noqa: E402  (calibration/)

#: Where the pre-rebuild copies live. Not a default anyone should rely on
#: forever -- it is the archive this study made -- but naming it here beats
#: retyping it into every invocation.
BEFORE_DIR = Path("/expscratch/sgreenberg/archive/pre-3667-vg_scale")


#: Datasets whose cells are the bare class. `vg_scale` is the only one of the
#: three keyed `class@band`; `vg_scale_any` collapses the band away (#3115) and
#: `vg_scale_deep` never designates on it (#3547). Spelling this wrong is the
#: same mistake #3672 made inside `_evaluable`, one level up.
BARE_KEYED = {"vg_scale_any", "vg_scale_deep"}


def cells_of(dataset: str) -> list[str]:
    """The cell names that dataset's loader designates, in its own keying."""
    if dataset in BARE_KEYED:
        return list(pc.SCALE_CLASSES)
    return [pc.scale_cell(c, b) for c in pc.SCALE_CLASSES for b in pc.BOX_BANDS]


def klass(cell: str) -> str:
    """The class a cell belongs to, under either keying."""
    return cell.split("@", 1)[0]


def evaluable_counts(medias: dict, cells: list[str]) -> Counter:
    out: Counter = Counter()
    for d in medias.values():
        for cell in d.get("evaluable_categories") or []:
            out[cell] += 1
    return Counter({c: out.get(c, 0) for c in cells})


def positives_of(medias: dict, cells: list[str]) -> dict[str, set[int]]:
    out: dict[str, set[int]] = defaultdict(set)
    for iid, d in medias.items():
        for cell in d.get("categories") or []:
            if cell in cells:
                out[cell].add(iid)
    return out


def priced(medias: dict, cells: list[str]) -> Counter:
    """The #3667 prediction, recomputed exactly as the pricing script did it.

    Read off `categories` -- what the image is a DESIGNATED positive for -- and
    not off the label read, which the pickle does not carry. Reproduced here so
    the prediction and the outcome are computed by one program over one file.
    """
    out: Counter = Counter()
    by_class: dict[str, set[str]] = defaultdict(set)
    for cell in cells:
        by_class[klass(cell)].add(cell)
    for d in medias.values():
        cats = list(d.get("categories") or [])
        if not cats:
            if d.get("evaluable_categories"):
                for cell in cells:
                    out[cell] += 1
            continue
        for cell in cats:
            out[cell] += 1
        if d.get("labels_exhaustive"):
            held = {klass(c) for c in cats}
            for c, owned in by_class.items():
                if c not in held:
                    for cell in owned:
                        out[cell] += 1
    return out


def check_invariants(medias: dict, cells: list[str]) -> list[str]:
    """The three properties the built data must have. Returns failures."""
    bad: list[str] = []
    cellset = set(cells)

    # 1. No name outside this dataset's own cells. The defect that shipped in
    #    #3672 wrote 36 band-suffixed strings into a bare-keyed pickle.
    stray: Counter = Counter()
    for d in medias.values():
        for cell in d.get("evaluable_categories") or []:
            if cell not in cellset:
                stray[cell] += 1
    if stray:
        bad.append(f"{sum(stray.values())} evaluable names outside the cell list, e.g. {sorted(stray)[:3]}")

    # 2. A positive is never evaluable in another band of its OWN class -- the
    #    #3156 guarantee, which is the one thing #3667 was not allowed to undo.
    n_own = 0
    for d in medias.values():
        cats = set(d.get("categories") or [])
        if not cats:
            continue
        mine = {klass(c) for c in cats}
        for cell in d.get("evaluable_categories") or []:
            if klass(cell) in mine and cell not in cats:
                n_own += 1
    if n_own:
        bad.append(f"{n_own} images evaluable in another band of a class they hold")

    # 3. The paired band contrast: within one class, the three bands must be
    #    scored against the IDENTICAL negatives. This is the property the shared
    #    pool existed to guarantee and the one #3667's trade had to preserve --
    #    cross-class comparisons may change, small-vs-large may not.
    pos = positives_of(medias, cells)
    ev: dict[str, set[int]] = defaultdict(set)
    for iid, d in medias.items():
        for cell in d.get("evaluable_categories") or []:
            if cell in cellset:
                ev[cell].add(iid)
    by_class: dict[str, list[str]] = defaultdict(list)
    for cell in cells:
        by_class[klass(cell)].append(cell)
    for c, owned in by_class.items():
        negs = {cell: ev[cell] - pos[cell] for cell in owned}
        first = negs[owned[0]]
        for cell in owned[1:]:
            if negs[cell] != first:
                d1 = len(negs[cell] ^ first)
                bad.append(f"{c}: {cell} and {owned[0]} differ by {d1} negatives -- the paired contrast is broken")
                break
    return bad


def reconstruct_before(after: dict, cells: list[str]) -> dict:
    """The pre-#3667 cell, derived from the rebuilt one by re-applying the old rule.

    Used for ``vg_scale_deep``, whose cell was overwritten without a copy being
    taken -- the backup glob was ``vg_scale__*``, which does not match
    ``vg_scale_deep__*``.

    This is a RECONSTRUCTION, not a measurement, and it is exact for every media
    the two builds share, because the rule it inverts reads nothing the rebuilt
    pickle does not carry:

        evaluable = categories or (all cells if in the shared pool else [])

    and "in the shared pool" is "no categories, but evaluable somewhere". What
    it cannot show is the medias the two builds do not share, or the 36 stray
    band-suffixed names the old deep cell also carried -- so the counts below
    are the old rule's, not necessarily the old file's. Say so wherever they
    appear.
    """
    out = {}
    for iid, d in after.items():
        cats = list(d.get("categories") or [])
        in_pool = not cats and bool(d.get("evaluable_categories"))
        out[iid] = {**d, "evaluable_categories": cats if cats else (list(cells) if in_pool else [])}
    return out


def n_pos_img_exh(medias: dict) -> int:
    """Positives whose labels rest on COCO -- the only ones the rule can touch."""
    return sum(1 for d in medias.values() if d.get("categories") and d.get("labels_exhaustive"))


def fingerprint(path: Path) -> dict:
    side = path.with_suffix("").with_suffix(".provenance.json")
    if not side.exists():
        return {}
    return json.loads(side.read_text())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="vg_scale")
    ap.add_argument("--embedder", default="siglip")
    ap.add_argument("--before-dir", type=Path, default=BEFORE_DIR)
    ap.add_argument("--after-dir", type=Path, default=None)
    ap.add_argument("--json", type=Path, default=None, help="write the measurements here")
    args = ap.parse_args()

    after_dir = args.after_dir or pc.EMBEDDINGS
    name = f"{args.dataset}__{args.embedder}.pkl"
    before_p, after_p = args.before_dir / name, after_dir / name
    if not after_p.exists():
        print(f"missing {after_p}", file=sys.stderr)
        return 2
    cells = cells_of(args.dataset)
    after = load_medias(after_p)
    reconstructed = not before_p.exists()
    if reconstructed:
        print(f"NOTE: no copy of {name} was taken; reconstructing the pre-#3667 rule from the rebuilt cell.")
        print("      Membership and vector comparisons are unavailable; label counts are exact.\n")
        before = reconstruct_before(after, cells)
    else:
        before = load_medias(before_p)
    print(f"{name}: {len(before)} medias before, {len(after)} after\n")

    failures: list[str] = []
    # Defined up here because the whole "what moved" block is skipped when the
    # before-cell is a reconstruction, and the JSON at the bottom reports them
    # either way.
    same_vec = False
    sb = sa = None
    n_cat = 0

    # --- 1. only the labels moved -------------------------------------------
    print("=== what moved ===")
    if reconstructed:
        print("  media ids       n/a (reconstructed from the rebuilt cell)")
        print("  vectors         n/a")
    elif set(before) != set(after):
        gone, new = sorted(set(before) - set(after)), sorted(set(after) - set(before))
        failures.append(f"media set changed: {len(gone)} dropped, {len(new)} added")
        print(f"  media ids       CHANGED ({len(gone)} dropped, {len(new)} added)")
        # Name them. A rebuild is from `dev`, not from the commit that built the
        # cell it replaces, so it also carries every `pile_config` ruling merged
        # in between -- and the difference between "one image left" and "the
        # designation moved" is the difference between a footnote and a redo.
        for iid in gone[:10]:
            print(f"    dropped {iid}: was {before[iid].get('categories') or 'pool'}")
        for iid in new[:10]:
            print(f"    added   {iid}: now {after[iid].get('categories') or 'pool'}")
    else:
        print(f"  media ids       identical ({len(after)})")

    shared = set(before) & set(after)
    if not reconstructed:
        n_cat = sum(1 for i in shared if (before[i].get("categories") or []) != (after[i].get("categories") or []))
        print(f"  designation     {'identical' if not n_cat else f'CHANGED on {n_cat} images'}")
        if n_cat:
            failures.append(f"{n_cat} images changed which cells they are a positive for")

        fb, fa = fingerprint(before_p), fingerprint(after_p)
        sb = fb.get("fingerprint", {}).get("vectors_sha256")
        sa = fa.get("fingerprint", {}).get("vectors_sha256")
        same_vec = bool(sb) and sb == sa
        print(f"  vectors_sha256  {'identical' if same_vec else 'DIFFER'}  {(sb or '?')[:16]} -> {(sa or '?')[:16]}")
        print(
            f"  built on        {fb.get('device', {}).get('hostname')} ({fb.get('device', {}).get('gpu_name')})"
            f" -> {fa.get('device', {}).get('hostname')} ({fa.get('device', {}).get('gpu_name')})"
        )
        # The sha is over the WHOLE cell, so it moves when the membership moves --
        # and a rebuild runs against `dev`, which carries every `pile_config` ruling
        # merged since the cell was built, not only the one being tested. The
        # question that separates "the pile changed" from "the machine changed" is
        # whether the images present in BOTH cells got the same vectors, which is
        # what #3160's ATEN_CPU_CAPABILITY pin is supposed to guarantee.
        import numpy as np  # noqa: PLC0415

        def vec(d: dict) -> "np.ndarray":
            emb = d.get("embeddings") or {}
            return np.asarray(next(iter(emb.values())), dtype=np.float64)

        worst, n_cmp = 0.0, 0
        for i in shared:
            a, b = vec(before[i]), vec(after[i])
            if a.shape == b.shape:
                worst = max(worst, float(np.abs(a - b).max()))
                n_cmp += 1
        print(f"  shared vectors  max |Δ| = {worst:.3g} over {n_cmp} images present in both")
        if worst > 1e-6:
            failures.append(f"vectors of shared images differ by up to {worst:.3g}; the rebuild is not reproducible")
        if not same_vec and worst <= 1e-6:
            print("                  (the whole-cell sha moved with the membership, not with the arithmetic)")
        elif not same_vec:
            failures.append("vectors changed: this is not a relabel")

    n_ev = sum(
        1
        for i in shared
        if sorted(before[i].get("evaluable_categories") or []) != sorted(after[i].get("evaluable_categories") or [])
    )
    print(f"  evaluable       changed on {n_ev} of {len(shared)} images ({100 * n_ev / len(shared):.1f}%)\n")

    # --- 2. invariants ------------------------------------------------------
    print("=== invariants on the built data ===")
    bad = check_invariants(after, cells)
    for line in bad:
        print(f"  FAIL {line}")
    if not bad:
        print("  no name outside the cell list")
        print("  no image evaluable in another band of a class it holds")
        print("  every class's bands share one negative set (the paired contrast)")
    failures += bad
    stray_before = check_invariants(before, cells)
    print(f"  (the cell it replaced: {len(stray_before)} of the same checks failed)\n")

    # --- 3. what it bought, against what it was priced at --------------------
    ev_b, ev_a, ev_p = (
        evaluable_counts(before, cells),
        evaluable_counts(after, cells),
        priced(before, cells),
    )
    pos = positives_of(after, cells)
    print("=== evaluable images per cell ===")
    print(f"{'cell':<20}{'before':>9}{'priced':>9}{'actual':>9}{'gain':>8}{'shortfall':>11}")
    print("-" * 66)
    rows = []
    for c in pc.SCALE_CLASSES:
        for cell in [x for x in cells if klass(x) == c]:
            gain = 100 * (ev_a[cell] - ev_b[cell]) / ev_b[cell] if ev_b[cell] else 0.0
            short = ev_p[cell] - ev_a[cell]
            rows.append(
                {
                    "cell": cell,
                    "before": ev_b[cell],
                    "priced": ev_p[cell],
                    "actual": ev_a[cell],
                    "gain_pct": gain,
                    "shortfall": short,
                    "positives": len(pos[cell]),
                    "negatives": ev_a[cell] - len(pos[cell]),
                }
            )
            if len(cells) <= 12 or cell.endswith("@medium"):
                print(f"{cell:<20}{ev_b[cell]:>9}{ev_p[cell]:>9}{ev_a[cell]:>9}{gain:>7.1f}%{short:>11}")

    mean_gain = sum(r["gain_pct"] for r in rows) / len(rows)
    mean_before = sum(r["before"] for r in rows) / len(rows)
    mean_after = sum(r["actual"] for r in rows) / len(rows)
    mean_priced = sum(r["priced"] for r in rows) / len(rows)
    if not sum(r["before"] for r in rows):
        print("\nno cell of this dataset matched a single evaluable name -- wrong keying?", file=sys.stderr)
        return 1
    prev_b = 100 * sum(r["positives"] for r in rows) / sum(r["before"] for r in rows)
    prev_a = 100 * sum(r["positives"] for r in rows) / sum(r["actual"] for r in rows)
    print(f"\nmean evaluable per cell: {mean_before:.0f} -> {mean_after:.0f} (priced {mean_priced:.0f})")
    print(f"mean gain: {mean_gain:.1f}%  (priced {100 * (mean_priced - mean_before) / mean_before:.1f}%)")
    print(f"prevalence: {prev_b:.2f}% -> {prev_a:.2f}%")

    # --- the shortfall, named ------------------------------------------------
    # The price and the rebuild disagree per IMAGE, and the disagreement has a
    # single cause: an image can HOLD a class without being DESIGNATED a
    # positive for it (its box fell outside every band, the cell was already
    # full, or the spelling was withheld as ambiguous). The price read
    # `categories` and saw "does not hold"; the rebuild read the labels and saw
    # "holds, undesignated". Print the images, not just the rate.
    print("\n=== images the price counted as negatives and the rebuild refused ===")
    by_class: dict[str, set[str]] = defaultdict(set)
    for cell in cells:
        by_class[klass(cell)].add(cell)
    withheld: Counter = Counter()
    examples = []
    for iid, d in after.items():
        cats = list(d.get("categories") or [])
        if not cats or not d.get("labels_exhaustive"):
            continue
        mine = {klass(c) for c in cats}
        got = set(d.get("evaluable_categories") or [])
        refused = sorted(c for c in by_class if c not in mine and not (by_class[c] & got))
        if refused:
            withheld[len(refused)] += 1
            if len(examples) < 8:
                examples.append((iid, cats, refused))
    for iid, cats, refused in examples:
        print(f"  {iid}: designated {cats} -- also holds {refused}, so it is not their negative")
    n_aff = sum(withheld.values())
    print(
        f"  {n_aff} of {n_pos_img_exh(after)} COCO-exhaustive positives hold at least one "
        f"undesignated class ({100 * n_aff / max(1, n_pos_img_exh(after)):.1f}%)"
    )

    n_exh = sum(1 for d in after.values() if d.get("labels_exhaustive"))
    n_pos_img = sum(1 for d in after.values() if d.get("categories"))
    print(
        f"\n{n_pos_img} positives, {len(after) - n_pos_img} pool; "
        f"{n_exh} images COCO-exhaustive ({100 * n_exh / len(after):.1f}%)"
    )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "dataset": args.dataset,
                    "embedder": args.embedder,
                    "n_medias": len(after),
                    "vectors_identical": same_vec,
                    "vectors_sha256": {"before": sb, "after": sa},
                    "designation_changed": n_cat,
                    "evaluable_changed": n_ev,
                    "invariant_failures": bad,
                    "mean_before": mean_before,
                    "mean_priced": mean_priced,
                    "mean_after": mean_after,
                    "mean_gain_pct": mean_gain,
                    "prevalence_before_pct": prev_b,
                    "prevalence_after_pct": prev_a,
                    "n_exhaustive": n_exh,
                    "cells": rows,
                },
                indent=1,
            )
            + "\n"
        )
        print(f"\nwrote {args.json}")

    if failures:
        print("\nFAILED:\n  " + "\n  ".join(failures), file=sys.stderr)
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
