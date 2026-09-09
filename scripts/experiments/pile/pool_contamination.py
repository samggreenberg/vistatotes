"""How often does `vg_scale`'s shared negative pool hold the class it denies?

``name_evidence.py`` asks a question *conditioned on a name*: given a VG box
named `sign` and no `stop sign`, is a stop sign there? (7.9%.) That prices one
row of :data:`pile_config.SCALE_VG_AMBIGUOUS`. It cannot say whether the pool is
in trouble, because the pool is not drawn by name -- it is drawn from the images
VG names *nothing* on, and a name-conditioned rate never looks at them.

This asks the unconditioned question instead, and it is the one the construction
rests on:

    of the images that would enter the shared negative pool on VG's evidence
    alone, what share actually hold class *c*?

Those are **false negatives**: `vg_scale` scores a detector wrong for finding the
object that is really there. ``coco_anchor.py`` measured the same defect over
VG's *labelled* images and found 1.35%; it is repaired on the 48% of VG that is
COCO-sourced, where :func:`anchor_to_coco` overwrites VG's reading wholesale. On
the other half nothing repairs it and nothing has ever measured it.

**The method is the overlap as a stand-in for the other half.** Every overlap
image is ``exhaustive`` in the real build, so the ambiguous table never fires
there and COCO is simply believed. Here the passes are run with
``exhaustive=set()`` -- i.e. *as if the image were off-COCO* -- and COCO is held
back as the answer key. That is the same trade ``anchor_to_coco`` and
``name_evidence.py`` already make: measure on the half with a reference, apply to
the half without one. It assumes the two halves have the same prevalence, which
is why ``--report`` prints the off-COCO pool population beside the rate rather
than folding them together silently.

**Two exclusion regimes, and the difference is the whole point of this script.**
:func:`band_candidates` admits an image to the pool only when it is a true
negative for *every* class in *C*::

    if not any((iid, c) in unbanded for c in classes):
        clean.append(iid)

So one ambiguous name costs **all twelve classes** the image, and a broad name
like `sign` (15,042 VG images) is unaffordable for that reason alone -- 12.7
images withheld per contaminated negative removed, charged to eleven classes
that never had the problem (#3635). But `vg_scale` already carries per-class
scorability: each media's ``evaluable_categories`` says which cells may score it,
and ``vtscore.eval.labels.evaluable_pool`` honours it. Under a **per-class** rule
the image stays in the pool and merely leaves `stop sign`'s cells. This script
prices both, because the choice between them decides whether `sign` is usable.

What the per-class rule costs is the construction's "identical negatives"
property *across* classes; it leaves the paired small-vs-large contrast *within*
a class untouched, since all three bands of a class share one exclusion set.

Usage::

    python pool_contamination.py --out contam.json
    python pool_contamination.py --propose prop.json --out contam.json

``prop.json`` is ``{"ambiguous": {class: [names]}}``, added to the shipped table.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import pile_config as pc

pc.setup_env()

import coco_folds as cf  # noqa: E402  (setup_env must run before vtscore resolves)
from pilebuild.loaders.vg_scale import canonicalise, lift_ambiguous  # noqa: E402

VG_ROOT = pc.DEMO_CACHE / "visual_genome"


def log(msg: str) -> None:
    print(f"[contam] {msg}", flush=True)


def wilson(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """``(lower, upper)`` Wilson bounds -- the same instrument ``name_evidence`` uses.

    Both ends, not just the lower one: this script's headline is sometimes a
    rate that must be shown to be SMALL, and a lower bound cannot do that.
    """
    if n <= 0:
        return 0.0, 0.0
    p = hits / n
    z2 = z * z
    centre = p + z2 / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    denom = 1 + z2 / n
    return max(0.0, (centre - half) / denom), min(1.0, (centre + half) / denom)


def read_labels(records: list, vdims: dict, wanted: set[str]) -> dict[int, dict[str, list[list[float]]]]:
    """``{iid: {vg_name: [box_norm]}}`` over *wanted*.

    Boxes are normalised here because nothing downstream of this script bands
    them -- the question is presence, not size. The read has to be wider than
    *C* for the same reason the loader's is: a spelling absent from ``wanted``
    makes its image look like an image holding nothing, i.e. like a negative.

    *wanted* is a parameter and not :func:`pile_config.scale_vg_wanted` because
    that function is built from the SHIPPED tables, and a proposed name absent
    from the read can never be suppressed -- the proposal would then score as a
    no-op for the most confusing possible reason.
    """
    labels: dict[int, dict[str, list[list[float]]]] = {}
    for rec in records:
        iid = int(rec["image_id"])
        vd = vdims.get(iid)
        if vd is None:
            continue
        by_name: dict[str, list[list[float]]] = {}
        for obj in rec.get("objects") or []:
            names = obj.get("names") or []
            if not names:
                continue
            name = str(names[0]).strip().lower()
            if name not in wanted:
                continue
            x, y = float(obj.get("x", 0)), float(obj.get("y", 0))
            w, h = float(obj.get("w", 0)), float(obj.get("h", 0))
            if w <= 0 or h <= 0:
                continue
            by_name.setdefault(name, []).append([x / vd[0], y / vd[1], (x + w) / vd[0], (y + h) / vd[1]])
        labels[iid] = by_name
    return labels


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--anchor-dir", default=str(pc.PILE / "coco_anchor"))
    ap.add_argument("--propose", default="", help='JSON {"ambiguous": {class: [names]}} to ADD to the shipped table')
    ap.add_argument(
        "--drop",
        default="",
        help="comma-separated `class:name` pairs to REMOVE from the shipped ambiguous table. Needed to "
        "score an entry that already ships: the counterfactual for `bike` is the pool WITHOUT it, and "
        "--propose can only add.",
    )
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    classes = list(pc.SCALE_CLASSES)
    anchor = Path(args.anchor_dir)

    _cboxes, cdims, cpresent = cf.coco_boxes(anchor)

    log("loading VG image_data.json")
    with (anchor / "image_data.json").open() as fh:
        meta = json.load(fh)
    coco_of = {int(m["image_id"]): int(m["coco_id"]) for m in meta if m.get("coco_id")}
    vdims = {int(m["image_id"]): (int(m["width"]), int(m["height"])) for m in meta}

    log(f"loading VG objects.json ({(VG_ROOT / 'objects.json').stat().st_size / 1e6:.0f} MB)")
    with (VG_ROOT / "objects.json").open() as fh:
        records = json.load(fh)
    log(f"  {len(records)} VG records")

    # Which images COCO can actually adjudicate. A re-framed copy (VG 500x375
    # against COCO 375x500) is not the same framing, so `anchor_to_coco` leaves
    # it on VG's own labels -- it behaves like an off-COCO image and is counted
    # as one here rather than being scored against a box that does not describe
    # its pixels.
    adjudicable: dict[int, int] = {}
    reframed = 0
    for iid, cid in coco_of.items():
        vd, cd = vdims.get(iid), cdims.get(cid)
        if vd is None or cd is None:
            continue
        if not pc.aspect_transferable(vd, cd):
            reframed += 1
            continue
        adjudicable[iid] = cid

    ambig = {c: list(ns) for c, ns in pc.SCALE_VG_AMBIGUOUS.items()}
    # A dropped entry is removed from the BASELINE, so the "proposal" is then the
    # shipped table and the two columns read as without-it -> with-it, exactly as
    # they do for an added name.
    for pair in (p for p in args.drop.split(",") if p.strip()):
        cls, _, nm = pair.partition(":")
        cls, nm = cls.strip(), nm.strip().lower()
        if cls not in classes:
            raise SystemExit(f"--drop names a class that is not in C: {cls!r}")
        if nm not in ambig.get(cls, []):
            raise SystemExit(f"--drop: {nm!r} is not in the shipped ambiguous table for {cls!r}")
        ambig[cls] = [n for n in ambig[cls] if n != nm]
    proposed = {c: list(ns) for c, ns in pc.SCALE_VG_AMBIGUOUS.items()}
    if args.propose:
        raw = json.loads(Path(args.propose).read_text())
        for c, ns in (raw.get("ambiguous") or {}).items():
            if c not in classes:
                raise SystemExit(f"proposal names a class that is not in C: {c}")
            proposed.setdefault(c, [])
            proposed[c] += [n.strip().lower() for n in ns if n.strip().lower() not in proposed[c]]

    def clean_sets(table: dict[str, list[str]]) -> tuple[set[int], dict[str, set[int]]]:
        """``(clean_global, clean_per_class)`` over every VG image, under *table*.

        The passes are the loader's own, run with ``exhaustive=set()`` so the
        ambiguous suppression fires everywhere -- which is the off-COCO regime
        this script is measuring. ``labels`` is rebuilt each call because both
        loader passes edit it in place.
        """
        labels = read_labels(records, vdims, read_wanted)
        # No dims and no mode, deliberately (#3637): pool membership asks only
        # whether a class NAME survives on the image, never what band its boxes
        # imply, and all three fold modes agree on that -- an image the class
        # already names is disqualified whether or not the alias box merges. The
        # returned counters are about bands and mean nothing here.
        canonicalise(labels, pc.SCALE_VG_NAMES)
        suppressed = lift_ambiguous(labels, {c: tuple(ns) for c, ns in table.items()}, set())
        sup_by_image: dict[int, set[str]] = defaultdict(set)
        for iid, c in suppressed:
            sup_by_image[iid].add(c)
        cg: set[int] = set()
        cpc: dict[str, set[int]] = {c: set() for c in classes}
        in_c = set(classes)
        for iid, by_name in labels.items():
            # After the loader's two passes every surviving name is a class name:
            # `scale_vg_wanted` reads nothing that is not a class, an alias or an
            # ambiguous spelling. The read here is deliberately WIDER than that --
            # it carries the proposal's names too -- so a proposed name that this
            # pass did not suppress is still sitting in `by_name`, and letting it
            # through would disqualify the image from the pool in the BASELINE
            # pass and score the proposal against a baseline it invented.
            if any(n in in_c for n in by_name):
                continue
            sup = sup_by_image.get(iid, set())
            if not sup:
                cg.add(iid)
            for c in classes:
                if c not in sup:
                    cpc[c].add(iid)
        return cg, cpc

    # One read set for BOTH passes, so the two are comparable image for image.
    read_wanted = pc.scale_vg_wanted() | {n for ns in proposed.values() for n in ns}
    log(f"reading {len(read_wanted)} VG names ({len(pc.scale_vg_wanted())} shipped + proposal)")

    log("pass 1: shipped tables")
    ship_global, ship_per_class = clean_sets(ambig)
    log("pass 2: proposal")
    prop_global, prop_per_class = clean_sets(proposed)

    adj = set(adjudicable)
    truth_has: dict[str, set[int]] = {
        c: {iid for iid in adj if c in cpresent.get(adjudicable[iid], set())} for c in classes
    }

    log(f"{len(adj)} adjudicable overlap images; {reframed} re-framed copies counted as off-COCO")

    def score(pool: set[int], c: str) -> tuple[int, int, float, float, float]:
        on_overlap = pool & adj
        hits = len(on_overlap & truth_has[c])
        n = len(on_overlap)
        lo, hi = wilson(hits, n)
        return hits, n, (hits / n if n else 0.0), lo, hi

    report: dict[str, object] = {
        "meta": {
            "adjudicable_overlap_images": len(adj),
            "reframed_counted_as_off_coco": reframed,
            "vg_images_read": len(vdims),
            "scale_n_neg": pc.SCALE_N_NEG,
            "scale_n_pos": pc.SCALE_N_POS,
            "proposal": {c: ns for c, ns in (json.loads(Path(args.propose).read_text()).get("ambiguous") or {}).items()}
            if args.propose
            else {},
            "dropped_from_baseline": [p.strip() for p in args.drop.split(",") if p.strip()],
        },
        "classes": {},
    }

    print("\n" + "=" * 100)
    print("POOL CONTAMINATION -- of the images that would enter the SHARED negative pool on VG's")
    print("evidence alone, the share that actually hold the class. Measured on the VG-COCO overlap")
    print("with the ambiguous pass forced on, i.e. as if those images were off-COCO.")
    print("`per-class` is the counterfactual rule: the image leaves this class's cells, not the pool.")
    print("=" * 100)
    print(
        "%-11s | %-22s | %-22s | %s"
        % ("class", "shipped, global rule", "shipped, per-class rule", "expected false neg /3900")
    )
    print("%-11s | %8s %13s | %8s %13s | %s" % ("", "rate", "95% CI", "rate", "95% CI", ""))
    for c in classes:
        hg, ng, rg, log_, hig = score(ship_global, c)
        hp, np_, rp, lop, hip = score(ship_per_class[c], c)
        exp = rp * pc.SCALE_N_NEG
        print(
            "%-11s | %7.2f%% [%.2f,%.2f]%s | %7.2f%% [%.2f,%.2f]%s | %6.0f  (vs %d positives/cell)"
            % (
                c,
                100 * rg,
                100 * log_,
                100 * hig,
                "",
                100 * rp,
                100 * lop,
                100 * hip,
                "",
                exp,
                pc.SCALE_N_POS,
            )
        )
        report["classes"][c] = {
            "global": {"hits": hg, "n": ng, "rate": rg, "lo": log_, "hi": hig},
            "per_class": {"hits": hp, "n": np_, "rate": rp, "lo": lop, "hi": hip},
            "pool_global": len(ship_global),
            "pool_per_class": len(ship_per_class[c]),
            "expected_false_negatives_per_pool": exp,
        }

    print("\npool population (VG images eligible as shared negatives):")
    print("  global rule, shipped tables : %d" % len(ship_global))
    for c in classes:
        report["classes"][c]["pool_per_class_shipped"] = len(ship_per_class[c])
    print(
        "  per-class rule, shipped     : min %d (%s), max %d"
        % (
            min(len(v) for v in ship_per_class.values()),
            min(ship_per_class, key=lambda c: len(ship_per_class[c])),
            max(len(v) for v in ship_per_class.values()),
        )
    )

    if args.propose or args.drop:
        print("\n" + "=" * 100)
        print("PROPOSAL -- what the changed ambiguous names buy, and what they cost, under each rule")
        print("=" * 100)
        print(
            "%-11s | %9s %9s | %9s %9s | %s"
            % ("class", "contam", "->", "pool", "->", "price (images withheld / contaminated neg removed)")
        )
        for c in classes:
            _hg, _ng, rg, _l, _h = score(ship_global, c)
            _hp2, _np2, rp2, _l2, _h2 = score(prop_global, c)
            removed = (rg - rp2) * len(ship_global & adj)
            withheld_global = len(ship_global) - len(prop_global)
            withheld_pc = len(ship_per_class[c]) - len(prop_per_class[c])
            price_g = withheld_global / removed if removed > 0 else float("inf")
            price_p = withheld_pc / removed if removed > 0 else float("inf")
            if withheld_pc == 0 and removed <= 0:
                continue
            print(
                "%-11s | %8.2f%% %8.2f%% | %9d %9d | global %.1f  per-class %.1f"
                % (
                    c,
                    100 * rg,
                    100 * rp2,
                    len(ship_per_class[c]),
                    len(prop_per_class[c]),
                    price_g,
                    price_p,
                )
            )
            report["classes"][c]["proposal"] = {
                "rate_before": rg,
                "rate_after": rp2,
                "withheld_global": withheld_global,
                "withheld_per_class": withheld_pc,
                "contaminated_removed": removed,
                "price_global": price_g,
                "price_per_class": price_p,
            }
        print("\npool population under the proposal:")
        print(
            "  global rule    : %d  (was %d, -%d)"
            % (len(prop_global), len(ship_global), len(ship_global) - len(prop_global))
        )
        print(
            "  per-class rule : min %d (%s), max %d"
            % (
                min(len(v) for v in prop_per_class.values()),
                min(prop_per_class, key=lambda c: len(prop_per_class[c])),
                max(len(v) for v in prop_per_class.values()),
            )
        )
        report["meta"]["pool_global_shipped"] = len(ship_global)
        report["meta"]["pool_global_proposed"] = len(prop_global)

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
        log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
