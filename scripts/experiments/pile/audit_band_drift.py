"""How often does VG's non-exhaustive annotation file an image in too small a band?

A `vg_scale` image sits in `class@band` because of the box it arrived with. When
a reviewer redraws that box onto a *different, more prominent* instance of the
same class, the image moves to the band the new box implies and leaves the cell
it was sampled to fill -- 6 of the first 13 redrawn boxes did (#3616).

That is a correction, not a reviewer error. VG's recall over *C* is 0.61, so an
image holding a small annotated bowl and a large unannotated one was banded
`small` by the only box anyone had written down, and the reviewer is the first
person to have seen the other one. The open question is not whether to accept
the move but **how much of this the un-reviewed half is still hiding**, and
review is far too expensive a way to find out.

**COCO answers it for free.** 48% of VG's images are COCO images, COCO annotates
*C* exhaustively, and `anchor_to_coco` already replaces VG's labels with COCO's
on that half. So the anchored images are a control group with both readings
available: band each one from VG's boxes *alone*, band it again from COCO's, and
every disagreement is one instance of exactly the defect a reviewer catches by
hand. The rate measured there is the rate to expect on the un-anchored half,
where nothing but a human can see it.

Which disagreements count::

    UP          VG says small, COCO says medium/large -- an unannotated larger
                instance. THE defect: the reviewer's rebox, found automatically.
    scattered   COCO finds several instances too far apart to be one region, so
                the image belongs in no cell of that class at all.
    absent      COCO annotates the image and no instance is there; VG's box was
                a false positive, which the boxed-positive review already covers.
    down        COCO's box is SMALLER. Not this defect -- an extent error in VG's
                box, or a differently-drawn instance -- and reported apart from it.

Bands are audited from the bottom up (`--bands small,medium` by default) because
the error only ever pushes an image *up*: a band can hide a larger instance, and
`large` has nowhere left to go.

Reads the VG source and COCO's instances; writes nothing but its report (and
``--out``, the affected pairs, for a targeted re-slate). In particular it does
NOT touch the roster.

Usage::

    python audit_band_drift.py                       # small + medium
    python audit_band_drift.py --bands small --out drift_small.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict

import pile_config as pc

sys.path.insert(0, str(pc.Path(__file__).resolve().parent))

from pilebuild.env import log  # noqa: E402
from pilebuild.loaders.vg_scale import (  # noqa: E402
    OVERSIZE,
    SCATTERED,
    anchor_to_coco,
    band_for,
    canonicalise,
    lift_ambiguous,
    read_vg_labels,
)
from pilebuild.vgsource import vg_image_paths, vg_source  # noqa: E402

#: What a pair is when the source annotates the image and puts no instance in it.
ABSENT = "absent"
#: The verdict on one pair, in the order the report prints them. `up` and
#: `scattered` are the defect; the rest are named so the total adds up.
VERDICTS = ("up", SCATTERED, ABSENT, "down", OVERSIZE, "agrees")


def _state(by_name: dict[str, list[list[float]]], cls: str, wh: tuple[int, int]) -> str:
    """One source's reading of one ``(image, class)`` pair: a band, or why not."""
    boxes = by_name.get(cls)
    if not boxes:
        return ABSENT
    return band_for(boxes, *wh)


def _verdict(vg_band: str, coco_state: str, order: list[str]) -> str:
    """How COCO's reading of a pair differs from VG's, as one of :data:`VERDICTS`."""
    if coco_state in (ABSENT, SCATTERED, OVERSIZE):
        return coco_state
    if coco_state == vg_band:
        return "agrees"
    return "up" if order.index(coco_state) > order.index(vg_band) else "down"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--bands",
        default="small,medium",
        help="VG-side bands to audit, comma-separated (default: small,medium; `large` cannot move up)",
    )
    ap.add_argument("--out", help="write the affected pairs here, for a targeted re-slate")
    args = ap.parse_args()

    # Imported here, not at module scope: `coco_anchor` runs `setup_env()` at
    # import, which rewrites `os.environ` and the process-wide import machinery.
    # Keeping it inside `main` is what lets the banding helpers above be imported
    # and tested without doing that to the caller (the loader does the same).
    import coco_anchor as ca  # noqa: PLC0415

    order = list(pc.BOX_BANDS)
    audited = [b.strip() for b in args.bands.split(",") if b.strip()]
    unknown = [b for b in audited if b not in pc.BOX_BANDS]
    if unknown:
        raise SystemExit(f"unknown band(s): {', '.join(unknown)}; known: {', '.join(order)}")

    wanted = set(pc.SCALE_CLASSES)
    paths = vg_image_paths()
    _, records, dims = vg_source()

    image_data, instances = ca.ensure_sources(pc.PILE / "coco_anchor", fetch=False)
    truth = ca.coco_truth(instances, wanted)
    with image_data.open() as fh:
        coco_of = {int(m["image_id"]): int(m["coco_id"]) for m in json.load(fh) if m.get("coco_id")}

    # The read is deliberately wider than the class list, exactly as the loader's
    # is: a VG spelling absent from it makes an image holding only that spelling
    # look like an image holding nothing (#3605).
    labels = read_vg_labels(records, paths, dims, pc.scale_vg_wanted())
    canonicalise(labels, pc.SCALE_VG_NAMES)  # the drift control reads VG's own bands, before any anchor

    # VG's own reading, taken before `anchor_to_coco` replaces it. `lift_ambiguous`
    # is applied with an EMPTY exhaustive set on purpose: that is what makes this
    # copy the UN-ANCHORED treatment of these images, which is the population the
    # measured rate is being carried over to. Applying the real exhaustive set
    # would let COCO's presence answer the ambiguity here and nowhere else, and
    # the control would flatter itself.
    vg_labels = {iid: {n: list(bs) for n, bs in by_name.items()} for iid, by_name in labels.items()}
    withheld = lift_ambiguous(vg_labels, pc.SCALE_VG_AMBIGUOUS, set())

    box_dims, exhaustive, n_anchored, n_reframed, _reband = anchor_to_coco(
        labels, dims, coco_of, truth, ca.COCO_DIMS, wanted
    )
    log(f"  labels: {len(labels)} VG images, {n_anchored} anchored to COCO, {n_reframed} skipped as re-framed")
    if not exhaustive:
        raise SystemExit("no image anchored to COCO; there is no control group to measure against")

    # --- the control: band every anchored pair twice, once from each source
    tally: dict[str, dict[str, Counter]] = {c: {b: Counter() for b in audited} for c in pc.SCALE_CLASSES}
    n_withheld: Counter = Counter()
    affected: list[dict] = []
    for iid in sorted(exhaustive):
        for cls in pc.SCALE_CLASSES:
            if (iid, cls) in withheld:
                n_withheld[cls] += 1
                continue
            vg_band = _state(vg_labels[iid], cls, dims[iid])
            if vg_band not in audited:
                continue
            coco_state = _state(labels[iid], cls, box_dims[iid])
            verdict = _verdict(vg_band, coco_state, order)
            tally[cls][vg_band][verdict] += 1
            if verdict in ("up", SCATTERED):
                affected.append({"image_id": iid, "class": cls, "vg_band": vg_band, "coco_band": coco_state})

    _report(tally, audited, n_withheld)
    _project(audited, exhaustive, tally)

    if args.out:
        pc.Path(args.out).write_text(json.dumps(affected, indent=1) + "\n")
        print(f"\n{len(affected)} affected pairs written to {args.out}")
    return 0


def _report(tally: dict[str, dict[str, Counter]], audited: list[str], n_withheld: Counter) -> None:
    """Per class and per band: where VG's reading of an anchored pair goes wrong."""
    print("\n=== VG-alone vs COCO on the anchored half, per (class, VG band) ===")
    print("`up` and `scattered` are the #3616 defect: an instance VG never annotated.\n")
    head = "%-12s %-7s %6s | " % ("class", "band", "n") + " ".join("%9s" % v for v in VERDICTS) + "   defect"
    print(head)
    print("-" * len(head))
    totals: dict[str, Counter] = {b: Counter() for b in audited}
    for cls in pc.SCALE_CLASSES:
        for band in audited:
            counts = tally[cls][band]
            n = sum(counts.values())
            if not n:
                continue
            totals[band].update(counts)
            defect = counts["up"] + counts[SCATTERED]
            print(
                "%-12s %-7s %6d | " % (cls, band, n)
                + " ".join("%9d" % counts[v] for v in VERDICTS)
                + "   %5.1f%%" % (100.0 * defect / n)
            )
    print("-" * len(head))
    for band in audited:
        counts, n = totals[band], sum(totals[band].values())
        if not n:
            continue
        defect = counts["up"] + counts[SCATTERED]
        print(
            "%-12s %-7s %6d | " % ("ALL", band, n)
            + " ".join("%9d" % counts[v] for v in VERDICTS)
            + "   %5.1f%%" % (100.0 * defect / n)
        )
    if n_withheld:
        # Not a defect and not a clean pair: an ambiguous spelling means the
        # un-anchored copy of this image would be in no cell and in no negative
        # pool either, so it is outside the population above rather than a zero in it.
        print(
            "\nwithheld by an ambiguous VG spelling (in no band, so not audited): "
            + ", ".join(f"{c}={n}" for c, n in sorted(n_withheld.items()))
        )


def _project(audited: list[str], exhaustive: set[int], tally: dict[str, dict[str, Counter]]) -> None:
    """Carry the measured rate onto the cells no reference can check.

    The roster is what says which images are actually *in* a cell, and the
    un-anchored ones are the seats where this error is invisible: their band
    rests on VG's word alone. Multiplying the two is the number that decides
    whether a re-slate is worth a human's time.
    """
    if not pc.ROSTER.exists():
        log(f"  no roster at {pc.ROSTER}; skipping the projection onto designated cells")
        return
    roster = json.loads(pc.ROSTER.read_text()).get("cells", {})
    if not roster:
        return
    print("\n=== projected onto the designated cells (the seats no reference can check) ===")
    print("%-12s %6s %10s %10s   %s" % ("band", "seats", "anchored", "un-anch.", "expected mis-banded"))
    per_band: dict[str, Counter] = defaultdict(Counter)
    for cell, members in roster.items():
        band = cell.rsplit("@", 1)[1] if "@" in cell else ""
        if band not in audited:
            continue
        per_band[band]["seats"] += len(members)
        per_band[band]["anchored"] += sum(1 for i in members if i in exhaustive)
    for band in audited:
        counts = per_band.get(band)
        if not counts:
            continue
        control: Counter = Counter()
        for cls in pc.SCALE_CLASSES:
            control.update(tally[cls][band])
        n = sum(control.values())
        unanchored = counts["seats"] - counts["anchored"]
        rate = (control["up"] + control[SCATTERED]) / n if n else 0.0
        print(
            "%-12s %6d %10d %10d   %.1f%% of %d = %.0f images"
            % (band, counts["seats"], counts["anchored"], unanchored, 100.0 * rate, unanchored, rate * unanchored)
        )
    print("\nThe anchored seats already carry COCO's band, so only the un-anchored ones")
    print("are exposed. Reviewing them is the only way to find the rest (#3616).")


if __name__ == "__main__":
    raise SystemExit(main())
