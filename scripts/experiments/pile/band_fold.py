"""Is a scattered fold the right outcome? (#3637)

``canonicalise`` merges an alias spelling's boxes into the class's, and
``band_for`` then reads the band off the union of all of them. VG's scatter
guard rejects a union more than :data:`pile_config.BAND_MAX_INFLATION` times the
largest single box, so folding a second spelling can take an image the class
**already banded** out of every band -- 248 of 11,857 on the non-COCO half, and
`clock` nets -16 (#3618).

Two readings disagree about that, and this script is what decides between them.
They are :data:`pilebuild.loaders.vg_scale.FOLD_MODES`: ``fold`` (merge and let
the guard judge the union), ``guarded`` (merge, but keep the class's own band
when the merge would leave every band) and ``additive`` (merge only where the
class had no box at all).

Two phases, and they answer two different questions:

**TRUTH** -- *which mode is right?* On the VG-COCO overlap COCO annotates C
exhaustively, so ``band_for`` over COCO's own boxes is the band the builder
would choose knowing everything. Each mode's band is scored against it, on every
image the class bands and again on just the contested ones. The overlap is also
where the question is already settled in practice and nobody noticed:
``anchor_to_coco`` replaces VG's labels with COCO's, so **the shipped build
already un-bands cleanly-banded images there** whenever the exhaustive box set
scatters. That rate is the base rate the 248 have to be read against, and it is
reported beside the arms.

The one assumption, stated: precision measured on the overlap is applied to the
non-COCO half. It is the same assumption ``anchor_to_coco`` and #3618 make.

**SUPPLY** -- *what does it cost?* Runs the real build passes (no pixels, no
embedding) under each mode and reports per-cell positive supply against
``SCALE_N_POS``, plus what each mode does to images a human has already reviewed
and to the shipped roster's designations. A band ledger is not a cost until it
moves a cell off its designated 100.

Usage::

    python band_fold.py --out band-fold.json --examples-out unbanded.json
    python band_fold.py --phase truth
    python band_fold.py --phase supply --inflation 1.5,2.0,3.0
    python band_fold.py --examples unbanded.json --sheet clock --sheet-out clock.jpg

The last form re-renders a sheet from a finished run and measures nothing, which
is what makes a figure cheap to redraw.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pile_config as pc

pc.setup_env()

import coco_anchor as ca  # noqa: E402  (setup_env must run before vtscore resolves)
import coco_folds as cf  # noqa: E402
from pilebuild.corrections import load_corrections  # noqa: E402
from pilebuild.loaders import vg_scale as vs  # noqa: E402
from pilebuild.vgsource import vg_image_paths, vg_source  # noqa: E402

VG_ROOT = pc.DEMO_CACHE / "visual_genome"

#: `dev`'s pass order -- fold BEFORE the anchor, with no dims -- carried as a
#: fourth supply arm so "the reorder changes nothing" is a measurement.
LEGACY = "fold@legacy-order"

#: What ``band_for`` is asked of, per mode, for an image the class already sees.
#: Kept beside the modes it mirrors rather than inside the loop, because the
#: whole question is which of these three descriptions of one image is true.
ABSENT = "absent"


def log(msg: str) -> None:
    print(f"[fold] {msg}", flush=True)


def band(boxes: list[list[float]]) -> str:
    """The band of a set of NORMALISED boxes.

    Every box here is already a share of its own image, so the unit square is
    the right frame and a VG box and a COCO box of the same object compare
    directly even though VG ships a downscaled copy.
    """
    return vs.band_for(boxes, 1, 1)


def mode_bands(own: list[list[float]], alias: list[list[float]]) -> dict[str, str]:
    """The band each mode assigns, for one class on one image.

    A single expression of :data:`pilebuild.loaders.vg_scale.FOLD_MODES` in
    terms of ``band_for``, so the scoring cannot drift from the builder: the
    build's own decision is re-derived here from the same two box sets it has.
    """
    merged = band(own + alias)
    kept = band(own) if own else merged
    return {
        "fold": merged,
        "guarded": merged if merged in pc.BOX_BANDS else kept,
        "additive": kept,
    }


def norm(rec: dict, wanted: set[str], W: int, H: int) -> dict[str, list[list[float]]]:
    """``{vg name: [normalised box]}`` for one VG record."""
    out: dict[str, list[list[float]]] = defaultdict(list)
    for obj in rec.get("objects") or []:
        names = [str(n).strip().lower() for n in (obj.get("names") or []) if str(n).strip()]
        if not names or names[0] not in wanted:
            continue
        x, y = float(obj.get("x", 0)), float(obj.get("y", 0))
        w, h = float(obj.get("w", 0)), float(obj.get("h", 0))
        if w > 0 and h > 0:
            out[names[0]].append([x / W, y / H, (x + w) / W, (y + h) / H])
    return dict(out)


# ---------------------------------------------------------------- phase: truth


def phase_truth(anchor: Path, records: list, inflations: list[float], examples: list | None = None) -> dict:
    """Score every mode against COCO on the overlap.

    Restricted to images where **COCO says the class is there**: where it is
    not, the VG box is a false positive and which band it would have taken is
    not a question about banding. That precision is #3618's subject, not this
    one, and mixing the two would let a mode win by being wrong less often about
    a different thing.
    """
    classes = list(pc.SCALE_CLASSES)
    alias = {c: tuple(ns) for c, ns in pc.SCALE_VG_NAMES.items()}
    wanted = set(classes) | {n for ns in alias.values() for n in ns}

    cboxes, cdims, _ = cf.coco_boxes(anchor)
    log("loading VG image_data.json")
    with (anchor / "image_data.json").open() as fh:
        meta = json.load(fh)
    coco_of = {int(m["image_id"]): int(m["coco_id"]) for m in meta if m.get("coco_id")}
    vdims = {int(m["image_id"]): (int(m["width"]), int(m["height"])) for m in meta}

    modes = list(vs.FOLD_MODES)
    # Three scopes, because the fold does two different things and they are two
    # different arguments. `unband` is the 248-shaped population -- the fold
    # leaves every band while the class's own boxes did not. `move` is #3616's:
    # both band, differently. `all` is the base the two must be read against.
    scopes = ("all", "unband", "move")
    hits = {m: dict.fromkeys(scopes, 0) for m in modes}
    seen = dict.fromkeys(scopes, 0)
    # What the truth says on the images the modes disagree about, which is the
    # finding: they only differ there, so this distribution IS the verdict.
    truth_says: dict[str, dict[str, int]] = {s: defaultdict(int) for s in scopes[1:]}
    unband_by_class: dict[str, dict[str, int]] = {c: defaultdict(int) for c in classes}
    # The base rate: images VG's own boxes band cleanly that COCO's exhaustive
    # box set does NOT band. This is `anchor_to_coco` un-banding a cleanly-banded
    # image on exhaustive evidence, which the build has always done in silence.
    anchor_unbands = {"banded_by_vg": 0, "unbanded_by_coco": 0, "moved_by_coco": 0, "absent_in_coco": 0}
    # Sensitivity of the contested population to the guard's own threshold.
    infl_contested = dict.fromkeys(inflations, 0)
    infl_truth_scatters = dict.fromkeys(inflations, 0)

    skipped_aspect = 0
    n_overlap = 0
    for rec in records:
        iid = int(rec["image_id"])
        vd = vdims.get(iid)
        cid = coco_of.get(iid)
        if vd is None or cid is None or cid not in cdims:
            continue
        cd = cdims[cid]
        if not pc.aspect_transferable(vd, cd):
            skipped_aspect += 1
            continue
        n_overlap += 1
        by_name = norm(rec, wanted, *vd)
        on_image = cboxes.get(cid, {})

        for c in classes:
            own = by_name.get(c, [])
            if not own:
                continue
            folded = [b for n in alias.get(c, ()) for b in by_name.get(n, [])]
            b_own = band(own)
            if b_own not in pc.BOX_BANDS:
                continue  # the class does not band this image today; nothing to lose

            truth_boxes = on_image.get(c, [])
            b_truth = band(truth_boxes) if truth_boxes else ABSENT

            # The base rate, independent of any alias: what the exhaustive
            # reference does to an image VG's own spelling banded.
            anchor_unbands["banded_by_vg"] += 1
            if b_truth == ABSENT:
                anchor_unbands["absent_in_coco"] += 1
            elif b_truth not in pc.BOX_BANDS:
                anchor_unbands["unbanded_by_coco"] += 1
            elif b_truth != b_own:
                anchor_unbands["moved_by_coco"] += 1

            if b_truth == ABSENT:
                continue  # a false positive, not a banding question

            bands = mode_bands(own, folded)
            seen["all"] += 1
            for m in modes:
                hits[m]["all"] += bands[m] == b_truth
            here = []
            if bands["fold"] not in pc.BOX_BANDS and bands["additive"] in pc.BOX_BANDS:
                here.append("unband")
                unband_by_class[c][b_truth] += 1
                if examples is not None:
                    # Every one of them, not a sample: 225 rows is small enough
                    # to keep whole, and a sampled population cannot be counted
                    # from afterwards.
                    examples.append(
                        {
                            "image_id": iid,
                            "class": c,
                            "alias_names": sorted(n for n in alias.get(c, ()) if n in by_name),
                            "own": own,
                            "folded": folded,
                            "coco": truth_boxes,
                            "band_own": b_own,
                            "band_fold": bands["fold"],
                            "band_coco": b_truth,
                        }
                    )
            elif bands["fold"] in pc.BOX_BANDS and bands["fold"] != bands["additive"]:
                here.append("move")
            for sc in here:
                seen[sc] += 1
                truth_says[sc][b_truth] += 1
                for m in modes:
                    hits[m][sc] += bands[m] == b_truth

            if folded:
                for t in inflations:
                    if _scatters(own + folded, t) and not _scatters(own, t):
                        infl_contested[t] += 1
                        if _scatters(truth_boxes, t):
                            infl_truth_scatters[t] += 1

    log(f"overlap: {n_overlap} images scored, {skipped_aspect} skipped on aspect drift")
    return {
        "overlap_images": n_overlap,
        "skipped_aspect_drift": skipped_aspect,
        "seen": dict(seen),
        "agreement": {m: dict(hits[m]) for m in modes},
        "truth_says": {s: dict(v) for s, v in truth_says.items()},
        "unband_by_class": {c: dict(v) for c, v in unband_by_class.items() if v},
        "anchor_unbands": anchor_unbands,
        "inflation": {
            str(t): {"contested": infl_contested[t], "truth_scatters": infl_truth_scatters[t]} for t in inflations
        },
    }


def _scatters(boxes: list[list[float]], threshold: float) -> bool:
    """Would the guard reject these boxes at *threshold*?

    Spelled out rather than reached through ``band_for`` because the threshold
    is the variable: ``band_for`` reads it from the config, and re-importing the
    config per value would make the sweep a test of module reloading.
    """
    if not boxes:
        return False
    ux0 = min(b[0] for b in boxes)
    uy0 = min(b[1] for b in boxes)
    ux1 = max(b[2] for b in boxes)
    uy1 = max(b[3] for b in boxes)
    union = max(0.0, ux1 - ux0) * max(0.0, uy1 - uy0)
    largest = max((b[2] - b[0]) * (b[3] - b[1]) for b in boxes)
    return union > largest * threshold


# --------------------------------------------------------------- phase: supply


def phase_supply(anchor: Path) -> dict:
    """Run the real build passes under each mode and diff what they designate.

    Everything up to ``designate_cells`` and no further: the pixels are not read
    and nothing is written, so this is the build's own answer to "which images
    would this cell hold", not a model of it.
    """
    classes = list(pc.SCALE_CLASSES)
    wanted = set(classes)
    wanted_vg = pc.scale_vg_wanted()

    paths = vg_image_paths()
    _, records, dims = vg_source()
    image_data, instances = ca.ensure_sources(anchor, fetch=False)
    truth = ca.coco_truth(instances, wanted)
    with image_data.open() as fh:
        coco_of = {int(m["image_id"]): int(m["coco_id"]) for m in json.load(fh) if m.get("coco_id")}
    corrections = load_corrections()
    roster = json.loads(pc.ROSTER.read_text()) if pc.ROSTER.exists() else {}
    log(f"{len(corrections)} human verdicts; roster pins {len(roster.get('cells', {}))} cells")

    # The fourth arm is not a mode: it is `dev`'s pass ORDER, folding before the
    # anchor with no dims, exactly as the loader ran it until #3637. It is here
    # to prove the claim the reorder rests on -- that folding an image COCO is
    # about to overwrite changes nothing -- rather than to leave it asserted.
    out: dict[str, dict] = {}
    for mode in (*vs.FOLD_MODES, LEGACY):
        # A fresh read per mode: `canonicalise` and every pass after it edit in
        # place, so a second mode over the first's labels would measure the two
        # composed. The read is ~90 s and the alternative is a deep copy of a
        # 100k-image dict, which is not cheaper.
        log(f"--- mode {mode}")
        labels = vs.read_vg_labels(records, paths, dims, wanted_vg)
        if mode == LEGACY:
            folded, contested = vs.canonicalise(labels, pc.SCALE_VG_NAMES)
            box_dims, exhaustive, *_ = vs.anchor_to_coco(labels, dims, coco_of, truth, ca.COCO_DIMS, wanted)
        else:
            box_dims, exhaustive, *_ = vs.anchor_to_coco(labels, dims, coco_of, truth, ca.COCO_DIMS, wanted)
            folded, contested = vs.canonicalise(labels, pc.SCALE_VG_NAMES, box_dims, mode)
        unbanded = vs.apply_corrections(labels, corrections, box_dims, exhaustive)
        unbanded |= vs.lift_ambiguous(labels, pc.SCALE_VG_AMBIGUOUS, exhaustive)
        supply, _, clean = vs.band_candidates(labels, box_dims, unbanded)
        chosen = vs.designate_cells(supply, corrections, roster)
        out[mode] = {
            "folded": folded,
            "contested": contested,
            "supply": {c: {b: len(v) for b, v in bands.items()} for c, bands in supply.items()},
            "clean": len(clean),
            "chosen": {cell: sorted(ids) for cell, ids in chosen.items()},
            "under_supplied": {
                pc.scale_cell(c, b): len(supply[c][b])
                for c in classes
                for b in pc.BOX_BANDS
                if len(supply[c][b]) < pc.SCALE_N_POS
            },
        }

    # What each mode does to work a human already did, and to the shipped
    # designations. A positive nobody has looked at is replaceable; a reviewed
    # one is not, and that is the whole of #3616's hazard.
    # The reorder is a no-op or it is not, and this is the whole test: same
    # supply, same designated ids, cell by cell.
    same = out[LEGACY]["supply"] == out["fold"]["supply"] and out[LEGACY]["chosen"] == out["fold"]["chosen"]
    log(f"reorder is a no-op on the built dataset: {same}")

    # Keyed on the PAIR, not the image: a human verdict on `(2409, backpack)`
    # says nothing about that image's `bird` seat, and counting it would report
    # reviewed work protected that nobody reviewed.
    pinned = {cell: set(ids) for cell, ids in roster.get("cells", {}).items()}
    for mode, d in out.items():
        churn = {}
        for cell, ids in d["chosen"].items():
            cls = cell.rsplit("@", 1)[0]
            was = pinned.get(cell, set())
            now = set(ids)
            churn[cell] = {
                "kept": len(was & now),
                "dropped": len(was - now),
                "dropped_reviewed": len({i for i in was - now if (i, cls) in corrections}),
                "added": len(now - was),
                # The denominator, without which `dropped_reviewed: 0` says
                # nothing: a cell nobody has reviewed cannot lose reviewed work,
                # and 18 of the 36 cells are in that position.
                "reviewed_designations": len({i for i in was if (i, cls) in corrections}),
            }
        d["churn"] = churn
        d["churn_total"] = {
            k: sum(v[k] for v in churn.values())
            for k in ("kept", "dropped", "dropped_reviewed", "added", "reviewed_designations")
        }
        del d["chosen"]
    out["reorder_is_a_no_op"] = same
    return out


# ----------------------------------------------------------------------- sheet


def sheet(examples: list, cls: str, out: Path, n: int) -> None:
    """The un-banded images as pictures, because the claim is about geometry.

    A band is a claim about one object's size, and "these two boxes are a
    scatter, not an object" is exactly the kind of claim a table cannot settle
    and a picture settles at a glance -- the argument #3281 lost for three
    studies. Each tile is the full frame with the class's own box in green, the
    box the alias spelling adds in purple, and COCO's exhaustive boxes dashed
    white over both.

    Pixels come from :func:`pilebuild.vgsource.vg_image_paths`, the loader's own
    resolver, so a sheet cannot point at a different copy of VG than the build.
    """
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.patches as mpatches  # noqa: PLC0415
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    rows = [e for e in examples if e["class"] == cls]
    if not rows:
        raise SystemExit(f"no un-banded examples for {cls!r}")
    # Widest union first: the biggest scatters are the clearest cases, and a
    # sheet that opens on a marginal one argues against itself.
    rows.sort(key=lambda e: -_union_area(e["own"] + e["folded"]))
    rows = rows[:n]

    paths = vg_image_paths()
    cols = 4
    nrow = (len(rows) + cols - 1) // cols
    fig, axes = plt.subplots(nrow, cols, figsize=(3.2 * cols, 3.0 * nrow), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, e in zip(axes.flat, rows, strict=False):
        path = paths.get(e["image_id"])
        if path is None:
            continue
        with Image.open(path) as im:
            ax.imshow(im.convert("RGB"))
            W, H = im.size
        for boxes, colour, style in (
            (e["coco"], "white", "--"),
            (e["own"], "#1b7837", "-"),
            (e["folded"], "#762a83", "-"),
        ):
            for b in boxes:
                ax.add_patch(
                    mpatches.Rectangle(
                        (b[0] * W, b[1] * H),
                        (b[2] - b[0]) * W,
                        (b[3] - b[1]) * H,
                        fill=False,
                        edgecolor=colour,
                        lw=2.0,
                        linestyle=style,
                    )
                )
        ax.set_title(
            f"{e['image_id']}  +{'/'.join(e['alias_names'])}\n"
            f"own {e['band_own']} -> fold {e['band_fold']} · COCO {e['band_coco']}",
            fontsize=8,
        )
    fig.suptitle(
        f"`{cls}`: what the fold un-bands. green = the class's own box, "
        f"purple = the alias box, dashed = COCO's exhaustive boxes",
        fontsize=10,
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    log(f"wrote {out}")


def _union_area(boxes: list[list[float]]) -> float:
    if not boxes:
        return 0.0
    return (max(b[2] for b in boxes) - min(b[0] for b in boxes)) * (max(b[3] for b in boxes) - min(b[1] for b in boxes))


# ---------------------------------------------------------------------- report


def report_truth(t: dict) -> None:
    modes = list(vs.FOLD_MODES)
    print("\n" + "=" * 96)
    print("TRUTH -- each mode's band against COCO's, on the VG-COCO overlap")
    print("Only images the class already bands and COCO confirms: a band is not a question about a")
    print("false positive. `unband` is the 248-shaped population (the fold leaves every band and the")
    print("class's own boxes did not); `move` is #3616's (both band, differently).")
    print("=" * 96)
    print(f"{'mode':<12}" + "".join(f"{sc + ' agree':>16}{'rate':>8}" for sc in ("all", "unband", "move")))
    for m in modes:
        row = f"{m:<12}"
        for sc in ("all", "unband", "move"):
            row += f"{t['agreement'][m][sc]:>10}/{t['seen'][sc]:<5}{_pct(t['agreement'][m][sc], t['seen'][sc]):>8}"
        print(row)

    for sc in ("unband", "move"):
        d = t["truth_says"].get(sc, {})
        tot = sum(d.values())
        print(f"\nOn the {tot} `{sc}` images, what COCO's exhaustive boxes actually say:")
        for k, v in sorted(d.items(), key=lambda kv: -kv[1]):
            print(f"  {k:<12}{v:>7}{_pct(v, tot):>9}")

    print("\nThe `unband` population per class -- and COCO's verdict on it:")
    keys = sorted({k for v in t["unband_by_class"].values() for k in v})
    print(f"{'class':<14}{'images':>8}" + "".join(f"{k:>12}" for k in keys))
    for c, v in sorted(t["unband_by_class"].items(), key=lambda kv: -sum(kv[1].values())):
        print(f"{c:<14}{sum(v.values()):>8}" + "".join(f"{v.get(k, 0):>12}" for k in keys))

    a = t["anchor_unbands"]
    n = a["banded_by_vg"]
    print("\n" + "=" * 96)
    print("THE BASE RATE -- what the exhaustive reference already does to a cleanly-banded image")
    print("`anchor_to_coco` replaces VG's labels with COCO's on 48% of VG, so this un-banding, on")
    print("evidence of exactly the same kind, is what the shipped build has always accepted in silence.")
    print("=" * 96)
    print(f"  banded by VG's own spelling      {n:>8}")
    print(
        f"  same band under COCO             {n - a['unbanded_by_coco'] - a['moved_by_coco'] - a['absent_in_coco']:>8}"
    )
    print(f"  un-banded by COCO's boxes        {a['unbanded_by_coco']:>8}{_pct(a['unbanded_by_coco'], n):>9}")
    print(f"  moved to another band by COCO    {a['moved_by_coco']:>8}{_pct(a['moved_by_coco'], n):>9}")
    print(f"  not there at all (a false box)   {a['absent_in_coco']:>8}{_pct(a['absent_in_coco'], n):>9}")

    print("\n" + "=" * 96)
    print("THE GUARD'S OWN THRESHOLD -- images the fold un-bands, and how often the truth agrees")
    print("=" * 96)
    print(f"{'inflation':<12}{'un-banded':>11}{'truth scatters':>17}{'share':>9}")
    for k, v in sorted(t["inflation"].items(), key=lambda kv: float(kv[0])):
        print(f"{k:<12}{v['contested']:>11}{v['truth_scatters']:>17}{_pct(v['truth_scatters'], v['contested']):>9}")


def report_supply(s: dict) -> None:
    modes = list(vs.FOLD_MODES)
    print("\n" + "=" * 92)
    print("SUPPLY -- the real build passes under each mode, to designation and no further")
    print("=" * 92)
    print(f"{'mode':<20}{'clean pool':>12}{'cells < N_POS':>15}{'boxes folded':>14}{'contested':>11}")
    for m in (*modes, LEGACY):
        d = s[m]
        print(
            f"{m:<20}{d['clean']:>12}{len(d['under_supplied']):>15}"
            f"{sum(d['folded'].values()):>14}{sum(d['contested'].values()):>11}"
        )
    print(f"\n`{LEGACY}` designates exactly what `fold` does: {s['reorder_is_a_no_op']}")

    print("\nPositive supply per cell, and what the mode changes (vs `fold`):")
    base = s["fold"]["supply"]
    print(f"{'cell':<24}{'fold':>8}" + "".join(f"{m:>12}" for m in modes[1:]))
    for c in pc.SCALE_CLASSES:
        for b in pc.BOX_BANDS:
            row = f"{pc.scale_cell(c, b):<24}{base[c][b]:>8}"
            for m in modes[1:]:
                row += f"{s[m]['supply'][c][b] - base[c][b]:>+12}"
            flag = "  UNDER" if base[c][b] < pc.SCALE_N_POS else ""
            print(row + flag)

    print("\nDesignated membership against the shipped roster (what a rebuild would do):")
    print(f"{'mode':<12}{'kept':>8}{'dropped':>10}{'of which reviewed':>19}{'added':>8}{'reviewed seats':>16}")
    for m in modes:
        t = s[m]["churn_total"]
        print(
            f"{m:<12}{t['kept']:>8}{t['dropped']:>10}{t['dropped_reviewed']:>19}"
            f"{t['added']:>8}{t['reviewed_designations']:>16}"
        )


def _pct(a: int, b: int) -> str:
    return f"{100.0 * a / b:.1f}%" if b else "--"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=("truth", "supply", "both"), default="both")
    ap.add_argument("--anchor-dir", default=str(pc.PILE / "coco_anchor"))
    ap.add_argument("--inflation", default="1.2,1.5,2.0,3.0,5.0")
    ap.add_argument("--out", default="")
    ap.add_argument("--examples-out", default="", help="every un-banded image, with its boxes and all three verdicts")
    ap.add_argument(
        "--sheet", default="", help="render that class's un-banded images (needs --examples-out or --examples)"
    )
    ap.add_argument("--examples", default="", help="read the examples file instead of re-measuring")
    ap.add_argument("--sheet-n", type=int, default=8)
    ap.add_argument("--sheet-out", default="")
    args = ap.parse_args()

    anchor = Path(args.anchor_dir)
    inflations = [float(x) for x in args.inflation.split(",") if x.strip()]
    payload: dict = {"meta": {"inflation": inflations, "band_max_inflation": pc.BAND_MAX_INFLATION}}

    examples: list = []
    if args.examples:
        examples = json.loads(Path(args.examples).read_text())
    elif args.phase in ("truth", "both"):
        log(f"loading VG objects.json ({(VG_ROOT / 'objects.json').stat().st_size / 1e6:.0f} MB)")
        with (VG_ROOT / "objects.json").open() as fh:
            records = json.load(fh)
        log(f"  {len(records)} VG records")
        payload["truth"] = phase_truth(anchor, records, inflations, examples)
        report_truth(payload["truth"])
        del records
        if args.examples_out:
            Path(args.examples_out).write_text(json.dumps(examples, indent=1) + "\n")
            print(f"wrote {args.examples_out} ({len(examples)} rows)")

    if args.sheet:
        sheet(examples, args.sheet, Path(args.sheet_out or f"{args.sheet}-unbanded.jpg"), args.sheet_n)

    # A sheet asked for off a finished run is a re-render, not a measurement.
    if args.sheet and args.examples:
        return 0

    if args.phase in ("supply", "both"):
        payload["supply"] = phase_supply(anchor)
        report_supply(payload["supply"])

    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=1) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
