"""What does a `vg_scale` name table actually buy, and what does it cost?

``coco_folds.py`` says which VG names land on a class's COCO boxes. It cannot
say what adding them to :data:`pile_config.SCALE_VG_NAMES` would *do*, for two
reasons: its counts are per box, so a class's names double-count every box two
of them share, and it only ever looks at the VG-COCO overlap -- the half where
the defect does not exist, because ``anchor_to_coco`` overwrites VG's labels
with COCO's there anyway.

The defect lives on the other half. On the ~52% of VG that COCO does not
annotate, VG's silence is the only evidence of absence, so an image whose only
`bicycle` is annotated `bike` is not a missing positive: it is a **negative for
its own class**, and a detector is scored wrong for finding the bicycle that is
really there (#3605). The number that matters is therefore how many such images
a proposed table repairs, counted on the non-COCO half, per class.

Three columns, and they answer three different questions:

* **overlap coverage** -- of the COCO boxes of class *c*, the share carried by
  some VG box named *c*, and the share carried by *c* or a proposed name. This
  is the only column with an exhaustive reference under it, so it is what
  validates the proposal. Counted as a **union over boxes**, not a sum over
  names.
* **repaired negatives** -- non-COCO images holding a proposed ALIAS name and no
  *c* box. These stop being negatives and become positives, which is the whole
  point of the alias table.
* **withheld negatives** -- non-COCO images holding a proposed AMBIGUOUS name
  and nothing else that settles the class. These stop being negatives and become
  nothing at all: ``lift_ambiguous`` drops them from the bands and from the
  shared pool alike. That is the price of the ambiguous table, and it is the
  number that decides whether a broad name like `sign` can be listed: the pool
  has to survive it.

A fourth column is free and worth printing, because it is a cheaper fix if it
is large. VG names an object with a *list* of synonyms and
``vg_boxes_by_name`` matches the primary only, so some share of the miss is
recoverable with no table at all -- just by reading the rest of the list.
``synonym`` reports how often a candidate-named object already carries the class
name further down its own ``names``.

Usage::

    python name_coverage.py                       # score the shipped tables
    python name_coverage.py --propose prop.json   # score a proposal instead
    python name_coverage.py --propose prop.json --out coverage.json

The proposal file is ``{"alias": {class: [names]}, "ambiguous": {class: [names]}}``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pile_config as pc

pc.setup_env()

import coco_folds as cf  # noqa: E402  (setup_env must run before vtscore resolves)
from pilebuild.loaders.vg_scale import OVERSIZE, SCATTERED, band_for  # noqa: E402

VG_ROOT = pc.DEMO_CACHE / "visual_genome"


def log(msg: str) -> None:
    print(f"[coverage] {msg}", flush=True)


def load_tables(path: str) -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
    """``(alias, ambiguous)`` from a proposal file, or the shipped tables."""
    if not path:
        return dict(pc.SCALE_VG_NAMES), dict(pc.SCALE_VG_AMBIGUOUS)
    raw = json.loads(Path(path).read_text())
    alias = {c: tuple(n.strip().lower() for n in ns) for c, ns in (raw.get("alias") or {}).items()}
    ambig = {c: tuple(n.strip().lower() for n in ns) for c, ns in (raw.get("ambiguous") or {}).items()}
    unknown = (set(alias) | set(ambig)) - set(pc.SCALE_CLASSES)
    if unknown:
        raise SystemExit(f"proposal names classes that are not in C: {sorted(unknown)}")
    return alias, ambig


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--propose", default="", help="JSON {alias: {...}, ambiguous: {...}}; default = the shipped tables")
    ap.add_argument("--anchor-dir", default=str(pc.PILE / "coco_anchor"))
    ap.add_argument("--iou", type=float, default=0.5, help="IoU above which two boxes are the same object")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    alias, ambig = load_tables(args.propose)
    classes = list(pc.SCALE_CLASSES)
    # Every name the read must see, so an image holding one is never mistaken
    # for an image holding nothing.
    wanted = set(classes)
    for table in (alias, ambig):
        for ns in table.values():
            wanted.update(ns)
    log(f"{len(classes)} classes; {len(wanted)} VG names to match")

    #: The COCO classes whose boxes count as *c* -- its whole footprint, not its
    #: namesake alone. `cup` is `cup` U `wine glass`
    #: (:data:`pile_config.SCALE_CLASS_MERGES`), so reading its coverage off COCO
    #: `cup` omits every `wine glass` box the merge claims: the denominator is
    #: short AND the stemware spellings appear to recover nothing (#3700).
    #: Returns ``{c}`` for an unmerged class, so nothing else moves.
    coco_for = {c: pc.coco_classes_for(c) for c in classes}

    cboxes, cdims, _ = cf.coco_boxes(Path(args.anchor_dir))

    log("loading VG image_data.json")
    with (Path(args.anchor_dir) / "image_data.json").open() as fh:
        meta = json.load(fh)
    coco_of = {int(m["image_id"]): int(m["coco_id"]) for m in meta if m.get("coco_id")}
    vdims = {int(m["image_id"]): (int(m["width"]), int(m["height"])) for m in meta}

    log(f"loading VG objects.json ({(VG_ROOT / 'objects.json').stat().st_size / 1e6:.0f} MB)")
    with (VG_ROOT / "objects.json").open() as fh:
        records = json.load(fh)
    log(f"  {len(records)} VG records")

    # --- counters -----------------------------------------------------------
    # overlap: COCO boxes of c carried by the class name / by class+alias /
    # by class+alias+ambiguous. Union over boxes, so a box two names share
    # counts once.
    n_coco = dict.fromkeys(classes, 0)
    hit_own = dict.fromkeys(classes, 0)
    hit_alias = dict.fromkeys(classes, 0)
    hit_any = dict.fromkeys(classes, 0)
    # non-COCO half: images by what evidence they carry for c
    off_total = 0
    off_own = dict.fromkeys(classes, 0)
    off_repaired = dict.fromkeys(classes, 0)
    off_withheld = dict.fromkeys(classes, 0)
    # synonym recovery: candidate-named objects already carrying c in names[1:]
    syn_seen = dict.fromkeys(classes, 0)
    syn_hit = dict.fromkeys(classes, 0)
    # what folding an alias does to the band of an image the class ALREADY sees,
    # and whether a repaired image lands in a band at all
    band_now = dict.fromkeys(classes, 0)
    band_same = dict.fromkeys(classes, 0)
    band_moved = dict.fromkeys(classes, 0)
    band_lost = dict.fromkeys(classes, 0)
    band_new = dict.fromkeys(classes, 0)

    skipped_aspect = 0
    for rec in records:
        iid = int(rec["image_id"])
        vd = vdims.get(iid)
        if vd is None:
            continue

        by_name: dict[str, list[list[float]]] = {}
        for obj in rec.get("objects") or []:
            names = [str(n).strip().lower() for n in (obj.get("names") or []) if str(n).strip()]
            if not names:
                continue
            name = names[0]
            if name not in wanted:
                continue
            x, y = float(obj.get("x", 0)), float(obj.get("y", 0))
            w, h = float(obj.get("w", 0)), float(obj.get("h", 0))
            if w <= 0 or h <= 0:
                continue
            by_name.setdefault(name, []).append([x / vd[0], y / vd[1], (x + w) / vd[0], (y + h) / vd[1]])
            for c in classes:
                if name != c and (name in alias.get(c, ()) or name in ambig.get(c, ())):
                    syn_seen[c] += 1
                    if c in names[1:]:
                        syn_hit[c] += 1

        cid = coco_of.get(iid)
        if cid is None or cid not in cdims:
            # --- the non-COCO half: VG's silence is the only evidence -------
            off_total += 1
            for c in classes:
                own = by_name.get(c, [])
                folded = [b for n in alias.get(c, ()) for b in by_name.get(n, [])]
                if own:
                    off_own[c] += 1
                    # Boxes are normalised here, so band_for is asked in a unit
                    # square: its rule is a share of image area either way.
                    before = band_for(own, 1, 1)
                    if before in pc.BOX_BANDS:
                        band_now[c] += 1
                        after = band_for(own + folded, 1, 1) if folded else before
                        if after == before:
                            band_same[c] += 1
                        elif after in (SCATTERED, OVERSIZE):
                            band_lost[c] += 1
                        else:
                            band_moved[c] += 1
                elif folded:
                    off_repaired[c] += 1
                    if band_for(folded, 1, 1) in pc.BOX_BANDS:
                        band_new[c] += 1
                elif any(n in by_name for n in ambig.get(c, ())):
                    off_withheld[c] += 1
            continue

        # --- the overlap: score the proposal against COCO -------------------
        cd = cdims[cid]
        if not pc.aspect_transferable(vd, cd):
            skipped_aspect += 1
            continue
        # `coco_boxes` carries the whole COCO vocabulary (#3640); only the
        # classes under proposal are scored, and the counters are keyed on them.
        #
        # A class this project defines as a UNION is scored against its whole
        # COCO footprint: reading `cup`'s coverage off COCO `cup` alone omits
        # every `wine glass` box the merge claims, understating the denominator
        # AND missing what the stemware spellings recover (#3700).
        on_image = cboxes.get(cid, {})
        for c in classes:
            boxes = [b for k in coco_for[c] for b in on_image.get(k, [])]
            if not boxes:
                continue
            own = by_name.get(c, [])
            folded = [b for n in alias.get(c, ()) for b in by_name.get(n, [])]
            amb = [b for n in ambig.get(c, ()) for b in by_name.get(n, [])]
            for cb in boxes:
                n_coco[c] += 1
                h_own = any(cf.iou(cb, vb) >= args.iou for vb in own)
                h_als = h_own or any(cf.iou(cb, vb) >= args.iou for vb in folded)
                hit_own[c] += h_own
                hit_alias[c] += h_als
                hit_any[c] += h_als or any(cf.iou(cb, vb) >= args.iou for vb in amb)

    log(f"non-COCO half: {off_total} images; skipped {skipped_aspect} overlaps on aspect drift")

    def pct(a: int, b: int) -> str:
        return f"{100.0 * a / b:.1f}%" if b else "--"

    print("\n" + "=" * 96)
    print("OVERLAP COVERAGE -- of a class's COCO boxes, the share some VG box of ours lands on")
    print("Union over boxes. `+alias` is what canonicalise would fold in; `+ambig` is what")
    print("lift_ambiguous would suppress, and is an upper bound on the alias table, not a gain.")
    print("=" * 96)
    print(
        f"{'class':<14}{'coco boxes':>11}{'own':>8}{'+alias':>9}{'+ambig':>9}   {'own %':>7}{'alias %':>9}{'ambig %':>9}"
    )
    for c in classes:
        print(
            f"{c:<14}{n_coco[c]:>11}{hit_own[c]:>8}{hit_alias[c]:>9}{hit_any[c]:>9}   "
            f"{pct(hit_own[c], n_coco[c]):>7}{pct(hit_alias[c], n_coco[c]):>9}{pct(hit_any[c], n_coco[c]):>9}"
        )

    print("\n" + "=" * 96)
    print(f"THE NON-COCO HALF -- {off_total} images where VG's silence is the only evidence of absence")
    print("`repaired` were negatives for their own class and become positives; `withheld` leave")
    print("the negative pool without becoming anything. Both are counted against `own`, the")
    print("images the class can already see.")
    print("=" * 96)
    print(f"{'class':<14}{'own imgs':>10}{'repaired':>10}{'withheld':>10}   {'repaired/own':>13}{'withheld/pool':>14}")
    for c in classes:
        print(
            f"{c:<14}{off_own[c]:>10}{off_repaired[c]:>10}{off_withheld[c]:>10}   "
            f"{pct(off_repaired[c], off_own[c]):>13}{pct(off_withheld[c], off_total):>14}"
        )

    print("\n" + "=" * 96)
    print("SYNONYM RECOVERY -- candidate-named objects already carrying the class name in names[1:]")
    print("A large share here means the miss is fixable in the reader, with no table at all.")
    print("=" * 96)
    print(f"{'class':<14}{'objects':>10}{'carry c':>10}{'share':>9}")
    for c in classes:
        if syn_seen[c]:
            print(f"{c:<14}{syn_seen[c]:>10}{syn_hit[c]:>10}{pct(syn_hit[c], syn_seen[c]):>9}")

    print("\n" + "=" * 96)
    print("BAND EFFECT on the non-COCO half -- canonicalise merges the alias boxes into the")
    print("class's, so the union box a band is read off can MOVE (#3616), or trip the scatter")
    print("filter and leave every band. `new` is repaired images that do land in a band.")
    print("=" * 96)
    print(f"{'class':<14}{'banded now':>12}{'same':>8}{'moved':>8}{'lost':>8}   {'repaired':>9}{'of which banded':>17}")
    for c in classes:
        print(
            f"{c:<14}{band_now[c]:>12}{band_same[c]:>8}{band_moved[c]:>8}{band_lost[c]:>8}   "
            f"{off_repaired[c]:>9}{band_new[c]:>17}"
        )

    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "meta": {
                        "iou": args.iou,
                        "proposal": args.propose or "(shipped tables)",
                        "non_coco_images": off_total,
                        "skipped_aspect_drift": skipped_aspect,
                    },
                    "alias": {c: list(ns) for c, ns in alias.items()},
                    "ambiguous": {c: list(ns) for c, ns in ambig.items()},
                    "overlap": {
                        c: {"coco_boxes": n_coco[c], "own": hit_own[c], "alias": hit_alias[c], "ambig": hit_any[c]}
                        for c in classes
                    },
                    "non_coco": {
                        c: {"own": off_own[c], "repaired": off_repaired[c], "withheld": off_withheld[c]}
                        for c in classes
                    },
                    "synonym": {c: {"objects": syn_seen[c], "carry_class": syn_hit[c]} for c in classes},
                    "bands": {
                        c: {
                            "banded_now": band_now[c],
                            "same": band_same[c],
                            "moved": band_moved[c],
                            "lost": band_lost[c],
                            "repaired_banded": band_new[c],
                        }
                        for c in classes
                    },
                },
                indent=1,
            )
            + "\n"
        )
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
