"""What does COCO fold into this class? Measure it, so a guide need not guess.

`book` was annotated by COCO's annotators over magazines, because COCO has no
magazine class. Our reviewer applied the narrower English reading, and the class
split into two definitions wearing one name -- 21 verdicts on COCO's reading and
49 on a different one (`make_definition_reslate.py`). Nothing in the pipeline
could see it: supply was fine, boxes were fine, every structural check passed.

The split was **visible in the data the whole time**. On the ~48% of VG that is
COCO-sourced, both vocabularies annotate the same pixels: COCO draws a box and
calls it `book`, VG draws the same box and calls it `magazine`. Asking which VG
names land on a COCO class's boxes therefore *enumerates the boundary cases*
before a human sees one, and it would have printed `magazine` under `book`.

Two directions, and they answer different questions:

* **fold-in** (COCO box -> VG names on it): what a reviewer must ACCEPT. If VG
  calls it `magazine` and COCO calls it `book`, then a slate built on COCO's
  reading contains magazines and the guide has to say so.
* **fold-out** (VG box -> COCO class under it): what the VG name actually
  DENOTES. VG `phone` may be a cell phone (COCO `cell phone`) or a landline
  (COCO has no such class, so the box lands on nothing). A VG name whose boxes
  mostly land on no COCO class is a name that does not mean what COCO means.

**The self-match rate is the canary.** COCO `truck` boxes should mostly be
called `truck` by VG. VG's and COCO's boxes live in different pixel spaces --
VG ships downscaled copies of the COCO originals, which is what made #3281 park
130 boxes on the frame origin -- so both are normalised by their OWN source's
dimensions before any IoU. If that normalisation were wrong the self-match would
collapse, which is why it is reported first and per class.

Usage::

    python coco_folds.py --classes truck,cup,bowl --out folds.json
"""

from __future__ import annotations

import argparse
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import pile_config as pc

pc.setup_env()

VG_ROOT = pc.DEMO_CACHE / "visual_genome"


def log(msg: str) -> None:
    print(f"[folds] {msg}", flush=True)


def iou(a: list[float], b: list[float]) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = ix1 - ix0, iy1 - iy0
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def coco_boxes(anchor: Path) -> tuple[dict, dict, dict]:
    """``({coco_id: {class: [norm boxes]}}, {coco_id: (W, H)}, {coco_id: set(all classes)})``.

    **Every COCO class is loaded**, not just the ones the caller asked about.
    Fold-out needs to say "this VG box sits on a COCO *bench*", and it can only
    say that if `bench`'s boxes are here to be tested against.

    This used to take a `classes` filter and keep only those boxes, which made
    fold-out answer with the caller's own question: a class nobody named had no
    boxes, so every VG box over it fell through to `(no COCO class)`. `bike`
    read 100% "means nothing" against a recorded 40.1%, because `bicycle` was
    not in the set -- and 100% is exactly the reading that banishes a good
    spelling to :data:`pile_config.SCALE_VG_AMBIGUOUS` and costs the class half
    its positives (#3640, the #3605 failure again). The filter saved nothing
    worth having: the boxes are small beside `objects.json`, which every caller
    of this function also loads.
    """
    zip_path = anchor / "annotations_trainval2017.zip"
    files = [anchor / "instances_val2017.json", anchor / "instances_train2017.json"]
    if not all(f.exists() for f in files) and zip_path.exists():
        log(f"extracting from {zip_path.name}")
        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.namelist():
                if member.endswith(("instances_val2017.json", "instances_train2017.json")):
                    (anchor / Path(member).name).write_bytes(zf.read(member))

    want_boxes: dict[int, dict[str, list[list[float]]]] = {}
    dims: dict[int, tuple[int, int]] = {}
    present: dict[int, set[str]] = defaultdict(set)
    for path in files:
        if not path.exists():
            log(f"  MISSING {path.name}; skipping")
            continue
        log(f"loading {path.name} ({path.stat().st_size / 1e6:.0f} MB)")
        with path.open() as fh:
            data = json.load(fh)
        cat = {c["id"]: c["name"] for c in data["categories"]}
        for img in data["images"]:
            iid = int(img["id"])
            dims[iid] = (int(img["width"]), int(img["height"]))
            want_boxes.setdefault(iid, {})
        for ann in data["annotations"]:
            name = cat.get(ann["category_id"])
            if name is None:
                continue
            iid = int(ann["image_id"])
            present[iid].add(name)
            x, y, w, h = (float(v) for v in ann["bbox"])
            if w <= 0 or h <= 0:
                continue
            W, H = dims[iid]
            want_boxes[iid].setdefault(name, []).append([x / W, y / H, (x + w) / W, (y + h) / H])
    return want_boxes, dims, present


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--classes", default="", help="comma-separated COCO class names (default: SCALE_CLASSES)")
    ap.add_argument("--anchor-dir", default=str(pc.PILE / "coco_anchor"))
    ap.add_argument("--iou", type=float, default=0.5, help="IoU above which two boxes are the same object")
    ap.add_argument("--top", type=int, default=12, help="VG names to print per class")
    ap.add_argument("--min-count", type=int, default=5, help="ignore VG names seen fewer times than this")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    classes = set(c.strip() for c in args.classes.split(",") if c.strip()) or set(pc.SCALE_CLASSES)
    anchor = Path(args.anchor_dir)

    cboxes, cdims, _ = coco_boxes(anchor)
    log(
        f"  {len(cdims)} COCO images; {sum(len(v) for v in cboxes.values())} class-groups over the full COCO vocabulary"
    )

    log("loading VG image_data.json")
    with (anchor / "image_data.json").open() as fh:
        meta = json.load(fh)
    coco_of = {int(m["image_id"]): int(m["coco_id"]) for m in meta if m.get("coco_id")}
    vdims = {int(m["image_id"]): (int(m["width"]), int(m["height"])) for m in meta}
    log(f"  {len(coco_of)} of {len(meta)} VG images carry a coco_id")

    log(f"loading VG objects.json ({(VG_ROOT / 'objects.json').stat().st_size / 1e6:.0f} MB)")
    with (VG_ROOT / "objects.json").open() as fh:
        records = json.load(fh)
    log(f"  {len(records)} VG records")

    # fold-in: COCO box of class c  ->  the VG names sitting on it
    fold_in: dict[str, dict[str, int]] = {c: defaultdict(int) for c in classes}
    n_coco_boxes: dict[str, int] = defaultdict(int)
    n_matched: dict[str, int] = defaultdict(int)
    # fold-out: VG box named n  ->  the COCO class under it ("" = no COCO class)
    fold_out: dict[str, dict[str, int]] = {c: defaultdict(int) for c in classes}
    n_vg_boxes: dict[str, int] = defaultdict(int)

    skipped_aspect = 0
    considered = 0
    for rec in records:
        iid = int(rec["image_id"])
        cid = coco_of.get(iid)
        if cid is None or cid not in cdims:
            continue
        vd, cd = vdims.get(iid), cdims[cid]
        if not vd:
            continue
        # A re-crop or rotation breaks normalised transfer; a pure rescale does not.
        if not pc.aspect_transferable(vd, cd):
            skipped_aspect += 1
            continue
        considered += 1
        W, H = vd

        vg_named: list[tuple[str, list[float]]] = []
        for obj in rec.get("objects") or []:
            names = obj.get("names") or []
            if not names:
                continue
            name = str(names[0]).strip().lower()
            x, y = float(obj.get("x", 0)), float(obj.get("y", 0))
            w, h = float(obj.get("w", 0)), float(obj.get("h", 0))
            if w <= 0 or h <= 0:
                continue
            vg_named.append((name, [x / W, y / H, (x + w) / W, (y + h) / H]))

        # fold-in is asked only of the classes the caller named; fold-out below
        # is asked of the whole vocabulary, which is why `cboxes` carries it.
        on_image = cboxes.get(cid, {})
        for c in classes:
            for cb in on_image.get(c, []):
                n_coco_boxes[c] += 1
                hits = {n for n, vb in vg_named if iou(cb, vb) >= args.iou}
                if hits:
                    n_matched[c] += 1
                for n in hits:
                    fold_in[c][n] += 1

        # fold-out is asked of the VG name we would actually build the class
        # from, which is the class name itself plus whatever merges into it.
        for n, vb in vg_named:
            if n not in classes:
                continue
            n_vg_boxes[n] += 1
            under = set()
            for c2, cbs in on_image.items():
                if any(iou(cb, vb) >= args.iou for cb in cbs):
                    under.add(c2)
            if not under:
                fold_out[n]["(no COCO class)"] += 1
            for c2 in under:
                fold_out[n][c2] += 1

    log(f"considered {considered} VG/COCO pairs; skipped {skipped_aspect} on aspect drift")

    print("\n" + "=" * 78)
    print("SELF-MATCH CANARY -- COCO's box for class c, called c by VG")
    print("If this collapses, the two coordinate spaces are not aligned (#3281).")
    print("=" * 78)
    print(f"{'class':<16}{'coco boxes':>11}{'any VG box':>12}{'named c':>10}{'self %':>8}")
    for c in sorted(classes):
        nb, nm, sm = n_coco_boxes[c], n_matched[c], fold_in[c].get(c, 0)
        pct = 100.0 * sm / nb if nb else 0.0
        print(f"{c:<16}{nb:>11}{nm:>12}{sm:>10}{pct:>7.1f}%")

    print("\n" + "=" * 78)
    print("FOLD-IN -- what a reviewer on COCO's reading must ACCEPT")
    print("VG names landing on a COCO box of the class, as % of that class's boxes.")
    print("=" * 78)
    for c in sorted(classes):
        nb = n_coco_boxes[c]
        if not nb:
            print(f"\n{c}: no COCO boxes on the VG overlap")
            continue
        rows = [(v, k) for k, v in fold_in[c].items() if v >= args.min_count and k != c]
        rows.sort(reverse=True)
        print(f"\n{c}  ({nb} COCO boxes; {100.0 * n_matched[c] / nb:.0f}% carry any VG box)")
        if not rows:
            print("    (no other VG name reaches the floor -- the name is unambiguous here)")
        for v, k in rows[: args.top]:
            print(f"    {k:<24}{v:>6}  {100.0 * v / nb:>5.1f}%")

    print("\n" + "=" * 78)
    print("FOLD-OUT -- what the VG name actually DENOTES")
    print("COCO classes under a VG box of this name. A large '(no COCO class)'")
    print("share means the VG name covers objects COCO does not have.")
    print("=" * 78)
    for c in sorted(classes):
        nb = n_vg_boxes[c]
        if not nb:
            print(f"\n{c}: no VG boxes of this name on the overlap")
            continue
        rows = [(v, k) for k, v in fold_out[c].items() if v >= args.min_count]
        rows.sort(reverse=True)
        print(f"\n{c}  ({nb} VG boxes)")
        for v, k in rows[: args.top]:
            print(f"    {k:<24}{v:>6}  {100.0 * v / nb:>5.1f}%")

    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "meta": {
                        "iou": args.iou,
                        "classes": sorted(classes),
                        "pairs_considered": considered,
                        "skipped_aspect_drift": skipped_aspect,
                    },
                    "coco_boxes": dict(n_coco_boxes),
                    "matched_any": dict(n_matched),
                    "vg_boxes": dict(n_vg_boxes),
                    "fold_in": {c: dict(v) for c, v in fold_in.items()},
                    "fold_out": {c: dict(v) for c, v in fold_out.items()},
                },
                indent=1,
            )
            + "\n"
        )
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
