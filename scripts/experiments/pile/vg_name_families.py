"""Every VG primary name that shares a class's head noun, with its supply.

``coco_folds.py`` can only see a name that lands on a COCO box, which is a
sample of the ~48% of VG that COCO annotates. That is the right place to
*validate* a name -- it is the only half with an exhaustive reference -- but it
is the wrong place to *find* one. A spelling used mostly on the other half
shows up there with a handful of boxes or none at all, and the half it is used
on is precisely the half where the miss turns into a **negative for its own
class** (#3605).

So this enumerates candidates from VG's own vocabulary instead, by head noun:
`blue umbrella`, `beach umbrella` and `umbrella.` all have `umbrella` as their
final token, and a name's head noun is what it denotes. Plurals are included by
a small stemmer, because `magazine`/`magazines` split in the data and a plural
is a spelling like any other.

**This is a recall aid and nothing else.** Two things it gets wrong on purpose,
both of which the fold-out column settles and neither of which a rule could:

* `hot dog` has the head noun `dog` and is a different object -- and a COCO
  class in its own right. So is `sea bird` (a bird) against `bird bath` (not).
  A head-noun family is a list of names to *measure*, never a list to fold.
* It cannot find a spelling that shares no head noun with the class: `bike`,
  `parasail`, `magazine`, `duck`. Those come from the fold-in column, which is
  measured against COCO and is the other half of the search.

Usage::

    python vg_name_families.py --min-images 5 --out families.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pile_config as pc

pc.setup_env()

VG_ROOT = pc.DEMO_CACHE / "visual_genome"


def log(msg: str) -> None:
    print(f"[families] {msg}", flush=True)


#: The head noun and its plurals are :mod:`pile_config`'s, not this module's.
#: `name_evidence.py --pooled` matches a construction against the same head, and
#: two implementations of "what is this name's head noun" would drift silently.
head = pc.name_head
singulars = pc.name_singulars
PUNCT = pc.NAME_PUNCT


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--classes", default="", help="comma-separated (default: SCALE_CLASSES)")
    ap.add_argument("--anchor-dir", default=str(pc.PILE / "coco_anchor"))
    ap.add_argument("--min-images", type=int, default=5, help="ignore a name seen on fewer VG images")
    ap.add_argument(
        "--also",
        default="",
        help="extra VG names to count, comma-separated. A spelling that shares no head noun with "
        "its class (`bike`, `duck`, `magazine`) comes from coco_folds.py, not from here -- but its "
        "supply is the same question, and it is one pass over the same file.",
    )
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    classes = [c.strip().lower() for c in args.classes.split(",") if c.strip()] or list(pc.SCALE_CLASSES)
    heads = {c: head(c) for c in classes}

    log("loading VG image_data.json")
    with (Path(args.anchor_dir) / "image_data.json").open() as fh:
        meta = json.load(fh)
    on_coco = {int(m["image_id"]) for m in meta if m.get("coco_id")}

    log(f"loading VG objects.json ({(VG_ROOT / 'objects.json').stat().st_size / 1e6:.0f} MB)")
    with (VG_ROOT / "objects.json").open() as fh:
        records = json.load(fh)
    log(f"  {len(records)} VG records; {len(on_coco)} carry a coco_id")

    n_images: dict[str, int] = defaultdict(int)
    n_boxes: dict[str, int] = defaultdict(int)
    n_off: dict[str, int] = defaultdict(int)
    for rec in records:
        iid = int(rec["image_id"])
        seen: set[str] = set()
        for obj in rec.get("objects") or []:
            names = obj.get("names") or []
            if not names:
                continue
            name = str(names[0]).strip().lower()
            if not name:
                continue
            n_boxes[name] += 1
            seen.add(name)
        for name in seen:
            n_images[name] += 1
            if iid not in on_coco:
                n_off[name] += 1

    log(f"  {len(n_images)} distinct primary names")

    extra = {
        n.strip().lower(): {
            "name": n.strip().lower(),
            "images": n_images.get(n.strip().lower(), 0),
            "boxes": n_boxes.get(n.strip().lower(), 0),
            "non_coco_images": n_off.get(n.strip().lower(), 0),
        }
        for n in args.also.split(",")
        if n.strip()
    }

    families: dict[str, list[dict]] = {}
    for c in classes:
        want = heads[c]
        rows = []
        for name, imgs in n_images.items():
            if name == c or imgs < args.min_images:
                continue
            if want in singulars(head(name)):
                rows.append({"name": name, "images": imgs, "boxes": n_boxes[name], "non_coco_images": n_off[name]})
        rows.sort(key=lambda r: -r["images"])
        families[c] = rows

    print("\n" + "=" * 84)
    print("HEAD-NOUN FAMILIES -- names to measure, not names to fold")
    print(f"VG primary names sharing a class's head noun, seen on >= {args.min_images} images.")
    print("=" * 84)
    for c in classes:
        rows = families[c]
        own = n_images.get(c, 0)
        tot = sum(r["images"] for r in rows)
        off = sum(r["non_coco_images"] for r in rows)
        print(f"\n{c}  ({own} images under the class name; {len(rows)} family names on {tot} images, {off} off-COCO)")
        for r in rows[:40]:
            print(f"    {r['name']:<28}{r['images']:>7} imgs{r['boxes']:>8} boxes{r['non_coco_images']:>8} off-COCO")
        if len(rows) > 40:
            print(f"    ... and {len(rows) - 40} more")

    if extra:
        print("\n" + "=" * 84)
        print("NAMED SEPARATELY -- supply for names that share no head noun with any class")
        print("=" * 84)
        for r in sorted(extra.values(), key=lambda r: -r["images"]):
            print(f"    {r['name']:<28}{r['images']:>7} imgs{r['boxes']:>8} boxes{r['non_coco_images']:>8} off-COCO")

    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "meta": {"min_images": args.min_images, "classes": classes, "distinct_names": len(n_images)},
                    "class_images": {c: n_images.get(c, 0) for c in classes},
                    "families": families,
                    "also": extra,
                },
                indent=1,
            )
            + "\n"
        )
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
