#!/usr/bin/env python3
"""VG spellings a head-noun family cannot reach: the misspellings (#3663).

`vg_name_families.py` enumerates a class's candidates by **head noun** --
`beach umbrella` and `umbrella's` share `umbrella` as their final token. That
search cannot reach a *misspelling* by construction: `umberella` has no token in
common with `umbrella`, so no head-noun family will ever contain it, and
`name_evidence.py` has therefore never scored one.

#3663 is the observation that this leaves a residue nobody has measured, and it
names five examples for `umbrella` alone. This closes that hole from the other
side: enumerate VG's primary names within a small **edit distance** of a class
name or one of its shipped aliases, and emit them as a `--candidates` file so
the existing evidence machinery scores them exactly like any other name.

**Edit distance is a recall aid and nothing else**, exactly as the head-noun
family is. `bowl` is one edit from `bow`, `bill`, `ball` and `boil`; `car` is
one from `cat`, `bar`, `care` and `cart`. Every one of those is a different
object, and the point is not to fold them -- it is to put them in front of the
same three cuts (`precision`, its Wilson lower bound, and box agreement) that
decide every other name. A candidate list is a list to *measure*.

Two guards keep the list honest rather than long:

* **a length floor**, because at four characters an edit distance of one is most
  of the vocabulary -- `cup` would drag in `cap`, `cut`, `can`, `cop`, `up`;
* **the shipped tables are excluded**, so what comes out is only what the
  curation actually dropped, which is the number #3663 asks for.

Usage::

    python name_misspellings.py --out cands.json
    python name_evidence.py --candidates cands.json --out evidence_misspellings.json
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
    print(f"[misspell] {msg}", flush=True)


def edit_distance(a: str, b: str, cap: int = 3) -> int:
    """Levenshtein distance, stopped once it exceeds *cap*.

    Small and local rather than a dependency: the strings are VG names, the cap
    makes the early exit cheap, and the alternative is adding a package to a
    script that runs once.
    """
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        if min(cur) > cap:
            return cap + 1
        prev = cur
    return prev[-1]


def near_misses(
    vocabulary: dict[str, int],
    targets: dict[str, set[str]],
    shipped: set[str],
    max_edits: int,
    min_len: int,
    min_images: int,
) -> dict[str, list[str]]:
    """``{class: [candidate names]}`` -- VG names close to a target spelling.

    *targets* is the spellings to be close **to**: the class name and whatever
    the shipped tables already declare for it, so `umberella` is reachable from
    `umbrella` and `bicyle` from `bicycle` without either being listed by hand.
    """
    out: dict[str, list[str]] = {}
    for cls, spellings in targets.items():
        hits: set[str] = set()
        for name, imgs in vocabulary.items():
            if imgs < min_images or name in shipped or name in spellings:
                continue
            for target in spellings:
                if len(target) < min_len:
                    continue
                if edit_distance(name, target, max_edits) <= max_edits:
                    hits.add(name)
                    break
        out[cls] = sorted(hits)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--anchor-dir", default=str(pc.PILE / "coco_anchor"))
    ap.add_argument("--max-edits", type=int, default=2)
    ap.add_argument("--min-len", type=int, default=5, help="ignore targets shorter than this (an edit is too cheap)")
    ap.add_argument("--min-images", type=int, default=3)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    log("loading VG image_data.json")
    with (Path(args.anchor_dir) / "image_data.json").open() as fh:
        on_coco = {int(m["image_id"]) for m in json.load(fh) if m.get("coco_id")}

    log(f"loading VG objects.json ({(VG_ROOT / 'objects.json').stat().st_size / 1e6:.0f} MB)")
    with (VG_ROOT / "objects.json").open() as fh:
        records = json.load(fh)

    n_images: dict[str, int] = defaultdict(int)
    n_off: dict[str, int] = defaultdict(int)
    for rec in records:
        iid = int(rec["image_id"])
        seen = set()
        for obj in rec.get("objects") or []:
            names = obj.get("names") or []
            if names and str(names[0]).strip().lower():
                seen.add(str(names[0]).strip().lower())
        for name in seen:
            n_images[name] += 1
            if iid not in on_coco:
                n_off[name] += 1
    log(f"  {len(n_images)} distinct primary names")

    shipped = {n for v in pc.SCALE_VG_NAMES.values() for n in v}
    for v in pc.SCALE_VG_AMBIGUOUS.values():
        shipped |= set(v)
    targets = {c: {c, *pc.SCALE_VG_NAMES.get(c, ())} for c in pc.SCALE_CLASSES}

    cands = near_misses(n_images, targets, shipped, args.max_edits, args.min_len, args.min_images)

    print("\n" + "=" * 78)
    print("NEAR-MISS SPELLINGS -- names to measure, not names to fold")
    print(f"within {args.max_edits} edits of a class name or a shipped alias, >= {args.min_images} images")
    print("=" * 78)
    total = off_total = 0
    for cls, names in cands.items():
        if not names:
            continue
        imgs = sum(n_images[n] for n in names)
        off = sum(n_off[n] for n in names)
        total += imgs
        off_total += off
        print(f"\n{cls}  ({len(names)} candidates, {imgs} images, {off} off-COCO)")
        for n in sorted(names, key=lambda n: -n_images[n])[:12]:
            print(f"    {n:<28}{n_images[n]:>7} imgs{n_off[n]:>8} off-COCO")
    print(f"\n{sum(len(v) for v in cands.values())} candidates over {total} images, {off_total} of them off-COCO")
    print("That off-COCO count is the ceiling on what this residue can be worth: on the")
    print("COCO half the labels are replaced wholesale, so a missed spelling costs nothing there.")

    if args.out:
        Path(args.out).write_text(json.dumps(cands, indent=1) + "\n")
        log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
