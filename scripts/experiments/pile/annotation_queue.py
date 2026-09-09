#!/usr/bin/env python3
"""The worklist for `vg_scale`'s exhaustive annotation pass (#3720, #3668).

`docs/plans/vg-scale-exhaustive-annotation.md` moves `vg_scale` from a
designation to an exhaustively annotated set, and its debt is one stratum: the
**off-COCO positives**. Every class in *C* is a COCO-2017 class and since #3670
the negative pool and the spares are drawn from the COCO-anchored half alone, so
COCO already answers for everything except the positives VG contributed on the
half COCO never annotated.

A reviewer cannot start on that stratum without being handed it. **Whether COCO
also annotated an image is not visible in the image** -- it is a join between VG
and COCO ids -- so the worklist has to be emitted rather than eyeballed. That is
all this does.

**One row per image, not per cell.** The pass asks "which of *C* is present?"
once per image, so an image designated in three cells is one row carrying three
cell names, not three judgements. Counting cells instead of images would
over-state the pass by the multiply-designated share.

**Two ways to answer "did COCO score this image?", and they are not
interchangeable.**

* The cell's own ``coco_scored`` stamp, which the build writes from the
  post-`anchor_to_coco` `exhaustive` set. This is the same fact #3670's pool
  composition is defined on, and it is what the build actually acted on.
* The **fallback**, for a cell built before the stamp: `coco_anchor`'s
  ``image_data.json``, joined on its ``coco_id`` field. That pairing is a slight
  *superset* of what the build anchors, because it does not re-apply the
  aspect-drift filter that drops 49 of 51,497 pairs -- so it can call an image
  anchored that the build did not, which **drops a row that needs annotating**.
  That is the unsafe direction here, and the exact opposite of how
  `check_review_coverage.eligible_under` uses the same join, where a superset
  inflates a denominator and understates coverage. Measured against the stamp on
  the shipped 25-class cell: the pairing drops 3 of the 3,391 owed images and
  adds none -- small, and one-sided in the direction that hides work. So every
  row says which source answered for it, and the summary says how much of the
  queue rests on the fallback: read a queue that is mostly ``pairing`` as a
  lower bound, and rebuild the cell rather than living with it.

**A missing stamp is not a `False` stamp.** ``coco_scored`` absent means no
answer; ``coco_scored`` false means COCO was asked and did not annotate this
image. The two are only distinguishable by testing for the *key*, which is why
this checks membership rather than truthiness -- the deep sibling silently
inherited #3667's cross-class rule with an empty world by making exactly this
mistake one level down (`vg_scale._emit_medias`).

**The order is a seeded shuffle, and that is the pilot's doing.** The plan reads
the first batch of the real queue as the pilot rather than annotating throwaway
images first, and what it reads off that batch is per-class recall. That is only
meaningful if the first N rows are a sample of the queue rather than of whatever
the id order happens to group together, so the queue is shuffled once under a
fixed seed: reproducible, and unbiased in class, band and source.

**The queue gates on the roster, because a worklist for a set that can move is
worse than no worklist.** `designate_cells` fills a cell from the roster first
and backfills the rest by `rank`, so an image that is *not* pinned can be
displaced by a rebuild that changed something else entirely -- 41 positives out
and 40 in with nothing relevant altered (#3667). Annotating an unpinned image is
therefore work with no guarantee of a home, which is why this exits non-zero and
names `make_roster.py` rather than printing a warning nobody reads. The queue
file is still written: the failure is "do not start yet", not "nothing to see".

**What the queue cannot carry is an answer key.** Every row is off-COCO by
construction, so no row in it can be scored against COCO. That is not a gap in
the worklist -- it is why the review tooling mixes a minority of anchored images
into a slate rather than into the worklist (`make_audit_slate.py`: "the reviewer
cannot tell them apart: files are named by image id alone"). Scoring the
annotators is the plan's own item and it acts at slate-build time, on top of
this file.

Usage::

    python annotation_queue.py                       # the shipped siglip cell
    python annotation_queue.py --cell <path.pkl> --out queue.jsonl
    python annotation_queue.py --summary summary.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pile_config as pc

#: Fixed so two people emitting the queue get the same first batch, and so a
#: re-emission after a rebuild can be diffed against the old one row by row.
QUEUE_SEED = 3720


def log(msg: str) -> None:
    print(f"[queue] {msg}", flush=True)


def class_of(cell: str) -> str:
    """``"chair@large"`` -> ``"chair"``; a bare class name passes through.

    ``vg_scale`` keys cells on ``class@band`` and ``vg_scale_deep`` keys them on
    the bare class, and the annotation pass does not care which -- it asks about
    the class. Splitting on the separator reads both without being told which
    dataset it was handed.
    """
    return cell.split("@", 1)[0]


def coco_answer(iid: int, media: dict, paired: set[int] | None) -> tuple[bool, str]:
    """``(did COCO score this image, which source said so)``.

    Raises when the media carries no stamp and no pairing was supplied, because
    the alternative -- treating "no answer" as "not scored" -- would put every
    anchored image in the queue and silently triple the pass.
    """
    if "coco_scored" in media:
        return bool(media["coco_scored"]), "stamp"
    if paired is None:
        raise ValueError(
            f"media {iid} carries no `coco_scored` stamp and no COCO pairing was supplied; "
            "pass the `image_data.json` join (see `paired_image_ids`)"
        )
    return iid in paired, "pairing"


def paired_image_ids(image_data: Path) -> set[int]:
    """VG image ids that COCO also holds, read from `coco_anchor/image_data.json`.

    The same file and the same field `check_review_coverage` and the loader read,
    deliberately: a second way of spelling "this image is in COCO" is a second
    thing to keep in sync.
    """
    if not image_data.exists():
        raise SystemExit(f"missing {image_data}; run `coco_anchor.py --fetch` before emitting a queue for this cell")
    return {int(m["image_id"]) for m in json.loads(image_data.read_text()) if m.get("coco_id")}


def queue_rows(medias: dict[int, dict], paired: set[int] | None = None, seed: int = QUEUE_SEED) -> list[dict]:
    """The off-COCO positives of *medias*, one row per image, shuffled once.

    A positive is an image with a non-empty ``categories`` -- what it was
    *designated* for. That is not what the image holds (the pickle never records
    that, which is #3678's whole finding), but it is the right population here:
    the pass is owed on the images the set contains.
    """
    rows = []
    for iid in sorted(medias):
        media = medias[iid]
        cells = list(media.get("categories") or [])
        if not cells:
            continue
        anchored, source = coco_answer(iid, media, paired)
        if anchored:
            continue
        rows.append(
            {
                "image_id": iid,
                "path": media.get("origin_name") or "",
                "filename": media.get("filename") or "",
                "cells": sorted(cells),
                "classes": sorted({class_of(c) for c in cells}),
                "coco_source": source,
            }
        )
    random.Random(seed).shuffle(rows)
    return rows


def counts(rows: list[dict]) -> tuple[Counter[str], Counter[str], Counter[str]]:
    """``(per class, per cell, per source)`` over the queue.

    Per *class* counts an image once however many of its cells name that class,
    so the columns of a class's row sum to its total only when no image is
    designated in two bands of one class.
    """
    per_class: Counter[str] = Counter()
    per_cell: Counter[str] = Counter()
    per_source: Counter[str] = Counter()
    for row in rows:
        per_class.update(row["classes"])
        per_cell.update(row["cells"])
        per_source[row["coco_source"]] += 1
    return per_class, per_cell, per_source


def band_columns(rows: list[dict], bands: tuple[str, ...]) -> tuple[str, ...]:
    """The bands actually designated in *rows*, in the config's order.

    `vg_scale_deep` keys its cells on the bare class, so its queue has no bands
    at all -- and a table printing three columns of zeros for it would read as a
    class list with nothing in it rather than as a dataset that is not banded.
    """
    _, per_cell, _ = counts(rows)
    seen = {cell.split("@", 1)[1] for cell in per_cell if "@" in cell}
    return tuple(b for b in bands if b in seen)


def roster_gaps(rows: list[dict], roster: dict) -> tuple[int, int, list[str]]:
    """``(pinned, unpinned, examples)`` over the queue's ``(image, cell)`` designations.

    Counted per designation rather than per image: an image pinned in two of its
    three cells is two-thirds safe, and reporting it as "pinned" would hide the
    cell it can still be displaced from.
    """
    cells = {c: {int(i) for i in ids} for c, ids in (roster.get("cells") or {}).items()}
    pinned = unpinned = 0
    missing: list[str] = []
    for row in rows:
        for cell in row["cells"]:
            if row["image_id"] in cells.get(cell, frozenset()):
                pinned += 1
            else:
                unpinned += 1
                if len(missing) < 5:
                    missing.append(f"{row['image_id']} in {cell}")
    return pinned, unpinned, missing


def print_table(rows: list[dict], bands: tuple[str, ...]) -> None:
    """Per-class counts, banded -- the numbers the plan says arrive for free."""
    per_class, per_cell, _ = counts(rows)
    cols = band_columns(rows, bands)
    by_class_band: dict[str, dict[str, int]] = defaultdict(dict)
    for cell, n in per_cell.items():
        if "@" in cell:
            by_class_band[class_of(cell)][cell.split("@", 1)[1]] = n
    width = 14 + 10 * (len(cols) + 1)
    print(f"\n{'class':<14}" + "".join(f"{b:>10}" for b in cols) + f"{'images':>10}")
    print("-" * width)
    for cls, total in sorted(per_class.items(), key=lambda kv: (-kv[1], kv[0])):
        cells = by_class_band.get(cls, {})
        print(f"{cls:<14}" + "".join(f"{cells.get(b, 0):>10}" for b in cols) + f"{total:>10}")
    print("-" * width)
    print(f"{'images owed':<14}" + " " * (10 * len(cols)) + f"{len(rows):>10}")


def main() -> int:
    # Deferred, like `check_review_coverage`'s: `setup_env` rewrites the import
    # machinery process-wide, and the selection above is worth a unit test that
    # does not pay for that.
    pc.setup_env()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    base = pc.PILE.parent / "vgscale-3156"
    ap.add_argument(
        "--cell",
        default=str(pc.EMBEDDINGS / "vg_scale__siglip.pkl"),
        help="any embedder's cell: membership is the same in all of them, only the vectors differ",
    )
    ap.add_argument("--out", default=str(base / "annotation_queue.jsonl"), help="the worklist, one JSON object a line")
    ap.add_argument("--summary", default=str(base / "annotation_queue.summary.json"))
    ap.add_argument("--image-data", default=str(pc.PILE / "coco_anchor" / "image_data.json"))
    ap.add_argument("--seed", type=int, default=QUEUE_SEED)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "calibration"))
    from _cells_io import load_medias  # noqa: PLC0415

    cell = Path(args.cell)
    medias = load_medias(cell)
    positives = {i: m for i, m in medias.items() if m.get("categories")}
    log(f"{cell.name}: {len(medias)} medias, {len(positives)} positives")

    # Read the ~100 MB join only when the cell cannot answer for itself.
    unstamped = sum(1 for m in positives.values() if "coco_scored" not in m)
    paired = None
    if unstamped:
        log(f"{unstamped} positives carry no `coco_scored` stamp; falling back to the COCO pairing")
        paired = paired_image_ids(Path(args.image_data))
        log(f"{len(paired)} VG images carry a coco_id")

    rows = queue_rows(medias, paired, seed=args.seed)
    per_class, per_cell, per_source = counts(rows)
    anchored = len(positives) - len(rows)

    print_table(rows, tuple(pc.BOX_BANDS))
    print()
    if not positives:
        log("NOTE: this cell designates no positives at all -- the wrong cell, or a build that produced none.")
        return 1
    log(
        f"{len(rows)} of {len(positives)} positives are off-COCO "
        f"({len(rows) / len(positives):.2%}); {anchored} are answered by COCO already"
    )
    log("source: " + ", ".join(f"{n} by {s}" for s, n in sorted(per_source.items())))
    if per_source.get("pairing"):
        # Loud, and next to the number it qualifies: the pairing over-counts
        # anchored images, so a queue resting on it is a lower bound on the pass.
        log(
            f"NOTE: {per_source['pairing']} rows were classified by the COCO pairing, which is a superset "
            "of what the build anchors -- those rows under-count the queue. Rebuild the cell to remove the doubt."
        )
    off_roster = sorted(set(per_class) - set(pc.SCALE_CLASSES))
    if off_roster:
        # The cell is the authority on what it holds and the config is the
        # authority on what C is; when they disagree the queue is the *build's*
        # debt, and saying so is cheaper than discovering it at review time
        # (#3678: nothing else tells you a cell predates a ruling).
        log(
            f"NOTE: {len(off_roster)} class(es) here are not in the config's roster ({', '.join(off_roster)}) "
            "-- this cell was built against a different class list."
        )
    if not rows:
        log("NOTE: no off-COCO positives -- every positive in this cell is already answered by COCO.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    summary = {
        "cell": str(cell),
        "seed": args.seed,
        "roster": list(pc.SCALE_CLASSES),
        "bands": list(pc.BOX_BANDS),
        "n_medias": len(medias),
        "n_positives": len(positives),
        "n_positives_coco_scored": anchored,
        "n_queue": len(rows),
        "by_source": dict(per_source),
        "per_class": dict(per_class),
        "per_cell": dict(per_cell),
    }
    # #3727: a worklist is only worth annotating if the set it names is pinned.
    # Checked after the queue is written, because the file is still useful --
    # the non-zero exit says "do not start yet", not "nothing was produced".
    gate = 0
    if not pc.ROSTER.exists():
        log(f"FAIL: no roster at {pc.ROSTER} -- run `make_roster.py --cell {cell}` before annotating")
        gate = 1
    else:
        pinned, unpinned, examples = roster_gaps(rows, json.loads(pc.ROSTER.read_text()))
        summary["roster"] = str(pc.ROSTER)
        summary["roster_pinned"] = pinned
        summary["roster_unpinned"] = unpinned
        log(f"roster pins {pinned} of {pinned + unpinned} of the queue's designations")
        if unpinned:
            log(f"FAIL: {unpinned} designation(s) unpinned, e.g. {', '.join(examples)}")
            log(f"      A rebuild can displace those. Run `make_roster.py --cell {cell}` first.")
            gate = 1

    Path(args.summary).write_text(json.dumps(summary, indent=2) + "\n")
    log(f"wrote {out} and {args.summary}")
    return gate


if __name__ == "__main__":
    raise SystemExit(main())
