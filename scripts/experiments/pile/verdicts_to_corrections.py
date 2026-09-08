"""Turn review verdicts into the corrections file the builder merges before banding.

Three sources feed one file, and they are *not* interchangeable:

* **human verdicts** (`ingest_slate.py`) -- the reviewer's Good/Bad on slate
  images, carrying a drawn box when they redrew one;
* **adjudications** -- a second opinion on the pairs where the reviewer and the
  reference disagree, recorded with a note so the reasoning survives;
* **triage flags** -- a model pass over the ranked negatives, which finds
  contaminated negatives efficiently but draws no boxes.

**Boxes are written NORMALISED, and say so.** A drawn box arrives as the app's
`region_box`, which is already in [0, 1], while VG's and COCO's are in pixels.
The builder normalises every box it merges, so an undeclared correction box was
normalised twice and landed on the frame origin -- 130 of them, taking their
band with them, invisible because the band is derived from the same box
(#3281). Each row therefore carries `box_space`, and `build_pile.py` refuses a
row whose boxes do not match what it declares.

**A correction without a box excludes rather than promotes.** "There is a bus in
this image" fixes a poisoned negative, but it cannot make the image a *positive*
for any band, because a band is a claim about size and no size was measured. The
builder therefore drops it from every cell of that class: not a positive, and no
longer a negative either. That is the whole point of the three-valued design --
the alternative is inventing a box to keep the arithmetic tidy.

**A `present` the class cannot hold is refused, and refusing it is not the same
as ignoring it.** A boxless `present` on a negative excludes the image; a boxed
one turns it into a positive. Both are wrong when the object the reviewer
correctly saw is one the class's own construction would never have admitted --
a wristwatch for `clock`, a pop-up canopy for `umbrella` -- because the class
then loses a good negative, or gains a positive it does not believe in, on the
strength of a reading it does not use. #3666 adjudicated nine such finds and
**four** were exactly this. The gate is the same adjudication file the positive
side already reads, with the same two fields (``"claude": "absent"`` plus
``"reason": "definition"``), because it is the same sentence pointed the other
way: *what the object is* settles it, and no amount of looking changes the
answer. It is a table of decided cases and never a heuristic -- a verdict
carries no object identity, so nothing here can infer one.

**A rebox can move an image between bands, and that is reported, not refused.**
An image entered a `class@band` cell because of the box it arrived with, so a
reviewer who retargets the box to a different instance of the same class moves
the image to the band the new box implies and vacates the cell it was sampled to
fill -- 6 of the first 13 redrawn boxes did (#3616). The move is a *correction*,
not a defect: VG is not exhaustive, so an image holding a small annotated bowl
and a large unannotated one was mis-banded from the start, and the reviewer is
the first person to have seen it. What was wrong is that it happened silently,
so the run now prints every band-changing rebox. `audit_band_drift.py` measures
how much of the same error the un-reviewed half is still hiding.

**A rejection is not a deletion in the small band -- unless it is definitional.**
Boxed review confirms only ~2/3 of sub-patch positives even when the box is drawn
for the reviewer, and the same objects defeat the model, so "not confirmed" there
is recorded as exactly that and the label stands. Above one patch a rejection
backed by adjudication does remove the positive.

That guard reads a small-band rejection as *"I cannot tell at this size"*, which
is the right default and the wrong one when the adjudicator has named what the
object actually is. Three of the ten ``bicycle@small`` positives are bicycle
pictograms on road signs; they are not bicycles at any resolution, and the guard
made them uncorrectable by a human rejection and an adjudicated one alike
(#3614). An adjudication may therefore carry ``"reason": "definition"``
alongside ``"claude": "absent"``, which removes the positive regardless of band.
Use it only where the identity of the object is settled -- never to force through
a rejection that is really about confirmability, which is what the guard is for.

Usage::

    python verdicts_to_corrections.py --verdicts verdicts.json --triage tri_flags_all.json \
        --adjudication adjudication_ml.json --out corrections.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import pile_config as pc
from pilebuild.corrections import dropped_rows, write_json_locked

pc.setup_env()


def log(msg: str) -> None:
    print(f"[corrections] {msg}", flush=True)


def _cell_band(cell: str) -> str:
    return cell.rsplit("@", 1)[1] if "@" in cell else ""


def _box_band(box: list[float]) -> str:
    """The band a NORMALISED correction box falls in, or ``""`` outside them all.

    A band is a fraction of the frame and a normalised box's area already *is*
    that fraction, so this needs no image dimensions -- which is the whole reason
    the drift can be reported here rather than only at build time. The builder's
    banding (``pilebuild.loaders.vg_scale.band_for``) additionally rejects a
    scattered *union* of several boxes; a correction carries exactly one box, so
    there is no scatter to reject and the two agree by construction.
    """
    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    for band, (lo, hi) in pc.BOX_BANDS.items():
        if lo <= area < hi:
            return band
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    base = pc.PILE.parent / "vgscale-3156"
    ap.add_argument(
        "--verdicts",
        default=f"{base / 'verdicts_20260820b.json'},/exp/sgreenberg/vgscale-3156-labelsets/verdicts_audit_20260825.json",
        help="verdict files, comma-separated; later files win",
    )
    ap.add_argument("--triage", default=str(base / "tri_flags_all.json"))
    ap.add_argument("--adjudication", default=str(base / "adjudication_ml_20260820.json"))
    ap.add_argument("--sheets", default=str(base / "sheets_neg"))
    ap.add_argument("--slates", default=f"{base / 'slates'},{base / 'slates_pos2'}")
    ap.add_argument("--include-maybes", action="store_true", help="apply triage maybes too (default: no)")
    ap.add_argument("--out", default=str(pc.PILE / "corrections.json"))
    ap.add_argument(
        "--allow-loss",
        action="store_true",
        help="write even when the result drops rows the existing file has (it is human work; see --help)",
    )
    args = ap.parse_args()

    # (image, class) -> manifest row, for the band of a reviewed positive.
    cells: dict[tuple[int, str], str] = {}
    for root in args.slates.split(","):
        for man in sorted(Path(root).glob("*/manifest.csv")):
            for r in csv.DictReader(man.open()):
                if r.get("cell"):
                    cells[(int(r["image_id"]), r["class"])] = r["cell"]

    out: dict[tuple[int, str], dict] = {}
    stats: Counter = Counter()
    # (class, image_id, band sampled, band the redrawn box implies) -- #3616.
    moves: list[tuple[str, int, str, str]] = []
    # (class, image_id, carried a box, the adjudicator's note) -- #3676.
    kept: list[tuple[str, int, bool, str]] = []

    # --- adjudications first, so a later source cannot silently overrule them
    adj = {}
    if Path(args.adjudication).exists():
        for a in json.loads(Path(args.adjudication).read_text()):
            adj[(int(a["image_id"]), a["class"])] = a

    # --- human verdicts
    # `ruled` records every pair a human decided, in EITHER direction. A triage
    # flag must never overrule one: the audit measured the flags at 0.44
    # precision, so applying an unaudited flag over a human "absent" would
    # inject more error than it removes.
    ruled: set[tuple[int, str]] = set()
    verdicts = []
    for path in args.verdicts.split(","):
        if Path(path).exists():
            verdicts += json.loads(Path(path).read_text())
    for v in verdicts:
        key = (int(v["image_id"]), v["class"])
        ruled.add(key)
        band = _cell_band(cells.get(key, ""))
        # Any stratum that reviews a current NEGATIVE folds in here. Listing
        # them explicitly means a new stratum is ignored rather than
        # mishandled -- safe, but silent, so the list has to be updated when one
        # is added. `redef*` came from re-reviewing a class whose definition
        # changed (`make_definition_reslate.py`).
        if v["stratum"] in ("boundary", "random", "flag", "audit", "redef", "redef_fresh"):
            if v["human"] == "present":
                # Refused only where an adjudication names the object and says
                # the class cannot hold it -- see the module docstring. The
                # reviewer is not being overruled about the pixels; the class is
                # being held to its own vocabulary.
                a = adj.get(key)
                if a and a.get("claude") == "absent" and a.get("reason") == "definition":
                    stats["negative_kept_definitional"] += 1
                    kept.append((key[1], key[0], bool(v.get("box")), a.get("note", "")))
                    continue
                box = v.get("box")
                out[key] = {
                    "image_id": key[0],
                    "class": key[1],
                    "present": True,
                    "boxes": [box] if box else [],
                    "box_space": pc.CORRECTION_BOX_SPACE,
                    "source": "human_review",
                }
                stats["negative_fixed" if box else "negative_excluded"] += 1
            continue
        if v["stratum"] == "positive_boxed":
            if v["human"] == "present":
                box = v.get("box")
                if box:  # reviewer redrew it: the box, hence the band, changes
                    out[key] = {
                        "image_id": key[0],
                        "class": key[1],
                        "present": True,
                        "boxes": [box],
                        "box_space": pc.CORRECTION_BOX_SPACE,
                        "source": "human_rebox",
                    }
                    stats["positive_reboxed"] += 1
                    # The redrawn box decides the band, so a rebox can move the
                    # image out of the cell it was sampled to fill. The move is
                    # kept -- see the module docstring -- but it is named here,
                    # because a cell that quietly rebalances is how the small
                    # band erodes without anyone deciding that it should.
                    new_band = _box_band(box)
                    if not band:
                        stats["positive_reboxed_UNSAMPLED"] += 1
                    elif new_band == band:
                        stats["positive_reboxed_band_kept"] += 1
                    else:
                        stats["positive_reboxed_band_moved"] += 1
                        moves.append((key[1], key[0], band, new_band or "oversize"))
                else:
                    # A confirmation changes no label, and until #3727 it wrote
                    # no row either -- so a pair a human had agreed with was
                    # indistinguishable from one nobody had opened, and
                    # `designate_cells` (which reads `corrections` to decide who
                    # keeps a seat) gave it no priority. It is written as a
                    # verdict with no boxes and its own source, never as a
                    # boxless `present`: see `pile_config`'s constant.
                    out[key] = {
                        "image_id": key[0],
                        "class": key[1],
                        "present": True,
                        "boxes": [],
                        "box_space": pc.CORRECTION_BOX_SPACE,
                        "source": pc.CORRECTION_SOURCE_CONFIRMED,
                    }
                    stats["positive_confirmed"] += 1
                continue
            # Rejected. Small band: not confirmed is not absent.
            a = adj.get(key)
            # ...unless the rejection is DEFINITIONAL. The band guard exists
            # because a small object is hard to confirm, so a rejection there is
            # ambiguous between "absent" and "I cannot tell at 26 px". That
            # ambiguity does not arise when the adjudicator names *what the
            # object is*: a bicycle pictogram on a road sign is not a bicycle at
            # any size, and no amount of resolution would change the answer.
            # Without this branch such a positive is uncorrectable -- the guard
            # swallows the human rejection and the adjudicated one alike (#3614).
            if a and a.get("claude") == "absent" and a.get("reason") == "definition":
                out[key] = {
                    "image_id": key[0],
                    "class": key[1],
                    "present": False,
                    "boxes": [],
                    "source": "human_reject+adjudicated_definition",
                    "note": a.get("note", ""),
                }
                stats["positive_removed_definitional"] += 1
            elif band == "small":
                stats["small_unconfirmed"] += 1
            elif a and a["claude"] == "absent":
                out[key] = {
                    "image_id": key[0],
                    "class": key[1],
                    "present": False,
                    "boxes": [],
                    "source": "human_reject+adjudicated",
                    "note": a.get("note", ""),
                }
                stats["positive_removed"] += 1
            elif a and a["claude"] == "present":
                stats["rejection_overturned"] += 1
            else:
                stats["rejection_unadjudicated"] += 1
            continue
        stats[f"IGNORED_unknown_stratum:{v['stratum']}"] += 1

    # --- triage flags: contaminated negatives, no boxes, so they exclude
    if Path(args.triage).exists():
        flags = json.loads(Path(args.triage).read_text())
        for cls, kinds in flags.items():
            idx_path = Path(args.sheets) / cls.replace(" ", "_") / "index.json"
            if not idx_path.exists():
                log(f"  no sheet index for {cls}; skipping its flags")
                continue
            idx = {(r["sheet"], r["tile"]): r["image_id"] for r in json.loads(idx_path.read_text())}
            wanted = list(kinds["definite"]) + (list(kinds["maybe"]) if args.include_maybes else [])
            for sheet, tile in wanted:
                iid = idx.get((sheet, tile))
                if iid is None:
                    continue
                key = (iid, cls)
                if key in ruled:  # a human ruled on this pair, either way
                    stats["triage_deferred_to_human"] += 1
                    continue
                out[key] = {
                    "image_id": iid,
                    "class": cls,
                    "present": True,
                    "boxes": [],
                    "source": "claude_triage",
                }
                stats["negative_excluded_by_triage"] += 1

    rows = sorted(out.values(), key=lambda r: (r["class"], r["image_id"]))
    # Refuse to shrink the file. See `dropped_rows`: the inputs that produced
    # what is on disk are not the ones this script defaults to, so a well-meant
    # re-run is a deletion of rows no one can regenerate.
    out_path = Path(args.out)
    if out_path.exists():
        lost = dropped_rows(json.loads(out_path.read_text()), rows)
        if lost and not args.allow_loss:
            print(f"\nREFUSING to write {out_path}: it holds {sum(lost.values())} rows this run does not produce")
            for source, n in sorted(lost.items(), key=lambda kv: -kv[1]):
                print(f"    {n:>4}  {source}")
            print("  Those came from inputs this invocation was not given. Add them with --verdicts /")
            print("  --adjudication / --slates / --triage, or pass --allow-loss if the drop is intended.")
            return 2
    write_json_locked(out_path, rows)

    print(f"\n{len(rows)} corrections written to {args.out}\n")
    for k, v in sorted(stats.items()):
        print(f"   {k:<32}{v:>6}")
    boxed = sum(1 for r in rows if r["boxes"])
    print(f"\n   {'of which carry a box':<32}{boxed:>6}  (can move an image between bands)")
    print(f"   {'excluded, no box':<32}{len(rows) - boxed:>6}  (dropped from every cell of that class)")
    _report_moves(moves, stats)
    _report_kept(kept)
    return 0


def _report_kept(kept: list[tuple[str, int, bool, str]]) -> None:
    """Name every correction refused because the class cannot hold the object (#3676).

    Printed rather than silent for the same reason the band moves are: a
    correction that does not happen is invisible in the output file, and this
    one is a *decision* about what the class means. The line also says whether
    the class has a written rule at all -- an unruled class here is a ruling
    somebody owes (#3673), not a defect in this script.
    """
    if not kept:
        return
    print(f"\n=== {len(kept)} `present` verdicts REFUSED: the class cannot hold the object ===")
    print("   The reviewer saw what they say they saw. The class's own names do not")
    print("   admit it, so applying the correction would spend a good negative (no box)")
    print("   or manufacture a positive (box) on a reading the build does not use.\n")
    print("   %-12s %-10s %-6s %s" % ("class", "image_id", "boxed", "why"))
    for cls, iid, boxed, note in sorted(kept):
        print("   %-12s %-10d %-6s %s" % (cls, iid, "yes" if boxed else "no", note or "(no note)"))
    unruled = sorted({c for c, _, _, _ in kept if c not in pc.SCALE_CLASS_RULES})
    if unruled:
        print("\n   NO WRITTEN RULE for: " + ", ".join(unruled))
        print("   Each of those is a sentence owed to `SCALE_CLASS_RULES` (#3673); until it")
        print("   exists the next reviewer re-derives the same call from scratch.")


def _report_moves(moves: list[tuple[str, int, str, str]], stats: Counter) -> None:
    """Name every rebox that left the cell it was sampled into (#3616)."""
    if not moves:
        return
    reboxed = stats["positive_reboxed_band_moved"] + stats["positive_reboxed_band_kept"]
    print(f"\n=== {len(moves)} of {reboxed} redrawn boxes LEAVE the cell they were sampled into ===")
    print("   Kept, not refused: VG is not exhaustive, so an image holding an unannotated")
    print("   larger instance was mis-banded before the reviewer ever saw it. Printed")
    print("   because it rebalances the cells, and the small band is the binding")
    print("   constraint on supply (#3603). `audit_band_drift.py` measures the same")
    print("   error on the images nobody has reviewed.\n")
    print("   %-14s %-10s %-8s    %s" % ("class", "image_id", "sampled", "redrawn"))
    for cls, iid, was, now in sorted(moves):
        print("   %-14s %-10d %-8s -> %s" % (cls, iid, was, now))
    per_class: Counter = Counter(m[0] for m in moves)
    vacated: Counter = Counter(m[2] for m in moves)
    print("\n   by class:      " + ", ".join(f"{c}={n}" for c, n in sorted(per_class.items())))
    print("   band vacated:  " + ", ".join(f"{b}={n}" for b, n in sorted(vacated.items())))


if __name__ == "__main__":
    raise SystemExit(main())
