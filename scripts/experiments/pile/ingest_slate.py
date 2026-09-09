"""Turn a VTSearch label export back into verdict rows, and score the ground truth.

The other half of ``make_audit_slate.py``. A ``server_folder`` import → vote →
``server_json_file`` export round-trip already emits everything a verdict needs:
the file name identifies the VG image (the slate names files ``<image_id>.jpg``),
``label`` is the human's Good/Bad, and ``region_box`` carries the box drawn on a
Good vote, normalised.

**Verdicts, not corrections.** Every reviewed ``(image, class)`` pair becomes a
row whether or not it disagrees with COCO. Corrections are then derived from the
disagreements, and review *coverage* falls out for free — without it, "no bus
here" is indistinguishable from "nobody looked", and every rate computed
afterwards is biased by an unknown amount.

**The rate comes from the random stratum alone.** The boundary stratum is chosen
to find errors, so its error rate is not the pool's error rate and averaging the
two together produces a number that means nothing. Both are reported, labelled,
and never pooled.

Usage::

    python ingest_slate.py --export ~/exports/bus.json --slates /expscratch/$USER/vgscale-3156/slates
    python ingest_slate.py --export 'exports/*.json' --slates ... --out verdicts.json
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

import pile_config as pc
from pilebuild.corrections import write_json_locked

pc.setup_env()


def log(msg: str) -> None:
    print(f"[ingest] {msg}", flush=True)


#: Which stratum wins when the same (image, class) was reviewed twice. The
#: boxed re-issue supersedes the bare thumbnail: it is the same question asked
#: in a form the reviewer could actually answer (`make_positive_slate.py`).
STRATUM_RANK = {"positive": 0, "boundary": 0, "random": 0, "positive_boxed": 1}


def load_manifests(roots: list[Path]) -> tuple[dict[tuple[int, str, str], dict], dict[str, str]]:
    """``({(image_id, class, detector): row}, {detector: folder name})``.

    Keyed by detector as well as class because a class can be reviewed through
    more than one detector -- the bare slate and the boxed positive re-issue --
    and their rows must not overwrite each other before the ranking above is
    applied.
    """
    out: dict[tuple[int, str, str], dict] = {}
    folders: dict[str, str] = {}
    for root in roots:
        for man in sorted(root.glob("*/manifest.csv")):
            with man.open() as fh:
                for row in csv.DictReader(fh):
                    det = row.get("detector") or row["class"]
                    out[(int(row["image_id"]), row["class"], det)] = row
                    folders[det] = man.parent.name
    return out, folders


def class_of(path: Path, elements: list[dict], folders: dict[str, str], explicit: str) -> str:
    """Which class this export is a review of.

    An export **cannot** be attributed by image id: the slates share images
    (801 of 3,600 rows are an image that appears under a second class), so a
    Good vote in the `bus` dataset would otherwise be recorded as a `dog`
    verdict too. The slate folder is what disambiguates, read from the
    importer's own origin, then from the file name, and never guessed.
    """
    if explicit:
        if explicit not in folders:
            raise SystemExit(f"--class {explicit!r} is not one of {sorted(folders)}")
        return explicit
    blob = " ".join(json.dumps(el.get("origin") or "") + " " + str(el.get("origin_name") or "") for el in elements[:50])
    hits = {c for c, folder in folders.items() if f"/{folder}/" in blob or f"/{folder}" in blob}
    if len(hits) == 1:
        return hits.pop()
    stem = path.stem.lower()
    hits = {c for c, folder in folders.items() if folder.lower() in stem}
    if len(hits) == 1:
        return hits.pop()
    raise SystemExit(
        f"{path.name}: cannot tell which class this export reviews "
        f"(origin paths and file name match {sorted(hits) or 'nothing'}). Pass --class."
    )


def read_api(base: str, detector: str) -> list[dict]:
    """One detector's saved votes, in the same shape as a file export.

    ``GET /api/detectors/<name>/labels-detail`` already returns the two things a
    verdict needs -- the file name (which is the VG image id) and the
    ``region_box`` drawn on a Good vote -- so pulling straight from the running
    app skips the export dialog entirely. The reviewer votes; nothing else is
    asked of them.
    """
    url = f"{base.rstrip('/')}/api/detectors/{urllib.parse.quote(detector)}/labels-detail"
    try:
        with urllib.request.urlopen(url, timeout=60) as r:  # noqa: S310 - caller-supplied http(s) base
            data = json.load(r)
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
        # A slate whose detector has been retired: reviewing it moved elsewhere
        # (a Claude triage pass, say) and its verdicts live in a snapshot. That
        # is a normal state, not a failure -- but it is reported, because a
        # silently empty detector and a finished one look identical downstream.
        log(f"  {detector}: no such detector (retired); contributing no verdicts")
        return []
    out = []
    for label in ("good", "bad"):
        for el in data.get(label) or []:
            out.append(
                {
                    "filename": el.get("filename") or el.get("origin_name"),
                    "label": label,
                    "region_box": el.get("region_box"),
                }
            )
    return out


def read_export(path: Path) -> list[dict]:
    """The labelled elements of one export, whichever shape it was written in."""
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "labels" in data:
        return list(data["labels"])
    if isinstance(data, list):
        return list(data)
    raise SystemExit(f"{path}: not a label export (no 'labels' key, not a list)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", default="", help="exported JSON (glob allowed)")
    ap.add_argument(
        "--api",
        default="",
        help="pull votes straight from a running VTSearch instead of a file export, "
        "e.g. --api http://rack7n03:11850 (one detector per class, named after it)",
    )
    ap.add_argument(
        "--slates",
        default=str(pc.PILE / "slates"),
        help="slate dir(s) from make_audit_slate.py / make_positive_slate.py, comma-separated",
    )
    ap.add_argument("--out", default="", help="write the verdict rows here")
    ap.add_argument("--class", dest="klass", default="", help="the class this export reviews (else inferred)")
    args = ap.parse_args()

    manifests, folders = load_manifests([Path(p) for p in args.slates.split(",") if p])
    if not manifests:
        raise SystemExit(f"no manifests under {args.slates}; run make_audit_slate.py first")
    log(f"{len(manifests)} slate entries over {len({c for _, c, _ in manifests})} classes, {len(folders)} detectors")

    if not args.export and not args.api:
        raise SystemExit("pass --export <file/glob> or --api <base-url>")

    # (label of the source, its elements, the class it reviews)
    sources: list[tuple[str, list[dict], str]] = []
    if args.api:
        det_class = {det: c for (_, c, det) in manifests}
        for det in sorted(folders):
            c = det_class[det]
            if args.klass and c != args.klass:
                continue
            els = read_api(args.api, det)
            sources.append((f"api:{det}", els, c, det))
    for path in sorted(glob.glob(args.export)) if args.export else []:
        els = read_export(Path(path))
        det = class_of(Path(path), els, folders, args.klass)
        sources.append((Path(path).name, els, {c for (_, c, d) in manifests if d == det}.pop(), det))

    verdicts: list[dict] = []
    unmatched = 0
    for source, elements, c, det in sources:
        for el in elements:
            name = Path(el.get("filename") or el.get("origin_name") or "").stem
            if not name.isdigit():
                unmatched += 1
                continue
            iid = int(name)
            row = manifests.get((iid, c, det))
            if row is None:
                unmatched += 1
                continue
            verdicts.append(
                {
                    "image_id": iid,
                    "class": c,
                    "stratum": row["stratum"],
                    "human": "present" if el.get("label") == "good" else "absent",
                    "reference": row["reference"],
                    "exhaustive": row["exhaustive"],
                    "box": el.get("region_box"),
                    "text_score": float(row["text_score"]),
                    "export": source,
                }
            )
        log(f"  {source}: {len(elements)} labelled elements, class {c!r}")
    if unmatched:
        log(f"  WARNING {unmatched} exported elements matched no slate entry")

    # One verdict per (image, class). A labelset can hold the same media twice:
    # observed 56 of 300 images duplicated in one detector, Bad votes as exact
    # copies and Good votes as a boxed entry plus an image-level one -- the app
    # appends a LabeledElement per vote event rather than replacing (#3174).
    # Counting both would double-weight a fifth of the review, so collapse
    # them, preferring the entry that carries a box because that is the one
    # that can place the image in a band.
    merged: dict[tuple[int, str], dict] = {}
    conflicts = 0
    for v in verdicts:
        key = (v["image_id"], v["class"])
        prev = merged.get(key)
        if prev is None:
            merged[key] = v
            continue
        rank_prev = STRATUM_RANK.get(prev["stratum"], 0)
        rank_new = STRATUM_RANK.get(v["stratum"], 0)
        if rank_new > rank_prev:
            merged[key] = v  # the boxed re-issue supersedes; not a conflict
            continue
        if rank_new < rank_prev:
            continue
        if prev["human"] != v["human"]:
            conflicts += 1
        if prev.get("box") is None and v.get("box") is not None:
            merged[key] = v
    dropped = len(verdicts) - len(merged)
    if dropped:
        log(f"  collapsed {dropped} duplicate labels into {len(merged)} distinct (image, class) verdicts")
    if conflicts:
        log(f"  WARNING {conflicts} images carry BOTH a Good and a Bad vote -- kept the boxed one")
    verdicts = sorted(merged.values(), key=lambda v: (v["class"], v["image_id"]))

    # Per stratum, per direction. Never pooled across strata.
    by: dict[tuple[str, str], int] = defaultdict(int)
    for v in verdicts:
        by[(v["stratum"], f"{v['reference']}->{v['human']}")] += 1

    # The calibration: pairs where COCO has already looked. A disagreement here
    # is NOT "the reviewer was wrong" -- it is reviewer error, COCO error, and
    # definition drift summed together, and nothing in this data separates them.
    # COCO is measurably better than VG (recall 0.61 vs COCO over C) but it is
    # not a gold standard: the same review already found images COCO annotates
    # as empty that plainly hold the object. Report it as disagreement, and
    # leave the decomposition to a third opinion on the disagreeing pairs.
    cal = [v for v in verdicts if v["exhaustive"] == "yes"]
    if cal:
        wrong = sum(1 for v in cal if v["human"] != v["reference"])
        print(
            f"\ncalibration: {len(cal)} pairs COCO has settled, {wrong} DISAGREEMENTS "
            f"({wrong / len(cal):.3f}) -- reviewer error + COCO error + definition drift, "
            f"not attributable without adjudication"
        )

    print(f"\n{len(verdicts)} verdicts\n")
    done: dict[str, int] = defaultdict(int)
    for v in verdicts:
        done[v["class"]] += 1
    total: dict[str, int] = defaultdict(int)
    for _, c, _det in manifests:
        total[c] += 1
    if done:
        print("progress (distinct images voted, duplicates already collapsed):")
        for c in sorted(total):
            n = done.get(c, 0)
            mark = " done" if n >= total[c] else ""
            print(f"  {c:<12}{n:>5} / {total[c]}{mark}")
        print()
    hdr = f"{'stratum':<10}{'agree':>8}{'ref absent, human present':>28}{'ref present, human absent':>28}{'rate':>9}"
    print(hdr)
    print("-" * len(hdr))
    for stratum in ("random", "boundary", "positive", "positive_boxed"):
        agree = by[(stratum, "absent->absent")] + by[(stratum, "present->present")]
        miss = by[(stratum, "absent->present")]
        over = by[(stratum, "present->absent")]
        n = agree + miss + over
        if not n:
            continue
        rate = (miss + over) / n
        note = "" if stratum == "random" else "  (biased by design)"
        print(f"{stratum:<10}{agree:>8}{miss:>28}{over:>28}{rate:>9.3f}{note}")
    print(
        "\nThe residual error rate of the ground truth is the RANDOM row only."
        "\nThe boundary row is chosen to surface errors and says nothing about the pool."
    )

    if args.out:
        write_json_locked(Path(args.out), verdicts)
        log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
