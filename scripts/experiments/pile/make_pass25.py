#!/usr/bin/env python3
"""Build the review set for the exhaustive pass: one dataset, 25 detectors (#3720).

The pass owes an answer for every `(image, class)` pair on the off-COCO
positives — 3,397 images, 25 classes. **It is run as 25 per-class passes, not as
one multi-class naming pass**, on the owner's ruling: holding one class in mind
while scanning is much easier than recognising twenty-five at once, and the app
is already built for exactly that shape. The cost framing that argued otherwise
(~85,000 votes) was wrong, because a per-class pass is mostly bulk in a ranked
review rather than 3,397 deliberate acts.

That ruling makes the recording problem disappear too: a per-class pass records
exactly the pairs it asked about, so "absent" can never be read back for a class
the reviewer was never shown — the hazard #3727 had to guard against under the
compact per-image shape.

**One dataset, twenty-five detectors.** The images are imported once and every
detector scores the same set. Importing per class would embed the same 3,397
images twenty-five times (#3669), and there is nowhere for the class's rule to
live in a shared folder's *dataset* name — but a detector's name is chosen by
the reviewer when they start a pass, so that is where the rule goes. Each
detector is named from `pile_config.SCALE_CLASS_RULES`, which is the wording the
reviewer sees while voting and the only place a definition survives (#3612).

**Controls are mixed in and are not distinguishable.** A fixed number of
COCO-anchored positives per class join the set; COCO answers exhaustively for
them, so every pass scores itself against a key the reviewer cannot see, at no
extra review cost. Files are named by image id alone, exactly as
`make_audit_slate.py` does it. The key is written beside the folder and is
**not** given to the app.

Usage::

    python make_pass25.py                      # build the folder and the key
    python make_pass25.py --api http://host:port --create   # import + 25 detectors
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

import pile_config as pc

#: Known positives per class among the controls. Twelve is enough to see a
#: reviewer missing a quarter of a class, and keeps the controls under a tenth
#: of the set -- the reviewer pays for every control image like any other.
CONTROLS_PER_CLASS = 12

#: Fixed so the control set is the same for whoever rebuilds it, and so a
#: second build can be diffed against the first rather than re-argued.
CONTROL_SEED = 3720


def log(msg: str) -> None:
    print(f"[pass25] {msg}", flush=True)


def draw_controls(
    medias: dict[int, dict],
    held_by: dict[int, list[str]],
    per_class: int = CONTROLS_PER_CLASS,
    seed: int = CONTROL_SEED,
) -> dict[int, list[str]]:
    """``{image_id: the classes COCO says it holds}`` for the control images.

    Drawn from the cell's **COCO-scored** positives, which is what makes them a
    key: COCO answered for all eighty of its classes at once, so a control is a
    known positive for its own class *and* a known negative for the other
    twenty-four. One control therefore scores every pass it appears in, not just
    the one it was drawn for.

    Images already in the queue cannot be drawn: the queue is off-COCO by
    construction and these are anchored, so the two sets are disjoint and this
    is an assertion rather than a filter.
    """
    rng = random.Random(seed)
    by_class: dict[str, list[int]] = {c: [] for c in pc.SCALE_CLASSES}
    for iid, media in sorted(medias.items()):
        if not media.get("coco_scored") or not media.get("categories"):
            continue
        if iid not in held_by:
            continue
        for cell in media["categories"]:
            cls = cell.split("@", 1)[0]
            if cls in by_class:
                by_class[cls].append(iid)
    chosen: dict[int, list[str]] = {}
    for cls in pc.SCALE_CLASSES:
        pool = sorted(set(by_class[cls]) - set(chosen))
        for iid in rng.sample(pool, min(per_class, len(pool))):
            chosen[iid] = list(held_by[iid])
    return chosen


def link_images(paths: dict[int, str], out_dir: Path) -> int:
    """Symlink each image into *out_dir* under its image id, and count them.

    Named by id alone, with nothing in the filename saying whether a row is a
    control -- that is the whole point of the controls, and a reviewer who could
    tell them apart would score differently on them.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.jpg"):
        old.unlink()
    n = 0
    for iid, src in sorted(paths.items()):
        dest = out_dir / f"{iid}.jpg"
        dest.symlink_to(src)
        n += 1
    return n


def detector_name(cls: str) -> str:
    """The string the reviewer sees while voting: the class's written rule.

    `SCALE_CLASS_RULES` carries the wording a class was reviewed under, and it
    is the only place a definition travels with the work (#3612 -- `book` split
    over magazines because the rule lived in a manifest). A class with no rule
    falls back to its bare name and is reported, because an unruled class is a
    ruling somebody owes (#3673) rather than a class without a boundary.
    """
    rule = pc.SCALE_CLASS_RULES.get(cls)
    return getattr(rule, "name", "") or cls


def api(base: str, path: str, payload: dict | None = None, method: str = "GET") -> dict:
    url = base.rstrip("/") + path
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(  # noqa: S310 - our own app on the cluster
        url, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as fh:  # noqa: S310
            body = fh.read().decode()
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"{method} {path} -> {exc.code}: {exc.read().decode()[:400]}") from exc
    return json.loads(body) if body.strip() else {}


def main() -> int:
    pc.setup_env()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    base = pc.PILE.parent / "vgscale-3156"
    ap.add_argument("--queue", default=str(base / "annotation_queue.jsonl"))
    ap.add_argument("--cell", default=str(pc.EMBEDDINGS / "vg_scale__siglip.pkl"))
    ap.add_argument("--out", default=str(base / "pass25"))
    ap.add_argument("--per-class", type=int, default=CONTROLS_PER_CLASS)
    ap.add_argument("--api", default="", help="app base URL, e.g. http://rack7n06:11850")
    ap.add_argument("--create", action="store_true", help="import the dataset and create the 25 detectors")
    ap.add_argument("--dataset-name", default="vg_scale off-COCO positives")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "calibration"))
    sys.path.insert(0, str(Path(__file__).resolve().parent / "pilebuild"))
    from _cells_io import load_medias  # noqa: PLC0415

    from pilebuild.audit import coco_held_by  # noqa: PLC0415

    rows = [json.loads(line) for line in Path(args.queue).read_text().splitlines() if line.strip()]
    log(f"{len(rows)} queue rows from {Path(args.queue).name}")

    medias = load_medias(Path(args.cell))
    held_by = coco_held_by()
    controls = draw_controls(medias, held_by, args.per_class)
    log(f"{len(controls)} controls drawn ({args.per_class}/class, COCO-answered)")

    queue_ids = {int(r["image_id"]) for r in rows}
    overlap = queue_ids & set(controls)
    if overlap:
        raise SystemExit(f"{len(overlap)} controls are also queue rows; the two sets must be disjoint")

    paths = {int(r["image_id"]): r["path"] for r in rows}
    for iid in controls:
        paths[iid] = medias[iid]["origin_name"]

    out = Path(args.out)
    n = link_images(paths, out / "images")
    key = {
        "dataset_name": args.dataset_name,
        "queue": sorted(queue_ids),
        "controls": {str(i): v for i, v in sorted(controls.items())},
        "per_class": args.per_class,
        "seed": CONTROL_SEED,
    }
    (out / "controls.json").write_text(json.dumps(key, indent=1) + "\n")
    log(f"{n} images linked into {out / 'images'}; key at {out / 'controls.json'}")

    unruled = [c for c in pc.SCALE_CLASSES if not pc.SCALE_CLASS_RULES.get(c)]
    if unruled:
        log(f"NOTE: no written rule for {', '.join(unruled)} -- the detector falls back to the bare name (#3673)")

    per_class_controls = Counter(c for v in controls.values() for c in v if c in set(pc.SCALE_CLASSES))
    thin = [c for c in pc.SCALE_CLASSES if per_class_controls[c] < args.per_class]
    if thin:
        log(f"NOTE: fewer than {args.per_class} known positives among the controls for: {', '.join(thin)}")

    if not args.create:
        log("built, nothing sent to the app. Re-run with --api URL --create to import.")
        return 0
    if not args.api:
        raise SystemExit("--create needs --api")

    log(f"importing {n} images as {args.dataset_name!r} ...")
    api(
        args.api,
        "/api/dataset/import/server_folder",
        {
            "path": str(out / "images"),
            "media_type": "image",
            "recursive": "false",
            "dig_archives": "false",
            "dataset_name": args.dataset_name,
        },
        method="POST",
    )
    log("  import submitted; poll /api/dataset/status")

    made = 0
    for cls in pc.SCALE_CLASSES:
        name = detector_name(cls)
        api(
            args.api,
            "/api/detectors",
            {
                "name": name,
                "media_type": "image",
                "text_query": cls,
                "embedder_type": "semantic",
                "examples": [{"type": "text", "value": cls}],
            },
            method="POST",
        )
        made += 1
    log(f"{made} detectors created, one per class, each named for its rule")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
