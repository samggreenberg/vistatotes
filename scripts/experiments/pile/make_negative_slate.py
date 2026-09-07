#!/usr/bin/env python3
"""Build the negative-pass slate: is the SHARED pool actually clean?

Every per-class slate measured one class against its own negatives. This one
measures the pool itself, and it is a different question. An image sits in the
shared pool because it carries no VG label for any candidate -- but the whole
finding of #3588 is that a missing VG name is not an absent object. The random
stratum of the thirteen class slates put per-class pool error between 0.0% and
7.1%; what nobody has measured is the JOINT rate, the share of pool images that
hold at least one of the thirteen. That number is what a negative in this
benchmark is worth.

Two strata, mirroring `make_class_slate.py` so the two passes stay comparable:

* ``random`` -- uniform from the pool. This is the estimator. Nothing else here
  measures anything; the boundary stratum is chosen to be wrong.
* ``boundary`` -- ranked by the best text score over all thirteen class names,
  so contamination is surfaced cheaply. Biased by design, and the manifest
  records WHICH class drove each row so the bias can be read afterwards.

Images VG already labels with a candidate are evicted first: they are known
contamination, they are counted by `make_class_slate.py` as `evicted.json`, and
asking a reviewer about them measures nothing.

**The pass covers all TWENTY-FIVE classes, not just the thirteen.** The shipped
twelve were never reviewed either -- same construction, same VG-silence
negatives, same unmeasured error (#3666) -- and treating the two halves
differently would leave the benchmark with two tiers of label quality that
any cross-class comparison would confound. Since the reviewer is looking at
the image anyway, the twelve cost one extra group rather than twelve extra
passes: `knife` and `book` fall in the table group, `bus`, `bicycle` and
`stop sign` in the street group, and the six that fit neither form their own.

**So are images the thirteen class passes already found the object in, and
missing that was a real bug.** The first build of this slate sampled the
PRE-CORRECTION pool: 69% of its 200 rows already carried a verdict, and 44 of
them were images the reviewer had personally marked `present`. Those images do
not survive `verdicts_to_corrections.py` -- they leave the pool as
`negative_fixed` or `negative_excluded` -- so putting them back in asks the
reviewer to rediscover their own findings and measures a pool that no longer
exists. The question this pass answers is about the pool **after** the thirteen
passes are applied.

Reviewed-and-ABSENT images stay in the frame on purpose. "No vase here" says
nothing about the other twelve, so the image is still a live unknown, and
dropping every reviewed image would bias the estimate downward: the reviewed set
is enriched in contamination by the boundary ranking that chose it.

**Polarity is stated positively on purpose.** The reviewer's question is "do you
see NONE of the thirteen?", so Good means clean. A conjunction of thirteen
negatives is not something anyone can hold in their head, which is why the
guide's checklist is ordered by measured hit rate rather than alphabetically --
six classes account for every pool error observed so far.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pile_config as pc  # noqa: E402
from make_class_slate import canonicalise, log  # noqa: E402
from pilebuild.loaders.vg_scale import read_vg_labels, vg_source  # noqa: E402

#: Five groups over the same 200 images, arrived at by the reviewer and matching
#: the co-occurrence data exactly -- which it did not at first.
#:
#: The route there is worth keeping. A measured three-group split was rejected as
#: too taxing; a semantic six-group split followed; then `Furniture` dissolved,
#: `chair` going to the table and `bench` to the street. That last move is what
#: the images had said all along: `chair` sits with a cup in 22% of its images
#: and `vase` sits with a chair in 43%, while `bench` sits with a car in 25%, a
#: bicycle in 16%, and a chair in only 6%. Chair is a dining-room object and
#: bench is a street object, whatever the word "furniture" suggests.
#:
#: So the grouping is both easy to hold AND scene-coherent, and it costs one
#: fewer pass than the version that was only the first of those. Grouping by
#: what a reviewer can hold and grouping by what a camera sees turned out to
#: agree once the semantic category was dropped as the organising idea.
#:
#: `clock` is the one placement no data supports, because none can: it is the
#: most scattered class of the twenty-five, with 38% of its images holding
#: nothing else. Street is a choice, not a finding -- and better than Handheld,
#: which would have invited skipping clock towers and wall clocks.
#:
#: What grouping buys beyond load: for a NEGATIVE pass the within-group boundary
#: does not exist. Cup-or-bowl, fork-or-spoon and van-or-SUV stop being
#: questions, because either answer makes the image not clean. The cost of a
#: group that is hard to hold is a MISSED positive, and a missed positive biases
#: the pool estimate DOWNWARD -- the one failure mode invisible in the result.
GROUPS: dict[str, tuple[str, ...]] = {
    "Table Objects": (
        "bowl",
        "cup",
        "bottle",
        "vase",
        "fork",
        "spoon",
        "sink",
        "knife",
        "chair",
    ),
    "Handheld Objects": ("cell phone", "book", "umbrella", "backpack"),
    "Vehicles": ("car", "truck", "bus", "bicycle"),
    "Street Objects": ("fire hydrant", "stop sign", "clock", "bench"),
    "Outdoor Objects": ("bird", "kite", "boat", "dog"),
}

#: Every class, for the frame and the ranking.
CHECKLIST = tuple(c for g in GROUPS.values() for c in g)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", default=str(pc.EMBEDDINGS / "vg_scale__siglip.pkl"))
    ap.add_argument("--embedder", default="siglip")
    ap.add_argument("--out", default=str(pc.PILE.parent / "classes-3588" / "slates"))
    ap.add_argument("--boundary", type=int, default=100)
    ap.add_argument("--random", dest="n_random", type=int, default=100)
    ap.add_argument(
        "--verdicts",
        default="",
        help="banked class-pass verdicts; images "
        "found POSITIVE there are corrected out of the pool and must not be re-asked",
    )
    ap.add_argument("--seed", type=int, default=20260905)
    args = ap.parse_args()

    from vtscore.embedding import embed_text_query  # noqa: PLC0415
    from vtscore.embedding.media_vectors import media_embedding  # noqa: PLC0415

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "calibration"))
    from _cells_io import load_medias  # noqa: PLC0415

    classes = tuple(CHECKLIST)
    # The CANDIDATE table, not the shipped one: these classes are candidates,
    # and `SCALE_VG_NAMES` would silently fall back to the bare class name for
    # every one of them -- reading `cup` while ignoring mug, goblet and the
    # rest, which is exactly the spelling-split blindness this study found.
    # Two tables, because the candidates and the shipped twelve keep their
    # measured aliases in different places; falling back to the bare class name
    # for either is the spelling-split blindness this study found.
    vg_names = {c: pc.scale_names_for(c) for c in classes}
    ambiguous = {c: pc.scale_ambiguous_for(c) for c in classes}
    wanted = {n for names in vg_names.values() for n in names}
    wanted |= {n for names in ambiguous.values() for n in names}
    log(
        f"reading VG source for {len(wanted)} names over {len(classes)} classes "
        f"({sum(len(v) for v in ambiguous.values())} ambiguous)"
    )
    paths, records, dims = vg_source()
    labels = read_vg_labels(records, paths, dims, wanted)
    canonicalise(labels, dict(vg_names))

    medias = load_medias(Path(args.cell))
    pool = [i for i in sorted(medias) if not medias[i].get("categories")]
    log(f"{len(pool)} shared negatives from {Path(args.cell).name}")

    # Known contamination is not the question -- evict it and say how much.
    holds = {i for i in pool if any(c in labels.get(i, {}) for c in classes)}
    # An ambiguous spelling is evidence of neither presence nor absence, so an
    # image carrying one is barred from the negative pool rather than counted
    # as clean -- the same three-valued treatment `lift_ambiguous` gives bands.
    amb_names = {n for names in ambiguous.values() for n in names}
    amb = {i for i in pool if i not in holds and amb_names & set(labels.get(i, {}))}
    # Images a class pass already found the object in are corrected out of the
    # pool. Sampling them re-asks a question that has an answer and measures the
    # pool as it was before the thirteen passes ran.
    found: set[int] = set()
    if args.verdicts:
        for v in json.loads(Path(args.verdicts).read_text()):
            if v.get("human") == "present":
                found.add(int(v["image_id"]))
    clean_pool = [i for i in pool if i not in holds and i not in amb and i not in found]
    in_pool = found & {i for i in pool if i not in holds and i not in amb}
    log(
        f"evicted {len(holds)} images VG already labels with a candidate "
        f"({100 * len(holds) / len(pool):.1f}%), {len(amb)} carrying an "
        f"ambiguous spelling, and {len(in_pool)} the class passes already found "
        f"the object in; {len(clean_pool)} remain"
    )

    mat = np.stack([np.asarray(media_embedding(medias[i]), dtype=np.float32) for i in clean_pool])
    mat /= np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12

    best = np.full(len(clean_pool), -np.inf, dtype=np.float32)
    driver = [""] * len(clean_pool)
    for c in classes:
        tv = embed_text_query(c, "image", embedder_name=args.embedder)
        if tv is None:
            raise SystemExit(f"no text tower for embedder {args.embedder!r}")
        v = np.asarray(tv, dtype=np.float32)
        v /= np.linalg.norm(v) + 1e-12
        s = mat @ v
        hit = s > best
        best = np.where(hit, s, best)
        for n in np.nonzero(hit)[0]:
            driver[n] = c

    order = sorted(range(len(clean_pool)), key=lambda n: -float(best[n]))
    boundary = order[: args.boundary]
    rest = [n for n in order[args.boundary :]]
    rng = random.Random(args.seed)
    uniform = rng.sample(rest, min(args.n_random, len(rest)))

    # One slate per group, over the SAME sampled images: the passes are N looks
    # at one sample, not N samples.
    #
    # Images are written ONCE, into the first group's directory, and the other
    # groups get a manifest alone. The negative set is identical every time and
    # only the question changes -- a question is a DETECTOR, not a dataset. The
    # earlier version copied the JPEGs per group and let `import_slates.py` build
    # a dataset from each, which embedded the same 200 images once per group.
    # See #3669.
    rows_by_group: dict[str, list[dict]] = {g: [] for g in GROUPS}
    for group in GROUPS:
        (Path(args.out) / group.replace(" ", "_")).mkdir(parents=True, exist_ok=True)
    rows = []
    for stratum, items in (("boundary", boundary), ("random", uniform)):
        for n in items:
            i = clean_pool[n]
            src = paths.get(i)
            if src is None:
                continue
            first = next(iter(GROUPS))
            (Path(args.out) / first.replace(" ", "_") / f"{i}.jpg").write_bytes(src.read_bytes())
            # COCO annotates all 80 of its classes on an image it annotates at
            # all, so one exhaustive flag settles the whole conjunction at once
            # -- a scored subset for the negative pass, free.
            rows.append(
                {
                    "image_id": i,
                    "class": "clean",
                    "stratum": stratum,
                    "cell": "",
                    "text_score": round(float(best[n]), 4),
                    "reference": "present" if medias[i].get("labels_exhaustive") else "",
                    "exhaustive": "yes" if medias[i].get("labels_exhaustive") else "no",
                    "n_boxes": 0,
                    "detector": "",
                    "driver": driver[n],
                }
            )
            for group in GROUPS:
                r = dict(rows[-1])
                r["detector"] = group
                r["class"] = group
                rows_by_group[group].append(r)

    for group, grows in rows_by_group.items():
        gdir = Path(args.out) / group.replace(" ", "_")
        with (gdir / "manifest.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(grows[0]))
            w.writeheader()
            w.writerows(grows)
        log(f"  {group}: {len(grows)} rows -> {gdir}")
    (Path(args.out) / "negative_pass.json").write_text(
        json.dumps(
            {
                "evicted_known": len(holds),
                "pool": len(pool),
                "clean_pool": len(clean_pool),
                "evicted_ambiguous": len(amb),
                "evicted_already_found": len(in_pool),
                "checklist": list(classes),
                "rows": len(rows),
            },
            indent=1,
        )
    )

    n_exh = sum(1 for r in rows if r["exhaustive"] == "yes")
    log(f"sampled {len(rows)} images ({n_exh} scored, {100 * n_exh / len(rows):.0f}%) into {len(GROUPS)} group slates")
    from collections import Counter

    log(f"boundary drivers: {dict(Counter(r['driver'] for r in rows if r['stratum'] == 'boundary').most_common(6))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
