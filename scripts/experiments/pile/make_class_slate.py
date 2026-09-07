"""Review slates for a class that is not in the built dataset yet (#3588).

``make_audit_slate.py`` audits a class ``vg_scale`` already carries: it reads
the pickle, which holds that class's positives and the shared negative pool.
A *candidate* class has neither. Its positives have never been banded and its
negatives have never been checked for it -- so the slate has to be built from
the VG source, while still landing on the same negatives every other cell uses.

**Positives come from the source, through the loader's own passes.** Banding is
``pilebuild.loaders.vg_scale.band_candidates`` called with the candidate list,
not a second implementation: the scatter filter, the band edges and the
``@band`` naming are the ones the dataset would actually be built with, so a
class that looks well-supplied here is well-supplied there. Boxes are anchored
to COCO first, and every box is normalised by the dimensions of the image its
coordinates were measured on -- VG ships downscaled copies of the COCO
originals, and mixing the two spaces is #3281. The alternate-spelling fold is
the loader's ``canonicalise`` for the same reason (#3605).

**Negatives come from the built pickle.** The shared pool is the same 4,200
images for every cell, which is what makes classes comparable, and they are
already embedded -- so ranking a candidate against them costs one text
embedding rather than a re-embed. It is also the work that has to happen
anyway: adding `truck` to C means every image in the shared pool must be free
of trucks, and the ranked stratum is how the ones that are not get found.

**Positives are re-issued with the box drawn**, as ``make_positive_slate.py``
established: a bare thumbnail cannot settle "is there a backpack here?" for a
sub-patch object, and taking the resulting 43% small-band rejection at face
value would delete half the band the study exists to measure.

The dataset name carries the definition (``pile_config.SCALE_CLASS_RULES``),
because a reviewer cannot see a manifest while voting and an unstated
convention is what split `book` over magazines.

Usage::

    python make_class_slate.py --out /expscratch/$USER/classes-3588/slates
    python make_class_slate.py --classes truck,cup --boundary 25 --random 20
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np

import pile_config as pc

pc.setup_env()

from make_positive_slate import draw_with_inset  # noqa: E402
from pilebuild.loaders.vg_scale import (  # noqa: E402
    anchor_to_coco,
    band_candidates,
    lift_ambiguous,
    canonicalise,
    read_vg_labels,
)
from pilebuild.vgsource import vg_source  # noqa: E402


def log(msg: str) -> None:
    print(f"[cslate] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", default=str(pc.EMBEDDINGS / "vg_scale__siglip.pkl"), help="for the shared negatives")
    ap.add_argument("--embedder", default="siglip")
    ap.add_argument("--classes", default="", help="default: pile_config.SCALE_CANDIDATES_3588")
    ap.add_argument("--out", default=str(pc.PILE.parent / "classes-3588" / "slates"))
    ap.add_argument("--boundary", type=int, default=200, help="top-scoring negatives per class")
    ap.add_argument("--random", dest="n_random", type=int, default=70, help="uniform negatives per class")
    ap.add_argument("--positive", type=int, default=30, help="boxed positives per class, spread over the bands")
    ap.add_argument("--anchor-frac", type=float, default=0.2, help="share of each negative stratum with a known answer")
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--supply-only", action="store_true", help="report per-band supply and stop")
    args = ap.parse_args()

    from vtscore.embedding import embed_text_query  # noqa: PLC0415
    from vtscore.embedding.media_vectors import media_embedding  # noqa: PLC0415

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "calibration"))
    from _cells_io import load_medias  # noqa: PLC0415

    import coco_anchor  # noqa: PLC0415
    from coco_anchor import coco_truth, ensure_sources  # noqa: PLC0415

    classes = tuple(c.strip() for c in args.classes.split(",") if c.strip()) or pc.SCALE_CANDIDATES_3588
    log(f"{len(classes)} candidate classes: {', '.join(classes)}")

    # ---- positives, from the VG source through the loader's own passes -----
    vg_names = {c: pc.scale_names_for(c) for c in classes}
    # Ambiguous spellings must be READ even though they are never folded: their
    # whole job is to bar an image from the shared negative pool, and a name that
    # was not read is invisible to `lift_ambiguous` -- it suppresses nothing and
    # the contaminated negative stays. `pile_config.scale_vg_wanted` reads both
    # tables for the built classes; this is the candidate-side equivalent.
    ambiguous_names = {c: pc.scale_ambiguous_for(c) for c in classes}
    wanted_vg = {n for names in vg_names.values() for n in names}
    wanted_vg |= {n for names in ambiguous_names.values() for n in names}
    log(f"reading VG source for {len(wanted_vg)} names")
    paths, records, dims = vg_source()
    labels = read_vg_labels(records, paths, dims, wanted_vg)
    folded, _ = canonicalise(labels, {c: vg_names[c] for c in classes})
    for c, n in sorted(folded.items()):
        if vg_names[c] != (c,):
            log(f"folded {n} boxes onto {c!r} from {[n_ for n_ in vg_names[c] if n_ != c]}")

    anchor = Path(pc.PILE / "coco_anchor")
    image_data, instances = ensure_sources(anchor, False)
    # A merged class needs both halves out of COCO, folded onto the primary
    # name before anything downstream keys on it -- `anchor_to_coco` and
    # `band_candidates` both take a flat {image: {class: boxes}}.
    coco_wanted = {n for c in classes for n in pc.coco_classes_for(c)}
    truth = coco_truth(instances, coco_wanted)
    for c in classes:
        extra = pc.coco_classes_for(c) - {c}
        if not extra:
            continue
        folded_boxes = 0
        for per_class in truth.values():
            merged = list(per_class.get(c, []))
            for other in extra:
                got = per_class.pop(other, [])
                merged += got
                folded_boxes += len(got)
            if merged or c in per_class:
                per_class[c] = merged
        log(f"merged {folded_boxes} COCO boxes from {sorted(extra)} onto {c!r}")
    with image_data.open() as fh:
        meta = json.load(fh)
    coco_of = {int(m["image_id"]): int(m["coco_id"]) for m in meta if m.get("coco_id")}
    # `coco_truth` fills this as a side effect of reading the instances files,
    # so it is only populated after the call above -- read it off the module
    # rather than binding the name at import time.
    box_dims, exhaustive, n_anchored, n_reframed, _reband = anchor_to_coco(
        labels, dims, coco_of, truth, coco_anchor.COCO_DIMS, set(classes)
    )
    log(f"anchored {n_anchored} images to COCO ({n_reframed} skipped as re-framed copies)")

    # An ambiguous spelling is evidence in neither direction, so it is dropped
    # from the bands and its image barred from the shared negative pool -- the
    # same `lift_ambiguous` pass the built classes get, which is why this runs
    # AFTER `anchor_to_coco`: on the COCO-annotated half the answer is already
    # known and suppressing there would discard good negatives to fix nothing.
    ambiguous = {c: names for c, names in ambiguous_names.items() if names}
    unbanded = lift_ambiguous(labels, ambiguous, exhaustive) if ambiguous else set()
    if ambiguous:
        log(f"suppressed {len(unbanded)} (image, class) pairs on {sorted(ambiguous)}")

    supply, boxes_for, _clean = band_candidates(labels, box_dims, unbanded, classes=classes)

    print(f"\n{'class':<16}{'small':>8}{'medium':>8}{'large':>8}   name the reviewer sees")
    print("-" * 78)
    for c in classes:
        b = {band: len(supply[c][band]) for band in pc.BOX_BANDS}
        print(f"{c:<16}{b['small']:>8}{b['medium']:>8}{b['large']:>8}   {pc.scale_class_dataset_name(c)!r}")
    if args.supply_only:
        return 0

    # ---- negatives, from the built pickle's shared pool --------------------
    medias = load_medias(Path(args.cell))
    ids = sorted(medias)
    pool = [i for i in ids if not medias[i].get("categories")]
    log(f"{len(pool)} shared negatives from {Path(args.cell).name}")
    mat_all = np.stack([np.asarray(media_embedding(medias[i]), dtype=np.float32) for i in pool])
    mat_all /= np.linalg.norm(mat_all, axis=1, keepdims=True) + 1e-12
    row_of = {i: n for n, i in enumerate(pool)}

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    index: list[dict] = []

    evicted: dict[str, list[int]] = {}
    for c in classes:
        name = pc.scale_class_dataset_name(c)
        # The shared pool was drawn as "holds none of the CURRENT twelve", so it
        # is NOT a negative pool for a candidate: an image can sit in it and
        # hold a car. Those images are neither negatives (the object is there)
        # nor positives of a band (they may be scattered or oversize), which is
        # exactly the `excluded` third state the vg_scale construction already
        # has. Draw them into a slate and the same image arrives twice, once
        # asking "is there a car here?" and once asserting there is -- and the
        # boxed render silently overwrites the bare one.
        #
        # The count is the measurement, not the leftovers: it is how much of the
        # shared pool this class would contaminate if it joined C.
        holds = {i for i in pool if c in labels.get(i, {})}
        evicted[c] = sorted(holds)
        negatives = [i for i in pool if i not in holds]
        mat = mat_all[[row_of[i] for i in negatives]]
        # Rank on the class name, not the rule string: the rule is prose for a
        # human and would drag the text tower somewhere the class is not.
        tvec = embed_text_query(c, "image", embedder_name=args.embedder)
        if tvec is None:
            raise SystemExit(f"no text tower for embedder {args.embedder!r}")
        tv = np.asarray(tvec, dtype=np.float32)
        tv /= np.linalg.norm(tv) + 1e-12
        scores = dict(zip(negatives, (mat @ tv).tolist()))

        # A minority of each stratum has an answer already: COCO looked at these
        # images and annotated this class exhaustively. Reviewing them corrects
        # nothing and scores the reviewer, which is what turns the open half's
        # residual error into a bounded number. Indistinguishable at voting time
        # -- every file is named by image id alone.
        anchored = [i for i in negatives if medias[i].get("labels_exhaustive")]
        open_ = [i for i in negatives if not medias[i].get("labels_exhaustive")]
        n_anchor_b = int(round(args.boundary * args.anchor_frac))
        n_anchor_r = int(round(args.n_random * args.anchor_frac))
        by_score = sorted(open_, key=lambda i: -scores[i])
        anchor_by_score = sorted(anchored, key=lambda i: -scores[i])
        boundary = by_score[: args.boundary - n_anchor_b] + anchor_by_score[:n_anchor_b]
        chosen_b = set(boundary)
        rest_open = [i for i in open_ if i not in chosen_b]
        rest_anchor = [i for i in anchored if i not in chosen_b]
        uniform = rng.sample(rest_open, min(args.n_random - n_anchor_r, len(rest_open))) + rng.sample(
            rest_anchor, min(n_anchor_r, len(rest_anchor))
        )

        per_band = max(1, args.positive // len(pc.BOX_BANDS))
        chosen_pos: list[tuple[int, str]] = []
        for band in pc.BOX_BANDS:
            # NOT `pool`: that name holds the shared negative pool, and
            # rebinding it here made every class after the first draw its
            # negatives from the previous class's last band.
            band_pool = sorted(supply[c][band])
            for i in rng.sample(band_pool, min(per_band, len(band_pool))):
                chosen_pos.append((i, pc.scale_cell(c, band)))

        cdir = out_root / name.replace(" ", "_")
        cdir.mkdir(parents=True, exist_ok=True)
        rows: list[dict] = []

        for stratum, items in (("boundary", boundary), ("random", uniform)):
            for i in items:
                src = paths.get(i)
                if src is None:
                    continue
                (cdir / f"{i}.jpg").write_bytes(src.read_bytes())
                rows.append(
                    {
                        "image_id": i,
                        "class": c,
                        "stratum": stratum,
                        "cell": "",
                        "text_score": round(float(scores.get(i, 0.0)), 4),
                        "reference": "absent",
                        "exhaustive": "yes" if medias[i].get("labels_exhaustive") else "no",
                        "n_boxes": 0,
                        "detector": name,
                    }
                )

        for i, cell in chosen_pos:
            src = paths.get(i)
            bs = boxes_for.get((i, cell))
            if src is None or not bs:
                continue
            W, H = box_dims[i]
            box = (
                min(b[0] for b in bs) / W,
                min(b[1] for b in bs) / H,
                max(b[2] for b in bs) / W,
                max(b[3] for b in bs) / H,
            )
            draw_with_inset(src, box, cdir / f"{i}.jpg")
            rows.append(
                {
                    "image_id": i,
                    "class": c,
                    "stratum": "positive_boxed",
                    "cell": cell,
                    "text_score": 0.0,
                    "reference": "present",
                    "exhaustive": "yes" if i in exhaustive else "no",
                    "n_boxes": len(bs),
                    "detector": name,
                }
            )

        with (cdir / "manifest.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        n_pos = sum(1 for r in rows if r["stratum"] == "positive_boxed")
        index.append({"class": c, "name": name, "dir": str(cdir), "n": len(rows), "n_positive": n_pos})
        log(f"  {c:<14} {len(rows):3d} images ({n_pos} boxed positives) -> {cdir.name}")

    (out_root / "slates.json").write_text(json.dumps(index, indent=1) + "\n")
    (out_root / "evicted.json").write_text(json.dumps(evicted, indent=1) + "\n")
    print(f"\n{'class':<16}{'evicted':>9}   images in the shared pool that already hold this class")
    print("-" * 78)
    for c in classes:
        print(f"{c:<16}{len(evicted[c]):>9}   {100.0 * len(evicted[c]) / len(pool):>5.2f}% of the pool")
    print(f"\n{sum(e['n'] for e in index)} images across {len(index)} classes under {out_root}")
    print(f"\nImport them and create the detectors with:\n  python import_slates.py --slates {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
