"""``vg_scale_any``: ``vg_scale`` with the box-size band collapsed away (#3115)."""

from __future__ import annotations

import pile_config as pc

from pilebuild.env import cells_io, log
from pilebuild.geometry import scale_label_digest


def load(dataset: str, medias: dict[int, dict], embedder_name: str) -> None:
    """``vg_scale`` with the box-size band collapsed away (#3115).

    #3156 built ``vg_scale`` to ask a question about *scale*, so its cells are
    keyed ``class@band`` and each holds exactly 100 positives.  A study that does
    not care how big the box is wants the same verified images with the band
    dropped: 12 classes, **300 positives each**, one shared negative pool, and
    identical prevalence everywhere.  That is what the calibration studies
    actually need, and what ``visual_genome_m`` conspicuously is not - its
    selected categories run from 25 positives (``banana``) to 1645
    (``building``), and the thin ones produce cells with *no trainable step at
    all* (``ball``, 51 positives, wrote a header and zero rows).

    Derived from the built ``vg_scale`` pickle rather than re-scanned from the
    VG source, so it is **the same images, the same boxes and the same
    hand-checked label corrections** with one string operation applied - and it
    costs no image decode and no embedding pass.

    The whole build is a relabel, ``class@band -> class``, applied to
    ``category`` / ``categories`` / ``evaluable_categories`` and to each region's
    ``label``.  What matters is what that does *not* change, because the
    exclusion semantics are the point of #3156 and the easy version of this
    build destroys them:

    * The **300 medias evaluable on nothing** stay evaluable on nothing.  They
      hold one of the 12 classes at a size outside every band, so scoring them
      as negatives would penalise a detector for finding a real bus - exactly
      what #3156 exists to prevent.  A naive "positive if in any band, negative
      otherwise" rule turns all 300 into negatives.
    * A media positive for one class stays evaluable only for that class **off
      COCO**, where ``labels_exhaustive`` is False and an image of a dog is not
      evidence of the absence of a clock. On the COCO half it is now a clock
      negative, because there the absence is annotated (#3667).
    * The 3900-image clean pool - images holding no instance of any of the 12 -
      stays evaluable for all of them.

    Result per class: 300 positives against the 3900 shared negatives **plus the
    COCO-exhaustive positives of the other eleven classes**. The pool is
    therefore no longer 4200 images at a prevalence identical for all twelve;
    the shared part is shared and the added part is per-class. #3156's paired
    contrast is unaffected here, because this dataset has already collapsed the
    bands it was about. See
    ``docs/experiments/2026-09-06-cross-class-negatives-3667/REPORT.md``.
    """
    src = pc.cell_path("vg_scale", embedder_name)
    if not src.exists():
        raise SystemExit(f"vg_scale_any is derived from {src.name}, which does not exist yet - build vg_scale first")

    def _base(label: str) -> str:
        return label.split("@", 1)[0]

    def _collapse(labels) -> list[str]:
        """Base classes, order-preserving and deduped."""
        return list(dict.fromkeys(_base(c) for c in (labels or [])))

    loaded = cells_io().load_medias(src)
    parent_digest = scale_label_digest(loaded)
    log(f"  derived from {src.name}: {len(loaded)} medias, parent labels {parent_digest[:12]}")
    for mid, media in loaded.items():
        d = dict(media)
        d["categories"] = _collapse(d.get("categories"))
        # `evaluable_categories` is collapsed rather than rebuilt: an empty list
        # must survive as an empty list (see the docstring), and rebuilding it
        # from the class roster is precisely the bug that would not.
        d["evaluable_categories"] = _collapse(d.get("evaluable_categories"))
        if d.get("category"):
            d["category"] = _base(d["category"])
        d["regions"] = [{**r, "label": _base(r["label"])} for r in (d.get("regions") or [])]
        # The parent's label digest travels with every media, so `--verify` can
        # tell a derived cell that is merely older than its parent from one that
        # no longer agrees with it. Without it a stale derivation is invisible.
        d["origin"] = {
            "importer": "vg_scale_any",
            "params": {"embedder": embedder_name, "derived_from": src.name, "parent_labels": parent_digest},
        }
        medias[mid] = d

    n_pos = sum(1 for d in medias.values() if d["categories"])
    n_clean = sum(1 for d in medias.values() if not d["categories"] and d["evaluable_categories"])
    n_excluded = sum(1 for d in medias.values() if not d["categories"] and not d["evaluable_categories"])
    log(f"  {n_pos} positives, {n_clean} shared negatives, {n_excluded} excluded-everywhere (must be nonzero)")
    if not n_excluded:
        raise SystemExit("vg_scale_any: no excluded-everywhere medias survived - the exclusion semantics were lost")


def check(dataset: str) -> str:
    """Derived from the built parent, so its "source" is a cell.

    Absent parents are not a problem here: a full run builds the parent first,
    and a purged pile rebuilds both in order.
    """
    built = [e for e in pc.EMBEDDERS if pc.cell_path("vg_scale", e).exists()]
    return f"derives from vg_scale ({len(built)} parent cells built)"
