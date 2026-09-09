#!/usr/bin/env python3
"""As pool_audit, but on the population the negative slate actually draws from,
and asking whether the shipped-class 0.0% is a measurement or a tautology."""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "scripts/experiments/pile")
import pile_config as pc  # noqa: E402
from pilebuild.loaders.vg_scale import read_vg_labels, vg_source  # noqa: E402
from make_class_slate import canonicalise  # noqa: E402

sys.path.insert(0, "scripts/experiments/calibration")
from _cells_io import load_medias  # noqa: E402

from coco_anchor import coco_truth, ensure_sources  # noqa: E402

SHIPPED, CAND = set(pc.SCALE_CLASSES_ORIGINAL), set(pc.SCALE_CANDIDATES_3588)
medias = load_medias(pc.EMBEDDINGS / "vg_scale__siglip.pkl")
ids = sorted(medias)
pool = [i for i in ids if not medias[i].get("categories")]
withcat = [i for i in ids if medias[i].get("categories")]

anchor = Path(pc.PILE / "coco_anchor")
image_data, instances = ensure_sources(anchor, False)
truth = coco_truth(instances, SHIPPED | CAND)
with image_data.open() as fh:
    meta = json.load(fh)
coco_of = {int(m["image_id"]): int(m["coco_id"]) for m in meta if m.get("coco_id")}


def present(i):
    cid = coco_of.get(i)
    return None if cid is None else {c for c, b in truth.get(cid, {}).items() if b}


# --- is the shipped 0.0% a tautology? ---------------------------------------
# If `categories` is COCO-derived, then "no categories" MEANS "COCO says no
# shipped class", and 0.0% is true by construction rather than measured. The
# test: on images that DO carry categories, does the stored set match COCO?
agree = mismatch = 0
for i in withcat[:4000]:
    p = present(i)
    if p is None:
        continue
    # `categories` holds CELLS -- "book@medium" -- not bare class names, so
    # intersecting it with a set of class names is always empty and the test
    # silently reported a 0.2% match. Split the cell at "@".
    stored = {x.split("@")[0] for x in (medias[i].get("categories") or [])}
    if stored & SHIPPED == p & SHIPPED:
        agree += 1
    else:
        mismatch += 1
tot = agree + mismatch
print(
    f"categories-vs-COCO on {tot} annotated non-pool images: {100 * agree / tot:.1f}% exact match on the shipped twelve"
)
print(
    "  -> pool membership tracks COCO: the shipped 0.0% is BY CONSTRUCTION\n"
    if agree / tot > 0.95
    else "  -> categories is NOT simply COCO; the 0.0% is a real measurement\n"
)

# --- the population the slate draws from ------------------------------------
vg_names = {c: pc.scale_names_for(c) for c in CAND}
amb = {c: pc.scale_ambiguous_for(c) for c in CAND}
wanted = {n for v in vg_names.values() for n in v} | {n for v in amb.values() for n in v}
paths, records, dims = vg_source()
labels = read_vg_labels(records, paths, dims, wanted)
canonicalise(labels, dict(vg_names))
amb_names = {n for v in amb.values() for n in v}
holds = {i for i in pool if any(c in labels.get(i, {}) for c in CAND)}
ambset = {i for i in pool if i not in holds and amb_names & set(labels.get(i, {}))}
clean = [i for i in pool if i not in holds and i not in ambset]

for label, group in (("whole pool", pool), ("slate population (post-eviction)", clean)):
    sc = [i for i in group if present(i) is not None]
    hit = [i for i in sc if present(i) & CAND]
    cnt = Counter(c for i in hit for c in present(i) & CAND)
    print(f"{label}: {len(group)} images, {len(sc)} COCO-scored")
    print(f"  COCO says a candidate is present in {len(hit)}/{len(sc)} = {100 * len(hit) / len(sc):.1f}%")
    print(f"  by class: {dict(cnt.most_common(8))}\n")
