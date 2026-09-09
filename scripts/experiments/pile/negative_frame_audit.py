#!/usr/bin/env python3
"""How big is the multi-object blind spot in vg_scale's evaluation?

Measured: 41.9% of the pile is dropped from EVERY class's evaluation for
holding a different class, and about 1,850 of those carry an exact COCO answer
already -- a 44% gain in negatives per class at zero labelling cost. See #3667.

`evaluable_categories` is `cats if cats else (cells if negative else [])`. So an
image that is a positive for ANY class is scorable ONLY in its own cells. An
image holding a book but no bus is therefore neither a bus positive nor a bus
negative -- it is dropped from the bus evaluation entirely.

That means a detector for A is never asked whether it fires on an image that
holds B and not A. This counts what that excludes.
"""

import sys

sys.path.insert(0, "scripts/experiments/pile")
sys.path.insert(0, "scripts/experiments/calibration")
import pile_config as pc  # noqa: E402
from _cells_io import load_medias  # noqa: E402

m = load_medias(pc.EMBEDDINGS / "vg_scale__siglip.pkl")
classes = list(pc.SCALE_CLASSES)
haspos = [i for i in m if m[i].get("categories")]
pool = [i for i in m if not m[i].get("categories")]

exh_pos = [i for i in haspos if m[i].get("labels_exhaustive")]
exh_pool = [i for i in pool if m[i].get("labels_exhaustive")]
print(
    f"positives           {len(haspos):5d}, of which COCO-exhaustive {len(exh_pos):5d} "
    f"({100 * len(exh_pos) / len(haspos):.1f}%)"
)
print(
    f"shared negatives    {len(pool):5d}, of which COCO-exhaustive {len(exh_pool):5d} "
    f"({100 * len(exh_pool) / len(pool):.1f}%)\n"
)

print(f"{'class':<12}{'blind':>8}{'blind & exact':>15}{'negatives now':>15}{'could be':>10}{'gain':>8}")
print("-" * 70)
for c in classes:
    blind = [i for i in haspos if all(x.split("@")[0] != c for x in m[i]["categories"])]
    exact = [i for i in blind if m[i].get("labels_exhaustive")]
    now, could = len(pool), len(pool) + len(exact)
    print(f"{c:<12}{len(blind):>8}{len(exact):>15}{now:>15}{could:>10}{100 * (could - now) / now:>7.1f}%")

# how "empty" are the current negatives vs the blind set?
print("\nWhat the current negatives look like, versus what is being excluded:")
print("  a shared-pool image holds 0 of the twelve, by construction.")
print(
    f"  a blind image holds at least 1 -- and {sum(1 for i in haspos if len({x.split('@')[0] for x in m[i]['categories']}) > 1)} of the pile's positives hold 2+."
)

print("\n" + "=" * 62)
print(f"{'class':<12}{'positives':>10}{'negatives':>11}{'blind (other cls)':>19}{'blind %':>9}")
print("-" * 62)
rows = []
for c in classes:
    pos = [i for i in haspos if any(x.split("@")[0] == c for x in m[i]["categories"])]
    # holds another class but not c -> excluded from c's evaluation entirely
    blind = [i for i in haspos if all(x.split("@")[0] != c for x in m[i]["categories"])]
    n_eval = len(pos) + len(pool)
    rows.append((c, len(pos), len(pool), len(blind), 100 * len(blind) / (n_eval + len(blind))))
    print(f"{c:<12}{len(pos):>10}{len(pool):>11}{len(blind):>19}{rows[-1][4]:>8.1f}%")

avg = sum(r[4] for r in rows) / len(rows)
print(f"\nOn average {avg:.1f}% of the pile is silently dropped from each class's evaluation")
print("because it holds a DIFFERENT class -- the AB/BC images, never scored for A.")
