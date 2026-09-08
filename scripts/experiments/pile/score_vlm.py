#!/usr/bin/env python3
"""Score the two VLM framings against COCO truth (#3720).

`ruled` answers in our class names already. `open` answers in its own words, so
it is scored *through a mapping* -- and the mapping is the point: a name we fail
to map is a vocabulary gap we can close without re-running the model, whereas a
`ruled` miss is gone unless we pay the GPU again.

The synonym table below is deliberately written out rather than derived. It
encodes our conventions, several of which are not what a reasonable person would
guess -- a jar is a `bottle`, a plate is a `bowl`, a magazine is a `book`, a
parasail is a `kite` -- and those are exactly the rows worth arguing over.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

SYN: dict[str, set[str]] = {
    "bird": {
        "pigeon",
        "duck",
        "goose",
        "gull",
        "seagull",
        "swan",
        "parrot",
        "owl",
        "chicken",
        "hen",
        "rooster",
        "sparrow",
        "crow",
        "raven",
        "eagle",
        "penguin",
        "ostrich",
        "flamingo",
        "peacock",
        "turkey",
        "dove",
        "pelican",
    },
    "car": {
        "sedan",
        "hatchback",
        "coupe",
        "suv",
        "minivan",
        "taxi",
        "cab",
        "jeep",
        "station wagon",
        "automobile",
        "police car",
        "race car",
        "sports car",
    },
    "truck": {
        "pickup",
        "pickup truck",
        "semi",
        "lorry",
        "tow truck",
        "fire truck",
        "box truck",
        "dump truck",
        "cargo van",
        "semi truck",
        "trailer truck",
    },
    "bus": {"coach", "school bus", "double decker bus", "double-decker bus", "shuttle"},
    "cell phone": {"phone", "smartphone", "mobile phone", "cellphone", "iphone", "mobile"},
    "cup": {
        "mug",
        "teacup",
        "glass",
        "drinking glass",
        "tumbler",
        "wine glass",
        "goblet",
        "coffee cup",
        "paper cup",
        "coffee mug",
        "pint",
        "wineglass",
    },
    "bowl": {"dish", "plate", "saucer", "serving bowl", "paper plate", "platter"},
    "bottle": {"jar", "water bottle", "wine bottle", "beer bottle", "soda bottle", "flask", "canteen", "vial"},
    "vase": {"flower pot", "flowerpot", "planter", "urn", "pot", "potted plant"},
    "book": {"magazine", "notebook", "textbook", "novel", "pamphlet", "booklet"},
    "knife": {"chef knife", "butter knife", "steak knife", "blade", "cleaver", "pocket knife", "bread knife"},
    "spoon": {"ladle", "teaspoon", "tablespoon", "soup spoon", "wooden spoon"},
    "chair": {
        "stool",
        "bar stool",
        "armchair",
        "high chair",
        "deck chair",
        "office chair",
        "folding chair",
        "recliner",
    },
    "bench": {"pew", "park bench", "church pew", "picnic bench"},
    "clock": {"wall clock", "alarm clock", "tower clock", "grandfather clock", "timer"},
    "backpack": {"rucksack", "daypack", "school bag", "knapsack", "book bag"},
    "umbrella": {"parasol", "beach umbrella", "patio umbrella", "sunshade"},
    "kite": {"parasail", "paraglider", "hang glider", "parachute"},
    "boat": {
        "ship",
        "ferry",
        "yacht",
        "sailboat",
        "canoe",
        "kayak",
        "raft",
        "dinghy",
        "gondola",
        "rowboat",
        "speedboat",
        "barge",
        "vessel",
    },
    "bicycle": {"bike", "cycle", "mountain bike", "road bike"},
    "dog": {
        "puppy",
        "hound",
        "labrador",
        "retriever",
        "poodle",
        "terrier",
        "bulldog",
        "beagle",
        "husky",
        "chihuahua",
        "german shepherd",
    },
    "sink": {"basin", "washbasin", "wash basin", "kitchen sink", "bathroom sink"},
    "fire hydrant": {"hydrant"},
    "stop sign": set(),
    "fork": set(),
}


def norm(s: str) -> str:
    s = re.sub(r"[^a-z0-9 ]+", " ", s.lower()).strip()
    s = re.sub(r"\s+", " ", s)
    return s[:-1] if s.endswith("s") and not s.endswith("ss") else s


def build_lookup(classes: list[str]) -> dict[str, str]:
    look: dict[str, str] = {}
    for c in classes:
        look[norm(c)] = c
        for a in SYN.get(c, set()):
            look[norm(a)] = c
    return look


def prf(tp: int, fp: int, fn: int) -> tuple[float, float]:
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    return p, r


def main() -> int:
    preds = Path(sys.argv[1])
    sys.path.insert(0, "scripts/experiments/pile")
    import pile_config as pc  # noqa: PLC0415

    classes = list(pc.SCALE_CLASSES)
    cset = set(classes)
    look = build_lookup(classes)

    rows = [json.loads(x) for x in preds.read_text().splitlines() if x.strip()]
    rows = [r for r in rows if "error" not in r]
    unmapped: Counter = Counter()

    stats = {m: defaultdict(lambda: [0, 0, 0]) for m in ("ruled", "open")}
    for r in rows:
        truth = set(r["truth"])
        for mode in ("ruled", "open"):
            got = r.get(f"{mode}_parsed") or []
            if mode == "ruled":
                pred = {g for g in (norm(x) for x in got) if g in cset}
                pred |= {x for x in got if x in cset}
            else:
                pred = set()
                for name in got:
                    hit = look.get(norm(name))
                    if hit:
                        pred.add(hit)
                    else:
                        unmapped[norm(name)] += 1
            for c in classes:
                s = stats[mode][c]
                if c in truth and c in pred:
                    s[0] += 1
                elif c in pred:
                    s[1] += 1
                elif c in truth:
                    s[2] += 1

    print(f"scored {len(rows)} images, {len(rows) * len(classes)} image-class pairs\n")
    hdr = f"{'class':<13}{'n':>4} | {'RULED  P':>9}{'R':>7} | {'OPEN   P':>9}{'R':>7}"
    print(hdr)
    print("-" * len(hdr))
    tot = {m: [0, 0, 0] for m in stats}
    for c in classes:
        a, b = stats["ruled"][c], stats["open"][c]
        for m, s in (("ruled", a), ("open", b)):
            for i in range(3):
                tot[m][i] += s[i]
        pa, ra = prf(*a)
        pb, rb = prf(*b)
        print(f"{c:<13}{a[0] + a[2]:>4} | {pa:>9.2f}{ra:>7.2f} | {pb:>9.2f}{rb:>7.2f}")
    print("-" * len(hdr))
    for m in ("ruled", "open"):
        p, r = prf(*tot[m])
        print(
            f"{m.upper():<13}{tot[m][0] + tot[m][2]:>4} | precision {p:.2f}  recall {r:.2f}  "
            f"(tp={tot[m][0]} fp={tot[m][1]} fn={tot[m][2]})"
        )
    print("\ntop unmapped names from `open` (vocabulary gaps, fixable without the GPU):")
    for name, n in unmapped.most_common(25):
        print(f"  {n:>4}  {name}")
    print(f"\n{len(unmapped)} distinct unmapped names, {sum(unmapped.values())} mentions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
