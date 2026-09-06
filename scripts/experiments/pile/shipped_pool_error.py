#!/usr/bin/env python3
"""Pool error for the SHIPPED twelve, out of the negative pass (#3666).

#3588 reviewed thirteen *candidate* classes at 300 images each and reported a
``pool error`` column -- 0.0% to 7.1% -- the share of a class's own negatives
that a human found the object in. #3666 says the same number was never produced
for the twelve classes ``vg_scale`` actually ships, so adding the thirteen would
leave the benchmark with two tiers of label quality.

**It was produced, by the negative pass, and this script reads it out.** That
pass put 200 shared-pool images (100 uniform, 100 text-ranked) in front of one
reviewer twelve times over, and five of the shipped twelve were asked as their
own question -- ``clock``, ``book``, ``backpack``, ``umbrella``, ``stop sign``.
The other seven were asked inside a group (``bus`` and ``bicycle`` with ``car``
and ``truck``; ``knife`` with seven table objects; ``bird``, ``kite``, ``boat``
and ``dog`` together), and a group verdict is not a per-class rate: a *clean*
verdict is a negative for every member, but a *present* verdict names no member.

So the per-class answer for the seven costs an **attribution**, not another
pass, and the attribution is 14 images rather than the 840 uniform draws #3666
priced. :data:`ADJUDICATION` is that work: every find, what the object in it
actually is, and -- the part that turned out to matter -- whether the class's
own construction would ever have admitted it as a positive.

**That last column is the study's result.** ``clock`` reads the VG names
``clock``, ``clock face`` and ``clocks``; it does not read ``watch``. So the
wristwatch a reviewer correctly saw in image 2408671 could never have become a
``clock`` positive, and counting it as contamination of a ``clock`` negative
scores the pool against a definition the benchmark does not use. ``book`` runs
the other way: ``magazine`` is a shipped fold-in (COCO annotates magazines as
``book``), so the open magazine in 2327535 is contamination and the reviewer is
simply right. At a rate near 1% the ruling moves the estimate further than the
sample size does -- three clocks are 3.0 points, and 380 extra uniform draws per
class buy +/-1.0.

Usage::

    python shipped_pool_error.py                       # read the committed verdicts
    python shipped_pool_error.py --rebank              # re-distil them from the passes
    python shipped_pool_error.py --figures             # + the report's figures

``--rebank`` is what makes the committed CSV honest: it re-reads the detector
JSONs on scratch and in the running app's data dir and rewrites
``verdicts.csv``, so the archive can be checked against the source rather than
trusted. Everything else reads the CSV and needs neither.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pile_config as pc  # noqa: E402

#: Where #3588's review left its material. Scratch is purgeable, which is why
#: `--rebank` distils out of it into the committed CSV rather than the analysis
#: reading it directly.
BANK = Path("/expscratch/sgreenberg/classes-3588")
#: The running app's detector store. Two passes (`Backpack`, `Umbrella`) were
#: finished after the last banking run and exist only here.
LIVE = Path("/exp/sgreenberg/projects/VTSearch/data/detectors")
STUDY = Path("docs/experiments/2026-09-06-shipped-pool-3666")

#: Per-class contamination predicted by `pool_contamination.py` (#3635), under
#: the per-class exclusion rule, on the shipped ambiguous tables. Measured on
#: the VG-COCO overlap with COCO held back and *extrapolated* to the off-COCO
#: half; this pass is the first human test of that extrapolation per class.
PREDICTED_3635 = Path("/expscratch/sgreenberg/stopsign-3635/contam-sign.json")

#: What each find actually is, and whether the class's own construction admits
#: it. Adjudicated from the image on 2026-09-06 (crops in `figures/`); the
#: reviewer's own verdict is the `present` in the pass and is not overridden
#: here -- `admits` answers a different question, which nobody had asked:
#: *would this object have become a positive?*
#:
#: `admits` is read off `pile_config` and off what COCO's annotators actually
#: did -- never off English:
#:   yes          -- a VG name the class reads, or COCO's own box, covers it
#:   no           -- neither vocabulary admits it, so an image holding it is a
#:                   negative by construction and finding it cannot be an error
#:   unverifiable -- the pixels do not settle it
#:
#: **A fold-in count is not admission, and reading one as admission mis-ruled two
#: of these nine.** `coco_folds.py` (run over the twelve for #3673) shows COCO's
#: annotators landing `watch` on a COCO clock box 35 times, and `canopy` 32 +
#: `tent` 26 on umbrella boxes -- which looks like the `book`/magazine split, and
#: is not. Fold-in is a BOX test conditioned the wrong way round (#3618): the
#: question the pool asks is the image one, and `name_evidence.py` answers it.
#: On the images where `watch` is the ONLY evidence, COCO finds a clock 11% of
#: the time against a 4.5% base; `canopy` scores 7% and `tent` 10% against
#: umbrella's 3.7%, all far under the 1/3 cut and all verdict `neither`. The
#: fold-in tail is COCO's own inconsistency, not a definition, so both are `no`.
ADJUDICATION: dict[int, dict] = {
    # ---- asked as their own question -------------------------------------
    2408671: {
        "cls": "clock",
        "what": "a wristwatch on a bystander's wrist, at a skate spot",
        "crop": (0.60, 0.55, 0.80, 0.95),
        "admits": "no",
        "why": "`watch` is not in clock's names or its ambiguous list, and it was measured "
        "for this study rather than assumed: over the 970 overlap images where `watch` is "
        "the SOLE evidence, COCO finds a clock 11% of the time (Wilson lower 0.09) against "
        "a 4.5% base and a 1/3 cut, with 3% box agreement -- verdict `neither`. The 35 "
        "COCO clock boxes a VG `watch` box lands on are the tail, not the rule",
    },
    2393325: {
        "cls": "clock",
        "what": "an analog clock WIDGET drawn on a computer monitor's desktop",
        "crop": (0.70, 0.10, 0.85, 0.28),
        "admits": "no",
        "why": "a depiction, and #3588's guide already rules on those for every class -- "
        "`vote on the object, not a depiction of it`, which it applies to a car on a "
        "billboard and cutlery printed on a menu. COCO annotated the image and listed "
        "no clock",
    },
    2392807: {
        "cls": "clock",
        "what": "the digital time on a railway platform's departure board",
        "crop": (0.02, 0.18, 0.26, 0.44),
        "admits": "no",
        "why": "not a clock face; VG names a departure board `sign`, `board` or `display`, "
        "none of which clock reads, and `display` scores 2% repair precision over 338 sole "
        "images. COCO listed no clock on this exhaustive image",
    },
    2327535: {
        "cls": "book",
        "what": "an open magazine on the desk beside a laptop",
        "crop": (0.00, 0.15, 0.42, 0.55),
        "admits": "yes",
        "why": "`magazine` is a SHIPPED fold-in for book -- COCO has no magazine class "
        "and annotates magazines as book -- so this is contamination, not a "
        "boundary call",
    },
    1593184: {
        "cls": "book",
        "what": "a printed booklet standing in an open box on the kitchen floor",
        "crop": (0.00, 0.20, 0.30, 0.52),
        "admits": "yes",
        "why": "same fold-in; small and off to one side, which is what a uniform draw finds and a ranked one does not",
    },
    2368984: {
        "cls": "backpack",
        "what": "a black backpack worn on the back of a passenger boarding an aircraft",
        "crop": (0.20, 0.45, 0.62, 0.72),
        "admits": "yes",
        "why": "unambiguous, and the only find of the nine that needs no ruling at all",
    },
    2315796: {
        "cls": "backpack",
        "what": "a pack or back-protector under a motorcyclist's leathers, in a showroom",
        "crop": (0.50, 0.05, 1.00, 0.40),
        "admits": "unverifiable",
        "why": "the shape is a rider's hump; `black bag` is on backpack's ambiguous list, "
        "so a VG-named one would have left the pool anyway",
    },
    2398287: {
        "cls": "umbrella",
        "what": "square pop-up canopy tents along the rail of a skate park",
        "crop": (0.55, 0.25, 1.00, 0.52),
        "admits": "no",
        "why": "umbrella reads `parasol` and four umbrella spellings and no canopy or tent "
        "name, and both were measured: `canopy` scores 7% repair precision over 225 sole "
        "images and `tent` 10% over 265, against a 3.7% base -- verdict `neither` for both, "
        "as for `awning` (4%) and `shade` (1%). COCO's 32 canopy and 26 tent boxes on "
        "umbrella boxes are a box-level tail the image test refutes",
    },
    2343839: {
        "cls": "stop sign",
        "what": "the blank aluminium BACK of a sign, on the pole carrying the street signs",
        "crop": (0.18, 0.62, 0.52, 1.00),
        "admits": "unverifiable",
        "why": "a sign seen from behind has no shape to read. COCO's annotators do box "
        "them -- VG `back` lands on a COCO stop-sign box 11 times, 1.1% of the class -- "
        "so the question is the pixels, not the vocabulary; `sign` is deliberately not "
        "listed for this class (#3618) and COCO listed no stop sign here",
    },
    # ---- asked inside a group: the attribution the group verdict owes ------
    498270: {
        "cls": "boat+dog",
        "what": "a dog sitting on the deck of a pontoon boat",
        "crop": (0.00, 0.00, 1.00, 1.00),
        "admits": "yes",
        "why": "two shipped classes in one image, both filling a large share of the frame",
    },
    2417852: {
        "cls": "boat",
        "what": "a pedal boat carrying three people down a river",
        "crop": (0.00, 0.00, 1.00, 1.00),
        "admits": "yes",
        "why": "unambiguous and large",
    },
    2323658: {
        "cls": "dog",
        "what": "a basset hound lying on a beach, filling a third of the frame",
        "crop": (0.00, 0.00, 1.00, 1.00),
        "admits": "yes",
        "why": "unambiguous",
    },
    2394898: {
        "cls": "dog",
        "what": "a dog riding a surfboard in the shore break",
        "crop": (0.00, 0.00, 1.00, 1.00),
        "admits": "yes",
        "why": "unambiguous",
    },
    2417678: {
        "cls": "bird",
        "what": "distant birds on the tideline behind a man catching a frisbee",
        "crop": (0.45, 0.20, 1.00, 0.50),
        "admits": "yes",
        "why": "small but real; the frisbee is not a kite, which is what the ranker was chasing when it drew this row",
    },
    4185: {
        "cls": None,
        "what": "parked cars on a Paris street, under a blue BUS-LANE sign",
        "crop": (0.55, 0.28, 0.90, 0.60),
        "admits": "no",
        "why": "the Vehicles group holds two candidates and two shipped classes; this "
        "image's vehicles are cars, and the bus is a pictogram on a sign -- the "
        "same failure mode as the bike-crossing signs in #3588",
    },
    2399434: {
        "cls": None,
        "what": "feed buckets in front of cattle at a fence",
        "crop": (0.00, 0.50, 0.60, 1.00),
        "admits": "no",
        "why": "a Table Objects find, attributable to `bowl` (a candidate); no knife",
    },
    2411445: {
        "cls": None,
        "what": "potted plants and a basin of greens on a vendor's motorbike",
        "crop": (0.00, 0.00, 1.00, 1.00),
        "admits": "no",
        "why": "attributable to `vase` under the pots-and-planters merge; no knife",
    },
    2413407: {
        "cls": None,
        "what": "doughnuts on a plate on a high-chair tray",
        "crop": (0.00, 0.40, 1.00, 1.00),
        "admits": "no",
        "why": "a plate is a `bowl` under the shipped merge; no knife",
    },
    2320159: {
        "cls": None,
        "what": "a slice of wedding cake held on a plate",
        "admits": "no",
        "why": "plate again, and COCO scores this image and calls it a chair image; "
        "no knife on the plate or in the frame",
    },
    1592013: {"cls": None, "what": "seeded from COCO (bottle)", "admits": "no", "why": "seed row"},
    2315792: {
        "cls": None,
        "what": "a bench, which COCO also boxes",
        "admits": "no",
        "why": "COCO's exhaustive answer for this image is `bench` and no knife, so the "
        "find is a candidate, not one of the twelve",
    },
    2383506: {
        "cls": None,
        "what": "a Table Objects find on a COCO-scored image COCO calls empty",
        "admits": "no",
        "why": "no shipped member of that group (knife) is visible",
    },
    1159420: {"cls": None, "what": "COCO boxes a truck", "admits": "no", "why": "candidate class"},
    2325332: {"cls": None, "what": "COCO boxes a car and a truck", "admits": "no", "why": "candidate class"},
    2384924: {"cls": None, "what": "COCO boxes a truck", "admits": "no", "why": "candidate class"},
}


def wilson(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """``(lower, upper)`` Wilson bounds -- the instrument #3635 and #3618 use.

    Both ends: at these rates the interesting claim is usually that a rate is
    SMALL, which a lower bound cannot make.
    """
    if n <= 0:
        return 0.0, 0.0
    p = hits / n
    z2 = z * z
    centre = p + z2 / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return max(0.0, (centre - half) / (1 + z2 / n)), min(1.0, (centre + half) / (1 + z2 / n))


def rebank(out: Path) -> None:
    """Distil every finished pass into one CSV the analysis can live on.

    The passes are detector JSONs in two places, under two polarities, with the
    COCO-seeded rows mixed in. All three are read here so that nothing
    downstream has to know about them:

    * ``polarity.json`` records which detectors mean Good = *clean* and which
      mean Good = *present*. The pass flipped mid-review; reading a banked file
      without it inverts nine of the twelve.
    * ``seeded.json`` records the rows `seed_pos.py` wrote from COCO to stop the
      trainer starving. They are the reference, not a judgement, and scoring
      them as finds would be circular.
    * a pass finished after the last banking run exists ONLY in the app's
      detector store, which is why both directories are read and the live copy
      wins.
    """
    pol = json.loads((BANK / "polarity.json").read_text())
    old, new = pol["old"]["detectors"], pol["new"]["detectors"]
    seeded = json.loads((BANK / "seeded.json").read_text())
    man = {int(r["image_id"]): r for r in csv.DictReader((BANK / "slates/Table_Objects/manifest.csv").open())}

    rows, seen = [], set()
    for p in sorted(list(LIVE.glob("*.json")) + list((BANK / "negbank").glob("*.json"))):
        body = json.loads(p.read_text())
        name, labels = body.get("name"), body.get("labelset", {}).get("labels", [])
        if not name or len(labels) < 200 or name in seen:
            continue
        seen.add(name)
        members = old.get(name) or new.get(name) or ()
        seeds = set(seeded.get(name, {}).get("seeded", []))
        for lb in labels:
            iid = int(Path(lb["origin_name"]).stem)
            present = (lb.get("label") != "good") if name in old else (lb.get("label") == "good")
            rows.append(
                {
                    "pass": name,
                    "members": "|".join(members),
                    "image_id": iid,
                    "present": int(present),
                    "seeded": int(iid in seeds),
                    "stratum": man[iid]["stratum"],
                    "driver": man[iid]["driver"],
                    "exhaustive": man[iid]["exhaustive"],
                }
            )
    rows.sort(key=lambda r: (r["pass"], r["image_id"]))
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[rebank] {len(seen)} passes, {len(rows)} judgements -> {out}")


def load(path: Path) -> tuple[dict, dict, dict]:
    """``(passes, strata, exhaustive)`` out of the committed CSV."""
    passes: dict[str, dict] = {}
    strata: dict[int, str] = {}
    exh: dict[int, bool] = {}
    for r in csv.DictReader(path.open()):
        p = passes.setdefault(r["pass"], {"members": tuple(r["members"].split("|")), "present": set(), "seeded": set()})
        strata[int(r["image_id"])] = r["stratum"]
        exh[int(r["image_id"])] = r["exhaustive"] == "yes"
        if int(r["seeded"]):
            p["seeded"].add(int(r["image_id"]))
        elif int(r["present"]):
            p["present"].add(int(r["image_id"]))
    return passes, strata, exh


def candidate_pool_error() -> dict[str, tuple[int, int]]:
    """#3588's own column, recomputed from its verdicts rather than transcribed.

    The comparison this study exists to make is *shipped against candidate*, and
    a number retyped out of a table is not evidence about the table.
    """
    src = BANK / "verdicts_20260904.json"
    if not src.exists():
        return {}
    per: dict[str, list] = defaultdict(list)
    for v in json.loads(src.read_text()):
        if v.get("stratum") == "random":
            per[v["class"]].append(v)
    return {c: (sum(1 for v in vs if v["human"] == "present"), len(vs)) for c, vs in per.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verdicts", default=str(STUDY / "verdicts.csv"))
    ap.add_argument("--rebank", action="store_true", help="re-distil verdicts.csv from the passes on scratch")
    ap.add_argument("--figures", action="store_true", help="also emit the report's figures")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    vpath = Path(args.verdicts)
    if args.rebank:
        vpath.parent.mkdir(parents=True, exist_ok=True)
        rebank(vpath)

    passes, strata, exhaustive = load(vpath)
    rand = {i for i, s in strata.items() if s == "random"}
    bound = {i for i, s in strata.items() if s == "boundary"}
    pred = json.loads(PREDICTED_3635.read_text())["classes"] if PREDICTED_3635.exists() else {}

    # Which pass carries each shipped class, preferring one that asked about it
    # alone. A group pass answers a weaker question and is labelled as such.
    carrier: dict[str, str] = {}
    for c in pc.SCALE_CLASSES:
        opts = [n for n, v in passes.items() if c in v["members"]]
        solo = [n for n in opts if len(passes[n]["members"]) == 1]
        carrier[c] = solo[0] if solo else (opts[0] if opts else "")

    # The adjudication, indexed the way the tables need it.
    adj_for: dict[str, set[int]] = defaultdict(set)
    for iid, a in ADJUDICATION.items():
        for c in (a["cls"] or "").split("+"):
            if c:
                adj_for[c].add(iid)
    admits = {i: a["admits"] for i, a in ADJUDICATION.items()}

    print("=" * 104)
    print("POOL ERROR FOR THE SHIPPED TWELVE -- the uniform stratum is the estimator; the")
    print("ranked one is chosen to be wrong and is reported beside it as an existence proof.")
    print("`as read` is the reviewer's verdict. `admissible` keeps only finds the class's own")
    print("construction would have made a positive -- see ADJUDICATION.")
    print("=" * 104)
    hdr = ("class", "asked as", "as read", "95% CI", "admissible", "ranked", "#3635")
    print("%-11s %-13s %10s %14s %13s %7s %8s" % hdr)
    print("-" * 104)
    table = {}
    for c in pc.SCALE_CLASSES:
        v = passes[carrier[c]]
        kind = "per-class" if len(v["members"]) == 1 else f"group/{len(v['members'])}"
        # As read: for a per-class pass every find is this class's. For a group
        # pass only the adjudicated attribution is.
        finds = v["present"] if len(v["members"]) == 1 else adj_for.get(c, set()) & v["present"]
        kr = len(finds & rand)
        kb = len(finds & bound)
        ok = len({i for i in finds & rand if admits.get(i) == "yes"})
        maybe = len({i for i in finds & rand if admits.get(i) == "unverifiable"})
        lo, hi = wilson(kr, len(rand))
        p3635 = 100 * pred.get(c, {}).get("per_class", {}).get("rate", float("nan")) if pred else float("nan")
        print(
            "%-11s %-13s %5d/%-4d %5.1f%% [%3.1f,%4.1f] %11s %7d %7.2f%%"
            % (
                c,
                kind,
                kr,
                len(rand),
                100 * kr / len(rand),
                100 * lo,
                100 * hi,
                f"{ok}" if not maybe else f"{ok}-{ok + maybe}",
                kb,
                p3635,
            )
        )
        table[c] = {
            "pass": carrier[c],
            "kind": kind,
            "random_hits": kr,
            "random_admissible": ok,
            "random_unverifiable": maybe,
            "random_n": len(rand),
            "ranked_hits": kb,
            "ci": [lo, hi],
            "predicted_3635": p3635 / 100,
            "expected_false_negatives": kr / len(rand) * pc.SCALE_N_NEG,
        }

    solo = [c for c in pc.SCALE_CLASSES if table[c]["kind"] == "per-class"]
    kr = sum(table[c]["random_hits"] for c in solo)
    ok = sum(table[c]["random_admissible"] for c in solo)
    n = len(solo) * len(rand)
    lo, hi = wilson(kr, n)
    p3635 = sum(table[c]["predicted_3635"] for c in solo) / len(solo)
    print("-" * 104)
    maybe = sum(table[c]["random_unverifiable"] for c in solo)
    print(
        f"pooled over the {len(solo)} asked per-class: as read {kr}/{n} = {100 * kr / n:.2f}% "
        f"[{100 * lo:.2f},{100 * hi:.2f}], admissible {ok}-{ok + maybe}/{n} = "
        f"{100 * ok / n:.2f}-{100 * (ok + maybe) / n:.2f}%; #3635 predicts {100 * p3635:.2f}%"
    )

    # ---- the union, and the two halves of the pool it is drawn from ---------
    # `vg_scale`'s pool is built by anchoring to COCO where COCO has an answer,
    # so on that half a shipped class's absence is COCO's own statement and on
    # the other half it is VG's silence. Splitting the union by that line is the
    # only way to see which half the contamination actually lives in -- and the
    # split is what says whether the next slate should be drawn from the whole
    # pool or only from the unverified half.
    print()
    union: set[int] = set()
    for c in pc.SCALE_CLASSES:
        v = passes[carrier[c]]
        union |= v["present"] if len(v["members"]) == 1 else adj_for.get(c, set()) & v["present"]
    yes = {i for i in union if admits.get(i) == "yes"}
    unv = {i for i in union if admits.get(i) == "unverifiable"}
    scored = {i for i, e in exhaustive.items() if e}
    for label, frame in (
        ("whole uniform stratum", rand),
        ("COCO-scored half", rand & scored),
        ("off-COCO half", rand - scored),
    ):
        n_f = len(frame)
        lo_f, hi_f = wilson(len(union & frame), n_f)
        print(
            f"union of the twelve, {label:<22} {len(union & frame):2d}/{n_f:<3d} = {100 * len(union & frame) / n_f:4.1f}% "
            f"[{100 * lo_f:.1f},{100 * hi_f:.1f}]   admissible "
            f"{len(yes & frame)}-{len(yes & frame) + len(unv & frame)}/{n_f} = "
            f"{100 * len(yes & frame) / n_f:.1f}-{100 * (len(yes & frame) + len(unv & frame)) / n_f:.1f}%"
        )
    print(
        f"  -> {len(union & rand & scored)} of the {len(union & rand)} uniform finds sit on a COCO-scored image, where the "
        "pool label IS COCO's;\n     a find there is the reviewer's English against COCO's boxes, not an "
        "inconsistency in the benchmark."
    )

    cand = candidate_pool_error()
    if cand:
        ch, cn = sum(h for h, _ in cand.values()), sum(n for _, n in cand.values())
        clo, chi = wilson(ch, cn)
        print(
            f"#3588's thirteen candidates, same statistic: {ch}/{cn} = {100 * ch / cn:.2f}% "
            f"[{100 * clo:.2f},{100 * chi:.2f}]  ({len(cand)} classes x 70 uniform draws)"
        )
        # A two-proportion difference, not an interval-overlap eyeball: #3666's
        # claim is about the GAP between the tiers, and overlapping intervals
        # are neither necessary nor sufficient for that gap to be zero.
        p1, p2 = kr / n, ch / cn
        se = math.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / cn)
        print(
            f"  shipped - candidate = {100 * (p1 - p2):+.2f} +/- {100 * 1.96 * se:.2f} pp (95%) -> "
            + ("the two tiers are NOT separable" if abs(p1 - p2) < 1.96 * se else "SEPARABLE")
        )

    print("\nWHAT A LARGER SAMPLE BUYS, against what a ruling buys")
    for p in (0.01, 0.02):
        for w in (0.01, 0.005):
            print(
                f"  p={100 * p:.0f}%, +/-{100 * w:.1f}pp needs {1.96**2 * p * (1 - p) / w**2:>6,.0f} uniform draws per class"
            )
    swing = 100 * (table["clock"]["random_hits"] - table["clock"]["random_admissible"]) / len(rand)
    print(f"  one ruling on `clock` (is a wristwatch a clock?) moves it {swing:.1f}pp, for one sentence")

    if args.out:
        Path(args.out).write_text(json.dumps({"classes": table, "candidates": cand}, indent=1) + "\n")
        print(f"\nwrote {args.out}")
    if args.figures:
        import figures_3666  # noqa: PLC0415  (optional, and it imports matplotlib)

        figures_3666.build(table, ADJUDICATION, STUDY / "figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
