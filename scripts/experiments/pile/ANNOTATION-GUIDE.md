# Annotation guide — the twelve shipped `vg_scale` classes (#3673)

The reviewer's document for `clock`, `bird`, `boat`, `umbrella`, `kite`, `book`,
`dog`, `backpack`, `knife`, `bicycle`, `bus` and `stop sign` — the classes
`vg_scale` ships today.

**Where the rules actually live.** Each class's Good/Bad wording is
`pile_config.SCALE_CLASS_RULES[cls].test`, and its short form is the `name` the
slate's dataset and detector carry, because that name is the only string a
reviewer sees while voting. This file is the long form: the same rulings with
the measurements that decided them, plus the boundaries that run *between*
classes and belong to no single entry. If the two ever disagree, `pile_config`
is the one the slate maker reads — fix this file.

The thirteen #3588 candidates have their own guide,
[`docs/experiments/2026-09-03-vg-scale-classes/ANNOTATION_GUIDE.md`](../../../docs/experiments/2026-09-03-vg-scale-classes/ANNOTATION_GUIDE.md),
which is also where the **protocol** lives — strata, boxing, the oversize rule,
cannot-tell → Good, depiction, toy, part-of-the-whole. All of it applies here
unchanged and is not repeated. The two guides merge when the class lists do
(#3604).

## Why this exists

#3666 measured the twelve's pool error for the first time and found the number
was not the problem: **six of the nine finds were boundary calls on rules that
did not exist.** A wristwatch. A clock drawn on a monitor. A railway departure
board. A rank of pop-up canopies. The blank back of a sign. At a ~1% rate, one
ruling moves a class further than 3,000 extra uniform draws would — `clock` went
from 3.0% to 0.0% on a single sentence about watches.

So the rulings are worth more than the sampling, and they are cheap: every one
below was decided from data that already existed, before any human labelled
anything.

## Two measurements decide a rule, and the cheap one gets it wrong

Both come from the ~51k images that are in **both** VG and COCO, where the two
vocabularies annotate the same pixels.

| test | script | asks | what it settles |
|---|---|---|---|
| **fold-in** (box) | `coco_folds.py` | which VG names land on a COCO box of the class | what a reviewer on COCO's reading must **accept** |
| **repair precision** (image) | `name_evidence.py` | where a VG name is the **only** evidence, does COCO find the class? | whether the name means the class at all |

**Run only the first and you will mis-rule.** It says COCO's annotators call a
wristwatch a `clock` **35** times, and a `canopy` **32** or a `tent` **26** an
`umbrella` — together more than `parasol`'s 38. Both look like the
`book`/magazine split that cost #3588 a whole pass. Neither is:

| name | class | sole images | COCO finds the class | base | verdict |
|---|---|---:|---:|---:|---|
| `watch` | clock | 970 | **11%** | 4.5% | `neither` |
| `display` | clock | 338 | 2% | 4.5% | `neither` |
| `tent` | umbrella | 265 | **10%** | 3.7% | `neither` |
| `canopy` | umbrella | 225 | **7%** | 3.7% | `neither` |
| `awning` | umbrella | 446 | 4% | 3.7% | `neither` |
| `shade` | umbrella | 634 | 1% | 3.7% | `neither` |

A fold-in tail is COCO's own inconsistency. **A box test says whether a name's
box IS the object; an image test says whether the object is THERE, and only the
second one prices a negative pool** (#3618).

## Definition risk, before anyone labels

The share of a class's VG boxes landing on **no** COCO class, over images COCO
annotated exhaustively. High means the VG name covers things COCO has no word
for — which is exactly how `book` broke.

| class | VG boxes | on its own COCO box | on **no** COCO class |
|---|---:|---:|---:|
| **`book`** | 1,524 | 56.6% | **43.1%** |
| `boat` | 2,614 | 71.0% | 28.5% |
| `umbrella` | 2,894 | 71.4% | 28.2% |
| `bird` | 2,427 | 73.3% | 25.9% |
| `kite` | 1,992 | 73.8% | 25.7% |
| `clock` | 2,397 | 76.3% | 23.3% |
| `stop sign` | 250 | 76.4% | 22.0% |
| `backpack` | 1,152 | 75.6% | 19.8% |
| `knife` | 1,015 | 80.6% | 18.0% |
| `bicycle` | 933 | 82.2% | 15.0% |
| `dog` | 1,865 | 87.7% | 9.9% |
| `bus` | 2,318 | 86.9% | 8.6% |

`book` calibrates the column at 43%, and the mechanical floor is ~7–15%: some
share of any name's boxes misses COCO's for reasons that have nothing to do with
vocabulary (a box drawn round the shelf rather than the volume, an object COCO
simply did not annotate). So read the column as an ordering of attention, not as
a rate of anything — and note that **`dog` and `bus` sit on the floor**, which
is why `dog` is the one class of the twelve with no rule: its name is the whole
question.

## The rulings, in one line each

The full wording is in `SCALE_CLASS_RULES`; these are the discriminations.

- **Clock** — a device whose job is showing the time and which stands, hangs or
  is mounted. **Not a wristwatch**, not a departure board, not a clock drawn on
  a screen.
- **Bird** — any live bird of any species. **Not a cooked one**: in VG,
  `chicken` (428 images, 10%) and `turkey` (53, 12%) are usually food, and
  `crane` (308, 2%) is a machine.
- **Boat** — anything built to travel on water. On a trailer still counts. Not a
  surfboard, which is COCO's own class.
- **Umbrella** — one central pole carrying a round canopy: hand-held, parasol,
  beach, patio. **Four legs or a wall fixing is a canopy, a tent or an awning**,
  and none of those.
- **Kite** — a kite, **and a parasail, paraglider or parachute**, which is
  COCO's reading and already folded in (`parasail` 57 boxes, `parachute` 26).
  Not a flag, a balloon or a windsock.
- **Book** — **is it bound along a spine?** Bound is a book, and that includes
  magazines and notebooks. Folded or loose sheets are not: newspapers, menus,
  posters, printouts.
- **Dog** — a domestic dog of any breed, puppies included. A hot dog is not one
  (8 boxes in the overlap, and it is the largest member of `dog`'s head-noun
  family at 405 images — a trap for a *name*, not for an eye).
- **Backpack** — carried on the back on two shoulder straps. Not a handbag, a
  shoulder bag or a suitcase; COCO carries those as their own classes.
- **Knife** — a bladed cutting or spreading implement, servers included. Not
  scissors (COCO's own class), not a spatula, and not a `silverware` box
  covering a whole place setting.
- **Bicycle** — human-powered pedal cycle; a tricycle counts. **Not a
  motorcycle**, and not a pictogram on a road sign even when COCO boxed one.
- **Bus** — carries passengers in rows, boarded through its own door. Not a tram
  on rails, not a cargo van, not an RV.
- **Stop sign** — the octagonal red one, from behind only if the octagon reads.
  No other sign, and no pictogram.

## Boundaries that run between two of the twelve

- **Bus against truck against car.** The full three-test version is in the
  thirteen's guide under `truck`, and it is the same one: is it self-propelled;
  does the body carry a load or perform work; was it built for goods or for
  people. A **minibus** is boarded through its own door rather than entered by
  row, which makes it a Bus; a **cargo van** is a Truck. `van` is genuinely
  three vehicles — COCO splits it 261 truck / 318 car / 37 bus — so disagreement
  there is a known cost, not a mistake.
- **Bicycle against bus.** Not a boundary. It is listed because the negative
  pass's one unattributed Vehicles find was a Paris street under a blue
  **bus-lane pictogram**, and a pictogram is neither.
- **Knife against fork against spoon.** Judge the object, not the drawer, and
  vote Good only when the boxed thing IS the utensil. A `silverware` or
  `utensil` box covering a place setting is Bad for all three.
- **Bird against kite.** They co-occur constantly on beaches and both are small
  and airborne. VG `kite` boxes land on COCO `bird` 3 times and VG `bird` boxes
  on COCO `kite` 10 — noise, but the two are the classes most often mistaken for
  each other at distance. Read the line: a kite has one, a bird does not.
- **Boat against kite.** `parasail` folds into `kite`, and a parasail is towed
  by a boat. Both are present in that scene; box the one you are asked about.

## What is still owed

- **The positives have never been reviewed** — 360 pre-boxed COCO positives, 30
  per class, which is the other half of the two-tier gap (#3674).
- **`bicycle` is missing roughly half its positives** on the non-COCO half,
  because it is built from the spelling `bicycle` while `bike` carries 638 of
  COCO's 3,683 boxes (#3605).
- **Four names measured well here and are not in any table**: `yacht`, `ferry`,
  `rowboat` and `barge` for `boat`, at 88–100% precision on small samples. They
  are candidates for `SCALE_VG_NAMES`, which is a build change and needs the
  #3618 gate run at its own sample sizes, not this guide's word.
