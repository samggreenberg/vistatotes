# `vg_scale` — one class list across three box-size bands

A dataset for asking **"how well can we find buses in the middleground?"** — the
same twelve classes at three object scales, so a small-vs-large difference is
about size rather than about which words happen to live at which size.

It exists because the published `vg_box_small/medium/large` sets cannot answer
that question: each category is banded by its *median* box area, so the three
sets carry disjoint vocabularies and their gap confounds box size with class
identity (`nose`/`glasses`/`watch` against `fence`/`hill`/`lady`). Those sets
remain valid for what they measured and are **not** comparable to this one.

## What it is

| | |
|---|---|
| medias | ~7,700 VG images, no pixels stored (vectors + `patch_grid` only) |
| cells | 36 = 12 classes × {small, medium, large} |
| positives | exactly **100 per cell**, each carrying its ground-truth box |
| negatives | one shared pool of **3,900**, identical for every cell |
| prevalence | **designed 0.0250** in all 36 cells; **realised 0.0172** since #3667 admitted other classes' positives as negatives (evaluable 4,000 → 5,806 per cell). `SCALE_PREVALENCE` is the designed number and every k\* is computed from it (#3681). |
| embedders | `siglip`, `siglip2_l`, `clip`, `clip_l`, `dinov3_patch` — five columns, all built from the same medias (`clip_l` is eval-only, not offered in the app) |
| region arm | the **pair** `siglip+dinov3_patch`: DINOv3 carries the patch grids that make region voting real, SigLIP carries the text tower the run opens on. Bare `dinov3_patch` is a *column of the pile*, never an arm of a study — with no text tower it cannot open the way the app does (#3276). |

Classes: `backpack` `bicycle` `bird` `boat` `book` `bus` `clock` `dog` `kite`
`knife` `stop sign` `umbrella` — every one also a COCO-2017 class, which is what
made the correction pass affordable.

**Bands** are anchored to the patch embedder's geometry, as fractions of image
area: `small` < 1/196 (below one DINOv3 patch), `medium` 1/196–1/12 (patch to
smallest HAC leaf), `large` 1/12–0.80. Size means the **union box** over a
class's instances — what one Good vote actually drags — and an image whose union
is scattered far wider than its largest instance is excluded rather than banded,
because there the box describes the scatter and not the object.

**Cells are designated, not inferred.** Each is exactly its 100 positives plus
the shared negatives; everything else is *excluded* from it. That third value is
the point: an image holding a large bus is not a `bus@small` positive, and
calling it a negative would penalise a detector for finding a real bus. Consumers
must honour it via `vtscore.eval.labels.evaluable_pool` — the harness does this
once per cell, and a pool built without it silently scores excluded images as
negatives.

## Where the labels come from, and what they are worth

VG's own annotation cannot support this construction: measured against COCO, its
recall over these twelve classes is **0.61**, and **1.35%** of the images it
treats as negatives actually hold the object. So labels are VG's, repaired:

- **48% of images are COCO-sourced** (`image_data.json`'s `coco_id`) and take
  COCO's exhaustive annotation, which replaces VG's for that image. Copies whose
  aspect ratio disagrees with the COCO original by >1% are left unrepaired: 49 of
  51,497 are re-cropped or rotated, and there a COCO box describes the wrong
  pixels.
- **The rest were reviewed by hand**, in VTSearch, across three passes: ranked
  negatives, a uniform random stratum, and every positive re-issued with its box
  drawn and a magnified inset.

### The numbers a reader should hold

| measured on | result |
|---|---|
| residual contamination of the negative pool, after review | **2.0%** (4/200), 95% CI **0.8–5.0%** |
| — concentrated in | `book` (3/20) and `bus` (1/20); zero in the other eight |
| small-band positives confirmable *with the box drawn* | **~2/3** |
| reviewer vs COCO on pairs COCO had settled | 9.0% disagreement — reviewer error, COCO error and definition drift, **not attributable without adjudication** |

**COCO is not a gold standard here.** The review found images COCO annotates as
empty that plainly hold the object, and four adjudicated COCO errors among the
twelve classes (two prohibition circles and a school-crossing paddle labelled
`stop sign`; a box on a hedge labelled `umbrella`).

**`bicycle` is built from one VG spelling, and the published pickle still is.**
VG names objects in free text and the builder matched an object's primary name
only, so a bicycle annotated `bike` was never a `bicycle` positive — and on the
non-COCO half, where VG's silence is the only evidence of absence, it was a
`bicycle` **negative**. Over the 51,411-image VG∩COCO overlap `bike` carries
**638** of COCO's 3,683 `bicycle` boxes against the `bicycle` spelling's 775, so
roughly half the class's positives on that half are missing and its negative pool
holds the ones it missed (#3605). The builder now withholds `bike` images from
both — `bike` cannot simply be merged, since only 40.1% of its boxes land on a
COCO `bicycle` and it is a measured alias of `motorcycle` too — but **the
published cells predate that**, so any per-class reading of `bicycle` in the
#3156 grid carries it.

**All twelve are built from one VG spelling, and the published pickle still is.**
#3618 measured the other eleven and every one of them had something: 32 spellings
now fold onto their class and 50 more are withheld from it, which on the 56,579
VG images COCO does not annotate **repairs 860 images that were negatives for
their own class** and withholds 2,664 more. `bird` gains the most (+18% on the
images it could already see), then `book` (+12%) and `boat` (+11%); `stop sign`
gains nothing, because the VG name carrying 46.6% of its COCO boxes is `sign`,
which is a stop sign 7.9% of the time (#3635). **None of this is in the published
cells** — the tables were empty when they were built — so a per-class reading of
any of the twelve carries the same defect `bicycle` does, in smaller measure. See
[`2026-09-04-vg-name-coverage/`](../2026-09-04-vg-name-coverage/REPORT.md).

**A class's definition is part of its label, and it now has a home.** A reviewer
votes on bare images — files are named by image id — so the dataset name is the
whole brief, and for a class whose plain English name does not settle the
question that name is where the rule has to live. `book` is what taught this:
COCO has no magazine class and annotates magazines as `book`, the human pass
applied the narrower English reading, and 21 verdicts landed on one definition
against 49 on another. The wordings now live in
`pile_config.SCALE_CLASS_RULES` and every slate maker builds its dataset name
from them, so a first pass and a re-review of one class ask the same question
(#3612). Only `book` of the twelve carries a rule; the published cells were
reviewed before the table existed.

**The small band is at the limit of verification, and this is a property of the
data, not a defect to hide.** A sub-patch object is under 1/196 of the frame;
reviewing bare thumbnails rejected 43% of small-band positives against 10% of
large, and drawing the box cut that to 18% vs 3%. Both a human and a
vision-language model confirm only ~2/3 of them even with the box. So a
small-band "not confirmed" is recorded as *unconfirmed* and the label stands —
**any small-band result should be read beside that fact.**

## What this dataset can and cannot be asked

The sections above say what the data *is*. This one says what a study may
**conclude** from it, because the two are not the same and the difference has
cost real GPU hours. Every verdict below points at the measurement that licenses
or blocks it; a question shape that is not listed has not been thought about, and
that is itself the finding.

The organising fact: **label error here is not uniform, and the design cancels it
in exactly one direction.** Cells within a class share their negatives, their
prevalence and their positives' provenance, so a contrast *across bands of one
class* is paired and the noise is common. Nothing about the construction makes
the noise comparable *across classes*, and nothing makes it harmless in a
*sequential* run, where a wrong label does not average out but steers every
acquisition after it.

| question shape | verdict | what decides it |
|---|---|---|
| Does target size change cost, **within one class**? | **supported** — the construction exists for it | Same 3,900 negatives and the same prevalence in every band of a class, and only the positives' *size* differs. But the published #3156 effect is a **lower bound**: the pre-#3667 pool let a head learn scene context, 2.50 ± 0.42× at `@small` against 1.25 ± 0.20× at `@large` (#3679). |
| **Method A vs method B**, one class, one band | **supported, with a stated floor** | Contamination is common to both arms, but its *cost* is not: a hidden positive among the negatives is expensive only for the method that ranks it highly. Do not call a difference smaller than that class's contamination rate (table below). |
| Method A vs B where the two are **confused by different things** | **unmeasured** | Since #3667 a class's negatives include images holding *other* classes, so the pool's co-occurrence structure is set by COCO's scene statistics rather than by design. Nobody has priced it. `cooccur.py` / `cooccur25.py` measure co-occurrence but were written to group review passes, not to price this confound. |
| Is class **X** harder than class **Y**? | **not currently supported** | Per-class label quality varies more than most task effects: pool contamination spans ~10× (`backpack` 2.8% → `kite`/`dog` ~0.3%, `pool_contamination.py`), definition risk ~5× (`book` 43.1% of VG boxes on no COCO class → `bus` 8.6%, #3673), and eleven of twelve classes had no written definition when they were reviewed — `SCALE_CLASS_RULES` now covers all twelve, but it postdates the review (#3612, #3673, #3771). A class ranking reads that spread as difficulty. |
| Absolute cost / AP **for one class** | **supported, with a stated floor** | Quote the class's own contamination rate beside the number, not the pooled 1.4%. |
| Anything about `@small` **in absolute terms** | **read beside the verification limit** | A sub-patch object is under 1/196 of the frame; both a human and a VLM confirm only **~2/3** of small-band positives *with the box drawn*. Small-band "not confirmed" is recorded as unconfirmed and the label stands. |
| Does more labelling monotonically help? Any claim about a **run's trajectory** | **unmeasured, and structurally hazardous** | A mislabelled image in a sequential loop is not additive error: it enters the training set and steers later acquisition. Worse, the loop samples contamination *adversarially* — hidden positives are by construction the most class-looking negatives available, so they are served early, when the head has fewest positives to outvote them. Unquantified; see the flip probe (#3686). |
| Compare a result here against `vg_box_small/medium/large` | **not supported** | Disjoint vocabularies; the gap confounds box size with class identity. Stated at the top of this datasheet. |
| Any **per-class** reading of `bicycle` (or, in smaller measure, the other eleven) in the published cells | **carries a known defect** | All twelve are built from one VG spelling and the published pickle still is. `bike` carries 638 of COCO's 3,683 `bicycle` boxes against the `bicycle` spelling's 775, and on the non-COCO half those became `bicycle` **negatives** (#3605, #3618). |
| Compare a number against one published **before 2026-09-06** | **not comparable** | #3667 changed both the evaluable set (4,000 → 5,806 per cell) and the realised prevalence (2.50% → 1.72%), which moves every k\* quoted from it. |
| Compare a number against one measured **before the 2026-09-08 rebuild** | **check what moved first** | The pile was rebuilt 2026-09-08 04:52 carrying three merged rulings: #3727 (a confirmed verdict now leaves a row, so `designate_cells` seats confirmed images ahead of unreviewed ones), #3726 (a reviewer's box designates an instance and the class's other instances are kept, so a cell carries **+270** region boxes) and #3662 (a tightened pooled-name criterion, +8 repaired images). Membership moved by **30 positives over 12 cells**; the shared negative pool did not move at all. Small, and not nothing: a per-class number on `stop sign@medium`, `clock@small`, `bus@{small,medium,large}` or `knife@large` rests on up to 6 different images out of 100. The pre-rebuild cells, roster and corrections are kept at `vts-cache/stash/pre-rebuild-20260908/`, so an old number can be re-measured against the pile it was taken on rather than argued about. |

**Per-class floors** (`pool_contamination.py`, #3635 — the estimated share of the
shared pool that actually holds the class, on VG's evidence alone):

| class | rate | | class | rate |
|---|---|---|---|---|
| `backpack` | **2.8%** | | `clock` | 1.1% |
| `book` | 1.7% | | `bus`, `umbrella` | 0.7% |
| `knife` | 1.6% | | `bicycle` | 0.6% |
| `stop sign` | 1.2% | | `bird`, `boat`, `dog`, `kite` | 0.3–0.5% |

`backpack` is the one to watch: ~110 expected hidden positives in a 3,900-image
pool against the 100 labelled ones per cell — more real backpacks among the
negatives than among the positives, which is the worst case `coco_anchor.py`
names.

**Region voting reads more boxes than it used to.** Since the 2026-09-08 rebuild a
reviewed positive carries every instance of its class rather than only the box the
reviewer drew (#3726), so a region-voting arm trains on +270 true boxes across the
reviewed images. That is a correction, not a regression — the boxes were always in
VG and the old merge discarded them — but it is a change in what an arm sees, so a
region-voting number measured across that boundary is one of the ones the row
above is about.

**Two of these are removable rather than permanent**, and cheaply: the negative
pool can be made *provable* instead of audited (#3668, #3670 — 18,986 images
outside the pile where COCO confirms none of the classes is present, no human
labelling involved), which retires the per-class spread and with it the
cross-class row. What cannot be made provable that way is the **off-COCO
positives** and their bands, because sourcing those from COCO would make this a
subset of COCO with extra steps — the thing the construction was corrected to
avoid.

## Reproducing and extending it

```bash
source scripts/experiments/pile/pile_env.sh
python build_pile.py --datasets vg_scale --force   # rebuild all three cells
python build_pile.py --verify --datasets vg_scale  # structure + review coverage
```

Corrections live in `corrections.json` as `(image_id, class, present, boxes)` and
are merged over `objects.json` **before** banding, so a corrected box can move an
image between bands. A correction with no box excludes the pair from every cell
of that class rather than promoting it: a band is a claim about size, and no size
was measured.

Membership is pinned by `vg_scale_roster.json` and selection is hash-stable, so a
rebuild does not reshuffle cells. **Run `check_review_coverage.py` after any
rebuild.** Coverage is the one property no structural check implies — cells can
be full, prevalence exact and boxes valid while the dataset no longer contains
the images its review was performed on, which happened here and cost a day.
