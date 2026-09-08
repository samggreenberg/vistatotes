# Make `vg_scale` a fully annotated set, not a single-question benchmark

## What this changes

`vg_scale` is built as a **designation**: a class list, a band rule, and a fixed
count of positives and negatives per cell, sampled to make one contrast
(small-vs-large, within a class) paired and clean. That was the right shape for
#3156 and it answered it.

It is the wrong shape for the actual goal, which is a set like COCO's — every
`(image, class)` pair answered — that many questions can be asked of. Under a
designation, prevalence, band definition, class list and pool composition are
all *build-time* choices, so changing any of them is a rebuild and a re-embed.
Under an exhaustively annotated set they are **queries**.

This plan closes the gap. It is small, because most of it is already done.

## What is already exhaustive, and what the debt is

Every class in `SCALE_CLASSES` is also a COCO-2017 class — chosen that way
deliberately, so COCO's exhaustive annotation can answer for all of them at once
on the images the two sources share. `scripts/experiments/pile/coco_anchor.py`
reads both `instances_train2017.json` and `instances_val2017.json`, so the
anchored half is the whole VG∩COCO overlap rather than val alone.

Since #3670 drew the negative pool entirely from that anchored half, the only
stratum lacking exhaustive labels is the **off-COCO positives**:

| stratum | 12-class build | exhaustive today |
|---|---:|---|
| negative pool | 9,900 | yes — COCO |
| spares | 1,000 | yes — COCO |
| positives, COCO-anchored | 2,037 (57.45%) | yes — COCO |
| **positives, off-COCO** | **1,509 (42.55%)** | **no — this is the whole debt** |

On the shipped 25-class cell the debt is **3,391 images**, 47.43% of its 7,150
positives — counted by `scripts/experiments/pile/annotation_queue.py`, which
emits the worklist itself (#3720). It is **one exhaustive pass per image** — a
reviewer naming which of the 25 classes are present — rather than 25 binary
votes. For scale, the pass already committed to for #3702 is ~400 cleared
negatives per class, ~10,000 images. The 12-class counts above are from
[the #3670 study](../experiments/2026-09-06-provable-negatives-3670/REPORT.md).

**The zero-annotation version was measured and does not reach.**
`scripts/experiments/pile/exact_supply.py` asks whether the COCO half alone
supplies the cells; 20 of 36 shipped cells and 17 of 39 candidate cells fall
short of 300 COCO-anchored positives (`dog@small` 114, `spoon@large` 106), so
the off-COCO half cannot simply be dropped — it has to be answered. See
[the class-list study](../experiments/2026-09-03-vg-scale-classes/REPORT.md).

## Why it is worth more than the pass costs

The annotation retires the machinery that exists **only** to substitute
inference for an answer on the off-COCO half. Each of these is a measured,
shipped subsystem whose reason for existing is VG's silence:

- the VG name tables and the two-search candidate hunt behind them
  (`scripts/experiments/pile/name_evidence.py`, `SCALE_VG_NAMES`,
  `SCALE_VG_AMBIGUOUS`) — spellings stop mattering once a human answered;
- pooled name adjudication and its homogeneity gate;
- pool-contamination measurement
  (`scripts/experiments/pile/pool_contamination.py`) — contamination goes to 0
  on *both* halves, not just COCO's;
- the `provable` / `matched` pool-composition choice, whose published
  justification has been retracted in `pile_config.SCALE_NEG_COMPOSITION` and is
  open as #3702 — it dissolves rather than being resolved, because the trade it
  makes (contamination against a provenance artifact) only exists while one half
  is unanswered;
- the global ambiguous exclusion (#3655), which costs every class an image
  because one spelling was ambiguous for one class;
- `anchor_to_coco`'s silent un-banding (#3659), which becomes a reconciliation
  between two human-grade answers rather than an overwrite.

It also ends the rebuild treadmill: membership currently drifts when rulings
change, 41 positives out and 40 in on a rebuild with nothing relevant altered
([the #3667 rebuild study](../experiments/2026-09-06-cross-class-negatives-3667/REPORT.md)).

## Two decisions this forces

**The band statistic — decided 2026-09-07: the band stays the union, and every
instance stays behind it.** The union is what one Good vote drags in the app and
is what #3156 measured, so the shipped statistic does not move; the instances
are kept because the union is derivable from them and they are not derivable
from it, which leaves largest, smallest, count and density available at eval
time for free. Recorded beside `BOX_BANDS` in `pile_config`. The build already
stores every instance — what this now commits us to defending is that **review**
does not quietly collapse the set (#3726).

**Two label sources of unequal quality.** COCO on one half and our annotators on
the other means provenance would correlate with *label noise* instead of with
*label* — a new form of #3702 rather than its elimination. The mitigation is
free and already available: the class list was chosen so annotator accuracy can
be scored against COCO on the anchored half without extra review. Build that
scoring into the pass rather than discovering the drift afterwards.

## Settle before the pass starts

Each of these is an open issue, carried here as a pointer because the *order*
matters and nothing else records it. They move the designated set, change what
the pass is asked to cover, or bill it for work it does not need. None is large
on its own — tens of images each. The cost is that every one of them forces a
**rebuild**, and a rebuild reshuffles which images fill a cell: #3667 measured 41
positives out and 40 in with nothing relevant altered, and three rebuilds once
retired 577 of 743 reviewed images. Human answers are the only input here that
cannot be regenerated, so a rebuild *after* the pass throws some of them away.

Land or dismiss all of them, rebuild once, then freeze the roster — in that
order.

Three have already left the list on measurement rather than on work. **#3686**
was never a blocker: I filed it here as overlapping the pass, and it does not —
the pass annotates off-COCO *positives* and that issue hunts false negatives in
the *pool*. Its own premise is also pre-#3670: the shipped pool is now
9,900/9,900 COCO-answerable with zero measured contamination, so the stratum it
searches is empty. `vg_scale_deep`, whose 6,264 unprovable negatives are the
only VG-silence population left in the pile, is where that instrument now points. **#3663** is
refuted: the misspellings it named are in VG and reachable by an edit-distance
search, they were scored by the same three cuts, and **none clears them** — the
real ones carry 3–5 images each. **#3655** moves the *negative* pool, which the
pass does not annotate, so it is not a blocker for it at all; its stated cost is
also a use the datasheet already refuses, and its stated benefit is nearly inert
against a pool that is 18x over-subscribed. The VG-name audit itself is no longer among them: `SCALE_VG_NAMES_AUDITED`
now covers all 25 classes, so #3618's "before the next rebuild" warning is spent.

<!-- item-sep -->

- [ ] #3729 — every human verdict exists only on purgeable scratch, with no second copy anywhere

<!-- item-sep -->

- [ ] #3727 — a confirmed verdict writes no correction, so the build cannot tell an answered image from an unopened one

<!-- item-sep -->

- [ ] #3726 — a rebox replaces the class's whole instance set, spending the band decision above (implemented as #3740, awaiting the owner's ruling — do not start it again)

<!-- item-sep -->


<!-- item-sep -->

- [ ] #3662 — one polysemous member sinks `boat`'s pooled group (criterion pre-registered and run; #3741 awaits the owner's ruling — do not start it again)

<!-- item-sep -->

- [ ] #3605 — `bike` is still unresolved for `bicycle`; measured 2026-09-07 as too big to fold into the pass (+9,963 images, ~4x)

<!-- item-sep -->


<!-- item-sep -->


<!-- item-sep -->

- [ ] #3616 — a reviewer's rebox silently moves an image between bands (the band statistic above, arriving as a bug)

<!-- item-sep -->

- [ ] #3669 — slate import re-embeds vectors the pile already holds (bills the pass three times over for its own images)

<!-- item-sep -->


<!-- item-sep -->

## Open work

<!-- item-sep -->

- [ ] #3720 — Emit the annotation queue (Sonnet 5)

<!-- item-sep -->

- **Freeze the roster, and say what freezing means.** The debt is ~3,200 images
  only if the designated set stops moving; otherwise rebuild churn keeps pulling
  in unannotated images. Decide what pins the roster (an explicit id list beside
  the config, versioned), what may still change it, and how a rebuild proves it
  did not. `scripts/experiments/pile/check_review_coverage.py` is the existing
  partial answer and is not sufficient on its own. The policy is the owner's
  call; the mechanism is Opus 4.8 work, since it is the invariant everything
  else rests on. (human + Opus 4.8)

<!-- item-sep -->

- **Treat the first batch as the pilot, rather than piloting separately.**
  Naming which of 25 classes are present is a materially harder task than a
  binary vote, and COCO itself used a staged per-class protocol rather than one
  sweep over 80 — so the interaction may well need changing. That is an argument
  for looking at the first few hundred judgements before running the other three
  thousand, not for annotating throwaway images first: start on the real queue,
  and read the first batch as the pilot. What to read off it is per-class recall
  and time per image, both against COCO on whatever part of the batch is
  anchored.
  `scripts/experiments/lessons/2026-09-02-one-pilot-cell-cleared-a-hazard-the-full-wave-hit.md`
  is the standing reason to look before committing the rest. (human + Sonnet 5)

<!-- item-sep -->

- **Score the annotators against COCO, continuously.** On the anchored half the
  answer key already exists, so every pass can carry a measured accuracy per
  annotator and per class at no extra review cost. Emit it with the pass rather
  than as a one-off audit; it is what keeps the two-source noise asymmetry
  visible. (Sonnet 5)

<!-- item-sep -->


<!-- item-sep -->

- **Run the pass, and rebuild once.** Off-COCO positives only, one exhaustive
  judgement per image, recorded as verdicts rather than corrections in VG's own
  shape (`scripts/experiments/pile/verdicts_to_corrections.py`'s existing
  contract). One rebuild at the end, not one per ruling. (human + Sonnet 5)

<!-- item-sep -->

- [ ] #3696 — emit VG's silence rate from the pass's own answers, as an upper bound (Sonnet 5)

<!-- item-sep -->

- **Retire the inference machinery, and delete it.** After the pass, the name
  tables, the pooled adjudication, the contamination measurement and the pool
  composition switch are all answering a question that no longer exists. They
  are experiment-tier code, so the backwards-compatibility rules do not protect
  them — delete rather than deprecate, and prune the plan pointers and README
  sections that describe them. Retire the published numbers conditioned on the
  old construction at the same time, or say plainly which construction they were
  measured under. The risk is deleting something still load-bearing for a
  published result. (Opus 4.8)

<!-- item-sep -->

- **Move prevalence, band and class list to eval-time queries.** The point of
  the exercise. `SCALE_N_POS` / `SCALE_N_NEG` / `SCALE_PREVALENCE` are
  build-time designations today, and `SCALE_PREVALENCE` is already documented as
  *designed, not realised*. With exhaustive labels a cell is a filter over a
  fixed set, so prevalence becomes a sampling parameter the harness applies and
  a class-list change stops being a re-embed. This is the item that turns the
  benchmark into a dataset. (Opus 4.8)

<!-- item-sep -->

- **Ask the questions the set was built for.** Co-occurrence (does a class get
  harder when a same-scene partner is present?), natural-composition negatives
  instead of a designed ratio, multi-label arms, and calibration at a prevalence
  chosen per question rather than pinned at build time. None of these are
  reachable under a designation; all of them are one query away under an
  annotated set. File them as they become concrete rather than listing them
  here. (Sonnet 5)

<!-- item-sep -->

## Related

- [`vg-scale-bands-and-corrections.md`](vg-scale-bands-and-corrections.md) — the
  band construction and the correction loop this builds on. Its open review
  items are about bounding the error on the *inferred* half; several stop being
  owed if that half is answered instead, so re-read it when the pass lands.
- #3668 — the origin of this idea, filed from the reviewer's side: a 13-way
  judgement recorded as one bit, with twelve thrown away.
- #3702, #3655, #3659 — open issues that dissolve rather than being fixed.
