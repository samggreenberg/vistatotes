# How safe should safe-thresholds be? (#2841)

> # ⚠️ SEEDING CAVEAT — these runs did not start the way the app does
>
> **Recorded 2026-08-26 (#3156).** Autopilot seeds its first three Good votes from
> a **text sort**: the user types a query and votes down that ranking. Until
> PR #3269 this harness instead ranked every item by cosine to a **crop of one
> boxed positive** — a ranking no user ever produces — and passed it as
> `seed_scores`, the argument that `al_strategies`, `EVAL.md` and
> `voting_iterations` all describe as "similarity to the **typed query**".
>
> **What to distrust here:** anything that depends on *how a run starts* —
> positive starvation, stuck or never-got-going runs, `n_good`, and
> early-trajectory cost. Measured on one cell after the fix, text seeding put the
> first positive at **rank 1** with five in the top 20, while the exemplar that
> crop-seeding made look like the dataset's hardest positive ranked **4006 of
> 7749** for its own class.
>
> **What still holds:** within-study contrasts where every arm seeded identically,
> which is most of what these reports conclude — the seeding is a shared baseline
> shift, not an arm-dependent one.
>
> See [the harness seeded from a crop](../../../scripts/experiments/lessons/2026-08-26-the-harness-seeded-from-a-crop.md).


## Verdict

**The handoff is too fast, production's cut is the worst-calibrated of the nine
schedules measured, and the two voting modes want different curves.**

| | ship | vs the old ramp | holds at `fpr x4`? |
|---|---|---|---|
| **Region voting** | **`slow_cap50`** (ramp to 40 labels, capped at half GMM) | −0.0577 (7-20 votes), −0.094 by 51-100 | **yes**, improves to −0.0745 |
| **Binary voting** | **`cap50`** (old ramp, capped at half GMM) | −0.0188 (7-20 votes), −0.051 by 201-300 | **yes**, improves to −0.0236 |

**Neither schedule ever hands over to the learned cut**, and both were re-tested
at 10x the original horizon to make sure that is a finding and not an artefact.
It survives: every arm that hands over gives its advantage back at the moment it
does, monotone in release point, and the capped schedules' margins *grow* with
depth. The first-pass answer for region (`slow`, a plain ramp to 40) was wrong
for exactly this reason — it reaches pure x-cal at 40 labels and decays to
nothing past it (+0.008 by 101-200 votes). `slow_cap50` keeps the ramp that won
the early window and the cap that wins the rest.

This is **not** because the GMM is better in the limit — it is an inconsistent
estimator and cannot be. It is because 300 clicks buys a median of **~13
positives**, and the learned cut converges in positives, not clicks. See
*The horizon problem* below.

If a single schedule is preferred over a per-mode one, ship **`cap50`
everywhere**: it is the only schedule that improves *both* modes under *every*
weighting tested (region −0.0210, binary −0.0173), though it leaves most of the
available region-voting gain on the table.

**Not recommended, despite scoring best at the shipped operating point:**
`vslow` (region −0.0539, binary −0.0328) and `pure_gmm` (region −0.0530). Their
binary-voting advantage is a *lower cut*, not better calibration, and reverses
hard when false positives are weighted more — `pure_gmm` goes to **+0.2446** at
`fpr x4`. Since the Inclusion knob is defined as `wf·FPR + wn·FNR`, that
reweighting is not hypothetical: it is roughly what an Inclusion-averse user
experiences.

**The issue's premise was inverted, then vindicated by a different route.**
"Just use the GMM all the time" is *not* obviously bad — on region voting it was
among the best schedules at the shipped operating point, and it is the closest
to the oracle cut of any schedule measured there (mean |gap| 0.0228 vs
production's 0.0359). But it is the least robust choice, so the answer to "how
safe should safe be" is **permanently a little safe** (a cap, or a much longer
ramp), not "always maximally safe".

## The question

`safe_thresholds` blends two candidate decision thresholds: a **GMM cut** fitted
on the score distribution (needs no labels, never wild) and a **cross-calibration
cut** (conformal, uses the labels, unreliable when there are few). #2799 settled
*whether* to blend and turned it on for everyone. It did not touch the schedule
inside the blend — **how much GMM, for how long** — which had been one
hard-coded line since it was written:

```python
x-cal weight = clip((n_labels - 6) / 14, 0, 1)   # pure GMM ≤6, pure x-cal ≥20
```

Three independent choices are baked into that line, and none had ever been
measured: the **endpoints** (6, 20), the **shape** (linear), and the
**statistic** it ramps on (total labels). A fourth question — whether a weighted
average is even the right combiner — could not be expressed in it at all.

The issue's framing was that pure GMM forever is presumably bad ("that's
probably not better than ignoring the learned threshold entirely, right?").
**The data says otherwise, and then says something more interesting than
either.**

## What was built

`vtscore/training/blend_schedules.py` — a registry of 18 named schedules across
five families, so each buried choice becomes an arm:

| Family | Schedules | What it varies |
|---|---|---|
| controls | `pure_gmm`, `pure_xcal` | the two ends of the axis |
| A. endpoints | `prod`, `fast`, `slow`, `vslow`, `early`, `late` | when the handoff starts and finishes |
| B. shape | `convex`, `concave`, `step`, `logistic` | the curve between the endpoints |
| C. statistic | `rare`, `pos` | ramps on the **rarer class** / positive count instead of the total |
| D. cap | `cap80`, `cap50` | never hand off completely |
| E. corridor | `corridor`, `corridor_ramp` | **clamp** x-cal between the GMM component means instead of averaging |

Family C exists because the binding constraint on conformal calibration is the
rarer class, not the total: a 19-bad/1-good labelset has 20 labels and one
positive, and #2790 traced the deep threshold spikes to exactly that starvation.
Family E exists because a weighted average taxes *every* x-cal cut to defend
against the rare wild one, whereas the pathology safe-thresholds actually fixes
is wild (#2788's cold-start "admit nothing" cuts).

`prod` is pinned **bit-identical** to the historical ramp by a 0..80
parametrized test, so the machinery is a no-op for users until a default flips.

### Fidelity: the app and the framework disagreed, and it hid two bugs

Both app training paths hard-coded *"skip cross-calibration below 6 votes"*;
the eval harness never skipped. That was safe only because the production weight
happens to be 0 there — a coincidence, not a guarantee. Deriving the skip from
the schedule (`xcal_is_discarded`) makes the two agree by construction, and any
schedule that trusts the learned cut earlier now automatically stops skipping.
Two real defects fell out:

- **The two app paths left different placeholders** (`NO_GOOD_THRESHOLD` on the
  Find path, `0.5` on the vote/labelset path). Normally discarded — but when the
  GMM fit degenerates, `blend_gmm_threshold` falls back to that placeholder, so
  the same collection would admit *nothing* through one path and *everything
  scoring ≥0.5* through the other. Both now admit nothing, the safe reading of
  "no threshold was ever computed".
- **An off-by-one against its own rationale**: the guard skipped *below* 6 but
  paid for two 200-epoch fold fits at *exactly* 6, where the weight is already
  zero. No user ever saw the difference — the app's first trained detector
  appears at 7 votes — but it was pure waste.

### COCO, from cache

The issue asks for VG **and** COCO, and COCO is not a VTSearch demo dataset. It
does not need to be: the #2790 sweep already embedded all 4952 COCO-2017-val
images, and the boxes are staged flat beside them, so `build_coco_pickle.py`
joins the two into an ordinary media pickle and every existing stage runs on it
unchanged. Nothing is re-embedded.

**One honest limit.** That cache stores each image's whole vector and its HAC
region vectors but **not** the raw 14×14 patch grid that `max_patch` pools over,
so COCO can only serve the **binary-voting** arm; a COCO region arm would need a
genuine re-embed. The builder refuses patch embedders outright rather than
emitting a pickle that would silently score as whole-image while being reported
as a region arm.

## Method

Two phases, because the blended threshold **feeds acquisition**: Autopilot's
Hard phase picks the item nearest the decision boundary, so two schedules label
different items and their trajectories diverge.

- **Phase 1 — screen.** One run on the production trajectory, with every
  schedule re-cut counterfactually at each step (one extra metric row each). All
  schedules see the same model, the same step, and the same held-out test
  scores, so they are exactly paired. Cheap — many schedules for the price of
  one run — but structurally blind to acquisition feedback.
- **Phase 2 — A/B.** One full independent trajectory per schedule, paired per
  cell. This is the verdict; the screen only decides who gets to run.

**Arms.** Region voting = VG × `dinov3_patch` × `max_patch` (the production
region path live decisions read). Binary voting = VG × `siglip` and COCO ×
`siglip`/`siglip2` × `whole_image`. Reported **separately, never pooled** — the
issue allows them to want different curves, and pooling would hide exactly that.

**Metric.** Inclusion-weighted `cost` (= FPR + FNR at inclusion 0), averaged over
the **7–20 vote window**: 7 is the app's first trained-detector step, 20 is where
the production ramp ends.

**Scale.** 1008 cells, 23 VG categories (scale-banded) + 19 COCO categories,
12 seeds, 30 steps, the production **linear** head. 42 cells emitted no rows
(rare small-object categories; deterministic and pre-vote, so symmetric across
schedules).

**Fidelity check, run before anything is reported:** `prod`'s counterfactual row
must reproduce the threshold and cost the run actually used. It did, to
**0.000e+00 over 26,142 paired rows**. Every other schedule row comes off the
same code path, so this is what licenses the rest.

## Phase 1 result: the ranking is monotone in "how long you keep the GMM"

Region voting (n = 254 paired cells, baseline `prod` cost 0.5376):

| schedule | cost | d_cost | % cells improved | p (Wilcoxon) | d_fnr | d_fpr |
|---|---|---|---|---|---|---|
| `pure_gmm` | 0.4839 | **−0.0537** | 83.5 | 9.7e-27 | −0.0738 | +0.0201 |
| `vslow` | 0.4875 | −0.0501 | 85.0 | 3.1e-30 | −0.0599 | +0.0098 |
| `slow` | 0.4981 | −0.0395 | 86.6 | 4.3e-32 | −0.0430 | +0.0035 |
| `late` | 0.4987 | −0.0389 | 86.2 | 8.4e-31 | −0.0465 | +0.0076 |
| `corridor_ramp` | 0.5144 | −0.0232 | 67.3 | 1.1e-17 | −0.0298 | +0.0066 |
| `cap50` | 0.5162 | −0.0214 | 86.2 | 1.1e-34 | −0.0205 | **−0.0009** |
| `convex` | 0.5197 | −0.0179 | 82.7 | 9.3e-28 | −0.0221 | +0.0042 |
| `rare` | 0.5230 | −0.0146 | 57.9 | 7.4e-06 | −0.0159 | +0.0014 |
| `pos` | 0.5289 | −0.0087 | 55.5 | 0.0087 | −0.0116 | +0.0029 |
| `cap80` | 0.5335 | −0.0041 | 76.8 | 1.3e-25 | −0.0039 | **−0.0002** |
| `corridor` | 0.5355 | −0.0021 | 44.9 | 0.24 (n.s.) | −0.0079 | +0.0057 |
| `logistic` | 0.5417 | +0.0042 | 33.1 | 4.5e-11 | +0.0004 | +0.0037 |
| `early` | 0.5497 | +0.0121 | 21.7 | 6.7e-27 | +0.0137 | −0.0016 |
| `step` | 0.5537 | +0.0161 | 22.8 | 1.6e-20 | +0.0047 | +0.0114 |
| `concave` | 0.5584 | +0.0208 | 16.9 | 1.6e-31 | +0.0216 | −0.0007 |
| `fast` | 0.5786 | +0.0410 | 11.8 | 4.7e-35 | +0.0385 | +0.0025 |
| `pure_xcal` | 0.6012 | +0.0636 | 13.0 | 3.4e-35 | +0.0606 | +0.0030 |

Binary voting (n = 694 paired cells, baseline `prod` cost 0.4931):

| schedule | cost | d_cost | % cells improved | p (Wilcoxon) | d_fnr | d_fpr |
|---|---|---|---|---|---|---|
| `rare` | 0.4621 | **−0.0310** | 86.0 | 6.8e-74 | −0.0592 | +0.0282 |
| `pos` | 0.4622 | −0.0308 | 85.9 | 1.6e-72 | −0.0591 | +0.0283 |
| `vslow` | 0.4624 | −0.0307 | 76.2 | 6.3e-50 | −0.0827 | +0.0520 |
| `slow` | 0.4640 | −0.0291 | 82.4 | 1.1e-68 | −0.0578 | +0.0287 |
| `late` | 0.4691 | −0.0239 | 75.4 | 2.8e-49 | −0.0656 | +0.0417 |
| `pure_gmm` | 0.4694 | −0.0236 | 69.3 | 2.5e-25 | −0.1060 | +0.0824 |
| `corridor` | 0.4714 | −0.0217 | 69.0 | 6.3e-34 | −0.0313 | +0.0096 |
| `corridor_ramp` | 0.4728 | −0.0202 | 72.8 | 1.5e-40 | −0.0601 | +0.0398 |
| `cap50` | 0.4752 | −0.0179 | 87.5 | 4.4e-89 | −0.0211 | +0.0032 |
| `convex` | 0.4826 | −0.0105 | 70.0 | 3.1e-33 | −0.0354 | +0.0249 |
| `cap80` | 0.4896 | −0.0035 | 75.5 | 2.2e-72 | −0.0034 | **−0.0001** |
| `early` | 0.5001 | +0.0070 | 30.3 | 2.5e-33 | +0.0228 | −0.0158 |
| `logistic` | 0.5013 | +0.0082 | 22.8 | 8.5e-57 | −0.0037 | +0.0119 |
| `concave` | 0.5090 | +0.0159 | 19.5 | 5.3e-63 | +0.0350 | −0.0190 |
| `step` | 0.5222 | +0.0291 | 15.3 | 1e-80 | +0.0005 | +0.0286 |
| `fast` | 0.5336 | +0.0405 | 11.5 | 3e-91 | +0.0596 | −0.0190 |
| `pure_xcal` | 0.5588 | +0.0657 | 11.2 | 1.7e-92 | +0.1011 | −0.0354 |

Three things read straight off these tables:

1. **`pure_xcal` is the worst schedule on both arms** (+0.064 / +0.066). That is
   safe-thresholds OFF, and it independently reproduces #2799's verdict on a
   different grid — a free replication.
2. **The issue's premise is inverted.** Pure GMM forever is not bad. On region
   voting it was the single **best** schedule (−0.0537), and on binary voting it
   still beat the incumbent. Every schedule that hands off *faster* than
   production (`fast`, `step`, `concave`, `early`) lost.
3. **Production sits in the wrong half of its own family.** Ten of seventeen
   schedules beat it.

## …but most of that is not calibration, it is a lower cut

Every schedule at the top of those tables has a strongly negative `d_fnr` and a
positive `d_fpr`. That is the signature of simply **cutting lower**, and at
inclusion 0 the cost weights are (1, 1), so trading a lot of FNR for a little
FPR scores as a win. It is a real win at the shipped operating point — but #2790
flagged exactly this trap, so the same paired cells were re-scored under
asymmetric weights.

Region voting, `d_cost` under each weighting:

| schedule | fpr ×1 (shipped) | fpr ×2 | fpr ×4 | fnr ×2 |
|---|---|---|---|---|
| `pure_gmm` | −0.0537 | −0.0336 | **+0.0066** | −0.1276 |
| `vslow` | −0.0501 | −0.0403 | −0.0208 | −0.1100 |
| `slow` | −0.0395 | −0.0360 | −0.0291 | −0.0825 |
| `cap50` | −0.0214 | **−0.0222** | **−0.0240** | −0.0419 |
| `cap80` | −0.0041 | −0.0043 | −0.0047 | −0.0080 |
| `corridor` | −0.0021 | +0.0036 | +0.0150 | −0.0100 |

Binary voting:

| schedule | fpr ×1 (shipped) | fpr ×2 | fpr ×4 | fnr ×2 |
|---|---|---|---|---|
| `rare` | −0.0310 | −0.0028 | **+0.0537** | −0.0902 |
| `vslow` | −0.0307 | +0.0213 | **+0.1253** | −0.1133 |
| `slow` | −0.0291 | −0.0004 | +0.0571 | −0.0869 |
| `pure_gmm` | −0.0236 | +0.0588 | **+0.2235** | −0.1297 |
| `corridor` | −0.0217 | −0.0121 | +0.0071 | −0.0529 |
| `cap50` | −0.0179 | −0.0147 | **−0.0084** | −0.0390 |
| `cap80` | −0.0035 | −0.0036 | −0.0037 | −0.0069 |

`pure_gmm`'s apparent win **flips** on both arms — catastrophically on binary
(+0.2235). So do `rare`, `pos`, `slow` and `vslow` on binary. The schedules that
improve at **every** weighting are the **cap family**, plus `slow` on region only.

The sharpest version of this: on region voting, **`cap50` and `cap80` are the
only schedules that improve *both* error types** — `cap50` at
d_fnr −0.0205 *and* d_fpr −0.0009, a Pareto move with no trade at all. Every
other winner buys FNR with FPR (`pure_gmm` gives up 0.27 FPR per FNR gained on
region, 0.78 on binary; `cap50` gives up 0.15 on binary and *nothing* on region).

**Method note, stated plainly:** this re-weighting was added *after* seeing the
first table, prompted by the FNR/FPR pattern — it was not pre-registered. And it
is a **scoring** sensitivity, not a simulation of a different Inclusion setting:
moving the Inclusion knob changes the conformal rule itself, not just the
weights. It answers "is this win an artefact of a symmetric metric", which is
the question it was added for, and nothing more.

## Reading after Phase 1

The two questions come apart:

- *"Should the handoff be slower?"* — At the shipped operating point, yes,
  dramatically. But the gain is mostly permissiveness, and a user who cares more
  about false alarms than misses would see it reverse. That makes it a
  **product/operating-point decision**, not a pure calibration improvement.
- *"How safe should safe be?"* — The weighting-independent answer is **never
  fully hand off**. A permanent GMM share is the one change that improves
  calibration rather than relocating the operating point, and on the production
  region path it improves both error types at once.

`cap50` was not in the pre-registered promotion list (which ranked on the
shipped metric alone), so it was **added** as an extra A/B arm rather than
quietly substituted; the pre-registered arms all still ran.

**The A/B upheld this reading.** The pre-registered rule — beat `prod` at
p < 0.01 on your own mode without losing on the other — is passed by `vslow`,
`slow`, `cap50`, `pure_gmm` and (region) `rare`, and on the shipped metric alone
it selects `vslow`. The recommendation is `slow`/`cap50` instead **because of
the post-hoc robustness analysis**, which is a deviation from the
pre-registration and is flagged as one. Anyone who prefers to hold the
pre-registration strictly should read the verdict as `vslow`, and should also
accept that its binary-voting gain reverses under reweighting.

## Phase 2 — A/B trajectories (the verdict)

Nine arms, 16 seeds, 1344 cells each (12,096 cells, 312,606 rows), each a full
independent trajectory so the blend's effect on *which items Autopilot asked the
user to label* is included. Paired per cell, since two arms' step *t* are not
the same state.

Region voting (n = 339 paired cells, `prod` cost 0.5336):

| schedule | d_cost | % improved | p (Wilcoxon) | d_fnr | d_fpr | `fpr x2` | `fpr x4` |
|---|---|---|---|---|---|---|---|
| `vslow` | −0.0539 | 78.5 | 1.3e-30 | −0.0621 | +0.0082 | −0.0458 | −0.0295 |
| `pure_gmm` | −0.0530 | 78.2 | 5.9e-25 | −0.0713 | +0.0183 | −0.0347 | **+0.0019** |
| **`slow`** | **−0.0422** | 78.8 | 1.4e-28 | −0.0440 | +0.0019 | −0.0403 | **−0.0366** |
| `cap50` | −0.0210 | 81.1 | 8.0e-35 | −0.0218 | +0.0009 | −0.0201 | −0.0184 |
| `rare` | −0.0188 | 59.3 | 1.6e-05 | −0.0186 | −0.0002 | −0.0190 | −0.0195 |
| `pos` | −0.0083 | 52.8 | 0.10 (n.s.) | −0.0118 | +0.0035 | −0.0048 | +0.0022 |
| `cap80` | −0.0041 | 71.4 | 5.0e-23 | −0.0037 | −0.0004 | −0.0045 | −0.0052 |
| `corridor` | −0.0021 | 46.3 | 0.62 (n.s.) | −0.0069 | +0.0048 | +0.0027 | +0.0123 |

Binary voting (n = 922 paired cells, `prod` cost 0.4940):

| schedule | d_cost | % improved | p (Wilcoxon) | d_fnr | d_fpr | `fpr x2` | `fpr x4` |
|---|---|---|---|---|---|---|---|
| `vslow` | −0.0328 | 66.6 | 9.6e-34 | −0.0879 | +0.0552 | +0.0224 | **+0.1325** |
| `pos` | −0.0303 | 69.5 | 6.8e-37 | −0.0561 | +0.0259 | −0.0040 | +0.0478 |
| `rare` | −0.0300 | 69.6 | 8.3e-37 | −0.0555 | +0.0255 | −0.0041 | +0.0470 |
| `slow` | −0.0273 | 62.0 | 2.0e-26 | −0.0592 | +0.0320 | +0.0050 | +0.0689 |
| `pure_gmm` | −0.0230 | 60.1 | 1.2e-14 | −0.1122 | +0.0892 | +0.0663 | **+0.2446** |
| `corridor` | −0.0226 | 62.1 | 4.2e-19 | −0.0332 | +0.0105 | −0.0121 | +0.0090 |
| **`cap50`** | **−0.0173** | 68.1 | 7.6e-43 | −0.0190 | +0.0018 | −0.0155 | **−0.0119** |
| `cap80` | −0.0034 | 62.0 | 1.3e-24 | −0.0032 | −0.0002 | −0.0036 | −0.0041 |

Every arm also cut the **degenerate-threshold rate** (−0.0025 binary, −0.0011
region) — the #2788 cold-start "admit nothing" failure — and moved AP by ≤0.014,
confirming these are calibration and acquisition effects, not ranking effects.

### The screen under-stated everything, and acquisition is why

The A/B effects are consistently *larger* than the screen's (region `vslow`
−0.0539 vs −0.0501). The reason shows up past the ramp, at **21+ votes**, where
every schedule has converged and the blend has no authority at all:

| | `pure_gmm` | `vslow` | `slow` | `cap50` |
|---|---|---|---|---|
| region d_cost | −0.1015 | −0.0968 | −0.0592 | (−0.0246 `cap80`) |
| binary d_cost | −0.0528 | −0.0568 | −0.0367 | −0.0452 |

Nothing but **acquisition feedback** can carry a difference there: the blended
threshold feeds Autopilot's Hard pick, so a better-calibrated blend asks the
user to label better items, and the resulting detector stays better long after
the blend has handed over. On region voting that channel is roughly **twice the
size of the direct effect** — and it is invisible to any within-step analysis,
which is exactly why the screen was never allowed to decide.

### Why the winners win: bias vs spread

Mean signed gap to each step's own oracle cut (`bias`; negative = cuts lower)
and the SD of that gap within a cell (`spread`; lower = steadier):

| region | bias | spread | \|gap\| | | binary | bias | spread | \|gap\| |
|---|---|---|---|---|---|---|---|---|
| `pure_gmm` | −0.0038 | 0.0105 | **0.0228** | | `rare` | −0.0009 | 0.0528 | **0.0634** |
| `vslow` | −0.0003 | 0.0115 | 0.0232 | | `pos` | −0.0010 | 0.0529 | 0.0635 |
| `slow` | +0.0035 | 0.0135 | 0.0250 | | `vslow` | −0.0354 | 0.0512 | 0.0676 |
| `cap50` | +0.0100 | 0.0172 | 0.0304 | | `slow` | −0.0151 | 0.0618 | 0.0704 |
| `rare` | +0.0131 | 0.0175 | 0.0326 | | `pure_gmm` | −0.0561 | 0.0426 | 0.0745 |
| `prod` | +0.0153 | 0.0230 | 0.0359 | | `cap50` | +0.0125 | 0.0702 | 0.0787 |
| | | | | | `prod` | +0.0277 | 0.0867 | 0.0935 |

Three things this settles:

1. **Production cuts too high and too noisily.** It has the largest positive
   bias on both modes and the largest spread on both — the worst-calibrated
   schedule of the nine on binary voting, and second-worst on region. The whole
   study's headline is really this: the shipped ramp hands over to a
   *systematically conservative and unstable* estimate too early.
2. **The prediction held.** The cap family buys **spread** (binary 0.0702 vs
   production's 0.0867; region 0.0172 vs 0.0230) while `pure_gmm` on binary buys
   **bias** (−0.0561, far below the oracle). Averaging with a label-free cut is
   a variance-reduction operation, and variance reduction helps under any
   weighting while bias only helps when the weighting likes its direction —
   which is precisely the `fpr x4` pattern.
3. **`rare` is the best-calibrated cut on binary voting** (|gap| 0.0634,
   essentially unbiased) — the rarer-class statistic really is the right thing
   to ramp on, as #2790's positive-starvation diagnosis predicted. It is not the
   recommendation only because its *cost* advantage still reverses at `fpr x4`;
   being closest to the oracle threshold and being robust under reweighting are
   not the same thing, because a small threshold error maps to a large rate
   change where the score density is steep.

On region voting `pure_gmm` is both near-unbiased and steadiest, yet still
loses its advantage at `fpr x4`. That tension is real and unexplained by the
decomposition alone; it is the clearest open thread this study leaves.

## The horizon problem: does the learned threshold ever win? (binary voting)

**Answer: not within a session anyone will ever have — because clicking buys
votes, and the learned cut needs *positives*.**

Six arms, 300 votes, 26 deep categories (≥50 positives in the simulation half),
8 seeds, 1824 cells. Binary voting only; region is a separate run below.

`d_cost` vs `prod`, by vote band (negative beats it, cell count in parens):

| schedule | 7-20 | 21-50 | 51-100 | 101-200 | 201-300 |
|---|---|---|---|---|---|
| `cap50` | −0.0188 | −0.0388 | −0.0376 | −0.0401 | **−0.0511** |
| `cap50_release_late` (150→400) | −0.0188 | −0.0388 | −0.0376 | −0.0402 | −0.0504 |
| `cap50_release` (50→200) | −0.0188 | −0.0388 | −0.0376 | −0.0235 | **+0.0039** |
| `cap50_release_early` (30→100) | −0.0188 | −0.0388 | −0.0259 | **+0.0029** | +0.0036 |
| `pure_xcal` | +0.0520 | +0.0040 | −0.0024 | −0.0061 | −0.0123 |

Three things, in order of how much they matter:

1. **Handing over always costs, and costs more the earlier you do it.** The
   release arms track `cap50` exactly until they release, then fall away from
   it: `cap50_release_early` gives up its entire advantage and ends *worse than
   the old ramp*; `cap50_release` follows one band later; `cap50_release_late`
   barely releases inside 300 votes and stays with `cap50`. The ordering is
   monotone in release point. That is as clean a refutation of "at some point
   just use the threshold you worked for" as this design can produce.
2. **`cap50`'s advantage *grows* with depth** (−0.019 → −0.051), rather than
   decaying as the learned cut converges.
3. **`pure_xcal` does eventually beat `prod`** — crossing zero around 51–100
   votes and reaching −0.0123 by 201–300. But this is *not* the learned cut
   winning: past 20 votes `prod` **is** pure x-cal (its ramp has weight 1), so
   the two arms run an identical rule and differ only in the trajectory their
   different early behaviour produced. It is an acquisition effect, not a
   threshold-policy one.

### Why: 300 clicks buys ~13 positives

The convergence argument is about **labelled positives** — the conformal rule
needs both tails and the positive tail is the scarce one. At realistic
prevalence, votes and positives come apart badly:

> After **300 votes**, the median cell holds **~13 positives**. Only
> `building` (39% prevalence) exceeds 48.

So banding on votes conflates "the learned cut has converged" with "the user
clicked a lot and still has almost nothing to calibrate on". Banding on
positives instead shows the convergence the theory predicts — and how far away
it still is:

| vs `prod` | 1-3 pos | 4-6 | 7-10 | 11-15 | 16-25 | 26+ |
|---|---|---|---|---|---|---|
| `cap50` | −0.0153 | −0.0578 | −0.0442 | −0.0356 | −0.0424 | −0.1333 |
| `pure_xcal` | +0.0959 | +0.0493 | +0.0346 | +0.0192 | **+0.0140** | +0.0264 |
| **`cap50` − `pure_xcal`** | −0.1113 | −0.1071 | −0.0780 | −0.0550 | **−0.0479** | −0.1449 |
| p (Wilcoxon) | 1e-40 | 1e-34 | 2e-26 | 4e-15 | 2e-05 | 3e-05 |

`pure_xcal`'s deficit shrinks monotonically from +0.096 to +0.014 as positives
accumulate — **the consistency argument is visibly true**. The blend's edge
narrows with it (−0.111 → −0.048). But it is still large and still highly
significant at 16–25 positives, and at that rate of closure you would need on
the order of 100+ positives to reach parity — **thousands of clicks at typical
prevalence**. The 26+ band reverses the trend, but it is 19 cells dominated by
one category and should be read as underpowered, not as a finding.

**So the theory is right and the recommendation is unchanged.** `cap50` stands
for binary voting, not because the GMM is better in the limit — it isn't — but
because a real labelling session never reaches the limit. The right way to state
it is not "never hand over" but **"do not hand over before ~100 positives, which
a session will not reach"**.

### What would change this

If autopilot ever accumulated positives much faster — a better Good phase, a
prevalence-aware acquisition rule, or class-balanced sampling — the crossover
would move into reach and `cap50_release_late` (or a positive-count-triggered
release) would become the right schedule. That is a more promising direction
than tuning the schedule further: the schedule is compensating for a data
problem, and fixing the data problem is worth more than the compensation.
A release keyed on **positives** rather than votes is the natural next arm.

## The horizon problem: region voting

Same design at 200 votes, 14 deep categories, 6 seeds, 6 arms. The region answer
is **stronger** than binary's and it overturned the first-pass recommendation.

`d_cost` vs `prod` by vote band:

| schedule | 7-20 | 21-50 | 51-100 | 101-200 |
|---|---|---|---|---|
| **`slow_cap50`** | **−0.0577** | **−0.0856** | −0.0937 | −0.0822 |
| `cap50` | −0.0260 | −0.0748 | **−0.0970** | **−0.0834** |
| `slow` | −0.0577 | −0.0417 | −0.0008 | **+0.0077** |
| `pure_xcal` | +0.0753 | +0.0168 | +0.0059 | +0.0046 |

**`slow` — the first-pass region winner — decays to nothing.** 40 labels is
exactly where it becomes pure x-cal, so past that band it *is* the losing arm.
The 30-vote study measured it right up to the point it falls apart. `cap50` does
the opposite, strengthening from −0.026 to −0.097.

`slow_cap50` is the synthesis: `slow`'s gentler early ramp (which is why it won
the early window) with `cap50`'s cap (which is why that one keeps winning). It
is best or tied in every band, wins the 21-50 band outright, and **dominates
`cap50` at every positive count**:

| vs `prod` | 1-3 pos | 4-6 | 7-10 | 11-15 | 16-25 | 26+ |
|---|---|---|---|---|---|---|
| **`slow_cap50`** | **−0.0285** | **−0.0628** | **−0.0776** | **−0.0763** | **−0.0846** | **−0.1051** |
| `cap50` | −0.0150 | −0.0444 | −0.0593 | −0.0732 | −0.0832 | −0.1017 |
| `slow` | −0.0205 | −0.0422 | −0.0326 | −0.0072 | −0.0030 | +0.0016 |
| `pure_xcal` | +0.0808 | +0.0663 | +0.0306 | +0.0281 | +0.0078 | +0.0037 |

It is also the opposite of permissive: under `fpr x4` it **improves**, −0.0577 →
−0.0745. (That sensitivity is computed on the headline 7-20 window, where
`slow_cap50` and `slow` are identical by construction; the depth advantage is
the band table above.)

### Region and binary fail differently

On **binary**, `cap50`'s edge over `pure_xcal` *narrows* as positives accumulate
(−0.111 → −0.048) — the learned cut is slowly converging, exactly as theory
says, just far too slowly to matter in a session.

On **region** it *widens* (−0.015 → −0.102), and `pure_xcal` flattens at +0.004
without ever crossing. That is the signature of a bias more labels cannot fix.
The likely cause is the scoring geometry: region scoring max-pools over ~196
patches, so the conformal cut is calibrated against an extreme-value
distribution and is biased **structurally**, not statistically. If that is right
the learned cut is inconsistent here *too*, and the GMM half is correcting a
model-misspecification error rather than a small-sample one — which would
explain why more data never rescues it. This is the study's most interesting
open thread and is worth a dedicated experiment.

### The synthesis does not transfer to binary

`slow_cap50` also beats `cap50` on binary at the headline (−0.0322 vs −0.0188)
— but it **flips to +0.0414 under `fpr x4`** while `cap50` holds at −0.0236. Its
extra binary gain is a lower cut, not better calibration (d_fnr −0.057 against
d_fpr +0.025, versus `cap50`'s near-Pareto −0.017/−0.002). Binary keeps `cap50`.
The same curve is genuinely better calibrated on region and merely more
permissive on binary, which is the sharpest evidence in this report that the two
modes are different problems rather than one problem with two datasets.

## Long-run design

`cap50` never hands over to the learned cut. **That cannot be literally right,
and this study could not have seen why.**

The GMM midpoint is an **inconsistent** estimator of the decision cut: it reads
no labels, so its error floors out at whatever its two-component symmetry
assumption gets wrong, however much data arrives. The cross-calibration cut is
**consistent** — more labels tighten the conformal quantile toward the right
threshold. Asymptotically pure x-cal *must* win. So the real question was never
*whether* to hand over but **when**, and every number above was measured over a
**30-vote** horizon, where the answer is "not yet" almost by construction.

The region-voting long run (200 votes, 5 arms including `slow` and
`cap50_release`) is still going; its bands will be added here. The binary
result above is the one the question was about.

Design of both long runs:

- **Deep categories only.** A long horizon is bounded by *positives in the
  simulation set*, not pool size: once autopilot exhausts them, every further
  vote is a negative and the conformal positive-quantile stops improving. With
  a floor of 50 sim positives, 14 VG and 12 COCO categories survive; the
  shallow ones (giraffe ~12, scissors ~14) cannot sustain the horizon and would
  contribute noise dressed as a plateau.
- **300 votes on binary voting** (`prod`, `cap50`, `pure_xcal`, and three
  cap-then-release arms bracketing the handoff at 30→100, 50→200, 150→400),
  **200 votes on region voting** (cheaper arm set; region cells cost ~10x).
- **Reported per vote band** (7-20, 21-50, 51-100, 101-200, 201-300) rather than
  as one average, because an average over a crossover is exactly the number that
  hides it.

If a release arm beats both `cap50` and `pure_xcal`, the shipped binary schedule
should become that arm. Until then `cap50` stands only as *the best schedule
over the first 30 votes*, which is where the evidence is.

## Limitations

- **COCO carries only binary voting.** The #2790 cache has whole-image and HAC
  region vectors but not the raw patch grid `max_patch` pools over, so the
  region-voting verdict rests on Visual Genome alone. A COCO region arm needs a
  real re-embed; until then, `slow`-for-region is a single-dataset result — the
  weakest link in this report.
- **The reweighting is a scoring sensitivity.** Moving the Inclusion knob
  changes the conformal rule *and* the weights, so `fpr x4` approximates an
  Inclusion-averse user rather than reproducing one. It was also added
  post-hoc.
- **One head, one inclusion, 30 steps.** Everything above is the production
  linear head at inclusion 0 over the first 30 votes — which is why `cap50`'s
  "never hand over" must be read as "not within 30 votes". See *The horizon
  problem* above; the follow-up run addresses exactly this.
- **A transient full-disk incident** on the shared `/exp/sgreenberg` volume
  killed ~950 cells mid-run. All were re-run; the 49 files it left behind were
  zero-byte rather than partially written (verified by field-count validation
  across every cell in every arm), so no cell contributed truncated data.

## Follow-ups

- **Re-embed COCO with patch grids** so region voting has a second dataset.
- **Why does `pure_gmm` lose at `fpr x4` on region voting** despite being the
  closest to the oracle cut and the steadiest? The bias/spread decomposition
  does not explain it; the likely answer is score-density curvature near the
  cut, which the harness could measure directly.
- **`rare` deserves another look.** It is the best-calibrated cut on binary
  voting and essentially unbiased, but its endpoints (1→8 rare-class labels)
  were guessed, not tuned. A tuned rare-class ramp may dominate `cap50`.
  **Tracked in #3551**, with the `corridor` item below.
- **`corridor` is not dead, it is untuned.** Clamping to the component means is
  a no-op on region voting (p=0.62) because the corridor is far wider than the
  x-cal error; a tighter corridor (a fraction of the mean gap) is the version
  worth testing. Note also that `corridor_ramp` releases its clamp entirely past
  its endpoint, which is discontinuous — see the class docstring.

## Provenance

- Screen: `/exp/sgreenberg/mixin-2841/results-screen/` (`REPORT_screen.md`,
  `screen_deltas.csv`, `screen_sensitivity.csv`).
- A/B: `/exp/sgreenberg/mixin-2841/results-ab/<arm>/`.
- Cached embeddings reused, nothing re-embedded: VG + Caltech pickles from the
  Max-Patch run (`/exp/sgreenberg/max-patch/datadir/embeddings/`), COCO from the
  #2790 sweep cache (`/exp/sgreenberg/threshold-stability/cache/regions/coco/`).
- Analyzer self-test (`selftest_analyze_mixin.py`) recovers planted effects,
  their magnitudes and signs, a null, an opposite-per-mode split, the window
  exclusion, and the fidelity abort.
