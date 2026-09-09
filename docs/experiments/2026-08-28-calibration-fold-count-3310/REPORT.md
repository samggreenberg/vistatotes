# Does more cross-calibration ever pay? Fold count vs its wall-clock price (#3314)

Executes [`PLAN.md`](PLAN.md), pre-registered before the run. Every decision rule
below was fixed at submission time; this report records the verdict they produce.

## The answer, in six lines

**Keep `calibrate_count = 2`.** The pre-registered gate is **closed**, and it
closed on **cost, not on benefit**:

- More folds **do** help, by more than the 0.005 margin, and where the theory
  said they would — early, and only on the DINOv3 geometries. At 1–25 votes,
  region voting: **−0.0057 ± 0.0012** cost at K=6.
- Every fold count that clears the margin costs **≥ 2.3×** the user's per-step
  retrain. The cheapest one that helps is 2.30×; the ceiling is 1.5×.
- The adaptive schedule the issue proposes **cannot rescue it**, because the
  premise it rests on is false here (see [The complementarity premise is
  wrong](#the-complementarity-premise-is-wrong)). All 18 members of the
  pre-registered family fail the same rule.
- **K=3 is the near miss**: affordable in every band (1.27–1.32×), a real
  4σ improvement — and **−0.0025 ± 0.0006**, half the margin. The margin
  rejects it, not the data.
- The effect follows the **embedder**, not the voting mode — the confound-breaking
  corner earned its cells.

**The blocking term is named and it is not the fold fits.** 86% of what a fold
costs is the anchored EM. Make that cheaper and K=6 lands at ~1.2×, inside the
ceiling, with a benefit this run has already measured. That is the follow-up.

## What was run

One stage-A run, the nested screen; stage B was gated off and is not in this
report (that is a result, not a shortfall — see [The gate](#the-gate)).

| axis | value |
|---|---|
| dataset | `vg_scale_any`, 12 hand-checked classes × 300 positives, one shared negative pool (identical prevalence in every cell) |
| geometries | `siglip/whole_image`, `dinov3_patch/whole_image`, `dinov3_patch/max_patch` |
| opening | SigLIP text sort in every cell (`CALIB_REQUIRE_OPENING=text`, asserted per cell and per grid) |
| live count | `calibrate_count = 2` — production's trajectory, so every counterfactual K is scored on the same votes |
| screen grid | `CALIB_FOLD_COUNTS = 1,2,3,4,6,8` |
| seeds × steps | 4 × 150, cell order `seed` |
| head / path | defaults: linear-SVM head, fused thresholds, per-mode blend schedule, per-space split fraction |
| cells | **96 / 96**, zero failed, zero zero-byte, zero header-only |
| rows analysed | 1,707,732 from 96/96 cells; 144 (category, seed, geometry) cells |

Nothing was dropped, so there is no survivorship to report. Launcher
[`launch_folds_3314.sh`](../../../scripts/experiments/calibration/launch_folds_3314.sh),
analyzer
[`analyze_folds_3314.py`](../../../scripts/experiments/calibration/analyze_folds_3314.py),
its planted-answer test
[`selftest_analyze_folds_3314.py`](../../../scripts/experiments/calibration/selftest_analyze_folds_3314.py).

Every number is printed to three significant digits — the decision constants are
0.005 and 0.01, so the third digit is where the rules are read — beside a
standard error bootstrapped over **cells**, never steps.

## What a fold actually costs, and why the old column could not say

This study's third rule is a latency, and a latency has to be measured against
something a user waits for. The obvious column, `fold_seconds`, is not it: it
prices the fold **fits** plus the conformal rule's overhead. The shipped
threshold also scores the sim set once per fold — so each fold's mixture can be
anchored on its own haystack — and fits one anchored EM per fold. Both scale
with K; both are paid inside `_safe_threshold_for_step`, which no timing column
in the frame covered.

Measured per fold on this study's own sizing cell:

| term | s/fold | share |
|---|---|---|
| fold fit | 0.010 | 7% |
| fold haystack scoring | 0.010 | 7% |
| **anchored EM** | **0.128** | **86%** |

So `fold_seconds` reports 0.079 s at K=8 where calibration costs 1.16 s — **15×
under**. Read through the old column every fold count sits comfortably under any
ceiling and this report says the opposite thing. #3314 therefore adds
`fold_fit_seconds`, `fold_score_seconds`, `anchored_seconds` and their sum
`cal_seconds` beside `fold_seconds`, which is left untouched so #2897's and
#3115's numbers keep meaning. Recorded as
[`lessons/2026-08-28-the-column-named-fold-seconds-was-a-third.md`](../../../scripts/experiments/lessons/2026-08-28-the-column-named-fold-seconds-was-a-third.md).

**The denominator is the app's retrain, not the harness cell.** A screen step
also computes six fold counts × eight arms of counterfactual rows, and a user
waits through none of it — the cell runs 4.7 s/step where the app's retrain
inside it is 0.33 s. Dividing by the cell would report every K as nearly free.
So the step is reconstructed from the pieces the app itself performs: fit the
head, score the haystack, calibrate, score the pool. `test_score_seconds` is
excluded — scoring a held-out test set is eval-only work no app step does, and
leaving it in would flatter every fold count.

## The benefit, per band

Paired **within the step** — the fold counts are nested prefixes of one Kmax
calibration, so every K re-cuts the same votes, the same final model and the same
held-out test scores — then collapsed to cell means before any resampling.
Negative is better. `cost` is the headline; `regret_honest` agrees to the third
digit everywhere and is in [`agg/paired_regret_honest.csv`](agg/paired_regret_honest.csv).

**`dinov3_patch/max_patch` (region voting)**

| band | K=1 | K=3 | K=4 | K=6 | K=8 |
|---|---|---|---|---|---|
| 1–25 | +0.0083 ± 0.0014 | −0.0025 ± 0.0006 | −0.0039 ± 0.0010 | **−0.0057 ± 0.0012** | **−0.0066 ± 0.0014** |
| 26–60 | +0.0050 ± 0.0022 | −0.0022 ± 0.0009 | −0.0043 ± 0.0012 | **−0.0051 ± 0.0016** | **−0.0054 ± 0.0019** |
| 61–100 | +0.0076 ± 0.0021 | −0.0012 ± 0.0007 | −0.0010 ± 0.0009 | −0.0008 ± 0.0012 | −0.0017 ± 0.0012 |
| 101–150 | +0.0017 ± 0.0014 | −0.0014 ± 0.0007 | −0.0019 ± 0.0008 | −0.0028 ± 0.0010 | −0.0033 ± 0.0012 |

**Bold** = clears both the 0.005 margin and 2 SE, which only the two early bands
do.

The horizon shape is close to the prediction but **not monotone**, and the
difference is worth stating rather than smoothing: the benefit is largest at 1–25
votes (−0.0057), collapses to nothing by 61–100 (−0.0008), and then **partially
returns** in the deep band (−0.0028). Variance reduction being worth most where
per-fold noise is largest predicts the first two; it does not predict the third.
With ±0.0012 on each, the late-band minimum and the deep-band recovery are
individually about 2 SE apart, so the late-band dip and the deep-band
recovery are worth naming and not worth building on. What matters for the decision is unaffected: the margin is cleared
early and nowhere else.

`dinov3_patch/whole_image` is the same story at the decision point (early K=6
**−0.0056 ± 0.0016**) with a flatter middle (−0.0038 mid, −0.0041 late, −0.0013
deep). `siglip/whole_image` never clears the margin at any K in any band — its
best is −0.0019 ± 0.0007, and at 61–100 votes it is very slightly *worse*
(+0.0004 ± 0.0007). Full table: [`agg/paired_cost.csv`](agg/paired_cost.csv).

### The effect follows the embedder, not the voting mode

The PLAN's own hypothesis was that the live question is region voting, "where
per-fold variance is much larger". That is **refuted**, and the middle corner is
what refutes it:

| contrast | holds fixed | early-band Δ at K=6 |
|---|---|---|
| `dinov3/whole` vs `dinov3/max_patch` | the embedder | −0.0056 vs −0.0057 — **no difference** |
| `siglip/whole` vs `dinov3/whole` | the voting mode | −0.0015 vs −0.0056 — **the whole effect** |

Switching the voting mode changes nothing; switching the embedder changes
everything. Without the `dinov3_patch/whole_image` cell this would have read as a
law about region voting, which is the confound #3115 and #3258 were caught by and
which #3287 found one knob over — its optimum followed the embedder too.

## The price, per band

`step_ratio` is the median over steps in the band of this K's whole per-step
retrain over K=2's on the same step — paired within the step, and a median
because one scheduler stall on a shared cluster moves a mean and cannot move a
median. `cal_share` is the fraction of a retrain the user spends calibrating.

**`dinov3_patch/max_patch`**

| band | cal_share at K=2 | K=3 | K=4 | K=6 | K=8 |
|---|---|---|---|---|---|
| 1–25 | 0.66 | 1.32 | 1.65 | 2.30 | 2.95 |
| 26–60 | 0.63 | 1.30 | 1.61 | 2.21 | 2.82 |
| 61–100 | 0.60 | 1.29 | 1.57 | 2.14 | 2.72 |
| 101–150 | 0.58 | 1.27 | 1.54 | 2.07 | 2.61 |

Ceiling is 1.5×. Only **K=3** is affordable, in every band. Full table:
[`agg/cost_ratios.csv`](agg/cost_ratios.csv).

Note the first column: **calibration is already 58–66% of a retrain at
production's K=2** on the region geometry, and 87–95% on the single-vector ones.
That is the fact that makes any extra fold expensive — there is no cheap
headroom to spend.

### The complementarity premise is wrong

The issue's proposal rests on a complementarity: *the folds that help most (few
votes) are also the cheapest to fit, because fit cost grows with the labelset*.
The first half is true — the benefit is early. **The second half is false.**

Read the price table across a row, not down a column: at K=6 the ratio is 2.30×
early and 2.07× deep. It is **flat, and slightly worse early**. The reason is the
cost breakdown above: the dominant per-fold term is the anchored EM on the
**haystack**, which is 7,747 media at 5 votes and 7,747 at 150. Only the fold
fit — 7% of the cost — scales with the labelset, and the labelset is tiny exactly
where the schedule wants to spend.

So there is no cheap-early regime for an adaptive count to exploit. Every member
of the pre-registered family `K(n_votes) = K_early while n_votes < N_cut, else 2`
(`K_early ∈ {4,6,8}` × `N_cut ∈ {25,60}` × 3 geometries = 18 arms) pays 1.65–3.83×
in exactly the bands where it raises the count, and **all 18 fail rule 3**.
[`agg/schedule_family.csv`](agg/schedule_family.csv).

This is the study's most useful negative result: the issue's mechanism is sound
and its cost model is not.

## The mechanism, measured directly

`sd(threshold)` across seeds at a fixed step, averaged over steps — the
variance-reduction claim, observed on the shipped cut rather than inferred from a
regret decomposition (#3116 established that decomposition cannot answer it).

`dinov3_patch/max_patch`, band 1–25:

| K | 1 | 2 | 3 | 4 | 6 | 8 |
|---|---|---|---|---|---|---|
| sd(threshold) | 0.0261 | 0.0222 | 0.0209 | 0.0203 | 0.0196 | 0.0191 |

Monotone in K, in **every** band and **every** geometry
([`agg/sd_threshold.csv`](agg/sd_threshold.csv)). The combined quantile is a mean
of K per-fold statistics, so its variance should fall like 1/K — and it does not:
K=2 → K=8 should halve the sd and instead drops it 14%. That gap is the finding
behind the finding. The anchored cut's variance has a **K-independent term** —
each fold's mixture is dominated by its κ=0.3 haystack anchor, which every fold
shares — so averaging more folds shrinks only the small half. That is why the
benefit saturates by K≈6 and why it is worth thousandths of cost rather than
hundredths, and it is the same mechanism the laptop bench (#3310) saw as
near-K-invariance on single-vector geometries.

**Degenerate folds are not the explanation.** A fold that saw one class
contributes no cut, so a K that looks like 8 can be a 5 — and a null at high K
would mean something different if it were. It is not: `n_folds_used` is within
1.3% of K in every (geometry, band, K), and the largest shortfall is at **K=1**
(0.988 of K, `dinov3_patch/whole_image`, 1–25 votes), not at the top of the grid.
No arm is quietly running on fewer folds than it claims
([`agg/folds_used.csv`](agg/folds_used.csv)).

## The gate

Pre-registered: stage B is booked only if some K clears the margin in some band
*within* the cost ceiling. Applied mechanically:

```
gate_open   = false
reason      = 4 (geometry, K) pairs beat the margin, and every one of them failed
              the 1.5x step ceiling (cheapest was 2.30x); no schedule cleared it
              either. The benefit is real and priced out of reach.
```

So stage B was not run, and the report ships without it — one of the outcomes
[`PLAN.md`](PLAN.md) names in advance.

**One correction, made after seeing the results, and it changed nothing.** The
first draft of `pick_stage_b` required a fixed-K ship candidate before it would
consider a schedule at all, which is stricter than the PLAN's band-local gate —
a schedule pays the ceiling only where it raises the count, which is the entire
reason the adaptive arm exists. Fixed, with a selftest that plants the case. The
verdict is identical either way, because no schedule cleared rule 3 on its own
merits. Flagged because a decision rule edited after the data is seen has to be.

**What the closed gate does and does not settle.** Stage A holds the trajectory
fixed, so it cannot see the votes a different K would have collected; that is
what stage B measures and it is unmeasured here. A *cost* rule cannot be
overturned by acquisition feedback — 2.30× is 2.30× however the votes fall — so
the gate's closure is safe. The *benefit* could be larger live than screened, and
that remains open. It would have to be ~2.5× larger to change the answer, since
the affordable K=3 would need to clear 0.005 from −0.0025.

## Figures

![cost over clicks](figures/cost_vs_clicks.png)

One panel per **geometry** (there is one dataset, and `max_patch` must never be
averaged with `whole_image`); colour is the fold count; the band is the
inter-quartile range over cells. **Click 0 is the free SigLIP text sort** — 0.387
in all three panels, since every cell opens the same way — so the distance from
the left marker to the right end is what the clicking bought. A line is dashed
where it describes fewer than 95% of that arm's cells. It does **not** license
comparing across panels: the three geometries have different absolute costs for
reasons that have nothing to do with this knob.

**Read this figure for the honest scale of the effect.** All six fold counts lie
on top of each other, inside an inter-quartile band an order of magnitude wider
than the difference between them. The paired within-step contrast resolves
−0.0057 at 4.6σ and a user would not see it. Both statements are true, and a
report that showed only the first would be misleading.

The same figure also prices the *loop*, which is a bigger number than anything K
does — at K=2, cost against the free text sort:

| geometry | click 0 | first trained (t=5) | crossover | t=150 |
|---|---|---|---|---|
| `dinov3_patch/max_patch` | 0.387 | 0.430 | **9 clicks** | 0.231 |
| `siglip/whole_image` | 0.387 | 0.674 | **23 clicks** | 0.310 |
| `dinov3_patch/whole_image` | 0.387 | 0.735 | **81 clicks** | 0.364 |

`dinov3_patch/whole_image` spends 81 clicks getting back to what typing the query
was already worth, and ends 150 clicks later barely past it. That is not this
study's question, but it is the axis this study's effect has to be read against.

![average precision over clicks](figures/average_precision_vs_clicks.png)

Per-run versions — every seed its own line, one file per geometry — are in
[`figures/`](figures/).

**Every other slice — per category, per seed, every metric the run emitted — is
in the interactive viewer: [`viewer.html`](viewer.html).**

A zoomable reading copy of this report, figures inlined, is beside it at
[`report.html`](report.html)
— generated from this file by
[`make_bench_html.py`](../../../scripts/experiments/calibration/make_bench_html.py),
never hand-written, so the two cannot drift.

## The steps themselves

Aggregates say a fold count is worth thousandths of cost; only rows say what that
looked like. [`agg/worked_examples_k8.csv`](agg/worked_examples_k8.csv) carries
the individual steps where K moved the cut most, in either direction — named
category, seed, click, both thresholds, both operating points, and the folds each
arm actually used.

## Secondary: #2897's monotone worsening does not reproduce

The PLAN kept the pooled-rule rows (`folds_k{K}_xcal`) as a replication check:
#2897 found the pooled combine rule getting monotonically *worse* with K, and
#3115 explained why (pooling K folds' scores builds a mixture of K models'
distributions and reads an extreme order statistic of it, so the target moves
with K).

Under a text-sort opening on this dataset the pooled rule gets **better** with K,
substantially, in every geometry and every band — `siglip/whole_image`, 1–25
votes: **−0.038 ± 0.006** at K=8. The old result does not reproduce.

This is not a contradiction of the mechanism, and it is not a decision input —
the pooled rule is not shipped. The bench in #3310 predicted the drift's sign is
*regime-dependent*: it helps at small n and hurts at moderate n on cleanly
separated classes. #2897's grid landed on the hurting side; this one lands on the
helping side throughout. What it does mean is that #2897's headline is a fact
about its conditions — different dataset, crop-seeded opening, pre-#3198 head,
pre-#3308 exclusion — and not a property of the combine rule.
[`agg/pooled_replication.csv`](agg/pooled_replication.csv).

## What this does not license

- **It is one dataset.** `vg_scale_any` was chosen because uniform prevalence is
  what makes a calibration question answerable, and that same choice means the
  result has not been seen at any other prevalence.
- **The cost ratio is not a constant of nature.** It is measured against a
  7,747-media haystack. The anchored EM's cost scales with the haystack and the
  fold fit with the labelset, so a *larger* corpus makes extra folds relatively
  **dearer**, not cheaper, and a smaller one cheaper. The 1.5× ceiling binds at
  K=3 here; where it binds elsewhere is unmeasured.
- **"Follows the embedder" rests on two embedders**, one of which
  (`dinov3_patch`) has no text tower and reaches its opening through the
  `siglip+dinov3_patch` pair. It is the most economical description of these
  three geometries, not an established law.
- **The margin is a choice, not a measurement.** K=3 is affordable and its
  −0.0025 ± 0.0006 is resolved at 4σ. It is rejected by a 0.005 threshold
  pre-registered before the data existed. Someone who thinks half a
  percentage point is the wrong bar should read the table, not this sentence.
- **No `rule_inefficiency` / `calibration_shift` claim is made at any K**, per the
  PLAN and #3116: those two terms sum to regret by construction and their
  reference is estimated from the very set K resizes.

## The follow-up this run earns

The blocking term is measured and it is one function: `fit_fold_anchored_cut`, at
0.128 s per fold, 86% of a fold's cost. Everything else about K is affordable.

If the anchored EM per fold were made ~4× cheaper — a warm start from the
previous step's mixture, a shared haystack fit across folds, or a cheaper
E-step — K=6's step ratio falls from 2.30× to roughly 1.2×, inside the ceiling,
with a benefit this run has **already measured** at −0.0057 ± 0.0012 early. That
is a much better-posed piece of work than re-running this grid, and it does not
need a new study to justify it: the two numbers it needs are both in this report.
**Tracked in #3558.**

## Reproducing

```bash
export VTS_REPO=/exp/$USER/projects/vts-folds-3314
cd $VTS_REPO/scripts/experiments/calibration
bash launch_folds_3314.sh prepare     # stage 0, once
bash launch_folds_3314.sh baseline    # the click-0 anchor
bash launch_folds_3314.sh size 0      # time a real cell before committing
bash launch_folds_3314.sh size 12     # ...and a real PAIR cell, which sets the limits
bash launch_folds_3314.sh screen      # stage A: the nested screen
bash status_folds_3314.sh             # what is true, read off disk
```

Measured on this grid, and what the array was sized from:

| cell | styles | elapsed | peak RSS |
|---|---|---|---|
| `vg_scale_any × siglip` | `whole_image` | 11m52s | 0.80 GB |
| `vg_scale_any × siglip+dinov3_patch` | `whole_image` + `max_patch` | 1h27m53s | 5.86 GB |

96 cells at `--mem 12G %72` (864 G of the 1,074 G per-user allowance); the array
drained in 1h40m. A cell runs every style of its embedder in one task, which is
why the pair cell is 7× the other and why it, not the binary cell, sets every
limit.

Stage B, had the gate opened, is
`bash launch_folds_3314.sh ab <K_BEST> <K_EARLY>@<N_CUT>` — two full live runs
against the screen's own K=2 trajectory, using the eval-only
`CALIB_FOLD_COUNT_SCHEDULE` knob added for it. The knob and its tests are in the
tree and unused: `calibrate_count` resolved per step from a schedule, off by
default, byte-identical to a constant when it never fires.
