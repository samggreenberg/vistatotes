# Which cut rule should answer the Inclusion knob? (issue #2865)

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


**Verdict: keep `mid_tilt`.** No candidate both delivered more of the knob than
the shipped rule and stayed within the pre-registered regret tolerance at every
stop of it. That is the outcome
[the plan](../../plans/population-anchored-calibration.md) flagged as the honest
possibility — *"`mid_tilt` may already be delivering most of the slider, in
which case the honest outcome is keep it"* — and the sweep says it is delivering
**95%** of it.

Three things this run settles that neither calibration run before it could,
because both scored every arm at inclusion 0:

- **The bug the issue reports is real and expensive.** The bare `mid` cut shipped
  by #2861 admits **exactly one set for the whole slider** — 13 stops, one
  answer, in **65,671 of 65,671** measured cell-steps across all four
  environments — and away from
  inclusion 0 it costs up to **+0.18±0.02** regret. #2868's `mid_tilt` was not a
  cosmetic repair.
- **The tilt itself is free.** `mid_tilt` and `rate` differ by a *constant offset
  in fold-quantile space* — that is exact, not empirical — so this sweep is a
  re-measurement of the **inclusion-0 choice** at thirteen different cost
  weightings. Above inclusion 0, `rate`'s gap to the incumbent never exceeds
  0.005 at any stop in any environment; its two largest gaps anywhere are a
  **+0.015±0.002** at k=0 on `coco_val × dinov3_patch` and a −0.021±0.005 at
  k=−1 on `coco_val × siglip`. The first is the only *material* loss on the
  table, and it is precisely the inclusion-0 contrast the anchor-mass run
  already decided in `mid`'s favour — reproduced here on a different detector
  head and a different environment.
- **Candidate 3 is dead and candidate 2-as-written is a lead, not a rule.**
  `q_tilt` is worse than the incumbent — averaged across the knob — at every one
  of the five step sizes in every environment, and buys knob only by giving up
  accuracy. `cross_tilt` is genuinely *better*
  below inclusion 0 (down to **−0.034±0.005**) and genuinely worse above it (up
  to **+0.073±0.012**) — an asymmetry worth a follow-up, not a rule to ship.

---

## 1. What the issue asked, and what changed under it

[#2865](https://github.com/samggreenberg/VTSearch/issues/2865) observed that
shipping #2864's recommendation verbatim (`κ=0.3`, cut rule `mid`) made the
Inclusion slider a **no-op** for every detector with usable calibration folds:
`mid` is the midpoint of two fitted component means, and it never looks at the
cost weights inclusion arrives as.

Two things moved before this run:

- **#2868 shipped candidate 1** (`mid_tilt`: the measured midpoint at inclusion
  0, rate-rule tilt away from it), so the incumbent is no longer `mid`. `mid`
  survives here as candidate 4, the honest inclusion-blind null.
- **#3124 disproved candidate 2's premise.** The candidate rests on the
  anchor-mass report's mechanism (*"`mid` ignores the mixture weights; `rate`
  needs them"*), and `rate` does not read them: the prior-odds factor in its
  `lam` cancels the `w_lo/w_hi` inside `_rate_cut`'s offset exactly. So
  candidate 2 *as described* is `rate`, already an arm; candidate 2 *as written*
  (`lam = fnr/fpr`, prior odds retained) is a different rule, and it runs here as
  `cross_tilt` so the issue's own text gets priced.

The arm set is therefore `mid` · `mid_tilt` · `rate` · `cross_tilt` · `q_tilt`,
with `q_tilt`'s free step size expanded over five values.

## 2. What was measured, and how faithfully

Every arm is a **re-cut of the estimator production caches**, not a
re-implementation of it. `_cut_inclusion_rows` fits the fold-anchored mixture
once per step through the app's own `fit_fold_anchored_cut`, then asks
`FoldAnchoredCut.threshold_at(k)` for each rule at each stop of the knob — which
is exactly what `recompute_detector_thresholds_for_inclusion` does when a user
drags the slider: no EM, no scoring pass, arithmetic on a cached fit. The two
eval-only candidates (`cross_tilt`, `q_tilt`) are arms *on the production
estimator*, added inside `gmm_cut_from_fit`, so the sweep cannot drift from what
ships. `tests_lib/detectors/test_cut_inclusion_sweep.py` pins the identity that
licenses the rest of the table: `mid_tilt` and `q_tilt` reproduce the measured
`mid` arm bit-for-bit at inclusion 0.

The trajectory those arms are re-cut on is the shipped one:

| knob | this run | why |
|---|---|---|
| detector head | `linear_svm` | PR #3198 made the linear SVM the shipped head, and an unset `CALIB_HEAD` resolves to `PRODUCTION_HEAD`. The launcher was drafted on 2026-08-12 with `CALIB_HEAD=linear` pinned, which *was* production then; carrying that pin forward would have measured the cut rule on a head no user has. |
| anchor mass κ | 0.3 | `FOLD_ANCHOR_WEIGHT`, shipped. Pinned, not swept: re-opening the anchor mass would confound it with the cut rule. |
| incumbent rule | `mid_tilt` | `FOLD_ANCHOR_CUT_RULE`, read off the module by the analyzer rather than hard-coded |
| fold combine | `qmean` | `FOLD_ANCHOR_COMBINE`, shipped |
| calibration folds | 2 | production |
| acquisition offset | −1 | `ACQUISITION_INCLUSION_OFFSET`, shipped |
| blend schedule | per-mode default | `slow_cap50` region / `cap50` binary (#2841) |
| patch geometry | `max_patch` only | the production patch pipeline; #2886 removed the region tree from ingest, so the HAC hybrids would have priced geometry no user gets |
| safe thresholds | on | the setting was deleted rather than left as a way to opt into a worse threshold (#2799) |

**Regret is reported on the rate scale.** `inclusion_cost_weights` *doubles* one
of the two cost weights per step of the knob, so a raw cost at k=10 is
denominated in units 1024× the ones at k=0. Pooled in those units, every summary
number is the k=±10 pair and a fixed tolerance means "a thousandth of an error
rate" at one end of the slider and "a whole error rate" at the other. Every
regret here is divided by `2**|k|`, the larger of the two weights — exactly 1 at
inclusion 0, so nothing the previous calibration studies measured moves, and
elsewhere a weighted mean of FPR and FNR that is bounded like a rate.

## 3. The grid

Four environments, chosen so the two the issue asks for are genuinely what it
says. Region voting needs **both** halves — ground-truth boxes (the dataset) and
a patch grid (the embedder) — and `preflight.sh --require-region-voting` asserted
the premise per arm rather than trusting the flag (`patch_grid` on 4193/4193 and
4952/4952 medias respectively):

| environment | voting | cells | deep cell-steps |
|---|---|---|---|
| `visual_genome_m × dinov3_patch × max_patch` | **region** | 92 | 18,062 |
| `coco_val × dinov3_patch × max_patch` | **region** | 74 | 14,782 |
| `visual_genome_m × siglip × whole_image` | binary | 90 | 18,060 |
| `coco_val × siglip × whole_image` | binary | 74 | 14,767 |

`siglip` (not `siglip2_l`) on the binary arms because it is the shipped default
embedder: the knob is a user-facing control on *every* detector, so a rule that
restores it on region voting while wrecking binary voting is not shippable.

**336 cells defined, 336 completed, 0 zero-byte, 6 contributed no rows.** The six
are `visual_genome_m` `ball`/s1 and `eye`/s0, and `coco_val` `refrigerator`/s0
and `sports ball`/s1 on **both** embedders — four produced no trajectory at all
and two produced a trajectory that never formed a fold-anchored fit. They are
seed-specific rather than category-wide (the same categories ran fine at other
seeds), and they are not the smallest categories in the grid — `giraffe`, with 24
positives, produced full cells. 11,055,447 cut-inclusion rows were read; the
tables below use the 7,683,507 of them past 100 votes, where the cut rule rather
than the anchor supply is what is being measured.

## 4. (a) Regret across the knob

Paired against the incumbent at each stop, per environment; **negative favours
the challenger**, ± is a bootstrap SE over cells (the resampling unit is a cell,
not a step: consecutive steps of one trajectory share a model).

Four environments × seven of the thirteen stops. The full grid — all thirteen
stops, all nine arms, with `d_regret_cost` beside each rate-scale figure — is
committed here as
[`cutincl_regret_vs_incumbent.csv`](cutincl_regret_vs_incumbent.csv), beside
[`cutincl_liveness.csv`](cutincl_liveness.csv),
[`cutincl_env_flatness.csv`](cutincl_env_flatness.csv) and the analyzer's
[`cutincl_summary.json`](cutincl_summary.json):

**`visual_genome_m × dinov3_patch × max_patch` — region voting, 92 cells**

| arm | k=−6 | k=−3 | k=−1 | k=0 | k=+1 | k=+3 | k=+6 |
|---|---|---|---|---|---|---|---|
| `mid` | +0.096±0.006 | +0.045±0.005 | +0.008±0.002 | +0.000 | +0.006±0.002 | +0.053±0.008 | +0.112±0.012 |
| `rate` | +0.010±0.001 | +0.002±0.002 | +0.001±0.002 | +0.003±0.002 | −0.000±0.001 | −0.000±0.001 | −0.001±0.001 |
| `cross_tilt` | −0.002±0.001 | −0.013±0.002 | −0.010±0.002 | +0.003±0.003 | +0.012±0.004 | +0.029±0.007 | +0.030±0.006 |
| `q_tilt` (0.02) | +0.008±0.003 | +0.021±0.003 | +0.025±0.006 | +0.000 | +0.001±0.001 | +0.023±0.005 | +0.054±0.008 |

**`coco_val × dinov3_patch × max_patch` — region voting, 74 cells**

| arm | k=−6 | k=−3 | k=−1 | k=0 | k=+1 | k=+3 | k=+6 |
|---|---|---|---|---|---|---|---|
| `mid` | +0.069±0.006 | +0.037±0.005 | +0.008±0.004 | +0.000 | −0.007±0.002 | +0.004±0.007 | +0.036±0.008 |
| `rate` | +0.012±0.001 | +0.006±0.002 | +0.007±0.004 | **+0.015±0.002** | +0.005±0.001 | +0.002±0.000 | +0.000±0.000 |
| `cross_tilt` | +0.001±0.001 | −0.010±0.002 | −0.012±0.004 | +0.000±0.007 | +0.004±0.007 | +0.002±0.006 | +0.006±0.005 |
| `q_tilt` (0.02) | +0.007±0.002 | +0.037±0.004 | +0.051±0.012 | +0.000 | −0.004±0.001 | −0.005±0.005 | +0.010±0.004 |

**`visual_genome_m × siglip × whole_image` — binary voting, 90 cells**

| arm | k=−6 | k=−3 | k=−1 | k=0 | k=+1 | k=+3 | k=+6 |
|---|---|---|---|---|---|---|---|
| `mid` | +0.131±0.006 | +0.074±0.004 | +0.016±0.002 | +0.000 | +0.017±0.003 | +0.096±0.011 | +0.181±0.016 |
| `rate` | +0.002±0.001 | −0.002±0.001 | +0.001±0.002 | +0.003±0.001 | +0.001±0.001 | +0.000±0.000 | +0.001±0.001 |
| `cross_tilt` | −0.011±0.001 | −0.018±0.002 | −0.001±0.003 | +0.035±0.006 | +0.054±0.007 | +0.073±0.012 | +0.048±0.010 |
| `q_tilt` (0.02) | +0.025±0.004 | +0.034±0.002 | +0.013±0.003 | +0.000 | +0.010±0.002 | +0.061±0.009 | +0.113±0.013 |

**`coco_val × siglip × whole_image` — binary voting, 74 cells**

| arm | k=−6 | k=−3 | k=−1 | k=0 | k=+1 | k=+3 | k=+6 |
|---|---|---|---|---|---|---|---|
| `mid` | +0.080±0.007 | +0.039±0.006 | −0.008±0.005 | +0.000 | +0.003±0.003 | +0.037±0.009 | +0.084±0.011 |
| `rate` | +0.005±0.002 | −0.007±0.002 | **−0.021±0.005** | −0.003±0.002 | −0.001±0.002 | −0.001±0.001 | −0.000±0.000 |
| `cross_tilt` | −0.005±0.002 | −0.020±0.002 | **−0.034±0.005** | −0.004±0.005 | +0.011±0.007 | +0.029±0.010 | +0.024±0.006 |
| `q_tilt` (0.02) | +0.009±0.002 | +0.039±0.003 | +0.078±0.014 | +0.000 | −0.008±0.003 | +0.011±0.007 | +0.038±0.008 |

**The scale those differences sit on.** The incumbent's own regret against the
per-`k` oracle, mean ± SE over cells, is largest in the middle of the knob and
falls away at the ends (where one error type dominates the cost and the oracle
itself has little room):

| environment | k=−6 | k=−3 | k=−1 | k=0 | k=+1 | k=+3 | k=+6 |
|---|---|---|---|---|---|---|---|
| VG · region | 0.059±0.006 | 0.058±0.005 | 0.033±0.002 | 0.039±0.003 | 0.038±0.004 | 0.037±0.004 | 0.023±0.003 |
| COCO · region | 0.015±0.002 | 0.026±0.002 | 0.037±0.004 | 0.043±0.004 | 0.037±0.004 | 0.031±0.002 | 0.011±0.001 |
| VG · binary | 0.029±0.002 | 0.037±0.002 | 0.027±0.002 | 0.039±0.002 | 0.050±0.007 | 0.055±0.008 | 0.020±0.003 |
| COCO · binary | 0.019±0.003 | 0.038±0.004 | 0.061±0.006 | 0.050±0.004 | 0.039±0.003 | 0.033±0.003 | 0.012±0.002 |

So `mid`'s +0.112 at k=+6 on VG region is not a small proportional miss: it is
**five times** the regret the shipped rule carries at that stop. Conversely, a
±0.002 difference — where most of `rate`'s column sits — is about 5% of the
incumbent's own error and well inside what this sample can resolve.

`mid`'s k=0 column is exactly zero in every environment, and `q_tilt`'s is too:
both rules are *defined* to reproduce the measured midpoint there, so those cells
are a fidelity check rather than a result. An arm counts as **harmed** at an
(environment, k) only when its whole CI sits above **+0.01** — the tolerance
PR #2891 pre-registered for the acquisition-offset decision, kept identical so
two threshold decisions in the same subsystem are not held to different bars.
Requiring the loss to be *material* rather than merely detectable is what makes
the rule decidable at all: the sweep produces ~100 intervals, and a
"significantly worse anywhere" test rejects a *perfect* arm on multiplicity
alone. Harmed counts: `mid` 37, `cross_tilt` 12, `rate` **1**, `q_tilt` 13–36
depending on step.

![Paired regret against the shipped rule at every stop of the knob](figures/fig1_regret_vs_k.png)

*Paired regret against `mid_tilt` at each stop of the Inclusion knob, one panel
per environment, mean with bootstrap CI over cells. Dashed lines are the ±0.01
harm tolerance; the incumbent is the zero line. Read the shape, not just the
sign: `mid` (red) is pinned to zero at k=0 by construction and diverges in both
directions, which is what "inclusion-blind" costs. The five orange curves are
`q_tilt`'s five step sizes. This figure does **not** license comparing
magnitudes across panels as if they were the same units of user pain — the
environments differ in how separable their haystacks are.*

Three readings:

- **`mid` is pinned at zero where it was measured and wrong everywhere else.**
  Its regret is +0.000 at k=0 in every environment — it *is* the incumbent there
  — and rises to +0.11 (VG region), +0.036 (COCO region), +0.18 and +0.084
  (binary) by k=+6. That V is the whole of issue #2865, drawn.
- **`rate` tracks the incumbent.** |Δ| ≤ 0.005 at every k ≥ +1 in all four
  environments, and ≤ 0.002 in three of them; the widest gaps anywhere are
  +0.010 and +0.012 at k=−6 on the two region arms. This is not a coincidence
  and not really a measurement:
  `_quantile_at` composes `mid_tilt` as `q_mid + (q_rate(k) − q_rate(0))`, so
  `mid_tilt − rate` is the **constant** `q_mid − q_rate(0)` in fold-quantile
  space at every k. The frame agrees to sixteen digits: on
  `visual_genome_m/dinov3_patch/max_patch/bag/s0` the offset is −0.00886 at all
  thirteen stops. What the sweep therefore measures at thirteen stops is one
  offset, re-priced under thirteen cost weightings — and it survives all of
  them. `rate`'s single material loss is at **k=0 on `coco_val × dinov3_patch`
  (+0.015±0.002)**, which is the inclusion-0 `mid`-vs-`rate` question #2864
  already answered, reproduced on the linear-SVM head in an environment #2864
  never ran.
- **`cross_tilt` has a real, signed asymmetry.** It beats the incumbent below
  inclusion 0 in three of four environments (−0.010 to −0.034) and loses above it
  (up to +0.073±0.012 on VG binary). It is the only candidate that genuinely
  reads the acquisition-biased mixture weights, and those weights push the cut in
  the "admit more" direction — which is what the knob wants when it is asking for
  fewer false alarms and the opposite of what it wants above zero. Twelve harmed
  (environment, k) points; rejected as a rule, kept as a lead.

![Regret over the ramp at three stops of the knob](figures/fig3_regret_vs_votes.png)

*The same metric over the axis a user actually spends — votes — at k=−3, 0 and
+3, mean ± SE over categories and seeds. The k=0 row is a fidelity check you can
read directly: `mid` and `mid_tilt` are the same line there, because `mid_tilt`
is defined to reproduce the measured midpoint at inclusion 0. `q_tilt` is drawn
at its shipped placeholder step (0.02); its step axis is Figure 5.*

![One line per run at k=+3](figures/fig4_per_run.png)

*One line per run at k=+3, all environments pooled, with the mean in black. The
mean understates what the null costs, because the damage is concentrated rather
than spread: under `mid`, **6.3%** of cell-steps sit above 0.3 regret and 2.1%
above 0.5, against **0.8%** and 0.2% for both `mid_tilt` and `rate`. Per run it
is starker — **20 of 330 cells** have a median deep regret above 0.3 under `mid`,
against **2 of 330** under the incumbent. So the slider is not uniformly a bit
worse under the null; it is badly mis-placed on a particular sixteenth of runs,
which is exactly what a mean over a heavy tail hides.*

## 5. (b) How much of the knob survives as distinct admitted sets

A rule that moves the threshold without moving the admitted set has fixed
nothing. `distinct_admitted` is how many different answers dragging the slider
through its 13 stops produces at a given step, averaged over steps and cells.

**Distinct admitted sets out of 13 stops** (mean over cell-steps past 100 votes):

| arm | VG · region | COCO · region | VG · binary | COCO · binary |
|---|---|---|---|---|
| `mid` | **1.0** | **1.0** | **1.0** | **1.0** |
| `mid_tilt` (incumbent) | 12.8 | 12.4 | 12.8 | 11.4 |
| `rate` | 13.0 | 12.5 | 13.0 | 11.6 |
| `cross_tilt` | 12.9 | 12.3 | 12.9 | 11.2 |
| `q_tilt` (0.005) | 12.9 | 12.8 | 13.0 | 11.0 |
| `q_tilt` (0.01) | 12.6 | 12.4 | 12.8 | 11.6 |
| `q_tilt` (0.02) | 12.1 | 11.4 | 12.5 | 11.2 |
| `q_tilt` (0.04) | 11.3 | 10.0 | 11.4 | 10.1 |
| `q_tilt` (0.08) | 9.9 | 8.9 | 9.9 | 8.9 |

**Dead-step rate** — the share of *adjacent* slider stops that admitted exactly
the same set, which is what a user experiences as a control that does nothing:

| arm | VG · region | COCO · region | VG · binary | COCO · binary |
|---|---|---|---|---|
| `mid` | **1.00** | **1.00** | **1.00** | **1.00** |
| `mid_tilt` (incumbent) | 0.01 | 0.05 | 0.02 | 0.13 |
| `rate` | 0.00 | 0.04 | 0.00 | 0.12 |
| `cross_tilt` | 0.01 | 0.06 | 0.01 | 0.15 |
| `q_tilt` (0.02) | 0.08 | 0.13 | 0.05 | 0.15 |

`mid` is not "coarse" — it is **inert**. Every adjacent pair of stops admits the
same set, in every environment: all 65,671 measured cell-steps have a knob that
produces exactly one answer end to end, with no exceptions. The incumbent's margin over it is the
entire difference between a working control and a decoration. Every other
challenger's margin over the incumbent is at most **0.4 of a stop** — and the
only 0.4 belongs to `q_tilt` (0.005), which pays for it with +0.020 regret in
that same environment. The largest margin from an arm that is *not* materially
worse on regret is `rate`'s 0.2.

![Distinct admitted sets per arm](figures/fig6_knob_yield.png)

*How many of the 13 slider stops are their own answer, per arm and environment.
`mid` sits on the "inert" line in every environment — that is one bar per
environment at exactly 1.0, not a rounding artifact.*

![What the slider does to the admitted set and to the cut](figures/fig2_knob_liveness.png)

*Top row: the admitted fraction against k — what the user sees the slider do.
Bottom row: the combined fold quantile the cut sits at — what the rule did.
`mid` is a flat line in **both** rows, so it is blind rather than defeated by the
haystack. The diagnostic this pairing exists for is a rule that moves in the
bottom row and not the top; the only arms that do it here are `q_tilt` at its
larger steps, whose quantile runs past 1.0 and clips.*

**The literal rows behind those counts** are in
[`figures/examples_slider.md`](figures/examples_slider.md) — one real cell per
environment, showing the threshold each rule chose at each stop and how many of
the test pool it admitted, with that environment's least and most live cell named
beside it. On `coco_val × dinov3_patch`'s median cell (`clock`, seed 3, 300
votes, 2476 test items), the slider under `mid` admits **382 items at every one
of the 13 stops**; under `mid_tilt` it runs 38 → 2442. That table is also where
`q_tilt`'s failure mode is visible: at step 0.02 and above it reaches k=−10 with
a quantile past 1.0 and admits **zero** items.

## 6. Candidate 3's free parameter

`q_tilt` shifts the combined fold quantile by a fixed amount per step of the
knob, decoupled from the mixture — the simplest rule that *cannot* be
inclusion-blind. Its price is a step size with no principled value, so the sweep
fits it rather than assuming the shipped placeholder.

![q_tilt's step size](figures/fig5_qtilt_step.png)

*Left: what each step buys or spends against the incumbent, pooled over k.
Right: how much of the slider it delivers. Every curve on the left is above
zero, in every environment, at every step — there is no value of the free
parameter at which candidate 3 is not, on balance across the knob, worse than the
rule already shipped (it does win at scattered individual stops; see the per-k
tables) — and
the knob it delivers falls monotonically as the step grows. Small steps keep the
knob and lose on regret because they cannot move far enough at large |k|; large
steps lose the knob to saturation.*

That is a clean negative result: candidate 3 does not need a better step size, it
needs to not be a candidate. The placeholder `FOLD_ANCHOR_QTILT_STEP = 0.02` is
now measured rather than assumed, and what it measures is that the rule it
parameterises should not ship.

## 7. The haystack's own ceiling

The plan carried a separate item — *inclusion resolution on cleanly separated
haystacks* — asking how often a cut lands inside an empty band between two
well-separated modes, where it realizes to the same threshold however far it
moves. The best knob yield any rule achieves in an environment bounds what any
cut rule could deliver there:

| environment | best arm | best knob yield | dead-step rate | admitted span | quantile span |
|---|---|---|---|---|---|
| `coco_val × siglip × whole_image` | `rate` | 0.89 | 0.12 | 0.77 | 0.77 |
| `coco_val × dinov3_patch × max_patch` | `q_tilt` (0.005) | 0.99 | 0.01 | **0.10** | **0.10** |
| `visual_genome_m × siglip × whole_image` | `rate` | 1.00 | 0.00 | 0.95 | 0.95 |
| `visual_genome_m × dinov3_patch × max_patch` | `rate` | 1.00 | 0.00 | 0.84 | 0.84 |

One caveat on that table, since it would otherwise read as "COCO region voting
resolves the knob better than COCO binary": `best_knob_yield` maximises *distinct
sets* without regard to *how far the knob travelled*, so a rule that inches
across a tenth of the pool in thirteen fine steps outscores one that crosses
three quarters of it in thirteen coarse ones. That is what `q_tilt` (0.005) is
doing in row two — an admitted span of 0.10 against `rate`'s 0.85 in the same
environment. Read that environment's ceiling off `rate` (0.96) instead, and treat
`best_knob_yield` as a **lower** bound on what the haystack permits rather than a
like-for-like ranking.

The ceiling is real but small: the worst environment still resolves ~89% of the
slider under its best rule, and in the two Visual Genome environments the ceiling
is ~100%. Comparing `quantile_span` against `admitted_span` in the liveness table
says why — they are nearly equal for every non-saturating arm, meaning the cut's
motion in quantile space is being realized in the admitted set rather than
absorbed by a gap. **On these haystacks the rule, not the data, was the binding
constraint** — which is the reason #2868's repair worked at all.

## 8. What this changes

- **Nothing ships.** `FOLD_ANCHOR_CUT_RULE` stays `mid_tilt`.
- **The plan's `#2865` item closes**, along with the two items it absorbed:
  *deeper-than-inclusion-0 evidence for the cut rule* (this sweep) and
  *inclusion resolution on cleanly separated haystacks* (§7).
- **`test_inclusion_slide_recut`'s assertion is now backed by a number.** It
  asserts that a slide moves the admitted *set*, not just the threshold — the
  right invariant, and one `mid` would have failed. What it could not say is by
  how much: the measured incumbent yield is 0.88–0.99 per environment, so the
  shipped rule clears that floor by an enormous margin rather than scraping it.
- **`FOLD_ANCHOR_QTILT_STEP`'s docstring stops being a placeholder** and starts
  being a measurement: the parameter was swept, and no value of it makes the rule
  competitive.
- **`preflight.sh` gained the check this run needed and did not have.** The
  launcher was drafted on 2026-08-12 pinning `CALIB_HEAD=linear` — production
  then, a legacy arm by the morning it ran — and the same pin would have
  produced a clean, plausible table about a detector nobody has. Check 12 now
  compares every knob with a named shipped constant against it and fails on any
  divergence the study has not *declared* (`--diverges head,anchor_weight`). To
  give the patch geometry something to be compared with, `voting_iterations`
  names its default as `PRODUCTION_PATCH_STYLE` rather than inlining it. Written
  up in
  [`lessons/2026-08-21-a-launcher-pinned-a-head-that-stopped.md`](../../../scripts/experiments/lessons/2026-08-21-a-launcher-pinned-a-head-that-stopped.md);
  the units bug the sizing run caught is in
  [`lessons/2026-08-21-a-pooled-number-over-an-axis-that-rescales.md`](../../../scripts/experiments/lessons/2026-08-21-a-pooled-number-over-an-axis-that-rescales.md).

### Follow-ups this run raises

Both are now tracked in **#3557** (a plan bullet is not a tracked follow-up).
They were previously parked as open items in
[`docs/plans/population-anchored-calibration.md`](../../plans/population-anchored-calibration.md),
in the space the closed `#2865` item vacated:

- **A sign-dependent tilt.** `cross_tilt` beats the incumbent at k ∈ [−4, −1] in
  three of four environments, by up to 0.034 — larger than anything else on the
  table — while losing above zero. A rule that reads the acquisition-biased
  mixture weights *only when the knob asks for fewer false alarms* is not
  obviously wrong, but it is a new rule: it needs its own pre-registration, and a
  hinge at k=0 has to be shown not to break the nesting contract. (`rate` shows
  the same sign, smaller: −0.021±0.005 at k=−1 on binary COCO.)
- **Explain the k=0 loss on `coco_val × dinov3_patch`.** `rate` is worse than
  `mid` there by 0.015 — five times its inclusion-0 gap in the other three
  environments, and the single reason `rate` did not ship. #2864's mechanism for
  `mid`-beats-`rate` is the components' **variance asymmetry**; if that is right,
  this environment should show the widest asymmetry, and that is checkable from
  the `__cutdiag` frame this run already wrote.

## 9. Reproducing

```bash
# on the GRID, from this branch's worktree
bash scripts/experiments/calibration/launch_incl_2865.sh prepare   # reads the pile in place
bash scripts/experiments/calibration/launch_incl_2865.sh size      # time one cell per arm first
bash scripts/experiments/calibration/launch_incl_2865.sh arms      # 336 cells + the analysis step

# analysis and figures (both read the same CSVs)
python scripts/experiments/calibration/selftest_analyze_cutincl.py  # planted answers, no cluster data
python scripts/experiments/calibration/analyze_cutincl.py
python scripts/experiments/calibration/make_cutincl_figs.py --out docs/experiments/2026-08-21-inclusion-cut-rule/figures
```

The run lived at `/expscratch/sgreenberg/cut-incl-2865`; its cells were sized
from a real cell rather than a guess (75 and 83 minutes for the two dinov3 region
arms at MaxRSS 5.3 G, 8–9 minutes for the two binary arms), and the array ran
120-wide on the cpu partition — the `cpu_limit` QOS charges 2 CPUs per task, so
120 is the real cap.
