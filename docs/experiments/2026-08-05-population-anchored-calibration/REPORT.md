# Population-anchored calibration: fusing the haystack into the trained threshold

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


**Issues #2852 / #2853 · design pre-registered in
`docs/plans/population-anchored-calibration.md` · estimators PR #2857 ·
shipped by PR #2861 · reports PR #2860 (run A) and PR #2864 (run B)**

Two runs:

| | Run A — the deep-regime run | Run B — the anchor-mass sweep |
|---|---|---|
| Reported in | PR #2860 | PR #2864 (this revision) |
| Date | 2026-08-05 | 2026-08-06 |
| Base | dev `0a54f0d7` | dev `7fbde84e` |
| Cells | 184/184, 0 failures (SLURM 468311 → 468312) | 384/384, 0 failures (SLURM 468874) |
| Scope | Visual Genome × {dinov3 max-patch, siglip whole-image} | **6 environments**: 3 datasets × 4 embedders × both voting modes |
| Anchor mass κ | 1 · 3 · 10 · 30 · 100 | **0.01 · 0.03 · 0.1 · 0.3 · 0.5 · 1 · 2 · 3** |
| Blend control | the 6→20 ramp (production at that base) | **`slow_cap50` / `cap50`** — what actually ships today |

> **Run A's recommendation shipped before run B finished, and then some.** PR
> #2861 merged `fold_anchored κ=1 rate` as the production threshold at 23:30
> EDT on 2026-08-05 — an hour into run B's cell array — and PR #2863 followed,
> deleting the `safe_thresholds` setting so the fused threshold is now
> **unconditional**. Everything below that reads like "if adopting" in run A's
> voice is therefore a statement about **code on dev**, and two of run B's
> findings are regressions against it rather than choices about it: the shipped
> κ and cut rule are beaten in 6/6 environments by `κ=0.3, mid`, and the
> shipped path covers binary-voting detectors, where it is *worse* than the
> `cap50` blend it replaced — with, after #2863, no way back to that blend.
>
> To be fair to #2863: on binary voting the full ranking is `cap50` (best) >
> fusion at `κ=0.3 mid` (−0.0004, a tie) > fusion at the shipped `κ=1 rate`
> (+0.0063) > pure cross-calibration (+0.0103). So removing the off-switch is
> an *improvement* for anyone who had safe thresholds off; the regression is
> specifically fusion-versus-blend, and it is what the mode split in
> *Recommendation* 2 exists to fix.

Visual report with mechanism figures:
<https://claude.ai/code/artifact/6bd39c84-5946-4dd8-ba80-334a88428920>

## BLUF

**Run A's winner sat on the bottom edge of its κ grid, and it shipped (PR
#2861) before run B finished. Run B extended the grid two decades down across
six environments: the optimum is interior, it is not where run A left it, and
the shipped setting is beaten in every environment measured — including two
where the shipped path is worse than the blend it replaced.**

1. **The setting moves from `κ=1, rate` to `κ=0.3, mid`.** Pooled over six
   environments in the deep regime, `fold_anchored κ=0.3 mid` cuts paired
   regret vs pure x-cal by **−0.0437** against the run-A winner's **−0.0392**.
   Head to head, cell-paired, the new setting wins in **6 of 6 environments**
   (pooled −0.0045, p<1e-4). It is also the best *single global* setting
   available: forcing it on every environment leaves each one within **0.0067**
   of its own best (mean gap 0.0026), against 0.0102 / 0.0069 for `κ=1, rate`.
2. **The κ answer does not transfer as a constant — but it barely has to.**
   Per-environment optima span κ 0.03–0.5 among the four environments where
   fusion does anything at all, and the tied plateaus overlap at κ=0.3 in three
   of the four. Within a decade the curve is nearly flat; outside it, it is not.
3. **The bigger finding is that the *win* does not transfer — and PR #2861
   shipped it everywhere.** Measured against the blend that ships today rather
   than the ramp run A had to use, fusion is a clear win on **region voting**
   (−0.026 to −0.032 vs `slow_cap50`), a **statistical dead heat** on COCO
   binary voting (−0.0004, n.s.), and a small **loss** on the 838-image
   caltech101 set (+0.003 to +0.009 — `cap50` beats every fusion arm there).
   The shipped `κ=1 rate` is worse than `cap50` on *both* binary environments
   (+0.0063 pooled), so binary-voting detectors currently have a slightly worse
   threshold than they had before #2861. Run A's "adopt it as *the* production
   threshold path" was measured on the two environments where it works.
4. **κ\* falls monotonically as votes accumulate** — argmin 3 → 3 → 1 → 0.3 →
   0.1 across the 20/50/100/200/300-vote windows. A constant κ is therefore a
   compromise across regimes, and the property that motivated κ (the labels'
   authority *growing* with data) turns out to grow too fast.

Hypothesis verdicts at the new setting: **H1 ✓ · H2 ✓ (region) / ✗ (binary) ·
H3 ✓ vs x-cal, ✗ vs the incumbent · H4 ✓ on binary, ✓ from 51 votes on region**
— see *Stability (H3) and the FNR budget (H4)*.

## Take-aways

- **The cut rule flips with the anchor mass.** `mid` peaks low (κ=0.3) and
  `rate` peaks high (κ=1) — and the two curves cross near κ=1–2, which is why
  run A, whose grid started at 1, saw `rate` in front.

  > **Correction (2026-08-12, from #2865's follow-up).** This take-away
  > originally explained the flip as "`mid` ignores the mixture weights; `rate`
  > needs them", with heavier anchoring letting the votes' acquisition-biased
  > prevalence into the weights that `rate` reads. **That mechanism is wrong.**
  > `rate` passes `lam = (fnr/fpr)·(w_lo/w_hi)` into a solve of the form
  > `w_lo·N_lo = lam·w_hi·N_hi`, where the prior-odds factor cancels the weights
  > exactly: the rule reduces to `N_lo = (fnr/fpr)·N_hi`, is prior-free, and its
  > interior root is invariant to the mixture weights at every inclusion (it
  > reads them only through the out-of-interval continuation slope). Verified
  > numerically over `w_lo ∈ {0.5, 0.9, 0.99}` and pinned by
  > `tests_lib/detectors/test_cut_inclusion_sweep.py::TestRateIsPriorFree`.
  >
  > What actually separates the two rules at inclusion 0 is the **variance
  > asymmetry**: `rate` solves the equal-density crossing, which sits off the
  > midpoint by `≈ var·ln(w_lo/(lam·w_hi))/(mu_hi − mu_lo)` whenever the
  > components differ in width, and heavier anchoring is what pulls the anchored
  > components' *widths* apart. The κ recommendation and every number in this
  > report are measurements and stand unchanged; only this explanation of them
  > does not. It is corrected rather than deleted because #2865's candidate
  > list was derived from it — its "candidate 2", *drop the mixture-weight
  > factor from `rate`*, is a no-op that describes what `rate` already computes.
- **"Honest anchors beat anchor mass" was a statement about heavy anchoring.**
  Run A found label-anchored fits flipping worse than x-cal at κ≥10. At κ=0.3
  they do not: pooled deep, `anchored κ=0.3 rate` is −0.0424, and on region
  voting it is the single best arm measured (−0.0293 vs `slow_cap50`, versus
  −0.0280 for its fold-anchored counterpart). The fold repair matters exactly
  in proportion to how much authority the anchors have — at κ=0.3 they have
  little, so there is little to repair — head to head on region voting the
  label arm is −0.0028 ahead of `fold κ=0.3 mid` (p=0.017) while skipping K
  haystack scoring passes. The fold path is still the one to ship: it is the
  only arm that does not go backwards on binary voting, and its anchors are
  honest by construction, so it degrades gently if κ is ever mis-set.
- **Fusion's value tracks how many *positive* anchors the regime supplies.**
  Deep-regime median positives per environment: VG dinov3 24, COCO 8, VG siglip
  7, caltech101 3. Deep Δregret vs x-cal at each environment's best: −0.093,
  −0.019/−0.012, −0.068, −0.002. The estimator needs a Good component it can
  actually locate.
- **Against today's schedule, not yesterday's.** `cap50` / `slow_cap50` (PR
  #2849, merged after run A's base) is a much stronger control than the 6→20
  ramp: on region voting pure x-cal is +0.052 *worse* than `slow_cap50`, and
  `rank_transfer` +0.037 worse. Run A's −0.085-vs-x-cal headline shrinks to
  −0.026 once the comparison is against what users have.
- **The new setting is less steady than the old one.** Mean |Δthreshold| per
  step past 20 votes: `κ=1 rate` 0.0056 · `cap50` 0.0068 · **`κ=0.3 mid`
  0.0098** · x-cal 0.0131. The accuracy gain costs stability; `κ=0.3 mid` is
  still 1.3× steadier than x-cal but is *less* steady than the shipped blend.
- **H4 improves.** On the run-A scope (VG), the recommended arm's FNR is
  0.264 / 0.255 / 0.235 / 0.208 / 0.183 across the five windows — over the
  0.25 nominal budget in the two shallowest, under it from 51 votes, against
  0.284 / 0.271 / 0.253 for run A's winner and 0.465 / 0.353 / 0.281 for x-cal.
  On binary voting it never exceeds 0.064.
- **The degeneracy machinery is all but inert, and it bottoms out at the
  recommended κ.** Across 1.74M fold-arm rows, 99.90% ran fully anchored (2/2
  folds), 0.094% at 1/2, 0.006% at 0/2; label-anchored fits fell back on 0.026%
  (all `inverted_means`). The fold fallback rate is **U-shaped in κ** — 0.118%
  at κ=0.01, minimum **0.059% at κ=0.5**, 0.180% at κ=3 — so the accuracy
  optimum and the numerical-stability optimum coincide.
- **`qmedian` is now answered, and it loses.** A K=4 addendum (92 cells,
  0 failures) un-degenerates the combine comparison that two folds made
  impossible: `qmean` beats `qmedian` at every κ, indistinguishably at the
  recommended κ=0.3 (p=0.90) and significantly from κ=1 up. Keep `qmean`.
  Four folds also beat two at all 16 grid points by a uniform −0.008, but no
  single comparison is significant and the contrast cannot be paired.

## Why this experiment exists

*(unchanged from run A)*

The threshold at run A's base treated its two estimators as rivals on a
hand-tuned schedule: an unsupervised 2-component GMM on the haystack's score
distribution ramps from weight 1 at 6 votes to 0 at 20; pure cross-calibration
ships thereafter. Three prior results said the framing is wrong (owner-side
run: naive GMM still competitive at ~300 votes; the selection-bias study
cleared the labels; #2790/#2799 isolated three structural deficits of the
conformal cut, none decaying with label count):

1. **Sample size** — the conformal cut is a low quantile over *tens* of
   held-out positives; the GMM fits up to 50k scores.
2. **Scale transfer** — the x-cal cut is measured on *fold models'* score
   scales but applied to the *final model's* scores.
3. **Per-retrain variance** — fold splits redraw every vote; the x-cal cut is
   a fresh noisy estimate each step.

The reframe under test: labels and haystack hold complementary information —
labels know which side is which and which quantile matters; the haystack knows
where that lives on the final model's actual score scale. They should feed
**one estimator**, not two rivals averaged on a schedule.

## The algorithm

*(unchanged from run A; see the visual report for the two mechanism figures)*

### Label-anchored mixture (semi-supervised EM; κ is the only knob)

`fit_anchored_score_gmm` fits the same 2-component 1-D Gaussian mixture as
production's `fit_score_gmm`, but semi-supervised: haystack scores are free;
every voted item's component membership is **clamped** to its label (Good →
high component, Bad → low). Classical anchored EM, initialized from the
unanchored seed-42 fit:

- **E-step** over free points only (log-domain responsibilities); anchors stay
  one-hot regardless of where their scores lie.
- **M-step** re-estimates means/variances/weights with each anchor counted
  **κ** times (`anchor_weight`). With n votes vs N haystack scores, the
  labels' share of the class-conditional evidence is **γ = κn / (κn + N)**.

What those numbers actually are in this run: at the deep-regime median of ~176
votes against N ≈ 2096 sim-set scores, κ=0.3 puts **γ ≈ 2.5%** of the fit in
the labels' hands, and κ=1 about 7.7%. The knob is not "how much do we trust
labels" in any colloquial sense — every setting that works leaves the fit
overwhelmingly population-driven, and the labels' job is to *identify* the
components, not to estimate them.

Anchors *force* the component identification rather than inherit it: if labels
contradict the population modes, the fit reports a named degeneracy
(`inverted_means`, `component_collapse`, …) and falls back to the
**unanchored** fit of the same sample — never to 0.5. The cut rule then applies
to the fitted pair: `mid` (production midpoint) or `rate` (the #2836
rate-optimal crossing `wn·f_pos = wf·f_neg`, midpoint fallback when rootless).

### The flaw in anchoring on the final model — and the fold repair

The label-anchored fit anchors on the **final model's** scores of the voted
items — but those items were in the final model's training set, so their scores
are optimistically separated, and votes are acquisition-biased (Autopilot
samples near the threshold).

The **fold-anchored mixture** ("cross-LabeledGMM") repairs this with machinery
cross-calibration already has. Per calibration fold k: score the haystack with
fold model k and fit the anchored mixture on those scores with fold k's
**held-out** labeled scores as anchors (honest anchors, one scale); apply the
cut rule; read the cut's empirical quantile q_k in fold k's own haystack
distribution; combine fold quantiles and realize the combined quantile on the
**final model's** haystack distribution (**rank-transfer**). No raw score ever
crosses scales. `rank_transfer` also runs as its own arm — fixing *only*
deficit 2, so its gain measures how much of the problem was ever about scale.

## Experiment design — run B

Within-step **paired variants**: every arm re-cuts the same per-step model
against the same held-out test scores, so contrasts are paired per
(environment, category, seed, step). 300-vote trajectories, 4 seeds, production
linear head, safe thresholds ON, 384 cells, 5.97M metric rows.

| Dimension | Values |
|---|---|
| Environments | `visual_genome_m × dinov3_patch × max_patch` (region) · `visual_genome_m × siglip × whole_image` (region) · `coco_val × siglip` (binary) · `coco_val × siglip2` (binary) · `caltech101_m × siglip` (binary) · `caltech101_m × siglip_l` (binary) |
| Anchor mass κ | 0.01 · 0.03 · 0.1 · 0.3 · 0.5 · 1 · 2 · 3 |
| Cut rules | `mid` · `rate` |
| Fold combine | `qmean` only (qmedian is byte-identical at 2 folds; the K=4 addendum below tests both) |
| Estimators | `anchored_w{κ}_{rule}` · `fold_anchored_w{κ}_{rule}_qmean` · `rank_transfer` |
| Controls | `xcal_only` · `pooled_mid` · counterfactual schedule rows `prod`, `slow_cap50`, `cap50`, `slow`, `pure_gmm`, `pure_xcal` |
| Trajectory | pinned to `prod` (the 6→20 ramp), i.e. run A's trajectory generator |
| Checkpoints | windows at 20 / 50 / 100 / 200 / 300 votes; deep regime = the ≥100 windows, i.e. votes 51–300 (run A's definition) |

The six environments span the three things that could plausibly move the
answer, and one that turned out to matter more than any of them:

| Environment | Voting | Fit population N | Deep median positives |
|---|---|---:|---:|
| VG × dinov3 · max_patch | region | 2096 | **24** |
| VG × siglip · whole_image | region | 2096 | 7 |
| COCO × siglip | binary | 2476 | 8 |
| COCO × siglip2 | binary | 2476 | 8 |
| caltech101 × siglip | binary | 419 | **3** |
| caltech101 × siglip_l | binary | 419 | 3.5 |

**Statistics.** Every p-value below is a paired Wilcoxon over **cell means**
— one number per (environment, category, seed, window) — not over raw steps.
300 steps of one trajectory are one experiment, not 300; the step counts are
reported only to show coverage. `analyze_rate.py` carries a planted-answer
self-test (`selftest_analyze_rate.py`) that fabricates a curve with a known
flat bottom and checks the argmin, the tied plateau, family separation, and the
κ-vs-γ table all come back right.

## Results

### Run A recap — what the deep-regime run established

Visual Genome only, deep regime, paired Δregret vs pure x-cal, κ ∈ {1…100}:

| κ | fold · rate | fold · mid | label · rate | label · mid |
|---:|---:|---:|---:|---:|
| 1 | **−0.0847** | −0.0796 | −0.0725 | −0.0626 |
| 3 | −0.0699 | −0.0660 | −0.0308 | −0.0246 |
| 10 | −0.0525 | −0.0564 | +0.0259 | +0.0166 |
| 30 | −0.0435 | −0.0493 | +0.0734 | +0.0482 |
| 100 | −0.0335 | −0.0360 | +0.1054 | +0.0697 |

`rank_transfer` = −0.0196. That table is what motivated run B: every column is
monotone toward the κ=1 boundary, so the grid could not say whether the
estimator wanted κ=1 or something smaller. It also establishes the part run B
did **not** re-measure — that heavy anchoring on final-model scores is actively
harmful, and that the deficit is mostly the conformal quantile's sample size
(anchoring recovers −0.073) rather than scale transfer (rank-transfer alone
recovers −0.020).

Run B's numbers are smaller throughout because it pools six environments,
four of which are easier for x-cal than Visual Genome; the VG-only columns of
run B (−0.093 to −0.098 at their optima) are the like-for-like comparison and
they are slightly *better* than run A's −0.085.

### The κ curve, pooled over all six environments (deep, paired vs `xcal_only`)

| κ | fold · mid | fold · rate | label · mid | label · rate |
|---:|---:|---:|---:|---:|
| 0.01 | −0.0407 | −0.0270 | −0.0409 | −0.0345 |
| 0.03 | −0.0413 | −0.0284 | **−0.0414** | −0.0364 |
| 0.1 | −0.0428 | −0.0318 | −0.0412 | −0.0399 |
| **0.3** | **−0.0437** | −0.0362 | −0.0377 | **−0.0424** |
| 0.5 | −0.0428 | −0.0381 | −0.0329 | −0.0416 |
| 1 | −0.0391 | **−0.0392** | −0.0214 | −0.0363 |
| 2 | −0.0326 | −0.0382 | −0.0056 | −0.0259 |
| 3 | −0.0285 | −0.0366 | +0.0053 | −0.0167 |

1121 cells / 93,060 paired steps per fold cell; every entry p<1e-3 vs zero
except `label·mid` at κ=2. Tied plateaus (cell-paired against each family's own
argmin, p≥0.05): fold·mid **{0.3, 0.5}**, fold·rate {1, 3}, label·mid
{0.03, 0.1}, label·rate {0.3, 1}.

**All four curves have an interior optimum**, so the grid-edge problem is
resolved. `fold · mid` at κ=0.3 is the best arm in the table.

### Challenger vs run A's winner, head to head

`fold_anchored κ=0.3 mid` − `fold_anchored κ=1 rate`, deep, cell-paired:

| Environment | κ=0.3 mid | κ=1 rate | diff | n | p |
|---|---:|---:|---:|---:|---:|
| VG × dinov3 · max_patch | −0.0913 | −0.0893 | −0.0020 | 270 | 0.012 |
| VG × siglip | −0.0669 | −0.0645 | −0.0023 | 268 | 0.48 |
| COCO × siglip | −0.0121 | −0.0020 | **−0.0101** | 220 | <1e-4 |
| COCO × siglip2 | −0.0183 | −0.0124 | −0.0059 | 219 | <1e-4 |
| caltech101 × siglip | +0.0010 | +0.0034 | −0.0024 | 72 | <1e-4 |
| caltech101 × siglip_l | +0.0029 | +0.0059 | −0.0030 | 72 | <1e-4 |
| **pooled** | **−0.0437** | −0.0392 | **−0.0045** | 1121 | <1e-4 |

The new setting is better in every environment, and its margin is *largest*
exactly where run A had no coverage.

### Which single global setting to ship

Each environment has its own argmin; the deployable question is what one
setting costs. Gap = distance from that environment's own best fold-anchored
arm, worst case over the six:

| rule | κ | mean Δregret | worst-env gap | mean gap |
|---|---:|---:|---:|---:|
| **mid** | **0.3** | **−0.0308** | **0.0067** | **0.0026** |
| mid | 0.5 | −0.0305 | 0.0103 | 0.0029 |
| mid | 0.1 | −0.0296 | 0.0068 | 0.0038 |
| mid | 1 | −0.0279 | 0.0192 | 0.0054 |
| rate | 1 *(run A)* | −0.0265 | 0.0102 | 0.0069 |

### Per-environment optima — and where fusion is worth anything

Best fold-anchored arm per environment (deep, vs x-cal), with the tied plateau:

| Environment | best (mid) | κ\* | plateau | best (rate) | κ\* |
|---|---:|---:|---|---:|---:|
| VG × dinov3 · max_patch | −0.0930 | 0.1 | 0.1, 0.3 | −0.0980 | 0.3 |
| VG × siglip | −0.0676 | 0.5 | 0.5, 1 | −0.0653 | 0.5 |
| COCO × siglip2 | −0.0193 | 0.03 | 0.03, 0.3, 0.5 | −0.0194 | 3 |
| COCO × siglip | −0.0121 | 0.3 | 0.3, 0.5 | −0.0112 | 3 |
| caltech101 × siglip | −0.0024 | 3 | 3 | −0.0030 | 3 |
| caltech101 × siglip_l | −0.0001 | 3 | 3 | +0.0022 | 3 |

Two readings, and the difference matters:

- **κ\* is not a constant.** Among the four environments with a real effect it
  spans 0.03–0.5 (17×). κ=0.3 is inside the tied plateau in three of the four
  (all but VG × siglip, where using 0.3 instead of 0.5 costs ≤0.009).
- **The two environments whose argmin sits at κ=3 have nothing to optimise.**
  caltech101's whole curve spans 0.007–0.027 around zero; picking its argmin is
  picking noise. Reporting them as "κ\*=3" would be the same grid-edge error
  run A made, one level up.

**Is the invariant κ or the label share γ?** Neither cleanly. At each
environment's own median deep vote count, the γ implied by its κ\* is 0.8%
(VG dinov3), 4.1% (VG siglip), 2.1% / 0.2% (COCO) — a 20× spread, no tighter
than κ itself. The one regularity worth flagging (**suggestive, not
established — four points, and it needs a plateau-member choice for one of
them**): κ\* × deep positives lands at 2.4, 2.4, 2.4, 3.5 for the four working
environments, i.e. "the positive votes should collectively be worth about three
haystack points." A *fixed total* anchor mass, not a fixed per-anchor mass.

### κ\* falls as votes accumulate

Argmin κ by window, pooled (fold-anchored):

| window (votes) | 2–20 | 21–50 | 51–100 | 101–200 | 201–300 |
|---|---:|---:|---:|---:|---:|
| `mid` argmin κ | 3 | 3 | 1 | 0.3 | 0.1 |
| `rate` argmin κ | 3 | 3 | 2 | 1 | 1 |
| `mid` best Δregret | −0.0711 | −0.0461 | −0.0434 | −0.0442 | −0.0441 |
| `mid` Δregret at κ=1 | −0.0630 | −0.0428 | −0.0434 | −0.0391 | −0.0348 |

This is the cleanest structural result in run B and it is a criticism of the
parameterisation itself. γ = κn/(κn+N) was chosen so the labels' authority
would *grow with data instead of a hand-tuned ramp*. The data say it grows too
fast: the best κ falls roughly like 1/n over the deep half, which is exactly
what holding **κ·n** constant would do. A fixed κ is a compromise between a
shallow regime that wants κ≈3 and a deep regime that wants κ≈0.1. The bottom
row prices that compromise for κ=1: it gives up 0.008 at 2–20 votes and 0.009
at 201–300 against the window-optimal setting, and is only exactly right in the
51–100 window. (κ=0.3, the recommended constant, buys back the deep end and
pays more at the shallow end; either way a constant cannot have both.)

### Against the blend that actually ships

Run A could only compare fusion to the 6→20 ramp, because PR #2849's per-mode
schedules merged after its base. Run B scores `slow_cap50` (region) and
`cap50` (binary) as counterfactual rows on the same trajectory. Deep,
cell-paired, pooled within voting mode (negative = beats the shipped blend):

| Arm | region voting | binary voting |
|---|---:|---:|
| `anchored κ=0.3 rate` (label) | **−0.0293** | +0.0042 |
| `fold_anchored κ=0.3 rate` | −0.0280 | +0.0161 |
| **`fold_anchored κ=0.3 mid`** | **−0.0256** | **−0.0004** |
| `fold_anchored κ=1 rate` *(run A)* | −0.0234 | +0.0063 |
| `pure_gmm` schedule | −0.0225 | +0.0052 |
| `rank_transfer` | +0.0367 | +0.0222 |
| `xcal_only` | +0.0520 | +0.0103 |

**This is the finding that most changes the recommendation.** On region voting
fusion beats the shipped blend by a wide, highly significant margin. On binary
voting the *only* arm that is not worse than `cap50` is `fold_anchored κ=0.3
mid`, and it is a dead heat (−0.0004, n.s.); run A's winner is +0.0063 *worse*
there, and on caltech101 alone every fusion arm loses to `cap50` by
+0.003…+0.025. The full binary ranking is worth stating plainly, because it is
what makes #2863 a mixed change rather than a simple mistake: `cap50` (best) >
fusion `κ=0.3 mid` > fusion `κ=1 rate` (shipped) > pure x-cal (worst). Deleting
the off-switch moved anyone who had it off from the worst option to the third;
it also removed the route back to the best one. `cap50` is simply a good estimator in the low-positive regime —
consistent with #2841's finding that caps buy *spread*, which is what a
threshold estimated from three positives is short of.

**On region voting alone, the cheap arm wins.** Head to head on the two region
environments (deep, cell-paired, n=538): `anchored κ=0.3 rate` — the
*label*-anchored fit, which needs no per-fold haystack scoring pass at all —
beats `fold_anchored κ=0.3 mid` by −0.0028 (p=0.017) and `fold_anchored κ=0.3
rate` by −0.0005 (p=0.016); the two fold arms are indistinguishable from each
other (−0.0024, p=0.23). So at this anchor mass the fold repair buys nothing
measurable on region voting — which is the mechanism talking again: the repair
corrects train-set optimism in the anchors, and at κ=0.3 the anchors move the
fit by ~2.5%. The margin is 0.003 on a −0.026 effect, and the fold path is the
conservative choice (honest anchors by construction, and a far gentler
degradation if κ is ever mis-set), but a deployment that cannot afford K extra
scoring passes has a measured, cheaper option.

Caveat carried from #2841: a counterfactual schedule row re-cuts *this*
trajectory, so it bounds the threshold-rule difference, not the whole-system
difference — a schedule that would have labelled different items cannot show
that here. Run A's acquisition-feedback finding suggests the true region-voting
gap is if anything larger.

### Stability (H3) and the FNR budget (H4)

Mean |Δthreshold| per step past 20 votes, pooled: `fold κ=1 rate` **0.0056** ·
`cap50`/`slow_cap50` 0.0068 · `fold κ=0.3 mid` **0.0098** · `pooled_mid` and
`xcal_only` 0.0131. The recommended arm is 1.3× steadier than x-cal but
**less** steady than both the shipped blend and run A's winner — the accuracy
gain is bought partly with jitter. H3 holds against x-cal, not against the
incumbent.

FNR by window against the 0.25 nominal conformal budget:

| | 2–20 | 21–50 | 51–100 | 101–200 | 201–300 |
|---|---:|---:|---:|---:|---:|
| **region**, `κ=0.3 mid` | 0.264 | 0.255 | **0.235** | 0.208 | 0.183 |
| region, `κ=1 rate` (run A) | 0.284 | 0.271 | 0.253 | 0.235 | 0.213 |
| region, `slow_cap50` | 0.320 | 0.311 | 0.247 | 0.190 | 0.156 |
| region, `xcal_only` | 0.465 | 0.353 | 0.281 | 0.218 | 0.182 |
| **binary**, `κ=0.3 mid` | 0.064 | 0.059 | 0.057 | 0.058 | 0.064 |
| binary, `xcal_only` | 0.190 | 0.110 | 0.071 | 0.051 | 0.043 |

The recommended arm is under the nominal budget **from 51 votes** on region
voting (run A's winner only manages it from 101) and never approaches it on
binary voting. The two shallow windows still exceed 0.25 — but every measured
arm does, x-cal by 1.8×, and this is the closest any of them gets. The
deep-regime recall trade is real and small: at 201–300 votes on region voting
the recommended arm gives up 0.027 FNR against `slow_cap50` and takes 0.058
FPR back, for −0.031 net cost.

### Fold count and the combine rule (addendum, SLURM 470106)

`calibrate_count` was a hard-coded constant, which is precisely why the
qmean/qmedian question had gone unanswered through two runs: at two folds the
two arms are byte-identical. Making it an env knob (default unchanged) and
re-running VG × siglip at **K=4**, 92 cells, 0 failures, answers both halves of
the last pre-registered open item.

**The combine rule barely matters, and where it does, production's `qmean`
is right.** Paired within the K=4 run (both arms re-cut the same per-step fold
fits, so this *is* a paired contrast), `qmean` beats `qmedian` at every κ:

| κ | 0.01 | 0.1 | 0.3 | 0.5 | 1 | 2 | 3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| qmedian − qmean (`mid`) | +0.0005 | +0.0005 | +0.0003 | +0.0002 | +0.0005 | +0.0009 | +0.0011 |
| p | 0.48 | 0.13 | 0.90 | 0.84 | 0.033 | 0.0005 | 2e-5 |

At the recommended κ=0.3 the two are indistinguishable (p=0.90); the gap only
becomes significant at κ≥1 and grows with κ. That is the same mechanism as
everywhere else in this report — the combine only has something to disagree
about once the anchors have enough authority to move the per-fold cuts apart.
**Keep `qmean`.**

**More folds helps, but not measurably.** K=4 is nominally better than K=2 at
**every one of the 16 (κ, rule) grid points**, by a very stable −0.008:

| κ | 0.01 | 0.1 | 0.3 | 0.5 | 1 | 3 |
|---|---:|---:|---:|---:|---:|---:|
| K=4 (`mid`) | −0.0662 | −0.0709 | −0.0745 | −0.0753 | −0.0751 | −0.0725 |
| K=2 (`mid`) | −0.0583 | −0.0630 | −0.0669 | −0.0676 | −0.0666 | −0.0634 |
| difference | −0.0079 | −0.0079 | −0.0076 | −0.0077 | −0.0085 | −0.0092 |

**But no single comparison is significant** (Mann-Whitney p = 0.14–0.34), and
it cannot be made paired: changing K changes the splits, the per-fold models
and therefore the trajectory, so the two runs are different experiments sharing
a design. The uniform sign is suggestive rather than conclusive — the 16 points
are cut from the same 268 cells and are strongly correlated, so a sign test
over them would badly overstate the evidence. The honest summary is: **no
detectable fold-count effect; if it is real it is worth about as much as
getting κ right, and it costs one extra haystack scoring pass per step.**

Two supporting observations from the same run. The argmin is **κ=0.5 under both
fold counts and both rules**, with κ=0.3 and κ=1 within 0.001 of it — the
flat-bottom finding survives the fold count, and this environment's K=2 argmin
reproduces the main run's per-environment table exactly. And `rank_transfer`
improves from −0.030 to −0.035 with more folds, which is what a better-pooled
fold quantile should do.

### Estimator provenance

Over 1,738,576 fold-arm rows: **99.900%** fully anchored (`fold_anchored[2/2]`),
0.094% at 1/2 folds, 0.006% at 0/2 — matching run A's 99.91%. Over 1,782,864
label-arm rows: 99.974% anchored, 0.026% `unanchored:inverted_means`, no other
degeneracy and no `gmm_failed` at all. The "never fall back to 0.5" policy is
correct and, as in run A, not load-bearing.

The fold fallback rate is **U-shaped in κ**: 0.118% at κ=0.01, minimum 0.059%
at κ=0.5, 0.180% at κ=3. Too little anchor mass leaves the fit near its
unanchored initialisation where a fold can still inherit inverted means; too
much pins a component on a handful of anchors and collapses it. The accuracy
optimum and the numerical-stability optimum land in the same place.

*(Measurement note: the first cut of this table read a flat 0% for the fold
family. The two families report their path differently — label arms say
`unanchored:<reason>`, fold arms say `fold_anchored[k/K]` and never contain the
word "unanchored" — so classifying both with the label family's rule silently
reports no fallbacks. Fixed in `analyze_rate.py`, and the analyzer self-test
now plants a fold fallback so the classifier is exercised.)*

## Caveats & open threads

- **Run B predates the ship PR, and #2861 moved two things under it.** The
  run's base is `7fbde84e`; #2861 merged after it. Two changes matter for
  reading these numbers across that boundary. (a) The shipped `rate` cut
  was redefined as a monotone supremum rather than a density-crossing
  root; #2861 states the two agree wherever a crossing exists **including
  at every equal-weight cut**, and run B scored everything at inclusion 0,
  so the `rate` arm here is the shipped rule. (b) #2861 made fold
  calibration run at every label count, where it used to be skipped
  wherever the schedule zeroed the x-cal cut — that changes trajectories
  below ~6 votes, so the 2–20 window is the least transferable one here.
- **The two runs are not on the same code either.** Run B is on dev `7fbde84e`, run A
  on `0a54f0d7`; PR #2849 changed the derived `<6 votes` x-cal skip in between,
  and the trajectories genuinely diverge (matched cells differ in their vote
  sequences). Run A's κ ∈ {10, 30, 100} points therefore **cannot** be spliced
  onto run B's curve, and κ=1/κ=3 are estimator-level replications, not byte
  replicates. What reproduces qualitatively: an interior optimum at low κ, and
  label-anchored degrading steeply with κ.
- **caltech101 is small on purpose and noisy in consequence.** 838 images,
  N=419, three positives in the deep regime. It is in the design as the lever
  for the κ-vs-γ question and as a low-positive stress case; its per-arm
  differences are mostly inside noise and its argmins should not be read as
  optima.
- **The folds addendum is one environment and an unpaired contrast.** VG ×
  siglip only, and K=4-vs-K=2 compares two trajectories rather than two arms —
  each run's own `Δ vs xcal_only` is what makes the comparison meaningful at
  all, and it still has no significant cell. Treat the −0.008 as a direction,
  not a number.
- **Counterfactual schedule rows cannot show acquisition feedback** (#2841's
  screen caveat). The region-voting margins are threshold-rule differences on a
  `prod`-driven trajectory.
- **Simulated voting, linear head, 4 seeds, image data only.** The MLP head,
  audio/video media, and real user sessions remain unmeasured.
- **The label-anchored family is still a trap at high κ** (+0.005 at κ=3
  pooled, and run A measured +0.105 at κ=100). Any adoption must pin κ, not
  expose it.
- Per-window FNR/cost aggregates weight steps equally; the paired contrasts —
  the decision numbers — are unaffected.

## Recommendation & next steps

1. **Change the shipped constant from `κ=1, rate` to `κ=0.3, mid`.** PR #2861
   shipped run A's recommendation; this is a one-line change to the same code
   path, and it wins head-to-head in 6/6 environments (pooled −0.0045,
   p<1e-4). It is also the best single global setting measured — worst-case
   distance from any environment's own optimum 0.0067, against 0.0102 for what
   shipped. On region voting it beats `slow_cap50` by −0.026 (p=2e-9 and
   p=3e-8 in the two region environments). (`anchored κ=0.3 rate` measures
   0.003 better on region still, and needs no per-fold scoring pass at all;
   take it only if that cost matters, and pin κ hard if you do — that family
   is a trap above κ≈1.)
2. **Scope the shipped path to region voting — this is a live regression.** PR
   #2861 made the fused threshold the cutoff for *every* detector with safe
   thresholds on, including binary-voting ones. Measured against the `cap50`
   blend it replaced there, the shipped `κ=1 rate` is **+0.0063 worse** on
   COCO and worse still on caltech101 (+0.008 / +0.003 by environment); even
   the best fusion setting, `κ=0.3 mid`, only reaches a dead heat (−0.0004,
   n.s.). Binary-voting users are getting a slightly worse threshold than they
   had before #2861. Restore `cap50` for that mode — which mirrors the
   per-mode split #2841 already shipped, for the same underlying reason: low
   positive counts want spread control, not a better-located cut. **PR #2863
   has since merged**, deleting the `safe_thresholds` setting, so the fused
   threshold is now unconditional and there is no configuration that gets a
   binary-voting detector back to the blend. That makes the mode split (or the
   positive gate in item 3) the only remaining route, and raises its priority
   from "worth doing" to "the fix". Note the fairness point: #2863 also removed
   pure cross-calibration as an option, and fusion beats *that* on binary
   voting by 0.004 — the regression is against the blend, not against x-cal.
3. **Gate on positives, not on dataset size.** The effect scales with the
   number of *positive* anchors (24 → −0.093, 8 → −0.019, 7 → −0.068, 3 →
   −0.002). A production gate of the form "use fusion once the fold anchors
   contain ≥ k positives, else the shipped blend" is directly supported; k is
   not yet estimated and deserves its own sweep of the gate value. **Tracked in
   #3550**, together with item 4.
4. **Test κ ∝ 1/n before pinning a constant.** The window trend (κ\* 3 → 0.1
   from 20 to 300 votes) says the fixed-κ parameterisation is wrong in a
   specific, correctable way. A `total_anchor_mass` variant — κ = M/n for
   fixed M, or M/n_good given the positives result — is a one-line change to
   `anchored_gmm_fit`'s caller and would plausibly recover the ~0.009 that the
   constant-κ compromise gives up at both ends. **Tracked in #3550** — note
   #2861 swept a *constant* κ and shipped 0.3, which cannot test this
   parameterisation.
5. **`qmean` is confirmed; folds is a maybe.** The addendum closes the
   combine question in production's favour. A properly powered folds A/B (more
   environments, more seeds, paired seeds where possible) is worth it only if
   −0.008 for an extra scoring pass per step looks like a good trade.
6. Still open from run A: the MLP head, and reconciling #2799's
   selection-feedback attribution (which stops at 30 votes) with the population
   term still paying at 300.

## Reproduction

**Run B.** Worktree `/exp/sgreenberg/projects/vts-rate-2861` (branch
`run/anchor-rate-2861` @ dev `7fbde84e`); launch
`scripts/experiments/calibration/launch_rate_2861.sh`. Prepare fully reused, no
GPU stage: VG + COCO pickles/crops from the #2841 mixin run, caltech101 derived
locally (boxless ⇒ whole-image exemplars, no model). Cells on the CPU partition
at **1 cpu / 8G** — the `cpu_limit` QOS caps a user at cpu=240 and SLURM charges
2 cpus per task, so that is 120 concurrent; run A's 4 cpu / 24G request bought
60 slots for cells that `sacct -o TotalCPU,Elapsed` shows are single-threaded
with 5.4G peak RSS. 384 cells in 1h51m wall, 0 failures.

- Jobs: cells array **468874**; `analyze_anchored.py` chained `afterany` as
  **468875**.
- Outputs: `/exp/sgreenberg/anchor-rate-2861/results/` — `REPORT_rate.md`,
  `rate_summary.json`, `agg/rate_*.csv` (curve / plateau / per-window /
  per-environment / vs-shipped-schedule / stability / FNR / provenance),
  `cells/task_*.csv`, `figs/*.svg`.
- Analyzers: `analyze_rate.py` (+ `selftest_analyze_rate.py`, planted-answer,
  passed before use), `make_rate_figs.py`.

**Folds addendum.** Same worktree; `launch_folds_2861.sh`
(`CALIB_CALIBRATE_COUNT=4`, VG × siglip, same κ grid, `qmean,qmedian`). Cells
array **470106**, 92/92, 0 failures. Results
`/exp/sgreenberg/anchor-folds-2861/results/`; analysis `analyze_folds.py`.

**Run A.** Worktree `/exp/sgreenberg/projects/vts-anchored-2852` @ dev
`0a54f0d7`, `launch_anchored.sh` knobs, prepare reused from
`/exp/$USER/calibration-safe-linear/results`, cells on the CPU partition
(24G / 4 cpu / %40). Jobs 468311 → 468312. Outputs under
`/exp/sgreenberg/calibration-anchored/results/`.
