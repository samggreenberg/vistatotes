# #3319 — what the acquisition offset is worth, where its frontier turns, and at what resolution

**COMPLETE.** Three waves, **4032 cells, 0 failures, 0 header-only cells**: the
12-arm shipped-arm sweep (2304 cells), the 400-click deep wave (768) and the
region cross-check (960).

Issue #3319 · branch `claude/acq-offset-3319` · base dev `faa9fa9ac` · worktree
`/exp/sgreenberg/projects/vts-acq-3319` · study `/expscratch/sgreenberg/acq-3319`.
Pre-registered plan: [`PLAN_3319.md`](PLAN_3319.md) — written before any arm cell
existed, and every rule below is the one it committed to.

---

## Summary

1. **What the offset is worth is SPEED, and the endpoint hides it.** Against no
   offset, the shipped cut reaches the same answer in **half the clicks** over a
   100-click session and **2.6× fewer** over a 400-click one. The advantage
   *compounds* with session length. A single `final_cost` number cannot express
   this and did not.
2. **The frontier is a plateau, not a peak — on every endpoint measured.** Final
   cost, area under the cost curve, and clicks-to-target are all flat from `−2`
   to `−5`; the curve only turns resolvably at `−8`. What eventually stops the
   sweep is the **spike guardrail**, not cost.
3. **Half steps are real and decision-irrelevant.** They are distinct operating
   points (0 duplicates in 2112 comparisons) and land resolvably half-way on the
   mechanism — and they separate on *no* decision endpoint, because the plateau
   is three bits wide.
4. **Inclusion is a log₂ likelihood-ratio threshold that under-delivers its own
   steps, by an environment-dependent amount** — a shortfall that shrinks when
   the ranking separates well (1.95 bits binary, 0.53 region). This is the most
   reusable finding and it retro-explains the constant's whole history.
5. **Ship `−4`** — the only arm passing the pre-registered rule in all three
   environments — **but the case is labelling efficiency, not quality or speed.**
   Within the plateau nothing distinguishes `−3` from `−4` on any endpoint.

---

## What a value means

Stated first because every number below is read in these units, and because it
is what makes a *fractional* grid principled rather than a fishing expedition.

`inclusion_cost_weights` defines the knob as a loss over the two error **rates**:
`cost = w_fp·FPR + w_fn·FNR`, with `(w_fp, w_fn) = (2^−k, 1)` for `k ≤ 0`. Because
each error is normalised by its own class, **the prevalence divides out** —
deliberately: `GmmFit1D.rate_crossing` puts the prior-odds factor back into `lam`
precisely so the realised cut does not carry it, which is what #2836 shipped.
Differentiating leaves `w_fn·f_pos(x) = w_fp·f_neg(x)`, i.e.

> **include *x* ⟺ f_pos(x) / f_neg(x) > 2^−k**

So inclusion is a **log₂ likelihood-ratio threshold**. `k = 0` is the
neutral-evidence point, and **each step of the knob is one bit of evidence**
(Good's *weight of evidence*, base 2). `−3` asks for 8:1, `−3.5` for 11.3:1,
`+2` accepts 4:1 against.

Three consequences the study leans on throughout:

* **A constant shift in evidence-bits is prior-free.** That is the real argument
  for parameterising this as an *offset* rather than an absolute inclusion, and
  why it has transferred across environments where an absolute cut would not.
* **The integer grid is coarse by construction** — one step *doubles* the
  evidence demanded, and #3318's CIs for `−3` and `−4` overlap. If the optimum
  sat at `−3.5`, an integer grid could not see it. Hence the half steps.
* **Prior-free in evidence, prevalence-bound in RANK.** Acquisition consumes the
  threshold as a *rank position*, and the bits→rank map runs through the fitted
  mixture. This is the mechanism behind the adaptive **ramp** #2876 found and
  `rank_pin` (constant by construction) lacks — and it is why the calibration
  section below is about the estimator, not about the arithmetic.

---

## The instrument, before any result is read

The direction of this knob is counter-intuitive — a *negative* offset *raises*
the cut — so a sign error would look exactly like the lever doing nothing.

| arm | k | median `acq_pool_percentile` | shift vs `prod` |
|---|---:|---:|---:|
| `prod` | 0 | 0.7252 | — |
| `acq_m3` | −3 | 0.8990 | +0.174 |
| `acq_m5` | −5 | 0.9480 | +0.223 |
| `acq_m8` | −8 | 0.9733 | +0.248 |
| `acq_p2` | +2 | 0.4009 | **−0.324** |

Every arm moved, monotonically, in the right direction, on 99% of steps.
**`acq_p2` degrades as required** — positives 7 → 4, cost +0.063 [+0.053,
+0.073], AP −0.047 — in all three environments. The mechanism is the one being
described.

---

## 1. What the offset is worth: speed

![the trajectory, not the endpoint](figures/fig6_3319_speed.png)

Every previous report on this constant, this one's first draft included, reported
`final_cost` — one point on a curve. Two arms can land in the same place having
taken very different routes, and **the route is what a user experiences**. Three
views, all paired at the (category, seed) cell:

**Clicks to reach the answer `k = 0` ends its session with**, measured the same
way on both sides (first crossing of that cell's own control final cost):

| | `k = 0` | `k = −1` | `k = −3` (shipped) | `k = −4` |
|---|---:|---:|---:|---:|
| 100-click session | 47.5 | 30.5 | **23.5** | 25.0 |
| 400-click session | 210.5 | 137.0 | **80.0** | **65.5** |

**Roughly 2× fewer clicks over a short session and 3.2× over a long one.** The
advantage *compounds*: paired against `−3`, running with no offset costs
**+17.1 clicks [+12.0, +21.8]** at a 100-click horizon and **+101.1 clicks
[+82.7, +119.5]** at a 400-click one.

**Area under the cost curve** (mean cost across warm steps) — rewards being
better *throughout*, which the endpoint cannot see:

| arm | k | AUC, 100 clicks | ΔAUC vs `prod` [95% CI] | AUC, 400 clicks | ΔAUC vs `prod` [95% CI] |
|---|---:|---:|---|---:|---|
| `prod` | 0 | 0.3600 | — | 0.3144 | — |
| `acq_m1` | −1 | 0.3528 | −0.0072 [−0.0141, −0.0003] | 0.3002 | −0.0142 [−0.0183, −0.0103] |
| `acq_m3` | −3 | 0.3384 | −0.0217 [−0.0305, −0.0131] | 0.2833 | −0.0311 [−0.0369, −0.0256] |
| `acq_m4` | −4 | **0.3340** | **−0.0260** [−0.0347, −0.0176] | **0.2814** | **−0.0330** [−0.0387, −0.0270] |

And the curve itself, at matched click budgets — the symmetric view, with no
crossing rule in it:

| arm | t=25 | t=50 | t=100 | t=200 | t=300 | t=400 |
|---|---:|---:|---:|---:|---:|---:|
| `prod` | 0.4108 | 0.3964 | 0.3574 | 0.3339 | 0.3157 | 0.3038 |
| `acq_m3` | 0.3959 | 0.3507 | 0.3242 | 0.2880 | 0.2666 | 0.2586 |
| `acq_m4` | 0.3999 | 0.3528 | 0.3224 | 0.2853 | 0.2703 | 0.2566 |

`−3` at **200 clicks** (0.2880) is already better than `prod` at **400** (0.3038).

**One honest caveat on the crossing metric.** It is computed per cell and
median-ed, which is not the same as the crossing of the median curve — on the
aggregate curves the crossings land at ~41 and ~145 clicks rather than 23.5 and
80. Both views say the same thing in the same direction; the per-cell figure is
the properly paired one and is quoted above, and the figure's markers show the
aggregate view so the two can be compared. A second caveat: `prod` reaches its
own target in 100% of cells by construction while the arms miss in 8–22%
(2.6–6.2% at 400 clicks), so the arm medians are conditioned on getting there.

---

## 2. The frontier is a plateau — on every endpoint

![the plateau, the guardrail, and the calibration gap](figures/fig4_3319_plateau_and_calibration.png)

Paired against `prod`, all 192 cells, 95% bootstrap CIs:

| arm | k | Δ final cost [95% CI] | Δ positives@100 | Δ AP | deep spikes |
|---|---:|---|---:|---:|---:|
| `acq_m1` | −1 | −0.011 [−0.019, −0.004] | +3.6 | +0.027 | 0.0% |
| `acq_m2` | −2 | −0.030 [−0.038, −0.022] | +10.1 | +0.058 | 0.5% |
| `acq_m2h` | −2.5 | **−0.034** [−0.041, −0.027] | +13.0 | +0.073 | 0.5% |
| `acq_m3` | −3 | −0.031 [−0.041, −0.022] | +17.7 | +0.083 | 0.0% |
| `acq_m3h` | −3.5 | −0.034 [−0.043, −0.025] | +22.8 | +0.088 | 0.0% |
| `acq_m4` | −4 | −0.033 [−0.042, −0.024] | +27.7 | +0.102 | 0.5% |
| `acq_m4h` | −4.5 | −0.033 [−0.042, −0.024] | +32.4 | +0.103 | 0.0% |
| `acq_m5` | −5 | −0.030 [−0.039, −0.021] | +36.6 | +0.108 | 0.0% |
| `acq_m6` | −6 | −0.023 [−0.033, −0.014] | +44.6 | +0.113 | 1.0% |
| `acq_m8` | −8 | −0.010 [−0.021, +0.000] | +52.1 | +0.111 | **2.6%** |

Contrasted **arm-to-arm against the best arm** — so the comparison does not
inherit the control's variance twice — the only arm resolvably worse is `−8`
(+0.024 [+0.015, +0.032]); `−6` is +0.011 [+0.003, +0.019], clearing zero but not
the ±0.010 tolerance. Everything from `−3` to `−5` sits within [−0.006, +0.011]
of the minimum with every CI spanning zero.

**The speed metrics agree, which is what makes the plateau a finding rather than
an artefact of one endpoint.** Paired arm-to-arm against the incumbent `−3`:

| arm | k | ΔAUC [95% CI] | Δ clicks-to-target [95% CI] |
|---|---:|---|---|
| `prod` | 0 | +0.0217 [+0.0132, +0.0304] | +17.1 [+12.0, +21.8] |
| `acq_m1` | −1 | +0.0145 [+0.0069, +0.0219] | +5.8 [+2.3, +9.4] |
| `acq_m2` | −2 | +0.0075 [+0.0009, +0.0140] | +6.3 [+2.8, +9.9] |
| `acq_m2h` | −2.5 | −0.0029 [−0.0101, +0.0041] | +2.8 [−1.0, +6.5] |
| `acq_m3h` | −3.5 | −0.0008 [−0.0072, +0.0053] | +0.0 [−3.1, +3.2] |
| `acq_m4` | −4 | −0.0044 [−0.0108, +0.0021] | −1.2 [−4.8, +2.4] |
| `acq_m4h` | −4.5 | +0.0033 [−0.0033, +0.0101] | +0.0 [−3.5, +3.4] |
| `acq_m5` | −5 | +0.0028 [−0.0034, +0.0092] | +0.4 [−3.0, +3.6] |
| `acq_m6` | −6 | +0.0059 [−0.0009, +0.0126] | −0.2 [−3.6, +3.1] |
| `acq_m8` | −8 | +0.0182 [+0.0112, +0.0255] | +3.7 [−0.7, +8.0] |

The plateau's boundaries are **sharper** here than on the endpoint — `prod`,
`−1`, `−2` and `−8` are all resolvably worse than `−3`, where on final cost `−2`
was not. But inside `−2.5 … −6` every contrast is a null on both speed measures,
exactly as on cost. At 400 clicks the same holds: `−4` vs `−3` is
ΔAUC −0.0018 [−0.0053, +0.0016], Δclicks −4.4 [−15.8, +7.1].

**So H1 is supported only in its weakest form.** The frontier turns, but at `−8`
— four bits past the shipped value and two past anything a ship rule would
consider. **`−3` sits mid-plateau: confirmed safe, and confirmed arbitrary.**

**What is not flat is the mechanism.** Positives per 100 clicks rise
monotonically and without saturation across the whole grid — 7 → 11 → 15 → 16 →
20 → 22 → 27.5 → 32 → 38 → 52 → **64** — and AP rises 0.568 → 0.722. What
eventually stops the sweep is the **guardrail**: deep-spike incidence is 0–0.5%
out to `−5`, then 1.0% at `−6` and **2.6% at `−8`**. This is closest to the
issue's second outcome: in this range the cost endpoint is not pricing
aggression at all, it is saturated, and threshold stability is the real
constraint — the same criterion #3318 found binding.

---

## 3. Half steps: real, and decision-irrelevant

**The prerequisite is discharged, emphatically.** A half step could have been an
artefact: `threshold_at` snaps its realised quantile to the haystack sample
(#3166), so one might have collapsed onto its integer neighbour and been a
*silent duplicate*. It does not. Across all 192 cells, **every adjacent arm pair
shares an identical `acq_pool_percentile` in 0.0% of cells** — not one cell in
2112 comparisons.

They are also genuinely *half-way*. Paired arm-to-arm against both neighbours:

| contrast | Δ positives@100 [95% CI] | Δ final cost [95% CI] |
|---|---|---|
| `−2.5` vs `−2` | **+2.9** [+1.6, +4.2] | −0.004 [−0.010, +0.003] |
| `−2.5` vs `−3` | **−4.7** [−6.2, −3.3] | −0.002 [−0.010, +0.005] |
| `−3.5` vs `−3` | **+5.1** [+3.5, +6.7] | −0.002 [−0.009, +0.005] |
| `−3.5` vs `−4` | **−4.8** [−6.5, −3.2] | −0.001 [−0.008, +0.006] |
| `−4.5` vs `−4` | **+4.7** [+3.2, +6.3] | −0.000 [−0.007, +0.006] |
| `−4.5` vs `−5` | **−4.2** [−5.7, −2.7] | −0.003 [−0.010, +0.003] |

**All six mechanism contrasts resolve; none of the six cost contrasts does — and
none of the six speed contrasts does either** (see the AUC/clicks table above).

So **H2 is falsified as pre-registered**, but the reason matters and is not the
one the hypothesis anticipated. The knob is *not* too coarse to have half-step
resolution: it has it, and the resolution is visibly, reliably half-way. **The
decision endpoints cannot see a half bit because they cannot see three whole bits
either.** Had the plateau been a peak, the half steps would have been exactly the
right instrument.

**Practical consequence: keep the integer grid.** The half-step arms are retired
as a tuning device, and the finer grid is not owed again unless an environment
shows a real interior optimum.

---

## 4. The calibration debt, and what it explains

![the deep regime and the replication](figures/fig5_3319_deep_and_replication.png)

The plan pre-registered a landmark. At prevalence π the selector's picks become
more likely Good than Bad only once the evidence clears the prior odds, at
`k* = −log₂((1−π)/π)`; at `vg_scale_any`'s designed **π = 7.1%** that is
**−3.71**, which is why the half-step grid was placed to bracket it.

Tested where the claim belongs — on the **pick log**, not a trajectory endpoint
(17.6k `hard` picks per arm, openings excluded):

| k | 0 | −1 | −2 | −2.5 | −3 | −3.5 | −4 | −4.5 | −5 | −6 | −8 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hard-pick precision | 4.7% | 8.6% | 15.7% | 18.9% | 24.0% | 29.5% | 34.8% | 39.9% | 44.5% | **52.9%** | 60.9% |

**The shape is exactly right.** Precision is smooth, monotone and well behaved in
`k` over eleven arms and four orders of evidence — what a log-odds threshold
should produce, and strong confirmation that the knob *is* the object the frame
says it is. Note the extreme: at `k = 0` the hard pick returns **4.7%, below the
7.1% base rate**. Sampling at the decision boundary is worse than sampling at
random, which is the cleanest possible statement of why this offset exists.

**The location is displaced.** Precision crosses 50% at **k ≈ −5.66**, not
−3.71. **H4 is not supported.**

### The displacement is a shortfall in the knob's steps, not an offset in its origin

Regressing log₂-odds of pick precision on `k` — calibrated would be intercept
`log₂(π/(1−π)) = −3.71` and slope `−1`:

| environment | intercept | slope | R² | reading |
|---|---:|---:|---:|---|
| binary `siglip × whole_image` | −3.79 | **−0.65** | 0.944 | origin right (−0.08 bits), steps ~⅔ size |
| REGION `pair × max_patch` | −3.32 | **−0.78** | 0.981 | origin +0.39 bits, steps ~⅘ size |

**At `k = 0` the knob is essentially calibrated.** What is wrong is the *gain*:
each nominal bit buys well under a bit of realised evidence.

**But it is not a constant gain, and the study's own out-of-sample point says so.**
The residuals curve systematically (−0.55 at `k=0`, +0.30 near `−4`, −0.74 at
`−8`), and `acq_p2` (`k = +2`), which was not in the fit, lands closer to the
*calibrated* line than the fitted one on the binary arm (resid −0.26 vs −0.88)
while sitting on the fitted line under region (−0.10 vs +0.73). The defensible
description is **"approximately calibrated near 0, increasingly compressed with
depth, saturating"** — a single gain number would be over-fitting, and the
earlier framing of this as "evidence overstated by 3.9×" was the accumulated
shortfall at the crossing, not a multiplier.

### What it explains

The shortfall is **environment-dependent, and it tracks how well the ranking
separates**: 1.95 bits on binary against 0.53 under region voting, which takes
oracle cost 0.382 → 0.218 and AP 0.517 → 0.762 on identical cells (#3318). A
better-separated ranking gives a mixture that is more nearly correct, so its
nominal bits are closer to true bits.

That single quantity retro-explains most of this constant's messy history: why
the optimal offset moved between environments, why region voting *tolerates*
aggression that binary voting does not, and why every attempt to ship one global
value has needed re-measurement. **The arms were never really sweeping
"aggression" — they were sweeping *nominal* bits against an environment-dependent
calibration debt.**

**It also lands on the user-facing Inclusion slider**, which drives the same
`threshold_at`. A user moving it three stops gets roughly two stops of effect.
The fix is not a global scale factor — the shortfall differs by environment and
is not a constant gain — but empirical per-detector calibration, which the
estimator has the held-out folds to do. **Tracked in #3546**, which carries the
measurement, the out-of-sample check that rules out a global gain correction, and
the two candidate knob shapes.

---

## 5. The deep regime

**768/768 cells**, `prod`/`−1`/`−3`/`−4` at `CALIB_MAX_STEPS=400`. Note the
column the harness calls `positives_100` is the trajectory's *final* value, so
here it is **positives at t=400**.

| arm | k | Δ final cost vs prod [95% CI] | Δ positives@400 | Δ AP | deep spikes vs prod |
|---|---:|---|---:|---:|---|
| `acq_m1` | −1 | −0.016 [−0.022, −0.011] | +16.8 | +0.063 | 0.5% → 1.6% (p=0.63) |
| `acq_m3` | −3 | **−0.033** [−0.039, −0.027] | +90.1 | +0.123 | 0.5% → **5.7% (p=0.006)** |
| `acq_m4` | −4 | −0.032 [−0.039, −0.026] | **+99.9** | **+0.128** | 0.5% → 2.1% (p=0.38) |

**H3's falsification condition was that the 400-click optimum be *shallower* than
the 100-click one. It is not.** `−3` and `−4` are statistically tied at both
horizons, and in absolute terms the offset is worth *more* at depth on every
measure — cost −0.033 against −0.031, positives +90 against +17.7, and the speed
advantage growing from +17 to +101 clicks.

**So the issue's worry does not materialise.** It expected the benefit to fade
because #2910 measured it as concentrated where positives are scarce, and deep
voting is where scarcity ends. The likelihood-ratio reading predicted the
opposite, on the grounds that the selector ranks the *unvoted pool*, whose
prevalence falls as positives are harvested. The measurement sides with the
latter — though weakly, for the reason immediately below.

**The genuinely new finding at depth is the guardrail.** Deep-spike incidence was
0.0% for every arm out to `−5` at 100 clicks. At 400 clicks it is live: `−3` goes
0.5% → **5.7%, p=0.006** (11 cells spiking that `prod` did not). That is the
first time this constant has shown a guardrail cost anywhere on the shipped arm.
It is **non-monotone** — `−4` is 2.1% (p=0.38), *lower* than `−3` — which is not
a shape any mechanism predicts and is what a low-rate count (11 vs 4 events in
192 cells) looks like when it is noisy. **Recorded as a live hazard for whoever
runs the next deep study, not as evidence about which arm is safer.** The deep
wave's own sizing is comfortable (binding SD 0.0437, n=74 needed, 192 run).

### The exhaustion hazard binds, and was mis-called from the pilot

The plan named positive exhaustion as the artefact that would masquerade as "the
offset stops mattering", and a single pilot cell (`backpack`, seed 0: 57 of ~150
positives, harvest rate still accelerating) was read as clearing it. **On the
full wave that reading was wrong**, and the correction matters more than the
original claim:

| arm | median positives @400 | median harvest of the ~150-positive sim half | cells >90% harvested |
|---|---:|---:|---:|
| `prod` | 22 | 14.7% | 0.0% |
| `acq_m1` | 36 | 24.0% | 0.0% |
| `acq_m3` | 123 | **82.0%** | **21.9%** |
| `acq_m4` | 128 | **85.3%** | **29.2%** |

**The aggressive arms run into their positive ceiling; the control never comes
near it.** One pilot cell was not a sample — `backpack` is simply a hard
category, and generalising from it is the mistake this table records.

Two consequences, pointing opposite ways:

* **It makes the H3 verdict conservative.** A ceiling the aggressive arms hit and
  the control does not can only *compress* their measured advantage in the late
  tail. They still win on cost, positives and speed, so "the sign does not flip"
  survives with margin.
* **But it weakens H3's positive half.** "Does the optimum *deepen*?" cannot be
  answered cleanly by arms that are ceiling-limited over the last quarter of
  their trajectory. The honest statement is the falsifier's: the optimum does not
  get *shallower*. Anything stronger needs a deeper haystack — a larger sim
  fraction or richer categories — not more seeds. **Tracked in #3547**, together
  with the guardrail finding above.

It is also a plausible mechanism for the spike rise: once positives are nearly
exhausted the remaining pool is almost all negatives, exactly the regime where a
cut fitted on a positive quantile gets unstable. That does not explain the
non-monotonicity (`−4` harvests *more* and spikes *less*), so both readings stay
on the table. **Control added:** `preflight.sh` check 16b.

---

## 6. Region cross-check — this is what decides the ship

**960/960 cells**, `prod`/`−3`/`−4`/`−5`/`+2` on `siglip+dinov3_patch`. The pair
runs both styles in one task, so the region environment is read at
`style == max_patch` **alone** — never pooled with the pair's `whole_image` rows,
which is the trap #2877 documented after a per-mode split that still pooled two
environments. Falsifier behaved (`+2`: positives −4.9, cost +0.037 [+0.028,
+0.046]); the lever moved on every arm.

| arm | k | Δ final cost vs prod [95% CI] | Δ positives@100 | Δ AP |
|---|---:|---|---:|---:|
| `acq_m3` | −3 | −0.002 [−0.009, +0.005] | +25.7 | +0.026 |
| `acq_m4` | −4 | +0.001 [−0.007, +0.009] | +35.1 | +0.024 |
| `acq_m5` | −5 | +0.004 [−0.003, +0.012] | +42.4 | +0.024 |

Against the incumbent `−3`, which is the comparison the ship rule reads:

| arm | k | Δ final cost [95% CI] | Δ positives | Δ AP | deep spikes | passes |
|---|---:|---|---:|---:|---|---|
| `acq_m4` | −4 | +0.0031 [−0.0025, **+0.0091**] | +9.4 | −0.002 | 1.6% → 1.6% | **YES** |
| `acq_m5` | −5 | +0.0064 [+0.0007, **+0.0123**] | +16.7 | −0.002 | 1.6% → 1.0% | no |

**`−4` clears the tolerance under region voting; `−5` does not.** That is the
constraint #3318 raised, and it removes `−5` despite its being free on the
shipped arm.

**The offset is worth much less here, on every axis.** Region voting reaches its
own final answer in 45.5 clicks against the binary arm's 47.5, and the offset
takes that to 32 (`−3`/`−4`) — a real but smaller speed-up, with **ΔAUC a null**
(`−3`: −0.0029 [−0.0095, +0.0036]) and 27–35% of cells never reaching the target
at all. Final cost is a null too. This is the same story as #3318's DiD: **region
geometry has already done most of the work the offset would buy** — and the
calibration section says why, since region's smaller debt means its nominal `−3`
is already closer to the real posterior flip.

**Stated rather than rounded away: `−4`'s region margin is thin.** Its upper
bound is +0.0091 against a +0.010 bar. #3318 measured this same contrast at
+0.006 [+0.001, +0.013] and rejected it; this run measures +0.0031 [−0.0025,
+0.0091] and passes. The CIs overlap heavily, so these are *consistent*
measurements that happen to straddle the bar — not a reversal, and not
independent confirmation either.

---

## 7. The ship decision

Against the incumbent `−3`, on the shipped arm, four arms pass; H2 says pick an
integer among them, and region removes `−5`:

| arm | k | Δ final cost [95% CI] | Δ positives | Δ AP | spikes | passes |
|---|---:|---|---:|---:|---|---|
| `acq_m3h` | −3.5 | −0.0024 [−0.0092, +0.0047] | +5.1 | +0.005 | 0.0% → 0.0% | YES |
| `acq_m4` | −4 | −0.0015 [−0.0081, +0.0050] | +9.9 | +0.019 | 0.0% → 0.5% | **YES** |
| `acq_m4h` | −4.5 | −0.0016 [−0.0087, +0.0053] | +14.7 | +0.020 | 0.0% → 0.0% | YES |
| `acq_m5` | −5 | +0.0015 [−0.0058, +0.0088] | +18.9 | +0.025 | 0.0% → 0.0% | YES |
| `acq_m6` | −6 | +0.0085 [+0.0008, +0.0158] | +26.8 | +0.030 | 0.0% → 1.0% | no |
| `acq_m8` | −8 | +0.0212 [+0.0124, +0.0301] | +34.4 | +0.028 | 0.0% → 2.6% | no |

**`−4` is shipped** (decided 2026-09-02, PR #3454) — the only arm passing in all
three environments (+9.9 / +9.7 / +9.4 positives, cost a null everywhere).

**But be clear what the case is.** It is *not* quality and it is *not* speed:
within the plateau, `−4` vs `−3` is a null on final cost, on AUC and on
clicks-to-target, at both horizons. The case is **labelling efficiency** —
hard-pick precision 24% → 35%, and ~28 matches surfaced per 100 clicks instead of
~20. Whether that is worth changing a shipped constant depends on how much
weight the product places on matches surfaced *during* a session, which is a
product question this study cannot answer. Staying at `−3` is defensible.

---

## Power, honestly

The realised paired SD on `final_cost` is **0.0747**, which needs **n ≈ 215** for
a ±0.010 half-width; the run has **192**, giving ±0.0106. Slightly under target,
stated rather than rounded away. It changes no verdict — the plateau's contrasts
are an order of magnitude inside the tolerance and the `−8` rejection far outside
it — but a study wanting to resolve two adjacent plateau arms on cost would need
roughly **5400 cells per arm**, since the half-step effect on cost is ~0.002.
That number is the honest reason the half-step grid is retired rather than re-run
bigger. The deep wave is comfortable at its own endpoint (binding SD 0.0437,
n = 74 needed).

---

## What this changes

* **The offset's value is speed, and it is large** — 2× fewer clicks over a
  100-click session, 3.2× over 400, compounding with length. Reporting this
  constant by `final_cost` alone understated what it does for users; any future
  report on it should carry the trajectory.
* **`−3` is confirmed safe and confirmed arbitrary.** It sits mid-plateau, and
  the plateau is flat on cost, AUC and speed alike.
* **`−4` is shipped**, on the labelling-efficiency case, with the thin region
  margin recorded in the constant's own rationale block as the number to
  re-measure first if the value is ever revisited. `−5` fails region;
  `−6`/`−8` fail the guardrail.
* **Stop tuning this constant.** There is no headroom on any decision endpoint,
  and the cost of finding some is bounded at ~5400 cells/arm.
* **The half-step question is answered and retired** — measurably half-way on
  the mechanism, invisible on endpoints that are flat across three bits.
* **The knob under-delivers its own steps, by an environment-dependent amount.**
  That is the most reusable finding, it reaches the user-facing Inclusion
  slider, and it points at a knob shaped like a *target pick precision* — which
  would be self-calibrating where a constant offset is not. **Tracked in #3546.**
* **Long sessions are untuned territory.** The guardrail is invisible at 100
  clicks and live at 400, and the harness cannot currently answer deep questions
  cleanly because the aggressive arms exhaust their positives. **Tracked in
  #3547**, which names the pile change that would fix it.

*Both open questions are issues, not paragraphs here — a follow-up written into a
report is not tracked.*
