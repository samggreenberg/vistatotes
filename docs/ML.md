# Machine Learning Details

VTSearch learns a binary classifier from user votes ("good" vs "bad") using a **linear SVM head** — a single `Linear(input_dim, 1)` fitted to the maximum-margin boundary between the two vote classes (hinge loss + L2, class-balanced). The model operates on embeddings produced by pretrained feature extractors (LAION-CLAP for audio, SigLIP for images, X-CLIP for video, E5-base-v2 for text) and outputs a score in [0, 1] for each item in the dataset.

Alongside the head, each dataset carries a **[Coverage Atlas](#coverage-atlas)** — a hierarchical partition of the embedding space that guides the autopilot's diversity sampling and provides typicality ranks for domain-shift detection.

## Architecture

The head's *architecture* is defined in `vtscore/training/mlp.py` via `build_model()`, and its *fit* in `vtscore/training/svm.py` via `fit_linear_svm_head()`. Production always builds the linear head, selected by the `hidden_dim=LINEAR_SVM_HEAD` (`-1`) sentinel:

```
Linear(input_dim, 1)
```

- **Input dimension**: Dynamic, depends on the embedding model for the current media type (see [Embedding Models](#embedding-models) below).
- **No hidden layer**: hence no ReLU and no dropout — a bare linear map has nothing to regularize with dropout.
- **Output**: A single number — the SVM's decision function, `w·x + b`. `torch.sigmoid` is applied at inference time to squash it into [0, 1].

The head is fitted by liblinear (scikit-learn's `LinearSVC`, `C=1.0`, `class_weight="balanced"`) rather than by a gradient loop, and its hyperplane is then copied into the `Linear(input_dim, 1)` module. Keeping the result a torch module is what lets the rest of the pipeline — calibration folds, max-pooled region scoring, weight serialization, the ONNX exporter, threshold fusion — stay exactly as it was.

**The sigmoid is not a calibrated probability**, and never was under the logistic head either: the decision point is the separately calibrated threshold below, not 0.5. The sigmoid is a monotone squash that keeps every downstream consumer (sorting, the score column, the quantile rules) working in [0, 1].

### The three heads: which one is shipped, and why

| Sentinel | Architecture | Fitted by | Status |
|---|---|---|---|
| `LINEAR_SVM_HEAD` (`-1`) | `Linear(D, 1)` | liblinear, hinge + L2, class-balanced | **shipped** |
| `LINEAR_HEAD` (`0`) | `Linear(D, 1)` | balanced BCE gradient loop = logistic regression | eval arm, tests |
| `hidden_dim > 0` | `Linear(D, H) -> ReLU -> Dropout -> Linear(H, 1)` | balanced BCE gradient loop | eval arm, tests |

The head used to be a small MLP with its width auto-sized by `_auto_hidden_dim(n_train)`. With only ~3–5 labeled positives that MLP is under-determined: each retrain wobbles the scores, and the calibrated cut lurches over the unlabeled positives — the threshold-stability spikes tracked in issue #2790. A linear boundary has no such freedom, and measurements on COCO/SigLIP2 and VG/SigLIP1+2 put the deep-spike rate roughly 55–73% lower at equal or better FNR and cost. That finding moved the head to `LINEAR_HEAD`.

The move from there to the SVM keeps that same linear boundary and changes only the objective that places it. A maximum-margin fit is decided by the votes nearest the boundary and is indifferent to how far the easy ones sit beyond it, whereas logistic regression keeps paying attention to every example — which is why the two place a visibly different line on the same handful of votes. Measurements in a separate environment put the SVM's ranking clearly ahead of the logistic head's; **why** it wins (better-behaved scores near the cut? more tolerance of a mis-vote?) is still open, and is the subject of follow-up sweeps. Until those land, the shipped head is the one that measures best.

Both retired heads remain reachable **by name** as eval arms (`head="linear"`, `head="mlp"`; see [`docs/EVAL.md`](EVAL.md)) and in unit tests; neither is reachable from the app. The Stage-2 structural-verification classifier (`vtscore/training/structural_similarity.py`) is a separate feature and keeps its own MLP.

## Training Configuration

| Setting | Value | Notes |
|---------|-------|-------|
| **Loss function** | Squared hinge + L2 | scikit-learn `LinearSVC`, solved by liblinear |
| **Regularization** | `C = 1.0` | `SVM_HEAD_C` in `config.py`, or the `VTSEARCH_SVM_HEAD_C` env var |
| **Solver iteration cap** | 5000 | liblinear's `max_iter`; the fit is milliseconds at any real vote count |
| **Head** | `Linear(input_dim, 1)` | The `LINEAR_SVM_HEAD` sentinel (`hidden_dim=-1`); no hidden layer, no dropout |
| **Batching** | Full-batch | All labeled data in one solve |
| **Reproducibility** | `random_state=seed` | liblinear is deterministic given the seed (default 42); the fit runs on CPU |

`TRAIN_EPOCHS`, `TRAIN_PATIENCE`, `MLP_DROPOUT` and `MLP_LABEL_SMOOTHING` no longer touch a production fit — they configure the BCE gradient loop, which now only the eval arms and the structural-verification classifier run.

### Class Weighting

Training balances classes by **inverse-frequency weighting** by default (`class_weight="balanced"`, liblinear's equivalent of the gradient loop's per-sample weights). The one exception is region flooding on patch datasets, where the caller supplies explicit per-bag `sample_weights` instead — those replace the class balance rather than stacking on it (see [Region-aware training](#region-aware-training-on-patch-datasets) below). Either way, inclusion does **not** enter training — the trained model, and therefore every item's score, is independent of inclusion. Inclusion is applied later as a pure threshold knob in `conformal_threshold` (see [Threshold Calibration](#threshold-calibration) below).

- **Weights**: `weight_true = num_false / num_true`, `weight_false = 1.0` (the same ratio `class_weight="balanced"` derives)

Keeping the model inclusion-independent is what lets the calibration cache reuse fold scores across cutoff slides: when the user changes inclusion, the labels are unchanged, the fold models are unchanged, and only the cheap quantile rule re-runs.

### Label Smoothing (BCE arms only)

The BCE gradient loop label-smooths its targets with ε = 0.05 (`MLP_LABEL_SMOOTHING` in `vtscore/config/runtime.py`): Good examples train toward 0.95, Bad toward 0.05, with class weights still derived from the hard labels. This is **not** a knob-mover — it exists as tie insurance for the conformal threshold rule below, which takes quantiles of the calibration scores and therefore needs distinct score values. Smoothing bounds the optimal logit (≈ ±2.9 at ε = 0.05), so a strongly-fit model cannot saturate every score to exact 0.0/1.0 sigmoids, where all quantiles would collapse to the same cut.

The SVM head needs no such insurance and does not use it: the margin objective has no incentive to run the decision function off to infinity, so its scores stay in a narrow band around the boundary and quantiles over them are naturally distinct.

### Threshold Calibration

A decision threshold separating "good" from "bad" predictions is computed via **cross-calibration**:

1. Split labeled data into Train (1 − `calibration_fraction`) and Calibrate (`calibration_fraction`). The two sizes are **dithered rather than rounded**: a fractional split rounds up with probability equal to its fractional part, so the count is unbiased instead of pinned to whichever side `round` picks. At a 0.5 split this fires only on an odd label count — an exact tie, where "nearest" has no answer. It matters because `round` is round-half-to-**even**, which made the odd vote's destination alternate with the label count (Train at `n % 4 == 1`, Calibrate at `n % 4 == 3`); one user never notices, but the eval casts one vote per step, so that seesaw was phase-locked across every simulated run and survived averaging as a spurious 4-vote ripple on the learning curves (issue #3286). The tie-break RNG is seeded from a digest of the labelset itself, so a threshold stays a pure function of the votes that produced it.
2. Train a fold model on Train **using the same head as the final model**, score the held-out Calibrate portion.
3. Repeat for `calibrate_count` independent random splits.
4. Pool every fold's held-out (score, label) pairs and apply the **conformal inclusion rule** (`conformal_threshold`) once. Pooling — rather than averaging per-fold thresholds — is deliberate: the knob's resolution is bounded by how many calibration scores the quantiles are taken over.

The `calibration_fraction` setting is the Calibrate share of each split, and its default is **per-embedder** (issue #3287): with no explicit user setting, a detector that learns in a single-vector space splits 70% Train / 30% Calibrate (`0.3`), while a patch-grid embedder keeps `0.5` — in *both* its voting styles, including the boxless whole-image fallback, because the measured optimum follows the embedder's capability rather than the voting mode. The table lives in `PRODUCTION_SPLIT_BY_SPACE` (`vtscore/training/thresholds/knobs.py`), resolved with an unknown-space fallback of `0.5` by `resolve_calibration_fraction` (`vtscore/detectors/training.py`); an explicit `calibration_fraction` user setting always wins, and clearing it returns to the automatic per-embedder default. Evidence: [`docs/experiments/2026-08-27-calibration-fraction-3287/REPORT.md`](experiments/2026-08-27-calibration-fraction-3287/REPORT.md).

The `calibrate_count` setting defaults to `2` (`DEFAULT_CALIBRATE_COUNT` in `vtscore/config/runtime.py`) and can be raised or lowered (`VTSEARCH_CALIBRATE_COUNT`) to trade calibration quality (and Inclusion-knob resolution — more folds means more pooled calibration scores) for latency. (The eval runner uses its own default of `2` for a separate, non-interactive path — see [`docs/EVAL.md`](EVAL.md).) The folds are trained at every label count: the fold *models* are an input to the fold-anchored estimator below — it anchors on their held-out scores — so there is nothing to skip. (Before the population-anchored adoption, the calibration was skipped wherever the mix-in schedule gave the cross-cal cut zero weight, at 6 labels or fewer; the schedule is no longer what combines the two estimators, so that skip is gone.) Every training entry point (vote-driven Train, labelset re-derivation, Find, and detector-load-from-origins) cross-calibrates below 6 labels rather than falling back to 0.5.

**Every trained threshold fuses the haystack into the cut.** There is no setting for this. It used to be the `safe_thresholds` per-user toggle, on by default since the #2799 A/B; the population-anchored adoption made the fused estimator uniformly better than the alternative at every label count, so the toggle was deleted rather than left as a way to opt into a worse threshold. The historical A/B is still worth reading for *why* the population distribution belongs in the cut: [`docs/experiments/2026-08-03-safe-thresholds/REPORT.md`](experiments/2026-08-03-safe-thresholds/REPORT.md).

**What it computes: the fold-anchored mixture, not the old blend.** The blend-vs-fusion question was measured by the #2852 deep-regime study and fusion won decisively, so the schedule was retired as the *combiner* (`fold_anchored_gmm_threshold`, adopting the study's recommendation). Per calibration fold *k*:

1. Score the whole haystack with fold model *k* and fit a **semi-supervised** 2-component mixture to those scores, with fold *k*'s **held-out** labeled scores clamped to their labeled component (Good → high, Bad → low). The anchors are honest — those items were not in fold *k*'s training set — and they share one score scale with the population they anchor.
2. Cut fold *k*'s fit at the **midpoint between its two component means**, and read that cut's empirical quantile in fold *k*'s own haystack distribution.
3. Average the fold quantiles and realize the result on the **final** model's haystack distribution (rank transfer: two models scoring the same haystack are related by an approximately monotone map, and quantiles survive monotone maps). No raw score ever crosses scales.

**The voted media are not part of any of those haystacks** (issue #3308). Every model in the chain was trained on some or all of the votes, so a voted item's score under it is optimistically shifted, and the calibration votes would additionally sit in the fold haystack twice — once as free points, once as anchors. All voted media are therefore dropped from the fold haystacks *and* from the final model's realization sample, which keeps every distribution in the quantile transfer over one identical population: the unlabeled remainder, which is also the only population the threshold still has to decide (the voted items' verdicts are their votes). The effect is bounded by the votes' share of the (≤50k-sampled) haystack — invisible on a large corpus, a measured win on small ones (simulation: −0.02 cost at 24% voted, −0.003 at 5%, 0 at ≤1%). The exclusion has one guard: it switches off entirely when it would leave fewer than `EXCLUSION_MIN_REMAINDER` (60) scores, because a drained remainder is both too coarse to read quantiles from and — after deep Autopilot voting — a selection-biased sample of exactly the items acquisition never wanted; the overlapping-data eval environment measures the exclusion as harmful below that line (+0.18 cost with <10 items left) and neutral-to-positive above it. The switch is all-or-nothing so the fold and final populations never diverge. The blend *fallback* below deliberately keeps the full distribution: it only runs when there are no usable folds, where the contamination is at most a few points.

The shipped anchor mass is **κ = 0.3** — each vote counts as three tenths of a haystack point among the ~50k the mixture is fitted on — paired with the **midpoint** cut. The first sweep only went *down* to κ=1, so its "performance degrades monotonically as κ grows" described one side of an interior optimum; extending the grid to 0.01 across six environments found the optimum at κ=0.3, and the cut rule flips with it. That flip has a mechanism: `mid` ignores the mixture weights, `rate` reads them, and anchor mass is what lets the votes' acquisition-biased prevalence into those weights. So `mid` peaks low and `rate` peaks high, the two curves cross near κ=1–2, and the first run — whose grid started at κ=1 — saw `rate` in front. Labels and haystack feed one estimator rather than being averaged as rivals on a label-count schedule; the labels' authority grows with the data instead of on a hand-tuned ramp — though the same run finds it grows *too fast*, with the best κ falling like 1/n, so a fixed *total* anchor mass is likely the better parameterisation.

Measured in the deep regime (votes 51–300) and pooled over all six environments, this cuts **−0.0437** paired regret against pure cross-calibration, and it is the best *single global* setting measured: forcing it everywhere leaves each environment within **0.0067** of its own optimum. On Visual Genome region voting alone it is −0.093, and it beats the previously shipped `κ=1, rate` head to head in **6 of 6** environments (pooled −0.0045, p<1e-4).

**One regression is on record and not yet fixed.** The same run scored the schedules that actually ship (`slow_cap50` region / `cap50` binary) as controls: against those rather than the old 6→20 ramp, fusion wins clearly on **region voting** (−0.026, p≤3e-8) but is a **dead heat on COCO binary voting and a loss on an 838-image set**, where `cap50` beats every fusion arm. The gain tracks how many *positive* anchors the regime supplies (24 → −0.093, 8 → −0.019, 3 → −0.002), so binary-voting detectors — which the fused path also covers, unconditionally since #2863 — get roughly what the blend gave them at best, with no way back to it. A voting-mode split or a positive-count gate is the fix; see [`docs/plans/population-anchored-calibration.md`](plans/population-anchored-calibration.md). The fold-count question is closed: `qmean` beats `qmedian` at every κ (indistinguishably at κ=0.3, significantly from κ=1 up), and four folds are nominally −0.008 better than two at every grid point with no significant cell. See [`docs/experiments/2026-08-05-population-anchored-calibration/REPORT.md`](experiments/2026-08-05-population-anchored-calibration/REPORT.md).

**Media that cannot be scored are not part of the population.** A head that emits a non-finite logit for a media has that media recorded at the `-1.0` sentinel (`NON_FINITE_SCORE_SENTINEL`, see [`vtscore/docs/packages/utils.md`](../vtscore/docs/packages/utils.md)), which sits outside the sigmoid range precisely so `score >= threshold` is always false for it. Every estimator here therefore drops those entries before fitting — the fold haystacks, the held-out anchors, the pooled conformal orderings, and the blend's GMM (`scored_only` in `vtscore/utils/scores.py`). Handing them to a fit instead is a sign flip, not a rounding error: a spike a full unit below the range pulls the cut below zero, where every real score clears it and the detector calls the entire dataset a match — issue #3180, where a CLI run reported a threshold of `-0.375` and a positive hit for every image. When *no* media is scorable there is no population at all, and the cross-calibration cut ships alone rather than being blended against a phantom distribution.

Degeneracy policy, per fold and then globally: a fold whose anchored fit degenerates (inverted means, collapsed component) falls back to that fold's **unanchored** GMM fit; if no fold yields a fit at all, the threshold falls back to the **blend** (`calculate_safe_threshold`), which is what still runs for label sets too small to form calibration folds — never to 0.5. A training path with no haystack at all (detector-load-from-origins, which re-derives from saved labels with no dataset in hand) ships the plain cross-calibration cut. The blend schedules therefore remain in the tree as that fallback and as harness arms, but they no longer decide the shipped cut.

The fitted mixtures are cached on the detector context and **re-cut** when the user slides Inclusion — no refit, no re-scoring — so a slide lands on exactly what a retrain at that inclusion would have stored.

**How Inclusion reaches the cut (`mid_tilt`, issue #2865).** Inclusion arrives at this estimator only as the rate weights a cut rule optimizes, and the bare midpoint ignores them — shipping it verbatim made the knob a no-op for every detector with usable folds. The shipped rule therefore anchors at the midpoint and tilts by the rate rule: in fold-quantile space, `q(k) = q_mid + (q_rate(k) − q_rate(0))` — the midpoint's combined fold quantile, shifted by however far the rate-optimal cut's own quantile moves from its inclusion-0 position. At inclusion 0 the shift is identically zero, so the shipped threshold is bit-for-bit the measured `κ=0.3, mid` arm; away from 0 it inherits the rate rule's monotone tilt without inheriting its weight-biased location, so raising Inclusion can only lower the cutoff and the included sets stay nested. (The rate cut is taken as the *highest* score at which the Bad component still out-densities the Good one under the cost tilt, which keeps it monotone in Inclusion even on fits where the density crossing leaves the inter-mean interval — and once the crossing runs off the interval the cut *continues* past the mode at the rule's own first-order slope, `var/d` per nat of log-cost, rather than pinning to the edge. Pinning made the cut constant over whole bands of the slider, which both deadened the knob there and silently collapsed the acquisition offset below to a no-op — issue #2896; the only plateau left is the honest one, where the cut runs off the haystack's support and the quantile pins at 0 or 1. A fold too degenerate for a rate cut contributes a zero shift and degrades to plain `mid`.) **The tilt is now measured** (issue #2865, [`docs/experiments/2026-08-21-inclusion-cut-rule/REPORT.md`](experiments/2026-08-21-inclusion-cut-rule/REPORT.md)): a 336-cell sweep across four environments and thirteen stops of the knob, on the shipped head, scored each row under the cost weights of its own `k`. `mid_tilt` was not beaten. Three results are worth carrying here. First, the bare `mid` cut it replaced is **inert**, not merely coarse — one admitted set for the whole slider in all 65,671 measured cell-steps — and costs up to +0.18 regret away from inclusion 0, so #2868 was a real repair. Second, `mid_tilt` and `rate` differ by the *constant* `q_mid − q_rate(0)` in fold-quantile space, so the sweep re-prices the inclusion-0 choice at thirteen cost weightings and `mid`'s location survives all of them; `rate`'s only material loss is at inclusion 0 itself. Third, the two candidate eval-only arms are priced and neither ships: `cross_tilt` (the rate solve with the mixture weights kept as priors) is better *below* inclusion 0 and worse above it, and `q_tilt` (a fixed quantile shift per inclusion step) is worse than `mid_tilt` at every one of the five step sizes swept, in every environment.

**A caveat on why `mid` beat `rate`.** The anchor-mass report attributes it to `mid` ignoring the acquisition-biased mixture weights while `rate` reads them. That mechanism is wrong: `rate` passes `lam = (fnr/fpr)·(w_lo/w_hi)` into a solve of the form `w_lo·N_lo = lam·w_hi·N_hi`, where the prior-odds factor cancels the weights *exactly*, leaving `N_lo = (fnr/fpr)·N_hi` — prior-free, and invariant to the mixture weights at every inclusion (it reads them only through the out-of-interval continuation slope). What actually separates `mid` from `rate` at inclusion 0 is the **variance asymmetry**: `rate` solves the equal-density crossing, which sits off the midpoint by `≈ var·ln(w_lo/(lam·w_hi))/(mu_hi − mu_lo)` whenever the components differ in width. The measurement stands; the explanation does not, which matters because #2865's candidate list was derived from it.

**The acquisition cut is not the decision line (`ACQUISITION_INCLUSION_OFFSET`, PR #2876; re-tuned #2905, #2909, #3318; now −4 by #3319/PR #3454).** The threshold does two unrelated jobs and they want opposite things from it. *Reporting* is the line the user sees, what Find calls a match, and what `cost = FPR + FNR` is scored at. *Acquisition* is what Autopilot's **Hard** and **New** picks consume — and those don't use it as a decision boundary at all: Hard ranks every item descending, finds the first rank position at or below the threshold, and takes the nearest unlabeled item **in rank space**, while New steers the atlas probe by a node's median score. A threshold is a *sampling position* to them.

That inverts the direction relative to the cost weights: a **negative** inclusion prices false alarms higher, *raises* the cut, moves it *up* the ranking, and so returns *more* positives. Production therefore re-cuts the same fitted estimator at `inclusion − 4` for the selector and leaves reporting where it is (`acquisition_inclusion` in `vtscore/training/thresholds/knobs.py`; `detector_acquisition_threshold` in `vtscore/state/core.py` derives it per request from the cached mixtures, so it costs a re-cut and no refit).

**The offset is −4, and it is deliberately not gated by voting mode.** The value has moved three times: shipped at −3 off one environment (#2876/#2878), cut to −1 when a second disagreed (#2891/#2909), restored to −3 by #2877's pile run (#3318), and set to −4 by **#3319** (PR #3454), which extended the grid past −4, added half-step resolution, a 400-click horizon and a region cross-check — 4032 cells, 0 failures.

**What this knob is worth is speed, and an endpoint cannot say so.** Against no offset, the shipped cut reaches the answer `k=0` ends its session with in **half the clicks** over a 100-click session (23.5 vs 47.5 median) and **3.2× fewer** over a 400-click one (65.5 vs 210.5); paired, running with no offset costs +17.1 clicks [+12.0, +21.8] at 100 and +101.1 [+82.7, +119.5] at 400. The advantage *compounds* with session length. Every report on this constant before #3319 read it through `final_cost` alone, which understated it — report a trajectory knob by its trajectory.

**Why −4 over −3, and how weak that case is.** The decision endpoint is a **plateau, not a peak**: final cost, area under the cost curve and clicks-to-target are all flat from −2 to −5 on both horizons, and −3 vs −4 is a null on all three (cost −0.0015 [−0.0081, +0.0050]; AUC −0.0044 [−0.0108, +0.0021]; clicks −1.2 [−4.8, +2.4]). The case for −4 is therefore **labelling efficiency alone** — hard-pick precision 24% → 35%, ~28 matches surfaced per 100 clicks instead of ~20 — which is a product judgement, not a quality or speed result. −3 was defensible and remains so. What bounds the value from below is the **guardrail**, not cost: positives and AP never saturate (7 → 64, AP 0.568 → 0.722 out to −8), while deep-spike incidence is 0–0.5% out to −5, then 1.0% at −6 and 2.6% at −8.

**Region voting is what rules out going deeper, and −4's margin there is thin.** On `siglip+dinov3_patch × max_patch`, against −3: −4 is +0.0031 [−0.0025, **+0.0091**] and passes; −5 is +0.0064 [+0.0007, **+0.0123**] and fails the +0.010 bar despite being free on the shipped arm. #3318 measured this same −4 contrast at +0.006 [+0.001, +0.013] and *rejected* it; #3319 measures +0.0031 and passes. The CIs overlap heavily — consistent measurements straddling the bar, not a reversal and not independent confirmation. **If this value is revisited, that is the number to re-measure first.**

**#2891's rejection is most likely a label-noise artefact** — `visual_genome_m`'s free-text labels have measured recall 0.76 over these classes, so ~a quarter of true positives are labelled negative and an arm that finds more of them is charged for them as false alarms; every clean-label environment measured adopts the aggressive arm. A well-supported explanation rather than a proven cause. **A mode gate is still refused**, for a demonstrated reason: the two *binary* environments disagree with each other more than the modes do, so the variance runs along score separability and a gate keyed on voting mode would put the wrong two on the same side. The one environment wanting −1 (`siglip+dinov3_patch × whole_image`, a patch embedder with no box supervision) rejects −2/−3/−4 on deep-spike incidence, 4.5% → 22.7/28.8/35.6%; it is not reachable today and is not what a mode gate would select. Contrast `PRODUCTION_SCHEDULE_BY_MODE`, which *is* mode-gated — that asymmetry is measured, not an oversight.

**The knob under-delivers its own steps, by an environment-dependent amount.** On the pick log, `k=0` returns **4.7% hard-pick precision — below the 7.1% base rate**, the cleanest statement of why this offset exists at all; and precision crosses 50% at `k ≈ −5.66` where the evidence semantics predict −3.71. Regressing log₂-odds of pick precision on `k` gives intercept −3.79 / slope **−0.65** on binary and −3.32 / **−0.78** under region, against a calibrated −3.71 / −1.00: the *origin* is right and the *steps* are short, and the shortfall shrinks when the ranking separates. It is not a constant gain (residuals curve; `k=+2` does not sit on the fitted line), so "approximately calibrated near 0, increasingly compressed with depth" is what the data supports. **This reaches the user-facing Inclusion slider**, which drives the same `threshold_at`, and it explains the history above: the arms were never sweeping "aggression", they were sweeping *nominal* bits against an environment-dependent debt.

**The deep regime does not flip the sign, but it wakes the guardrail.** At 400 clicks the offset is worth *more* (cost −0.033 vs no offset, +90 positives) — but deep-spike incidence, 0.0% for every arm at 100 clicks, goes 0.5% → 5.7% (p=0.006) at −3 and 2.1% (p=0.38) at −4. Non-monotone, so a hazard to watch rather than evidence about which arm is safer. Long sessions remain the least-measured regime, and the harness cannot currently answer them cleanly: the aggressive arms harvest 82–85% of their sim positives by t=400 while the control harvests 15%.

**What the conservative reading gave up, and why the successor still matters.** On a starved environment −1 finds 6 positives per 100 votes where −3 finds 18 — the gain #3318 recovered and #3319 extended. Under binary voting the benefit is sharply concentrated in *starved* cells and turns negative in well-supplied ones — measured on axes independent of the arm being scored (AP response slope **−0.0207** on log category prevalence, CI [−0.0259, −0.0159]; **−0.0402** on a leave-one-out baseline). The offset is a starvation remedy whose price is charged everywhere, so the principled successor is a **supply-dependent** offset — aggressive while positives are scarce, relaxing as they accumulate — which the detector can drive off its own positive count and which subsumes the voting-mode question (#2910). See [`REPORT.md`](experiments/2026-08-07-acquisition-inclusion/REPORT.md) (COCO), [`REPORT_SECOND_ENVIRONMENT.md`](experiments/2026-08-07-acquisition-inclusion/REPORT_SECOND_ENVIRONMENT.md) (VG binary), [`REPORT_REGION_VOTING.md`](experiments/2026-08-07-acquisition-inclusion/REPORT_REGION_VOTING.md) (VG region — **voided by #2943, read its banner before citing anything in it**) and [`REPORT_PILE_2877.md`](experiments/2026-08-07-acquisition-inclusion/REPORT_PILE_2877.md) (the pile run: three environments, both voting modes, and the region check that was owed). **Note the pile run complicates #2910's premise rather than supporting it**: prevalence is held flat there by construction, and the arms still split — along score separability, not supply. A supply-dependent offset should be re-argued against those three environments before it is built. **#3319 sharpens the successor question rather than settling it**: the decision endpoint is flat across three bits, so a supply-dependent *offset* would be tuning inside a plateau. What that study points at instead is a rule targeting a **pick precision** — self-calibrating, and immune to the step-shortfall above, which a constant offset is not (**#3546**). The deep regime it could not settle, and the pile change that would settle it, are **#3547**.

**That measurement is one environment, and a second one disagrees (issue #2877).** Rerun verbatim on `visual_genome_m × siglip`, the mechanism reproduces exactly — the sampling position moves *further* (+0.121 pool percentile against COCO's +0.058), positives per 100 votes go **6 → 12**, the `+2` falsifier falsifies on every endpoint, and the adaptive ramp is identical — but the payoff inverts: final cost degrades roughly monotonically in `|k|`, and `-3` **fails the same pre-registered ship rule**, with a 95% CI of **[+0.003, +0.022]** on the mean cost delta against a +0.01 tolerance. Only `k=-1` passes. The mechanism is legible: `regret` (cost minus oracle cost) is *flat* in every negative-`k` arm, so the cut estimator is blameless; what changes is the learned ranking, in two directions at once. Average precision **rises** (0.349 → 0.371, p<1e-5) while oracle cost — `min_θ (FPR+FNR)`, a statement about *global* separability — **rises too** (0.395 → 0.410). Aggressive acquisition sharpens the head of the ranking and blurs its tail; AP sees the first, and a globally-placed reporting cut sees the second. COCO escaped this only because it was starved hard enough that any positive helped everywhere. The ranking benefit also **saturates at `k=-2`** while the cost penalty keeps growing. **Note that this second environment is also binary voting** — `visual_genome_m × siglip` carries no `patch_grid`, so its `region_voting` flag was a no-op — so `-3` is over-fitted to `coco_val × siglip2` rather than to binary voting as a class, and the region-voting generalisation check needed `dinov3_patch`. **It has since been run** — see [`REPORT_PILE_2877.md`](experiments/2026-08-07-acquisition-inclusion/REPORT_PILE_2877.md), which finds region voting *tolerant* of the aggressive offsets this environment rejected, and finds the head-sharpening/tail-blurring split above to be a property of the environment rather than of the voting mode. That reading is superseded: #3318 restored −3 and #3319 shipped **−4**, on clean-label environments only, with the region cross-check passing. See [`docs/experiments/2026-08-07-acquisition-inclusion/REPORT_SECOND_ENVIRONMENT.md`](experiments/2026-08-07-acquisition-inclusion/REPORT_SECOND_ENVIRONMENT.md).

Two things about the shape are worth keeping. **It is an offset, not an absolute cut** — the mechanism is the *gap* between where the line is drawn and where sampling happens, so reading −3 absolutely would collapse the gap to nothing at Inclusion −3 and invert it below that. And **the ramp is not a parameter anyone chose**: early on the anchored mixture is wide, so the tilt has little leverage and the acquisition cut sits near the reporting line; as the fit sharpens it climbs (pool percentile 0.840 → 0.932 → 0.961 across a run). A *pinned* quantile — the same intent with one fewer indirection — is constant by construction, is maximally aggressive from step 1 against a model trained on almost nothing, and returned 6 positives against −3's 18. The adaptive ramp is doing the work, which was the strongest evidence that `mid_tilt` tilts usefully before its *reporting* role was measured directly (issue #2865; the tilt held).

Everything else still reads the reporting cut: the green/red line, the above-threshold count, Find verdicts and the Find boundary walk, exports, and the labeling-progress indicators. Only the two Autopilot picks moved, and only the learned sort's response carries an `acq_threshold` for them.

**Why fold models share the final model's head:** a different head produces a different score distribution, so a threshold found on fold models would not transfer faithfully to the final model. The training code threads one `hidden_dim` sentinel through both the final fit and every fold fit, so the two can't drift apart. With a linear head that value is a constant (`LINEAR_SVM_HEAD`), which makes the property automatic — the head has no capacity to size. It mattered more under the old MLP, whose width was auto-sized from the training-set size: left alone, each fold would have trained on fewer examples and got a narrower hidden layer than the final model.

The **Calibration Fraction** setting (0–1, default 0.5) controls how much data is reserved for threshold calibration vs. model training in each split. For example, a value of 0.2 means 80% Train / 20% Calibrate. If the fraction is so extreme that a valid Train/Calibrate split cannot be formed (fewer than 2 training examples or fewer than 1 calibration example), the system returns a maximum threshold so that nothing is predicted as Good.

The threshold is a **split-conformal quantile rule** over the pooled held-out scores, governed by the `inclusion_value` parameter (integer in range [-10, +10]). This is where inclusion biases the result toward recall or precision — at calibration/threshold time, **not** at training time. For `k = inclusion_value` (with `BASE = 0.25`, `QPOS_MAX = 0.75` — `CONFORMAL_BASE_BUDGET` / `CONFORMAL_QPOS_MAX` in `vtscore/training/thresholds/conformal.py`):

- A **false-negative cap** `α(k) = min(1, BASE·2⁻ᵏ)`: the threshold never exceeds the α-quantile of the held-out *positive* scores, so an estimated at-most-`α` fraction of true matches falls below the cut. `+k` therefore has a portable meaning — "the fraction of true matches I'm willing to miss, halving per step" — independent of dataset or detector (e.g. `+3` ≈ miss at most ~3%, `+10` ≈ miss at most ~0.02%). The cap is an upper bound, not a target: when the classes separate cleanly the cut drops into the gap below the calibration positives and the budget goes unspent.
- A **false-positive guard** for `k ≤ 0`: the threshold stays at or above the `1 − BASE·2ᵏ` quantile of the held-out *negative* scores (so overlap-heavy tasks keep FPR control) and above a walk *up* toward the positive score distribution. The walk interpolates linearly in score space from the **gap midpoint** at `k = 0` to the `QPOS_MAX` quantile of positives at `k = −10` ("just the surest matches").

The **gap midpoint** is the cut's default anchor. When the classes separate cleanly there is an empty band between the top of the negatives and the lowest calibration positive, and every cut inside it has identical empirical error on the calibration set. The band's top edge — the lowest calibration positive — is therefore an arbitrary choice among equals, and it is the worst one: it is an extreme order statistic over a handful of held-out votes (so it lurches from vote to vote), and it is measured on the *fold* models' score scale while being applied to the *final* model's scores. The fold models train on half the votes and saturate, so their lowest held-out positive routinely lands above every score the final model produces — a cut that admits nothing at all, not even the items the user voted Good, self-healing on the next click when the fold split is redrawn (issue #2781). Sitting in the middle of the band is the max-margin choice among cuts the calibration data cannot distinguish, and it spends no FN budget: the midpoint is strictly below every calibration positive. Under class overlap there is no band, the midpoint collapses to the false-positive guard, and the rule is unchanged.

The rule is **monotone in `k` by construction**: raising inclusion can only lower the threshold, so included sets are *nested* — everything included at Inclusion 1 stays included at Inclusion 4. That makes "cut off at Inclusion 1, then verify the extra band up to Inclusion 4" a well-defined workflow. (The previous min-cost argmin over observed cuts had exactly as many distinct optima as the calibration folds had ranking errors, so on well-separated votes the knob provably never moved; see `docs/experiments/2026-07-27-inclusion-knob/REPORT.md` and issue #2693.)

Because the fold models are inclusion-independent, the pooled held-out scores can be cached once and re-thresholded at any inclusion (this powers the Find Stats sweep across all inclusion values).

For semantic (text/example) sorts, a **GMM-based threshold** is used instead: a 2-component Gaussian Mixture Model is fitted to the score distribution and the cut is placed at the **midpoint between the two fitted component means**. The same cut is the GMM half of the safe-threshold blend, which is now only the fallback for label sets too small to form calibration folds.

**Why not the equal-density crossing?** Issue #2798 replaced the midpoint with the crossing of the two weighted components — solving `w_lo·N(x; μ_lo, σ²_lo) = w_hi·N(x; μ_hi, σ²_hi)`, the Bayes boundary between them — on the argument that under **region voting** (a media's score is the max over ~24 region-node scores) the Bad mode is an extreme-value statistic: right-skewed, wider and much heavier than the Good mode, which puts the crossing *above* the midpoint and means the midpoint cuts inside Bad mass. The #2799 study measured the two rules as paired within-step variants (each re-cutting the same model on the same votes) and the crossing lost on cost in every max-pooled window: +0.0036 at 6–20 votes, +0.0059 at 2–5 (`docs/experiments/2026-08-03-safe-thresholds/REPORT.md`). The geometry argument holds in *direction* — the crossing does cut higher and does buy FPR — but the exchange rate is ~1.3 FNR per 1 FPR, and for a needle-finding tool the missed match is the worse error. So issue #2833 reverted production to the midpoint. The crossing solver stays in the tree as an eval variant, and issue #2836 is the open question of why the midpoint wins (leading hypothesis: the crossing is the count-optimal cut while we score a *rate* loss, so its prior-odds term is a bias, and the right cut is a third point rather than either of these two).

### "All my Goods rank above all my Bads" — is the head over-training?

No, and the symptom cannot tell you either way. Embeddings are 512–768-dimensional, so a few hundred labeled points are almost always linearly separable for **any** labeling — including a pure coin flip. A head that fits its own training labels perfectly is therefore doing what a correctly-specified model does on separable data, not memorizing; shrinking the head would not change the symptom, because the symptom is a property of the geometry rather than of the model's capacity.

Over-training is only visible on items the head did **not** train on. The shuffled-label control makes that concrete: run the same dataset, the same train/held-out split, and the same head twice, changing only the labels.

- **REAL** — label each item by its true category (a signal that exists).
- **NOISE** — label each item by a fixed coin flip (a signal that does not).

Both arms reach ≈1.0 training AUC. That is the whole point: perfect training separation happens even for pure noise, so it proves nothing. The held-out numbers are what separate them:

| Arm | Train AUC | Held-out AUC | Reading |
|---|---|---|---|
| REAL | ≈ 1.0 | well above 0.5 | The head is genuinely learning; the perfect training separation is *good behavior*. |
| NOISE | ≈ 1.0 | ≈ 0.5 | The head correctly fails to generalize noise — it is not a memorizer faking held-out signal. |

A high held-out AUC on the NOISE arm would be the real alarm: that is label leakage or a broken split, not over-training.

The same control answers the neighbouring question — "is the capacity right?" — by sweeping the hidden width (or the head sentinel) with the split, votes and seeds held fixed, and reading held-out AUC per width. Note that production ships the **linear SVM head**, which has no width to sweep: capacity questions apply to the BCE eval arms in [The three heads](#the-three-heads-which-one-is-shipped-and-why) above.


## PyTorch Environment Settings

| Setting | Where | Value |
|---------|-------|-------|
| `OMP_NUM_THREADS` | `app.py` | `1` |
| `MKL_NUM_THREADS` | `app.py` | `1` |
| `torch.set_num_threads` | `vtscore/embedding/loader.py` | `1` |
| dtype | `training.py` | `torch.float32` |
| Device | default | CPU (GPU supported, see tests) |

Threading is restricted to 1 to minimize memory overhead; the real cost is the embedding models, not the head. (The head's own fit is liblinear on the CPU and is milliseconds either way.)

## Embedding Models

Each embedder produces fixed-size embedding vectors from a pretrained model. The full roster is generated from the live registry (the `document` media type has no embedder of its own, so it is absent here — documents are converted to other media types before embedding):

<!-- BEGIN GENERATED: embedders -->
<!-- Generated by scripts/gen-docs-inventories.py; do not edit by hand. Refresh with: python scripts/gen-docs-inventories.py -->

| Media type | Embedder | Display name | Model | Dim | Notes |
|---|---|---|---|---|---|
| `audio` | `clap_general` | CLAP (general, larger) | `laion/larger_clap_general` | 512 | **default** for its media type |
| `audio` | `ast` | AST (audio spectrogram) | `MIT/ast-finetuned-audioset-10-10-0.4593` | 768 | no text queries |
| `audio` | `beats` | BEATs (audio events) | `lpepino/beats_ckpts` | 768 | no text queries |
| `audio` | `clap` | CLAP (general, faster) | `laion/clap-htsat-unfused` | 512 | — |
| `audio` | `clap_music` | CLAP (music) | `laion/larger_clap_music_and_speech` | 512 | — |
| `audio` | `paraspeechclap` | ParaSpeechCLAP (speech style) | `ajd12342/paraspeechclap-combined` | 768 | — |
| `audio` | `whisper_encoder` | Whisper encoder (speech) | `openai/whisper-base` | 512 | no text queries |
| `face` | `face` | FaceNet (face identity, 512d) | — | 512 | no text queries |
| `image` | `siglip` | SigLIP (general images) | `google/siglip-base-patch16-224` | 768 | **default** for its media type |
| `image` | `clip` | CLIP (general images) | `openai/clip-vit-base-patch32` | 512 | — |
| `image` | `clip_l` | CLIP ViT-L/14 (evaluation only) | `openai/clip-vit-large-patch14` | 768 | **evaluation only** — registered, but withheld from the app's pickers |
| `image` | `dinov2_patch` | DINOv2 patch (region-aware images) | `facebook/dinov2-base` | 768 | no text queries; patch grid (region-aware) |
| `image` | `dinov2_single` | DINOv2 single (image vector) | `facebook/dinov2-base` | 768 | no text queries |
| `image` | `dinov3_patch` | DINOv3 patch (region-aware images) | `facebook/dinov3-vitb16-pretrain-lvd1689m` | 768 | no text queries; patch grid (region-aware) |
| `image` | `dinov3_single` | DINOv3 single (image vector) | `facebook/dinov3-vitb16-pretrain-lvd1689m` | 768 | no text queries |
| `image` | `eupe_patch` | EUPE patch (region-aware images) | `https://huggingface.co/facebook/EUPE-ViT-B/resolve/main/EUPE-ViT-B.pt` | 768 | no text queries; patch grid (region-aware); restricted model license |
| `image` | `eupe_single` | EUPE single (image vector) | `https://huggingface.co/facebook/EUPE-ViT-B/resolve/main/EUPE-ViT-B.pt` | 768 | no text queries; restricted model license |
| `image` | `sift_vlad` | SIFT/VLAD (instance matching) | — | 8192 | no text queries; geometric verification |
| `image` | `siglip2` | SigLIP 2 (general images) | `google/siglip2-base-patch16-224` | 768 | — |
| `image` | `siglip2_l` | SigLIP2-L (SO400M/384) | `google/siglip2-so400m-patch14-384` | 1152 | — |
| `image` | `siglip_l` | SigLIP-L (SO400M/384) | `ViT-SO400M-14-SigLIP-384` | 1152 | — |
| `text` | `e5` | E5 (text) | `intfloat/e5-base-v2` | 768 | **default** for its media type |
| `text` | `bge` | BGE (text) | `BAAI/bge-base-en-v1.5` | 768 | — |
| `video` | `xclip` | X-CLIP (video) | `microsoft/xclip-base-patch32` | 768 | **default** for its media type |
| `video` | `languagebind` | LanguageBind (video) | `LanguageBind/LanguageBind_Video_V1.5_FT` | 768 | — |
| `video` | `videomae` | VideoMAE v2 (action features) | `OpenGVLab/VideoMAEv2-Base` | 768 | no text queries |

<!-- END GENERATED: embedders -->

Each embedder lives in its own `embedder_<name>.py` file inside the media-type package and exposes a module-level `EMBEDDER` sentinel; the default for a given media type is whichever embedder overrides `is_default` to return `True` (exactly one per media type). Each embedder declares its output dimensionality via the `embedding_dim` descriptor property (the "Dim" column above).

Most media types ship alternative embedders alongside the default. The image variants come in **single/patch pairs**: `_single` embedders produce one CLS-pooled vector per image (cheap, same shape as SigLIP); `_patch` embedders additionally produce the raw `H × W × D` patch grid (196 vectors on a DINOv3 14×14), enabling region-level similarity, region-aware detector scoring, and region voting on yes-votes.  See [`docs/plans/patch-embedder.md`](plans/patch-embedder.md) for the full design.

Embedders carry capability flags consumed by the routes layer and the frontend:

- `supports_text: bool`: whether the embedder can embed text queries. Text-sort returns HTTP 400 + `supports_text: false` when this is false.
- `supports_patch_regions: bool`: set on the `_patch` variants. Loaders that see this flag populate `media["patch_grid"]` (raw `H × W × D` fp16) in addition to the embedder's vector in `media["embeddings"]`.
- `license_notice: Optional[str]`: non-None for embedders with usage restrictions (e.g. EUPE's FAIR Noncommercial Research Licence). Surfaced as a warning chip on the embedder picker.

The **document** media type has no embedding model of its own. Documents (PDF, DOC, PPT) are intended to be converted to other media types (images or text) via media converters in `vtscore/converters/` before embedding.

Embeddings are computed once when a dataset is loaded. The full-image vector lands in each clip's `"embeddings"` dict, keyed by embedder name (`numpy.ndarray` values; read it through the `media_embedding` accessor); patch embedders additionally populate `"patch_grid"` (`H × W × D` fp16 ndarray, re-derived at load — never persisted). The detector head trains on these pre-computed vectors, so training is fast (typically < 1 second for 200 epochs on a few hundred labeled examples).

### Description enrichment

The Settings toggle **Enrich descriptions** (`enrich_descriptions`, off by default) changes one thing: the *query* vector. With it on, `embed_text_query` replaces the embedding of what you typed with the L2-normalised mean over the embedder's `description_wrappers` applied to it — `clap_general`'s are `"the sound of {text}"`, `"a recording of {text}"`, `"{text}"`, `"audio of {text}"`, `"the noise of {text}"` — via `MediaEmbedder.embed_text_enriched`. Media vectors are untouched, and the Text Sort box (`vtsearch/routes/sorting.py`) is its only consumer. It costs 5 text-encoder passes instead of 1 on the query alone (~20-30 ms), cached per `(embedder, media type, enrich, text)`.

**It is not a general improvement, and the default stays off.** Measured across every media-type default on 22 eval datasets and 560 paired queries ([`docs/experiments/2026-08-31-enrich-descriptions-3127/`](experiments/2026-08-31-enrich-descriptions-3127/REPORT.md), issue #3127), paired difference in text-sort average precision:

| media type | embedder | enrichment |
|---|---|---|
| `audio` | `clap_general` **(default)** | **+0.014 ± 0.009** — helps, but below 2 SE |
| `audio` | `clap` | −0.010 ± 0.008 |
| `image` | `siglip` **(default)** | −0.001 ± 0.002 — inert on average, and the sign varies by corpus (`enrico` +0.01, `vggface2` −0.008) |
| `text` | `e5` **(default)** | **−0.057 ± 0.009** — worse on 45 of 45 categories |
| `text` | `bge` | −0.059 ± 0.009 — worse on 45 of 45 |
| `video` | `xclip` **(default)** | +0.008 ± 0.014 on AP, but −0.010 on P@10 |

Two things worth knowing before changing any of this:

- **The effect belongs to the embedder-and-corpus pair, not to enrichment.** The wrappers were written per media type, and only two templates in the tree have a positive point estimate on their own embedder: `"the sound of {text}"` on `clap_general` (+0.014, which is the whole of that media type's gain) and `"a media showing {text}"` on `xclip`. Every wrapper measured negative on `siglip`, `e5`, `bge` and `clap`.
- **`{text}` is itself one of the five templates**, so where an embedder has wrappers, enrichment always averages the plain query back in. Where the other four hurt, that is the only reason the ensemble did not hurt as much as they did — it is why `siglip`'s four negative templates netted out to only −0.001.

**Which is why the wrapper list is now per-embedder** (issue #3341). Because the sign belongs to the model rather than the media type, `siglip`, `e5`, `bge` and `clap` declare **no wrappers at all**: `embed_text_enriched` falls back to `embed_text`, and the toggle is a no-op for them instead of a small silent cost. `clap_general` and `xclip` — the only two with a positive point estimate on their own model — keep theirs. The global default still stays off: `clap_general`'s +0.014 does not clear its own 2 SE, ESC-50's 50 categories are already fully used, and resolving it needs a second audio corpus.

Two general CLAP checkpoints disagree on the *identical* five templates, so **a sibling model's prompts are not evidence for yours**. A new embedder should leave `description_wrappers` empty unless it has measured otherwise — see [EXTENDING-media.md](EXTENDING-media.md#adding-a-media-embedder).

`face` (no text tower) and `document` (no embedder of its own) cannot text-sort, so the setting is inert for them.

### Region-aware training on patch datasets

Every patch media has one **score-row stack** — `media_score_rows` in `vtscore/embedding/matrix.py` — and it is the single definition of the geometry:

| row | vector | box |
|---|---|---|
| `0` | the image-level (CLS) vector | whole image |
| `1 .. H·W` | every raw patch, row-major (`1 + r·W + c` is grid cell `(r, c)`) | that one cell |

Inference max-pools the head over that stack (an image scores by its **best** row — see `score_media`). Training is shaped to match, and it is deliberately asymmetric between Good and Bad votes — the multiple-instance-learning treatment of a max-pool bag:

- **Good vote** — a positive bag needs only *one* good region, and the user tells us which via an optional `region_box` (drawn by Shift-drag on the focus pane). The vote trains on **the raw patch nearest the box** (`nearest_patch_to_box`), which is by construction one of the rows the max-pool will score. A Good vote with no box falls back to the image-level vector — row 0 of the same stack.
- **Bad vote** — a negative bag asserts that *no* region is good, so a Bad vote **floods the entire stack** (image-level vector + every raw patch) as negatives. This trains every row down, so the max-pool can't surface a look-alike sub-region of a rejected image.

**The invariant tying the two together: every vector a vote can train on must also be a row that is scored.** Both bullets above call the same `media_score_rows`, so it holds by construction rather than by two implementations agreeing. It is not cosmetic — an early MaxPatch prototype scored raw patches only, so a *boxless* Good vote trained on the image-level vector, a vector nothing ever scored. The classifier learned to separate "full-image-like" from "raw-patch-like" (every Bad vote floods raw patches), and the calibrated threshold landed in a gap the score distribution never reaches: perfect ranking, zero FPR, catastrophic FNR.

This geometry (**MaxPatch**) replaced a HAC region tree in #2886. The old pipeline pooled patches into ~12 saliency-weighted k-means leaves, merged them into a 24-node binary tree, snapped Good votes to the best-IoU node, and flooded only the childless nodes. Over 23 scale-band Visual Genome categories × 3 seeds, tree-free MaxPatch beat it on ErrorCost by a paired Δ = −0.064 (Holm p = 0.002) and was best-or-tied-best in *every* scale band, on both halves of the error; the edge is largest on small objects, where a raw patch is a near-pure object sample while the tree's smallest pooled leaf already blends object with context. Dropping the tree also removed the #2731 flood/score gap (internal HAC nodes were scored but never flooded, because renormalised merge vectors are not dominated by their own leaves) — MaxPatch has no internals, so the exception is gone rather than inherited. Ingest gets cheaper and the payload gets *smaller*: the grid was already stored alongside the tree. See [`docs/experiments/2026-07-29-max-patch/REPORT.md`](experiments/2026-07-29-max-patch/REPORT.md).

**Measured cost of the swap** (DINOv3 14×14, D = 768, CPU):

| | MaxHAC (before) | MaxPatch (now) |
|---|---|---|
| ingest, per image | 2.52 ms (k-means leaves + O(k³) merges + fp16 cast) | 0.46 ms (fp16 cast only) |
| scored rows per image | 24 | 197 |
| flattened score matrix, per 10k images | ~740 MB float32 | ~3.0 GB float16 |
| scoring forward pass, per image | ~13 µs | ~110 µs |

So ingest gets ~2 ms/image cheaper (≈21 s per 10k images) and the stored payload gets *smaller* — the grid was already being pickled alongside the tree, so the tree's ~150 MB per 10k images is pure saving. Scoring is where the cost went: ~8× the rows, and a retrain runs three scoring passes (the final model plus one per calibration fold), so a 10k-image collection pays roughly 3 s per vote instead of 0.4 s.

Two things keep that bounded. The flattened matrix is kept **float16** (the grid's own dtype) and upcast chunk-wise by both consumers (`_forward_sigmoid_chunked`, `chunked_row_scores`), so peak float32 memory is `ROW_CHUNK × D × 4` regardless of dataset size; and `_build_region_arrays` allocates the matrix once and fills it in place rather than concatenating per-media blocks, which would hold 2× the matrix at peak. The matrix itself is cached on the `DatasetContext` and rebuilt only when the media-id set changes — never per vote.

Because flooding turns one Bad vote into ~197 correlated rows, class balance and calibration are **per-bag, not per-row**:

- The final fit is `train_model(..., sample_weights=...)` where each Bad image's rows share one image's worth of negative mass (`_per_bag_fit_weights`), so a rejected image counts once regardless of row count. Good votes weigh `n_bad_bags / n_good`, matching the default inverse-frequency balance but with the *bag* as the unit.
- Cross-calibration (`compute_fold_orderings(groups=...)`) splits Train/Calibrate **by bag** (a Bad image's rows never straddle the boundary), sizes fold counts over votes not rows, weights fold fits per-bag, and **max-pools each calibration group to one score** — so the threshold is placed on the per-image score scale the detector actually deploys. Hidden-layer width and the fallback blend's ramp likewise size on vote count.

Flooding applies wherever a detector is trained from labels the app will then score by region-aware max-pool: the Learned-sort vote path (`train_and_score`), the saved-detector labelset path (`labelset_train_and_score` / `train_from_labelset`), and — since #3525 and #3544 — **both on-demand cold trains, which delegate to that same labelset path** rather than re-deriving a head of their own: cross-dataset Find's (`_score_with_cold_detector`) and the active-dataset one behind `/api/find-label`, `/api/auto-detect` and the portable export (`resolve_or_train_detector`). That delegation is what makes one labelset mean one detector: Find used to read each label's image-level vector directly, so a Good vote's `region_box` was discarded instead of pooled to the raw patch under it, a Bad vote contributed one vector instead of its flooded stack, and calibration collapsed per row instead of per bag — a detector trained by drawing boxes silently became a whole-image detector the moment Find pointed it at a dataset it was not loaded against. Both of Find's scorers therefore now read `scoring_rows_for_snap` at the ordinary max-pooled geometry: the *live* one scores a head that **was** flooded and whose `det_ctx.threshold` was cut on the max-pooled distribution, and the *cold* one scores a head just built the same way. Reading the image-level row alone for either would compare a systematically lower score against a cut placed on a higher one, and the detector would under-return on every patch dataset it is pointed at.

The remaining unflooded path is label-file sort, which scores each image by one vector and is intentionally *not* flooded (flooding patch negatives while scoring one image vector would be a train/score space mismatch). `scoring_rows_for_snap(..., region_pooling=False)` exists for exactly that shape of caller — a head that never saw a patch as a negative must not have a distractor patch promoted into its media score — and has no shipped caller now that neither cold path builds a whole-image head. On any dataset whose embedder produces no patch grid, every bag holds one row and the whole path collapses byte-for-byte to the historical single-vector BCE — fully backward-compatible.

## Coverage Atlas

The Coverage Atlas (`vtscore/coverage/atlas.py`, class `CoverageAtlas`) is a hierarchical k-means partition of a dataset's embedding space that remembers, per region, how much labeled evidence of each class the user has provided. It serves two jobs:

1. **Diversity sampling** — the Training autopilot's "Explore Diversity" phase asks it for the next item to label, so a handful of clicks covers the whole collection and stress-tests the model where it is most likely to be wrong.
2. **Domain-shift detection** — it answers "how typical is this item of the data this atlas was built on?" with a calibrated p-value, so a detector trained on dataset A can be sanity-checked against dataset B before anyone trusts its scores there.

One atlas exists per dataset (`DatasetContext.coverage_atlas`). It replaced the earlier Diversity Tree, which kept only a boolean "seen" flag per region; the atlas keeps the geometry and statistics the tree threw away. The full design study (including the not-yet-built portable artifact, blob scan, and active auditor) lives in [`docs/plans/coverage-atlas.md`](plans/coverage-atlas.md).

### Geometry: center, then normalize

All stored embeddings are unit vectors (L2-normalized at ingest), and contrastive embedders concentrate them in a narrow cone — raw cosines between any two items are uniformly high, which makes raw directions nearly useless for partitioning or typicality. The atlas therefore works in a **centered spherical frame**: it subtracts the dataset's mean vector and re-normalizes to the unit sphere. The centering vector is part of the structure and every query is mapped into the same frame.

One consequence worth remembering: the **root node is directionally degenerate by construction**. Centering makes the sum of all vectors (the "resultant") vanish, so the root has no preferred direction. Everything below the root — the k-means cells — is cohesive and directional. Several behaviors key off this via the resultant length `rbar` (see the calibration gate below).

### Build

Built automatically at dataset load for datasets up to 50 000 items (`COVERAGE_ATLAS_AUTO_THRESHOLD`), on demand via `POST /api/datasets/registry/<id>/coverage-atlas` for larger ones, and cached inside the dataset pickle (key `"coverage_atlas"`, format `"coverage-atlas/1"`) so reloads skip the k-means. The build is recursive k-means (k = 3) over the centered vectors, splitting until a node has fewer than 20 items (`min_node_size`) or the depth cap is hit (`auto_max_depth` bounds the leaf count at ~4 000 for very large datasets). K-means runs on cuML when a usable GPU is present, sklearn otherwise, with restart counts scaled down for large nodes.

Each node stores:

| Field | Contents |
|-------|----------|
| `ids` | The node's item IDs, sorted **most-typical-first** (descending `mu . x`) — `ids[0]` is the region's representative |
| `children` | Child node names, stored **largest-first** so breadth-first traversal reaches big unexplored regions before small ones |
| `n` | Item count |
| `mu`, `rbar` | Mean direction and resultant length — the sufficient statistics of a von Mises–Fisher component, so reading the tree at any depth gives a multiresolution mixture model of the dataset |
| `t_quantiles` | A 21-point quantile grid of the node's own points' **leave-one-out** typicality scores, used to calibrate query p-values |

Node records are **immutable** once built. Labeled evidence lives in a separate per-atlas **overlay** — `n_pos` / `n_neg` counts keyed by node name (session state, not serialized), plus the labeled-ID set — so two atlases can share one node table by reference while keeping independent labels. `structural_clone()` exploits this: an atlas over the same id set (e.g. the labeling-progress per-step atlas mirroring the dataset context's) is cloned with the node table shared and a fresh overlay, skipping the hierarchical-k-means re-fit.

Evidence flows in from votes: every good/bad vote calls `label(id, good=...)`, which increments the class counter in the item's leaf and every ancestor; un-voting decrements; clearing votes or swapping detectors resets and replays (`resync_coverage_atlas_to_detector`, via `reset_labeled()`). A node is **covered** when `n_pos + n_neg > 0` (read through `atlas.n_pos(name)` / `atlas.n_neg(name)`).

### Diversity sampling (`next_sample`)

`GET|POST /api/coverage-atlas/next` returns the next item the autopilot should show. The walk is breadth-first from the root: the first node carrying **no evidence** is the next region to explore. Because siblings are stored largest-first, ties break toward the biggest unexplored region — best coverage gain per click.

Within the chosen node, the pick is a **surprise probe** when sort scores are supplied (the autopilot always supplies the current learned-sort scores and threshold):

- Node's median score ≥ threshold (**presumed good**) → return the **lowest**-scored element: the item most likely to be a hidden bad in a region the model calls good.
- Otherwise (**presumed bad**) → return the **highest**-scored element: the item most likely to be a hidden good.

The extremum probe is informative in both outcomes. If the probe *flips* (the greenest item of a presumed-red region is actually good), the user just found a hidden pocket the model was wrong about — maximum training value for one click. If it *doesn't* flip, the region's presumption has been stress-tested at its weakest point: nothing else in the node was more likely to surprise.

Two refinements:

- **Typicality tempering.** In nodes with a concentrated direction (`rbar ≥ 0.1`), the extremum is taken over the node's **typical half** (`ids` is typicality-sorted, so this is just the first half). An extreme score on an *atypical* item is disproportionately often a lone oddball — a corrupt file, a weird crop — whose flip says nothing about the region; a flip on a typical item is evidence of a real pocket. Degenerate nodes (the root) probe the whole node, since their typicality ordering is noise.
- **Regional median.** The median that decides the probe direction always spans the whole node, not the pool — the presumption being tested is about the region.

Without scores, the pick is the node's most typical element (`ids[0]`), a representative of the unexplored region.

The response's `coverage_level` — the number of consecutive covered nodes in breadth-first order — is the autopilot's **Span** indicator: it turns green at `autopilot_goal_diversity` (default 40) covered nodes, ending the diversity phase. `exhausted: true` means every node carries evidence.

### Typicality and domain shift

`CoverageAtlas.typicality_pvalues(matrix)` answers, per query vector: *what fraction of the data this atlas was built on looks less typical than this?* Small p-value = the atlas has essentially never seen anything like it.

How a query is scored:

1. Map the query into the atlas frame (subtract `center`, renormalize).
2. Route it down the tree, at each node descending into the cosine-nearest child.
3. At every **calibrated** node along the path — at least 20 points *and* `rbar ≥ 0.1` — compute the alignment `t = mu . x` and read a p-value off the node's stored quantile grid.
4. Average the p-values along the path.

Three details are meant to make the p-values honest rather than merely monotone. Each does real work, and together they still fall short of calibration — see [How good are these p-values?](#how-good-are-these-p-values) below:

- **Leave-one-out calibration.** Each node's quantile grid is built from scores of its own points against the mean direction of the *other* points (closed form on the sphere: `(R.x - 1) / ||R - x||`). Scoring a point against a mean it helped shape is optimistic, and without the correction fresh in-domain queries systematically read as atypical.
- **The `rbar` gate.** A node with no concentrated direction has a meaningless `mu` and pathological leave-one-out scores; the gate excludes it — notably the always-degenerate root. Sparse branches terminate shallow, which is the adaptive bandwidth: dense regions are judged at fine scale, sparse ones at coarse scale.
- **Path averaging.** A hard partition has boundary artifacts (a fresh in-domain query near a k-means cell edge looks atypical at leaf scale); averaging across scales smooths them the way a tree ensemble would, at zero extra build cost.

`domain_shift_report(atlas, matrix, alpha=0.05)` aggregates the p-values into a dataset-level verdict: the observed fraction below `alpha` (`frac_atypical` — roughly the shifted proportion), a binomial z-score for the excess against the *nominal* rate, the median p-value, and a headline `shifted` boolean (excess both statistically clear, z > 3, and practically large, ≥ 2×`alpha`).

#### How good are these p-values?

Not calibrated, and the gap was unmeasured for as long as the atlas existed. Measured on held-out in-domain data across five embedders ([the #3329 fit-quality study](experiments/2026-08-30-fit-quality-3329/REPORT.md), B1–B6):

- The values are **under-dispersed** — sd 0.250 against the 0.289 a uniform would give — and sit **KS 0.103** from uniform. An in-domain query does not have an `alpha` chance of scoring below `alpha`, so `expected_atypical` is nominal, not measured.
- **No combiner is calibrated.** Median 0.071, mean (shipped) 0.103, deepest-only 0.132, Fisher 0.186, min 0.329. Dropping the path averaging — the obvious suspect — makes it *worse*, not better.
- **The shipped combiner looks right at exactly the number a spot check reads** (4.3 % flagged against a nominal 5 %) while being second-worst in overall shape. That is why this went unnoticed.
- **Patch embedders are where it breaks.** Their spaces are the least concentrated (node `rbar` 0.61 against 0.66–0.70), and `dinov3_patch` puts 13.7 % of its own in-domain items below 0.05 — the verdict fires on 80 % of the atlas's own held-out data against 93 % cross-corpus, a separation of 0.13. The domain-shift route therefore **refuses patch embedders outright** rather than answering with noise. Refitting `alpha` per embedder was priced and does not rescue it; the fix, if it is ever wanted there, is the per-node vMF model.

The **ranking** is unaffected, which is what the autopilot's diversity sampling actually uses (`next_sample` reads each node's score-sorted `ids` and gates on `rbar`, never a p-value). The defect is confined to the calibrated *value* and its one consumer, the domain-shift endpoint.

### Tutorial: how a diversity session works

What actually happens when the autopilot enters its "Explore Diversity" phase, click by click:

1. **The atlas already exists.** It was built (or restored from the pickle cache) when the dataset loaded, and every vote cast during the earlier good/bad/refine phases has already been counted into its evidence channels.
2. **The frontend asks for a sample.** `POST /api/coverage-atlas/next` with the current learned-sort scores and decision threshold in the body.
3. **The atlas walks breadth-first** to the first evidence-free node — say a 900-item region of the collection no vote has ever touched — preferring the largest such region among siblings.
4. **It probes for a surprise.** Suppose the node's median score is 0.81 against a threshold of 0.5: the model presumes the whole region is good. The atlas returns the *lowest*-scored item from the region's typical half — the most plausible hidden bad that is still representative of the region.
5. **The user votes.** The vote lands in the detector's labels *and* increments `n_neg` (or `n_pos`) in the item's leaf and all its ancestors — the region is now covered, and the next `next_sample` call moves on to the next uncovered region.
6. **The Span indicator advances.** Each labeling-status poll reads `span_info()`; when 40 consecutive breadth-first nodes carry evidence, Span turns green and the autopilot declares the collection covered.

Either outcome of step 5 helped: a flip hands the head a training example from a region it was confidently wrong about (the next retrain bends the boundary there); a non-flip certifies the region at its weakest point for one click.

### Tutorial: checking for domain shift before reusing a detector

You trained a detector on dataset A and want to run it on dataset B. Should you trust it? Ask the atlas:

```
# Both datasets loaded; A's coverage atlas built (automatic ≤ 50k items).
# The X-Dataset-Id header names the ACTIVE dataset (B); the URL names the
# REFERENCE dataset (A, the training domain).
GET /api/datasets/registry/<dataset_A_id>/domain-shift
X-Dataset-Id: <dataset_B_id>
```

```json
{
  "reference_dataset_id": "…",
  "n_items": 40000,
  "alpha": 0.05,
  "frac_atypical": 0.31,
  "expected_atypical": 0.05,
  "z_score": 24.1,
  "median_pvalue": 0.18,
  "shifted": true
}
```

Reading this: 31% of dataset B sits in regions of embedding space where dataset A had essentially no mass (against the 5% that chance would produce), so the verdict is `shifted` — the detector will be *extrapolating* on a third of B, and its scores there are unfalsified guesswork. Verify by hand before trusting it. A same-domain report instead shows `frac_atypical` near `alpha`, a median p-value near 0.5, and `shifted: false`.

The endpoint refuses (HTTP 400) when the two datasets use different embedders — typicality across embedding spaces would be confident nonsense — or when the reference has no atlas yet (build it via the endpoint above).

The same machinery is available in the library tier:

```python
import numpy as np
from vtscore.coverage.atlas import CoverageAtlas, domain_shift_report

atlas = CoverageAtlas({mid: vec for mid, vec in train_vectors.items()}, k=3)

atlas.label(42, good=True)          # count labeled evidence
print(atlas.next_sample(scores, threshold))  # next diversity probe

pvals = atlas.typicality_pvalues(np.stack(list(other_vectors.values())))
print(domain_shift_report(atlas, np.stack(list(other_vectors.values()))))
```

### Costs

Build is the same order as the embedding-matrix work a dataset load already does — seconds on CPU for 50k items, with progress reported to the load bar. Queries are microseconds per item (`O(depth × k × dim)`); a full domain-shift sweep over a 40k-item dataset is well under a second. Nothing needs a GPU.

## Key Files

- `vtscore/training/mlp.py`: `build_model`, `train_model`, `build_model_from_weights`
- `vtscore/coverage/atlas.py`: `CoverageAtlas`, `domain_shift_report`
- `vtscore/state/coverage.py`: atlas build/restore/resync helpers, vote wiring
- `vtscore/training/thresholds/`: `calculate_cross_calibration_threshold`, `fold_anchored_gmm_threshold` (`anchored.py`), `calculate_safe_threshold` (`blend.py`), `calculate_gmm_threshold` (`gmm.py`), `conformal_threshold` (`conformal.py`)
- `vtscore/detectors/training.py`: `train_and_score`, `train_and_threshold`, origin-based detector training
- `vtscore/detectors/labeling_progress.py`: Per-step cache of the detectors the app trained, and their stability analysis
- `vtscore/embedding/loader.py`: Model initialization and thread configuration
- `vtscore/eval/voting_iterations.py`: Voting simulation evaluation
- `vtscore/config/runtime.py`: `TRAIN_EPOCHS`; `vtscore/config/models.py`: the model IDs
