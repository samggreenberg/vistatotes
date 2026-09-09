"""Knob semantics: the constants and pure functions the threshold rules are keyed on.

Nothing here fits a model or touches a score distribution.  It is the layer
that says what an *inclusion* value means (its cost weights, the acquisition
offset), what a Train/Calibrate split defaults to, and what the sentinels are.
Every other module in :mod:`vtscore.training.thresholds` reads from here; this
one reads from nothing.
"""

from __future__ import annotations

# Sentinel threshold meaning "predict nothing as Good". Sigmoid scores are
# in [0, 1], so any value > 1.0 makes every ``score >= threshold`` check
# evaluate to False. Kept finite (vs. ``float("inf")``) so it cannot poison
# downstream blends - ``0.0 * inf`` evaluates to NaN, which would then be
# stored on ``DetectorContext.threshold`` and break every comparison.
NO_GOOD_THRESHOLD = 2.0


# Bounds of the Inclusion knob.  Every threshold rule keyed on inclusion is
# defined over this closed range, and every sweep of it (the UI slider's stops,
# the Find Stats chart) runs over exactly these values.
INCLUSION_MIN = -10
INCLUSION_MAX = 10


#: The shipped Train/Calibrate split of each calibration fold, per the **space
#: the detector learns in** (issue #3287 measured them separately; see
#: ``docs/experiments/2026-08-27-calibration-fraction-3287/REPORT.md``).  The value is the
#: **Calibrate** share, so ``0.3`` means 70% Train / 30% Calibrate.
#:
#: * ``single_vector`` - one embedding vector per media.  Spending more votes
#:   on fitting each fold's model and fewer on reading its threshold is worth
#:   −0.012 to −0.013 ± 0.003 in cost on both single-vector embedders
#:   measured, winning in every vote band; the gain is largest when votes are
#:   scarce and decays toward 150 clicks.
#: * ``patch`` - a patch-grid embedder (whatever style it currently votes in).
#:   Nothing measured beats the incumbent 0.5 here, and 0.3 is +0.015 ± 0.005
#:   *worse* on ``dinov3_patch/whole_image`` - which is why the key is the
#:   embedder's capability rather than the voting mode: the same row-wise
#:   calibrator wants opposite splits on ``siglip/whole`` vs ``dinov3/whole``,
#:   while both ``dinov3`` styles agree on 0.5.
PRODUCTION_SPLIT_BY_SPACE: dict[str, float] = {
    "single_vector": 0.3,
    "patch": 0.5,
}

#: Fallback when the space is unknown.  0.5 is the incumbent and the
#: never-harmful choice: it is not significantly worse than any arm measured,
#: on any geometry.
PRODUCTION_SPLIT = 0.5


def production_split_for(*, patch_space: bool | None) -> float:
    """The shipped ``calibration_fraction`` for the space a detector learns in.

    *patch_space* says whether the detector's embedder produces a patch grid
    (its capability, not what it is doing in the current configuration -
    ``dinov3_patch`` wants 0.5 in both its styles, including the boxless
    fallback that emits no patches at all).  ``None`` means "unknown", which
    takes :data:`PRODUCTION_SPLIT` rather than guessing - the same three-state
    contract as :func:`vtscore.training.blend_schedules.production_schedule_for`.

    An explicit user setting always wins over this table; callers resolve that
    precedence via
    :func:`vtscore.detectors.training.resolve_calibration_fraction`.
    """
    if patch_space is None:
        return PRODUCTION_SPLIT
    return PRODUCTION_SPLIT_BY_SPACE["patch" if patch_space else "single_vector"]


def classify_threshold_provenance(fallback: float | None) -> str:
    """Name the code path a trained threshold came from, from its *fallback*.

    :func:`compute_fold_orderings` returns a ``fallback`` that fully discriminates
    which branch produced the threshold: ``None`` means the conformal quantile
    rule ran on real fold orderings; :data:`NO_GOOD_THRESHOLD` (2.0) means the
    "no valid Train/Calibrate split" sentinel; ``0.5`` means a too-few-labels
    early return.  Used by the calibration study (issue #2781) to attribute the
    runaway-threshold bug; the safe-threshold GMM blend is a separate caller and
    is tagged ``"gmm_blend"`` at that site, not here.
    """
    if fallback is None:
        return "conformal"
    if fallback == NO_GOOD_THRESHOLD:
        return "no_good_sentinel"
    if fallback == 0.5:
        return "too_few_default"
    return "unknown"


def inclusion_cost_weights(inclusion_value: float) -> tuple[float, float]:
    """``(fpr_weight, fnr_weight)`` - the rate loss the Inclusion knob names.

    Inclusion is defined as a trade-off between the two error *rates*:
    ``cost = fpr_weight * FPR + fnr_weight * FNR``.  Each ``+1`` step doubles
    the price of a miss (matching :func:`conformal_threshold`'s halving
    false-negative budget) and each ``-1`` step doubles the price of a false
    alarm, so the knob means the same thing to every rule that reads it - the
    conformal quantile, the rate-optimal GMM cut
    (:meth:`GmmFit1D.rate_crossing`), and the eval harness's scoring.

    This is the single definition; :mod:`vtscore.eval.calibration_metrics` and
    :mod:`vtscore.eval.voting_iterations` delegate here so a measured arm and
    the shipped path can never disagree about what an inclusion value costs.

    **What a value MEANS, in one line: inclusion is a log2 likelihood-ratio
    threshold.**  Because the loss is a weighted sum of *rates*, each normalised
    by its own class, the prevalence divides out (see
    :meth:`GmmFit1D.rate_crossing`, which puts the prior-odds factor back into
    ``lam`` precisely so the cut does not carry it).  Minimising
    ``w_fp*FPR + w_fn*FNR`` therefore admits exactly the items whose class-
    conditional likelihood ratio clears ``w_fp / w_fn``::

        include x  <=>  f_pos(x) / f_neg(x)  >  2**-k

    So ``k = 0`` is the neutral-evidence point (LR > 1: admit whatever the Good
    class explains better than the Bad class), and **each step of the knob is one
    bit of evidence** - Good's *weight of evidence*, in base 2.  ``k = -4`` asks
    for 16:1 evidence, ``k = +2`` accepts 4:1 against.  That is what makes the
    knob portable across datasets and what makes the acquisition *offset* the
    right parameterisation: a constant shift in evidence-bits is prior-free,
    while the rank position it lands on is not.

    *inclusion_value* is a float: fractional steps are well defined (a half step
    is a factor of sqrt(2) in the evidence ratio) and issue #3319 sweeps them.
    The UI slider still stops at integers.
    """
    if inclusion_value >= 0:
        return 1.0, 2.0**inclusion_value
    return 2.0 ** (-inclusion_value), 1.0


#: How far *below* the reporting inclusion the **acquisition** cut sits.
#:
#: The threshold does two unrelated jobs.  Reporting is the decision line the
#: user sees and every metric is scored at.  Acquisition is what Autopilot's
#: ``hard`` and ``new`` picks consume - and those read the threshold as a **rank
#: position** in the descending ranking, not as a decision boundary, so they want
#: the opposite thing from it.
#:
#: The direction is therefore the opposite of the intuition from the cost
#: weights: a *negative* offset prices false alarms higher, *raises* the cut,
#: moves it *up* the ranking, and so returns *more* positives.
#:
#: ``-4`` is the value that passes the pre-registered ship rule in **every**
#: environment measured on clean labels, and this constant is deliberately
#: **not** gated by voting mode.  The history is worth keeping, because the
#: value moved three times before it settled:
#:
#: * ``coco_val x siglip2`` (binary, PR #2876) found an interior optimum at
#:   ``-3``: positives per 100 votes 4 -> 18, final cost 0.137 -> 0.129 (95% CI
#:   [-0.025, -0.005]), average precision 0.696 -> 0.817.  #2878 shipped it.
#: * ``visual_genome_m x siglip`` (binary, PR #2891) **rejected** ``-3``: cost CI
#:   [+0.003, +0.022] against a +0.01 tolerance.  #2909 cut the value to ``-1``.
#: * ``visual_genome_m x dinov3_patch`` (region, PR #2909) was **voided** by
#:   #2943 - see ``REPORT_REGION_VOTING.md``'s banner.
#: * ``vg_scale_any`` (PR #3318, #2877 on the pile) measured three environments
#:   at once, on **verified labels at 7.1% prevalence in every cell**, and
#:   restored ``-3`` - noting ``-4`` was at least as good on every endpoint but
#:   sat at the edge of the grid with the trend unbroken.
#: * **#3319 (PR #3454) extended the grid and shipped ``-4``**: 12 arms x 192
#:   paired cells past ``-4`` and at half-step resolution, plus a 400-click wave
#:   and a region cross-check.  4032 cells, 0 failures.
#:
#: **What this knob is worth is SPEED, and an endpoint cannot say so.**  Against
#: no offset at all, the shipped cut reaches the answer ``k = 0`` ends its
#: session with in **half the clicks** over a 100-click session (23.5 vs 47.5
#: median) and **3.2x fewer** over a 400-click one (65.5 vs 210.5) - and the
#: advantage *compounds* with session length: paired, running with no offset
#: costs +17.1 clicks [+12.0, +21.8] at 100 and +101.1 [+82.7, +119.5] at 400.
#: Every report on this constant before #3319 read it through ``final_cost``
#: alone, which understated it.  Report a trajectory knob by its trajectory.
#:
#: **Why ``-4`` and not ``-3``, and how weak that case is.**  The decision
#: endpoint is a **plateau**, not a peak: final cost, area under the cost curve
#: and clicks-to-target are all flat from ``-2`` to ``-5``, on both horizons, and
#: ``-3`` vs ``-4`` is a null on all three (cost -0.0015 [-0.0081, +0.0050];
#: AUC -0.0044 [-0.0108, +0.0021]; clicks -1.2 [-4.8, +2.4]).  So the case for
#: ``-4`` is **labelling efficiency alone** - hard-pick precision 24% -> 35%, and
#: ~28 matches surfaced per 100 clicks instead of ~20 - and it is a product
#: judgement about how much matches-surfaced-during-a-session are worth, not a
#: quality or speed result.  ``-3`` was a defensible choice and remains one.
#:
#: **What bounds the value from below.**  Positives and AP never saturate across
#: the grid (7 -> 64 positives, AP 0.568 -> 0.722 out to ``-8``), so nothing in
#: the *mechanism* stops the sweep.  The **guardrail** does: deep-spike incidence
#: is 0-0.5% out to ``-5``, then 1.0% at ``-6`` and 2.6% at ``-8``, and ``-6``
#: also regresses cost (+0.0085 [+0.0008, +0.0158] against ``-3``).  Threshold
#: stability is the binding criterion, exactly as #3318 found.
#:
#: **Region voting is what rules out going deeper, and ``-4``'s margin there is
#: thin.**  On ``siglip+dinov3_patch x max_patch``, against ``-3``: ``-4`` is
#: +0.0031 [-0.0025, **+0.0091**] and passes; ``-5`` is +0.0064 [+0.0007,
#: **+0.0123**] and fails the +0.010 bar despite being free on the shipped arm.
#: #3318 measured this same ``-4`` contrast at +0.006 [+0.001, +0.013] and
#: rejected it; #3319 measures +0.0031 and passes.  The CIs overlap heavily, so
#: those are *consistent* measurements straddling the bar - not a reversal, and
#: not independent confirmation either.  **If this value is ever revisited, that
#: is the number to re-measure first.**
#:
#: **Why #2891's rejection does not block it.**  That environment is
#: ``visual_genome_m``, whose free-text labels have measured recall **0.76** over
#: these classes (``scripts/experiments/pile/coco_anchor.py``).  Roughly a
#: quarter of true positives are labelled negative there, so an arm that finds
#: *more* true positives is charged for them as false alarms - a bias against
#: precisely the aggressive arms under test.  ``vg_scale_any`` exists to remove
#: it (COCO-exhaustive labels plus a human review pass).  The pattern fits: every
#: clean-label environment adopts the aggressive arm; the one noisy one rejects
#: it.  This is a well-supported explanation, not a proven cause - #2877's cells
#: are archived and the counterfactual was not re-run.
#:
#: **The one environment that wants ``-1`` is not a mode.**  A patch embedder
#: with no box supervision (``siglip+dinov3_patch x whole_image`` - a DINOv3
#: detector whose users vote whole images instead of dragging boxes) rejects
#: ``-2``/``-3``/``-4`` on deep-spike incidence, 4.5% -> 22.7/28.8/35.6%
#: (p<1e-4), while their *cost* deltas are negative.  That arm is not reachable
#: today (DINOv3 does not ship), and it is **not** what a voting-mode gate would
#: select: the two *binary* environments disagree with each other more than the
#: modes do.  If a patch embedder ever ships, gate on the **scoring geometry**
#: that actually resolves (a patch embedder falling back to ``whole_image``), not
#: on how the user voted.
#:
#: **The mechanism is threshold stability, not cost.**  Measured within one
#: embedder on the same 264 cells, region voting takes oracle cost 0.382 -> 0.218
#: and AP 0.517 -> 0.762, and the 8x spike rise that rejects the aggressive arm
#: under whole-image scoring does not happen at all (2.7% -> 1.5%, p=0.51).
#: Aggressive acquisition destabilises the cut when the ranking is poorly
#: separated; it is safe when the ranking separates well.
#:
#: **The knob under-delivers its own steps, by an environment-dependent amount.**
#: Measured on the pick log, ``k = 0`` returns 4.7% hard-pick precision - *below*
#: the 7.1% base rate, which is the cleanest statement of why this offset exists
#: at all - and precision crosses 50% at ``k ~ -5.66`` where the evidence
#: semantics above predict -3.71.  Regressing log2-odds of pick precision on
#: ``k`` gives intercept -3.79 / slope **-0.65** on binary and -3.32 / **-0.78**
#: under region, against a calibrated -3.71 / -1.00: the *origin* is right and
#: the *steps* are short, and the shortfall shrinks when the ranking separates.
#: It is not a constant gain (the residuals curve, and ``k = +2`` does not sit on
#: the fitted line) - "approximately calibrated near 0, increasingly compressed
#: with depth" is what the data supports.  This reaches the **user-facing
#: Inclusion slider**, which drives the same :meth:`FoldAnchoredCut.threshold_at`.
#: It also explains the history above: the arms were never sweeping "aggression",
#: they were sweeping *nominal* bits against an environment-dependent debt.
#:
#: The step-shortfall above is tracked in **#3546**; the deep-regime question
#: below, and the pile change that would settle it, in **#3547**.
#:
#: **The deep regime does not flip the sign, but it does wake the guardrail.**
#: At 400 clicks the offset is worth *more*, not less (cost -0.033 vs prod,
#: +90 positives, and the speed gain above) - but deep-spike incidence, 0.0% for
#: every arm at 100 clicks, goes 0.5% -> 5.7% (p=0.006) at ``-3`` and 2.1%
#: (p=0.38) at ``-4``.  Non-monotone, so a hazard to watch rather than evidence
#: about which arm is safer.  Long sessions remain the least-measured regime.
#:
#: See ``docs/experiments/2026-08-07-acquisition-inclusion/REPORT_3319.md`` (the
#: study this value rests on), ``REPORT_PILE_2877.md`` (the three environments
#: before it), ``REPORT.md`` (COCO), and ``REPORT_SECOND_ENVIRONMENT.md`` /
#: ``REPORT_REGION_VOTING.md`` for the two superseded readings.
ACQUISITION_INCLUSION_OFFSET = -4


def acquisition_inclusion(inclusion_value: float, offset: float = ACQUISITION_INCLUSION_OFFSET) -> float:
    """The inclusion the **selector's** cut is taken at, given the reporting one.

    One definition, shared by the app and the eval harness, so a measured arm
    and the shipped path cannot disagree about where acquisition samples - the
    same discipline :func:`inclusion_cost_weights` follows.  *offset* exists for
    the harness's arms; production always takes the default.

    An *offset*, not an absolute value.  The runs that measured ``-3`` held
    reporting at inclusion 0, where the two readings coincide; away from 0 only
    the offset preserves what was measured, because the mechanism is the *gap*
    between where the line is drawn and where sampling happens.  Reading ``-3``
    absolutely would collapse the gap to nothing at reporting inclusion -3 and
    invert it below that - the direction the ``acq_p2`` arm falsified.

    *offset* may be fractional (issue #3319).  One step is one bit of evidence
    (:func:`inclusion_cost_weights`), so a half step is a real, realisable
    operating point of factor sqrt(2), not an interpolation between two settings.

    Deliberately unclamped.  The reporting inclusion is clamped to ``[-10, 10]``
    at the API edge, so this can reach -13; the cost weights are exponential but
    finite there, and :meth:`FoldAnchoredCut.threshold_at` clamps the quantile it
    realizes to ``[0, 1]`` anyway.  Clamping here would instead silently switch
    the mechanism off at the bottom of the slider, which is the failure mode that
    is hard to notice.
    """
    return inclusion_value + offset
