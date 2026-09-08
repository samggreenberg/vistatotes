"""Evaluate learned-sort cost over simulated voting iterations.

For each combination of seed *s*, dataset *d*, and target category *c*:

1. Load the dataset and split medias into **D_sim** (simulation) and
   **D_test** (held-out) using *s* to control the random split.
2. Assign ground-truth labels based on *c*: medias whose ``"category"``
   matches *c* are positive (``good``), others are negative (``bad``).
3. Vote on D_sim one item at a time, choosing *which* item to vote on next by
   reproducing the app's **Autopilot** flow (order seeded by *s*).
4. At each step *t* (once at least one good **and** one bad vote exist),
   train a model on votes so far, find a threshold, score D_test, and record
   the inclusion-weighted cost (``fpr_weight * FPR + fnr_weight * FNR``).

Which item the simulated user votes on at each step is chosen by the
``autopilot`` vote-order strategy (see :mod:`vtscore.eval.al_strategies`): seed
from text sort (or a few random known-good examples), then the standard
Good / Bad / Hard / New phases.  This is the only strategy the eval runs — the
point is to measure how the tool itself would function, not to compare
acquisition heuristics.

The result is a :class:`pandas.DataFrame` with columns
``seed, dataset, category, strategy, t, n_good, n_bad, cost, fpr, fnr``.

``n_good``/``n_bad`` are the number of good/bad votes the model was trained
on for that row. The very first scored step has only one of each, so its
``cost``/``fpr``/``fnr`` are extremely noisy; these counts let downstream
analysis filter or weight rows by how many votes actually informed them
rather than treating a 1-vs-1 model as if it were as reliable as a 50-vs-50
one.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

    from vtscore.training.thresholds import FoldAnchoredCut

from vtscore.embedding.media_vectors import media_embedding
from vtscore.eval.al_strategies import ALContext, select_next
from vtscore.eval.autopilot_flow import SMART_WINDOW, AutopilotFlow, app_has_detector
from vtscore.eval.startup_schedule import StartupState, parse_startup_schedule, round_cut
from vtscore.eval.arms_anchored import (
    _ANCHORED_FOLD_COMBINES,
    _ANCHORED_RULES,
    _ANCHORED_WEIGHTS,
    _anchored_variant_rows,
)
from vtscore.eval.arms_fit_quality import _fit_quality_rows
from vtscore.eval.arms_fold_count import _fold_count_variant_rows, parse_fold_count_schedule
from vtscore.eval.arms_inclusion import _cut_inclusion_rows, _inclusion_sweep_rows
from vtscore.eval.arms_safe_gmm import _safe_gmm_variant_rows
from vtscore.eval.arms_schedule import _schedule_variant_rows
from vtscore.eval.row_metrics import operating_metrics, round6
from vtscore.eval.step_model import (
    HEADS,
    PRODUCTION_HEAD,
    StepModel,
    score_sim_set_with_model,
)
from vtscore.eval.step_trainers import (
    _build_eval_atlas,
    _labelset_error_costs,
    _score_pool,
    _train_and_calibrate,
)
from vtscore.eval.labels import evaluable_pool, media_is_positive
from vtscore.eval.score_dumps import maybe_dump_predictions
from vtscore.eval.voting_columns import (
    FIT_QUALITY_STRIDE_DEFAULT,
    VOTING_COLUMNS,
)
from vtscore.training.blend_schedules import BlendContext
from vtscore.training.thresholds import (
    ACQUISITION_INCLUSION_OFFSET,
    apply_vote_exclusion,
    FOLD_ANCHOR_QTILT_STEP,
    NO_GOOD_THRESHOLD,
    acquisition_inclusion,
    calculate_safe_threshold,
    threshold_from_fold_orderings,
)


#: The detection geometry a live detector uses on a patch dataset - the style an
#: unspecified ``style=`` resolves to below.  Named rather than inlined so that
#: "is this run measuring the shipped geometry?" is a question something can
#: *ask*: `scripts/experiments/preflight.sh` compares a study's configured
#: styles against it, the way it compares a study's head against
#: :data:`PRODUCTION_HEAD`.  The HAC hybrids in
#: :mod:`vtscore.eval.patch_styles` are experiment-only arms; #2886 removed the
#: region tree from ingest.
PRODUCTION_PATCH_STYLE: str = "max_patch"

#: The one detection style the #3322 skyline is defined for in v1 - spelled here
#: rather than imported for the same reason :data:`_SAFE_GMM_VARIANTS` spells its
#: rule names: this module is deliberately import-light at module scope.
#: ``test_skyline_arm`` pins it against :class:`~vtscore.eval.patch_styles.WholeImageStyle`.
_WHOLE_IMAGE_STYLE: str = "whole_image"


#: The **supervised-skyline** arms (issue #3322), tagged in the ``gmm_variant``
#: column exactly like every other variant family, and emitted **once per run**
#: rather than once per step: a skyline is vote-independent, so its row is a
#: constant of the ``(cell, seed)`` and repeating it 150 times would only invite
#: a reader to average it against itself.
#:
#: * ``skyline_train_full`` - the **primary** arm and the one the decomposition
#:   is defined against.  The same head, trained through the same trainer, on the
#:   **entire simulation split with full ground-truth labels**, evaluated on the
#:   untouched test split.  The standard supervised-skyline arm from the
#:   active-learning literature: same hypothesis class, same features, full
#:   supervision, disjoint eval, so its cost is "how learnable is this class in
#:   this embedding with this head" and nothing else.
#: * ``skyline_test_xfit`` - the optional **bracket** partner, cross-fitted over
#:   the test split (train on K-1 folds, score the held-out one).  Never a naive
#:   train-on-test fit: a ~769-parameter linear head on a test set of comparable
#:   size can shatter near-arbitrary labelings, so a naive fit would report
#:   ``d / n_test`` under the name "learnability" and hand back near-zero cost on
#:   a class nothing can learn.  Cross-fitting is the SVM analogue of
#:   :func:`~vtscore.eval.transfer_rules.honest_test_oracle`, which does the same
#:   thing one level down for the *cut*.
SKYLINE_TRAIN_FULL: str = "skyline_train_full"
SKYLINE_TEST_XFIT: str = "skyline_test_xfit"
SKYLINE_ARMS: tuple[str, ...] = (SKYLINE_TRAIN_FULL, SKYLINE_TEST_XFIT)

#: ``threshold_provenance`` on a skyline row.  A skyline is a statement about a
#: **ranking**, so it is deliberately *not* routed through a calibrated cut - that
#: would re-mix in exactly the term ``regret`` already isolates.  Its threshold is
#: the test oracle's, which makes ``cost == oracle_cost`` and ``regret == 0`` on
#: the row **by construction**: read a skyline row's ``oracle_cost``, never its
#: ``regret``.
SKYLINE_PROVENANCE: str = "skyline_test_oracle"

#: RNG salt for the cross-fitted arm's test-set partition.  Combined with the
#: run's own ``seed`` so the folds are reproducible per cell, and kept off the
#: trajectory's ``RandomState`` so turning the arm on cannot move a single vote.
_SKYLINE_SEED: int = 3322


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


#: Minimum positives an arm must retain after prevalence downsampling.  Below
#: this the held-out test set has too few positives for a stable FNR estimate,
#: so the arm is skipped rather than reported with a noisy denominator.
_MIN_PREVALENCE_POSITIVES = 15


def _prevalence(clips_dict: dict[int, dict[str, Any]], target_category: str) -> float:
    """Fraction of *clips_dict* that is positive for *target_category*."""
    if not clips_dict:
        return 0.0
    n_pos = sum(1 for m in clips_dict.values() if media_is_positive(m, target_category))
    return n_pos / len(clips_dict)


def _downsample_to_prevalence(
    clips_dict: dict[int, dict[str, Any]],
    target_category: str,
    target_prevalence: float,
    rng: np.random.RandomState,
) -> Optional[dict[int, dict[str, Any]]]:
    """Return a copy of *clips_dict* with positives thinned to ~*target_prevalence*.

    All negatives are kept; positives are randomly downsampled (via *rng*, so the
    arm is deterministic in the eval seed) to the largest count ``k`` with
    ``k / (k + n_neg) <= target_prevalence``.  Returns ``None`` when that leaves
    fewer than :data:`_MIN_PREVALENCE_POSITIVES` positives (the arm is then
    skipped).  Multi-label datasets are handled through ``media_is_positive``.
    """
    import numpy as np  # noqa: PLC0415

    pos_ids = [cid for cid in clips_dict if media_is_positive(clips_dict[cid], target_category)]
    neg_ids = [cid for cid in clips_dict if not media_is_positive(clips_dict[cid], target_category)]
    n_neg = len(neg_ids)
    if n_neg == 0:
        return None
    keep_k = int(target_prevalence * n_neg / (1.0 - target_prevalence))
    keep_k = min(keep_k, len(pos_ids))
    if keep_k < _MIN_PREVALENCE_POSITIVES:
        return None
    chosen = rng.choice(np.array(pos_ids, dtype=np.int64), size=keep_k, replace=False)
    keep = set(int(c) for c in chosen) | set(neg_ids)
    return {cid: clips_dict[cid] for cid in clips_dict if cid in keep}


def _split_media_ids(
    clips_dict: dict[int, dict[str, Any]],
    sim_fraction: float,
    rng: np.random.RandomState,
) -> tuple[list[int], list[int]]:
    """Randomly partition media IDs into simulation and test sets."""
    all_ids = sorted(clips_dict.keys())
    shuffled = rng.permutation(all_ids).tolist()
    n_sim = max(1, int(len(shuffled) * sim_fraction))
    return shuffled[:n_sim], shuffled[n_sim:]


def _pool_percentile(pool_scores: dict[int, float], threshold: float) -> float:
    """Fraction of the *unlabelled pool* scoring below *threshold*.

    The selector's ``hard`` pick works in rank space over the pool, so this - not
    the threshold's value, and not its percentile in the held-out test scores -
    is the number that says where the next item comes from.  Returns NaN on an
    empty pool rather than a misleading 0.0.
    """
    import numpy as np  # noqa: PLC0415

    if not pool_scores:
        return float("nan")
    arr = np.asarray(list(pool_scores.values()), dtype=np.float64)
    return round(float((arr < threshold).mean()), 6)


def _sorted_percentile(descending: list[float], value: float) -> float:
    """Where *value* cuts a **descending** score list, as a fraction from the top.

    ``0`` = above every score, ``1`` = below every score.  Used for the pick
    log's ``startup_cut_percentile``, so a round's cut is reported as the
    sampling *position* it actually is rather than as a bare similarity whose
    scale differs per category.  Infinite cuts (the ``top`` round) read 0.
    """
    import numpy as np  # noqa: PLC0415

    if not descending:
        return float("nan")
    idx = int(np.searchsorted(-np.asarray(descending, dtype=np.float64), -value, side="left"))
    return round6(idx / len(descending))


def _blend_xcal_input(threshold: float, details: dict[str, Any]) -> float:
    """The x-cal side of the schedule blend, with the app's sentinel substitution.

    When the fold computation could not calibrate at all it returns a *sentinel*
    rather than a cut - ``0.5`` on the too-few-labels / fewer-than-two-per-class
    paths, :data:`~vtscore.training.thresholds.NO_GOOD_THRESHOLD` when the split
    itself is degenerate (see
    :func:`~vtscore.training.thresholds.compute_fold_orderings`).  Production
    does **not** blend whichever sentinel came back: ``_fused_threshold`` feeds
    the blend ``NO_GOOD_THRESHOLD`` ("we never computed a cut, so admit
    nothing") whenever ``folds.fallback is not None``, regardless of the
    sentinel's value.  This applies that same substitution, so the harness's
    shipped-threshold arm blends what the app blends.

    It matters exactly in the cold start: the production schedules ramp from
    ``lo=6`` labels, so a step past that with one class still under two votes -
    the rare-class starvation the autopilot flow reaches whenever the Bad phase
    keeps surfacing positives - carries real weight on the x-cal side, and
    ``2.0`` vs ``0.5`` moves both the recorded operating point and the
    acquisition cut.

    *details* carries ``fold_fallback`` on every torch path (``None`` when the
    folds are real).  The SVM arms carry no fold fallback at all - their
    threshold comes from the trainer-agnostic port, which has no production
    counterpart to mirror - so they blend their own returned value unchanged.
    """
    return NO_GOOD_THRESHOLD if details.get("fold_fallback") is not None else threshold


def _safe_threshold_for_step(
    threshold: float,
    step: StepModel,
    details: dict[str, Any],
    region_aware: bool,
    sim_clips: dict[int, dict[str, Any]] | None,
    X_all_clips: Any,
    ctx: "BlendContext",
    sim_ids: list[int],
    inclusion: int,
    style_obj: Any = None,
    schedule: str | None = None,
    voted_ids: "set[int] | None" = None,
    exclusion_min_remainder: float | None = None,
) -> tuple[float, list[float], list[int], list[Any], str, "FoldAnchoredCut | None"]:
    """The harness's **shipped** safe threshold - the same rule the app applies.

    Scores the simulation set (the harness's haystack) with the final model and
    with each calibration fold model, then cuts via
    :func:`~vtscore.training.thresholds.fold_anchored_gmm_threshold` at the
    production defaults (the ``FOLD_ANCHOR_*`` constants).  This is the
    estimator :func:`vtscore.detectors.training._safe_threshold` ships, called
    with the same arguments, so the harness's baseline arm cannot drift from
    the app's behaviour - the paired ``*_variant`` rows are where deliberate
    deviations live.

    Falls back to the schedule blend
    (:func:`~vtscore.training.thresholds.calculate_safe_threshold`) exactly
    where production does: no usable calibration folds.  The blend's x-cal side
    carries production's sentinel substitution (see :func:`_blend_xcal_input`) -
    a step whose folds fell back blends ``NO_GOOD_THRESHOLD``, not whichever
    sentinel the fold rule returned.  The SVM arms always land on the blend -
    their fold models are standalone sklearn estimators rather than the head
    the app trains, so there is no production path for them to match.

    Returns ``(threshold, sim_scores, sim_ids, fold_haystacks, provenance, cut)``.
    The fitted :class:`~vtscore.training.thresholds.FoldAnchoredCut` rides along
    (``None`` on the blend fallback) so a caller can re-cut the *same* fit at
    another inclusion without refitting - which is what the acquisition cut does
    (``acq_inclusion_offset``; see ``docs/ML.md``, threshold calibration).
    The sim scores ride along so the #2799 / #2836 / #2852 variant rows can
    re-cut the same distribution without a second scoring pass, their media ids
    with them so a variant can attach each score's true label without assuming
    the scorer preserved any ordering, and the per-fold haystack score arrays
    so the fold-anchored variant grid re-fits without re-scoring.

    *voted_ids* drives the app's #3308 exclusion, and drives it through the app's
    own :func:`~vtscore.training.thresholds.apply_vote_exclusion` /
    :func:`~vtscore.training.thresholds.drop_voted` rather than a copy of them:
    the voted items are dropped from every haystack the fold-anchored estimator
    fits on - the per-fold arrays (which are also what the returned
    ``fold_haystacks`` carry, so the variant grids inherit the same population
    convention) and the final model's realization sample - unless the remainder
    falls under the floor, in which case the whole step keeps its full
    haystacks.  The *returned* sim scores stay complete: they feed evaluation
    and acquisition, not the fit.

    *exclusion_min_remainder* is the **#3312 arm knob** and the one thing here
    that is allowed to differ from the app: ``None`` (every production caller,
    and the harness's own default arm) resolves to the shipped floor, ``0``
    excludes unconditionally, and ``math.inf`` switches the exclusion off
    entirely - the pre-#3308 baseline.  Because the resolution happens inside
    :func:`~vtscore.training.thresholds.resolve_exclusion_floor`, the default
    arm cannot drift from the app even though the knob exists.
    """
    import numpy as np  # noqa: PLC0415

    from vtscore.training.thresholds import drop_voted, fit_fold_anchored_cut  # noqa: PLC0415

    final_model = step.torch_model
    # The final model's pass over the haystack (#3314).  Real app work - it is
    # what the shipped cut is realized on and what the browse view ranks - and
    # it is K-INDEPENDENT, so it belongs in the denominator of a cost ratio and
    # not in `cal_seconds`.  Untimed it would simply be missing from the step,
    # which makes every fold count look more expensive than it is.
    t_final = time.monotonic()
    if style_obj is not None or region_aware:
        assert final_model is not None
        ids, all_scores = score_sim_set_with_model(
            final_model, region_aware, sim_clips, X_all_clips, sim_ids, style_obj
        )
    else:
        # Trainer-agnostic: the SVM arms have no torch model to forward.
        ids = sorted(sim_ids)
        all_scores = np.asarray(step.predict(np.asarray(X_all_clips))).ravel().tolist()
    details["final_score_seconds"] = time.monotonic() - t_final

    # The floor decision is taken once, on the final model's remainder, and
    # applies to every haystack in the step - all-or-nothing, so the fold and
    # final populations can never diverge.  ``apply_vote_exclusion`` is the
    # app's own decision function, so the default arm reproduces production by
    # construction rather than by a comment promising it does.
    fit_final, excluding = apply_vote_exclusion(all_scores, ids, voted_ids, min_remainder=exclusion_min_remainder)

    def _hay(scores: list[float], score_ids: list[int]) -> "np.ndarray":
        """One fold's haystack under this step's exclusion decision."""
        return drop_voted(scores, score_ids, voted_ids or ()) if excluding else np.asarray(scores, dtype=np.float64)

    fold_models = details.get("fold_models") or []
    fold_orderings = details.get("fold_orderings") or []
    n_folds = min(len(fold_models), len(fold_orderings))
    fold_haystacks: list[Any] = []
    # Per-fold haystack scoring seconds (#3314).  Scoring the sim set with each
    # fold model is a real, K-proportional part of the calibration a live run at
    # K pays for - the shipped rule anchors every fold's mixture on that fold's
    # own haystack - and it is paid *here*, outside `train_seconds`,
    # `xcal_seconds`, `pool_score_seconds` and `test_score_seconds`.  Left
    # unmeasured it is invisible to every cost model built out of those four,
    # which would make a fold-count affordability ceiling read a fraction of
    # what K actually costs.
    haystack_seconds: list[float] = []
    for model in fold_models[:n_folds]:
        t_hay = time.monotonic()
        fids, fscores = score_sim_set_with_model(model, region_aware, sim_clips, X_all_clips, sim_ids, style_obj)
        hay = _hay(fscores, fids)
        haystack_seconds.append(time.monotonic() - t_hay)
        fold_haystacks.append(hay)

    # #3116: the #2897 fold-count arms need a haystack per fold to re-fit the
    # *shipped* rule at each K, and `details["fold_models"]` is trimmed to the
    # live `calibrate_count`.  Score the extra Kmax-run folds here, where the
    # sim set and the scoring machinery are already in hand, and stash them
    # beside the orderings for :func:`_fold_count_variant_rows`.  Only the
    # models past the live prefix are scored - the folds are nested, so the
    # first `n_folds` haystacks are the ones just computed above, and this adds
    # `Kmax - calibrate_count` scoring passes rather than Kmax of them.
    fold_data = details.get("fold_count_data")
    if fold_data is not None and fold_data.get("models"):
        extended = list(fold_haystacks)
        for model in fold_data["models"][len(extended) :]:
            t_hay = time.monotonic()
            fids, fscores = score_sim_set_with_model(model, region_aware, sim_clips, X_all_clips, sim_ids, style_obj)
            hay = _hay(fscores, fids)
            haystack_seconds.append(time.monotonic() - t_hay)
            extended.append(hay)
        fold_data["haystacks"] = extended
        # Aligned with `haystacks`, so a K-prefix of one is a K-prefix of the
        # other.  Every entry is one scoring pass over the same sim set, so the
        # seconds are comparable across folds and across K by construction.
        fold_data["haystack_seconds"] = haystack_seconds

    cut = (
        fit_fold_anchored_cut(fold_haystacks, fold_orderings[:n_folds], fit_final.tolist()) if fold_haystacks else None
    )
    if cut is not None:
        anchored = cut.threshold_at(inclusion)
        if np.isfinite(anchored):
            return anchored, all_scores, ids, fold_haystacks, cut.provenance, cut
    blended = calculate_safe_threshold(_blend_xcal_input(threshold, details), all_scores, ctx, schedule=schedule)
    return blended, all_scores, ids, fold_haystacks, "gmm_blend", None


def _evaluate_on_test(
    step: StepModel,
    threshold: float,
    clips_dict: dict[int, dict[str, Any]],
    test_ids: list[int],
    target_category: str,
    inclusion: int,
    region_aware: bool = False,
    style_obj: Any = None,
) -> dict[str, float]:
    """Score *test_ids* with *step* and return the per-step metrics.

    Returns the operating-point metrics the user cares about — inclusion-weighted
    ``cost``, ``fpr``, ``fnr``, ``precision``, ``recall`` and ``f1`` (all
    computed at *threshold*, the last three via
    :func:`~vtscore.eval.calibration_metrics.detection_metrics`) — plus the
    threshold-independent ranking metrics ``auroc`` and ``average_precision``,
    which isolate "how good is the ranking" from "how good is the threshold".

    When *region_aware* the test media carry a ``patch_grid`` (a patch
    embedder), so scoring max-pools the MLP over every score row of each image -
    exactly the live detector's inference for patch datasets (an image scores
    by its best-matching row).  Otherwise each media is scored by its single
    whole-image vector through the step's trainer-agnostic ``predict``.
    """
    import numpy as np  # noqa: PLC0415

    from vtscore.eval.label_curve import _auroc, _average_precision  # noqa: PLC0415

    nan = float("nan")
    if not test_ids:
        return {
            "cost": nan,
            "fpr": nan,
            "fnr": nan,
            "precision": nan,
            "recall": nan,
            "f1": nan,
            "n_test_pos": nan,
            "n_test_neg": nan,
            "n_flagged": nan,
            "auroc": nan,
            "average_precision": nan,
        }

    if style_obj is not None:
        # Explicit detection style (see vtscore.eval.patch_styles): the style
        # owns the whole image-scoring rule (whole-image / region max-pool /
        # raw-patch max-pool), replacing both branches below.
        assert step.torch_model is not None
        test_clips = {cid: clips_dict[cid] for cid in test_ids}
        score_map = style_obj.score_media(step.torch_model, test_clips)
        scores = [score_map[cid] for cid in test_ids]
    elif region_aware:
        from vtscore.detectors.training import score_media_with_model  # noqa: PLC0415

        assert step.torch_model is not None
        test_clips = {cid: clips_dict[cid] for cid in test_ids}
        score_map = {r["id"]: r["score"] for r in score_media_with_model(step.torch_model, test_clips)}
        scores = [score_map[cid] for cid in test_ids]
    else:
        embs = np.array([media_embedding(clips_dict[cid]) for cid in test_ids])
        scores = np.asarray(step.predict(embs)).ravel().tolist()

    true_labels = [1.0 if media_is_positive(clips_dict[cid], target_category) else 0.0 for cid in test_ids]

    maybe_dump_predictions(clips_dict, test_ids, scores, true_labels, threshold, target_category, suffix="__eval")

    scores_arr = np.asarray(scores, dtype=np.float64)
    labels_arr = np.asarray(true_labels, dtype=np.float64)
    from vtscore.eval.calibration_metrics import (  # noqa: PLC0415
        detection_metrics,
        inclusion_weights,
        operating_cost,
    )

    cost, fpr, fnr = operating_cost(scores_arr, labels_arr, threshold, *inclusion_weights(inclusion))

    det = detection_metrics(scores_arr, labels_arr, threshold)
    return {
        "cost": round(cost, 6),
        "fpr": round(fpr, 6),
        "fnr": round(fnr, 6),
        **{k: round6(v) for k, v in det.items()},
        "auroc": round(_auroc(scores_arr, labels_arr), 6),
        "average_precision": round(_average_precision(scores_arr, labels_arr), 6),
    }


def _calibration_metric_rows(
    step: StepModel,
    threshold: float,
    details: dict[str, Any],
    clips_dict: dict[int, dict[str, Any]],
    test_ids: list[int],
    target_category: str,
    inclusion: int,
    style_obj: Any,
    repool_variants: list[str],
    topk: int,
) -> tuple[list[dict[str, Any]], "np.ndarray", "np.ndarray"]:
    """Per-step metric rows for the base pooling plus each remedial re-pool.

    Scores the test set's per-node sigmoids once through *style_obj*, then pools
    them ``max`` (base) and — for the raw-patch tree arm, which carries
    ``fold_node_data`` — ``topk`` / ``pnorm``.  Each remedial variant recalibrates
    its own threshold by re-pooling the same fold models' held-out node scores,
    so every arm has a genuine *trained* cost and an *oracle* cost.  Returns one
    row dict per pooling, each tagged with ``pool_variant``.
    """
    import numpy as np  # noqa: PLC0415

    from vtscore.eval import calibration_metrics as cm  # noqa: PLC0415
    from vtscore.eval.patch_styles import _forward_sigmoid_chunked  # noqa: PLC0415

    assert step.torch_model is not None  # the calibration study only runs the MLP style path
    model = step.torch_model
    provenance = details.get("provenance", "conformal")
    fold_orderings = details.get("fold_orderings") or []
    fold_node_data = details.get("fold_node_data")

    test_clips = {cid: clips_dict[cid] for cid in test_ids}
    ids, flat, seg = style_obj.node_scores(model, test_clips)
    labels = np.array([1.0 if media_is_positive(clips_dict[cid], target_category) else 0.0 for cid in ids])
    n_pool_rows = float(cm.segment_counts(seg, flat.shape[0]).mean()) if len(ids) else float("nan")

    rows: list[dict[str, Any]] = []

    # --- Base pooling (max): the arm's real operating point. ---
    base_scores = cm.segment_max_pool(flat, seg)
    # dump: calibration path -- `ids` is aligned with base_scores and labels.
    maybe_dump_predictions(clips_dict, list(ids), base_scores, list(labels), threshold, target_category)
    base_cal_scores = np.array([s for scores, _ in fold_orderings for s in scores]) if fold_orderings else None
    base_cal_labels = np.array([lb for _, labels_ in fold_orderings for lb in labels_]) if fold_orderings else None
    base = operating_metrics(
        base_scores,
        labels,
        threshold,
        inclusion,
        base_cal_scores,
        base_cal_labels,
        pool_variant="max",
        provenance=provenance,
        n_pool_rows=n_pool_rows,
    )
    if "xcal_threshold" in details:
        # Under safe_thresholds the base row's threshold is the blended one;
        # record the pre-blend conformal cut alongside it (issue #2799).
        base["xcal_threshold"] = round6(float(details["xcal_threshold"]))
    # How many held-out scores the conformal quantile was actually taken over,
    # on the SHIPPED row rather than only on the fold-count variant rows (issue
    # #3287).  It was declared in `CALIBRATION_COLUMNS` and filled only by the
    # #2897 arms, so the one quantity `calibration_fraction` directly controls -
    # the resolution of the quantile the threshold is read from - was NaN on
    # every production row.  A knob whose mechanism is invisible in the output
    # can only be argued about; this makes it a column.
    if base_cal_scores is not None:
        base["n_cal_scores"] = int(np.asarray(base_cal_scores).size)
    rows.append(base)

    # --- Remedial re-pools: only where the same fold models exposed node data
    # (the raw-patch tree arm) and the base threshold was a real conformal cut. ---
    if fold_node_data and repool_variants:
        # Final-model node scores over the bad-voted bags -> the pnorm test null.
        neg_rows = details.get("neg_score_rows") or []
        if neg_rows:
            null_concat = np.concatenate([np.asarray(r, dtype=np.float32) for r in neg_rows], axis=0)
            test_null = np.sort(np.asarray(_forward_sigmoid_chunked(model, null_concat), dtype=np.float64))
        else:
            test_null = np.empty(0, dtype=np.float64)

        for variant in repool_variants:
            # Recalibrate the threshold: re-pool each fold's held-out calibration
            # groups under this variant, then run the conformal rule on the pool.
            v_orderings: list[tuple[list[float], list[float]]] = []
            for blocks, blk_labels in fold_node_data:
                if variant == "pnorm":
                    fold_null = cm.negative_block_null(blocks, blk_labels)
                    pooled = cm.pool_blocks(blocks, "pnorm", null_sorted=fold_null)
                else:
                    pooled = cm.pool_blocks(blocks, variant, topk=topk)
                v_orderings.append((pooled, list(blk_labels)))
            v_threshold = threshold_from_fold_orderings(v_orderings, inclusion)

            # Re-pool the test node scores under this variant.
            if variant == "pnorm":
                v_scores = cm.segment_pnorm_pool(flat, seg, test_null)
            else:
                v_scores = cm.segment_topk_mean_pool(flat, seg, topk)
            v_cal_scores = np.array([s for scores, _ in v_orderings for s in scores])
            v_cal_labels = np.array([lb for _, labels_ in v_orderings for lb in labels_])
            rows.append(
                operating_metrics(
                    v_scores,
                    labels,
                    v_threshold,
                    inclusion,
                    v_cal_scores,
                    v_cal_labels,
                    pool_variant=variant,
                    provenance="conformal",
                    n_pool_rows=n_pool_rows,
                )
            )

    return rows, base_scores, labels


# ------------------------------------------------------------------
# Supervised skyline (issue #3322)
# ------------------------------------------------------------------


def _skyline_fit_and_score(
    good_ids: list[int],
    bad_ids: list[int],
    score_ids: list[int],
    clips_dict: dict[int, dict[str, Any]],
    target_category: str,
    *,
    trainer: str,
    head: str,
    style_obj: Any,
    region_voting: bool,
    input_dim: int,
    inclusion: int,
    calibrate_count: int,
    calibration_fraction: float,
) -> tuple[dict[int, float], StepModel, dict[str, float], float]:
    """Train one fully-supervised head and score *score_ids* with it.

    The whole point of the skyline is that it differs from a mortal step in
    **the labels and nothing else**, so this goes through
    :func:`_train_and_calibrate` rather than fitting an estimator of its own:
    same head, same inclusion class-weights, same pinned fit seed, same backend,
    same bag-aware flooding.  A skyline that trained through a private
    ``LinearSVC`` would report "fewer labels" and "different trainer" as one
    number, and would need its own ``Mirror(...)`` entry in
    ``scripts/check-eval-app-sync.py`` to stay honest; delegation needs neither.

    The threshold the trainer calibrates is computed and **discarded** - the
    caller takes the test oracle's cut instead (see :data:`SKYLINE_PROVENANCE`).
    It is paid for rather than skipped because the alternative is a second entry
    point into the trainer that only the skyline uses, which is exactly the kind
    of near-copy this module keeps out.

    Returns ``({media_id: score}, step, timings, test_score_seconds)``.
    """
    from vtscore.eval import calibration_metrics as cm  # noqa: PLC0415

    step, _threshold, _n_labels, timings, _details = _train_and_calibrate(
        trainer,
        dict.fromkeys(good_ids),
        dict.fromkeys(bad_ids),
        clips_dict,
        target_category,
        region_voting=region_voting,
        input_dim=input_dim,
        inclusion=inclusion,
        calibrate_count=calibrate_count,
        calibration_fraction=calibration_fraction,
        head=head,
        style_obj=style_obj,
        emit_calibration_metrics=False,
    )
    assert step.torch_model is not None  # v1 runs the styled torch path only
    t_score = time.monotonic()
    ids, flat, seg = style_obj.node_scores(step.torch_model, {cid: clips_dict[cid] for cid in score_ids})
    pooled = cm.segment_max_pool(flat, seg)
    score_seconds = time.monotonic() - t_score
    return ({cid: float(v) for cid, v in zip(ids, pooled, strict=True)}, step, timings, score_seconds)


def _skyline_arm_rows(
    arms: list[str],
    clips_dict: dict[int, dict[str, Any]],
    target_category: str,
    sim_ids: list[int],
    test_ids: list[int],
    inclusion: int,
    *,
    trainer: str,
    head: str,
    style_obj: Any,
    region_voting: bool,
    input_dim: int,
    calibrate_count: int,
    calibration_fraction: float,
    seed: int,
) -> list[dict[str, Any]]:
    """One metric row per requested skyline arm (issue #3322), or ``[]``.

    Both arms are scored on the **untouched test split** - the same ``test_ids``
    every mortal row is scored on, pooled through the same
    ``style_obj.node_scores`` + segment max, so the two sides of
    ``training_regret`` differ in the model and in nothing else.  The #3308
    population convention is satisfied trivially rather than by re-applying
    :func:`~vtscore.training.thresholds.apply_vote_exclusion`: that convention is
    about the *haystack a threshold's population estimate is fitted on*, and a
    skyline fits no such estimate - it takes the test oracle's cut - so there is
    no haystack here to exclude votes from.

    ``skyline_test_xfit`` pools **cross-fitted** scores: the test split is
    partitioned into :data:`~vtscore.eval.transfer_rules.HONEST_ORACLE_FOLDS`
    folds and each item is scored by a head that never saw it.  Two caveats ride
    on that row and neither is a defect:

    * The pooled scores come from K different heads, so they share a *ranking*
      but not a calibrated scale.  That is fine for ``oracle_cost`` / ``auroc`` /
      ``average_precision``, which is all this arm is read for.
    * Its ``oracle_cost`` is still a sample minimum over those pooled scores, and
      is optimistic for exactly the reason ``oracle_cost`` always is - which is
      why ``oracle_cost_honest`` ships beside it, cross-fitting the *cut* on top
      of the cross-fitted *model*.

    Returns each row already carrying its ``gmm_variant`` tag and its own timing
    / backend columns; the caller supplies the identifying columns.
    """
    import numpy as np  # noqa: PLC0415

    from vtscore.eval import calibration_metrics as cm  # noqa: PLC0415

    nan = float("nan")
    started = time.monotonic()
    ordered_test = sorted(test_ids)
    test_labels = np.array(
        [1.0 if media_is_positive(clips_dict[cid], target_category) else 0.0 for cid in ordered_test],
        dtype=np.float64,
    )
    wf, wn = cm.inclusion_weights(inclusion)

    def _row(name: str, score_map: dict[int, float], step: StepModel, timings: dict[str, float], secs: float):
        scores = np.array([score_map[cid] for cid in ordered_test], dtype=np.float64)
        o_thr, _o_cost, _o_fpr, _o_fnr = cm.oracle_cut(scores, test_labels, wf, wn)
        if not np.isfinite(o_thr):
            return None
        row = operating_metrics(
            scores,
            test_labels,
            float(o_thr),
            inclusion,
            None,
            None,
            pool_variant="max",
            provenance=SKYLINE_PROVENANCE,
            n_pool_rows=1.0,
        )
        row["gmm_variant"] = name
        row["schedule"] = ""
        row.update(
            {
                "calibrate_count": calibrate_count,
                "train_seconds": round(timings["train_seconds"], 6),
                "final_score_seconds": nan,
                "xcal_seconds": round(timings["xcal_seconds"], 6),
                "pool_score_seconds": nan,
                "test_score_seconds": round(secs, 6),
                "backend": step.backend,
                "device": step.device,
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
        )
        return row

    rows: list[dict[str, Any]] = []

    if SKYLINE_TRAIN_FULL in arms:
        sim_pos = [cid for cid in sorted(sim_ids) if media_is_positive(clips_dict[cid], target_category)]
        sim_neg = [cid for cid in sorted(sim_ids) if not media_is_positive(clips_dict[cid], target_category)]
        if sim_pos and sim_neg:
            score_map, step, timings, secs = _skyline_fit_and_score(
                sim_pos,
                sim_neg,
                ordered_test,
                clips_dict,
                target_category,
                trainer=trainer,
                head=head,
                style_obj=style_obj,
                region_voting=region_voting,
                input_dim=input_dim,
                inclusion=inclusion,
                calibrate_count=calibrate_count,
                calibration_fraction=calibration_fraction,
            )
            row = _row(SKYLINE_TRAIN_FULL, score_map, step, timings, secs)
            if row is not None:
                rows.append(row)

    if SKYLINE_TEST_XFIT in arms:
        xfit = _skyline_xfit_scores(
            ordered_test,
            clips_dict,
            target_category,
            trainer=trainer,
            head=head,
            style_obj=style_obj,
            region_voting=region_voting,
            input_dim=input_dim,
            inclusion=inclusion,
            calibrate_count=calibrate_count,
            calibration_fraction=calibration_fraction,
            seed=seed,
        )
        if xfit is not None:
            row = _row(SKYLINE_TEST_XFIT, *xfit)
            if row is not None:
                rows.append(row)

    return rows


def _skyline_xfit_scores(
    ordered_test: list[int],
    clips_dict: dict[int, dict[str, Any]],
    target_category: str,
    *,
    trainer: str,
    head: str,
    style_obj: Any,
    region_voting: bool,
    input_dim: int,
    inclusion: int,
    calibrate_count: int,
    calibration_fraction: float,
    seed: int,
) -> tuple[dict[int, float], StepModel, dict[str, float], float] | None:
    """Cross-fitted test-side skyline scores: every item scored by a head that never saw it.

    Partitions *ordered_test* into :data:`~vtscore.eval.transfer_rules.HONEST_ORACLE_FOLDS`
    folds, trains the fully-supervised head on the complement of each, and scores
    the held-out fold with it.  ``None`` when the split cannot be made honestly -
    fewer items than folds, or a fold whose complement carries only one class -
    which leaves the bracket arm out of the frame rather than quietly falling back
    to the train-on-test fit it exists to avoid.

    Returns the same tuple shape as :func:`_skyline_fit_and_score`, with the
    per-fold wall clocks summed and the *last* fold's step standing in for the
    backend/device columns (every fold trains on the same backend).
    """
    import numpy as np  # noqa: PLC0415

    from vtscore.eval.transfer_rules import HONEST_ORACLE_FOLDS  # noqa: PLC0415

    if len(ordered_test) < HONEST_ORACLE_FOLDS:
        return None
    # Seeded off the run's own seed, never off the trajectory's RandomState:
    # drawing from `rng` would move every subsequent vote, so turning the arm on
    # would silently change the run it is meant to describe.
    order = np.random.default_rng([_SKYLINE_SEED, seed]).permutation(len(ordered_test))
    scores: dict[int, float] = {}
    timings = {"train_seconds": 0.0, "xcal_seconds": 0.0}
    total_secs = 0.0
    step: StepModel | None = None
    for part in np.array_split(order, HONEST_ORACLE_FOLDS):
        held = [ordered_test[i] for i in part]
        if not held:
            continue
        held_set = set(held)
        fit_pos: list[int] = []
        fit_neg: list[int] = []
        for cid in ordered_test:
            if cid in held_set:
                continue
            (fit_pos if media_is_positive(clips_dict[cid], target_category) else fit_neg).append(cid)
        if not fit_pos or not fit_neg:
            return None
        score_map, step, fold_timings, secs = _skyline_fit_and_score(
            fit_pos,
            fit_neg,
            held,
            clips_dict,
            target_category,
            trainer=trainer,
            head=head,
            style_obj=style_obj,
            region_voting=region_voting,
            input_dim=input_dim,
            inclusion=inclusion,
            calibrate_count=calibrate_count,
            calibration_fraction=calibration_fraction,
        )
        scores.update({cid: score_map[cid] for cid in held})
        timings["train_seconds"] += fold_timings["train_seconds"]
        timings["xcal_seconds"] += fold_timings["xcal_seconds"]
        total_secs += secs
    if step is None or len(scores) != len(ordered_test):
        return None
    return scores, step, timings, total_secs


def _apply_skyline_decomposition(rows: list[dict[str, Any]], skyline_rows: list[dict[str, Any]]) -> None:
    """Fill the :data:`SKYLINE_COLUMNS` on *rows* from the primary skyline arm.

    ``training_regret = oracle_cost(row) - oracle_cost(skyline)`` on the naive
    reference and the cross-fitted one alike, so both decompositions telescope
    exactly against the ``regret`` / ``regret_honest`` the row already carries.
    Applied to the skyline rows too, which makes ``skyline_train_full``'s own
    ``training_regret`` exactly ``0`` and ``skyline_test_xfit``'s the bracket
    between the two references - both of which are the right readings.

    A no-op when :data:`SKYLINE_TRAIN_FULL` did not run: the columns stay NaN
    rather than being re-based on the bracket partner, which measures capacity on
    the test sample and not learnability.
    """
    ref = next((r for r in skyline_rows if r["gmm_variant"] == SKYLINE_TRAIN_FULL), None)
    if ref is None:
        return
    floor = ref["oracle_cost"]
    floor_honest = ref["oracle_cost_honest"]
    for row in [*rows, *skyline_rows]:
        row["skyline_oracle_cost"] = floor
        row["skyline_oracle_cost_honest"] = floor_honest
        row["training_regret"] = round6(row["oracle_cost"] - floor)
        row["training_regret_honest"] = round6(row["oracle_cost_honest"] - floor_honest)


# ------------------------------------------------------------------
# Single (seed, dataset, category) evaluation
# ------------------------------------------------------------------


@dataclass(frozen=True)
class _RunKnobs:
    """The pre-registered knobs of one :func:`simulate_voting_iterations` cell.

    A record rather than a tuple, so adding a knob cannot silently rebind the
    existing ones at the unpacking call site - the same hazard the keyword-only
    marker on :func:`simulate_voting_iterations` closes for its callers.
    """

    fold_schedule: "Callable[[int], int] | None"
    startup_state: StartupState | None
    skyline_arms: list[str]
    head: str


def _resolve_head(head: Optional[str], trainer: str) -> str:
    """Validate an explicit MLP head name, or resolve ``None`` to the app's.

    Raises:
        ValueError: If *head* is not a known head, or is given for a trainer
            that fits its own estimator rather than a head.
    """

    if head is not None:
        if head not in HEADS:
            raise ValueError(f"unknown head {head!r}; expected one of {HEADS}")
        if trainer != "mlp":
            raise ValueError(f"head={head!r} only applies to the production trainer; got trainer={trainer!r}")
    # **The default arm must be the app's default.**  Production pins the linear
    # SVM head on every fit (``hidden_dim = LINEAR_SVM_HEAD`` in
    # ``vtscore.detectors.training.train_and_threshold``), so an unspecified head
    # resolves to it — the same way *style* and *blend_schedule* resolve to the
    # app's geometry and schedule below.  The head fits the final model *and*
    # the calibration folds, so it moves the thresholds and, through the vote
    # order, the whole trajectory: defaulting to a retired head would make every
    # unqualified run measure a detector nobody ships.  ``head="linear"`` (the
    # logistic head) and ``head="mlp"`` stay available as named legacy arms.
    head = head or PRODUCTION_HEAD
    return head


def _resolve_startup_state(
    startup_schedule: Optional[str],
    seed_scores: Optional[dict[int, float]],
    autopilot_fidelity: bool,
    strategy: str,
) -> StartupState | None:
    """Parse an explicit startup schedule, checking the run can actually honour it.

    Raises:
        ValueError: If a schedule is named without the seed sort it addresses
            positions on, or outside the faithful autopilot strategy it steers.
    """

    startup_state: StartupState | None = None
    if startup_schedule:
        if seed_scores is None:
            raise ValueError(
                "startup_schedule needs seed_scores: a schedule names positions on the "
                "seed sort, and there is no sort to name them on without one"
            )
        if not autopilot_fidelity or strategy != "autopilot":
            raise ValueError("startup_schedule requires autopilot_fidelity and the autopilot strategy")
        startup_state = StartupState(parse_startup_schedule(startup_schedule))
    return startup_state


def _resolve_run_knobs(
    *,
    fold_count_schedule: str | None,
    calibrate_count: int,
    startup_schedule: Optional[str],
    seed_scores: Optional[dict[int, float]],
    autopilot_fidelity: bool,
    strategy: str,
    skyline_arms: Optional[list[str]],
    emit_calibration_metrics: bool,
    acq_inclusion_offset: float,
    acq_rank_percentile: Optional[float],
    head: Optional[str],
    trainer: str,
    style: Optional[str],
) -> _RunKnobs:
    """Validate a cell's pre-registered knobs and resolve the defaults among them.

    Called from the top of :func:`simulate_voting_iterations`, before anything
    expensive runs, and the only place these checks belong: a run that dies
    forty minutes in on a typo has held a cluster slot for nothing, so a
    malformed knob has to kill the cell at second zero.  Gathering them under
    one name is what keeps that contract visible - a ``raise`` added deep in the
    stepping loop now reads as an obvious departure rather than as one more line
    among seven hundred.

    Note this resolves only the defaults that can be decided from the arguments
    alone.  The two that need the loaded pool - the detection *style* and the
    pair in :func:`_resolve_production_defaults` - are keyed on ``region_aware``
    and so are resolved later, once the medias have been read.

    Raises:
        ValueError: If a knob is malformed, or two knobs are mutually exclusive.
    """

    # These are pre-registered experiment knobs, so they are validated beside
    # the other argument checks rather than deep in the loop: a run that dies
    # forty minutes in on a typo has held a cluster slot for nothing.
    # Parsed here, with the other pre-registered knobs, and not in the loop: a
    # malformed schedule must kill the cell at second zero rather than forty
    # minutes in, and the parse is what validates the spec at all.
    _fold_schedule = parse_fold_count_schedule(fold_count_schedule, calibrate_count)

    startup_state = _resolve_startup_state(startup_schedule, seed_scores, autopilot_fidelity, strategy)

    skyline_arms = list(skyline_arms or [])
    if skyline_arms:
        unknown = [a for a in skyline_arms if a not in SKYLINE_ARMS]
        if unknown:
            raise ValueError(f"unknown skyline arm(s) {unknown}; expected a subset of {list(SKYLINE_ARMS)}")
        if not emit_calibration_metrics:
            raise ValueError(
                "skyline_arms requires emit_calibration_metrics: the decomposition is defined "
                "against the calibration frame's oracle_cost/regret columns, which the plain "
                "frame does not carry"
            )

    if acq_rank_percentile is not None:
        if acq_inclusion_offset != 0:
            raise ValueError(
                "acq_inclusion_offset and acq_rank_percentile are mutually exclusive; "
                "pass acq_inclusion_offset=0 to run the rank-pinned arm "
                f"(the default is {ACQUISITION_INCLUSION_OFFSET}, the shipped acquisition cut)"
            )
        if not 0.0 <= acq_rank_percentile <= 1.0:
            raise ValueError(f"acq_rank_percentile must be in [0, 1], got {acq_rank_percentile}")

    head = _resolve_head(head, trainer)

    if style is not None and trainer != "mlp":
        raise ValueError(f"detection styles only support the MLP trainer; got trainer={trainer!r}")
    return _RunKnobs(
        fold_schedule=_fold_schedule,
        startup_state=startup_state,
        skyline_arms=skyline_arms,
        head=head,
    )


def _resolve_production_defaults(
    *,
    blend_schedule: Optional[str],
    calibration_fraction: Optional[float],
    region_aware: bool,
) -> tuple[str, float]:
    """Resolve the unnamed blend schedule and split fraction to the app's own.

    **The eval default arm must be the app's default.**  Both ``default``
    mirrors in ``scripts/check-eval-app-sync.py`` -
    ``training.blend_schedule_default`` and ``training.split_fraction_default`` -
    name *this* function as their harness side, so a tripped gate points the
    reconciler at twenty lines instead of at the thousand that make up
    :func:`simulate_voting_iterations`.  It also gives that gate something real
    to check: its harness-side test is an existence check on the named symbol,
    which the enclosing function would satisfy even if both resolutions below
    were deleted outright.

    Both resolutions are keyed on *region_aware* - whether any media in the pool
    carries a ``patch_grid`` - which is why they run here rather than in
    :func:`_resolve_run_knobs`, alongside the knobs that need no pool.

    Args:
        blend_schedule: The named schedule arm, or ``None`` for the default.
        calibration_fraction: The explicit Train/Calibrate split, or ``None``.
        region_aware: Whether the pool carries patch grids.

    Returns:
        The resolved ``(blend_schedule, calibration_fraction)`` pair.
    """

    # Mirror the app's per-mode schedule default (#2841): with no explicit arm, a
    # patch dataset blends under the region schedule and a single-vector one
    # under the binary schedule, exactly as `_blend_schedule_for_snap` decides in
    # `vtscore.detectors.training`.  Without this the harness would measure a
    # schedule no detector actually uses.
    if blend_schedule is None:
        from vtscore.training.blend_schedules import production_schedule_for  # noqa: PLC0415

        blend_schedule = production_schedule_for(region_voting=region_aware)

    # Mirror the app's per-space split default (#3287/#3290): with no explicit
    # arm, the Train/Calibrate fraction of each fold is the one a live
    # detector would resolve for this dataset's embedder.  ``region_aware``
    # (any media carrying a ``patch_grid``) is the harness's spelling of "the
    # pickle was built by a patch embedder" - the same capability the app
    # reads off ``supports_patch_regions`` in
    # ``vtscore.detectors.training.resolve_calibration_fraction``, which the
    # ``training.split_fraction_default`` mirror in
    # ``scripts/check-eval-app-sync.py`` pins against this block.  Note it is
    # deliberately NOT the voting mode: ``dinov3_patch`` datasets take 0.5 in
    # both their styles, including boxless ``whole_image``.
    if calibration_fraction is None:
        from vtscore.training.thresholds import production_split_for  # noqa: PLC0415

        calibration_fraction = production_split_for(patch_space=region_aware)
    return blend_schedule, calibration_fraction


def simulate_voting_iterations(  # noqa: C901
    clips_dict: dict[int, dict[str, Any]],
    target_category: str,
    seed: int,
    *,
    dataset_name: str = "",
    inclusion: int = 0,
    sim_fraction: float = 0.5,
    safe_thresholds: bool = True,
    calibrate_count: int = 2,
    fold_count_schedule: str | None = None,
    calibration_fraction: Optional[float] = None,
    region_voting: bool = False,
    strategy: str = "autopilot",
    max_steps: Optional[int] = None,
    atlas_min_node_size: int = 20,
    seed_scores: Optional[dict[int, float]] = None,
    trainer: str = "mlp",
    head: Optional[str] = None,
    target_prevalence: Optional[float] = None,
    style: Optional[str] = None,
    emit_calibration_metrics: bool = False,
    repool_variants: Optional[list[str]] = None,
    repool_topk: int = 4,
    inclusion_sweep_ks: Optional[list[int]] = None,
    sweep_sink: Optional[list[dict[str, Any]]] = None,
    blend_schedule: Optional[str] = None,
    schedule_variants: Optional[list[str]] = None,
    cut_diag_sink: Optional[list[dict[str, Any]]] = None,
    fit_quality_sink: Optional[list[dict[str, Any]]] = None,
    fit_quality_stride: int = FIT_QUALITY_STRIDE_DEFAULT,
    autopilot_fidelity: bool = True,
    anchored_thresholds: bool = False,
    anchored_weights: Optional[list[float]] = None,
    anchored_rules: Optional[list[str]] = None,
    anchored_fold_arms: bool = True,
    anchored_fold_combines: Optional[list[str]] = None,
    fold_count_variants: Optional[list[int]] = None,
    cut_inclusion_ks: Optional[list[int]] = None,
    cut_inclusion_sink: Optional[list[dict[str, Any]]] = None,
    cut_inclusion_qtilt_steps: Optional[list[float]] = None,
    acq_inclusion_offset: float = ACQUISITION_INCLUSION_OFFSET,
    acq_rank_percentile: Optional[float] = None,
    startup_schedule: Optional[str] = None,
    pick_sink: Optional[list[dict[str, Any]]] = None,
    exclusion_min_remainder: Optional[float] = None,
    skyline_arms: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    """Simulate voting on *clips_dict* and evaluate at every step.

    Args:
        clips_dict: Pre-loaded media dict (``{id: clip_data}``).
        target_category: Category treated as the positive class.
        seed: Random seed for splitting and vote ordering.
        dataset_name: Label included in result rows.
        inclusion: Inclusion setting in ``[-10, 10]``.
        trainer: Which ranker to train at each step — ``"mlp"`` (default, the
            production path, whose head is chosen by *head*) or a standalone
            SVM name
            (``"svm_linear"``, ``"svm_rbf"``, or a parameterised spec such as
            ``"svm_rbf@C=3,gamma=scale"``).  The autopilot vote order adapts to
            the chosen model, so MLP and SVM trajectories diverge after the
            first retrain even at the same seed — by design (the question is
            which model makes *VTSearch* better, and VTSearch's vote order
            depends on the model).
        head: Which head the ``"mlp"`` trainer fits at each step (see
            :data:`HEADS`).  ``None`` (default) resolves to the **app's** head,
            :data:`PRODUCTION_HEAD` — the linear SVM a live VTSearch detector
            actually has, so a default run's thresholds and costs are the ones
            users see.  ``"linear"`` (the logistic head the SVM replaced) and
            ``"mlp"`` (the harness's auto-sized hidden layer, #2781) are the
            explicitly-named legacy arms.  The head is threaded into the
            calibration folds as well, mirroring how production threads one
            sentinel through ``_train_and_score_xy``.  Rejected on the
            standalone SVM trainers, which fit their own estimator rather than
            a head; the *resolved* name is recorded in the ``head`` result
            column (blank on those trainers).
        style: Optional detection-style name (see
            :mod:`vtscore.eval.patch_styles`): ``"whole_image"``,
            ``"max_patch"`` (the production geometry), or one of the
            ``"max_patch_hac"`` hybrids.  When set (MLP trainer only), the style owns the
            vote-to-vector assembly, the test/sim scoring rule, and the
            bag-aware flooding of Bad votes - the Max-Patch experiment arms.
            ``None`` (default) resolves to the **app's** geometry: a patch
            dataset (any media with a ``patch_grid``) on the MLP trainer gets
            ``"max_patch"``, everything else keeps the historical single-vector
            path byte-for-byte.  The *resolved* name is what lands in the
            ``style`` result column, so a row always says which geometry
            produced it.
        target_prevalence: When set (e.g. ``0.01`` for the 1%-prevalence rare
            arm), positives across the whole dataset are deterministically
            downsampled — using ``seed`` — to that fraction *before* the
            sim/test split, so every FPR/FNR is measured at the target
            prevalence.  ``None`` (default) uses the category's natural
            prevalence and is numerically identical to the pre-prevalence
            harness.  The arm is skipped (returns ``[]``) if it would leave
            fewer than :data:`_MIN_PREVALENCE_POSITIVES` positives, to keep the
            test-set FNR estimable.
        sim_fraction: Fraction of medias used for simulated voting.
        safe_thresholds: The shipped threshold path - fuse the haystack score
            distribution into the trained cut (the fold-anchored estimator, see
            :func:`vtscore.training.thresholds.fold_anchored_gmm_threshold`).
            **On by default, matching the app**, which has no switch for it.
            Set ``False`` only to run the no-fusion control arm: pure
            cross-calibration, which the app can no longer produce.
            Under ``emit_calibration_metrics`` with a *style*, each step
            additionally emits one metric row per safe-threshold cut variant
            (:data:`_SAFE_GMM_VARIANTS`, tagged in the ``gmm_variant`` column) -
            the #2799/#2836 measurement arms - and, when *cut_diag_sink* is
            given, one :data:`CUT_DIAGNOSTIC_COLUMNS` row per (step, geometry)
            carrying the fitted mixture parameters and the #2836 decomposition
            chain.
        calibrate_count: Number of random Train/Calibrate splits for threshold
            calibration (default 2).
        fold_count_schedule: **Eval-only** (#3314).  ``"K@N"`` resolves
            *calibrate_count* per step from the vote count -
            ``K(n_votes) = K while n_votes < N, else calibrate_count`` - so a
            run can spend more folds where they are cheapest and decay to
            production's count as the labelset grows.  ``None`` (every other
            caller) keeps *calibrate_count* constant, which is byte-identical
            to the behaviour before the knob existed.  See
            :func:`parse_fold_count_schedule` for why this is a harness knob
            and not a shipped setting.
        calibration_fraction: Fraction of labelled data reserved for
            calibration in each split.  ``None`` (default) resolves to the
            **app's** per-space split
            (:func:`vtscore.training.thresholds.production_split_for`, issue
            #3287): 0.5 when the dataset carries a ``patch_grid`` (built by a
            patch embedder), 0.3 otherwise - so a default run's folds are
            split the way a live detector's are.
        region_voting: When ``True``, each Good vote trains on the region-pooled
            vector of the media's ground-truth box for *target_category* (the
            minimal box covering every annotated instance), instead of the
            whole-image vector - simulating a user who drags a region around the
            object.  Requires a patch embedder: media without a ``patch_grid``
            or without an annotated box fall back to the whole-image vector.
            Scoring is unaffected by this flag - a patch dataset always scores
            region-aware (max-pool over regions), so the only thing this toggles
            is the Good-vote training vector, isolating region voting's effect.
        strategy: Vote-order strategy naming *which* pool item the simulated
            user labels next (see :data:`vtscore.eval.al_strategies.STRATEGIES`).
            Only ``"autopilot"`` (the default) exists: it reproduces the app's
            real user flow — seed from text sort (or random known-good examples),
            then the standard Good / Bad / Hard / New phases.
        max_steps: Cap on the number of voting steps (pool items labelled).
            ``None`` (default) votes on the entire simulation set.
        atlas_min_node_size: Minimum leaf population for the coverage atlas the
            autopilot New phase reads (default 20, the production floor).  Lower
            it for small simulation sets so diversity cells actually resolve.
        seed_scores: Optional ``{media_id: similarity}`` text-sort ranking (each
            item's cosine to the typed query).  When provided the autopilot seed
            follows the text sort (top items for the initial goods, the sort's
            cutoff for the initial bads); ``None`` (default) means the dataset
            has no text sort, so autopilot seeds from random known-good examples.
        cut_inclusion_ks: Inclusion values the **fold-anchored cut rules** are
            swept over for issue #2865, into *cut_inclusion_sink* (columns
            :data:`CUT_INCLUSION_COLUMNS`).  Orthogonal to
            *inclusion_sweep_ks*, which sweeps the conformal rule's budget: this
            one asks which cut rule should answer the Inclusion knob, so its
            rows are scored at their own ``k`` rather than at *inclusion*.  The
            arms come from *anchored_weights* x *anchored_rules* x
            *anchored_fold_combines*, so ``anchored_rules=["mid", "mid_tilt",
            "rate", "cross_tilt", "q_tilt"]`` is the candidate set the issue
            names.  ``None`` (default) = off, and every other study is unchanged.
        cut_inclusion_sink: List the #2865 rows are appended to.  Required for
            *cut_inclusion_ks* to do anything.
        cut_inclusion_qtilt_steps: Step sizes the eval-only ``q_tilt`` rule is
            expanded over (its free parameter; every other rule ignores this).
            Defaults to the single placeholder
            :data:`~vtscore.training.thresholds.FOLD_ANCHOR_QTILT_STEP`.
        acq_inclusion_offset: Cut the threshold handed to the **selector** at
            ``inclusion + acq_inclusion_offset``, leaving reporting and every
            metric at *inclusion* so arms stay comparable.  Defaults to
            :data:`~vtscore.training.thresholds.ACQUISITION_INCLUSION_OFFSET`
            (-4), **the shipped app behaviour** - the harness matches production
            here as it does everywhere else, so a baseline arm measures what
            users get.  Pass ``0`` for the pre-#2876 control, where one threshold
            did both jobs.

            The direction is the opposite of the intuition from the cost
            weights, because Autopilot's ``hard`` pick reads the threshold as a
            **rank position**, not a decision boundary: a *negative* offset
            prices false alarms higher, *raises* the cut, moves it *up* the
            ranking, and so returns *more* positives.  Requires a fold-anchored
            cut for the step; steps that fall back to the schedule blend keep
            the reporting threshold (the blend has no inclusion-aware form).
        startup_schedule: A parameterised Autopilot **opening** (issue #3267),
            e.g. ``"n6@k-6,n6@k-2,n6@k0"``; see
            :mod:`vtscore.eval.startup_schedule` for the grammar.  ``None``
            (default) is the **app's own** opening - three positives off the top
            of the seed sort, four negatives at its cutoff - and leaves the
            trajectory byte-for-byte what it was before the knob existed.  A
            schedule replaces only the pre-detector phases; the learned Hard
            sort that follows is unchanged and still samples at
            *acq_inclusion_offset*.  Requires *seed_scores*: a schedule names
            positions on the seed sort, so there has to be one.
        pick_sink: List the per-click :data:`PICK_COLUMNS` rows are appended
            to - one per vote, including the opening's, which emit no main row
            because no model exists yet.  ``None`` (default) = off.
        acq_rank_percentile: Alternative acquisition cut - place it at this
            quantile of the simulation-set score distribution directly, rather
            than by naming an inclusion.  This is the ``rank_pin`` arm: same
            intent, one fewer indirection.  Requires
            ``acq_inclusion_offset=0``, since the two name the same cut.
        anchored_thresholds: When ``True`` (requires ``safe_thresholds``,
            ``emit_calibration_metrics``, and a *style*), each step additionally
            emits one metric row per anchored-mixture arm (issue #2852): the
            label-anchored family (``anchored_w{W}_{rule}``), the fold-anchored
            "cross-LabeledGMM" family (``fold_anchored_w{W}_{rule}_{combine}``),
            and the ``rank_transfer`` attribution arm - see
            :func:`_anchored_variant_rows`.  The fold arms score the sim set
            once per calibration fold model per step, so they cost roughly one
            extra scoring pass per fold.
        anchored_weights: Anchor-weight grid for the anchored arms (default
            :data:`_ANCHORED_WEIGHTS`).  Each labelled score counts as this
            many haystack scores in the anchored EM's M-step.
        anchored_rules: Cut rules applied to each anchored fit (default
            :data:`_ANCHORED_RULES`): ``"mid"`` (plain midpoint), ``"rate"``
            (rate-optimal crossing at the live inclusion weights), and/or
            ``"mid_tilt"`` (the shipped rule: midpoint anchored at inclusion 0,
            rate tilt away from it).  ``"mid_tilt"`` is defined in
            fold-quantile space, so it applies to the fold-anchored family
            only; the label-anchored family skips it.
        anchored_fold_arms: Include the fold-anchored + rank-transfer arms
            (default ``True``); ``False`` keeps only the cheap label-anchored
            family (no per-fold scoring passes).
        anchored_fold_combines: How the fold arms combine per-fold cuts in
            quantile space (default :data:`_ANCHORED_FOLD_COMBINES`):
            ``"qmean"`` and/or ``"qmedian"``.
        fold_count_variants: Calibration fold counts to score counterfactually
            (issue #2897; requires ``emit_calibration_metrics`` and a *style*).
            Each step trains ``max(calibrate_count, *variants)`` folds instead of
            ``calibrate_count`` and emits one ``folds_k{K}_xcal`` row - plus a
            ``folds_k{K}_blend`` row where the step has a safe-threshold fit -
            per K, carrying that K's regret and its measured ``fold_seconds``.
            The folds are nested, so the live threshold and the trajectory are
            byte-identical to a plain run at ``calibrate_count`` and the arm at
            ``K == calibrate_count`` reproduces this step's own conformal cut;
            see :func:`_fold_count_variant_rows`.  Costs ``Kmax - calibrate_count``
            extra fold fits per step and nothing else.
        skyline_arms: **Supervised-skyline** arms to measure once per run
            (issue #3322; see :data:`SKYLINE_ARMS`).  ``None``/``[]`` (default)
            = off, and every other study runs exactly as before.  Requires
            ``emit_calibration_metrics``, because the arm exists to split that
            frame's ``oracle_cost`` into a learnability floor plus a
            ``training_regret``; skipped with a warning on a patch column, whose
            skyline needs a supervision decision this does not improvise (see
            :data:`SKYLINE_COLUMNS` and issue #3321).  Costs one extra fit per
            arm per run - the skyline is vote-independent - not one per step.
        autopilot_fidelity: When ``True`` (default) the simulated user follows
            the app's own phase machine
            (:class:`vtscore.eval.autopilot_flow.AutopilotFlow`): no detector is
            consulted before the Good/Bad quorum, Bad votes come from the text
            sort's cutoff, Hard picks are nearest-by-rank, and Hard → New → Done
            are driven by the smart/stable/span indicators rather than step
            parity.  ``False`` restores the older approximation so previously
            published studies reproduce byte-for-byte; see ``docs/EVAL.md``.
            Metrics are recorded at every trainable step in both modes — only
            the *vote order* and the ``app_trained`` flag differ.

    Returns:
        List of row dicts.  Keys: ``seed, dataset, category, strategy, trainer,
        head, style, prevalence_arm, realized_prevalence, t, n_good, n_bad, phase,
        app_trained, cost, fpr, fnr, auroc, average_precision, train_seconds,
        xcal_seconds, pool_score_seconds, test_score_seconds, backend, device,
        elapsed_seconds``.  ``n_good``/``n_bad`` report the vote counts behind
        each row so callers can tell apart metrics learned from a 1-vs-1 model
        and a many-vs-many one.  ``app_trained`` is 1 exactly when the app would
        have had a trained detector on screen at that step: a threshold recorded
        where it is 0 is one no user would ever see, which is what issue #2788's
        cold-start degenerates turned out to be.
    """
    import numpy as np  # noqa: PLC0415

    # One filter for the whole cell, before anything reads a label: on a
    # scale-banded dataset an image can hold the category at the wrong size,
    # and every "not positive means negative" test below would score it as a
    # negative.  A no-op for every dataset that does not designate its cells.
    clips_dict = evaluable_pool(clips_dict, target_category)

    rng = np.random.RandomState(seed)
    # Note: no torch.manual_seed() here - train_model handles its own
    # RNG seeding via fork_rng, keeping it thread-safe.
    start_time = time.monotonic()

    knobs = _resolve_run_knobs(
        fold_count_schedule=fold_count_schedule,
        calibrate_count=calibrate_count,
        startup_schedule=startup_schedule,
        seed_scores=seed_scores,
        autopilot_fidelity=autopilot_fidelity,
        strategy=strategy,
        skyline_arms=skyline_arms,
        emit_calibration_metrics=emit_calibration_metrics,
        acq_inclusion_offset=acq_inclusion_offset,
        acq_rank_percentile=acq_rank_percentile,
        head=head,
        trainer=trainer,
        style=style,
    )
    _fold_schedule = knobs.fold_schedule
    startup_state = knobs.startup_state
    skyline_arms = knobs.skyline_arms
    head = knobs.head

    prevalence_arm = "natural" if target_prevalence is None else f"rare_{target_prevalence:g}"
    if target_prevalence is not None:
        # Thin positives to the target prevalence *before* splitting, so both the
        # votable sim pool and the held-out test pool sit at that prevalence.
        downsampled = _downsample_to_prevalence(clips_dict, target_category, target_prevalence, rng)
        if downsampled is None:
            return []  # too few positives survive - skip this arm
        clips_dict = downsampled
    realized_prevalence = round(_prevalence(clips_dict, target_category), 6)

    sim_ids, test_ids = _split_media_ids(clips_dict, sim_fraction, rng)

    # Ensure the test set has both positive and negative medias.  Routes through
    # ``media_is_positive`` so multi-label (Visual Genome) images - where the
    # target may be a non-primary category - are counted correctly.
    test_pos = [cid for cid in test_ids if media_is_positive(clips_dict[cid], target_category)]
    test_neg = [cid for cid in test_ids if not media_is_positive(clips_dict[cid], target_category)]
    if not test_pos or not test_neg:
        return []

    # A patch dataset exposes a ``patch_grid`` per media; such datasets are
    # scored region-aware (max-pool over the image's score rows) the same way
    # the live detector scores them, regardless of how the Good votes were
    # assembled.
    region_aware = any(clips_dict[cid].get("patch_grid") is not None for cid in clips_dict)

    # `region_voting` is a request, not a guarantee: `good_training_vec` pools
    # the ground-truth box only when the media carries a stored `patch_grid`,
    # and falls back to the whole-image embedding otherwise - which is the same
    # condition `region_aware` above tests.  On a single-vector embedder that
    # fallback fires for EVERY vote, so the run is plain binary voting under a
    # flag that says otherwise, and it scores whole-image and blends under the
    # binary schedule too.  None of that shows up in the output: #2877 shipped a
    # report calling `visual_genome_m x siglip` a region-voting environment
    # before anyone checked, because the dataset is boxed and the harness config
    # said "region voting" next to its name.  Say so loudly.
    if region_voting and not region_aware:
        import warnings  # noqa: PLC0415

        warnings.warn(
            "region_voting=True but no media carries a patch_grid, so every Good "
            "vote falls back to its whole-image embedding: this run is BINARY "
            "voting. Region voting needs a patch embedder (e.g. dinov3_patch). "
            "See docs/experiments/2026-08-07-acquisition-inclusion/REPORT_SECOND_ENVIRONMENT.md.",
            RuntimeWarning,
            stacklevel=2,
        )

    # **The default arm must be the app's default.**  On a patch dataset the
    # live detector floods a Bad vote over the image's whole score-row stack
    # (``bad_negative_vecs``) and trains/calibrates bag-aware; the style-less
    # path here trains a Bad vote on one image-level row.  That gap predates
    # #2886 but MaxPatch widened it from 1-vs-24 to 1-vs-197 rows: the default
    # arm would train ~196 patch rows per rejected image down never, while
    # scoring max-pools all of them, so it would systematically under-suppress
    # and its numbers would not describe the shipped tool.  An eval default that
    # doesn't match the app default can't be trusted, so a patch dataset
    # defaults to the ``max_patch`` style - which *is* the production geometry
    # (its methods delegate to ``pool_box_from_media`` / ``bad_negative_vecs`` /
    # ``media_score_rows``).  The resolved name is recorded in the ``style``
    # column, so a result row always says which geometry produced it.
    #
    # Single-vector datasets are untouched: no patch grid, no style, and the
    # historical ``_mlp_train_and_calibrate`` path runs byte-for-byte.  Non-MLP
    # trainers are untouched too - they have no head for a style to drive.
    if style is None and region_aware and trainer == "mlp":
        style = PRODUCTION_PATCH_STYLE

    style_obj: Any = None
    if style is not None:
        from vtscore.eval.patch_styles import resolve_style  # noqa: PLC0415

        style_obj = resolve_style(style)

    # **v1 is the whole-image columns.**  Full supervision hands out *image*
    # labels, but the mortal max_patch flow trains on GT-box-pooled vectors (the
    # sim user drags a box), so a patch column's skyline is a design decision -
    # "oracle boxes + all images", or the multiple-instance problem of which
    # patch of a positive image is the positive - and not something to improvise
    # inside a metric row.  That decision is the named open item on #3321.  Skip
    # rather than raise: a sweep over `styles=["whole_image", "max_patch"]` should
    # get the decomposition on the column that has one instead of dying on the
    # column that doesn't, and the skip is loud here and visible in the frame
    # (no `skyline_*` rows, NaN decomposition columns) rather than silent.
    if skyline_arms and (style_obj is None or style_obj.name != _WHOLE_IMAGE_STYLE):
        import warnings  # noqa: PLC0415

        warnings.warn(
            f"skyline_arms={skyline_arms} skipped for style={style or 'none'!r}: the supervised "
            f"skyline is scoped to the {_WHOLE_IMAGE_STYLE!r} column in v1, because a patch "
            "column's skyline needs a supervision decision (GT boxes vs. multiple-instance) "
            "that is still open - see issue #3321.",
            RuntimeWarning,
            stacklevel=2,
        )
        skyline_arms = []
    blend_schedule, calibration_fraction = _resolve_production_defaults(
        blend_schedule=blend_schedule,
        calibration_fraction=calibration_fraction,
        region_aware=region_aware,
    )

    import torch  # noqa: PLC0415

    # Whole-image embeddings of the simulation pool.  These feed the autopilot
    # selector (the example-sort good centroid and the coverage atlas); the
    # Good-vote *training* vector can still be region-pooled below when
    # ``region_voting`` is on.
    sim_embeddings: dict[int, np.ndarray] = {
        cid: np.asarray(media_embedding(clips_dict[cid]), dtype=np.float32) for cid in sim_ids
    }
    input_dim = int(next(iter(sim_embeddings.values())).shape[0])

    # The autopilot New phase reads a coverage atlas built over the pool; it is
    # labelled in lock-step with the votes below so its coverage advances.
    atlas = _build_eval_atlas(sim_embeddings, atlas_min_node_size) if strategy == "autopilot" else None

    # Pre-compute embeddings for safe-threshold GMM scoring.  Restrict to the
    # simulation set so the held-out ``test_ids`` never feed into the GMM that
    # picks the threshold - otherwise the test scores leak into calibration
    # and the reported metrics are biased upward.  Region-aware datasets keep a
    # sim-set snapshot and score it per-step via region max-pool (to match how
    # the test set is scored); single-vector datasets pre-stack whole-image
    # embeddings once.
    # The snapshot is built for every region-aware / styled run, not only the
    # safe-threshold ones: the pool scorer needs it too, and it is a dict of
    # references to media already in memory.
    sim_clips: dict[int, dict[str, Any]] | None = None
    X_all_clips: Any = None
    if region_aware or style_obj is not None:
        sim_clips = {cid: clips_dict[cid] for cid in sim_ids}
    elif safe_thresholds:
        gmm_clip_embs = np.array([media_embedding(clips_dict[cid]) for cid in sorted(sim_ids)])
        X_all_clips = torch.tensor(gmm_clip_embs, dtype=torch.float32)

    # The #2799 safe-threshold variant rows additionally fit a GMM on the sim
    # set's *whole-image* scores (the historical pre-#2797 fit geometry), so
    # the whole-image matrix is pre-stacked once here.
    X_sim_image: "np.ndarray | None" = None
    if safe_thresholds and emit_calibration_metrics and style_obj is not None:
        X_sim_image = np.stack([sim_embeddings[cid] for cid in sorted(sim_ids)])

    good_votes: dict[int, None] = {}
    bad_votes: dict[int, None] = {}
    labeled: dict[int, float] = {}
    rows: list[dict[str, Any]] = []

    # Voting proceeds one item at a time: the autopilot selector picks the next
    # pool item using the *current* detector (trained at the previous step), the
    # item's ground-truth label is revealed, a fresh model is trained on all
    # votes so far, and the coverage atlas is labelled so its New-phase coverage
    # advances.  Before a trainable model exists the selector runs its seed/bad
    # phases (text sort or example sort), so a cold start still makes real picks.
    pool = sorted(sim_ids)
    # Ground-truth pool labels: autopilot draws its random known-good seed
    # examples from the positives here when no text sort is available.  Cheap to
    # build once up front.
    pool_labels = {cid: (1.0 if media_is_positive(clips_dict[cid], target_category) else 0.0) for cid in sim_ids}
    step: StepModel | None = None
    threshold = 0.5
    #: The selector's threshold - cut ``acq_inclusion_offset`` steps below the
    #: reporting one.  Kept as its own name so the two jobs cannot silently
    #: re-merge (they were one variable, and that is how the #2847 positives
    #: regression got in).
    acq_threshold = 0.5
    pool_scores: dict[int, float] = {}
    n_steps = len(pool) if max_steps is None else min(max_steps, len(pool))

    # The app's phase machine, driving the vote order the way Autopilot does.
    # Disabled (``None``) under ``autopilot_fidelity=False``, which leaves the
    # selector on its legacy parity interleave.
    flow: Any = None
    if autopilot_fidelity and strategy == "autopilot":
        flow = AutopilotFlow(startup=startup_state)
    # Each schedule round's cut on the seed sort, resolved once: the app fits a
    # cosine sort's GMM over the whole sort and never refits it as votes come
    # in, so these are constants of the run rather than per-step state.
    startup_cuts: list[float] = []
    if startup_state is not None and seed_scores is not None:
        sort_values = list(seed_scores.values())
        startup_cuts = [round_cut(sort_values, rnd) for rnd in startup_state.rounds]
    # The seed sort as a ranking, for the pick log: where in the sort each click
    # landed is the mining record the study reads.
    seed_rank: dict[int, int] = {}
    if pick_sink is not None and seed_scores is not None:
        seed_rank = {cid: i for i, cid in enumerate(sorted(seed_scores, key=lambda c: seed_scores[c], reverse=True))}
    seed_sorted_scores: list[float] = sorted(seed_scores.values(), reverse=True) if seed_scores else []
    # Recent per-step models (each with the threshold it was calibrated at),
    # re-scored every step against the *current* labelset so the Smart
    # indicator's slope regresses over one shared eval set - exactly what the
    # app's ``_eval_cached_models`` does over its per-step cache.
    recent_steps: list[tuple[Any, float]] = []

    for t in range(1, n_steps + 1):
        if not pool:
            break
        phase = flow.phase if flow is not None else None
        startup_round = startup_state.index if (startup_state is not None and not startup_state.done) else -1
        startup_cut = startup_cuts[startup_round] if startup_round >= 0 else None
        ctx = ALContext(
            pool_ids=pool,
            embeddings=sim_embeddings,
            labeled=labeled,
            scores=pool_scores,
            model=step,
            # The ONLY consumer that moves.  Reporting, the metric rows and the
            # phase machine all stay on ``threshold``.
            threshold=acq_threshold,
            atlas=atlas,
            rng=rng,
            pool_labels=pool_labels,
            seed_scores=seed_scores,
            phase=phase,
            startup_cut=startup_cut,
        )
        cid = select_next(strategy, ctx)
        pool.remove(cid)
        is_positive = media_is_positive(clips_dict[cid], target_category)
        if is_positive:
            good_votes[cid] = None
            labeled[cid] = 1.0
        else:
            bad_votes[cid] = None
            labeled[cid] = 0.0
        # Mirror the vote onto the coverage atlas so the New phase's next_sample
        # advances past covered regions (the app labels the atlas the same way).
        if atlas is not None and cid in atlas.vector_to_leaf:
            atlas.label(cid, good=is_positive)

        if pick_sink is not None:
            rank = seed_rank.get(cid, -1)
            n_sorted = len(seed_rank)
            pick_sink.append(
                {
                    "seed": seed,
                    "dataset": dataset_name,
                    "category": target_category,
                    "startup_schedule": startup_schedule or "",
                    "style": style or "",
                    "t": t,
                    "phase": phase or "",
                    "startup_round": startup_round,
                    "startup_held": bool(startup_state.held_for_quorum) if startup_state is not None else False,
                    "startup_extended_clicks": int(startup_state.extended_clicks) if startup_state is not None else 0,
                    "startup_cut": round6(startup_cut) if startup_cut is not None else float("nan"),
                    "startup_cut_percentile": (
                        _sorted_percentile(seed_sorted_scores, startup_cut) if startup_cut is not None else float("nan")
                    ),
                    "picked_id": cid,
                    "picked_label": 1 if is_positive else 0,
                    "picked_seed_rank": rank,
                    "picked_seed_percentile": (
                        round6(rank / (n_sorted - 1)) if n_sorted > 1 and rank >= 0 else float("nan")
                    ),
                    "picked_seed_score": round6(seed_scores[cid])
                    if seed_scores and cid in seed_scores
                    else float("nan"),
                    "picked_detector_score": round6(pool_scores[cid]) if cid in pool_scores else float("nan"),
                    "acq_threshold": round6(acq_threshold),
                    "n_good": len(good_votes),
                    "n_bad": len(bad_votes),
                    "n_pool": len(pool),
                }
            )

        n_votes_now = len(good_votes) + len(bad_votes)
        # Need at least 1 good and 1 bad to train
        if not good_votes or not bad_votes:
            step = None
            # The phase still advances - the app's Good phase ends on its third
            # positive whether or not a detector could be trained, and without
            # this the flow would never leave ``good`` and the run would vote
            # positives forever.
            if flow is not None:
                flow.update(
                    len(good_votes),
                    len(bad_votes),
                    remaining_unlabeled=len(pool),
                    span=atlas.span_info() if atlas is not None else None,
                )
            continue

        # The live fold count for THIS step.  Constant unless #3314's schedule
        # knob is set, and then a function of the vote count only - never of
        # anything the step has already computed, so the count is decided
        # before the fit and cannot depend on it.
        step_calibrate_count = calibrate_count if _fold_schedule is None else _fold_schedule(n_votes_now)
        step, threshold, n_labels, timings, details = _train_and_calibrate(
            trainer,
            good_votes,
            bad_votes,
            clips_dict,
            target_category,
            region_voting=region_voting,
            input_dim=input_dim,
            inclusion=inclusion,
            calibrate_count=step_calibrate_count,
            calibration_fraction=calibration_fraction,
            head=head,
            style_obj=style_obj,
            emit_calibration_metrics=emit_calibration_metrics,
            fold_count_variants=fold_count_variants,
        )

        # Apply the shipped safe threshold if enabled
        sim_pooled_scores: list[float] | None = None
        sim_pooled_ids: list[int] = []
        sim_fold_haystacks: list[Any] = []
        if safe_thresholds:
            # The x-cal side of the blend, not the raw fold return: a step whose
            # folds fell back blends NO_GOOD_THRESHOLD (see _blend_xcal_input),
            # and the variant families below re-blend this same input, so their
            # rows stay paired with the shipped one.
            xcal_threshold = _blend_xcal_input(threshold, details)
            # Vote-level class counts, so the fallback blend's schedule can ramp
            # on the rarer class (#2841).  The harness votes one media at a
            # time, so bags and votes coincide here and the counts are the two
            # vote dicts' sizes.
            blend_ctx = BlendContext(n_labels=n_labels, n_good=len(good_votes), n_bad=len(bad_votes))
            threshold, sim_pooled_scores, sim_pooled_ids, sim_fold_haystacks, safe_provenance, safe_cut = (
                _safe_threshold_for_step(
                    threshold,
                    step,
                    details,
                    region_aware,
                    sim_clips,
                    X_all_clips,
                    blend_ctx,
                    sim_ids,
                    inclusion,
                    style_obj=style_obj,
                    schedule=blend_schedule,
                    voted_ids=set(good_votes) | set(bad_votes),
                    exclusion_min_remainder=exclusion_min_remainder,
                )
            )
            if emit_calibration_metrics:
                details["pre_blend_provenance"] = details.get("provenance", "conformal")
                details["provenance"] = safe_provenance
                details["xcal_threshold"] = xcal_threshold
                details["n_votes"] = n_labels
                details["n_good"] = len(good_votes)
                details["n_bad"] = len(bad_votes)

        # The selector's cut.  Recomputed from scratch every step - never
        # carried over - so a step with nothing to re-cut falls back to the
        # reporting threshold rather than sampling this step's scores against
        # the last step's cut.
        acq_threshold = threshold
        if safe_thresholds:
            if acq_rank_percentile is not None:
                if sim_pooled_scores:
                    acq_threshold = float(
                        np.quantile(np.asarray(sim_pooled_scores, dtype=np.float64), acq_rank_percentile)
                    )
            elif acq_inclusion_offset != 0 and safe_cut is not None:
                # Re-cut the *same* fold-anchored fit.  O(1) - the mixture was
                # fitted above; ``threshold_at`` is monotone by construction, so
                # the arms are nested and offset 0 reproduces the reporting cut
                # exactly.  ``safe_cut is None`` is the schedule-blend fallback
                # (~5% of steps, concentrated in the cold start): the blend has
                # no inclusion-aware form, so there is nothing honest to re-cut.
                cand = safe_cut.threshold_at(acquisition_inclusion(inclusion, acq_inclusion_offset))
                if np.isfinite(cand):
                    acq_threshold = float(cand)

        # Evaluate on the held-out test set.  The calibration study (#2781)
        # emits one row per pooling (base + remedial) instead of the single
        # metrics row, but both paths score the same test set here.
        calibration: tuple[list[dict[str, Any]], np.ndarray, np.ndarray] | None = None
        metrics: dict[str, float] = {}
        t_test = time.monotonic()
        if emit_calibration_metrics and style_obj is not None:
            calibration = _calibration_metric_rows(
                step,
                threshold,
                details,
                clips_dict,
                test_ids,
                target_category,
                inclusion,
                style_obj,
                repool_variants or [],
                repool_topk,
            )
        else:
            metrics = _evaluate_on_test(
                step,
                threshold,
                clips_dict,
                test_ids,
                target_category,
                inclusion,
                region_aware=region_aware,
                style_obj=style_obj,
            )
        test_score_seconds = time.monotonic() - t_test

        # Score the remaining pool with the fresh model so the next step's
        # autopilot Hard pick can rank it - in the geometry the cut it will be
        # compared against was fitted in (#2943).  The safe-threshold path has
        # already scored the whole sim set that way, so hand those scores over
        # rather than paying for a second pass.
        t_pool = time.monotonic()
        pool_scores = _score_pool(
            step,
            pool,
            clips_dict,
            region_aware=region_aware,
            style_obj=style_obj,
            sim_clips=sim_clips,
            sim_scored=(sim_pooled_ids, sim_pooled_scores) if sim_pooled_scores else None,
        )
        pool_score_seconds = time.monotonic() - t_pool

        # Advance the app's phase machine on this step's model: the Smart
        # indicator needs the labelset error cost, Stable the prediction flips
        # over the still-unlabeled pool, Span the atlas's coverage.
        if flow is not None:
            recent_steps.append((step, threshold))
            # The app regresses over its last SMART_WINDOW *models*; here every
            # step trains one, so the last SMART_WINDOW steps are the same set.
            del recent_steps[:-SMART_WINDOW]
            flow.record_step(
                _labelset_error_costs(
                    recent_steps,
                    good_votes,
                    bad_votes,
                    clips_dict,
                    inclusion,
                    region_aware=region_aware,
                    style_obj=style_obj,
                ),
                {cid: (1 if s >= threshold else 0) for cid, s in pool_scores.items()},
            )
            flow.update(
                len(good_votes),
                len(bad_votes),
                remaining_unlabeled=len(pool),
                span=atlas.span_info() if atlas is not None else None,
            )

        # Identifying columns shared by every row this step emits.
        base_row = {
            "seed": seed,
            "dataset": dataset_name,
            "category": target_category,
            "strategy": strategy,
            "trainer": trainer,
            # Blank on the standalone SVM trainers: they fit no head, so
            # naming one here would attribute the row to a head never trained.
            "head": head if trainer == "mlp" else "",
            "style": style or "",
            "prevalence_arm": prevalence_arm,
            "realized_prevalence": realized_prevalence,
            "t": t,
            "n_good": len(good_votes),
            "n_bad": len(bad_votes),
            # The haystack the threshold was fitted on, and what is left of it
            # after this step's votes.  `n_remainder` is *exactly* the quantity
            # the #3308 exclusion floor is compared against
            # (`apply_vote_exclusion` counts the unvoted scores), and `pool` has
            # already had this step's vote removed by the time this row is
            # built - so an analyzer can reconstruct, per step, whether the
            # exclusion fired, without the harness having to report it (#3312).
            # Their ratio is the axis the mechanism runs on: the effect is
            # bounded by the votes' share of the haystack.
            "n_haystack": len(sim_ids),
            "n_remainder": len(pool),
            "phase": flow.phase if flow is not None else "",
            # The three lights behind that phase (#3560).  Already computed by
            # `flow.update` above and previously discarded; the phase alone
            # cannot say whether Smart or Stable is what holds a run in `hard`.
            "smart": flow.smart if flow is not None else "",
            "stable": flow.stable if flow is not None else "",
            "span": flow.span if flow is not None else "",
            "span_level": flow.span_level if flow is not None else -1,
            "span_depth": flow.span_depth if flow is not None else -1,
            "app_trained": 1 if (flow is None or app_has_detector(flow.phase)) else 0,
            "startup_schedule": startup_schedule or "",
            "acq_threshold": round(float(acq_threshold), 6),
            # Measured against the pool the selector ranks, not the test set, so
            # the pair answers "how much did the sampling position move".
            "acq_pool_percentile": _pool_percentile(pool_scores, acq_threshold),
            "report_pool_percentile": _pool_percentile(pool_scores, threshold),
        }
        timing_cols = {
            # The fold count this step actually LIVED at.  Constant on every run
            # but #3314's scheduled arm - and recorded regardless, because a
            # run-level knob whose only other record is the directory the cells
            # were read out of is unreadable in a frame concatenated across arms
            # (#3287's `calibration_fraction` lesson, one knob over).
            "calibrate_count": step_calibrate_count,
            "train_seconds": round(timings["train_seconds"], 6),
            #: The final model's own pass over the haystack, on the shipped
            #: safe-threshold path (#3314).  NaN when safe thresholds are off,
            #: where there is no such pass.  K-independent, so a per-step cost
            #: ratio wants it in the denominator.
            "final_score_seconds": round(float(details.get("final_score_seconds", float("nan"))), 6),
            "xcal_seconds": round(timings["xcal_seconds"], 6),
            "pool_score_seconds": round(pool_score_seconds, 6),
            "test_score_seconds": round(test_score_seconds, 6),
            "backend": step.backend,
            "device": step.device,
            "elapsed_seconds": round(time.monotonic() - start_time, 3),
        }

        if calibration is not None:
            metric_rows, base_scores, base_labels = calibration
            # The final model's haystack under the #3308 population convention:
            # the voted items dropped, exactly as `_safe_threshold_for_step`
            # dropped them from the fold haystacks - so every fold-anchored
            # variant fit below stays paired with the shipped cut's population.
            # The same floor applies, via the same decision function, so a
            # variant grid can never sit on a different population than the
            # shipped cut it is measured against.
            _voted_step_ids = set(good_votes) | set(bad_votes)
            sim_fit_scores: list[float] | None = None
            if sim_pooled_scores is not None:
                _fit_arr, _ = apply_vote_exclusion(
                    sim_pooled_scores,
                    sim_pooled_ids,
                    _voted_step_ids,
                    min_remainder=exclusion_min_remainder,
                )
                sim_fit_scores = _fit_arr.tolist()
            # One extra row per safe-threshold GMM variant (issue #2799), all
            # evaluated against the same held-out max-pooled test scores.
            if X_sim_image is not None and sim_pooled_scores is not None:
                sim_image_ids = sorted(sim_ids)
                sim_image_scores = np.asarray(step.predict(X_sim_image)).ravel().tolist()
                variant_rows, diag_rows = _safe_gmm_variant_rows(
                    details,
                    base_scores,
                    base_labels,
                    {"pooled": sim_pooled_scores, "image": sim_image_scores},
                    {
                        "pooled": np.array([pool_labels[cid] for cid in sim_pooled_ids], dtype=np.float64),
                        "image": np.array([pool_labels[cid] for cid in sim_image_ids], dtype=np.float64),
                    },
                    inclusion,
                    n_pool_rows=metric_rows[0]["n_pool_rows"],
                    schedule=blend_schedule,
                )
                metric_rows.extend(variant_rows)
                if cut_diag_sink is not None:
                    for dr in diag_rows:
                        cut_diag_sink.append({**base_row, **dr})
            # The #3329 goodness-of-fit frame.  Deliberately NOT nested in the
            # block above: that one needs `X_sim_image`, which only a region run
            # has, and the binary control arm is half of this study's design -
            # gating on it would have emitted nothing for exactly the arm the
            # max-pooling hypothesis is contrasted against.  The `image`
            # geometry rides along only where it exists.
            if (
                fit_quality_sink is not None
                and sim_pooled_scores is not None
                and (t <= 3 or t % max(1, fit_quality_stride) == 0)
            ):
                fq_scores: dict[str, Any] = {"pooled": sim_pooled_scores}
                fq_labels: dict[str, Any] = {
                    "pooled": np.array([pool_labels[cid] for cid in sim_pooled_ids], dtype=np.float64)
                }
                if X_sim_image is not None:
                    fq_image_ids = sorted(sim_ids)
                    fq_scores["image"] = np.asarray(step.predict(X_sim_image)).ravel().tolist()
                    fq_labels["image"] = np.array([pool_labels[cid] for cid in fq_image_ids], dtype=np.float64)
                fit_quality_sink.extend(_fit_quality_rows(base_row, safe_cut, fq_scores, fq_labels, threshold))
            # One extra row per mix-in schedule (issue #2841), on the production
            # cut.  Independent of the cut-variant rows above: the schedule
            # screen only needs the pooled sim scores the blend actually fits.
            if schedule_variants and sim_pooled_scores is not None:
                metric_rows.extend(
                    _schedule_variant_rows(
                        details,
                        base_scores,
                        base_labels,
                        sim_pooled_scores,
                        inclusion,
                        n_pool_rows=metric_rows[0]["n_pool_rows"],
                        schedules=schedule_variants,
                    )
                )
            # The #2897 fold-count arms.  Unlike the arms above these need no
            # sim scores of their own - they re-cut fold orderings the step
            # already trained - so they run whether or not safe_thresholds is on;
            # the pooled sim scores, when present, only add the blended arm.
            if fold_count_variants:
                metric_rows.extend(
                    _fold_count_variant_rows(
                        details,
                        base_scores,
                        base_labels,
                        inclusion,
                        n_pool_rows=metric_rows[0]["n_pool_rows"],
                        counts=fold_count_variants,
                        sim_pooled_scores=sim_pooled_scores,
                        schedule=blend_schedule,
                        sim_fit_scores=sim_fit_scores,
                    )
                )
            # The #2852 anchored-mixture arms, paired against the same test
            # scores (and against pooled_mid / xcal_only above).
            if anchored_thresholds and sim_pooled_scores is not None:
                metric_rows.extend(
                    _anchored_variant_rows(
                        details,
                        base_scores,
                        base_labels,
                        sim_pooled_scores,
                        sim_pooled_ids,
                        list(good_votes),
                        list(bad_votes),
                        sim_fold_haystacks,
                        inclusion,
                        n_pool_rows=metric_rows[0]["n_pool_rows"],
                        weights=anchored_weights if anchored_weights is not None else list(_ANCHORED_WEIGHTS),
                        rules=anchored_rules if anchored_rules is not None else list(_ANCHORED_RULES),
                        fold_combines=(
                            anchored_fold_combines
                            if anchored_fold_combines is not None
                            else list(_ANCHORED_FOLD_COMBINES)
                        ),
                        fold_anchored=anchored_fold_arms,
                        sim_fit_scores=sim_fit_scores,
                    )
                )
            for mr in metric_rows:
                rows.append({**base_row, **mr, **timing_cols})
            # The near-free inclusion-budget sweep, into the side sink.
            if inclusion_sweep_ks and sweep_sink is not None:
                for sr in _inclusion_sweep_rows(details, base_scores, base_labels, inclusion_sweep_ks):
                    sweep_sink.append({**base_row, **sr})
            # The #2865 cut-rule x inclusion sweep, into its own side sink.
            # Needs the per-fold sim scores the fold-anchored arms use, so it
            # rides the same `sim_pooled_scores is not None` gate they do.
            if cut_inclusion_ks and cut_inclusion_sink is not None and sim_pooled_scores is not None:
                for cr in _cut_inclusion_rows(
                    details,
                    base_scores,
                    base_labels,
                    sim_fold_haystacks,
                    sim_fit_scores if sim_fit_scores is not None else sim_pooled_scores,
                    cut_inclusion_ks,
                    weights=anchored_weights if anchored_weights is not None else list(_ANCHORED_WEIGHTS),
                    rules=anchored_rules if anchored_rules is not None else list(_ANCHORED_RULES),
                    fold_combines=(
                        anchored_fold_combines if anchored_fold_combines is not None else list(_ANCHORED_FOLD_COMBINES)
                    ),
                    qtilt_steps=(
                        cut_inclusion_qtilt_steps if cut_inclusion_qtilt_steps is not None else [FOLD_ANCHOR_QTILT_STEP]
                    ),
                ):
                    cut_inclusion_sink.append({**base_row, **cr})
        else:
            rows.append({**base_row, **metrics, **timing_cols})

    # --- The supervised skyline (issue #3322), once per run. ---
    #
    # Deliberately **after** the loop rather than before it: every fit here draws
    # from some RNG somewhere, and a skyline computed up front would have to be
    # proved not to perturb the trajectory it is meant to describe.  Computed
    # here it cannot, whatever the trainer does internally - the votes are
    # already cast.  `test_skyline_does_not_perturb_the_trajectory` pins that.
    if skyline_arms:
        skyline_rows = _skyline_arm_rows(
            skyline_arms,
            clips_dict,
            target_category,
            sim_ids,
            test_ids,
            inclusion,
            trainer=trainer,
            head=head,
            style_obj=style_obj,
            region_voting=region_voting,
            input_dim=input_dim,
            calibrate_count=calibrate_count,
            calibration_fraction=calibration_fraction,
            seed=seed,
        )
        _apply_skyline_decomposition(rows, skyline_rows)
        # `t=0` and `app_trained=0`: the skyline belongs to no step, so it is
        # given a step index no trajectory can occupy (the first *trainable*
        # step is t>=2) rather than being duplicated onto all of them.  The vote
        # counts are the full supervision it was handed, and `n_remainder=0`
        # says exactly what that means - nothing in the haystack was left
        # unlabelled.
        n_sky_good = sum(1 for cid in sim_ids if pool_labels[cid] == 1.0)
        skyline_ident = {
            "seed": seed,
            "dataset": dataset_name,
            "category": target_category,
            "strategy": strategy,
            "trainer": trainer,
            "head": head if trainer == "mlp" else "",
            "style": style or "",
            "prevalence_arm": prevalence_arm,
            "realized_prevalence": realized_prevalence,
            "t": 0,
            "n_good": n_sky_good,
            "n_bad": len(sim_ids) - n_sky_good,
            "n_haystack": len(sim_ids),
            "n_remainder": 0,
            "phase": "",
            # A skyline belongs to no step, so no phase ran and no indicator
            # was ever read for it (#3560).  Blank / -1 is the "not measured"
            # spelling every other unphased row uses; a `red` here would say
            # the rules had looked and refused, which they never did.
            "smart": "",
            "stable": "",
            "span": "",
            "span_level": -1,
            "span_depth": -1,
            "app_trained": 0,
            "startup_schedule": startup_schedule or "",
            "acq_threshold": float("nan"),
            "acq_pool_percentile": float("nan"),
            "report_pool_percentile": float("nan"),
        }
        rows.extend({**skyline_ident, **sr} for sr in skyline_rows)

    return rows


# ------------------------------------------------------------------
# Full evaluation across seeds x datasets x categories
# ------------------------------------------------------------------


def run_voting_iterations_eval(
    dataset_clips: dict[str, dict[int, dict[str, Any]]],
    seeds: list[int],
    categories: Optional[dict[str, list[str]]] = None,
    inclusion: int = 0,
    sim_fraction: float = 0.5,
    safe_thresholds: bool = True,
    calibrate_count: int = 2,
    calibration_fraction: Optional[float] = None,
    region_voting: bool = False,
    strategies: Optional[list[str]] = None,
    max_steps: Optional[int] = None,
    atlas_min_node_size: int = 20,
    seed_scores: Optional[dict[str, dict[str, dict[int, float]]]] = None,
    trainers: Optional[list[str]] = None,
    prevalence_arms: Optional[list[Optional[float]]] = None,
    styles: Optional[list[Optional[str]]] = None,
    autopilot_fidelity: bool = True,
    startup_schedule: Optional[str] = None,
) -> pd.DataFrame:
    """Run the voting-iterations evaluation over multiple seeds/datasets/categories.

    Args:
        dataset_clips: Mapping of dataset name to a pre-loaded medias dict.
            Each medias dict maps ``int`` media IDs to media data dicts
            (must carry a resolvable embedding in the per-embedder
            ``"embeddings"`` store and a ``"category"`` key).
        seeds: List of random seeds to iterate over.
        categories: Optional mapping of dataset name to list of target
            categories.  If ``None`` or a dataset is missing from the dict,
            all unique categories in that dataset are used.
        inclusion: Inclusion setting in ``[-10, 10]``.
        sim_fraction: Fraction of medias reserved for simulated voting.
        safe_thresholds: The shipped fused threshold path; on by default,
            matching the app.  ``False`` is the no-fusion control arm.
            (see :func:`simulate_voting_iterations`).
        calibrate_count: Number of random Train/Calibrate splits for threshold
            calibration (default 2).
        calibration_fraction: Fraction of labelled data reserved for
            calibration in each split.  ``None`` (default) resolves per
            dataset to the app's per-space split (see
            :func:`simulate_voting_iterations`).
        region_voting: When ``True``, Good votes train on the ground-truth
            region-pooled vector for patch datasets (see
            :func:`simulate_voting_iterations`).
        strategies: Vote-order strategies to run (see
            :data:`vtscore.eval.al_strategies.STRATEGIES`).  ``None`` (default)
            runs ``["autopilot"]``, the only strategy; the name is recorded in
            the ``strategy`` result column.
        max_steps: Cap on the number of voting steps per run (see
            :func:`simulate_voting_iterations`).
        atlas_min_node_size: Minimum coverage-atlas leaf population for the
            autopilot New phase (see :func:`simulate_voting_iterations`).
        seed_scores: Optional text-sort rankings keyed
            ``{dataset: {category: {media_id: similarity}}}``.  When a
            (dataset, category) has an entry, the autopilot seed follows that
            text ranking; otherwise it seeds from random known-good examples.
        trainers: Which rankers to run at each cell (see
            :func:`simulate_voting_iterations`).  ``None`` (default) runs
            ``["mlp"]``; pass e.g. ``["mlp", "svm_linear", "svm_rbf"]`` for the
            head-to-head comparison.  Recorded in the ``trainer`` column.
        prevalence_arms: Which prevalence arms to run per (dataset, category).
            ``None`` (default) runs ``[None]`` (natural prevalence only); pass
            e.g. ``[None, 0.01]`` to add the 1%-prevalence rare arm.  Recorded
            in the ``prevalence_arm`` / ``realized_prevalence`` columns.
        styles: Which detection styles to run per cell (see
            :func:`simulate_voting_iterations`).  ``None`` (default) runs
            ``[None]``, which resolves per dataset to whatever the **app** does
            - ``max_patch`` on a patch dataset, the single-vector path
            otherwise; pass e.g. ``["whole_image", "max_patch"]`` to pin the
            Max-Patch experiment arms explicitly.  The *resolved* name is
            recorded in the ``style`` column (``""`` only when no style ran).
        autopilot_fidelity: Follow the app's own Autopilot phase machine
            (default ``True``); see :func:`simulate_voting_iterations`.  Pass
            ``False`` to reproduce studies published before the flow was
            aligned.
        startup_schedule: A parameterised Autopilot opening (issue #3267); see
            :func:`simulate_voting_iterations`.  ``None`` (default) is the app's
            own opening.  Requires a *seed_scores* entry for every cell run.

    Returns:
        A :class:`~pandas.DataFrame` with the columns listed in
        :data:`VOTING_COLUMNS`.
    """
    import pandas as pd  # noqa: PLC0415

    strategy_list = strategies if strategies is not None else ["autopilot"]
    trainer_list = trainers if trainers is not None else ["mlp"]
    arm_list = prevalence_arms if prevalence_arms is not None else [None]
    style_list = styles if styles is not None else [None]
    all_rows: list[dict[str, Any]] = []

    for ds_name, clips_dict in dataset_clips.items():
        # Determine target categories.  For multi-label datasets each image's
        # ``category`` is only its primary, so fall back to the union of every
        # image's ``categories`` list when present.
        if categories and ds_name in categories:
            target_cats = categories[ds_name]
        else:
            cat_set: set[str] = set()
            for cid in clips_dict:
                media = clips_dict[cid]
                cat_set.update(media.get("categories") or [media["category"]])
            target_cats = sorted(cat_set)

        for seed in seeds:
            for cat in target_cats:
                cat_seed_scores = (seed_scores or {}).get(ds_name, {}).get(cat)
                for arm in arm_list:
                    for strategy in strategy_list:
                        for trainer in trainer_list:
                            for style in style_list:
                                rows = simulate_voting_iterations(
                                    clips_dict,
                                    target_category=cat,
                                    seed=seed,
                                    dataset_name=ds_name,
                                    inclusion=inclusion,
                                    sim_fraction=sim_fraction,
                                    safe_thresholds=safe_thresholds,
                                    calibrate_count=calibrate_count,
                                    calibration_fraction=calibration_fraction,
                                    region_voting=region_voting,
                                    strategy=strategy,
                                    max_steps=max_steps,
                                    atlas_min_node_size=atlas_min_node_size,
                                    seed_scores=cat_seed_scores,
                                    trainer=trainer,
                                    target_prevalence=arm,
                                    style=style,
                                    autopilot_fidelity=autopilot_fidelity,
                                    startup_schedule=startup_schedule,
                                )
                                all_rows.extend(rows)

    return pd.DataFrame(all_rows, columns=pd.Index(list(VOTING_COLUMNS)))


def run_voting_iterations_eval_from_pickles(
    dataset_paths: dict[str, str],
    seeds: list[int],
    categories: Optional[dict[str, list[str]]] = None,
    inclusion: int = 0,
    sim_fraction: float = 0.5,
    safe_thresholds: bool = True,
    calibrate_count: int = 2,
    calibration_fraction: Optional[float] = None,
    region_voting: bool = False,
    strategies: Optional[list[str]] = None,
    max_steps: Optional[int] = None,
    atlas_min_node_size: int = 20,
    seed_scores: Optional[dict[str, dict[str, dict[int, float]]]] = None,
    trainers: Optional[list[str]] = None,
    prevalence_arms: Optional[list[Optional[float]]] = None,
    styles: Optional[list[Optional[str]]] = None,
    autopilot_fidelity: bool = True,
    startup_schedule: Optional[str] = None,
) -> pd.DataFrame:
    """Convenience wrapper that loads datasets from pickle files.

    Args:
        dataset_paths: Mapping of dataset name to pickle file path.
        seeds: List of random seeds.
        categories: Optional category filter (see :func:`run_voting_iterations_eval`).
        inclusion: Inclusion setting in ``[-10, 10]``.
        sim_fraction: Fraction of medias for simulation.
        safe_thresholds: The shipped fused threshold path; on by default,
            matching the app.  ``False`` is the no-fusion control arm.
        calibrate_count: Number of random Train/Calibrate splits for threshold
            calibration (default 2).
        calibration_fraction: Fraction of labelled data reserved for
            calibration in each split.  ``None`` (default) resolves per
            dataset to the app's per-space split (see
            :func:`simulate_voting_iterations`).
        region_voting: When ``True``, Good votes train on the ground-truth
            region-pooled vector for patch datasets (see
            :func:`simulate_voting_iterations`).
        strategies: Vote-order strategies to run (see
            :func:`run_voting_iterations_eval`).
        max_steps: Cap on the number of voting steps per run.
        atlas_min_node_size: Minimum coverage-atlas leaf population for the
            autopilot New phase.
        seed_scores: Optional text-sort rankings keyed
            ``{dataset: {category: {media_id: similarity}}}`` (see
            :func:`run_voting_iterations_eval`).

    Returns:
        A :class:`~pandas.DataFrame` identical to :func:`run_voting_iterations_eval`
        (columns: ``seed, dataset, category, strategy, t, n_good, n_bad, cost,
        fpr, fnr, elapsed_seconds``).
    """
    from vtscore.datasets.loader import load_dataset_from_pickle

    dataset_clips: dict[str, dict[int, dict[str, Any]]] = {}
    for name, path in dataset_paths.items():
        medias: dict[int, dict[str, Any]] = {}
        load_dataset_from_pickle(Path(path), medias)
        dataset_clips[name] = medias

    return run_voting_iterations_eval(
        dataset_clips,
        seeds=seeds,
        categories=categories,
        inclusion=inclusion,
        sim_fraction=sim_fraction,
        safe_thresholds=safe_thresholds,
        calibrate_count=calibrate_count,
        calibration_fraction=calibration_fraction,
        region_voting=region_voting,
        strategies=strategies,
        max_steps=max_steps,
        atlas_min_node_size=atlas_min_node_size,
        seed_scores=seed_scores,
        trainers=trainers,
        prevalence_arms=prevalence_arms,
        styles=styles,
        autopilot_fidelity=autopilot_fidelity,
        startup_schedule=startup_schedule,
    )
