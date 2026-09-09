"""Detector training helpers: validate, train, threshold, serialise.

Consolidates the repeated train → calibrate → fuse → serialise pipeline used
by detector route handlers and test helpers.

This module also holds the vote-aware detector training entry points -
:func:`train_and_score` (online, called every time the user toggles a
vote) and :func:`train_detector_from_origins` (the **library-tier**
re-derivation of an MLP from a saved labelset, for embedders driving
``vtscore`` directly; the app does not call it - see that function's
"Who calls this" note). Both build on the generic
:mod:`vtscore.training` primitives but layer in the patch-region max-
pool and origin-based file resolution that are detector-specific.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from vtscore.embedding.precomputed import stack_vectors
from vtscore.utils.scores import sigmoid_to_finite_array

if TYPE_CHECKING:
    import torch.nn as nn


log = logging.getLogger(__name__)


def _score_embedder_for_snap(snap: dict[int, dict[str, Any]] | None) -> str | None:
    """Resolve the score embedder (structural ▸ patch ▸ text) for the medias in *snap*.

    The detector MLP trains and scores against this embedder's vectors (the v3
    routing table).  Derived from the embedder names *those* medias carry
    rather than the active context, since :func:`_score_all_media` /
    :func:`_build_vote_xy` may be handed an arbitrary snapshot (a test fixture,
    a cross-dataset dict).  Returns ``None`` for a slot-less single-vector
    dataset (e.g. ``dinov2_single``) or medias whose embedder is unregistered,
    so the matrix layer falls back to each media's primary vector.  For a
    single-embedder dataset the resolved name equals the primary, which the
    matrix layer collapses to the cached primary path.

    This is deliberately **not** :func:`~vtscore.embedding.binding.score_marker_embedder_for_snap`:
    that one falls back to the media's primary *name* and returns ``""``, where
    the matrix layer here wants ``None`` to mean "no bound slot, read each
    media's primary vector".
    """
    from vtscore.embedding.binding import slot_embedders_for_snap  # noqa: PLC0415

    text, patch, structural = slot_embedders_for_snap(snap)
    return structural or patch or text


def detector_score_embedder(det_ctx: Any, snap: dict[int, dict[str, Any]] | None) -> str | None:
    """Embedder a *detector* trains and scores in.

    The concrete embedder of the detector's locked type
    (``det_ctx.embedder_type``) that the snap supplies wins; otherwise the
    dataset score precedence - the legacy-migration default and the
    cross-dataset portability fallback (a detector pointed at a dataset that
    lacks its type re-embeds against that dataset's space).

    This is :func:`~vtscore.embedding.binding.keying_embedder_for_snap` with the
    empty marker normalised to ``None``, which is the *only* reason the name
    exists: the training and scoring paths read ``None`` as "no explicit
    primary, fall back to each media's own vector", so the resolver's ``""``
    for an empty snap has to become ``None`` before it reaches them.  Kept as a
    public name because out-of-tree code may import it.  See
    ``docs/plans/patch-embedder.md`` → "Per-detector embedder type".
    """
    from vtscore.embedding.binding import keying_embedder_for_snap  # noqa: PLC0415

    return keying_embedder_for_snap(det_ctx, snap) or None


def _patch_embedder_for_snap(snap: dict[int, dict[str, Any]] | None) -> str | None:
    """Resolve the patch-slot embedder name for *snap*'s medias, or ``None``."""
    from vtscore.embedding.binding import slot_embedders_for_snap  # noqa: PLC0415

    return slot_embedders_for_snap(snap)[1]


def _scores_in_patch_space(snap: dict[int, dict[str, Any]] | None, embedder_name: str | None) -> bool:
    """Whether a detector whose primary is *embedder_name* lives in *snap*'s patch space.

    **The single definition of the patch gate**, so training and scoring cannot
    disagree about which geometry a detector is in:

    * An explicit primary (the per-detector embedder) is in the patch space
      **only** when it *is* the snap's patch-slot embedder.  A detector locked to
      the text or structural space of a multi-embedder dataset scores against
      that space's full-image vectors (:func:`_score_all_media`), so it must
      train on those vectors too - flooding it with patch rows would mix two
      unrelated embedding spaces into one model (or, when their dimensions
      differ, fail outright with a
      :class:`~vtscore.embedding.precomputed.MismatchedVectorError`).
    * ``None`` (no per-detector primary) keeps the dataset-level score
      precedence, where any media carrying a ``patch_grid`` takes the patch
      path - the pre-per-detector behaviour.

    Callers pair this with "does any media actually carry a grid?"; on a
    grid-less dataset both sides collapse to the single image-level vector
    regardless.
    """
    if embedder_name is None:
        return True
    patch = _patch_embedder_for_snap(snap)
    return patch is not None and embedder_name == patch


def validate_good_bad_split(y_list: list[float]) -> tuple[int, int]:
    """Check that *y_list* contains at least one good and one bad label.

    Returns ``(num_good, num_bad)``.
    Raises ``ValueError`` when either count is zero.
    """
    num_good = sum(1 for y in y_list if y == 1.0)
    num_bad = len(y_list) - num_good
    if num_good == 0 or num_bad == 0:
        raise ValueError("Need at least one good and one bad labeled example")
    return num_good, num_bad


def _flood_context(
    X_list: list,
    y_list: list[float],
    groups: list | None,
) -> tuple[int, list | None, Any]:
    """Resolve the bag-aware training context for a possibly-flooded label set.

    Returns ``(n_votes, cal_groups, sample_weights)``:

    * ``n_votes`` - the number of distinct bags (votes/images); the unit the
      hidden-layer width and the fallback blend's ramp should size on, so region
      flooding (many rows per Bad vote) doesn't inflate either.
    * ``cal_groups`` - *groups* when flooding actually occurred (a bag holds
      more than one row), else ``None`` so the calibrator takes its historical
      row-wise path unchanged.
    * ``sample_weights`` - per-bag loss weights when flooding occurred, else
      ``None`` so :func:`~vtscore.training.mlp.train_model` computes its default
      inverse-frequency weights.

    Shared by :func:`train_and_threshold` and :func:`_train_and_score_xy` so the
    vote, labelset, and Find paths flood identically.
    """
    import torch  # noqa: PLC0415

    from vtscore.training.thresholds import _per_bag_fit_weights  # noqa: PLC0415

    n_votes = len(set(groups)) if groups is not None else len(X_list)
    flooded = groups is not None and len(X_list) != n_votes
    cal_groups = groups if flooded else None
    sample_weights = None
    if flooded and groups is not None:
        sample_weights = torch.tensor(
            _per_bag_fit_weights(np.asarray(y_list, dtype=np.float32), groups), dtype=torch.float32
        )
    return n_votes, cal_groups, sample_weights


def _blend_schedule_for_snap(snap: dict | None) -> str:
    """The fallback mix-in schedule these medias should be blended under.

    #2841 measured the two voting modes separately and they want different
    curves, so the schedule is resolved per training call rather than being one
    global constant.  The mode follows the *scoring* geometry, not how the user
    happened to vote: a patch dataset always scores by max-pooling over its raw
    patches (see :func:`_score_all_media`), which is exactly the ``region`` arm of the
    study, while a single-vector dataset is the ``binary`` arm.
    """
    from vtscore.training.blend_schedules import production_schedule_for  # noqa: PLC0415

    if not snap:
        return production_schedule_for(region_voting=None)
    return production_schedule_for(region_voting=_patch_embedder_for_snap(snap) is not None)


def _fused_threshold(
    xcal_threshold: float,
    folds: Any,
    rows: "ScoringRows | None",
    final_scores: list[float],
    inclusion_value: int,
    blend_ctx: Any,
    schedule: str,
    det_ctx: Any = None,
    final_ids: list[int] | None = None,
    voted_ids: "set[int] | None" = None,
) -> float:
    """The shipped threshold: the fold-anchored cut, schedule blend as fallback.

    Scores the haystack (*rows*, the ``(media, score row)`` matrix
    :func:`scoring_rows_for_snap` built for the snapshot) through each
    calibration fold model, so
    each fold's anchored mixture is fitted on that fold model's own score scale
    with that fold's *held-out* votes as anchors, then carries the per-fold cuts
    to the final model in quantile space
    (:func:`~vtscore.training.thresholds.fold_anchored_gmm_threshold`).  This is
    the estimator the 2026-08-05 deep-regime run picked over both pure
    cross-calibration and the schedule-blended GMM - see
    ``docs/experiments/2026-08-05-population-anchored-calibration/REPORT.md``.

    **Voted media are excluded from every haystack the estimator fits on**
    (issue #3308).  Each model in the chain was trained on the votes, so the
    votes' own scores under it are optimistically shifted - and the calibration
    votes additionally sit in the haystack twice, once as free points and once
    as anchors.  Dropping all of *voted_ids* from the fold haystacks *and* from
    the final model's realization sample keeps every distribution in the
    quantile transfer over the identical population (the unlabeled remainder),
    which is the population the threshold actually decides - the voted items'
    verdicts are already known.  The effect is bounded by the votes' share of
    the (<=50k-sampled) haystack: invisible on a large corpus, a measured win
    on small ones.  The exclusion switches off entirely when it would leave
    fewer than :data:`~vtscore.training.thresholds.EXCLUSION_MIN_REMAINDER`
    scores - a remainder that small is both too coarse to read quantiles from
    and, after deep Autopilot voting, selection-biased toward whatever
    acquisition never wanted; the full contaminated haystack measures better
    there (see the constant's rationale).  The schedule-blend *fallback* below
    deliberately keeps the full distribution: it only fires when there are no
    usable folds (<4 votes, or a single class), where the contamination is at
    most a few points, and its blend weights were measured on the full
    population.  *final_ids* names the media each entry of *final_scores*
    belongs to; the exclusion needs both it and *voted_ids*, and the
    historical include-everything behaviour is kept otherwise.

    The extra scoring passes (one per fold) are the estimator's whole marginal
    cost.  Production trains the *linear* head, so a pass is a matrix multiply
    and the cost is trivial; a heavier head would want measuring first.  Only
    the head changes between passes, never the rows, so *rows* is built once by
    the caller and reused by every fold **and** by the final model's own pass -
    which is what keeps the marginal cost a matmul rather than a restack.  On
    the active dataset the builders cache anyway; on a cross-dataset snapshot
    (Find, the CLI's importer chunks) they do not, and re-deriving per fold
    would have restacked the whole corpus ``calibrate_count + 1`` times - a
    multi-gigabyte float16 rebuild on a patch dataset, where the row stack is
    ~197 rows per media.

    Falls back to the schedule blend
    (:func:`~vtscore.training.thresholds.calculate_safe_threshold`) when there
    are no usable folds - the "dataset too small to fit the population
    estimator" case of the plan's decision rule 1.  The x-cal side of that
    blend is then :data:`~vtscore.training.thresholds.NO_GOOD_THRESHOLD`
    ("we never computed a cut, so admit nothing"), which is what the pre-fusion
    pure-GMM branch fed it.

    When *det_ctx* is given, the fitted estimator is parked on
    ``det_ctx.anchored_cut_cache`` so an Inclusion slide can re-cut it without
    refitting or re-scoring anything (see
    :func:`vtscore.state.core.recompute_detector_thresholds_for_inclusion`).

    **Unscorable media never reach the fit.**  A media the head cannot score
    (a broken vector, a destabilised model) is recorded at
    :data:`~vtscore.utils.scores.NON_FINITE_SCORE_SENTINEL`, a full unit below
    the sigmoid range; both estimators drop those before fitting, and both do
    it *inside* the shared :mod:`vtscore.training.thresholds` functions, so the
    eval harness's default arm inherits the same population without copying
    anything.  Only the warning below lives here, where the media count is a
    thing the user can act on.
    """
    from vtscore.training.thresholds import (  # noqa: PLC0415
        NO_GOOD_THRESHOLD,
        apply_vote_exclusion,
        calculate_safe_threshold,
        drop_voted,
        fit_fold_anchored_cut,
    )
    from vtscore.utils.scores import scored_only  # noqa: PLC0415

    n_unscorable = len(final_scores) - int(scored_only(final_scores).size)
    if n_unscorable:
        log.warning(
            "threshold fit: %d of %d media could not be scored and are excluded from the "
            "population estimate (check for media with broken embeddings)",
            n_unscorable,
            len(final_scores),
        )

    # One decision, taken on the final model's scores and then obeyed by every
    # fold haystack in this fit - the all-or-nothing contract lives in
    # ``apply_vote_exclusion`` rather than being re-derived per haystack here.
    fit_final, excluding = (
        apply_vote_exclusion(final_scores, final_ids, voted_ids)
        if final_ids is not None
        else (np.asarray(final_scores, dtype=np.float64), False)
    )
    # ``excluding`` is only ever True when ``voted_ids`` is a non-empty set, so
    # binding it here narrows the type for the fold loop AND makes the coupling
    # explicit: one name carries both "are we excluding" and "what by".
    exclude: set[int] | None = voted_ids if (excluding and voted_ids) else None

    cut = None
    if folds.fallback is None and rows is not None:
        n_folds = min(len(folds.models), len(folds.orderings))
        fold_haystacks = []
        for model in folds.models[:n_folds]:
            scores, _best = score_rows_with_model(model, rows)
            fold_haystacks.append(drop_voted(scores, rows.ids, exclude) if exclude else np.asarray(scores, np.float64))
        cut = fit_fold_anchored_cut(fold_haystacks, folds.orderings[:n_folds], fit_final)

    if det_ctx is not None:
        det_ctx.anchored_cut_cache = cut

    if cut is not None:
        threshold = cut.threshold_at(inclusion_value)
        if np.isfinite(threshold):
            return threshold
    xcal = NO_GOOD_THRESHOLD if folds.fallback is not None else xcal_threshold
    return calculate_safe_threshold(xcal, final_scores, blend_ctx, schedule=schedule)


def _calibration_score_rows(
    groups: list | None,
    cal_groups: list | None,
    score_rows: dict | None,
) -> dict | None:
    """Per-bag **inference** row stacks for the calibrator, or ``None``.

    Calibration collapses each held-out bag with ``max``.  Left to the training
    rows that is an unfair comparison on a patch dataset: a Good vote holds one
    row while a Bad vote holds its ~197 flooded patches, and ``max`` over 197
    draws beats ``max`` over 1 with no signal at all - so the cut lands high and
    the detector over-rejects true matches.  Handing the calibrator each voted
    image's *scoring* stack (all ~197 rows, exactly what
    :func:`_score_all_media` max-pools) puts both classes in the geometry
    inference actually uses.

    Coverage is **all-or-nothing**: unless every bag has a stack this returns
    ``None`` and the calibrator keeps pooling over the training rows.  Partial
    coverage is worse than none - a Good bag left on its 1 training row while
    the Bad bags stay at their full row count would deepen the very bias this
    corrects - so an unresolvable media declines the correction rather than
    skewing it.

    Also returns ``None`` when *cal_groups* is ``None``: nothing flooded, so
    every bag is already a single row and the row-wise path runs unchanged.
    """
    if cal_groups is None or groups is None or not score_rows:
        return None
    bag_ids = set(groups)
    if not bag_ids <= set(score_rows):
        return None
    return {g: np.asarray(score_rows[g], dtype=np.float32) for g in bag_ids}


def _patch_space_for(embedder_name: str | None) -> bool | None:
    """Whether *embedder_name* learns in a patch space, or ``None`` for unknown.

    Reads the embedder's **capability** (``supports_patch_regions``), not what
    it is doing in the current configuration: a patch embedder falling back to
    whole-image voting on a boxless dataset still learns in its patch space's
    pickle, and #3287 measured that it wants the patch split there too.  An
    empty name or one the registry doesn't know maps to ``None`` ("unknown"),
    not ``False`` - the split default has a three-state contract and an
    unrecognised embedder takes the mode-agnostic fallback rather than a guess.
    """
    if not embedder_name:
        return None
    from vtscore.media import get_embedder  # noqa: PLC0415

    try:
        embedder = get_embedder(embedder_name)
    except (KeyError, ValueError):
        return None
    return bool(getattr(embedder, "supports_patch_regions", False))


def resolve_calibration_fraction(calibration_fraction: float | None, embedder_name: str | None) -> float:
    """Resolve the Train/Calibrate split for a detector on *embedder_name*.

    An explicit *calibration_fraction* (the user's persisted setting, or a
    caller that pinned one) always wins.  ``None`` means "unset", which takes
    the per-space production default (issue #3287): 0.3 when the detector
    learns in a single-vector space, 0.5 on a patch grid, and 0.5 when the
    space is unknown - see
    :func:`vtscore.training.thresholds.production_split_for`.

    Mirrored by the eval harness: ``_resolve_production_defaults`` resolves
    ``calibration_fraction=None`` through the same
    :func:`~vtscore.training.thresholds.production_split_for` table, keyed on
    whether the dataset carries a ``patch_grid`` (its spelling of "built by a
    patch embedder").  If the predicate here changes, the harness's has to
    move with it - the ``training.split_fraction_default`` mirror in
    ``scripts/check-eval-app-sync.py`` pins this function for that reason.
    """
    if calibration_fraction is not None:
        return float(calibration_fraction)
    from vtscore.training.thresholds import production_split_for  # noqa: PLC0415

    return production_split_for(patch_space=_patch_space_for(embedder_name))


def train_and_threshold(
    X_list: list,
    y_list: list[float],
    snap: dict | None = None,
    embedder_name: str | None = None,
    det_ctx: Any = None,
    groups: list | None = None,
    score_rows: dict | None = None,
    voted_ids: "set[int] | None" = None,
    haystack: dict | None = None,
    haystack_rows: "ScoringRows | None" = None,
) -> tuple[Any, float]:
    """Train the detector head and compute a calibrated threshold.

    This is the canonical training pipeline used by all detector routes:

    1. K-fold calibration (respects ``calibrate_count`` /
       ``calibration_fraction`` settings), giving both the cross-calibration
       cut and the fold models.
    2. Full-data model training (respects ``inclusion`` setting).
    3. The fold-anchored population threshold whenever *snap* is provided -
       see :func:`_safe_threshold`.  It is fitted on the per-media scores
       :func:`_score_all_media` produces - region max-pooled on a patch
       dataset - so it sees the distribution the threshold will actually cut.
       Without a *snap* there is no haystack to fuse and the cross-calibration
       cut ships alone.

    ``inclusion`` is read from ``get_inclusion()``, which resolves to the
    *active detector context's* inclusion (seeded from the user's settings
    default the first time it's read for a detector). Both Train and Find
    therefore train at the same per-detector inclusion within a session.

    Args:
        X_list: Embedding vectors (list of numpy arrays).
        y_list: Binary labels (1.0 = good, 0.0 = bad).
        snap: Optional media snapshot - the haystack the population
            estimator is fitted on, unless *haystack* overrides it.  Without
            either there is nothing to fit on and the threshold is the plain
            cross-calibration cut.
        embedder_name: The detector's primary embedder, used so the
            haystack scoring pass reads vectors from the same space the
            ``X_list`` were built in.  ``None`` falls back to the dataset score
            precedence for *snap* (the pre-per-detector behaviour).
        det_ctx: When provided, the inclusion-independent K fold orderings are
            cached on ``det_ctx.calibration_cache`` (and the fold models are
            sized to match the final model).  This is what lets a later
            Inclusion slide re-derive the threshold over the cached orderings
            instead of being a no-op — see
            :func:`vtscore.state.core.recompute_detector_thresholds_for_inclusion`.
            ``None`` keeps the
            legacy (uncached) behaviour for callers that don't own a context.
        groups: Per-row bag ids (one voted image per bag); see
            :func:`_flood_context`.
        score_rows: Per-bag inference row stacks; see
            :func:`_calibration_score_rows`.  Only consulted when *groups*
            reveals flooding.
        haystack: The population the cut is *realized* on, when that is not
            *snap*.  The two are the same set for every caller that scores the
            snapshot it calibrated against - the GUI, which scores the loaded
            dataset it was handed - and differ for the CLI, whose scoring pass
            converts, re-clips and re-embeds the loaded medias into the
            detector's own granularity before scoring them
            (:func:`~vtscore.detectors.converter_routing.route_and_embed`).
            The fold-anchored estimator carries its per-fold cuts to the final
            model *in quantile space* and realizes them against this
            distribution, so handing it the loaded medias while inference reads
            the routed ones lands the quantile on the wrong ruler: a cut fitted
            on whole images and applied to the max over their clips sits far
            lower in the scored population than the algorithm intended (issue
            #3647).  *snap* keeps its other job either way - it is the snapshot
            a caller's labels are resolved against, which the routed items,
            with their throwaway ids and recomputed hashes, cannot serve as.
            The blend schedule follows *haystack* too, since it is chosen off
            the scoring geometry.
        voted_ids: Media ids in the haystack whose labels the training set
            carries.
            Excluded from every haystack the fold-anchored estimator fits on -
            their scores under the models trained on them are optimistically
            shifted (issue #3308; see :func:`_fused_threshold`).  ``None`` (the
            callers with no way to name their labels' media) keeps the full
            haystack.
        haystack_rows: the haystack's already-built
            :func:`scoring_rows_for_snap` matrix, when the caller has one in
            hand.  Purely an optimisation - it must be the rows this call would
            have built itself, i.e. ``scoring_rows_for_snap(haystack or snap,
            embedder_name)``.  A caller that scores the same snapshot for its
            own purposes (cross-dataset Find, which scores every media it just
            loaded) passes its copy so the corpus is stacked once for the whole
            operation rather than once here and once there; on a patch dataset
            that stack is ~197 float16 rows per media, so the saving is memory
            as much as time.  ``None`` builds it here.

    Returns:
        ``(model, threshold)``
    """
    import torch

    from vtscore.state import (
        get_calibrate_count,
        get_calibration_fraction,
        get_inclusion,
    )
    from vtscore.training import (
        calibration_folds,
        calibration_folds_cached,
        threshold_from_folds,
        train_model,
    )
    from vtscore.training.blend_schedules import BlendContext
    from vtscore.training.mlp import LINEAR_SVM_HEAD

    X = torch.from_numpy(stack_vectors(X_list, label="training vector"))
    y = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1)
    input_dim = X.shape[1]

    # Bag-aware setup (region flooding): size on votes not rows, split/weight
    # per bag.  On a legacy label set every bag is one row, so this collapses
    # to the historical behaviour.
    _n_votes, cal_groups, sample_weights = _flood_context(X_list, y_list, groups)
    cal_score_rows = _calibration_score_rows(groups, cal_groups, score_rows)

    # The production detector head is the **linear SVM**: one Linear(d, 1) fitted
    # to the maximum-margin boundary (hinge + L2) rather than to balanced BCE.
    # Like the logistic head it replaced it has no capacity to wobble under
    # sparse positives (the threshold-stability #2790 finding); unlike it, it
    # ranks measurably better.  The same head fits the final model and the
    # calibration fold models on *both* branches below (cached and uncached), so
    # the calibrated threshold is always measured on the head the final model
    # actually has.
    hidden_dim = LINEAR_SVM_HEAD

    inclusion = get_inclusion()
    # The user's persisted split wins; unset resolves to the per-space
    # production default for this detector's embedder (issue #3287).
    calibration_fraction = resolve_calibration_fraction(get_calibration_fraction(), embedder_name)
    blend_ctx = BlendContext.from_labels(y_list, cal_groups)
    # The calibration folds are computed at *every* label count.  The pre-fusion
    # path skipped them below the blend schedule's floor, where the schedule
    # discarded the cross-cal cut anyway; the fold-anchored estimator that
    # replaced the schedule needs the fold *models* (it anchors on their
    # held-out scores), so there is nothing left to skip.  Two extra linear-head
    # fits at 4-5 votes is the whole cost.
    if det_ctx is not None:
        # Cache the K folds on the context so an Inclusion slide can re-derive
        # the cutoff without a no-op (the find-label / detector-load paths land
        # here; without the cache the slide can't move the line).
        folds = calibration_folds_cached(
            X_list,
            y_list,
            input_dim,
            calibrate_count=get_calibrate_count(),
            calibration_fraction=calibration_fraction,
            hidden_dim=hidden_dim,
            det_ctx=det_ctx,
            groups=cal_groups,
            score_rows_by_group=cal_score_rows,
        )
    else:
        folds = calibration_folds(
            X_list,
            y_list,
            input_dim,
            calibrate_count=get_calibrate_count(),
            calibration_fraction=calibration_fraction,
            hidden_dim=hidden_dim,
            groups=cal_groups,
            score_rows_by_group=cal_score_rows,
        )
    threshold = threshold_from_folds(folds, inclusion)

    if sample_weights is not None:
        model = train_model(X, y, input_dim, hidden_dim=hidden_dim, sample_weights=sample_weights)
    else:
        model = train_model(X, y, input_dim, hidden_dim=hidden_dim)

    # Fit the population estimator on the *inference* score distribution.
    # `scoring_rows_for_snap` + `score_rows_with_model` is `_score_all_media`,
    # the same call scoring makes (`score_media_with_model`), split so the rows
    # are built once and reused by the fold passes below: on a patch dataset the
    # mixture sees the region max-pooled per-media scores the threshold will
    # actually cut.  Scoring the image-level embedding matrix instead fitted it
    # on a systematically lower distribution (the region max is ≥ the single
    # image-level row), biasing the cut low → over-inclusion on
    # region-voting detectors.  Plain single-vector datasets take
    # `scoring_rows_for_snap`'s embedding-matrix fallback, so their behaviour is
    # unchanged.  *embedder_name* is forwarded as-is (not pre-resolved) so
    # the region-vs-plain gating matches inference exactly.
    # Mirrors `_train_and_score_xy` and
    # `eval.voting_iterations._safe_threshold_for_step`.
    #
    # The row builder skips media it cannot score, so an empty score list means
    # the haystack contributed nothing to fit on - either there was no haystack
    # at all, or none of its media carry a usable vector in this space.  Both
    # take the no-haystack branch rather than fitting the estimator on an empty
    # distribution.
    #
    # A *partly* embedded haystack has no such branch, and that is why the CLI
    # must embed at load: fitting the cut on the subset that happens to carry a
    # vector is silent, plausible, and wrong - a lower threshold over fewer
    # items (issue #3556).  `vtscore.cli._embed_loaded_medias` is what keeps the
    # CLI's snap embedded before this call, mirroring the GUI's load pipeline.
    #
    # The population the quantile is realized on is almost always *snap* - the
    # caller scores what it calibrated against - but the CLI converts and
    # re-clips before scoring, so it names the routed snapshot instead (#3647).
    hay = haystack if haystack is not None else snap
    rows: ScoringRows | None = None
    all_ids: list[int] = []
    all_scores: list[float] = []
    if hay:
        rows = haystack_rows if haystack_rows is not None else scoring_rows_for_snap(hay, embedder_name)
        all_ids = rows.ids
        all_scores, _best_region = score_rows_with_model(model, rows)
    if rows is not None and all_scores:
        threshold = _fused_threshold(
            threshold,
            folds,
            rows,
            all_scores,
            inclusion,
            blend_ctx,
            _blend_schedule_for_snap(hay),
            det_ctx=det_ctx,
            final_ids=all_ids,
            voted_ids=voted_ids,
        )
    elif det_ctx is not None:
        # Safe thresholds off: no population estimator to re-cut on a slide.
        det_ctx.anchored_cut_cache = None

    return model, threshold


def serialize_weights(model) -> dict[str, list]:
    """Convert a PyTorch model's state dict to JSON-serialisable nested lists."""
    return {key: value.cpu().tolist() for key, value in model.state_dict().items()}


# ---------------------------------------------------------------------------
# Vote-aware detector training (online, called from sort/vote handlers)
# ---------------------------------------------------------------------------


def pool_box_from_media(
    media: dict[str, Any],
    region_box: tuple[float, float, float, float] | None,
) -> np.ndarray | None:
    """Return the region training vector for *media*, or ``None``.

    **MaxPatch Good-vote rule.**  When *region_box* is set and the media
    carries a ``patch_grid``, return the single raw patch vector nearest the
    box (:func:`vtscore.media.patch_embed.nearest_patch_to_box`) - one of the
    very rows :func:`vtscore.embedding.matrix.media_score_rows` will score the
    image over, so the Good vote is a fair representative of what the detector
    actually evaluates.  Returns ``None`` (the caller then uses the image-level
    ``embedding``, which is row 0 of that same stack) for a boxless vote and
    for legacy single-vector embedders that carry no grid.

    This replaced the HAC snap-to-node rule in #2886: over 23 scale-band Visual
    Genome categories the raw patch beat the tree's best-IoU node on both
    halves of the error at every scale band, and by the largest margin exactly
    where the hypothesis said it would - below leaf scale, where the tree's
    smallest pooled candidate already blends object with context while a raw
    patch is a near-pure object sample.  See
    ``docs/experiments/2026-07-29-max-patch/REPORT.md``.

    Note the drawn box's width and height are discarded in essentially every
    case: ``nearest_patch_to_box`` collapses to "the patch nearest the box
    centre" unless the box is thinner than one cell.  That is the shipped
    design and what the study measured, not an oversight - the natural 4-DOF
    alternative (mean of the patches inside the box) is a per-vote amalgam that
    can never be a per-image scored row, so it breaks the train/score
    invariant by construction.

    Shared by the in-dataset vote path (:func:`_training_vec_for_vote`) and
    the cross-dataset labelset path
    (:func:`vtscore.detectors.labelset_training._resolve_uncached_embedding`).
    """
    if region_box is None:
        return None

    grid = media.get("patch_grid")
    if grid is None:
        return None
    from vtscore.media.patch_embed import nearest_patch_to_box  # noqa: PLC0415

    return nearest_patch_to_box(np.asarray(grid), region_box)


def _training_vec_for_vote(
    media: dict[str, Any],
    region_box: tuple[float, float, float, float] | None,
    embedder_name: str | None = None,
) -> np.ndarray:
    """Return the training vector for one vote on *media*.

    Region-pools via :func:`pool_box_from_media` when the vote designated a
    box and *media* carries a ``patch_grid``; otherwise falls back to the
    image-level vector of *embedder_name* (the score embedder) - or the
    primary vector when ``None``.
    """
    from vtscore.embedding.media_vectors import media_embedding  # noqa: PLC0415

    pooled = pool_box_from_media(media, region_box)
    return pooled if pooled is not None else media_embedding(media, embedder_name)


def bad_negative_vecs(
    media: dict[str, Any],
    embedder_name: str | None = None,
) -> list[np.ndarray]:
    """Negative training vectors contributed by one Bad vote on *media*.

    On **patch** media (carrying a ``patch_grid``) a Bad vote floods the
    image-level vector plus **every raw patch** - i.e. exactly
    :func:`vtscore.embedding.matrix.media_score_rows`, the same ~197 rows
    :func:`_score_all_media` max-pools.  This is the multiple-instance-learning
    treatment of a rejected image: since inference scores an image by its
    **best** row, a Bad vote asserts that *no* row of it should score high, so
    every row is trained down.

    The flood and the scoring stack are **the same function call**, which is
    the point.  Under the old HAC tree they were deliberately different - the
    flood covered the CLS node and the leaves but skipped the internal merge
    nodes, a measured exception (#2731) forced by internals being renormalised
    convex-hull points that are not dominated by their own leaves.  MaxPatch
    has no internals, so the gap closes and every scored row is a flooded row.

    All ~197 rows share one bag id in :func:`_build_vote_xy`, so a rejected
    image still counts as **one** vote for weighting, splitting, and the
    threshold's small-count ramp.

    Non-patch media contribute a single image-level vector, exactly as before,
    so every legacy single-vector dataset is byte-for-byte unchanged.
    """
    from vtscore.embedding.matrix import media_score_rows  # noqa: PLC0415
    from vtscore.embedding.media_vectors import media_embedding  # noqa: PLC0415

    rows = media_score_rows(media, embedder_name)
    if rows is None:
        return [media_embedding(media, embedder_name)]
    return list(rows)


def inference_score_rows(
    media: dict[str, Any],
    embedder_name: str | None = None,
) -> np.ndarray | None:
    """The row stack *media* is max-pooled over at inference, or ``None``.

    A thin alias for :func:`vtscore.embedding.matrix.media_score_rows` - the
    same rows :func:`vtscore.embedding.matrix._build_region_arrays` flattens
    into the matrix :func:`_score_all_media` scores: image-level vector + every
    raw patch on a patch media (~197 rows), a single image-level vector on a
    grid-less one.

    Used to calibrate in inference geometry: a voted image's bag must collapse
    over the same rows the scorer will pool, not over the (Good: 1, Bad: ~197)
    rows the fold model happened to train on.  See
    :func:`_calibration_score_rows`.
    """
    from vtscore.embedding.matrix import media_score_rows  # noqa: PLC0415

    return media_score_rows(media, embedder_name)


def _vote_score_rows(
    media: dict[str, Any],
    embedder_name: str | None,
    row_embedder: str | None,
    *,
    patch_rows: bool,
) -> np.ndarray | None:
    """The rows the scorer max-pools *media* over, in the detector's own space.

    The patch stack (:func:`inference_score_rows`) when the detector scores in
    the patch space, else the single image-level row of *embedder_name* - which
    is exactly what :func:`_score_all_media`'s embedding-matrix fallback scores
    that media by.  See :func:`_scores_in_patch_space`.
    """
    if patch_rows:
        return inference_score_rows(media, row_embedder)
    from vtscore.embedding.media_vectors import media_embedding  # noqa: PLC0415

    vec = media_embedding(media, embedder_name)
    return None if vec is None else np.asarray(vec, dtype=np.float32).reshape(1, -1)


def _vote_negative_vecs(
    media: dict[str, Any],
    embedder_name: str | None,
    row_embedder: str | None,
    *,
    patch_rows: bool,
) -> list[np.ndarray]:
    """The rows one Bad vote on *media* trains down - the rows the scorer pools.

    The flood (:func:`bad_negative_vecs`) when the detector scores in the patch
    space, else the one image-level vector of *embedder_name*.  A non-patch
    detector never max-pools the grid, so flooding it would train the model on
    rows from another embedding space that nothing ever scores.
    """
    if patch_rows:
        return bad_negative_vecs(media, row_embedder)
    from vtscore.embedding.media_vectors import media_embedding  # noqa: PLC0415

    return [media_embedding(media, embedder_name)]


def _build_vote_xy(
    clips_dict: dict[int, dict[str, Any]],
    good_votes: dict[int, None],
    bad_votes: dict[int, None],
    region_boxes: dict[int, tuple[float, float, float, float]],
    embedder_name: str | None = None,
) -> tuple[list[np.ndarray], list[float], list, dict]:
    """Build ``(X_list, y_list, groups, score_rows)`` from filtered votes.

    Good votes that designated a region train on the nearest raw patch via
    :func:`_training_vec_for_vote` (one row each).  Bad votes are expanded by
    :func:`bad_negative_vecs`: one row per image-level vector on a legacy
    dataset, or the image-level vector + every raw patch on a patch dataset
    (region flooding).

    ``groups`` carries one bag id per row - ``("g", cid)`` for a Good vote,
    ``("b", cid)`` shared across all of a Bad vote's ~197 flooded rows - so the
    downstream trainer/calibrator can balance and split by **image**, not by
    row.  On a legacy dataset every bag holds exactly one row, so ``groups`` is
    1:1 with the rows and the whole path collapses to the pre-flood behaviour.
    The caller (:func:`_train_and_score_xy`) enforces the ≥2-samples /
    ≥1-good / ≥1-bad guard.

    ``score_rows`` maps each bag id to the row stack that voted image is
    *scored* over at inference (:func:`inference_score_rows`) - the whole
    ~197-row patch stack on a patch media - so calibration can collapse a Good
    bag and a Bad bag the same way :func:`_score_all_media` collapses any
    image.  Without it a Good bag is a max over its 1 training row against a
    Bad bag's max over ~197, and the calibrated cut lands high.

    *embedder_name* is the detector's primary embedder; when ``None`` the
    dataset score precedence for *clips_dict* is used (the pre-per-detector
    behaviour).  Either way the MLP trains in the same space
    :func:`_score_all_media` scores against.

    All three patch behaviours above - the Good vote's region pool, the Bad
    vote's flood, and the recorded calibration stack - are gated by
    :func:`_scores_in_patch_space`, the same gate :func:`_score_all_media`
    pools under.  A detector locked to the text or structural space of a
    multi-embedder dataset therefore trains on that space's **full-image**
    vectors, exactly the rows it will be scored over; without the gate its Bad
    votes flooded patch-space rows (and its boxed Good votes pooled one) into a
    model scored in a different space entirely - a
    :class:`~vtscore.embedding.precomputed.MismatchedVectorError` when the two
    dimensions differ, silent garbage negatives when they happen to match.
    """
    patch_rows = _scores_in_patch_space(clips_dict, embedder_name)
    if embedder_name is None:
        embedder_name = _score_embedder_for_snap(clips_dict)
    # The image-level row of a patch stack belongs to the *patch-slot* embedder
    # (the space the grid lives in), which is what the scoring matrix reads -
    # see ``matrix._patch_embedder_for_region_snap``.  On a single-embedder
    # patch dataset this is the score embedder anyway; on a text+patch dataset
    # it keeps the flooded / calibrated rows out of the text space.
    row_embedder = (_patch_embedder_for_snap(clips_dict) or embedder_name) if patch_rows else embedder_name
    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    groups: list = []
    score_rows: dict = {}

    def _record_score_rows(group: tuple, cid: int) -> None:
        rows = _vote_score_rows(clips_dict[cid], embedder_name, row_embedder, patch_rows=patch_rows)
        if rows is not None:
            score_rows[group] = rows

    for cid in good_votes:
        if cid in clips_dict:
            box = region_boxes.get(cid) if patch_rows else None
            X_list.append(_training_vec_for_vote(clips_dict[cid], box, embedder_name))
            y_list.append(1.0)
            groups.append(("g", cid))
            _record_score_rows(("g", cid), cid)
    for cid in bad_votes:
        if cid in clips_dict:
            for vec in _vote_negative_vecs(clips_dict[cid], embedder_name, row_embedder, patch_rows=patch_rows):
                X_list.append(vec)
                y_list.append(0.0)
                groups.append(("b", cid))
            _record_score_rows(("b", cid), cid)
    return X_list, y_list, groups, score_rows


def _forward_sigmoid_chunked(model: nn.Sequential, matrix: np.ndarray) -> np.ndarray:
    """``sigmoid(model(matrix))`` as float64, upcasting a float16 matrix chunk-wise.

    The flattened patch matrix is stored float16 (see
    :func:`vtscore.embedding.matrix._build_region_arrays`); torch has no
    float16 CPU linear kernel and a whole-matrix upcast would allocate a
    float32 copy twice the size of the matrix - gigabytes on a large patch
    dataset, where MaxPatch already stacks ~197 rows per image.  Chunking
    bounds that copy at ``ROW_CHUNK`` rows.

    :func:`~vtscore.utils.scores.sigmoid_to_finite_array` replaces NaN/±Inf
    with the ``NON_FINITE_SCORE_SENTINEL`` (-1.0) so a destabilised MLP cannot
    leak non-finite floats into the JSON response.  The downstream segmented
    max-pool then incidentally drops sentinels in favour of any real score (in
    ``[0, 1]``) for the same media.
    """
    import torch  # noqa: PLC0415

    from vtscore.embedding.matrix import ROW_CHUNK  # noqa: PLC0415

    device = next(model.parameters()).device
    out = np.empty(matrix.shape[0], dtype=np.float64)
    with torch.no_grad():
        for start in range(0, matrix.shape[0], ROW_CHUNK):
            block = np.ascontiguousarray(matrix[start : start + ROW_CHUNK])
            chunk = torch.from_numpy(block).to(device=device, dtype=torch.float32)
            out[start : start + chunk.shape[0]] = sigmoid_to_finite_array(model(chunk))
    return out


class ScoringRows(NamedTuple):
    """The ``(media, row)`` matrix one snapshot is scored over.

    ``ids`` are the media ids in sorted order; ``matrix`` holds one row per
    ``(media, score row)`` pair; ``media_index`` maps each row to its media's
    index in ``ids`` and ``region_index`` to its position within that media's
    :func:`vtscore.embedding.matrix.media_score_rows` stack.  Built once by
    :func:`scoring_rows_for_snap` and reusable across every head that scores
    the same snapshot in the same space (:func:`score_rows_with_model`).
    """

    ids: list[int]
    matrix: np.ndarray
    media_index: np.ndarray
    region_index: np.ndarray


def scoring_rows_for_snap(
    clips_dict: dict[int, dict[str, Any]],
    embedder_name: str | None = None,
    *,
    region_pooling: bool | None = None,
) -> ScoringRows:
    """Build the rows every media in *clips_dict* is scored over.

    Patch datasets (those whose media expose a ``patch_grid``) contribute every
    media's :func:`vtscore.embedding.matrix.media_score_rows` stack - image-level
    vector + all ``H*W`` raw patches - so the head's score for a media is the max
    over its rows.  Plain datasets contribute one row per media, from the cached
    embedding matrix.

    *embedder_name* is the detector's primary embedder (the space the head was
    trained in).  When it is given, the patch rows are used **only** if that
    primary is the dataset's patch-slot embedder - a detector scoring in the
    text or structural space of a multi-embedder dataset must score against
    that space's full-image vectors, not the patch grid.  When ``None`` (the
    pre-per-detector behaviour) any media carrying a ``patch_grid`` takes the
    patch path, matching the dataset-level score precedence.

    Split out from :func:`_score_all_media` so a caller with several heads over
    one snapshot - the CLI's per-group detector scoring - builds the rows once,
    and so no caller has to re-derive the patch gate (or the skip policy below)
    for itself.

    *region_pooling* overrides that gate for a caller whose **head** is not a
    MaxPatch head.  ``None`` (every ordinary caller) resolves it as described
    above.  ``False`` forces one image-level row per media even on a patch
    dataset - the geometry a whole-image head has to be scored at, because it
    was never shown a patch as a negative and would fire on distractors the
    max-pool then promotes to the media's score (see ``docs/ML.md``, region
    flooding).  ``True`` is accepted for symmetry and still yields one row per
    media on a grid-less dataset, which has no patch rows to pool.

    **No shipped caller passes ``False`` today.**  Cross-dataset Find's cold
    path used to, and was the case this override was written for; #3525 replaced
    the whole-image head it needed the override for with the app's own
    region-flooded labelset training, so the pin went with the head.  The knob
    stays because it is the correct answer for any caller that does build a
    whole-image head, and because it is the only thing separating "this head
    is not MaxPatch" from the dataset-level patch gate.

    Media that carry no usable vector in that space are **skipped**, not fatal:
    the builders raise on one (no vector, or a row of the wrong width), and this
    catches that, drops the offending media via
    :func:`~vtscore.embedding.matrix.scoreable_snapshot`, and rebuilds from what
    is left.  A snapshot is not the dataset's own medias dict - the CLI scores
    importer output that never went through the load pipeline's drop-none stage,
    and one bound embedder of a multi-embedder dataset can have failed on media
    another succeeded on - so a single unembeddable image must cost one skipped
    item, not the whole run (issue #3179).  Every caller reads the media list
    back off :attr:`ScoringRows.ids`, so a shorter list flows through unchanged
    - which is also how a caller names what was skipped, without this having to
    report it.

    The filter runs **on the failure path only**, not as a pre-pass: this is the
    per-vote retrain path over the whole dataset (and ``_fused_threshold`` calls
    it once more per calibration fold), where an unconditional O(N) scan would
    tax every vote on a 300k-media dataset to catch a case that the load
    pipeline has already made impossible there.  A clean snapshot pays nothing
    and keeps its cached-matrix fast path.
    """
    from vtscore.embedding.matrix import (  # noqa: PLC0415
        get_embedding_matrix_for_snap,
        get_region_matrix_for_snap,
        scoreable_snapshot,
    )

    resolved = embedder_name if embedder_name is not None else _score_embedder_for_snap(clips_dict)
    # Explicit per-detector primary: patch-pool only when scoring in the patch
    # space (the grid lives in the patch embedder's vectors).  Same gate
    # ``_build_vote_xy`` trains under - see :func:`_scores_in_patch_space`.
    has_regions = any(clips_dict[cid].get("patch_grid") is not None for cid in clips_dict) and _scores_in_patch_space(
        clips_dict, embedder_name
    )
    if region_pooling is not None:
        has_regions = has_regions and region_pooling

    def _build(snapshot: dict[int, dict[str, Any]]) -> ScoringRows:
        if has_regions:
            # One row per (media, score row) pair, built once and cached on the
            # dataset context (the patch vectors never change between votes -
            # only the MLP weights do), so online retraining no longer rebuilds
            # a multi-million-row matrix on every vote.
            return ScoringRows(*get_region_matrix_for_snap(snapshot))
        ids, matrix = get_embedding_matrix_for_snap(snapshot, resolved)
        n = len(ids)
        return ScoringRows(ids, matrix, np.arange(n, dtype=np.int64), np.zeros(n, dtype=np.int64))

    try:
        rows = _build(clips_dict)
    except ValueError:
        # Both refusals land here: the missing-vector ``ValueError`` and the
        # wrong-width ``MismatchedVectorError`` that subclasses it.  Filter
        # against the same key the chosen builder reads - the patch-slot
        # embedder for the region path (``_build_region_arrays`` requires an
        # image-level row per media in that space), the resolved score embedder
        # otherwise - then rebuild.  A rebuild that still refuses is a different
        # problem and propagates.
        survivors, skipped = scoreable_snapshot(clips_dict, resolved, region_rows=has_regions)
        log.warning(
            "Scoring skipped %d media with no usable vector under %r: %s%s",
            len(skipped),
            resolved or "(primary)",
            skipped[:10],
            "…" if len(skipped) > 10 else "",
        )
        if not survivors:
            empty_rows = np.empty((0,), dtype=np.int64)
            return ScoringRows([], np.empty((0, 0), dtype=np.float32), empty_rows, empty_rows)
        rows = _build(survivors)

    if int(rows.matrix.shape[0]) != int(rows.media_index.size):
        # Deliberately outside the retry above: a builder that hands back an id
        # list and a matrix of different lengths is not the "this media has no
        # vector" case that filtering fixes, and rebuilding would only hide it
        # for one round. Left alone it surfaces as an out-of-bounds `reduceat`
        # deep in the max-pool, or - worse, under a plain zip - as silently
        # truncated scores (audit M11). Fail here, naming both counts.
        raise ValueError(
            f"scoring rows for {len(rows.ids)} media claim {rows.media_index.size} rows "
            f"but the matrix holds {rows.matrix.shape[0]}; the embedding matrix and its id list disagree."
        )
    return rows


def score_rows_with_model(model: nn.Sequential, rows: ScoringRows) -> tuple[list[float], list[int]]:
    """Forward *rows* through *model* and max-pool them down to one score per media.

    Returns ``(scores_per_media, best_row_index_per_media)``; the winning row's
    index is what surfaces the best-match overlay in the UI.
    """
    from vtscore.embedding.matrix import segmented_max_pool  # noqa: PLC0415

    if not rows.ids:
        return [], []
    flat_scores = _forward_sigmoid_chunked(model, rows.matrix)
    return segmented_max_pool(flat_scores, rows.media_index, rows.region_index, len(rows.ids))


def _score_all_media(
    model: nn.Sequential,
    clips_dict: dict[int, dict[str, Any]],
    embedder_name: str | None = None,
) -> tuple[list[int], list[float], list[int]]:
    """Score every media in *clips_dict* with the trained detector head.

    The composition of :func:`scoring_rows_for_snap` and
    :func:`score_rows_with_model`, and **the** definition of what a detector
    scores a media at: every path that turns a head into per-media scores -
    online voting, Find, the population estimator behind the threshold, and CLI
    autodetect - goes through it, so none of them can score a media at a
    different geometry than the threshold was cut on.

    Returns ``(all_ids, scores_per_media, best_row_index_per_media)``.
    """
    rows = scoring_rows_for_snap(clips_dict, embedder_name)
    scores, best_region = score_rows_with_model(model, rows)
    return rows.ids, scores, best_region


def _format_results(
    all_ids: list[int],
    scores: list[float],
    best_region: list[int],
    clips_dict: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Sort by score (descending) and produce JSON-serialisable result dicts.

    Raw float scores are used for sorting so tiny differences still
    affect ordering; only the response ``score`` field is rounded.
    Patch media gain a ``best_region`` key holding the winning row's box - the
    whole image for row 0, a single grid cell otherwise
    (:func:`vtscore.embedding.matrix.media_row_box`).
    """
    from vtscore.embedding.matrix import media_row_box  # noqa: PLC0415

    paired = sorted(
        zip(all_ids, scores, best_region, strict=True),
        key=lambda t: t[1],
        reverse=True,
    )
    results: list[dict[str, Any]] = []
    for cid, s, bri in paired:
        entry: dict[str, Any] = {"id": cid, "score": round(s, 4)}
        box = media_row_box(clips_dict[cid], bri)
        if box is not None:
            entry["best_region"] = box
        results.append(entry)
    return results


def score_media_with_model(
    model: nn.Sequential,
    clips_dict: dict[int, dict[str, Any]],
    embedder_name: str | None = None,
) -> list[dict[str, Any]]:
    """Score every media in *clips_dict* with an already-trained *model*.

    Returns sorted (descending by score) result dicts of the same shape the
    vote-driven training path produces: ``{"id", "score"}`` plus a
    ``best_region`` box for patch-region-aware media (the argmax region that
    drove the media's score).  Use this from any route that scores with a
    pre-trained detector - e.g. the Find / detector-scoring path - so the
    best-match highlight is populated regardless of which entry point ran the
    scoring.  Plain single-vector datasets are scored via the cached embedding
    matrix and gain no ``best_region`` field, exactly as before.

    *embedder_name* is the detector's primary embedder, so the scoring space
    matches the one the *model* was trained in (the per-detector primary).
    """
    all_ids, scores, best_region = _score_all_media(model, clips_dict, embedder_name)
    return _format_results(all_ids, scores, best_region, clips_dict)


def _train_and_score_xy(
    X_list: list[np.ndarray],
    y_list: list[float],
    clips_dict: dict[int, dict[str, Any]],
    *,
    inclusion_value: int,
    calibrate_count: int,
    calibration_fraction: float | None,
    det_ctx: Any,
    groups: list | None = None,
    score_rows: dict | None = None,
    voted_ids: "set[int] | None" = None,
    rows: ScoringRows | None = None,
) -> tuple[list[dict[str, Any]], float, nn.Sequential | None]:
    """Train the detector head on ``(X_list, y_list)`` and score every media in *clips_dict*.

    Shared core of :func:`train_and_score` (vote-driven) and
    :func:`vtscore.detectors.labelset_training.labelset_train_and_score`
    (labelset-driven): the two pipelines differ only in how they assemble
    ``(X_list, y_list)``, so the guard → threshold → train → score → format
    tail lives here once.

    The head is the linear SVM at every label count, so region
    flooding can't inflate its capacity; the threshold's label count is still
    sized from the **vote** count (distinct *groups*) rather than the row count,
    so flooding - which turns one Bad vote into many leaf rows - doesn't shift
    the fallback blend's small-count ramp.  When *groups*
    reveals at least one multi-row bag (flooding actually happened), the
    calibration split, fold fits, and final fit all run **bag-aware**
    (grouped fold split, per-bag loss weights), and each calibration bag
    collapses over the *scoring* rows *score_rows* supplies rather than the
    rows it trained on (see :func:`_calibration_score_rows`); otherwise every
    row is its own bag and the path is byte-for-byte the pre-flood behaviour.
    *voted_ids* names the media in *clips_dict* the labels came from, so the
    fold-anchored estimator can drop them from its haystacks (issue #3308; see
    :func:`_fused_threshold`).
    Returns ``([], 0.5, None)`` when the labels don't satisfy ≥2 samples AND
    ≥1 good AND ≥1 bad.

    *rows* lets a caller that has **already** built this snapshot's score rows
    hand them in instead of having them rebuilt.  It is a pure cache
    pass-through - the rows must be
    :func:`scoring_rows_for_snap`'s output for *clips_dict* in the same space
    this function resolves - and exists for cross-dataset Find, which scores
    several detectors over one ``temp_medias`` that
    :func:`~vtscore.embedding.matrix.get_region_matrix_for_snap` will not cache
    (its cache is keyed to the *active* dataset context).  Without it each
    detector would restack the corpus.
    """
    import torch  # noqa: PLC0415

    from vtscore.training.mlp import LINEAR_SVM_HEAD, train_model  # noqa: PLC0415
    from vtscore.training.blend_schedules import BlendContext  # noqa: PLC0415
    from vtscore.training.thresholds import (  # noqa: PLC0415
        calibration_folds_cached,
        threshold_from_folds,
    )

    num_good = sum(1 for v in y_list if v == 1.0)
    num_bad = len(y_list) - num_good
    if len(X_list) < 2 or num_good == 0 or num_bad == 0:
        return [], 0.5, None

    # The detector's primary embedder (the explicit space it scores in), or the
    # dataset score precedence when the detector has no primary yet.  Scoring
    # reads vectors from this same space the X_list were assembled in.
    score_emb = detector_score_embedder(det_ctx, clips_dict)

    # ``None`` = no explicit user setting: take the per-space production split
    # for the embedder the detector learns in (issue #3287).
    calibration_fraction = resolve_calibration_fraction(calibration_fraction, score_emb)

    X = torch.from_numpy(stack_vectors(X_list, label="training vector"))
    y = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1)
    input_dim = X.shape[1]

    # Bag-aware setup (region flooding): size on votes not rows, split/weight
    # per bag when a Bad vote flooded its patch stack; a no-op on legacy datasets.
    _n_votes, cal_groups, sample_weights = _flood_context(X_list, y_list, groups)
    cal_score_rows = _calibration_score_rows(groups, cal_groups, score_rows)
    blend_ctx = BlendContext.from_labels(y_list, cal_groups)
    # Linear SVM production head - see train_and_threshold for why.
    hidden_dim = LINEAR_SVM_HEAD

    # K-fold calibration runs at every label count: the fold-anchored estimator
    # anchors on the fold models' held-out scores, so their models are an input
    # rather than something to skip.  (The pre-fusion path skipped the fold fits
    # below the blend schedule's floor, where the schedule multiplied the
    # cross-cal cut by zero; the schedule is no longer what combines the two
    # estimators.)
    folds = calibration_folds_cached(
        X_list,
        y_list,
        input_dim,
        calibrate_count=calibrate_count,
        calibration_fraction=calibration_fraction,
        hidden_dim=hidden_dim,
        det_ctx=det_ctx,
        groups=cal_groups,
        score_rows_by_group=cal_score_rows,
    )
    threshold = threshold_from_folds(folds, inclusion_value)

    # A Good vote trains on one row (the raw patch nearest the drawn box); a
    # Bad vote trains on the image's whole score-row stack (region flooding),
    # per-bag weighted so a rejected image counts once.  On a legacy dataset
    # there are no per-bag weights, so the call stays identical to the
    # historical one-vector-per-media fit.
    if sample_weights is not None:
        model = train_model(X, y, input_dim, hidden_dim=hidden_dim, sample_weights=sample_weights)
    else:
        model = train_model(X, y, input_dim, hidden_dim=hidden_dim)

    # One row build for the whole step: the final model's scoring pass and every
    # fold pass inside `_fused_threshold` read the same matrix (only the head
    # changes between them).  A caller that already holds that matrix passes it
    # in (see *rows* above) so a multi-detector pass over one snapshot builds it
    # once rather than once per head.
    if rows is None:
        rows = scoring_rows_for_snap(clips_dict, score_emb)
    all_ids = rows.ids
    scores, best_region = score_rows_with_model(model, rows)

    # The label counts feeding the fallback blend are votes, not flooded rows,
    # so its small-count ramp is unmoved by region flooding.
    threshold = _fused_threshold(
        threshold,
        folds,
        rows,
        scores,
        inclusion_value,
        blend_ctx,
        _blend_schedule_for_snap(clips_dict),
        det_ctx=det_ctx,
        final_ids=all_ids,
        voted_ids=voted_ids,
    )

    results = _format_results(all_ids, scores, best_region, clips_dict)
    return results, threshold, model


def train_and_score(
    clips_dict: dict[int, dict[str, Any]],
    good_votes: dict[int, None],
    bad_votes: dict[int, None],
    inclusion_value: int = 0,
    calibrate_count: int = 2,
    calibration_fraction: float | None = None,
    vote_region_boxes: dict[int, tuple[float, float, float, float]] | None = None,
    det_ctx: Any = None,
) -> tuple[list[dict[str, Any]], float, nn.Sequential | None]:
    """Train the detector head on voted media embeddings and score every media.

    Uses k-fold calibration to determine an appropriate decision threshold,
    then trains a final model on all labelled data and scores every media in
    ``clips_dict``.

    Args:
        clips_dict: Mapping of media ID to media data dict. Each value must carry
            a resolvable embedding vector in its per-embedder ``"embeddings"``
            dict store (read via ``media_embedding``).
        good_votes: Dict whose keys are media IDs labelled as good (values are ``None``).
        bad_votes: Dict whose keys are media IDs labelled as bad (values are ``None``).
        inclusion_value: Integer in ``[-10, 10]`` passed to the training and
            threshold-finding functions to control the inclusion/exclusion bias.
        calibrate_count: Number of random Train/Calibrate splits for threshold
            calibration (default 2).
        calibration_fraction: Fraction of labelled data reserved for calibration
            in each split.  For example, 0.2 means 80% Train / 20% Calibrate.
            ``None`` (default) resolves to the per-space production split for
            the detector's embedder - 0.3 single-vector, 0.5 patch (see
            :func:`resolve_calibration_fraction`).
        vote_region_boxes: Optional ``media_id -> (x0, y0, x1, y1)`` map from
            yes-votes that designated a region.  When set and the source
            media carries a ``patch_grid``, the raw patch nearest the box
            (:func:`vtscore.media.patch_embed.nearest_patch_to_box`) trains the
            vote instead of ``media["embeddings"]``.  Falls back to the
            full-image vector when the media lacks a patch grid (legacy
            datasets, single-vector embedders) or the box is missing.

    Returns:
        A tuple ``(results, threshold, model)`` where:

        - ``results`` is a list of ``{"id": int, "score": float}`` dicts, sorted
          by score in descending order (highest confidence first).
        - ``threshold`` is the decision boundary as a float: the fold-anchored
          population cut fitted on ``clips_dict``'s score distribution (see
          :func:`_fused_threshold`).
        - ``model`` is the trained ``nn.Sequential`` model (``None`` when
          training was not possible).
    """
    region_boxes = vote_region_boxes or {}
    X_list, y_list, groups, score_rows = _build_vote_xy(
        clips_dict, good_votes, bad_votes, region_boxes, detector_score_embedder(det_ctx, clips_dict)
    )
    results, threshold, model = _train_and_score_xy(
        X_list,
        y_list,
        clips_dict,
        inclusion_value=inclusion_value,
        calibrate_count=calibrate_count,
        calibration_fraction=calibration_fraction,
        det_ctx=det_ctx,
        groups=groups,
        score_rows=score_rows,
        voted_ids=set(good_votes) | set(bad_votes),
    )

    # Stage-2 structural re-rank: a no-op for every non-structural dataset
    # (gated on media carrying ``local_features``), so existing datasets are
    # untouched.  For a structural (SIFT/VLAD) dataset it geometrically
    # verifies the VLAD shortlist against the RegionYes templates and re-ranks
    # by the match-statistic classifier (or the cold-start inlier gate).  See
    # docs/plans/structural-embedder.md.
    from vtscore.training.structural_similarity import maybe_structural_rerank  # noqa: PLC0415

    results, threshold = maybe_structural_rerank(
        results, threshold, clips_dict, good_votes, bad_votes, region_boxes, det_ctx
    )
    return results, threshold, model


# ---------------------------------------------------------------------------
# Origin-based helpers (for weight-free detector serialisation)
# ---------------------------------------------------------------------------


def collect_media_origins(
    media_ids: dict[int, None] | list[int],
    snap: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Collect origin info for a set of media IDs from a medias snapshot.

    Each returned dict contains ``origin``, ``origin_name``, ``filename``,
    and ``md5`` - enough to re-resolve the original file later.

    Args:
        media_ids: Media IDs (keys of a votes dict, or a plain list).
        snap: Snapshot of all loaded medias (from :func:`snapshot_medias`).

    Returns:
        A list of origin dicts, one per matched media.
    """
    origins: list[dict[str, Any]] = []
    for cid in media_ids:
        if cid not in snap:
            continue
        media = snap[cid]
        origins.append(
            {
                "origin": media.get("origin"),
                "origin_name": media.get("origin_name", ""),
                "filename": media.get("filename", ""),
                "md5": media.get("md5", ""),
            }
        )
    return origins


def train_detector_from_origins(
    good_origins: list[dict[str, Any]],
    bad_origins: list[dict[str, Any]],
    inclusion: int,
    media_type: str,
    embedder_name: str,
    calibrate_count: int = 2,
    calibration_fraction: float | None = None,
) -> tuple[dict[str, list] | None, float]:
    """Resolve origin entries to files, embed them, and train a detector head.

    This is the load-time counterpart of file-based detector export: given
    the origin lists that were saved to disk, it re-derives the head weights
    by resolving the original media files, embedding them, and training.

    **Who calls this: library consumers, not the app.**  Nothing in
    ``vtsearch`` reaches here.  The app re-derives a saved detector through
    ``POST /api/detectors/registry/load`` ->
    :func:`vtscore.detectors.labelset_training.train_from_labelset` ->
    :func:`train_and_threshold`, which is handed the active dataset's medias
    as a haystack and therefore ships the **fold-anchored** cut, exactly as a
    freshly trained detector does (pinned by
    ``tests/detectors/test_load_time_threshold_provenance.py``).  So the plain
    cross-calibration cut below is what a *library* caller with no haystack
    gets - it is not the threshold a resumed VTSearch session starts on
    (issue #3257, evaluated and closed unrun).

    Args:
        good_origins: Origin dicts for media labelled Good.
        bad_origins: Origin dicts for media labelled Bad.
        inclusion: The inclusion value to use for training.
        media_type: Media type string (e.g. ``"audio"``, ``"image"``).
        embedder_name: Name of the embedder the detector was originally
            trained with. Passed through to :func:`embed_file` so every
            re-embedded media is encoded by the same model that produced
            the saved vectors - otherwise the MLP trains on a mix of
            embedder outputs and learns garbage. Pass ``""`` only when you
            genuinely want the media type's default embedder (e.g. a
            brand-new detector with no recorded embedder yet).
        calibrate_count: Number of k-fold calibration splits.
        calibration_fraction: Fraction reserved for calibration.  ``None``
            (default) resolves to the per-space production split for
            *embedder_name* (see :func:`resolve_calibration_fraction`).

    Returns:
        A ``(weights, threshold)`` tuple.  ``weights`` is ``None`` if
        resolution/embedding failed for too many entries (need at least
        one good and one bad).
    """
    import torch  # noqa: PLC0415

    from vtscore.detectors.resolver import embed_file, resolve_file_context
    from vtscore.training.mlp import LINEAR_SVM_HEAD, train_model
    from vtscore.training.thresholds import calculate_cross_calibration_threshold

    X_list: list = []
    y_list: list[float] = []

    for entry in good_origins:
        with resolve_file_context(
            entry.get("origin"),
            entry.get("origin_name", ""),
            entry.get("filename", ""),
        ) as file_path:
            if file_path is None:
                continue
            emb = embed_file(file_path, media_type, embedder_name)
        if emb is None:
            continue
        X_list.append(emb)
        y_list.append(1.0)

    for entry in bad_origins:
        with resolve_file_context(
            entry.get("origin"),
            entry.get("origin_name", ""),
            entry.get("filename", ""),
        ) as file_path:
            if file_path is None:
                continue
            emb = embed_file(file_path, media_type, embedder_name)
        if emb is None:
            continue
        X_list.append(emb)
        y_list.append(0.0)

    num_good = sum(1 for v in y_list if v == 1.0)
    num_bad = len(y_list) - num_good
    if len(X_list) < 2 or num_good == 0 or num_bad == 0:
        return None, 0.5

    X = torch.from_numpy(stack_vectors(X_list, label="training vector"))
    y = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1)
    input_dim = X.shape[1]

    # Plain cross-calibration: this load-time path re-derives a detector from
    # saved origins and has no haystack to fuse against, so there is no
    # population estimator to fit and the conformal cut ships alone.  The trainer
    # degrades gracefully below 4 labels / <2-per-class via its own 0.5
    # fallback, so no separate small-label short-circuit is needed here.
    threshold = calculate_cross_calibration_threshold(
        X_list,
        y_list,
        input_dim,
        inclusion,
        calibrate_count=calibrate_count,
        calibration_fraction=resolve_calibration_fraction(calibration_fraction, embedder_name),
        hidden_dim=LINEAR_SVM_HEAD,
    )
    model = train_model(X, y, input_dim, hidden_dim=LINEAR_SVM_HEAD)

    state_dict = model.state_dict()
    weights = {k: v.cpu().tolist() for k, v in state_dict.items()}
    return weights, threshold
