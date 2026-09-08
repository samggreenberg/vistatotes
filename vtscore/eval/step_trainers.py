"""Per-step trainers for the voting-iterations eval, and the pool scorers.

Everything here answers one of two questions the eval loop asks at every step:
*what does the pool look like right now* (:func:`_score_pool`,
:func:`_labelset_error_costs`, :func:`_build_eval_atlas`) and *what model do
these votes produce* (the ``*_train_and_calibrate`` family).  Each trainer takes
the votes so far and returns a :class:`~vtscore.eval.step_model.StepModel`, its
threshold, and the timings the result rows record, so the loop can swap one for
another without knowing anything about the backend behind it.

:func:`_style_train_and_calibrate` is the **production-faithful** trainer and is
pinned against ``vtscore.detectors.training.train_and_threshold`` by
`scripts/check-eval-app-sync.py`; the MLP and SVM trainers beside it are
experiment arms.  See the "Eval Default Arm IS the App" rule before changing
what the default resolves to.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

from vtscore.embedding.media_vectors import media_embedding
from vtscore.eval.labels import region_box_for_category
from vtscore.eval.calibration_metrics import inclusion_weights
from vtscore.eval.step_model import (
    PRODUCTION_HEAD,
    StepModel,
    good_training_vec,
    resolve_hidden_dim,
    score_sim_set_with_model,
)
from vtscore.eval.trainers import _cross_calibrated_threshold, _parse_trainer_spec
from vtscore.training.mlp import train_model
from vtscore.training.thresholds import (
    calibration_folds,
    classify_threshold_provenance,
    compute_fold_orderings,
    compute_grouped_fold_node_scores,
    threshold_from_fold_orderings,
    threshold_from_folds,
)


def _score_pool(
    step: StepModel,
    pool_ids: list[int],
    clips_dict: dict[int, dict[str, Any]],
    *,
    region_aware: bool = False,
    style_obj: Any = None,
    sim_clips: dict[int, dict[str, Any]] | None = None,
    sim_scored: tuple[list[int], list[float]] | None = None,
) -> dict[int, float]:
    """Return ``{pool_id: score}`` for the current model over the pool.

    **In the same score space the thresholds are cut in** (issue #2943).  That
    is not a refinement, it is a correctness requirement: the Hard pick locates
    its cutoff with the *absolute* comparison ``ranking[cid] <= threshold``
    (:func:`~vtscore.eval.al_strategies._hard_pick_by_index`), so a ranking and
    a cut that live in different spaces put the cutoff index in the wrong place.
    On a patch dataset the reporting/acquisition cuts are fitted on the style's
    region max-pooled scores, and a max over ~197 patch rows stochastically
    dominates the single whole-image row - so scoring the pool whole-image would
    depress every pool score relative to the cut and drag the cutoff index
    systematically toward the top of the ranking.  The app has no such gap: its
    learned sort ranks the very same pooled scores its threshold cuts.

    Three paths, mirroring :func:`_evaluate_on_test` / :func:`score_sim_set_with_model`:

    * *sim_scored* - the ``(ids, scores)`` the safe-threshold step already
      computed over the whole simulation set, in exactly this geometry.  The
      pool is a subset of that set, so restricting it is free and removes the
      scoring pass entirely.
    * a *style_obj* / *region_aware* dataset with no such scores (the
      ``safe_thresholds=False`` control arm) - score through the style.  The
      **full** sim set is scored rather than just the pool: the style memoises
      its flattened patch matrix per media-id set, and the pool loses an item
      every step, so scoring the shrinking pool would re-flatten from scratch
      each step *and* leak a cache entry per step.
    * everything else (single-vector datasets, the SVM arms) - the trainer-
      agnostic whole-image ``predict``, which is already the threshold's space.
    """
    import numpy as np  # noqa: PLC0415

    if not pool_ids:
        return {}
    if sim_scored is not None:
        ids, scores = sim_scored
        pool_set = set(pool_ids)
        return {cid: float(s) for cid, s in zip(ids, scores, strict=True) if cid in pool_set}
    if (style_obj is not None or region_aware) and sim_clips:
        assert step.torch_model is not None
        pool_set = set(pool_ids)
        ids, scores = score_sim_set_with_model(
            step.torch_model, region_aware, sim_clips, None, sorted(sim_clips), style_obj
        )
        return {cid: float(s) for cid, s in zip(ids, scores, strict=True) if cid in pool_set}
    embs = np.array([media_embedding(clips_dict[cid]) for cid in pool_ids])
    scores = np.asarray(step.predict(embs)).ravel().tolist()
    return dict(zip(pool_ids, scores, strict=True))


def _labelset_error_costs(
    model_steps: list[tuple[Any, float]],
    good_votes: dict[int, None],
    bad_votes: dict[int, None],
    clips_dict: dict[int, dict[str, Any]],
    inclusion: int,
    *,
    region_aware: bool = False,
    style_obj: Any = None,
) -> list[float]:
    """Weighted FPR/FNR of **every** recent model on the current labelled set.

    Feeds the Smart indicator.  Reproduces ``labeling_progress._eval_cached_models``
    /``_score_step`` - and shares their arithmetic, via
    :func:`~vtscore.training.thresholds.weighted_error_cost`, so only the input
    plumbing differs: every model in the window is re-scored against the
    *current* labelset — the only ground truth the app has — with its own cached
    threshold, so all points of the slope regression share one eval set and the
    trend isolates model improvement.  Scoring each model against the labelset
    it was trained on instead would confound model change with labelset growth:
    autopilot deliberately votes boundary items, which are mispredicted at first
    and inflate the later costs of a frozen-cost history.

    **In the arm's own scoring geometry** (issue #3757), for the same reason
    :func:`_score_pool` is: a MaxPatch head is trained on a Good vote's boxed
    patch and against every row of a Bad vote, and served by a max over those
    rows, so scoring it on whole-image vectors measures a geometry it was never
    fitted for - and the app, whose Smart curve this arm has to *be*, now scores
    its cached heads through
    :func:`~vtscore.detectors.training.scoring_rows_for_snap`.  Whole-image arms
    keep the trainer-agnostic ``predict``, which is already their space.

    Deliberately *not* the held-out test split: those labels must never reach
    the vote order.  Returns ``[]`` when the labelset has no usable eval set
    (either class empty), matching ``_eval_cached_models``, which leaves the
    Smart indicator on its "not enough points" branch.
    """
    import numpy as np  # noqa: PLC0415

    from vtscore.training.thresholds import weighted_error_cost  # noqa: PLC0415

    ids = list(good_votes) + list(bad_votes)
    labels = [1.0] * len(good_votes) + [0.0] * len(bad_votes)
    if not model_steps or not ids or not good_votes or not bad_votes:
        return []

    fpr_weight, fnr_weight = inclusion_weights(inclusion)

    if style_obj is not None or region_aware:
        # The labelled subset only - the app builds its eval rows over exactly
        # the voted media too.  The style memoises the flattened stack per
        # media-id set, so the window's models share one build; the memo is
        # LRU-bounded, so the id set growing by one each step costs a re-flatten
        # rather than a retained matrix per step.
        labelled = {cid: clips_dict[cid] for cid in ids if cid in clips_dict}
        costs = []
        for step, threshold in model_steps:
            assert step.torch_model is not None
            scored_ids, scored = score_sim_set_with_model(
                step.torch_model, region_aware, labelled, None, sorted(labelled), style_obj
            )
            by_id = dict(zip(scored_ids, scored, strict=True))
            scores = np.array([by_id[cid] for cid in ids if cid in by_id], dtype=np.float64)
            kept = [lbl for cid, lbl in zip(ids, labels, strict=True) if cid in by_id]
            costs.append(weighted_error_cost(scores, kept, threshold, fpr_weight, fnr_weight)[0])
        return costs

    # One eval matrix, reused by every model in the window - the app's
    # ``_build_eval_rows`` builds its rows once for the same reason.
    embs = np.array([media_embedding(clips_dict[cid]) for cid in ids])

    costs = []
    for step, threshold in model_steps:
        scores = np.asarray(step.predict(embs)).ravel()
        costs.append(weighted_error_cost(scores, labels, threshold, fpr_weight, fnr_weight)[0])
    return costs


def _build_eval_atlas(embeddings: dict[int, np.ndarray], min_node_size: int) -> Any:
    """Build a coverage atlas over *embeddings* for the autopilot New phase.

    Returns ``None`` when there are no vectors.  Uses the same hierarchical
    k-means partition the live dataset builds (see
    :class:`~vtscore.coverage.atlas.CoverageAtlas`); *min_node_size* is
    exposed so a caller with a small simulation set can drive the partition
    deeper than the production floor (20) and actually resolve density cells.
    """
    from vtscore.coverage.atlas import CoverageAtlas, auto_max_depth  # noqa: PLC0415

    if not embeddings:
        return None
    return CoverageAtlas(
        embeddings,
        k=3,
        max_depth=auto_max_depth(len(embeddings), k=3, min_node_size=min_node_size),
        min_node_size=min_node_size,
    )


def _train_and_calibrate(
    trainer: str,
    good_votes: dict[int, None],
    bad_votes: dict[int, None],
    clips_dict: dict[int, dict[str, Any]],
    target_category: str,
    *,
    region_voting: bool,
    input_dim: int,
    inclusion: int,
    calibrate_count: int,
    calibration_fraction: float,
    head: str = PRODUCTION_HEAD,
    style_obj: Any = None,
    emit_calibration_metrics: bool = False,
    fold_count_variants: list[int] | None = None,
) -> tuple[StepModel, float, int, dict[str, float], dict[str, Any]]:
    """Train the step's ranker and calibrate its threshold from the current votes.

    *head* selects the head on both production paths (see :data:`HEADS`):
    ``"linear_svm"`` (the default, :data:`PRODUCTION_HEAD`) trains the head the
    live detector has, ``"linear"`` the logistic head it replaced, ``"mlp"`` the
    legacy auto-sized hidden layer.  It is ignored by the standalone SVM path,
    which fits its own estimator rather than a head.

    Dispatches on *trainer*: ``"mlp"`` runs the production MLP path unchanged
    (see :func:`_mlp_train_and_calibrate`); any ``svm_*`` name runs the SVM path
    (see :func:`_svm_train_and_calibrate`).  Returns ``(step, threshold,
    n_labels, timings, details)`` where *timings* has ``train_seconds`` and
    ``xcal_seconds`` for the fit and threshold-calibration wall clocks, and
    *details* is empty unless *emit_calibration_metrics* (the #2781 study),
    carrying the fold orderings, node scores, and threshold provenance the
    calibration metrics need.

    With an explicit *style_obj* (MLP only) the vote-to-vector assembly is
    delegated to the style (see :func:`_style_train_and_calibrate`).
    """
    if style_obj is not None:
        return _style_train_and_calibrate(
            style_obj,
            good_votes,
            bad_votes,
            clips_dict,
            target_category,
            region_voting=region_voting,
            input_dim=input_dim,
            inclusion=inclusion,
            calibrate_count=calibrate_count,
            calibration_fraction=calibration_fraction,
            head=head,
            emit_calibration_metrics=emit_calibration_metrics,
            fold_count_variants=fold_count_variants,
        )
    if trainer == "mlp":
        return _mlp_train_and_calibrate(
            good_votes,
            bad_votes,
            clips_dict,
            target_category,
            region_voting=region_voting,
            input_dim=input_dim,
            inclusion=inclusion,
            calibrate_count=calibrate_count,
            calibration_fraction=calibration_fraction,
            head=head,
        )
    return _svm_train_and_calibrate(
        trainer,
        good_votes,
        bad_votes,
        clips_dict,
        target_category,
        inclusion=inclusion,
        calibrate_count=calibrate_count,
        calibration_fraction=calibration_fraction,
    )


def _mlp_train_and_calibrate(
    good_votes: dict[int, None],
    bad_votes: dict[int, None],
    clips_dict: dict[int, dict[str, Any]],
    target_category: str,
    *,
    region_voting: bool,
    input_dim: int,
    inclusion: int,
    calibrate_count: int,
    calibration_fraction: float,
    head: str = PRODUCTION_HEAD,
) -> tuple[StepModel, float, int, dict[str, float], dict[str, Any]]:
    """The production arm — numerically identical to the pre-trainer harness at ``head="mlp"``.

    At ``head="linear_svm"`` (the default, :data:`PRODUCTION_HEAD`) this trains
    the live detector's head: production pins the linear SVM on every fit (see
    ``vtscore.training.mlp.LINEAR_SVM_HEAD``), so the reported thresholds and
    costs are the shipped detector's.  ``head="linear"`` (the logistic head the
    SVM replaced) and ``head="mlp"`` (the small-MLP candidate #2781 measured)
    are the named legacy arms.  Everything *around* the head mirrors the
    production ``_train_and_score_xy`` / ``train_and_threshold`` pipeline
    whichever is chosen:

    Good votes region-pool their ground-truth box when *region_voting* is on
    (and the media supports it); Bad votes always train on the whole-image
    vector.

    **This is the single-vector path.**  Bad votes here are one row because a
    single-vector media *has* one row - not because the live detector works that
    way.  On a patch dataset the live detector floods a Bad vote over the
    image's whole score-row stack, and
    :func:`simulate_voting_iterations` routes such datasets to the
    ``max_patch`` style (:func:`_style_train_and_calibrate`) rather than here,
    so the default arm matches the app.  Do not "restore" whole-image Bad votes
    on patch data: that trains ~196 rows per rejected image down never while
    inference max-pools them.

    * ``hidden_dim`` comes from the head (sized from the *full* label count on
      the MLP head, 0 on the linear one) and is forced onto the
      calibration folds, so the fold models share the final model's architecture
      (production likewise threads one width into
      ``cross_calibration_threshold_cached``).  Letting each fold auto-size to
      its own smaller train split would train narrower fold nets and report a
      threshold no single-architecture pipeline ever produces.
    * the fold splits use a fresh ``RandomState(42)`` - the fixed seed
      ``cross_calibration_threshold_cached`` always calibrates with - rather than
      the shared per-seed simulation RNG, so the calibration is byte-for-byte
      what production runs for this vote set.  The eval seed still varies the
      data (which media are voted, in what order, and the held-out test split);
      only the calibration folds are pinned, as they are in production.
    """
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    for vid in good_votes:
        X_list.append(good_training_vec(clips_dict[vid], target_category, region_voting))
        y_list.append(1.0)
    for vid in bad_votes:
        X_list.append(media_embedding(clips_dict[vid]))
        y_list.append(0.0)

    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    y = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1)
    n_labels = len(good_votes) + len(bad_votes)

    hidden_dim = resolve_hidden_dim(head, n_labels)
    t_xcal = time.monotonic()
    # The folds' orderings *and* models ride out in ``details`` unconditionally:
    # the shipped safe threshold anchors on the fold models' held-out scores, so
    # they are an input to the baseline arm, not study-only extras.
    folds = calibration_folds(
        X_list,
        y_list,
        input_dim,
        calibrate_count=calibrate_count,
        calibration_fraction=calibration_fraction,
        hidden_dim=hidden_dim,
        rng=np.random.RandomState(42),
    )
    threshold = threshold_from_folds(folds, inclusion)
    xcal_seconds = time.monotonic() - t_xcal
    t_train = time.monotonic()
    model = train_model(X, y, input_dim, hidden_dim=hidden_dim)
    train_seconds = time.monotonic() - t_train

    device = str(next(model.parameters()).device)

    def predict(X_test: Any) -> np.ndarray:
        with torch.no_grad():
            t = torch.tensor(np.asarray(X_test), dtype=torch.float32).to(next(model.parameters()).device)
            return torch.sigmoid(model(t)).squeeze(1).cpu().numpy()

    step = StepModel(
        predict=predict,
        torch_model=model,
        backend="torch-cuda" if device.startswith("cuda") else "torch-cpu",
        device=device,
    )
    details = {
        "fold_orderings": folds.orderings,
        "fold_models": folds.models,
        # Which sentinel (if any) the fold rule returned: the blend's x-cal side
        # is NO_GOOD_THRESHOLD whenever this is set, as production's does
        # (see :func:`_blend_xcal_input`).
        "fold_fallback": folds.fallback,
    }
    return step, threshold, n_labels, {"train_seconds": train_seconds, "xcal_seconds": xcal_seconds}, details


def _style_train_and_calibrate(
    style_obj: Any,
    good_votes: dict[int, None],
    bad_votes: dict[int, None],
    clips_dict: dict[int, dict[str, Any]],
    target_category: str,
    *,
    region_voting: bool,
    input_dim: int,
    inclusion: int,
    calibrate_count: int,
    calibration_fraction: float,
    head: str = PRODUCTION_HEAD,
    emit_calibration_metrics: bool = False,
    fold_count_variants: list[int] | None = None,
) -> tuple[StepModel, float, int, dict[str, float], dict[str, Any]]:
    """Style-driven torch path (the Max-Patch experiment arms).

    The detection style (see :mod:`vtscore.eval.patch_styles`) supplies the
    vote-to-vector rules: each Good vote contributes ``style.good_vec`` (given
    the ground-truth box when *region_voting* and the media has one), each Bad
    vote floods ``style.bad_vecs`` - one row on a whole-image style, the
    image-level vector + every raw patch on ``max_patch``, every tree node on
    the HAC hybrids.

    Training and calibration are **bag-aware**, exactly like the production
    vote path (:func:`vtscore.detectors.training._train_and_score_xy`): the
    head (see :data:`HEADS`) and the safe-threshold ramp size on distinct *votes* rather
    than flooded rows, the calibration folds split by bag, and the final fit
    weights each bag equally.  On a whole-image style every bag is one row, so
    this collapses to the historical single-vector behaviour.

    Calibration additionally runs in **inference geometry**: each bag is handed
    its ``style.score_rows`` stack so a Good bag collapses the same way a Bad
    bag (and every held-out image) does.  Without this a Good bag is a max over
    its 1 training row while a Bad bag is a max over the ~197 rows it flooded,
    and the calibrated cut lands above the score range production actually
    produces - see :func:`vtscore.training.thresholds.compute_fold_orderings`.
    """
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    from vtscore.detectors.training import _flood_context  # noqa: PLC0415

    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    groups: list = []
    score_rows_by_group: dict = {}
    for vid in good_votes:
        box = region_box_for_category(clips_dict[vid], target_category) if region_voting else None
        X_list.append(np.asarray(style_obj.good_vec(clips_dict[vid], box), dtype=np.float32))
        y_list.append(1.0)
        groups.append(("g", vid))
        score_rows_by_group[("g", vid)] = style_obj.score_rows(clips_dict[vid])
    for vid in bad_votes:
        for vec in style_obj.bad_vecs(clips_dict[vid]):
            X_list.append(np.asarray(vec, dtype=np.float32))
            y_list.append(0.0)
            groups.append(("b", vid))
        score_rows_by_group[("b", vid)] = style_obj.score_rows(clips_dict[vid])

    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    y = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1)
    n_votes, cal_groups, sample_weights = _flood_context(X_list, y_list, groups)

    hidden_dim = resolve_hidden_dim(head, n_votes)
    t_xcal = time.monotonic()
    details: dict[str, Any] = {}
    if emit_calibration_metrics:
        threshold, details = _calibrate_with_details(
            X_list,
            y_list,
            input_dim,
            inclusion,
            calibrate_count=calibrate_count,
            calibration_fraction=calibration_fraction,
            hidden_dim=hidden_dim,
            cal_groups=cal_groups,
            score_rows_by_group=score_rows_by_group if cal_groups is not None else None,
            fold_count_variants=fold_count_variants,
        )
        # Bad-voted bags' inference row stacks: the final model scores these to
        # form the pnorm null (F_neg) at test time (see _calibration_metric_rows).
        details["neg_score_rows"] = [score_rows_by_group[("b", vid)] for vid in bad_votes]
    else:
        # Same fold work as the metrics branch, minus the study extras: the
        # shipped safe threshold anchors on the fold models, so they ride out
        # in ``details`` on every path (see :func:`_safe_threshold_for_step`).
        folds = calibration_folds(
            X_list,
            y_list,
            input_dim,
            calibrate_count=calibrate_count,
            calibration_fraction=calibration_fraction,
            hidden_dim=hidden_dim,
            rng=np.random.RandomState(42),
            groups=cal_groups,
            score_rows_by_group=score_rows_by_group if cal_groups is not None else None,
        )
        threshold = threshold_from_folds(folds, inclusion)
        details = {
            "fold_orderings": folds.orderings,
            "fold_models": folds.models,
            "fold_fallback": folds.fallback,
        }
    xcal_seconds = time.monotonic() - t_xcal
    # Under the #2897 screen this step trained Kmax folds, not ``calibrate_count``
    # of them.  Bill the reported wall clock for the live count only, so the
    # baseline row's timing stays the one an uninstrumented run would report; the
    # per-K costs live in each fold-count arm's own ``fold_seconds``.
    extra = (details.get("fold_count_data") or {}).get("seconds")
    if extra:
        xcal_seconds -= sum(extra[calibrate_count:])
    t_train = time.monotonic()
    if sample_weights is not None:
        model = train_model(X, y, input_dim, hidden_dim=hidden_dim, sample_weights=sample_weights)
    else:
        model = train_model(X, y, input_dim, hidden_dim=hidden_dim)
    train_seconds = time.monotonic() - t_train

    device = str(next(model.parameters()).device)

    def predict(X_test: Any) -> np.ndarray:
        with torch.no_grad():
            t = torch.tensor(np.asarray(X_test), dtype=torch.float32).to(next(model.parameters()).device)
            return torch.sigmoid(model(t)).squeeze(1).cpu().numpy()

    step = StepModel(
        predict=predict,
        torch_model=model,
        backend="torch-cuda" if device.startswith("cuda") else "torch-cpu",
        device=device,
    )
    return step, threshold, n_votes, {"train_seconds": train_seconds, "xcal_seconds": xcal_seconds}, details


def _calibrate_with_details(
    X_list: list[np.ndarray],
    y_list: list[float],
    input_dim: int,
    inclusion: int,
    *,
    calibrate_count: int,
    calibration_fraction: float,
    hidden_dim: int | None,
    cal_groups: list | None,
    score_rows_by_group: dict | None,
    fold_count_variants: list[int] | None = None,
) -> tuple[float, dict[str, Any]]:
    """Compute the trained threshold **and** the calibration study's provenance.

    Replaces the plain :func:`calculate_cross_calibration_threshold` call on the
    style path when the #2781 metrics are requested.  Trains the calibration
    folds exactly once and returns ``(threshold, details)`` where *details* holds:

    * ``provenance`` — which code path set the threshold (``conformal`` /
      ``no_good_sentinel`` / ``too_few_default``), via
      :func:`~vtscore.training.thresholds.classify_threshold_provenance`.
    * ``fold_orderings`` — the pooled ``(scores, labels)`` per fold under the
      base (max) pooling, for the calibration-set oracle and the inclusion sweep.
    * ``fold_node_data`` — per-fold, per-group **node** scores (grouped path
      only), so a remedial pooling variant can recalibrate off the same fold
      models without retraining; ``None`` on the row-wise (whole-image) path.
    * ``fold_fallback`` — the sentinel the fold rule returned, or ``None`` when
      the folds are real.  The shipped blend substitutes ``NO_GOOD_THRESHOLD``
      for the x-cal side whenever this is set, as production does (see
      :func:`_blend_xcal_input`).
    * ``fold_count_data`` — only under *fold_count_variants* (issue #2897): the
      **full** Kmax fold orderings, their per-fold seconds, and the
      count-independent overhead, for :func:`_fold_count_variant_rows`.

    On the grouped path the fold models are trained once via
    :func:`~vtscore.training.thresholds.compute_grouped_fold_node_scores` and the
    base orderings are the max-pool of the node data, so the threshold is
    identical to what production's grouped calibration produces for this arm.

    *fold_count_variants* raises the number of folds actually trained to
    ``max(calibrate_count, *variants)`` while leaving everything the step
    returns computed off the first ``calibrate_count`` of them.  That is exact,
    not an approximation: the folds are nested (see
    :func:`~vtscore.training.thresholds.compute_fold_orderings`) and
    ``train_model`` is seeded per call, so the extra folds cannot perturb the
    live threshold, the fold models, or the trajectory - they only cost time.
    """
    import numpy as np  # noqa: PLC0415

    k_max = max(calibrate_count, *(fold_count_variants or [calibrate_count]))
    t_folds = time.monotonic()
    fold_seconds: list[float] = []

    def _with_fold_data(details: dict[str, Any], orderings: list) -> dict[str, Any]:
        """Attach the fold-count screen's inputs and trim *details* to K live folds."""
        if fold_count_variants:
            details["fold_count_data"] = {
                "orderings": orderings,
                # The **untrimmed** fold models, so the #3116 anchored arm can
                # re-fit production's rule at every K.  `details["fold_models"]`
                # is deliberately cut to the live count so nothing downstream
                # can accidentally widen the shipped threshold's own fit.
                "models": list(fold_models),
                "seconds": fold_seconds,
                # Everything in the calibration wall clock that is *not* a fold
                # fit (the pooled conformal rule, the node max-pool): paid once
                # at every K, so it belongs in each arm's cost.
                "overhead_seconds": max(0.0, (time.monotonic() - t_folds) - sum(fold_seconds)),
            }
        return details

    # The trained fold models ride along in details["fold_models"] so the
    # #2852 fold-anchored arm can score the haystack on each fold's own scale
    # without retraining; production callers never see them.
    fold_models: list = []
    if cal_groups is not None:
        fold_node_data, fallback = compute_grouped_fold_node_scores(
            X_list,
            y_list,
            input_dim,
            groups=cal_groups,
            rng=np.random.RandomState(42),
            calibrate_count=k_max,
            calibration_fraction=calibration_fraction,
            hidden_dim=hidden_dim,
            score_rows_by_group=score_rows_by_group,
            model_sink=fold_models,
            seconds_sink=fold_seconds,
        )
        if fallback is not None:
            return fallback, {
                "provenance": classify_threshold_provenance(fallback),
                "fold_orderings": [],
                "fold_node_data": None,
                "fold_models": [],
                "fold_fallback": fallback,
            }
        # Base (max) orderings from the same fold node data -> identical to
        # production's grouped calibration for this arm.
        all_orderings = [([float(np.max(b)) for b in blocks], labels) for blocks, labels in fold_node_data]
        fold_orderings = all_orderings[:calibrate_count]
        threshold = threshold_from_fold_orderings(fold_orderings, inclusion)
        return threshold, _with_fold_data(
            {
                "provenance": classify_threshold_provenance(None),
                "fold_orderings": fold_orderings,
                "fold_node_data": fold_node_data[:calibrate_count],
                "fold_models": fold_models[:calibrate_count],
                "fold_fallback": None,
            },
            all_orderings,
        )

    # Row-wise path (whole-image styles): no bag flooding, no node re-pooling.
    all_orderings, fallback = compute_fold_orderings(
        X_list,
        y_list,
        input_dim,
        rng=np.random.RandomState(42),
        calibrate_count=k_max,
        calibration_fraction=calibration_fraction,
        hidden_dim=hidden_dim,
        model_sink=fold_models,
        seconds_sink=fold_seconds,
    )
    if fallback is not None:
        return fallback, {
            "provenance": classify_threshold_provenance(fallback),
            "fold_orderings": [],
            "fold_node_data": None,
            "fold_models": [],
            "fold_fallback": fallback,
        }
    fold_orderings = all_orderings[:calibrate_count]
    threshold = threshold_from_fold_orderings(fold_orderings, inclusion)
    return threshold, _with_fold_data(
        {
            "provenance": classify_threshold_provenance(None),
            "fold_orderings": fold_orderings,
            "fold_node_data": None,
            "fold_models": fold_models[:calibrate_count],
            "fold_fallback": None,
        },
        all_orderings,
    )


def _svm_train_and_calibrate(
    trainer: str,
    good_votes: dict[int, None],
    bad_votes: dict[int, None],
    clips_dict: dict[int, dict[str, Any]],
    target_category: str,
    *,
    inclusion: int,
    calibrate_count: int,
    calibration_fraction: float,
) -> tuple[StepModel, float, int, dict[str, float], dict[str, Any]]:
    """SVM path — single-vector only (the experiment never region-votes an SVM).

    Threshold uses the trainer-agnostic cross-calibration port
    (:func:`vtscore.eval.trainers._cross_calibrated_threshold`) — the natural
    analogue of the MLP's production calibration — with the fold models pinned
    to the sklearn CPU backend (they are tiny and only feed the threshold, so
    paying GPU launch overhead per fold would be wasteful).  The *final* fit
    honours the ambient backend (cuML on a GPU unless ``VTSEARCH_DISABLE_CUML``
    forces sklearn), and that backend is what the row records and what produces
    the scores.  The SVM fit seed is pinned to 42, mirroring the MLP's fixed
    calibration seed; the eval seed still varies which items are voted.
    """
    import numpy as np  # noqa: PLC0415

    from vtscore.eval.trainers import _train_svm_factory  # noqa: PLC0415
    from vtscore.training.svm import train_svm  # noqa: PLC0415

    X = np.array(
        [media_embedding(clips_dict[vid]) for vid in good_votes]
        + [media_embedding(clips_dict[vid]) for vid in bad_votes],
        dtype=np.float32,
    )
    y = np.array([1] * len(good_votes) + [0] * len(bad_votes), dtype=np.int32)
    n_labels = len(good_votes) + len(bad_votes)

    kernel, kwargs = _parse_trainer_spec(trainer)

    # Fold models for the threshold are pinned to sklearn CPU (tiny fits).
    fold_trainer = _train_svm_factory(kernel, backend="sklearn", **kwargs)
    t_xcal = time.monotonic()
    threshold = _cross_calibrated_threshold(
        X,
        y,
        fold_trainer,
        42,
        inclusion_value=inclusion,
        calibrate_count=calibrate_count,
        cal_fraction=calibration_fraction,
    )
    xcal_seconds = time.monotonic() - t_xcal

    t_train = time.monotonic()
    clf = train_svm(X, y, kernel=kernel, inclusion_value=inclusion, seed=42, **kwargs)  # type: ignore[arg-type]
    train_seconds = time.monotonic() - t_train

    step = StepModel(
        predict=clf.predict_proba,
        torch_model=None,
        backend=clf.backend,
        device="cuda" if clf.backend == "cuml" else "cpu",
    )
    return step, threshold, n_labels, {"train_seconds": train_seconds, "xcal_seconds": xcal_seconds}, {}
