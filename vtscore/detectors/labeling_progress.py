"""Labeling-session analyzer: per-step model cache and stopping-condition metrics.

Caches the detectors the app actually trained, plus their stability metrics,
per labelling step, so repeated queries (the progress button, the
auto-indicator) never recompute what is already known.

Only the shipped detector is ever measured (issue #3757)
-------------------------------------------------------
A step carries a model **only** when the app trained one for exactly that
label set - i.e. a learned sort ran and ``inject_live_model`` handed its head
and threshold over.  Every other step carries ``model=None`` and contributes
no point to the Smart curve and no entry to the Stable series.

That leaves gaps, and the gaps are the honest answer.  This module used to
fill them by training a *stand-in* of its own: a linear SVM over the
labelset's image-level vectors with an in-sample threshold.  On a patch
dataset that is a different detector from the shipped one, which trains each
Good vote on its boxed patch and floods every Bad vote's whole row stack, and
scores a media by the **max** over those rows.  Because a learned sort is
coalesced when votes outrun it (one job at a time, one pending slot - see
:class:`~vtscore.concurrency.async_jobs.JobManager`), roughly every other step
got a real model and the rest got a stand-in, so the plotted curve alternated
between two unrelated model families and read as a detector violently
changing its mind.  Worse, Smart and Stable gate Autopilot's phase machine, so
the stand-ins steered the run as well as the picture.

The same rule fixes the geometry, which is the other half of #3757: a model is
now scored the way it is *served*, over
:func:`~vtscore.detectors.training.scoring_rows_for_snap` rows with a
segmented max-pool, rather than over image-level vectors it was never fitted
on.  Both the Smart eval set and the Stable comparison pool go through that
one definition, so neither can drift from what the user's ranking shows.

Cache shape
-----------
All cache state lives in :class:`_ProgressCache` instances held in ``_caches``,
an LRU-bounded map keyed by ``(dataset_id, detector_id)``.  Every entry
point opens with ``cache = _active_cache()`` (or ``_ensure_cache``, which
returns one) under ``_progress_lock`` and works through that object.  Keying by
the pair is a correctness requirement, not a convenience: without it one
detector's history gets replayed onto another's accumulated label sets, and one
detector's models get served as another's Smart / Stable indicators (issue
#2914).

Unrelated to :mod:`vtscore.concurrency.progress`, which is the
infrastructure for tracking and cancelling long-running operations
(``ProgressTracker`` and the dataset/sort/eval/find singletons).  The
two modules used to share the ``progress.py`` name; this one was
renamed to make the distinction obvious.

Lock ordering (audit M1)
------------------------
``_progress_lock`` is acquired strictly *outside* ``vtscore.state.core._state_lock``.
Every callsite that needs to invalidate or clear the cache after a
state-lock'd mutation must release ``_state_lock`` before invoking a
function in this module.  Conversely, code inside ``_progress_lock`` must
not call into anything that acquires ``_state_lock`` - including helpers
on ``DatasetContext`` / ``DetectorContext`` that take the state lock,
and any of the resolve-context-then-mutate functions in
:mod:`vtscore.state.votes` / :mod:`vtscore.state.coverage`.  Holding
both locks in the opposite order would establish a cross-module cycle
and could deadlock.

That ordering is why the score rows this module measures over are built by the
*caller*, outside the lock, and passed in: :func:`_build_pool` and
:func:`_build_eval_rows` both reach the embedding-matrix layer, which takes
``_state_lock``.  :func:`_advance_cache` is the wrapper that does the dance.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from vtscore.concurrency.async_jobs import check_job_cancelled
from vtscore.embedding.media_vectors import media_embedding
from vtscore.training.thresholds import inclusion_cost_weights, weighted_error_cost

if TYPE_CHECKING:
    import torch.nn as nn

    from vtscore.detectors.training import ScoringRows

# ---------------------------------------------------------------------------
# Per-(dataset, detector) cache
# ---------------------------------------------------------------------------


@dataclass
class _ProgressCache:
    """Everything cached for one ``(dataset_id, detector_id)`` pair.

    Each entry in ``steps`` corresponds to one index in ``label_history`` and
    stores the model, threshold, label sets, and stability result for that step.
    ``good_ids`` / ``bad_ids`` track the running label sets so the next step
    only needs to apply a single delta.

    Every input a cache is built from is resolved *per request* from the
    ``X-Dataset-Id`` / ``X-Detector-Id`` headers (``label_history``,
    ``good_votes`` / ``bad_votes`` via the detector context; ``clips_dict`` and
    the coverage atlas via the dataset context).  Multiple detectors stay loaded
    at once and the UI switches between them freely - re-selecting an *already
    loaded* detector never goes through ``register_detector_context`` /
    ``unregister_detector_context``, so those clears do not cover the switch.
    Keying the cache by the pair is what stops one detector's history being
    replayed on top of another's accumulated label sets, or one detector's
    models being served as another's Smart / Stable indicators (issue #2914).
    """

    key: tuple[str, str]

    #: Inclusion value every cached step was trained under.  A different value
    #: rebuilds the cache in place (see :func:`_ensure_cache`).
    inclusion: Optional[int] = None

    steps: list[dict[str, Any]] = field(default_factory=list)
    good_ids: set[int] = field(default_factory=set)
    bad_ids: set[int] = field(default_factory=set)
    prev_predictions: Optional[dict[int, int]] = None
    coverage_atlas: Any = None  # CoverageAtlas | None

    #: Last fully-computed ``/api/labeling-status`` payload (minus the transient
    #: ``stale`` flag).  ``compute_labeling_status`` refreshes it on every full
    #: compute; the route returns it immediately to pollers while a background
    #: worker advances the per-step cache, so the 2 s poll never blocks on an
    #: MLP retrain (issue #2397).
    status_snapshot: Optional[dict[str, Any]] = None

    #: Live models injected by ``train_and_score`` during sorting.  Keyed by
    #: ``(frozenset(good_ids), frozenset(bad_ids))`` so that ``_ensure_cache``
    #: can look up the actual model that was used at each label step.  Per-pair
    #: because the lookup is by labelset alone, so a model must not outlive the
    #: detector it was trained for.
    #:
    #: This is the **only** source of models in this module: a step whose label
    #: set never had a sort run against it stays modelless rather than being
    #: filled with a locally-trained stand-in (see the module docstring).
    live_models: dict[tuple[frozenset[int], frozenset[int]], tuple[Any, float]] = field(default_factory=dict)

    #: Memoised Smart status, as ``(key, status)``.  ``_compute_smart_status``
    #: re-scores the whole recent-model window against the current labelset -
    #: forward passes over every labelled media's score rows - which is far too
    #: expensive to repeat on each 2 s ``/api/labeling-status`` poll when
    #: nothing has changed.  The key covers everything the status reads, so a
    #: hit is exactly the answer a recompute would produce.
    smart_memo: Optional[tuple[Any, dict[str, Any]]] = None

    def reset(self) -> None:
        """Drop everything derived from labels, keeping the pair identity.

        Used by the in-place rebuild (an inclusion change) that keeps the cache
        bound to the same pair.  Callers that want the cache gone entirely
        should use :func:`clear_progress_cache`.
        """
        self.steps.clear()
        self.good_ids.clear()
        self.bad_ids.clear()
        self.prev_predictions = None
        self.inclusion = None
        self.coverage_atlas = None
        self.live_models.clear()
        self.smart_memo = None
        # Drop the status snapshot too: it belonged to the just-cleared
        # labelset and would otherwise be served (stale) for the rebuild until
        # its first background refresh lands.
        self.status_snapshot = None


# Caches keyed by ``(dataset_id, detector_id)``, most-recently-used last.
# Bounded so that a session cycling through many detectors cannot grow without
# limit; the LRU victim is simply rebuilt on demand if it is selected again.
_caches: OrderedDict[tuple[str, str], _ProgressCache] = OrderedDict()

# How many ``(dataset, detector)`` pairs stay warm at once.  Small on purpose:
# each cache holds the app's trained head for every label-history step that had
# a sort, plus a ``prev_predictions`` map over the scored pool, so the point is
# to keep an A-to-B-and-back detector toggle from throwing away work, not to
# cache everything a long session ever touched.
_MAX_CACHED_PAIRS = 3

# Reentrant lock protecting ``_caches`` and every field of every cache in it.
# RLock is used because public functions call _ensure_cache which may call
# ``_ProgressCache.reset`` internally (inclusion change) while already holding
# the lock.
_progress_lock = threading.RLock()

# How many models back the Smart trend regresses.  Counted in *models*, not in
# label-history steps: most steps carry none (see the module docstring), so a
# step-counted window would silently shrink to a handful of points - or to
# fewer than the three the regression needs - exactly when sorts are slowest
# and the coalescing is heaviest.
_SMART_WINDOW_MODELS = 10

# How long :func:`cached_indicator_history` will wait for ``_progress_lock``
# before declaring the cache unreadable.  Long enough to ride out the brief
# holds taken by status reads, short enough that a click landing mid-build
# falls through to the async job instead of hanging on it.
_CACHE_READ_LOCK_TIMEOUT = 0.25


def _active_cache_key() -> tuple[str, str]:
    """Return ``(dataset_id, detector_id)`` for the current execution context.

    Read *without* acquiring ``_state_lock``: this runs under ``_progress_lock``
    (see the lock-ordering note at module top), which forbids taking
    ``_state_lock`` while held.  Neither ``get_active_context()`` nor
    ``get_active_detector_context()`` takes a lock - both resolve from the
    request-scoped resolver or a thread-local - so the read is safe.  In the
    ``/api/labeling-status`` background worker both contexts are bound
    thread-locally by ``JobManager``, so the worker resolves the same key the
    request thread did.

    Falls back to ``("", "")`` when resolution fails (e.g. a request naming an
    unloaded detector, whose resolver raises); such a caller cannot reach a
    usable ``label_history`` anyway.  That fallback is a key like any other, so
    two such callers share one (empty) cache rather than corrupting a real
    detector's.
    """
    try:
        from vtscore.state.core import get_active_context, get_active_detector_context  # noqa: PLC0415

        return (get_active_context().dataset_id, get_active_detector_context().detector_id)
    except Exception:
        return ("", "")


def _active_cache() -> _ProgressCache:
    """Return the cache for the active ``(dataset, detector)`` pair.

    Creates it on first use and marks it most-recently-used, evicting the LRU
    victim once more than :data:`_MAX_CACHED_PAIRS` pairs are warm.  Must be
    called with ``_progress_lock`` held.

    This is what makes the old identity-stamp invariant structural.  Cache
    state used to live in module globals guarded by a ``_bind_cache_identity()``
    call that every entry point had to remember to make first, or it would
    serve detector A's models as detector B's (issue #2914) - an invariant
    nothing enforced.  Now the only way to reach cache state at all is through
    the key, so a new entry point cannot forget.
    """
    key = _active_cache_key()
    cache = _caches.get(key)
    if cache is None:
        cache = _ProgressCache(key=key)
        _caches[key] = cache
    _caches.move_to_end(key)
    while len(_caches) > _MAX_CACHED_PAIRS:
        _caches.popitem(last=False)
    return cache


def clear_progress_cache() -> None:
    """Drop every cached pair's progress data.

    Must be called whenever votes are cleared, medias change, or inclusion
    is altered so that stale models are not reused.

    Deliberately global rather than scoped to the active pair: the callers
    (``clear_votes``, ``clear_medias``, ``set_inclusion``,
    ``register_detector_context`` / ``unregister_detector_context``) each
    invalidate *at least* the active pair, and some - a dataset's medias
    changing, the global inclusion knob moving - invalidate every pair over
    that dataset or every pair outright.  Clearing everything is the
    conservative reading and costs only a rebuild.  What the per-pair keying
    buys is the path that does *not* come through here: switching between two
    already-loaded detectors, which no longer throws either one's work away.
    """
    with _progress_lock:
        _caches.clear()


def invalidate_progress_cache_from(media_id: int) -> None:
    """Truncate the active pair's progress cache to just before *media_id* first appeared.

    Called when a vote switches polarity (good→bad or bad→good).  Steps
    before the media was first labeled are still valid - their models never
    included this media in training data.  Only steps from the first
    appearance onward are discarded so they can be retrained and their
    stability/evaluation metrics recomputed.

    Scoped to the active pair: a polarity flip on one detector says nothing
    about another's cache.
    """
    with _progress_lock:
        cache = _active_cache()

        # Find the first cached step that includes media_id in its training data.
        truncate_at = None
        for i, step in enumerate(cache.steps):
            if media_id in step["good_ids"] or media_id in step["bad_ids"]:
                truncate_at = i
                break

        if truncate_at is None:
            # Media never appeared in any cached step.  Still need to clear
            # live models - they may have been injected by learned-sort
            # without building the progress cache.
            cache.live_models.clear()
            return

        # Keep steps [0, truncate_at); discard the rest.
        del cache.steps[truncate_at:]

        # Restore the running ID sets to the surviving prefix's final state.
        cache.good_ids.clear()
        cache.bad_ids.clear()
        if cache.steps:
            last = cache.steps[-1]
            cache.good_ids.update(last["good_ids"])
            cache.bad_ids.update(last["bad_ids"])
        else:
            # truncate_at == 0: media was present from the very first step, so
            # the whole prefix is gone and no label survives.  No cached step
            # remains to source the Smart / Stable indicators from, so drop the
            # stale snapshot (parity with the old step-0 full-clear path).
            cache.status_snapshot = None

        # Reset the stability prediction chain - it will restart from the
        # truncation point when _ensure_cache replays the remaining history.
        cache.prev_predictions = None

        # Clear live models - some may have been trained with the old label.
        cache.live_models.clear()

        # The Smart memo was computed over the discarded suffix's models.
        cache.smart_memo = None

        # Rewind the coverage-atlas overlay and replay the surviving labels
        # rather than nulling the atlas (which would force a full hierarchical
        # k-means rebuild on the next /api/labeling-status poll, starving the
        # request pool at scale).  The structure is unchanged - only labels
        # moved - so the atlas object identity survives the invalidate.
        if cache.coverage_atlas is not None:
            cache.coverage_atlas.reset_labeled()
            for mid in cache.good_ids | cache.bad_ids:
                if mid in cache.coverage_atlas.vector_to_leaf:
                    cache.coverage_atlas.label(mid, good=mid in cache.good_ids)


def inject_live_model(
    good_votes: dict[int, None],
    bad_votes: dict[int, None],
    model: nn.Sequential,
    threshold: float,
) -> None:
    """Register a live model from ``train_and_score`` for progress-cache reuse.

    Called by the learned-sort route after each live training run.  The model
    is stored on the active pair's cache, keyed by its label set, so
    ``_ensure_cache`` can look it up instead of retraining from scratch.
    """
    key = (frozenset(good_votes), frozenset(bad_votes))
    with _progress_lock:
        _active_cache().live_models[key] = (model, threshold)


def _active_context_atlas() -> Any:
    """Return the active dataset context's coverage atlas, or ``None``.

    Read *without* acquiring ``_state_lock``: this runs under ``_progress_lock``
    (see the lock-ordering note at module top), which forbids taking
    ``_state_lock`` while held.  ``get_active_context()`` itself takes no lock,
    and the atlas it returns is only ever *read* here (its structure is
    immutable once built), so the lock-free read is safe - at worst a stale
    reference fails the id-set match below and we fall back to a fresh build.
    In the ``/api/labeling-status`` background worker the dataset context is
    bound thread-locally by ``JobManager``, so this resolves the right atlas.
    """
    try:
        from vtscore.state.core import get_active_context  # noqa: PLC0415

        return get_active_context().coverage_atlas
    except Exception:
        return None


def _build_coverage_atlas(clips_dict: dict[int, dict[str, Any]]) -> Any:
    """Build a CoverageAtlas from clip embeddings, or ``None`` if no embeddings.

    When the active dataset context already holds an atlas over *exactly* this
    id set, its hierarchical-k-means structure is identical to what a rebuild
    would produce, so we :meth:`~CoverageAtlas.structural_clone` it (sharing the
    node table by reference, fresh label overlay) instead of re-fitting under
    ``_progress_lock`` - which otherwise starves the request pool at N in the
    few-thousands on every polarity-flip invalidate.
    """
    vectors: dict[int, np.ndarray] = {
        cid: np.asarray(media_embedding(media), dtype=np.float32)
        for cid, media in clips_dict.items()
        if media_embedding(media) is not None
    }
    if not vectors:
        return None

    ctx_atlas = _active_context_atlas()
    if ctx_atlas is not None and ctx_atlas.vector_to_leaf.keys() == vectors.keys():
        return ctx_atlas.structural_clone()

    from vtscore.coverage.atlas import CoverageAtlas, auto_max_depth  # noqa: PLC0415

    # Cap the depth exactly as every other build site does
    # (``build_coverage_atlas`` / ``build_coverage_atlas_for_context``).
    # Omitting it left this fallback on ``COVERAGE_ATLAS_MAX_DEPTH``, so the
    # atlas built here was *deeper* - and cost many more k-means fits - than
    # the context atlas it stands in for.  That is the whole cost of a cold
    # progress-cache build on a dataset large enough to skip the load-time
    # atlas build, and it runs under ``_progress_lock``.
    return CoverageAtlas(vectors, k=3, max_depth=auto_max_depth(len(vectors), k=3))


def _apply_label_event(cache: _ProgressCache, media_id: int, label: str) -> bool:
    """Update *cache*'s running good/bad ID sets for one label event.

    Returns ``True`` if *media_id* was already labeled before this event.
    """
    was_labeled = media_id in cache.good_ids or media_id in cache.bad_ids
    if label == "unlabel":
        cache.good_ids.discard(media_id)
        cache.bad_ids.discard(media_id)
    elif label == "good":
        cache.bad_ids.discard(media_id)
        cache.good_ids.add(media_id)
    else:
        cache.good_ids.discard(media_id)
        cache.bad_ids.add(media_id)
    return was_labeled


def _sync_coverage_atlas(
    cache: _ProgressCache, media_id: int, label: str, was_labeled: bool
) -> Optional[dict[str, Any]]:
    """Mirror a label event onto the coverage atlas and return level info."""
    atlas = cache.coverage_atlas
    if atlas is None:
        return None
    if label == "unlabel":
        # Only unlabel on the atlas when the item is no longer labeled at all
        # (guards against good→bad re-labels going through "unlabel").
        if was_labeled and media_id not in cache.good_ids and media_id not in cache.bad_ids:
            if media_id in atlas.vector_to_leaf:
                atlas.unlabel(media_id)
    else:
        if media_id in atlas.vector_to_leaf:
            atlas.label(media_id, good=label == "good")
    return {
        "num_labels": len(cache.good_ids) + len(cache.bad_ids),
        "diversity_level": atlas.coverage_level(),
        "depth": atlas.total_nodes,
    }


@dataclass
class _ScoredPool:
    """The rows every stability pass scores, and the ids they belong to.

    ``rows`` is exactly what a learned sort scores the same snapshot over -
    :func:`~vtscore.detectors.training.scoring_rows_for_snap` output, image-level
    row plus every raw patch on a patch dataset, one row per media otherwise -
    so a step's predictions here are the predictions the user's ranking showed.
    ``id_set`` is carried alongside for O(labels) unlabeled counting.

    On the active dataset the underlying matrix is the one
    :func:`~vtscore.embedding.matrix.get_region_matrix_for_snap` already caches
    on the :class:`~vtscore.state.core.DatasetContext`, so holding this costs no
    memory beyond what the sort path holds anyway - which is also the bound on
    what a stability pass can cost: one scoring pass over rows the app scores on
    every sort, run once per step that has a model, i.e. once per sort.
    """

    rows: "ScoringRows"
    id_set: set[int]


def _detector_score_embedder(clips_dict: dict[int, dict[str, Any]]) -> Optional[str]:
    """The embedder the active detector trains and scores in, or ``None``.

    Delegates to :func:`~vtscore.detectors.training.detector_score_embedder`, the
    same resolver ``train_and_score`` uses, so the rows built here land in the
    space the cached heads were fitted in.  Falls back to ``None`` - "read each
    media's primary vector" - when no detector context resolves, which is the
    same fallback a detector with no chosen primary already gets.
    """
    from vtscore.detectors.training import detector_score_embedder  # noqa: PLC0415

    try:
        from vtscore.state.core import get_active_detector_context  # noqa: PLC0415

        det_ctx = get_active_detector_context()
    except Exception:
        det_ctx = None
    return detector_score_embedder(det_ctx, clips_dict)


def _build_pool(clips_dict: dict[int, dict[str, Any]]) -> _ScoredPool:
    """Build the stability pool for *clips_dict*.

    **Must be called without** ``_progress_lock`` **held**: it reaches
    :func:`~vtscore.embedding.matrix.get_region_matrix_for_snap`, which takes
    ``_state_lock``, and this module's lock ordering forbids that nesting (see
    the module docstring).  Every caller therefore builds the pool first and
    hands it to :func:`_ensure_cache`.
    """
    from vtscore.detectors.training import scoring_rows_for_snap  # noqa: PLC0415

    rows = scoring_rows_for_snap(clips_dict, _detector_score_embedder(clips_dict))
    return _ScoredPool(rows=rows, id_set=set(rows.ids))


def _build_eval_rows(
    clips_dict: dict[int, dict[str, Any]],
    current_good_votes: dict[int, None],
    current_bad_votes: dict[int, None],
) -> Optional[tuple["ScoringRows", list[float]]]:
    """Build the evaluation rows and labels from the current votes.

    Returns ``(rows, labels)`` aligned on ``rows.ids``, or ``None`` when there
    is nothing usable to evaluate against.  Built once and reused by every model
    in a window, so all points of the Smart indicator's slope regression share
    one eval set.

    The rows are the labelled media's **scoring** rows, not their image-level
    vectors: a production head is served by max-pooling over that stack, and a
    patch head is fitted on a Good vote's boxed patch and against every row of a
    Bad vote, so measuring it on image-level vectors alone scores it on a
    geometry it was never fitted for - half of issue #3757.  Media with no
    usable vector in the detector's space are dropped by
    :func:`~vtscore.detectors.training.scoring_rows_for_snap` rather than being
    fatal, and the labels follow the surviving ids.

    **Must be called without** ``_progress_lock`` **held** - see
    :func:`_build_pool`.
    """
    from vtscore.detectors.training import scoring_rows_for_snap  # noqa: PLC0415

    labels: dict[int, float] = {}
    for cid in current_good_votes:
        labels[cid] = 1.0
    for cid in current_bad_votes:
        labels[cid] = 0.0

    subset = {cid: clips_dict[cid] for cid in labels if cid in clips_dict}
    if not subset:
        return None

    rows = scoring_rows_for_snap(subset, _detector_score_embedder(clips_dict))
    if not rows.ids:
        return None
    return rows, [labels[cid] for cid in rows.ids]


def _compute_step_stability(
    cache: _ProgressCache,
    model: nn.Sequential,
    threshold: float,
    pool: Optional[_ScoredPool],
    t: int,
    num_labels: int,
) -> Optional[dict[str, Any]]:
    """Compute prediction stability by comparing to the previous step's predictions.

    Scores *pool* the way the detector is served - one forward pass over its
    score rows, max-pooled per media (:func:`score_rows_with_model`) - so a flip
    means an item the user would have seen move across the cut, not an item a
    differently-shaped scorer would have.

    The comparison is against the previous **model**, which is not necessarily
    the previous step: steps whose label set no sort ran against carry no
    detector, so nothing moved across them.  One entry therefore covers one
    retraining, however many votes the sort was coalesced over.

    *pool* is ``None`` only when the caller decided no step could carry a model
    and a sort injected one in the meantime.  There is then nothing to score, so
    the chain is dropped and restarted at the next step rather than comparing
    against a baseline this step cannot itself refresh.
    """
    from vtscore.detectors.training import score_rows_with_model  # noqa: PLC0415

    if pool is None:
        cache.prev_predictions = None
        return None

    labeled_ids = cache.good_ids | cache.bad_ids
    # Labels are few relative to the pool, so count the overlap from the
    # labelset rather than rescanning the pool.
    num_unlabeled = len(pool.rows.ids) - sum(1 for cid in labeled_ids if cid in pool.id_set)

    if num_unlabeled <= 0 or not pool.rows.ids:
        return {"time_index": t, "num_labels": num_labels, "num_flips": 0, "num_unlabeled": 0}

    # Score the whole pool in one pass and drop the currently-labeled ids
    # afterwards.  Scoring the handful of extra (labeled) media is far cheaper
    # than re-materialising a per-step row stack of the unlabeled subset.
    scores, _best_rows = score_rows_with_model(model, pool.rows)

    predictions: dict[int, int] = {
        cid: 1 if score >= threshold else 0
        for cid, score in zip(pool.rows.ids, scores, strict=True)
        if cid not in labeled_ids
    }

    stability: Optional[dict[str, Any]] = None
    if cache.prev_predictions is not None:
        prev = cache.prev_predictions
        num_flips = sum(1 for cid in predictions.keys() & prev.keys() if predictions[cid] != prev[cid])
        stability = {
            "time_index": t,
            "num_labels": num_labels,
            "num_flips": num_flips,
            "num_unlabeled": num_unlabeled,
        }
    # else: no prior predictions to compare - leave stability as None.

    cache.prev_predictions = predictions
    return stability


def _resolve_step_model(
    cache: _ProgressCache,
    pool: Optional[_ScoredPool],
    t: int,
    num_labels: int,
    good_ids: list[int],
    bad_ids: list[int],
    prev: Optional[dict[str, Any]],
) -> tuple[Optional[nn.Sequential], Optional[float], Optional[dict[str, Any]]]:
    """Resolve the model, threshold, and stability for one cache step.

    Reuses the previous step's model when the training data is unchanged;
    otherwise takes the model ``train_and_score`` injected for this exact label
    set, and **leaves the step modelless when there isn't one**.  Returns
    ``(model, threshold, stability)``.

    There is deliberately no fallback that trains something here.  A model this
    module fitted for itself is not the detector the user is building - on a
    patch dataset it is not even the same shape of detector - so plotting it
    beside the real ones produced the alternating Smart curve of issue #3757 and
    fed Autopilot's phase machine a mix of the two.  A gap is the honest answer,
    and the curve is plotted against label count, so a gap reads as a gap.
    """
    # Check whether the training data actually changed compared to the
    # previous step.  If the good/bad ID sets are identical, the model
    # would be the same - skip the stability recording so the line graph and
    # Stable indicator only reflect genuine model updates.
    training_data_changed = (
        prev is None or set(good_ids) != set(prev["good_ids"]) or set(bad_ids) != set(prev["bad_ids"])
    )

    if not training_data_changed:
        # Reuse previous model - no new stability entry.
        model = prev["model"] if prev else None
        threshold = prev["threshold"] if prev else None
        return model, threshold, None

    live = cache.live_models.get((frozenset(cache.good_ids), frozenset(cache.bad_ids)))
    if live is None:
        # No sort ran against this label set, so the app never had a detector
        # here.  The prediction baseline is deliberately *kept*: Stable measures
        # movement between successive detectors, and nothing moved across a gap
        # in which no detector existed.  Dropping it would mean that a session
        # whose sorts are coalesced onto every other vote - the exact shape of
        # issue #3757 - never has two adjacent model-bearing steps, produces no
        # stability entries at all, and leaves the indicator stuck on "not
        # enough history" forever, which also stops Autopilot ever finishing.
        return None, None, None

    model, threshold = live
    stability = _compute_step_stability(cache, model, threshold, pool, t, num_labels)
    return model, threshold, stability


def _ensure_cache(
    clips_dict: dict[int, dict[str, Any]],
    label_history: list[tuple[int, str, float]],
    inclusion_value: int,
    pool: Optional[_ScoredPool] = None,
) -> _ProgressCache:
    """Bring the active pair's cache up to date with *label_history*.

    Only computes steps that are not yet cached.  If *inclusion_value*
    differs from the value used for existing cache entries the entire cache
    is rebuilt.  Returns the cache, so callers never have to re-resolve it.

    *pool* is the stability pool, built by the caller **outside** this module's
    lock (see :func:`_build_pool`); ``None`` means the caller established that
    no step could carry a model, and any step that turns out to have one records
    no stability entry rather than reaching for ``_state_lock`` from here.

    Must be called with ``_progress_lock`` held.
    """
    cache = _active_cache()

    if cache.inclusion is not None and cache.inclusion != inclusion_value:
        # Same pair, different inclusion: rebuild in place.
        cache.reset()

    if cache.inclusion is None:
        cache.inclusion = inclusion_value

    start = len(cache.steps)
    if start >= len(label_history):
        return cache  # already up to date

    if cache.coverage_atlas is None:
        cache.coverage_atlas = _build_coverage_atlas(clips_dict)
        # A freshly built (or cloned) atlas starts with an empty label overlay.
        # Defensively seed it with any labels already accumulated in the
        # running ID sets so coverage_level() is correct before the history
        # replay below runs; normally these sets are empty at first build
        # (invalidate rewinds and replays its atlas in place rather than
        # nulling it, so this branch no longer runs mid-history).
        if cache.coverage_atlas is not None:
            for mid in cache.good_ids | cache.bad_ids:
                if mid in cache.coverage_atlas.vector_to_leaf:
                    cache.coverage_atlas.label(mid, good=mid in cache.good_ids)

    for t in range(start, len(label_history)):
        # Each step may run a scoring pass; honour a cancel of the owning eval
        # job here so a long history doesn't run to completion after cancel.
        # The partially-built cache is a valid prefix (steps 0..t-1), so the
        # next run resumes cleanly from ``len(cache.steps)``.  No-op outside a
        # job (see ``async_jobs.check_job_cancelled``).
        check_job_cancelled()
        media_id, label, _ = label_history[t]

        was_labeled = _apply_label_event(cache, media_id, label)
        diversity_info = _sync_coverage_atlas(cache, media_id, label, was_labeled)

        good_ids = list(cache.good_ids)
        bad_ids = list(cache.bad_ids)
        num_labels = len(good_ids) + len(bad_ids)

        prev = cache.steps[-1] if cache.steps else None
        model, threshold, stability = _resolve_step_model(cache, pool, t, num_labels, good_ids, bad_ids, prev)

        cache.steps.append(
            {
                "model": model,
                "threshold": threshold,
                "good_ids": good_ids,
                "bad_ids": bad_ids,
                "stability": stability,
                "diversity": diversity_info,
            }
        )

    # Any Smart status memoised before this advance was computed over a shorter
    # step list, so it no longer describes the cache.
    cache.smart_memo = None
    return cache


def _advance_cache(
    clips_dict: dict[int, dict[str, Any]],
    label_history: list[tuple[int, str, float]],
    inclusion_value: int,
) -> _ProgressCache:
    """Bring the active pair's cache up to date, building its pool off-lock.

    The lock dance is the point: :func:`_build_pool` must run **outside**
    ``_progress_lock`` (it takes ``_state_lock``, and the ordering is fixed the
    other way), while :func:`_ensure_cache` must run **inside** it.  So the
    state is sampled under the lock, the pool is built between the two holds,
    and ``_ensure_cache`` re-derives everything it needs from the live cache -
    a concurrent advance in the gap costs nothing but a redundant pool.

    The pool is skipped entirely when the cache holds no live models, because
    then no step can resolve one and nothing will be scored.  That is the
    common cold-start shape - a session that has loaded a dataset and voted but
    not yet sorted - and it keeps the first poll off the whole-dataset matrix
    build.
    """
    with _progress_lock:
        cache = _active_cache()
        rebuilding = cache.inclusion is not None and cache.inclusion != inclusion_value
        behind = rebuilding or len(cache.steps) < len(label_history)
        needs_pool = behind and bool(cache.live_models)

    pool = _build_pool(clips_dict) if needs_pool else None

    with _progress_lock:
        return _ensure_cache(clips_dict, label_history, inclusion_value, pool)


# ---------------------------------------------------------------------------
# Helper: evaluate cached models against a label set
# ---------------------------------------------------------------------------


def _score_step(
    step: dict[str, Any],
    eval_rows: "ScoringRows",
    eval_labels: list[float],
    fpr_weight: float,
    fnr_weight: float,
    t: int,
) -> dict[str, Any]:
    """Score one cached step against the evaluation set (forward pass only).

    Returns an error-cost dict for the step.  The caller guarantees
    ``step["model"]`` is not ``None``.

    Scoring goes through :func:`~vtscore.detectors.training.score_rows_with_model`,
    the same call the sort path makes, so a media's score here is the max over
    the rows the detector is actually served on.  The weighted FPR/FNR
    arithmetic is :func:`~vtscore.training.thresholds.weighted_error_cost`,
    shared with the eval harness
    (``vtscore.eval.step_trainers._labelset_error_costs``) so the Smart
    indicator a study measures is the one the app shows.
    """
    from vtscore.detectors.training import score_rows_with_model  # noqa: PLC0415

    scores, _best_rows = score_rows_with_model(step["model"], eval_rows)

    error_cost, fpr, fnr = weighted_error_cost(scores, eval_labels, step["threshold"], fpr_weight, fnr_weight)

    return {
        "time_index": t,
        "num_labels": len(step["good_ids"]) + len(step["bad_ids"]),
        "error_cost": round(error_cost, 4),
        "fpr": round(fpr, 4),
        "fnr": round(fnr, 4),
    }


def _model_step_indices(cache: _ProgressCache) -> list[int]:
    """Indices of the cached steps that carry a model the app actually trained."""
    return [t for t, step in enumerate(cache.steps) if step["model"] is not None]


def _eval_cached_models(
    cache: _ProgressCache,
    eval_set: Optional[tuple["ScoringRows", list[float]]],
    inclusion_value: int,
    indices: Optional[list[int]] = None,
) -> list[dict[str, Any]]:
    """Score *cache*'s models against the current labelset (forward passes only).

    Returns one error-cost dict per cached step in *indices* (every
    model-bearing step when ``None``).  Steps with no model contribute nothing:
    the app never had a detector at that label count, so there is no accuracy to
    report there - see the module docstring.

    The Inclusion weights come from the shipped
    :func:`~vtscore.training.thresholds.inclusion_cost_weights`, so the
    indicator prices a miss exactly as the threshold rule that produced the cut
    does.  *eval_set* comes from :func:`_build_eval_rows`, built by the caller
    outside ``_progress_lock``; ``None`` means there is nothing to measure
    against and the series is empty.
    """
    if eval_set is None:
        return []
    eval_rows, eval_labels = eval_set

    fpr_weight, fnr_weight = inclusion_cost_weights(inclusion_value)

    if indices is None:
        indices = _model_step_indices(cache)

    return [_score_step(cache.steps[t], eval_rows, eval_labels, fpr_weight, fnr_weight, t) for t in indices]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def recreate_model_at_time(
    clips_dict: dict[int, dict[str, Any]],
    label_history: list[tuple[int, str, float]],
    time_index: int,
    inclusion_value: int = 0,
) -> tuple[Optional[nn.Sequential], Optional[float], list[int], list[int]]:
    """Return the cached model for a given labelling step.

    The model is the one the app trained for that step's label set, or ``None``
    when no sort ever ran against it - this module trains nothing of its own
    (see the module docstring).

    Args:
        clips_dict: Mapping of media ID to media data dict with ``"embedding"``.
        label_history: Ordered labelling events.
        time_index: Index into *label_history*.
        inclusion_value: FPR/FNR trade-off in ``[-10, 10]``.

    Returns:
        ``(model, threshold, good_ids, bad_ids)`` - same contract as before.
    """
    if time_index < 0 or time_index >= len(label_history):
        return None, None, [], []

    cache = _advance_cache(clips_dict, label_history, inclusion_value)
    with _progress_lock:
        step = cache.steps[time_index]
        return step["model"], step["threshold"], step["good_ids"], step["bad_ids"]


def calculate_error_cost_over_time(
    clips_dict: dict[int, dict[str, Any]],
    label_history: list[tuple[int, str, float]],
    current_good_votes: dict[int, None],
    current_bad_votes: dict[int, None],
    inclusion_value: int = 0,
) -> list[dict[str, Any]]:
    """Calculate classification error cost at each labelling step that had a detector.

    Uses cached models - nothing is trained here.  Steps the app never trained a
    model for are absent from the series, so it is shorter than the label
    history and its ``num_labels`` values are not contiguous.
    """
    cache = _advance_cache(clips_dict, label_history, inclusion_value)
    eval_set = _build_eval_rows(clips_dict, current_good_votes, current_bad_votes)
    with _progress_lock:
        return _eval_cached_models(cache, eval_set, inclusion_value)


def calculate_prediction_stability_over_time(
    clips_dict: dict[int, dict[str, Any]],
    label_history: list[tuple[int, str, float]],
    inclusion_value: int = 0,
) -> list[dict[str, Any]]:
    """Return cached prediction-stability metrics for every step that had a detector."""
    cache = _advance_cache(clips_dict, label_history, inclusion_value)
    with _progress_lock:
        return [step["stability"] for step in cache.steps if step["stability"] is not None]


def _smart_memo_key(cache: _ProgressCache, model_steps: list[int], good: int, bad: int, inclusion_value: int) -> Any:
    """Everything :func:`_compute_smart_status` reads, as a comparable key.

    Steps are append-only between resets (a polarity flip truncates *and* drops
    the memo), so a matching step count and model count mean the same models;
    a matching vote count means the same eval set, because the only ways to
    change the labelset without changing its size - a polarity flip - reset the
    memo too.
    """
    return (len(cache.steps), tuple(model_steps[-_SMART_WINDOW_MODELS:]), good, bad, inclusion_value)


def _smart_status_memoized(
    cache: _ProgressCache,
    inclusion_value: int,
    good: int,
    bad: int,
) -> Optional[dict[str, Any]]:
    """The memoised Smart status when it still describes *cache*, else ``None``.

    Split out from :func:`_compute_smart_status` so a caller can ask "will this
    need the eval rows?" *before* paying to build them - the rows are a
    whole-labelset row stack, and the poll that asks for them arrives every 2 s
    whether or not anything moved.

    Must be called with ``_progress_lock`` held.
    """
    if cache.smart_memo is None:
        return None
    key = _smart_memo_key(cache, _model_step_indices(cache), good, bad, inclusion_value)
    if cache.smart_memo[0] != key:
        return None
    return dict(cache.smart_memo[1])


def _compute_smart_status(
    cache: _ProgressCache,
    eval_set: Optional[tuple["ScoringRows", list[float]]],
    inclusion_value: int,
    good: int,
    bad: int,
) -> dict[str, Any]:
    """Compute Smart (error-cost flatness) red/yellow/green status.

    Regresses over the last :data:`_SMART_WINDOW_MODELS` **models**, not the
    last N label-history steps: most steps carry no model, so a step-counted
    window shrinks with the sort backlog and would report "not enough history"
    exactly when the run is busiest.

    Memoised on *cache* because every branch is re-read by each 2 s poll while
    the answer cannot have changed.
    """
    model_steps = _model_step_indices(cache)
    memo_key = _smart_memo_key(cache, model_steps, good, bad, inclusion_value)
    if cache.smart_memo is not None and cache.smart_memo[0] == memo_key:
        return dict(cache.smart_memo[1])

    status = _smart_status_uncached(cache, model_steps, eval_set, inclusion_value, good, bad)
    cache.smart_memo = (memo_key, dict(status))
    return status


def _smart_status_uncached(
    cache: _ProgressCache,
    model_steps: list[int],
    eval_set: Optional[tuple["ScoringRows", list[float]]],
    inclusion_value: int,
    good: int,
    bad: int,
) -> dict[str, Any]:
    """The body of :func:`_compute_smart_status`, before memoisation."""
    if good < 5 or bad < 5:
        return {
            "status": "red",
            "reason": f"Need at least 5 good and 5 bad. Currently {good}g, {bad}b.",
        }

    if len(model_steps) < 3:
        return {
            "status": "yellow",
            "reason": "Not enough trained detectors yet to assess trend. Sort to train one.",
        }

    recent_entries = _eval_cached_models(cache, eval_set, inclusion_value, model_steps[-_SMART_WINDOW_MODELS:])
    recent_error_costs = [e["error_cost"] for e in recent_entries]

    if len(recent_error_costs) < 3:
        return {"status": "yellow", "reason": "Not enough valid model steps in recent history to assess trend."}

    # Linear regression slope over the recent error-cost values
    n_pts = len(recent_error_costs)
    x_vals = list(range(n_pts))
    x_mean = sum(x_vals) / n_pts
    y_mean = sum(recent_error_costs) / n_pts

    numer = sum((x_vals[i] - x_mean) * (recent_error_costs[i] - y_mean) for i in range(n_pts))
    denom = sum((x_vals[i] - x_mean) ** 2 for i in range(n_pts))
    slope = numer / denom if denom != 0 else 0.0
    relative_slope = slope / y_mean if y_mean > 0 else slope

    FLAT_THRESHOLD = -0.015
    if relative_slope < FLAT_THRESHOLD:
        return {
            "status": "yellow",
            "reason": "Error cost is still declining. Keep labeling.",
            "slope": round(relative_slope, 4),
        }
    return {
        "status": "green",
        "reason": "Error cost has leveled off. You can likely stop labeling.",
        "slope": round(relative_slope, 4),
    }


def _compute_stable_status(
    cache: _ProgressCache,
    good: int,
    bad: int,
    total: int,
) -> dict[str, Any]:
    """Compute Stable (prediction-flip) red/yellow/green status."""
    if good < 5 or bad < 5:
        return {
            "status": "red",
            "reason": f"Need at least 5 good and 5 bad. Currently {good}g, {bad}b.",
        }

    stability = [step["stability"] for step in cache.steps if step["stability"] is not None]

    MIN_STABLE_ENTRIES = 5
    if len(stability) < MIN_STABLE_ENTRIES:
        return {"status": "yellow", "reason": "Not enough history to assess prediction stability."}

    recent = stability[-10:]

    # Use flip *rate* (fraction of unlabeled predictions that changed) so the
    # threshold scales with dataset size instead of using a fixed absolute count.
    flip_rates: list[float] = []
    for s in recent:
        n_unlabeled = s.get("num_unlabeled", 0)
        if n_unlabeled > 0:
            flip_rates.append(s["num_flips"] / n_unlabeled)
        else:
            flip_rates.append(0.0)

    avg_flip_rate = sum(flip_rates) / len(flip_rates)
    max_flip_rate = max(flip_rates)

    STABLE_RATE_THRESHOLD = 0.005  # average less than 0.5% of predictions flipping
    STABLE_MAX_THRESHOLD = 0.01  # no single recent step above 1%

    if avg_flip_rate < STABLE_RATE_THRESHOLD and max_flip_rate < STABLE_MAX_THRESHOLD:
        return {"status": "green", "reason": "Predictions have stabilized.", "avg_flip_rate": round(avg_flip_rate, 4)}
    return {
        "status": "yellow",
        "reason": f"Average {avg_flip_rate:.1%} of predictions flipping in recent steps.",
        "avg_flip_rate": round(avg_flip_rate, 4),
    }


def _compute_span_status(span_info: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Compute the Span (diversity-coverage) red/yellow/green status.

    Depends only on the coverage-atlas ``span_info`` passed in from the route
    (``level`` = consecutive BFS-order seen nodes, ``depth`` = total nodes),
    not on the per-step MLP cache, so it stays cheap and is reused verbatim by
    the pending-status placeholder.
    """
    # The old metric required 4 full tree levels for green, which in a k=3
    # tree is 1+3+9+27 = 40 nodes.  We preserve that scale: green at 40
    # nodes (capped at total), yellow at 10, red below 10.
    from vtscore.config import CoreConfig  # noqa: PLC0415

    SPAN_GREEN = CoreConfig.from_settings().autopilot_goal_diversity
    SPAN_YELLOW = 10
    if span_info is None:
        return {
            "status": "red",
            "reason": "Diversity tree not available.",
            "level": 0,
            "depth": 0,
        }

    level = span_info["level"]
    tree_total = span_info["depth"]  # total nodes
    green_at = min(SPAN_GREEN, tree_total)
    yellow_at = min(SPAN_YELLOW, green_at)
    if tree_total <= 0:
        return {"status": "green", "reason": "Degenerate tree.", **span_info}
    if level >= green_at:
        return {
            "status": "green",
            "reason": "All tree nodes covered." if level >= tree_total else f"{level}/{tree_total} nodes covered.",
            **span_info,
        }
    if level >= yellow_at:
        return {
            "status": "yellow",
            "reason": f"{level}/{tree_total} nodes covered.",
            **span_info,
        }
    return {
        "status": "red",
        "reason": "No tree coverage yet." if level == 0 else f"{level}/{tree_total} nodes covered.",
        **span_info,
    }


def compute_labeling_status(
    clips_dict: dict[int, dict[str, Any]],
    label_history: list[tuple[int, str, float]],
    current_good_votes: dict[int, None],
    current_bad_votes: dict[int, None],
    inclusion_value: int = 0,
    span_info: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Compute per-metric red/yellow/green labeling statuses.

    Returns a dict with ``good_count``, ``bad_count``, ``total_count``, and
    three sub-dicts: ``smart``, ``stable``, and ``span``, each with a
    ``status`` field of ``"red"``, ``"yellow"``, or ``"green"``.

    Advancing the per-step cache scores the whole pool once per step that has a
    detector, and Smart re-scores its recent-model window against the current
    labelset, so this is the *heavy* path.  The result is stashed in the pair's
    ``status_snapshot`` so the ``/api/labeling-status`` route can serve it
    immediately (marked ``stale``) on subsequent polls while a background worker
    calls this to advance the cache off the request thread (issue #2397).
    """
    good = len(current_good_votes)
    bad = len(current_bad_votes)
    total = good + bad

    cache = _advance_cache(clips_dict, label_history, inclusion_value)

    # The eval rows are built off-lock (they reach the embedding-matrix layer,
    # which takes ``_state_lock``), which means deciding *before* the build
    # whether Smart will read them at all.  It will not when the memo still
    # holds - the common case, since the 2 s poll outruns the vote rate by an
    # order of magnitude - nor below the 5g/5b quorum, where the status is red
    # without looking at a model.
    with _progress_lock:
        memo = _smart_status_memoized(cache, inclusion_value, good, bad)
    needs_eval = memo is None and good >= 5 and bad >= 5
    eval_set = _build_eval_rows(clips_dict, current_good_votes, current_bad_votes) if needs_eval else None

    with _progress_lock:
        smart = memo if memo is not None else _compute_smart_status(cache, eval_set, inclusion_value, good, bad)
        stable = _compute_stable_status(cache, good, bad, total)

    # Span status from coverage atlas info (passed in from the route).
    span = _compute_span_status(span_info)

    result = {
        "good_count": good,
        "bad_count": bad,
        "total_count": total,
        "smart": smart,
        "stable": stable,
        "span": span,
    }
    # Refresh the snapshot the poll route hands back while the cache is behind.
    # Store a copy so a caller that mutates the returned dict (e.g. adding the
    # ``stale`` flag) doesn't retroactively corrupt the snapshot.  The lock was
    # released for the (settings-reading) Span computation above, so another
    # thread may have dropped this pair's cache meanwhile (a vote clear, a
    # detector unload, an LRU eviction); the identity check republishes only
    # onto the very object these indicators were computed from, never onto a
    # successor that has been rebuilt or belongs to someone else.
    with _progress_lock:
        if _caches.get(cache.key) is cache:
            cache.status_snapshot = dict(result)
    return result


def cached_indicator_history(
    metric: str,
    clips_dict: dict[int, dict[str, Any]],
    label_history: list[tuple[int, str, float]],
    current_good_votes: dict[int, None],
    current_bad_votes: dict[int, None],
    inclusion_value: int = 0,
) -> tuple[list[dict[str, Any]], bool]:
    """Read *metric*'s per-step history **without advancing the cache**.

    Returns ``(history, complete)``.  ``complete`` is ``False`` - with an empty
    history - whenever the per-step cache does not already cover the whole of
    *label_history*; the caller is expected to fall back to the async
    ``/api/eval/train-and-score`` job, which does the same work on a background
    thread with progress and cancellation.

    ``complete`` says the *cache* is caught up, not that every label step has a
    point: a step the app never trained a detector for has no accuracy to
    report and is simply absent (see the module docstring).  A complete-but-
    empty ``smart`` series is the honest answer for a session that has voted
    but never sorted, and the modal renders it as "no history yet" rather than
    inventing one.

    This is the counterpart to the ``calculate_*_over_time`` functions, which
    call :func:`_advance_cache` and therefore score the pool once per uncached
    step on the calling thread.  Doing that inline is exactly what
    ``/api/labeling-status`` refuses to do (issue #2397), so the read path that
    backs the progress-plot modal must not do it either.

    When the cache *is* complete every branch is cheap: ``smart`` runs one
    forward pass per cached model over the (small) labelled set, and ``stable``
    / ``diverse`` are plain reads of values recorded during the cache build.
    """
    # Coverage first, and on its own: while the user is labelling the answer is
    # usually "behind", and the Smart branch below would otherwise have built a
    # row stack over every labelled media only to throw it away.
    #
    # Reading the cache needs ``_progress_lock``, but a background worker holds
    # that lock for the *entire* duration of a cache build - which is exactly
    # the multi-second work this function exists to avoid waiting on.  Blocking
    # here would reintroduce the hang whenever the click lands mid-refresh, so
    # give up quickly and report the cache as unavailable: the caller falls back
    # to the async job, which is the right answer in that state anyway.
    if not _progress_lock.acquire(timeout=_CACHE_READ_LOCK_TIMEOUT):
        return [], False
    try:
        if not _cache_covers_history(_active_cache(), label_history, inclusion_value):
            return [], False
    finally:
        _progress_lock.release()

    # The Smart series measures the cached models against the *current* votes,
    # and building those rows reaches the embedding-matrix layer, which takes
    # ``_state_lock`` - so it has to happen with ``_progress_lock`` released,
    # never inside it (see the module docstring's lock-ordering note).
    eval_set = _build_eval_rows(clips_dict, current_good_votes, current_bad_votes) if metric == "smart" else None

    if not _progress_lock.acquire(timeout=_CACHE_READ_LOCK_TIMEOUT):
        return [], False
    try:
        cache = _active_cache()
        # Re-checked: the gap above is exactly when a background worker can
        # advance or reset the cache, and a series read off a cache that no
        # longer covers this history would be a truncated plot.
        if not _cache_covers_history(cache, label_history, inclusion_value):
            return [], False

        if metric == "smart":
            data = _eval_cached_models(cache, eval_set, inclusion_value)
        elif metric == "stable":
            data = [step["stability"] for step in cache.steps if step["stability"] is not None]
        else:
            data = [step["diversity"] for step in cache.steps if step.get("diversity") is not None]
        return data, True
    finally:
        _progress_lock.release()


def _cache_covers_history(
    cache: _ProgressCache,
    label_history: list[tuple[int, str, float]],
    inclusion_value: int,
) -> bool:
    """Whether *cache* already holds a step for every event in *label_history*.

    A mismatched ``inclusion_value`` counts as not-covered because
    :func:`_ensure_cache` would rebuild the cache from scratch.  The length
    comparison is against the pair's own cache, so another detector's longer
    history can never be read as covering this one's.

    Must be called with ``_progress_lock`` held.
    """
    if cache.inclusion is not None and cache.inclusion != inclusion_value:
        return False
    return len(cache.steps) >= len(label_history)


def is_status_cache_fresh(label_history: list[tuple[int, str, float]], inclusion_value: int) -> bool:
    """Return ``True`` when the per-step cache already covers *label_history*.

    A fresh cache means ``compute_labeling_status`` will not advance a step, and
    therefore will not score the pool, so the route can compute the status
    inline instead of deferring to a background worker.

    It does **not** promise the Smart status is memoised, and deliberately so.
    The worker that advances the cache computes the status in the same call, so
    after every vote the memo is warm before the next poll arrives; requiring it
    here would only mean a brand-new detector - no votes, no steps, nothing to
    compute - reported its indicators as "computing" for one extra poll.
    """
    with _progress_lock:
        return _cache_covers_history(_active_cache(), label_history, inclusion_value)


def _pending_labeling_status(
    current_good_votes: dict[int, None],
    current_bad_votes: dict[int, None],
    span_info: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build a status from only the cheap fields (counts + Span).

    The MLP-derived Smart / Stable indicators show a transient "computing"
    state; :func:`stale_labeling_status` overlays the real ones from the last
    snapshot when one exists.
    """
    good = len(current_good_votes)
    bad = len(current_bad_votes)
    computing = {"status": "yellow", "reason": "Computing indicators..."}
    return {
        "good_count": good,
        "bad_count": bad,
        "total_count": good + bad,
        "smart": dict(computing),
        "stable": dict(computing),
        "span": _compute_span_status(span_info),
    }


def stale_labeling_status(
    current_good_votes: dict[int, None],
    current_bad_votes: dict[int, None],
    span_info: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build the poll response served while a background cache refresh is pending.

    Counts and the coverage-atlas Span status are recomputed live (both cheap),
    so the panel's counters and diversity chip stay accurate the instant a vote
    lands.  Only the expensive Smart / Stable indicators lag: they come from the
    last ``compute_labeling_status`` snapshot, or - when none exists yet (first
    poll after a detector switch / session start) - a transient "computing"
    placeholder.  The caller stamps ``stale = True`` on the result.
    """
    status = _pending_labeling_status(current_good_votes, current_bad_votes, span_info)
    with _progress_lock:
        # Reading the snapshot off the *active pair's* cache is what stops one
        # detector's indicators being handed to another; a detector with no
        # snapshot of its own shows the "computing" placeholder instead.
        snapshot = _active_cache().status_snapshot
        if snapshot is not None:
            status["smart"] = dict(snapshot["smart"])
            status["stable"] = dict(snapshot["stable"])
    return status


def calculate_diversity_level_over_time(
    clips_dict: dict[int, dict[str, Any]],
    label_history: list[tuple[int, str, float]],
    inclusion_value: int = 0,
) -> list[dict[str, Any]]:
    """Return cached per-step diversity levels.

    Diversity levels are computed and stored by :func:`_ensure_cache` as it
    processes each label-history step, so this function ensures the cache is
    current before reading it.  Unlike Smart and Stable this series has a point
    for every step: coverage is a property of the votes, not of a detector, so
    it does not depend on whether a sort ever ran.
    """
    cache = _advance_cache(clips_dict, label_history, inclusion_value)
    with _progress_lock:
        return [step["diversity"] for step in cache.steps if step.get("diversity") is not None]


def analyze_labeling_progress(
    clips_dict: dict[int, dict[str, Any]],
    label_history: list[tuple[int, str, float]],
    current_good_votes: dict[int, None],
    current_bad_votes: dict[int, None],
    inclusion_value: int = 0,
) -> dict[str, Any]:
    """Run a comprehensive analysis of labelling progress.

    Models and stability metrics are read from the per-step cache.  Error
    cost is recomputed cheaply using cached models (forward passes only).
    The error-cost and stability series cover only the steps the app trained a
    detector for, so they are shorter than ``total_labels``; diversity covers
    every step.
    """
    cache = _advance_cache(clips_dict, label_history, inclusion_value)
    eval_set = _build_eval_rows(clips_dict, current_good_votes, current_bad_votes)

    with _progress_lock:
        error_cost = _eval_cached_models(cache, eval_set, inclusion_value)

        stability = [step["stability"] for step in cache.steps if step["stability"] is not None]

        diversity = [step["diversity"] for step in cache.steps if step.get("diversity") is not None]

    return {
        "error_cost_over_time": error_cost,
        "stability_over_time": stability,
        "diversity_level_over_time": diversity,
        "total_labels": len(current_good_votes) + len(current_bad_votes),
        "total_medias": len(clips_dict),
    }
