"""Blueprint for evaluation and labeling progress routes.

Migrated to ``flask_smorest`` so the routes are described in
``/api/openapi.json``. Schema-level
failures (missing ``metric`` / ``job_id``, invalid metric value) surface
as 422 with the standard ``errors`` envelope; handler-level rejects
(no votes / no label history, missing job) keep their HTTP codes
(400 / 404 / 500) with the standard ``message`` envelope.
"""

from flask_smorest import Blueprint, abort

from vtscore.detectors.labeling_progress import (
    analyze_labeling_progress,
    cached_indicator_history,
    calculate_diversity_level_over_time,
    calculate_error_cost_over_time,
    calculate_prediction_stability_over_time,
    compute_labeling_status,
    is_status_cache_fresh,
    stale_labeling_status,
)
from vtsearch.schemas.eval import (
    EvalTrainAndScoreCancelResponseSchema,
    EvalTrainAndScoreRequestSchema,
    EvalTrainAndScoreResponseSchema,
    EvalTrainAndScoreResultQuerySchema,
    IndicatorScoreHistoryQuerySchema,
    IndicatorScoreHistoryResponseSchema,
    LabelingProgressResponseSchema,
    LabelingStatusResponseSchema,
)
from vtsearch.state import (
    bad_votes,
    get_coverage_atlas,
    get_inclusion,
    good_votes,
    label_history,
    snapshot_medias,
)
from vtscore.concurrency.progress import (
    CancelledError,
    update_eval_progress,
)
from vtscore.state.core import (
    DatasetNotLoadedError,
    DetectorNotLoadedError,
    RequestMissingContextError,
)

eval_bp = Blueprint(
    "eval",
    __name__,
    description="Labeling-progress analysis and learned-sort eval indicators.",
)


# Failures of *context resolution* are not this blueprint's 500s.  Every route
# below reads the request-scoped proxies (``label_history``, ``good_votes``,
# ``snapshot_medias()``, ``get_inclusion()``), and each of those raises when
# the client named a dataset / detector the backend has not finished loading -
# which ``vtsearch/errors.py`` maps to the app-wide 409 ``dataset_not_loaded``
# / ``detector_not_loaded`` (or a 400 for the request-missing sentinel).  A
# blanket ``except Exception`` swallows that contract and reports a load-window
# poll as an opaque "computation failed" 500, which is what made issue #3644
# read as a bug in the empty-labelset branch rather than a detector that was
# still loading.  Let these through; only genuine computation faults are 500s.
_CONTEXT_ERRORS = (
    DatasetNotLoadedError,
    DetectorNotLoadedError,
    RequestMissingContextError,
)


@eval_bp.route("/api/labeling-progress", methods=["POST"])
@eval_bp.response(200, LabelingProgressResponseSchema)
@eval_bp.alt_response(400, description="No good/bad votes, or no label history.")
@eval_bp.alt_response(500, description="Labeling-progress computation failed.")
def labeling_progress():
    """Analyze labeling progress and calculate stopping condition metrics."""
    if not good_votes or not bad_votes:
        abort(400, message="need at least one good and one bad vote")

    if not label_history:
        abort(400, message="no label history available")

    try:
        return analyze_labeling_progress(snapshot_medias(), label_history, good_votes, bad_votes, get_inclusion())
    except _CONTEXT_ERRORS:
        raise
    except Exception:
        import logging

        logging.getLogger(__name__).exception("labeling-progress failed")
        abort(500, message="Labeling progress computation failed")


def _schedule_status_refresh(span_info) -> None:
    """Kick (or coalesce into) a single background labeling-status cache build.

    Snapshots the inputs on the request thread - like ``eval_train_and_score``
    - so the worker advances the per-step cache against a consistent labelset
    even if more votes arrive mid-refresh; the next stale poll simply schedules
    another pass.  ``JobManager`` coalesces the rapid poll burst into one
    in-flight refresh per (dataset, detector) so we never fan out parallel
    retrains.
    """
    from vtscore.concurrency.async_jobs import labeling_status_jobs
    from vtscore.state.core import get_active_context, get_active_detector_context

    clips = snapshot_medias()
    inclusion = get_inclusion()
    history = list(label_history)
    good_snap = dict(good_votes)
    bad_snap = dict(bad_votes)

    ds_ctx = get_active_context()
    det_ctx = get_active_detector_context()

    signature = (ds_ctx.dataset_id, det_ctx.detector_id, len(history), inclusion)

    def _run(job):
        # ``compute_labeling_status`` advances the per-step cache under
        # ``_progress_lock`` and refreshes ``_status_snapshot``; the return
        # value is unused - the next poll reads the snapshot.  The lock is
        # taken only inside the worker, never across the HTTP response.
        compute_labeling_status(clips, history, good_snap, bad_snap, inclusion, span_info=span_info)

    labeling_status_jobs.start(
        signature,
        _run,
        dataset_id=ds_ctx.dataset_id,
        detector_id=det_ctx.detector_id,
    )


@eval_bp.route("/api/labeling-status", methods=["GET"])
@eval_bp.response(200, LabelingStatusResponseSchema)
@eval_bp.alt_response(500, description="Labeling-status computation failed.")
def labeling_status_indicator():
    """Return per-metric red/yellow/green labeling statuses.

    Returns ``smart``, ``stable``, and ``span`` sub-objects, each with a
    ``status`` field of ``"red"``, ``"yellow"``, or ``"green"``, plus a
    ``stale`` flag.

    The frontend polls this every 2 s during labeling.  When the per-step
    cache already covers the full ``label_history`` the status is computed
    inline (cheap: cached-model forward passes over the small labeled set).
    When the cache is *behind* - the common case on the first poll after a new
    vote, or after a polarity flip truncated the cache - advancing it would
    retrain one-or-more MLPs and score every unlabeled media, so we return the
    last-computed snapshot immediately with ``stale = true`` and defer the
    advancement to a background worker (issue #2397).
    """
    try:
        tree = get_coverage_atlas()
        span = tree.span_info() if tree is not None else None
        inclusion = get_inclusion()

        if is_status_cache_fresh(label_history, inclusion):
            status = compute_labeling_status(
                snapshot_medias(), label_history, good_votes, bad_votes, inclusion, span_info=span
            )
            status["stale"] = False
            return status

        # Cache is behind: serve the last snapshot now, advance the cache in
        # the background.  ``stale_labeling_status`` keeps the cheap fields
        # (counts + Span) live and lags only the MLP-derived Smart / Stable
        # indicators, falling back to a "computing" placeholder when no
        # snapshot exists yet (rapid votes at session start, or the first poll
        # after a detector switch cleared the cache).
        # Read the snapshot *before* kicking the refresh: the worker rewrites
        # ``_status_snapshot`` as soon as it finishes, so scheduling first
        # races it against this request's read and the response could reflect
        # the in-flight recompute instead of the previous snapshot.
        status = stale_labeling_status(good_votes, bad_votes, span)
        status["stale"] = True
        _schedule_status_refresh(span)
        return status
    except _CONTEXT_ERRORS:
        raise
    except Exception:
        import logging

        logging.getLogger(__name__).exception("labeling-status failed")
        abort(500, message="Labeling status computation failed")


@eval_bp.route("/api/indicator-score-history", methods=["GET"])
@eval_bp.arguments(IndicatorScoreHistoryQuerySchema, location="query")
@eval_bp.response(200, IndicatorScoreHistoryResponseSchema)
@eval_bp.alt_response(500, description="Score-history computation failed.")
def indicator_score_history(query: dict):
    """Return the cached indicator score history for a given metric.

    Reads only the per-step cache advanced by the ``/api/labeling-status``
    background worker; **no models are retrained and the cache is never
    advanced here**, so the response is always fast regardless of dataset size
    or label-history length.

    When the cache does not yet cover the whole ``label_history`` the response
    carries ``complete: false`` and an empty ``history``.  Clients should then
    fall back to ``POST /api/eval/train-and-score``, which computes the same
    series on a background thread with live progress and cancellation.

    This route used to call the ``calculate_*_over_time`` helpers directly.
    Those advance the cache via ``_ensure_cache`` - retraining one MLP per
    uncached label step plus a forward pass over the unlabeled pool, and on a
    cold cache a full hierarchical-k-means coverage-atlas build - all inline on
    the request thread under ``_progress_lock``.  Since ``/api/labeling-status``
    deliberately defers exactly that work to a background worker (issue #2397),
    the cache is behind for most of a labeling session, and this endpoint
    absorbed the entire deferred build: tens of seconds on a mid-size dataset,
    growing with both media count and vote count.
    """
    metric = query["metric"]

    clips = snapshot_medias()
    inclusion = get_inclusion()

    try:
        data, complete = cached_indicator_history(metric, clips, label_history, good_votes, bad_votes, inclusion)
        return {"metric": metric, "history": data, "complete": complete}
    except _CONTEXT_ERRORS:
        raise
    except Exception:
        import logging

        logging.getLogger(__name__).exception("indicator-score-history failed")
        abort(500, message="Score history computation failed")


_METRIC_KEY = {"smart": "error_cost", "stable": "stability", "diverse": "diversity"}


def _eval_done_payload(job) -> dict:
    """Build the JSON body for a finished eval job, including metric data."""
    result = job.result or {}
    metric = result.get("metric") or ""
    data_key = _METRIC_KEY.get(metric, "data")
    return {
        "job_id": job.job_id,
        "status": "done",
        "metric": metric,
        data_key: result.get("data", []),
    }


@eval_bp.route("/api/eval/train-and-score", methods=["POST"])
@eval_bp.arguments(EvalTrainAndScoreRequestSchema)
@eval_bp.response(200, EvalTrainAndScoreResponseSchema)
@eval_bp.alt_response(500, description="Evaluation computation failed (only when ``wait=true``).")
def eval_train_and_score(body: dict):
    """Start (or short-circuit) an eval train-and-score computation.

    The work walks the full ``label_history`` retraining a small MLP at
    every step, which used to block every other request on the gthread
    pool.  We now run it on a background daemon thread and return a
    ``job_id``; clients poll ``/api/eval/train-and-score/result`` for the
    metric data; the ``eval`` SSE channel on ``/api/events`` carries
    live progress.

    A signature cache keyed by ``(metric, history, votes, inclusion,
    dataset, detector)`` short-circuits identical re-runs.

    Tests can pass ``{"wait": true}`` to block until the job completes.
    """
    from vtscore.concurrency.async_jobs import eval_jobs
    from vtscore.state.core import (
        get_active_context,
        get_active_detector_context,
        thread_dataset_context,
        thread_detector_context,
    )

    metric = body["metric"]
    wait = body["wait"]

    clips = snapshot_medias()
    inclusion = get_inclusion()
    history = list(label_history)
    good_snap = dict(good_votes)
    bad_snap = dict(bad_votes)

    ds_ctx = get_active_context()
    det_ctx = get_active_detector_context()

    signature = (
        metric,
        ds_ctx.dataset_id,
        det_ctx.detector_id,
        tuple(sorted(clips.keys())),
        tuple(history),
        tuple(sorted(good_snap)),
        tuple(sorted(bad_snap)),
        inclusion,
    )

    cached = eval_jobs.cached_for(signature)
    if cached is not None:
        return _eval_done_payload(cached)

    n_total = max(len(history) - 1, 0)

    def _run(job):
        with thread_dataset_context(ds_ctx), thread_detector_context(det_ctx):
            # Progress lives on the job, not on the global ``eval_progress``
            # singleton, so overlapping evals don't decorrelate the poll from
            # job identity.  The singleton is written only from the actually-
            # running job (here, inside ``_run``) so the live SSE ``eval`` bar
            # reflects the running job rather than a job still parked pending.
            job.update_progress(0, n_total, f"Computing {metric}...")
            update_eval_progress("running", f"Computing {metric}...", 0, n_total)
            try:
                if metric == "smart":
                    data = calculate_error_cost_over_time(clips, history, good_snap, bad_snap, inclusion)
                elif metric == "stable":
                    data = calculate_prediction_stability_over_time(clips, history, inclusion)
                else:
                    data = calculate_diversity_level_over_time(clips, history, inclusion)
                job.result = {"metric": metric, "data": data}
                job.update_progress(n_total, n_total, "Done")
                update_eval_progress("idle", "Done", n_total, n_total)
            except CancelledError:
                # User cancelled a running job: not a failure.  Clear the
                # progress bar and re-raise so the JobManager marks the job
                # ``cancelled`` rather than ``error``.
                update_eval_progress("idle", "Cancelled", 0, 0)
                raise
            except Exception:
                update_eval_progress("idle", "Error", 0, 0)
                raise

    job = eval_jobs.start(
        signature,
        _run,
        dataset_id=ds_ctx.dataset_id,
        detector_id=det_ctx.detector_id,
    )
    # Seed this job's own total so a poll that lands before ``_run`` starts
    # (the job may be parked pending behind another in-flight eval) still
    # reports this job's numbers, not a global singleton shared with the
    # currently-running job.
    job.total = n_total

    if wait:
        job.done_event.wait(timeout=300)
        if job.status == "error":
            abort(500, message=job.error or "Evaluation computation failed")
        if job.status == "done":
            return _eval_done_payload(job)

    return {"job_id": job.job_id, "status": "running", "current": 0, "total": n_total}


@eval_bp.route("/api/eval/train-and-score/result", methods=["GET"])
@eval_bp.arguments(EvalTrainAndScoreResultQuerySchema, location="query")
@eval_bp.response(200, EvalTrainAndScoreResponseSchema)
@eval_bp.alt_response(404, description="Job not found.")
@eval_bp.alt_response(500, description="Background evaluation job failed.")
def eval_train_and_score_result(query: dict):
    """Poll a background eval train-and-score job."""
    from vtscore.concurrency.async_jobs import eval_jobs

    job_id = query["job_id"]

    job = eval_jobs.get(job_id)
    if job is None:
        # ``job_status``, not ``status``: the error envelope's own ``status``
        # is the HTTP status name, and an extra kwarg never overwrites an
        # envelope field (see ``VTSearchApi.handle_http_exception``).
        abort(404, message="Job not found", job_id=job_id, job_status="missing")

    if job.status in ("running", "pending"):
        # Read progress from the job itself (not the global ``eval_progress``
        # singleton) so overlapping evals report their own numbers.
        return {
            "job_id": job.job_id,
            "status": "running",
            "current": job.current,
            "total": job.total,
        }
    if job.status == "error":
        abort(500, message=job.error or "Evaluation computation failed", job_id=job.job_id)
    if job.status == "cancelled":
        return {"job_id": job.job_id, "status": "cancelled"}
    return _eval_done_payload(job)


@eval_bp.route("/api/eval/train-and-score/cancel/<job_id>", methods=["POST"])
@eval_bp.response(200, EvalTrainAndScoreCancelResponseSchema)
@eval_bp.alt_response(404, description="Job not found.")
def cancel_eval_train_and_score(job_id: str):
    """Cancel an in-flight eval train-and-score job.

    Sets the cancel flag on the :class:`AsyncJob`; the per-step retrain
    loop polls it cooperatively. Returns 200 even when the job has
    already finished; see ``cancel_learned_sort`` for the rationale.
    """
    from vtscore.concurrency.async_jobs import eval_jobs

    job = eval_jobs.get(job_id)
    if job is None:
        abort(404, message="Job not found")
    job.cancel()
    return {"ok": True}
