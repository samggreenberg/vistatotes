"""Blueprint for sorting and voting routes.

Migrated to ``flask_smorest`` so the routes are described in
``/api/openapi.json``.

Schema-level validation failures (missing required ``text`` / ``job_id`` /
``examples`` / ``inclusion``; type-mismatched ``inclusion``
values) surface as 422 with the
standard ``errors`` envelope. Handler-level rejects (empty / whitespace
``text``, no votes, no medias, bad files in the multipart routes, etc.)
keep their HTTP codes (400 / 404 / 500) with the standard ``message``
envelope. The two multipart routes (``/api/example-sort``,
``/api/label-file-sort``) omit ``arguments`` and declare their error
responses via ``alt_response``; same pattern as ``add-to-pile`` and
``server-media-files/upload``.
"""

import json
import logging
import threading
from pathlib import Path

from flask import request
from flask_smorest import Blueprint, abort

from vtsearch.routes._context import require_detector_header
from vtsearch.routes._http import format_exception_detail
from vtsearch.routes._progress import sort_idle
from vtsearch.routes._sort_window import windowed_sort_response

from vtscore.config import DATA_DIR
from vtscore.embedding import embed_text_query
from vtsearch.schemas.sorting import (
    CoverageAtlasNextResponseSchema,
    InclusionRequestSchema,
    InclusionResponseSchema,
    LabelFileSortResponseSchema,
    LearnedSortCancelResponseSchema,
    LearnedSortRequestSchema,
    LearnedSortResponseSchema,
    LearnedSortResultQuerySchema,
    OkResponseSchema,
    SortPageQuerySchema,
    SortPageResponseSchema,
    SortRequestSchema,
    SortResponseSchema,
    TextsortSuggestionRequestSchema,
    TextsortSuggestionsResponseSchema,
    VotesResponseSchema,
)
from vtscore.training.query_sort import (
    apply_crop_or_keep,
    cosine_sort_active,
    embed_external_labels,
    example_sort_from_paths,
    parse_label_file,
    score_embedder_for_active,
    train_and_score_active,
)
from vtsearch.state import (
    add_textsort_suggestion,
    bad_votes,
    coverage_atlas_next_sample,
    get_calibrate_count,
    get_calibration_fraction,
    get_coverage_atlas,
    get_inclusion,
    get_learned_scores,
    get_textsort_suggestions,
    get_vote_click_times,
    good_votes,
    set_inclusion,
    snapshot_medias,
    vote_region_boxes,
)
from vtscore.concurrency.progress import sort_progress, update_sort_progress

sorting_bp = Blueprint(
    "sorting",
    __name__,
    description="Text / example / learned sort, votes, inclusion, safe-thresholds, coverage atlas.",
)

# Text-sort proceeds in three phases: load the embedding model, embed the text
# query, then score every media by cosine similarity.  Reported as
# ``step``/``total_steps`` (not ``current``/``total``) so the sort channel gets
# the unified whole-job bar and an overall ETA, with the model load surfaced as
# real sub-progress within step 1.
_SORT_STEPS = 3
#: Timing-profile task name; its step names and shipped fallback weights live in
#: :data:`vtscore.timing.tasks.TASKS`. An admin ``VTSEARCH_TIMING_PROFILE``
#: measured on this deployment replaces those weights with real seconds, which
#: is what keeps the sort ETA from drifting on a cold model load.
_SORT_TASK = "text_sort"


_embedder_load_lock = threading.Lock()


def _get_embedder_for_loaded_data(snap=None):
    """Return the appropriate embedder for the currently loaded dataset.

    *snap* threads in an already-taken medias snapshot to avoid re-copying the
    medias dict under ``_state_lock``; when ``None`` a fresh snapshot is taken.
    """
    from vtscore.media import embedder_for_medias

    if snap is None:
        snap = snapshot_medias()
    return embedder_for_medias(snap)


def _sort_will_load_model(snap=None) -> bool:
    """Whether this sort will actually pay its ``load_model`` step.

    ``True`` only on the branch a user waits seconds for: an encoder exists and
    is not yet resident. Mirrors the two early returns in
    :func:`_load_embedder_with_progress` exactly — both of them leave step 1
    *unreported*, so its slice of the bar should be zero rather than merely
    small. See the call site for why the bar needs the distinction and why
    reading it here is safe.

    *snap* threads in an already-taken medias snapshot, like its neighbours.
    """
    emb = _get_embedder_for_loaded_data(snap)
    return emb is not None and not emb.models_loaded()


def _load_embedder_with_progress(snap=None):
    """Load the embedding model, forwarding its load progress to step 1 of the bar.

    If the model is already loaded this is a no-op.  A lock serialises
    concurrent callers so only one request drives the load and the others
    return early once it is warm (``_on_progress`` itself is thread-scoped —
    see ``MediaEmbedder.progress_scope`` — so the save/restore below can no
    longer trample another request's callback).  The model-load ``cur``/``tot``
    is reported as the within-step progress of step 1, so the unified bar
    animates during the load instead of parking at a step floor.

    *snap* threads in an already-taken medias snapshot to avoid re-copying the
    medias dict under ``_state_lock``; when ``None`` a fresh snapshot is taken.
    """
    emb = _get_embedder_for_loaded_data(snap)
    if emb is None:
        return

    with _embedder_load_lock:
        if emb.models_loaded():
            return

        update_sort_progress("sorting", "Loading embedder…", 0, 0, step=1, total_steps=_SORT_STEPS)
        original_cb = emb._on_progress
        emb._on_progress = lambda status, msg, cur, tot: update_sort_progress(
            "sorting", msg, cur, tot, step=1, total_steps=_SORT_STEPS
        )
        try:
            emb.load_models()
        finally:
            emb._on_progress = original_cb


@sorting_bp.route("/api/sort", methods=["POST"])
@sorting_bp.arguments(SortRequestSchema)
@sorting_bp.response(200, SortResponseSchema)
@sorting_bp.alt_response(400, description="Empty/whitespace text, no medias, or embedder doesn't support text.")
@sorting_bp.alt_response(500, description="Text sort failed (embedder error or unexpected exception).")
def sort_clips(body: dict):
    """Return medias sorted by cosine similarity to a text query."""
    text = body.get("text", "").strip()
    if not text:
        sort_idle()
        abort(400, message="text is required")

    snap = snapshot_medias()
    if not snap:
        sort_idle()
        abort(400, message="No medias loaded")

    first = next(iter(snap.values()))
    media_type = first.get("media_type", "audio")

    # Resolve the dataset's bound text embedder (v3 routing table). Reject the
    # request up-front when no text slot is bound (a vision-only dataset, e.g.
    # DINOv3 patch). Without this short-circuit we'd waste time loading a model
    # just to discover it can't embed text; the frontend already reads the same
    # ``supports_text`` signal per embedder to hide its text-search UI.
    from vtscore.state.core import get_active_context  # noqa: PLC0415

    ctx = get_active_context()
    embedder_name = ctx.routed_embedder("text")
    if embedder_name is None:
        sort_idle()
        primary = first.get("embedder", "") or "this dataset's embedder"
        abort(
            400,
            message=(
                f"Embedder '{primary}' does not support text queries. Use learned sort or load a saved sort instead."
            ),
        )

    from vtscore import timing  # noqa: PLC0415

    # A warm sort never enters step 1: ``_load_embedder_with_progress`` returns
    # before reporting it when the encoder is already resident, which is 47 of
    # every 48 sorts in a served process. Budgeting the bar for a load that will
    # not happen is what put 0.80-0.85 of this task's bar in the wrong step under
    # every profile #3521 fitted and under the shipped defaults alike (#3596), so
    # the residency the route can simply *look up* is passed to the pacing.
    #
    # Racy in one harmless direction only: models are never unloaded, so a warm
    # answer stays true, and a cold answer that another request warms first
    # merely paces this run as it was paced before this call existed.
    sort_progress.set_step_weights(
        timing.step_weights(
            _SORT_TASK,
            media_type=media_type,
            embedder=embedder_name,
            n=len(snap),
            skip_steps=() if _sort_will_load_model(snap) else ("load_model",),
        )
    )
    # Every exit below — success and abort alike — parks the tracker at "idle"
    # via ``sort_idle()``, which is what closes the recorder.
    recorder = timing.record_task(
        sort_progress, _SORT_TASK, media_type=media_type, embedder=embedder_name, auto_finish=True
    )
    recorder.start()
    recorder.set_scale(n=len(snap))
    try:
        _load_embedder_with_progress(snap)
        update_sort_progress("sorting", "Embedding text query…", 0, 0, step=2, total_steps=_SORT_STEPS)

        from vtsearch import settings

        enrich = settings.get_enrich_descriptions()
        text_vec = embed_text_query(text, media_type, enrich=enrich, embedder_name=embedder_name)
        if text_vec is None:
            sort_idle()
            abort(500, message=f"Could not embed text for media type {media_type}")

        update_sort_progress("sorting", "Computing similarities…", 0, 0, step=3, total_steps=_SORT_STEPS)
        results, threshold = cosine_sort_active(text_vec, role="text", snap=snap)
        sort_idle()
        return windowed_sort_response(results, threshold)
    except Exception as exc:
        recorder.finish(ok=False)
        from werkzeug.exceptions import HTTPException

        if isinstance(exc, HTTPException):
            # ``abort()`` above raises HTTPException; let flask-smorest's
            # handler render its envelope unchanged instead of wrapping it
            # in a 500.
            raise
        logging.getLogger(__name__).exception("text sort failed")
        sort_idle()
        abort(500, message=f"Text sort failed: {format_exception_detail(exc)}")


@sorting_bp.route("/api/sort/page", methods=["GET"])
@sorting_bp.arguments(SortPageQuerySchema, location="query")
@sorting_bp.response(200, SortPageResponseSchema)
@sorting_bp.alt_response(404, description="Unknown or expired sort token; re-run the sort.")
def sort_page(query: dict):
    """Return one window of a previously-computed ranking.

    A sort route (``/api/sort``, ``/api/example-sort``, ``/api/label-file-sort``)
    stores its full descending ``results`` list and hands back a ``sort_token``;
    this endpoint slices ``[offset, offset + limit)`` out of that cached list so
    the client can scroll deep into a large ranking without receiving the whole
    thing up front (``docs/plans/scalability.md`` S3/S17/S19).

    The token doubles as a sort-generation guard: a re-sort mints a new token,
    and an evicted/unknown token 404s so the client refetches from the top.
    """
    from vtscore.state.core import get_active_context  # noqa: PLC0415
    from vtscore.state.sort_results_cache import sort_results_cache  # noqa: PLC0415

    dataset_id = getattr(get_active_context(), "dataset_id", "") or None
    page = sort_results_cache.page(query["token"], query["offset"], query["limit"], dataset_id=dataset_id)
    if page is None:
        abort(404, message="Unknown or expired sort token; re-run the sort.")
    return page


def _learned_sort_done_payload(job) -> dict:
    """Build the JSON body returned when a learned-sort job is finished.

    ``job.result`` is the windowed response body produced by
    :func:`windowed_sort_response` in the job target, so the window metadata
    (``sort_token`` / ``total`` / ``above_threshold`` / ``has_more_below``)
    rides straight through to the client, as does the ``acq_threshold`` the
    Autopilot picks sample around.
    """
    result = job.result or {}
    return {
        "job_id": job.job_id,
        "status": "done",
        "results": result.get("results", []),
        "threshold": result.get("threshold", 0.0),
        "acq_threshold": result.get("acq_threshold"),
        "sort_token": result.get("sort_token"),
        "total": result.get("total"),
        "above_threshold": result.get("above_threshold"),
        "has_more_below": result.get("has_more_below", False),
    }


def _validate_learned_sort_inputs(labelset, good, bad) -> None:
    if labelset is not None:
        good_count = sum(1 for el in labelset.elements if el.label == "good")
        bad_count = sum(1 for el in labelset.elements if el.label == "bad")
        if good_count == 0 or bad_count == 0:
            abort(400, message="need at least one good and one bad vote")
        return
    if not good or not bad:
        abort(400, message="need at least one good and one bad vote")


@sorting_bp.route("/api/learned-sort", methods=["POST"])
@sorting_bp.arguments(LearnedSortRequestSchema)
@sorting_bp.response(200, LearnedSortResponseSchema)
@sorting_bp.alt_response(400, description="No good/bad votes available for training.")
@sorting_bp.alt_response(500, description="Background learned-sort job failed (only when ``wait=true``).")
def learned_sort(body: dict):
    """Kick off (or short-circuit) a learned-sort training job.

    Training is GIL-bound and ran inline used to stall every other request
    served by the small ``gthread`` pool (votes polls, thumbnails, and even
    media bytes).  The endpoint now hands the work off to a background daemon
    thread and returns immediately with a ``job_id``; clients poll
    :func:`learned_sort_result` until ``status == "done"``.

    A small signature cache short-circuits the no-op case: when the votes,
    detector, inclusion and thresholding settings are unchanged from the
    most recent successful run, the previous result is returned directly.

    Tests can pass ``{"wait": true}`` in the body to block until the job
    completes and receive the result inline.  The frontend leaves it false.
    """
    from vtscore.concurrency.async_jobs import learned_sort_jobs
    from vtscore.detectors.learned_sort import (
        build_learned_sort_signature,
        resolve_active_labelset,
        run_learned_sort,
    )
    from vtscore.state.core import (
        detector_acquisition_threshold,
        get_active_context,
        get_active_detector_context,
    )

    wait = body["wait"]

    snap = snapshot_medias()

    det_ctx = get_active_detector_context()
    ds_ctx = get_active_context()
    labelset, det_media_type = resolve_active_labelset(det_ctx)

    # Freeze the votes at request time, exactly as region boxes already are.
    # good_votes/bad_votes are lazy context proxies; if we passed them live,
    # the signature would capture request-time membership while the background
    # job's dict(good)/dict(bad) copy at run time could see a different set
    # (a vote POST or an ensure_votes_match_active_dataset rehydrate in
    # between), poisoning _last_done: key says V1, result trained on V2.
    good_snapshot = dict(good_votes)
    bad_snapshot = dict(bad_votes)

    _validate_learned_sort_inputs(labelset, good_snapshot, bad_snapshot)

    inclusion_value = get_inclusion()
    calibrate_count_value = get_calibrate_count()
    calibration_fraction_value = get_calibration_fraction()
    region_boxes_snapshot = dict(vote_region_boxes)

    signature = build_learned_sort_signature(
        det_ctx=det_ctx,
        ds_ctx=ds_ctx,
        snap=snap,
        labelset=labelset,
        good=good_snapshot,
        bad=bad_snapshot,
        region_boxes_snapshot=region_boxes_snapshot,
        inclusion_value=inclusion_value,
        calibrate_count_value=calibrate_count_value,
        calibration_fraction_value=calibration_fraction_value,
    )

    cached = learned_sort_jobs.cached_for(signature)
    if cached is not None:
        return _learned_sort_done_payload(cached)

    # _run closes over the resolved inputs and delegates the train → score →
    # reconcile pipeline to the library; the route only owns the job-result
    # envelope.
    def _run(job):
        results, threshold = run_learned_sort(
            det_ctx=det_ctx,
            ds_ctx=ds_ctx,
            snap=snap,
            labelset=labelset,
            det_media_type=det_media_type,
            good=good_snapshot,
            bad=bad_snapshot,
            region_boxes_snapshot=region_boxes_snapshot,
            inclusion_value=inclusion_value,
            calibrate_count_value=calibrate_count_value,
            calibration_fraction_value=calibration_fraction_value,
        )
        # The acquisition cut is read *inside* the job's dataset/detector
        # context, after training parked the fitted estimator on ``det_ctx`` -
        # this is the only sort with a detector behind it, so the only one that
        # carries one.
        acq = detector_acquisition_threshold(det_ctx, inclusion_value)
        job.result = windowed_sort_response(results, round(threshold, 4), round(acq, 4))

    job = learned_sort_jobs.start(
        signature,
        _run,
        dataset_id=ds_ctx.dataset_id,
        detector_id=det_ctx.detector_id,
    )

    if wait:
        job.done_event.wait(timeout=120)
        if job.status == "error":
            abort(500, message=job.error or "learned-sort failed")
        if job.status == "done":
            return _learned_sort_done_payload(job)

    return {"job_id": job.job_id, "status": "running", "current": 0, "total": 1}


@sorting_bp.route("/api/learned-sort/result", methods=["GET"])
@sorting_bp.arguments(LearnedSortResultQuerySchema, location="query")
@sorting_bp.response(200, LearnedSortResponseSchema)
@sorting_bp.alt_response(404, description="Job not found.")
@sorting_bp.alt_response(500, description="Background learned-sort job failed.")
def learned_sort_result(query: dict):
    """Poll a learned-sort background job.

    Returns the same shape as the POST endpoint's ``done`` response when the
    job has finished, or a ``running`` snapshot otherwise.
    """
    from vtscore.concurrency.async_jobs import learned_sort_jobs

    job_id = query["job_id"]

    job = learned_sort_jobs.get(job_id)
    if job is None:
        abort(404, message="Job not found")

    if job.status in ("running", "pending"):
        return {
            "job_id": job.job_id,
            "status": "running",
            "current": job.current,
            "total": job.total,
        }
    if job.status == "error":
        abort(500, message=job.error or "learned-sort failed", job_id=job.job_id)
    if job.status == "cancelled":
        return {"job_id": job.job_id, "status": "cancelled"}
    return _learned_sort_done_payload(job)


@sorting_bp.route("/api/learned-sort/cancel/<job_id>", methods=["POST"])
@sorting_bp.response(200, LearnedSortCancelResponseSchema)
@sorting_bp.alt_response(404, description="Job not found.")
def cancel_learned_sort(job_id: str):
    """Cancel an in-flight learned-sort job.

    Sets the cancel flag on the :class:`AsyncJob`; the training loop
    polls it cooperatively. Returns 200 even when the job has already
    finished; the caller's contract is "make sure it's no longer
    running", which also holds for done / errored / already-cancelled
    jobs.
    """
    from vtscore.concurrency.async_jobs import learned_sort_jobs

    job = learned_sort_jobs.get(job_id)
    if job is None:
        abort(404, message="Job not found")
    job.cancel()
    return {"ok": True}


@sorting_bp.route("/api/votes", methods=["GET"])
@sorting_bp.response(200, VotesResponseSchema)
def get_votes():
    """Return current good/bad votes, click times, and learned scores."""
    from vtscore.state.core import _empty_detector_context, get_active_detector_context
    from vtscore.utils.scores import finite_or  # noqa: PLC0415

    click_times = get_vote_click_times()
    learned_scores = get_learned_scores()
    det_ctx = get_active_detector_context()
    if det_ctx is not _empty_detector_context and det_ctx.detector_id:
        labelset_good_count = det_ctx.labelset_good_count
        labelset_bad_count = det_ctx.labelset_bad_count
    else:
        labelset_good_count = len(good_votes)
        labelset_bad_count = len(bad_votes)
    # Defensive guard against non-finite scores poisoning the response: every
    # write site is already sanitised via ``sigmoid_to_finite_scores``, but
    # ``round(NaN, 4)`` returns ``NaN`` and Flask's default JSON provider
    # emits the literal token ``NaN``, which is invalid JSON that breaks every
    # browser ``JSON.parse``. Belt-and-braces audit M13.
    return {
        "good": sorted(good_votes),
        "bad": sorted(bad_votes),
        "verified": sorted(det_ctx.verified_ids),
        "click_times": {str(k): v for k, v in click_times.items()},
        "learned_scores": {str(k): round(finite_or(v), 4) for k, v in learned_scores.items()},
        "labelset_good_count": labelset_good_count,
        "labelset_bad_count": labelset_bad_count,
        # Region boxes live only on good votes (popped on bad/un-vote); intersect
        # with ``good_votes`` defensively so a stale entry can't leak through.
        "good_region_boxes": {
            str(k): [float(c) for c in box]
            for k, box in vote_region_boxes.items()
            if k in good_votes and box is not None
        },
    }


@sorting_bp.route("/api/votes/clear", methods=["POST"])
@sorting_bp.response(200, OkResponseSchema)
@require_detector_header
def clear_votes_route():
    """Clear all votes without clearing medias.

    Used by the Label flow to reset votes before importing a model's labelset
    so that labels from a previous session don't contaminate the new model.
    """
    from vtsearch.state import clear_votes

    clear_votes()
    return {"ok": True}


@sorting_bp.route("/api/textsort-suggestions", methods=["GET"])
@sorting_bp.response(200, TextsortSuggestionsResponseSchema)
def get_textsort_suggestions_route():
    """Return stored text-sort suggestions (most recent last)."""
    return {"suggestions": get_textsort_suggestions()}


@sorting_bp.route("/api/textsort-suggestions", methods=["POST"])
@sorting_bp.arguments(TextsortSuggestionRequestSchema)
@sorting_bp.response(200, OkResponseSchema)
@sorting_bp.alt_response(400, description="Empty or whitespace-only ``text``.")
def add_textsort_suggestion_route(body: dict):
    """Store a text-sort query as a suggested name for detectors/labelsets."""
    text = body.get("text", "").strip()
    if not text:
        abort(400, message="text is required")
    add_textsort_suggestion(text)
    return {"ok": True}


@sorting_bp.route("/api/inclusion", methods=["GET"])
@sorting_bp.response(200, InclusionResponseSchema)
def get_inclusion_route():
    """Get the current Inclusion setting and the cutoff it resolves to."""
    return {"inclusion": get_inclusion(), "threshold": _active_detector_threshold()}


@sorting_bp.route("/api/inclusion", methods=["POST"])
@sorting_bp.arguments(InclusionRequestSchema)
@sorting_bp.response(200, InclusionResponseSchema)
def set_inclusion_route(body: dict):
    """Set the Inclusion setting (clamped to ``[-10, 10]``).

    Inclusion is a pure cutoff knob: this re-derives the active detector's
    threshold from its cached fold orderings (no MLP retrain) and, in Find
    mode, re-splits the unverified items over the frozen scores.  The new
    cutoff is returned so the Find slider can move the green/red line.
    """
    # The clamp is not spelled out here: ``settings.validate_inclusion`` is
    # generated from the ``[-10, 10]`` bound declared once on
    # ``UserSettings.inclusion``, so this endpoint and ``PUT /api/settings``
    # cannot drift apart (issue #3416). The schema admits any number
    # (``fields.Raw`` plus a numeric check), so truncate toward zero first --
    # the pydantic field is an ``int`` and rejects a fractional value, and
    # truncate-then-clamp is what this route has always done.
    #
    # This note stays a comment rather than joining the docstring above:
    # flask-smorest publishes the docstring as the endpoint's OpenAPI
    # ``description``, and internal wiring is not part of the contract.
    from vtsearch import settings  # noqa: PLC0415

    try:
        new_inclusion = settings.validate_inclusion(int(body["inclusion"]))
    except (TypeError, ValueError) as exc:
        abort(400, message=str(exc))
    set_inclusion(new_inclusion)
    return {"inclusion": get_inclusion(), "threshold": _active_detector_threshold()}


def _active_detector_threshold() -> float | None:
    """The active detector context's current cutoff, or ``None`` if unset.

    Returns ``None`` for the empty / request-missing sentinels (no detector
    identified), which don't carry a meaningful threshold.
    """
    from vtscore.state.core import _empty_detector_context, get_active_detector_context

    det_ctx = get_active_detector_context()
    if det_ctx is _empty_detector_context:
        return None
    return getattr(det_ctx, "threshold", None)


def _parse_crop_params(raw: str | None) -> dict | None:
    """Parse a JSON string of crop bounds, or return None if absent.

    Returns ``None`` when *raw* is empty/missing.  Returns a dict otherwise
    (caller validates the contents against the target media type).
    """
    if not raw:
        return None
    try:
        params = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(params, dict):
        return None
    return params


@sorting_bp.route("/api/example-sort", methods=["POST"])
@sorting_bp.response(200, SortResponseSchema)
@sorting_bp.alt_response(
    400,
    description="No file provided, no filename, or no medias loaded.",
)
@sorting_bp.alt_response(500, description="Example sort failed (embedder error or unexpected exception).")
def example_sort():
    """Sort medias by similarity to an uploaded example media file.

    Optional ``crop_params`` form field carries a JSON object with the
    bounds for a user-cropped sub-region (e.g. ``{"start": 1.5, "end": 3}``
    for audio or ``{"box": [x1, y1, x2, y2]}`` for images).  When present
    the file is cropped server-side before embedding.
    """
    if "file" not in request.files:
        abort(400, message="No file provided")

    file = request.files["file"]
    if not file.filename:
        abort(400, message="No file selected")

    if not snapshot_medias():
        abort(400, message="No medias loaded")

    try:
        # Save uploaded file to a unique temp location to avoid race conditions
        import uuid

        suffix = Path(file.filename).suffix or ".bin"
        temp_path = DATA_DIR / f"temp_example_{uuid.uuid4().hex}{suffix}"
        DATA_DIR.mkdir(exist_ok=True)
        file.save(temp_path)

        try:
            crop_params = _parse_crop_params(request.form.get("crop_params"))
            apply_crop_or_keep(temp_path, crop_params)
            results, thresh = example_sort_from_paths([temp_path])
        finally:
            # Clean up temp file even if sorting raises
            temp_path.unlink(missing_ok=True)

        return windowed_sort_response(results, thresh)

    except Exception as exc:
        from werkzeug.exceptions import HTTPException

        if isinstance(exc, HTTPException):
            raise
        logging.getLogger(__name__).exception("example-sort failed")
        abort(500, message=f"Example sort failed: {format_exception_detail(exc)}")


@sorting_bp.route("/api/label-file-sort", methods=["POST"])
@sorting_bp.response(200, LabelFileSortResponseSchema)
@sorting_bp.alt_response(
    400,
    description=(
        "No file / no filename, no medias loaded, no embedder, invalid label file, "
        "no labels in file, too few valid labeled files, or missing good/bad split."
    ),
)
@sorting_bp.alt_response(500, description="Label file sort failed (unexpected exception).")
def label_file_sort():
    """Train MLP on external media files from a label file, then sort all medias.

    On a patch dataset each result also carries the ``best_region`` box of the
    region that drove its score, matching the cosine- and learned-sort paths.
    """
    if "file" not in request.files:
        abort(400, message="No file provided")

    file = request.files["file"]
    if not file.filename:
        abort(400, message="No file selected")

    if not snapshot_medias():
        abort(400, message="No medias loaded")

    # Embed external labels with (and later score against) the dataset's
    # score embedder so training and scoring share one space.
    emb, score_name = score_embedder_for_active()
    if emb is None:
        abort(400, message="No embedder available for loaded dataset")

    try:
        try:
            labels = parse_label_file(file)
        except ValueError as exc:
            abort(400, message=str(exc))
        X_list, y_list, loaded, skipped = embed_external_labels(labels, emb)

        if loaded < 2:
            abort(
                400,
                message=f"Need at least 2 valid labeled files (loaded {loaded}, skipped {skipped})",
            )

        from vtscore.detectors.training import validate_good_bad_split

        try:
            validate_good_bad_split(y_list)
        except ValueError:
            abort(400, message="Need at least one good and one bad labeled example")

        results, threshold = train_and_score_active(X_list, y_list, score_name)
        threshold = round(threshold, 4)
        return {**windowed_sort_response(results, threshold), "loaded": loaded, "skipped": skipped}

    except Exception as exc:
        from werkzeug.exceptions import HTTPException

        if isinstance(exc, HTTPException):
            raise
        logging.getLogger(__name__).exception("label-file-sort failed")
        abort(500, message="Label file sort failed")


@sorting_bp.route("/api/coverage-atlas/next", methods=["GET", "POST"])
@sorting_bp.response(200, CoverageAtlasNextResponseSchema)
@sorting_bp.alt_response(400, description="Invalid score keys/values or threshold value (POST only).")
def coverage_atlas_next():
    """Return the next diverse sample from the Coverage Atlas.

    Accepts an optional POST body with ``{"scores": {id: score, ...},
    "threshold": <float>}`` so the sort mode influences which element is
    picked from the next evidence-free node.  When a threshold is provided,
    the node's median score determines direction: above-threshold nodes yield
    the lowest-scored element (surprise in a "good" region), while
    below-threshold nodes yield the highest-scored element (surprise in a
    "bad" region).  In nodes with a concentrated direction the surprise
    extremum is drawn from the node's typical half, so the probe lands on a
    representative counterexample rather than a lone oddball.  Without
    scores the node's most typical element is returned.

    Returns ``{"id": <media_id>}`` or ``{"id": null}`` when the atlas is
    exhausted or not yet built.  Also includes ``coverage_level`` (the
    number of consecutive BFS-order evidence-bearing nodes) so the frontend
    can display progress, and ``exhausted`` (bool) which is true when the
    atlas exists but every node already carries evidence.
    """
    scores: dict[int, float] | None = None
    threshold: float | None = None
    if request.method == "POST":
        # ``request.get_json(silent=True)`` keeps the legacy lenient body
        # handling; flask-smorest's ``arguments`` would 422 on a missing
        # body, but we want GET / POST to behave identically when nothing
        # is sent. The shape-level validation lives in the schema; per-
        # value int-coercion stays in the handler so we can return a 400
        # with a custom message.
        data = request.get_json(silent=True) or {}
        raw_scores = data.get("scores")
        if isinstance(raw_scores, dict):
            try:
                scores = {int(k): float(v) for k, v in raw_scores.items()}
            except (ValueError, TypeError):
                abort(400, message="Invalid score keys or values")
        raw_threshold = data.get("threshold")
        if raw_threshold is not None:
            try:
                threshold = float(raw_threshold)
            except (ValueError, TypeError):
                abort(400, message="Invalid threshold value")

    atlas = get_coverage_atlas()
    next_id = coverage_atlas_next_sample(scores=scores, threshold=threshold)
    level = atlas.coverage_level() if atlas is not None else 0
    exhausted = atlas is not None and next_id is None
    return {"id": next_id, "coverage_level": level, "exhausted": exhausted}
