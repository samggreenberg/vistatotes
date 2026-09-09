"""Dataset loading orchestration: background threading, gate handoff, staging.

This module strings the post-import :mod:`vtscore.datasets.stages` together
into a background-threaded dataset load: it acquires the download/embed
concurrency gates, runs the importer, then drives the clipper, embed, dedup,
coverage-atlas, registry, and (optional) projection stages while routing each
stage's progress into the shared loading-task tracker. The per-stage work
itself lives under :mod:`vtscore.datasets.stages`; the
:class:`~vtscore.concurrency.gate.ConcurrencyGate` primitive lives in
:mod:`vtscore.concurrency.gate`.
"""

from __future__ import annotations

import gc
import json
import time
import traceback
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Callable
from uuid import uuid4

from vtscore.config import CoreConfig, DATA_DIR
from vtscore.concurrency.gate import ConcurrencyGate
from vtscore.concurrency.progress import (
    CancelledError,
    clear_thread_progress,
    loading_tasks,
    set_thread_progress,
)
from vtscore.datasets import export_dataset_to_file
from vtscore.datasets.clipper_chain import append_cleaner_steps
from vtscore.datasets.loader import apply_custom_metadata_md5
from vtscore.datasets.registry import unregister_dataset as _reg_unregister
from vtscore.state import DatasetContext, clear_all, register_context

from vtscore.datasets.stages._common import (
    AdaptiveLoadPacer,
    FinalizeProgress,
    _STATUS_TO_STEP,
    _TOTAL_LOAD_STEPS,
    _origin_to_str,
    load_cost_terms,
    load_step_weights,
)
from vtscore.datasets.stages._load_profiler import resolve_download_size_mb, start_profiler
from vtscore.datasets.stages.clipper import _apply_clipper_stage, _relazify_reference_clips_stage
from vtscore.datasets.stages.embedding import embed_missing, _embed_missing_stage
from vtscore.datasets.stages.finalize import (
    _build_coverage_atlas_stage,
    _collapse_duplicates_stage,
    _collapse_near_duplicates_stage,
    _drop_none_embeddings_stage,
)
from vtscore.datasets.stages.projection import _build_projection_stage, _maybe_signpost_texts_stage
from vtscore.datasets.stages.registry import _register_and_migrate
from vtscore.datasets.thumbnail_warm import start_archive_thumbnail_warm
from vtscore.timing import record_task, step_weights


# Two independent gates control how many dataset loads can run concurrently
# in each phase.  The download/import phase is bandwidth- and disk-bound;
# the embedding phase is CPU/GPU- and RAM-bound.  Splitting the gates lets
# one dataset download while another is still embedding, instead of forcing
# strict end-to-end serialisation.  Limits are user-configurable via the
# ``max_concurrent_dataset_downloads`` and ``max_concurrent_dataset_embeddings``
# settings; defaults derive from the host's CPU/GPU counts (see
# :func:`vtscore.embedding.loader.default_concurrent_downloads` and
# :func:`vtscore.embedding.loader.default_concurrent_embeddings`).
_download_gate = ConcurrencyGate(lambda: CoreConfig.from_settings().max_concurrent_dataset_downloads)
_embed_gate = ConcurrencyGate(lambda: CoreConfig.from_settings().max_concurrent_dataset_embeddings)


def reset_load_gates_for_tests() -> None:
    """Rebind both load gates to fresh instances.  For test isolation.

    The gates are process globals, so under ``pytest -n auto`` every test in a
    worker shares them.  A test that leaves a background load thread running
    leaves that thread holding a permit, and the next test in the process sees
    a gate that is already partly full — which is how ``TestLoadingGates``
    became a lottery over xdist worker assignment (issue #3613).

    Rebinding rather than zeroing the counters is what makes this safe in the
    presence of a thread we could not stop: the leaked thread still holds — and
    eventually releases — the *old* gate object, so it cannot drive the new
    gate's count negative.  :class:`_LoadGateController` cooperates by
    remembering the gate object it acquired instead of re-resolving these
    globals at release time.
    """
    global _download_gate, _embed_gate

    _download_gate = ConcurrencyGate(lambda: CoreConfig.from_settings().max_concurrent_dataset_downloads)
    _embed_gate = ConcurrencyGate(lambda: CoreConfig.from_settings().max_concurrent_dataset_embeddings)


# ---------------------------------------------------------------------------
# App-side persistence hook
# ---------------------------------------------------------------------------
# The library remembers the user's per-media-type embedder pick by calling
# whatever the app installs here.  Default is a no-op so this module doesn't
# need to import ``vtsearch.settings`` (Phase 2 of
# ``../docs/architecture.md``).  ``vtsearch/shim/`` registers the
# real implementation; ``vtsearch.settings.set_last_embedder_for_media_type``
# is wired at app startup.
_last_embedder_persistence_hook: Callable[[str, str], None] | None = None


def register_last_embedder_persistence_hook(fn: Callable[[str, str], None]) -> None:
    """Install the callback used to persist the user's per-media-type embedder pick.

    The Flask app installs ``vtsearch.settings.set_last_embedder_for_media_type``
    as the hook at startup so library callers don't have to know about the
    user-pref persistence layer.  Library-only callers can leave the default
    in place (no persistence).
    """
    global _last_embedder_persistence_hook
    _last_embedder_persistence_hook = fn


def clear_dataset():
    """Clear the current dataset, votes, and all related state."""
    clear_all()


def _get_embedder_for_medias(media_dict: dict):
    """Resolve the embedder for *media_dict*.

    Imported lazily to keep this module's import graph shallow;
    ``vtscore.media`` pulls in the whole media-plugin discovery pass.
    """
    from vtscore.media import embedder_for_medias  # noqa: PLC0415

    return embedder_for_medias(media_dict)


def _recorded_embedder_name(media_dict: dict, requested: str) -> str:
    """The embedder name a timing row should carry for this load.

    The caller's pick wins when there is one. When there is not, the load still
    *used* an encoder — the media type's default, resolved deep inside the embed
    stage and stamped onto every media — and recording the blank instead files
    the run under the media rollup, so the exact ``(device, media, embedder)``
    cell the profile is keyed on can never be populated by an import that did
    not name one (#3345). Reads the name off the medias rather than resolving an
    embedder object, because by this point they carry it and the object is not
    needed.
    """
    if requested:
        return requested
    if not media_dict:
        return ""
    first = next(iter(media_dict.values()), None) or {}
    return str(first.get("embedder", "") or "")


def _parse_bool(value: Any) -> bool:
    """Coerce a request-supplied flag to ``bool``.

    Thin alias for :func:`vtscore.plugins.parse_checkbox`, kept so the many
    call sites in this module read unchanged; the coercion itself is shared
    with the CLI so the two cannot drift.
    """
    from vtscore.plugins import parse_checkbox  # noqa: PLC0415

    return parse_checkbox(value)


def _parse_embedder_list(value: Any) -> list[str] | None:
    """Coerce a request-supplied embedder list to ``list[str]`` (or ``None``).

    The v3 create-time three-role picker sends the bound trio (text / patch /
    structural picks, deduped) under the ``embedders`` field.  It arrives as a
    native list on JSON-body routes and as a string on multipart routes (a JSON
    array string, or a comma-separated fallback).  ``None`` / empty in →
    ``None`` (the caller falls back to the single ``embedder`` field — the
    pre-trio path).  Order is preserved and blanks/dupes are dropped.
    """
    if value is None:
        return None
    items: list[Any]
    if isinstance(value, (list, tuple)):
        items = list(value)
    else:
        text = str(value).strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
            items = parsed if isinstance(parsed, list) else [parsed]
        except (TypeError, ValueError):
            items = text.split(",")
    out: list[str] = []
    for item in items:
        name = str(item).strip()
        if name and name not in out:
            out.append(name)
    return out or None


def _normalize_media_type(value: str) -> str:
    """Normalize a media type string (folder_import_name or type_id) to a canonical type_id."""
    value = (value or "").strip()
    if not value:
        return ""
    try:
        from vtscore.media import get_by_folder_name, normalize_type_id  # noqa: PLC0415

        try:
            return get_by_folder_name(value).type_id
        except KeyError:
            return normalize_type_id(value)
    except Exception:
        return value


def _parse_chain_field(raw: Any) -> list[dict] | None:
    """Decode a ``clipper_chain`` importer field value into a step list.

    The field may arrive as a JSON string (typical client encoding) or as
    a native list (programmatic callers). Returns ``None`` for missing /
    malformed values so the legacy single-clipper path stays in effect.
    """
    if raw is None or raw == "":
        return None
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        import json as _json

        try:
            decoded = _json.loads(raw)
        except (TypeError, ValueError):
            return None
        if isinstance(decoded, list):
            return decoded
        return None
    return None


# ---------------------------------------------------------------------------
# Parallel-safe background loading
# ---------------------------------------------------------------------------


@dataclass
class _ImportTask:
    """Everything :func:`_start_import_task` sets up before the worker runs.

    ``tracker`` is the per-task :class:`~vtscore.concurrency.progress.ProgressTracker`
    (registered under ``task_id`` on ``loading_tasks``), ``recorder`` is the
    already-started :mod:`vtscore.timing` recorder, and ``request_user`` is the
    caller identity the worker replays so per-user settings writes land in the
    right file.  ``total_steps`` is the flow's step structure, carried here so
    everything downstream (notably :class:`_LoadGateController`) reports the
    same shape the tracker was built with.
    """

    task_id: str
    tracker: Any
    recorder: Any
    request_user: str
    total_steps: int


def _start_import_task(
    *,
    prefix: str,
    family: str,
    display_name: str,
    total_steps: int,
    media_type: str = "",
    embedder: str = "",
    weights: list[float] | None = None,
    extra_fields: dict[str, Any] | None = None,
    status_phases: dict[str, str] | None = None,
    created_by: str = "",
) -> _ImportTask:
    """Register a background import on ``loading_tasks`` and arm its recorder.

    The two import pipelines below — a full dataset load and a combine-flow
    staging import — open identically: mint a task id, create the per-task
    tracker (so two concurrent imports never interleave one channel), start the
    timing recorder that labels each measured phase, and snapshot the user who
    asked for the work.  Only the family name, the step structure, and the
    tracker's extra fields differ, so they are parameters here.

    The caller writes its own first ``tracker.update`` (rather than this
    function writing a generic one) because the load flow subscribes its
    env-gated phase profiler in between: a profiler that missed the step-1
    update would date the download phase from the wrong instant.
    """
    from vtscore.state.current_user import get_current_user  # noqa: PLC0415

    task_id = f"{prefix}{uuid4().hex[:8]}"
    tracker = loading_tasks.create_task(
        task_id,
        display_name,
        media_type=media_type,
        embedder=embedder,
        extra_fields=extra_fields,
        step_weights=weights,
    )
    recorder = record_task(
        tracker,
        family,
        media_type=media_type,
        embedder=embedder,
        status_phases=status_phases,
    )
    recorder.start()
    return _ImportTask(
        task_id=task_id,
        tracker=tracker,
        recorder=recorder,
        request_user=created_by or get_current_user(),
        total_steps=total_steps,
    )


def _spawn_import_worker(task: _ImportTask, body: Callable[[], None]) -> str:
    """Run *body* on a daemon thread scoped to *task*; return its ``task_id``.

    Closes the harness :func:`_start_import_task` opens.  *body* is a zero-arg
    closure over whatever the pipeline needs; everything around it is identical
    for both flows:

    * the requesting user is replayed into the worker, so background per-user
      settings writes resolve to the right file;
    * the thread-local progress hook is cleared however the body exits, so a
      later job on this thread cannot narrate itself on a dead channel;
    * ``mark_finished`` runs last, so a caller waiting on
      ``has_active_tasks() == False`` (or ``is_finished``) sees fully
      cleaned-up worker state — gates released, contexts restored;
    * the thread is registered with ``set_worker`` *before* it starts, so a
      cancel arriving in the same instant can tell "not started yet" from
      "nothing here" (#3167).
    """

    def run() -> None:
        from vtscore.state.current_user import thread_user  # noqa: PLC0415

        try:
            with thread_user(task.request_user):
                body()
        finally:
            clear_thread_progress()
            loading_tasks.mark_finished(task.task_id)

    worker = threading.Thread(target=run, daemon=True)
    loading_tasks.set_worker(task.task_id, worker)
    worker.start()
    return task.task_id


class _LoadGateController:
    """Tracks which import gate (download / embed) is currently held.

    Splits gate-acquisition concerns out of the task body: the importer
    runs under the download gate (bandwidth-bound), and we swap to the
    embed gate as soon as the importer signals it's started embedding so
    another dataset can begin downloading in parallel.

    Both import pipelines use it, so *total_steps* is a parameter rather than
    ``_TOTAL_LOAD_STEPS``: the "waiting for other datasets" messages have to
    carry the step structure of whichever flow is queueing, or the whole-job
    bar rescales the moment a load parks on a full gate.
    """

    def __init__(self, tracker, total_steps: int = _TOTAL_LOAD_STEPS) -> None:
        self._tracker = tracker
        self._total_steps = total_steps
        self._held: str | None = None
        #: The gate object :meth:`acquire` last took a permit from.  Held as an
        #: object rather than re-resolved from the module global at release
        #: time so a controller always releases *the gate it acquired*, even if
        #: the global has since been rebound (which is what
        #: :func:`reset_load_gates_for_tests` does between tests).
        self._held_gate: ConcurrencyGate | None = None

    @property
    def held(self) -> str | None:
        return self._held

    def acquire(self, gate: ConcurrencyGate, name: str, wait_msg: str) -> None:
        if gate.acquire(blocking=False):
            self._held = name
            self._held_gate = gate
            return
        self._tracker.update("loading", wait_msg, 0, 0, step=1, total_steps=self._total_steps)
        while not gate.acquire(timeout=0.5):
            self._tracker.check_cancelled()
        self._held = name
        self._held_gate = gate

    def acquire_download(self) -> None:
        self.acquire(_download_gate, "download", "Waiting for other datasets to finish downloading…")

    def swap_to_embed(self) -> None:
        if self._held == "embed":
            return
        if self._held == "download":
            self.release()
        self.acquire(_embed_gate, "embed", "Waiting for other datasets to finish embedding…")

    def release(self) -> None:
        if self._held_gate is not None:
            self._held_gate.release()
        self._held = None
        self._held_gate = None


def _make_stepped_progress(controller: _LoadGateController, pacer):
    """Build the importer-side progress callback.

    Routes status updates into the load's :class:`AdaptiveLoadPacer` (which
    maps them onto the unified bar) with the right step number, and triggers
    the download→embed gate swap on the first ``"embedding"`` status so a
    queued download can start.
    """

    def stepped(status: str, message: str = "", current: int = 0, total: int = 0) -> None:
        pacer.check_cancelled()
        if status == "idle":
            return
        if status == "embedding" and controller.held != "embed":
            controller.swap_to_embed()
        step = _STATUS_TO_STEP.get(status)
        pacer.update(status, message, current, total, step=step, total_steps=_TOTAL_LOAD_STEPS)

    return stepped


def _run_importer(load_fn, ctx: DatasetContext, stepped) -> None:
    """Invoke *load_fn* under thread-local progress, populating ctx.medias."""
    import inspect  # noqa: PLC0415

    set_thread_progress(stepped)
    try:
        sig = inspect.signature(load_fn)
        if sig.parameters:
            load_fn(ctx.medias)
        else:
            load_fn()
    finally:
        clear_thread_progress()


def _tag_origins(media_dict: dict, origin: dict) -> None:
    """Stamp *origin* onto medias that don't already carry one.

    Each media gets its own fresh copy of the origin dict (including a
    fresh ``params``).  Sharing one dict by reference across siblings
    means any later mutation of ``media["origin"]["params"]`` on one
    media silently corrupts every other media stamped by the same load;
    and that aliasing also survives pickle round-trips via backreferences.
    """
    for media in media_dict.values():
        if media.get("origin") is None:
            media["origin"] = {
                "importer": origin.get("importer", ""),
                "params": dict(origin.get("params", {})),
            }
        if not media.get("origin_name"):
            media["origin_name"] = media.get("filename", "")


def _warmup_embedder_async(media_dict: dict) -> None:
    """Warm up the embedder (model load + text-encoder prime) in a daemon thread.

    Fire-and-forget: the caller doesn't wait, and there is no progress
    surface; the dataset is usable for grid-browsing immediately, and
    text sort waits behind its own ``_embedder_load_lock`` (see
    ``vtsearch/routes/sorting.py:_load_embedder_with_progress``) on first
    use.  ``MediaEmbedder.load_models`` is idempotent and serialised by
    a per-class lock, so racing this thread against an on-demand sort
    load is safe.
    """

    def _run() -> None:
        emb = _get_embedder_for_medias(media_dict)
        if emb is None:
            return
        try:
            # Explicitly silent: this thread has no progress surface (see the
            # docstring), and an unscoped load would narrate itself on the
            # global dataset channel instead — a phantom import, right as the
            # real one finished (#3167).
            with emb.silent_progress():
                emb.load_models()
                emb.embed_text("warmup")
        except Exception:
            pass

    threading.Thread(target=_run, name="warmup-embedder", daemon=True).start()


def _failure_message(exc: BaseException, fallback: str) -> str:
    """Map a background-import exception onto the string shown to the user.

    Both import pipelines (a dataset load and a combine-flow staging import)
    run the same kinds of work — an importer, then an embed — so they fail in
    the same ways and must say the same things.  ``"Cancelled"`` in particular
    is load-bearing rather than cosmetic: ``dashboard-loading-tasks.service.ts``
    and ``toast.service.ts`` both test for that exact string to tell a
    user-requested stop from a genuine failure, so a flow that surfaced
    ``CancelledError``'s own text instead ("Operation cancelled by user") popped
    a red *failed* toast for a cancel the user asked for.

    *fallback* is the last-resort text for an exception carrying no message.
    The traceback is printed for the two cases where the server log is the only
    place the detail survives; a cancel and an OOM carry all they have in the
    returned string.
    """
    if isinstance(exc, CancelledError):
        return "Cancelled"
    if isinstance(exc, ImportError):
        traceback.print_exc()
        return f"Missing dependency: {exc}. Install all required packages with: pip install -e '.[cpu,dev]'"
    if isinstance(exc, MemoryError):
        return "Out of memory: this dataset is too large. Try a smaller dataset or free up system RAM."
    traceback.print_exc()
    return str(exc) or repr(exc) or fallback


def _handle_load_failure(
    exc: BaseException,
    context_id: str,
    tracker,
    registry_entry_id: str | None = None,
) -> None:
    """Unregister the context and write the failure into *tracker*.

    If *registry_entry_id* is set, the on-disk registry entry (and its
    backing pkl) is also removed; this prevents an orphaned dashboard
    row when a load fails after :func:`_register_and_migrate` has
    already written the entry.
    """
    from vtscore.state.core import unregister_context  # noqa: PLC0415

    error = _failure_message(exc, "Unknown error during dataset loading")

    unregister_context(context_id)
    if registry_entry_id:
        try:
            _reg_unregister(registry_entry_id)
        except Exception:
            traceback.print_exc()
    gc.collect()
    tracker.update("idle", "", 0, 0, error=error, step=None, total_steps=None)


def _park_load_terminal(tracker, n_items: int) -> None:
    """Park *tracker* at a terminal state once the load thread has unwound.

    The failure paths already do this — :func:`_handle_load_failure` writes
    ``idle`` plus an ``error`` — but the success path historically wrote
    nothing, so a load that finished cleanly left its tracker on whatever
    "loading …" message happened to fire last.  Nothing ever cleared it:
    :meth:`~vtscore.concurrency.progress.LoadingTasksTracker.has_active_tasks`
    stayed true until the finished entry aged out, the dashboard's
    just-finished test (``status == "idle" && !error``) never fired so the
    registry refresh waited on the prune instead, and every external signal
    kept saying "still importing" about a thread that had exited (#3167).

    Terminal states belong in a ``finally``.  A tracker that already carries an
    error is left alone: that failure *is* its terminal state.
    """
    if tracker.get().get("error"):
        return
    tracker.update(
        "idle",
        f"Loaded {n_items} item(s)",
        n_items,
        n_items,
        step=_TOTAL_LOAD_STEPS,
        total_steps=_TOTAL_LOAD_STEPS,
    )


def _run_origin_load_in_background(
    load_fn,
    origin: dict,
    *,
    name: str = "",
    clipper: str = "",
    clipper_params: dict | None = None,
    chain_steps: list[dict] | None = None,
    embedder: str = "",
    embedders: list[str] | None = None,
    created_by: str = "",
    media_type: str = "",
    build_projection: bool = False,
    merge_near_duplicates: bool = False,
    dataset_id: str = "",
    n_hint: int | None = None,
    download_size_mb_hint: float | None = None,
) -> str:
    """Run a dataset load in a background thread with standard error handling.

    *load_fn* is called with a single argument (the target medias dict);
    and should populate it in-place.  Everything after (origin tagging,
    clipping, dedup, coverage atlas, registry, embedder warm-up) is handled
    automatically.

    *embedder* is the primary create-time embedder (recorded as each media's
    primary and used for the per-media-type persistence hint / task display).
    *embedders* is the optional v3 trio (text / patch / structural create-time
    picks): when set, every name is embedded during ingest so a multi-embedder
    dataset is produced.  ``None`` falls back to the single *embedder* — the
    pre-trio create path, unchanged.

    The dataset context is NOT activated during loading.  It is activated
    only upon successful completion, and only if no other dataset is
    currently active.

    Returns the task_id that can be used to poll progress or cancel.
    """
    # Remember the user's embedder pick per media type so the next dataset
    # importer modal can pre-select it even when no loaded dataset is
    # around to supply the same hint via ``guessedMediaEmbedder``.
    if media_type and embedder and _last_embedder_persistence_hook is not None:
        try:
            _last_embedder_persistence_hook(media_type, embedder)
        except Exception:
            pass

    ingest_started_at = time.time()
    # ``_start_import_task`` mints the id, creates the per-task tracker, arms
    # the generic cross-task recorder (VTSEARCH_TIMING_RECORD, which every other
    # long-running family also feeds — without it an admin who armed only the
    # documented env var got rows for every task *except* the imports, #2845),
    # and snapshots the user that triggered the load so background per-user
    # state (settings writes, settings_source sync) resolves correctly.
    # ``status_phases`` splits step 1 into its two byte-scaled phases, which
    # only the status string tells apart.
    task = _start_import_task(
        prefix="_loading_",
        family="dataset_load",
        display_name=name or _origin_to_str(origin),
        total_steps=_TOTAL_LOAD_STEPS,
        media_type=media_type,
        embedder=embedder,
        weights=load_step_weights(media_type, n=n_hint, download_size_mb=download_size_mb_hint, embedder=embedder),
        status_phases={"extracting": "extract"},
        created_by=created_by,
    )
    task_id, tracker, timing_recorder = task.task_id, task.tracker, task.recorder
    # Env-gated per-phase timing recorder (VTSEARCH_PROFILE_LOAD); a no-op
    # stand-in and zero-cost when off. It runs alongside the generic recorder
    # above rather than replacing it: the two answer different questions (that
    # one fits the shared timing profile; this one additionally splits
    # cold/warm model loads and finalize sub-slots), they write to separate
    # files, and each is independently armed. Subscribed before the first phase
    # fires — hence before the step-1 update below, not inside
    # ``_start_import_task``. See scripts/profiling/README.md.
    profiler = start_profiler(tracker, media_type, embedder)
    tracker.update("loading", "Preparing dataset...", step=1, total_steps=_TOTAL_LOAD_STEPS)

    def load_task():
        from vtscore.state.core import thread_dataset_context  # noqa: PLC0415

        ctx = DatasetContext(task_id)
        ctx.merge_near_duplicates = merge_near_duplicates
        # Pin the in-flight context to this thread so importers, clippers,
        # dedup, coverage-atlas, and label-sync helpers that resolve via
        # ``get_active_context()`` see the dataset being built, not the
        # empty fallback context.  Without this, mutations addressed at
        # the active context (e.g. label restoration, vote replay) land
        # on ``_empty_dataset_context`` and are silently lost.
        #
        # ``thread_dataset_context`` snapshots the prior thread-local value on
        # entry and restores it on exit, so a future pooled / reused worker
        # thread cannot leak context across jobs; ``_spawn_import_worker`` does
        # the same for the user identity, and runs ``mark_finished`` after both
        # scopes have exited so callers waiting on ``has_active_tasks() ==
        # False`` see fully cleaned-up worker state.
        context_id = task_id
        registry_entry_id: str | None = None
        controller = _LoadGateController(tracker, task.total_steps)
        # Pace the unified bar from the per-phase cost terms, rebasing on what
        # actually happens (cached archives, observed bandwidth, skipped
        # phases). All stage progress below routes through the pacer.
        cost_terms, terms_calibrated = load_cost_terms(
            media_type, n=n_hint, download_size_mb=download_size_mb_hint, embedder=embedder
        )
        pacer = AdaptiveLoadPacer(tracker, cost_terms, calibrated=terms_calibrated)
        stepped = _make_stepped_progress(controller, pacer)
        profiler.bind_thread()  # so FinalizeProgress.begin stamps land here (no-op when off)
        # Same reason, for the generic recorder: the stage that decides whether
        # this import embeds or reads a cached pkl is many frames below here,
        # and binds the fact to the thread rather than to an argument (#3521).
        timing_recorder.bind_thread()

        try:
            with thread_dataset_context(ctx):
                try:
                    controller.acquire_download()
                    pacer.update("loading", "Preparing new dataset…", 0, 0, step=1, total_steps=_TOTAL_LOAD_STEPS)
                    register_context(ctx)
                    gc.collect()

                    _run_importer(load_fn, ctx, stepped)
                    tracker.check_cancelled()

                    # Backstop: an importer that completes without raising but
                    # produces zero medias would otherwise sail through clipping,
                    # dedup, and registry steps and surface as a green dashboard
                    # row with 0 items.  Fail loudly instead, mirroring the
                    # staging-flow guard at ``_stage_importer_in_background``.
                    if not ctx.medias:
                        raise ValueError("Import produced no medias.")

                    # Post-load stages are CPU/GPU-bound and touch embeddings;
                    # gate them on the embed semaphore.  Calling swap here
                    # unconditionally is also the safety net for minimalist
                    # importers that complete without firing an ``"embedding"``
                    # status: ``_make_stepped_progress``'s callback-driven swap
                    # never fires for them, so without this call the download
                    # gate would stay held through every post-load stage.  The
                    # ``finally: controller.release()`` below is a second-line
                    # backstop that releases whichever gate is held on any
                    # error path.  No-op if the importer already swapped
                    # mid-load.
                    controller.swap_to_embed()

                    apply_custom_metadata_md5(ctx.medias)
                    _tag_origins(ctx.medias, origin)
                    _apply_clipper_stage(ctx, pacer, clipper, clipper_params, chain_steps)
                    _embed_missing_stage(ctx, pacer, embedders if embedders else [embedder])
                    # Step 4 (finalize) bundles several sub-stages. Route them
                    # through a FinalizeProgress proxy so each maps into its own
                    # ordered slice of the step-4 bar instead of independently
                    # filling (and pinning at 100%) the whole slice — keeps the
                    # bar advancing and the ETA self-correcting through the
                    # serialize/disk-write window. See FinalizeProgress.
                    fin = FinalizeProgress(pacer, media_type)
                    fin.begin("cleanup")
                    _drop_none_embeddings_stage(ctx, fin)
                    # Re-lazify clips from reference (thin) parents now that
                    # embedding is done: strip their materialized bytes so the
                    # dataset stores recipes, not duplicated clip payloads.
                    _relazify_reference_clips_stage(ctx, fin)
                    fin.begin("dedup")
                    _collapse_duplicates_stage(ctx, fin)
                    _collapse_near_duplicates_stage(ctx, fin)
                    fin.begin("coverage")
                    _build_coverage_atlas_stage(ctx, fin)
                    tracker.check_cancelled()
                    # Opt-in (rides the projection opt-in): cache a signpost
                    # text per media BEFORE the registry save, so the texts —
                    # the sign pipeline's only full-corpus model cost — are
                    # pickled with the dataset and later browse / Find→Browse
                    # re-fits skip the text models entirely.
                    _maybe_signpost_texts_stage(ctx, fin, build_projection)
                    fin.begin("registry")
                    context_id, registry_entry_id = _register_and_migrate(
                        ctx, fin, task_id, origin, name, clipper, embedder, created_by, ingest_started_at
                    )
                    # Opt-in: compute + persist the 2-D Browse projection now,
                    # so the Browse canvas opens instantly instead of building
                    # UMAP lazily on first visit.  Best-effort and runs after
                    # registration: the dataset is already saved and usable, so
                    # a failure (or a cancel during the fit) leaves it intact
                    # and just defers the projection to the lazy Browse path.
                    if build_projection:
                        fin.begin("projection")
                        try:
                            _build_projection_stage(ctx, fin)
                        except Exception:
                            traceback.print_exc()
                    # Embedder warm-up is fire-and-forget so the dashboard row goes
                    # green immediately.  Text sort waits behind its own progress
                    # bar on first use if the model isn't ready yet.
                    _warmup_embedder_async(ctx.medias)
                    # Same deal for archive-member thumbnails: the importer reads
                    # no member bytes by design, so those media land with no
                    # thumbnail and every browse tile would stream a tar member
                    # and decode it on the request thread.  Warm them off the
                    # request path now that the dataset is registered and
                    # browsable; a no-op for every other import path.  Kicked on
                    # reload too (this runs for pickle loads as well), since the
                    # save above necessarily predates the pass.
                    start_archive_thumbnail_warm(ctx)

                    from vtscore.achievements_hooks import record_achievement  # noqa: PLC0415

                    record_achievement("dataset_load", str(origin.get("importer", "")))
                except Exception as exc:
                    _handle_load_failure(exc, context_id, tracker, registry_entry_id=registry_entry_id)
                finally:
                    controller.release()
        finally:
            # Pass the demo dataset id (empty for non-demo loads) so profiler
            # rows carry it and can resolve the archive size via
            # ``download_size_mb_for`` — otherwise app-recorded rows land with
            # ``dataset_id: ""`` and can't feed fit_load_weights.py (see #2614).
            # Both recorders learn the resolved embedder here rather than at
            # construction: it is only known once the medias exist (#3345).
            recorded_embedder = _recorded_embedder_name(ctx.medias, embedder)
            # writes JSONL + unbinds (no-op when off)
            profiler.finish(len(ctx.medias), dataset_id, embedder=recorded_embedder)
            # A load that failed measured an abort, not a cost: ``ok=False`` tells
            # the fitter to drop the run rather than fit a slope to it. The
            # tracker is authoritative here because every failure path funnels
            # through ``_handle_load_failure``, which stamps the error on it.
            size_mb = download_size_mb_hint
            if size_mb is None:
                size_mb = resolve_download_size_mb(dataset_id)
            timing_recorder.finish(
                n=len(ctx.medias),
                size_mb=size_mb,
                ok=not tracker.get().get("error"),
                embedder=recorded_embedder,
            )
            _park_load_terminal(tracker, len(ctx.medias))

    return _spawn_import_worker(task, load_task)


def consume_chunks_into(
    target: dict[int, dict[str, Any]],
    chunks: Iterable[dict[int, dict[str, Any]]],
) -> None:
    """Drain *chunks* into *target* with sequential IDs.

    Each chunk yielded by an importer's ``run_chunked()`` re-uses IDs
    starting at 1, so naive ``target.update(chunk)`` would overwrite
    earlier chunks.  Renumber every media to a unique ID continuing from
    whatever IDs are already present in *target*.
    """
    next_id = max(target.keys(), default=0) + 1
    for chunk in chunks:
        for media in chunk.values():
            media["id"] = next_id
            target[next_id] = media
            next_id += 1


_CHUNK_SIZE_BY_MEDIA_TYPE: dict[str, int] = {
    "text": 5000,
    "image": 500,
    "audio": 100,
    "video": 25,
    "document": 50,
}


def auto_chunk_size(media_type: str) -> int:
    """Pick a chunk size for *media_type* that bounds peak memory.

    Tuned roughly so a single in-flight chunk's raw bytes + embeddings stay
    below ~1 GB on typical inputs.  Returns a positive int.  Importers that
    do not support chunked loading silently ignore the value.
    """
    return _CHUNK_SIZE_BY_MEDIA_TYPE.get(_normalize_media_type(media_type), 100)


def _run_importer_in_background(importer, field_values: dict) -> str:
    """Start *importer*.run() in a daemon thread.

    When the importer reports ``supports_chunked``, the loader streams
    medias in via ``run_chunked`` to bound peak memory during the
    import/embedding phase.  The chunk size is auto-selected from the
    field's ``media_type`` (see :func:`auto_chunk_size`); there is no
    user-facing knob.

    Returns the task_id for progress tracking.
    """
    from vtscore.plugins.uploads import wrap_cli_file_fields  # noqa: PLC0415

    # Normalize ``field_type="file"`` values to UploadedFile.  The
    # request path supplies a FileStorage / BytesIOUploadedFile already;
    # the reload-from-origin path supplies a server path string that
    # needs CliUploadedFile wrapping so ``run()`` doesn't have to
    # branch on the input shape.
    from vtscore.state.current_user import get_current_user  # noqa: PLC0415

    field_values = wrap_cli_file_fields(importer.fields, field_values)
    created_by = get_current_user()
    origin = importer.build_origin(field_values)
    # Reference mode (server importers): store path references instead of
    # copying media bytes.  Maps onto the loader's ``thin`` flag.  Popped so
    # it isn't forwarded into the importer's field_values (run() takes ``thin``
    # as a parameter, not a field).
    reference_files = _parse_bool(field_values.pop("reference_files", None))
    clipper_name = field_values.pop("clipper", "") or ""
    clipper_params = field_values.pop("clipper_params", None)
    chain_steps = _parse_chain_field(field_values.pop("clipper_chain", None))
    # Enabled cleanup gates always run last, on the finished units, so they are
    # appended to the chain here rather than positioned by the client.
    chain_steps = append_cleaner_steps(chain_steps, field_values.pop("cleaners", None))
    # Keep clipper in field_values for importers that need it (e.g. demo
    # importer stores it in the container metadata for readiness tracking).
    field_values["clipper"] = clipper_name
    # Importers that clip themselves (``handles_own_clipping``) own the full
    # clip config: hand the params/chain back so their ``run`` clips with the
    # user's real settings, and suppress the pipeline-level clipper so the
    # media isn't clipped a second time on top of the importer's own pass.
    if getattr(importer, "handles_own_clipping", False):
        if clipper_params is not None:
            field_values["clipper_params"] = clipper_params
        if chain_steps is not None:
            field_values["clipper_chain"] = chain_steps
        pipeline_clipper, pipeline_clipper_params, pipeline_chain_steps = "", None, None
    else:
        pipeline_clipper, pipeline_clipper_params, pipeline_chain_steps = clipper_name, clipper_params, chain_steps
    embedder_name = field_values.get("embedder", "")
    embedders = _parse_embedder_list(field_values.pop("embedders", None))
    # The primary picker's choice always leads the embed order (it becomes each
    # media's recorded primary); the trio's patch/structural picks ride behind
    # it.  Defensive: include the primary even if the client omitted it.
    if embedders and embedder_name and embedder_name not in embedders:
        embedders = [embedder_name, *embedders]
    build_projection = _parse_bool(field_values.pop("build_projection", None))
    merge_near_duplicates = _parse_bool(field_values.pop("merge_near_duplicates", None))

    # Extract media_type from field_values so in-progress tasks can expose it
    # to the frontend (used for guessing the type in subsequent add dialogs).
    media_type_hint = _normalize_media_type(field_values.get("media_type", ""))

    use_chunked = getattr(importer, "supports_chunked", False)
    chunk_size = auto_chunk_size(media_type_hint) if use_chunked else 0

    def _load(target_medias):
        if use_chunked:
            consume_chunks_into(target_medias, importer.run_chunked(field_values, chunk_size, thin=reference_files))
        else:
            importer.run(field_values, target_medias, thin=reference_files)

    # For demo datasets we know the expected item count + archive size up front,
    # which lets load_step_weights pace by the measured n-aware cost model rather
    # than the static asymptote. Unknown for streaming folder imports -> None.
    demo_id, n_hint, download_size_mb_hint = _demo_load_hints(importer, field_values)

    return _run_origin_load_in_background(
        _load,
        origin,
        name=importer.resolve_display_name(field_values),
        clipper=pipeline_clipper,
        clipper_params=pipeline_clipper_params,
        chain_steps=pipeline_chain_steps,
        embedder=embedder_name,
        embedders=embedders,
        created_by=created_by,
        media_type=media_type_hint,
        build_projection=build_projection,
        merge_near_duplicates=merge_near_duplicates,
        dataset_id=demo_id,
        n_hint=n_hint,
        download_size_mb_hint=download_size_mb_hint,
    )


def _demo_load_hints(importer, field_values: dict) -> tuple[str, int | None, float | None]:
    """Demo dataset id, expected item count, and archive size for a demo load,
    else ``("", None, None)``.

    The ``n`` / size hints enable the ``n``-aware cost-model weights (see
    :func:`vtscore.datasets.stages._common.load_step_weights`); the id is also
    threaded to the load profiler so its rows carry ``dataset_id`` (and can
    resolve the archive size) rather than landing empty (see #2614). Only the
    demo importer knows these up front; folder importers stream, so they fall
    back to the static weight profile.
    """
    if getattr(importer, "name", "") != "demo":
        return "", None, None
    dataset_id = field_values.get("name", "")
    if not dataset_id:
        return "", None, None
    from vtscore.datasets.config import DEMO_DATASETS  # noqa: PLC0415
    from vtscore.datasets.demo_counts import exact_demo_count  # noqa: PLC0415

    n = exact_demo_count(dataset_id)
    info = DEMO_DATASETS.get(dataset_id) or {}
    dl = info.get("download_size_mb")
    return dataset_id, n, (float(dl) if dl is not None else None)


# ---------------------------------------------------------------------------
# Staging – import datasets to temporary pkl files for the combine flow
# ---------------------------------------------------------------------------

STAGING_DIR = DATA_DIR / "staging"

#: Task family and step count for a staging import (see vtscore.timing.tasks).
_STAGE_TASK = "dataset_stage"
_TOTAL_STAGE_STEPS = 3  # acquire, embed, serialize

#: Maps the status strings an importer emits onto ``dataset_stage``'s steps, the
#: way :data:`_STATUS_TO_STEP` does for a load. Staging folds the load's four
#: steps into three, so every pre-embed status shares the acquire slice.
#:
#: This map is what makes the ``embed`` step measure an embed. An importer that
#: embeds *inside* ``run()`` — every demo source does, and the demo importer is
#: what the combine flow stages — used to report stepless, so the tracker kept
#: step 1 and the whole embedding leg was recorded as ``acquire``. #3521 §5
#: fitted that step at ``b = 0.0136 s/item, r² 0.9995`` — an embed curve under
#: the wrong name — beside an ``embed`` at ``b = 7.2e-07``, on a sweep that
#: cleared the embeddings cache before every rep. Clearing the cache moved
#: 11–40 s of real embedding into the run and ``embed`` did not move, because
#: the boundary, not the cache, was what put it there (#3593).
_STAGE_STATUS_TO_STEP = {
    "downloading": 1,
    "extracting": 1,
    "loading": 1,
    "converting": 1,
    "embedding": 2,
}


def _make_staging_progress(controller: _LoadGateController, tracker):
    """Build the importer-side progress callback for a staging run.

    Mirrors :func:`_make_stepped_progress`: it stamps the step each status
    belongs to — so the timing recorder labels a duration with the phase that
    actually ran — and swaps the download gate for the embed gate on the first
    ``"embedding"``, so a queued import can start fetching while this one holds
    only the embed slot. Staging used to swap only after ``run()`` returned,
    which meant a demo staging did its embedding under the *download* gate.

    Two deliberate differences from the load pipeline's version, both because
    staging has no :class:`AdaptiveLoadPacer` between the importer and the
    tracker:

    - a status the map does not know leaves ``step`` **unset** rather than
      passing ``None``, so the tracker keeps the step it was last told instead
      of nulling the whole-job fraction for that update;
    - the step never moves backwards. A demo's clipper reports its clip
      embedding under a plain ``"loading"`` status, which would otherwise walk
      the bar back to acquire after the embed slice had already started.
    """

    def stepped(status: str, message: str = "", current: int = 0, total: int = 0, **kwargs) -> None:
        # An importer signalling *its* completion is not the staging job's:
        # serialization still has to run, and the terminal update is the one at
        # the bottom of ``stage_task``. A failure riding along is another
        # matter and is forwarded. (``load_demo_dataset`` ends with exactly such
        # an ``"idle"``, which used to park the whole staging task at idle
        # mid-run.)
        if status == "idle" and "error" not in kwargs:
            return
        if status == "embedding" and controller.held != "embed":
            controller.swap_to_embed()
        if "step" not in kwargs:
            step = _STAGE_STATUS_TO_STEP.get(status)
            if step is not None and step >= (tracker.get().get("step") or 0):
                kwargs["step"] = step
        kwargs.setdefault("total_steps", _TOTAL_STAGE_STEPS)
        tracker.update(status, message, current, total, **kwargs)

    return stepped


def _stage_importer_in_background(importer, field_values: dict, label: str = "") -> str:
    """Run *importer*.run() in a daemon thread, saving the result to a staging pkl.

    Unlike ``_run_importer_in_background``, this does **not** modify the global
    ``medias`` dict and does **not** register a dataset.  Instead it writes a
    temporary ``.pkl`` file to :data:`STAGING_DIR` and publishes the terminal
    ``staging_result`` on this operation's own per-task progress tracker.

    Each call gets a dedicated :class:`ProgressTracker` (via ``loading_tasks``),
    keyed by the returned ``task_id``, mirroring
    :func:`_run_origin_load_in_background`, so two concurrent stagings never
    interleave one channel and their terminal ``staging_result``s cannot
    collide.  The shared setup/teardown around the body — tracker, timing
    recorder, user replay, worker registration — is
    :func:`_start_import_task` / :func:`_spawn_import_worker`, the same harness
    the load pipeline runs on.

    Staging is gated exactly like a load: the importer runs under the download
    gate and the embed under the embed gate.  It has the same appetite for
    bandwidth, RAM, and GPU as a regular import, and the combine flow stages
    several datasets at once — so without this, N stagings ran fully in
    parallel with each other *and* with gated loads, defeating the
    ``max_concurrent_dataset_downloads`` / ``max_concurrent_dataset_embeddings``
    limits whose whole purpose is bounding that pressure.

    Returns the ``task_id`` a caller can poll (via the ``loading-tasks`` SSE
    channel) for progress and the final ``staging_result``.
    """
    from vtscore.plugins.uploads import wrap_cli_file_fields  # noqa: PLC0415

    field_values = wrap_cli_file_fields(importer.fields, field_values)

    staged_media_type = _normalize_media_type(field_values.get("media_type", ""))
    staged_embedder = field_values.get("embedder", "") or ""
    # Staging reports the same step structure every other long-running family
    # does, which is what earns it a whole-job bar and an ETA — and what lets the
    # timing recorder label each measured duration with the phase it belongs to.
    # The boundaries stamped below mark the steps this function drives; the
    # importer's own progress calls are mapped onto the same three steps by
    # ``_make_staging_progress``, because an importer that embeds inside
    # ``run()`` is embedding whatever step number the last stamp left behind
    # (#3593).
    task = _start_import_task(
        prefix="_staging_",
        family=_STAGE_TASK,
        display_name=label or importer.resolve_display_name(field_values),
        total_steps=_TOTAL_STAGE_STEPS,
        media_type=staged_media_type,
        embedder=staged_embedder,
        weights=step_weights(_STAGE_TASK, media_type=staged_media_type, embedder=staged_embedder),
        extra_fields={"staging_result": None},
    )
    tracker, timing_recorder = task.tracker, task.recorder
    tracker.update("loading", "Preparing dataset…", 0, 0, step=1, total_steps=_TOTAL_STAGE_STEPS)

    def stage_task():
        controller = _LoadGateController(tracker, task.total_steps)
        # Route the importer's own progress calls (and embedding progress)
        # into this task's tracker instead of the global singleton, mapped onto
        # this task's steps on the way.
        set_thread_progress(_make_staging_progress(controller, tracker))
        # A staging import of a demo reads the same embeddings pkl a full import
        # writes, so it forks on the same cache and must record which branch it
        # took.
        timing_recorder.bind_thread()
        try:
            controller.acquire_download()
            temp_medias: dict = {}
            importer.run(field_values, temp_medias)
            apply_custom_metadata_md5(temp_medias)
            # Backstop for the handoff ``_make_staging_progress`` normally makes
            # on the importer's first ``"embedding"`` status: an importer that
            # embeds nothing itself (every non-demo source) never fires one, and
            # would otherwise hold the download gate through the embed below.
            # No-op when the swap already happened mid-run.
            controller.swap_to_embed()
            tracker.update("embedding", "Embedding…", 0, 0, step=2, total_steps=_TOTAL_STAGE_STEPS)
            embed_missing(temp_medias, field_values.get("embedder", "") or "", on_progress=tracker.update)
            from vtscore.embedding.media_vectors import media_embedding  # noqa: PLC0415

            temp_medias = {mid: m for mid, m in temp_medias.items() if media_embedding(m) is not None}

            if not temp_medias:
                tracker.update("idle", "", 0, 0, error="Import produced no medias.")
                return

            first = next(iter(temp_medias.values()))
            media_type = first.get("media_type", "audio")
            count = len(temp_medias)
            # Same late resolution as the import path: staging that named no
            # embedder still ran the media type's default (#3345).
            timing_recorder.set_scale(
                n=count,
                embedder=_recorded_embedder_name(temp_medias, staged_embedder),
            )
            name = label or importer.resolve_display_name(field_values)

            tracker.update("loading", "Writing staged file…", 0, 0, step=3, total_steps=_TOTAL_STAGE_STEPS)
            data_bytes = export_dataset_to_file(temp_medias)
            del temp_medias
            gc.collect()

            STAGING_DIR.mkdir(parents=True, exist_ok=True)
            staging_path = STAGING_DIR / f"stage_{uuid4().hex}.pkl"
            staging_path.write_bytes(data_bytes)
            del data_bytes
            gc.collect()

            tracker.update(
                "idle",
                f"Staged: {name} ({count} medias)",
                100,
                100,
                step=_TOTAL_STAGE_STEPS,
                total_steps=_TOTAL_STAGE_STEPS,
                staging_result={"path": str(staging_path), "name": name, "count": count, "media_type": media_type},
            )
        except Exception as exc:
            gc.collect()
            tracker.update("idle", "", 0, 0, error=_failure_message(exc, "Unknown error during staging"))
        finally:
            controller.release()
            # Every branch above parks the tracker at "idle", setting
            # ``error`` when it failed — which is what says whether these
            # phase timings describe a staging run worth fitting.
            timing_recorder.finish(ok=not tracker.get().get("error"))

    return _spawn_import_worker(task, stage_task)
