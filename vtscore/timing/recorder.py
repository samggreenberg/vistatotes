"""Env-gated recorder that measures how long each step of a task really took.

Arm it by pointing ``VTSEARCH_TIMING_RECORD`` at a JSONL path. Every task that
wraps its work in :func:`record_task` then appends one row per step::

    {"task": "text_sort", "device": "cuda", "cuml": true, "media_type": "image",
     "embedder": "siglip", "n": 12403, "size_mb": 0.0, "step": "score",
     "seconds": 1.83, "ok": true, "cold_model": false}

``scripts/profiling/tune_timing_profile.py`` fits those rows into the profile
JSON that :mod:`vtscore.timing.profile` reads back. Because the recorder lives
behind an env var and touches nothing when unset, an admin has two ways to
gather data, and both produce the same file:

- **Drive it.** Run the tuning script, which exercises each task family against
  exemplar datasets with the recorder armed.
- **Watch it.** Set ``VTSEARCH_TIMING_RECORD`` on the real server and let real
  users generate the timings, then fit the accumulated JSONL. This measures the
  production mix directly — the datasets people actually load, at the sizes they
  actually are — which no synthetic sweep can reproduce.

When disarmed the cost is one ``os.environ`` lookup per task and a couple of
no-op method calls; there is no tracker subscription and no file handle.

**Branches.** A step's cost often forks on a cache the row does not otherwise
record, and a fitter that cannot see the fork averages the two. Two markers
carry the fork:

- ``cold_model`` is a property of the **run**: whether it was the first in the
  process to need its ``(media_type, embedder)`` encoder (#3520, below).
- ``branch`` is a property of one **step**: which path that step took, named by
  the code that chose it via :func:`note_branch`. The vocabulary lives in
  :mod:`vtscore.timing.tasks` (``CHEAP_BRANCHES`` / ``DEAR_BRANCHES``).

The second exists because the first cannot see the caches that are not the
encoder, and those are on **disk**, so they outlive the process a cold/warm
ledger is scoped to. #3345 measured two: a ``dataset_open`` whose coverage step
restored the atlas cached in its pickle on all 16 opens and never once rebuilt
it, and a ``dataset_load``/``dataset_stage`` embed that read the demo
embeddings pkl instead of embedding. (The ``dataset_stage`` half of that second
one turned out to be a misdiagnosis: its embed step read 0.000–0.002 s across
all four image tiers because the importer's embedding was being recorded under
``acquire``, not because a cache was hit. #3521 §5 cleared the cache and the
step still read zero; #3593 moved the boundary. The branch marker is what made
that legible in one look.)

**Cold vs warm.** Every row also carries ``cold_model``: whether this run was
the first in the process to need its ``(media_type, embedder)`` encoder, and so
the one that paid to download and instantiate it. Without that marker the rows
for a once-per-process cost are unfittable — the model load of a text sort
measures 15 s once and 0 s on the next 47, and a fitter that cannot tell the two
populations apart medians them into "free" (#3520). Only tasks whose
:class:`~vtscore.timing.tasks.TaskSpec` declares ``loads_encoder`` take part, so
a task that never touches an encoder neither claims a key nor carries the field.

The dataset-load pipeline *also* carries an older, richer recorder
(:mod:`vtscore.datasets.stages._load_profiler`) that additionally distinguishes
cold from cached downloads and splits finalize into its sub-slots. The two run
side by side on the same load: they answer different questions, are armed by
different env vars, and write to different files. The tuning script's fitter
reads both row shapes, so a pre-existing dataset-load calibration sweep still
folds into a profile — but an admin who armed only ``VTSEARCH_TIMING_RECORD``
gets import rows too, which for a while they did not.

Kept in ``vtscore`` (no Flask) so a plain CLI or library run records the same
way a served request does.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Optional

from vtscore.timing.profile import cuml_active, resolve_device_name
from vtscore.timing.tasks import task_spec

logger = logging.getLogger(__name__)

#: Environment variable naming the JSONL sink. Unset means "record nothing".
RECORD_ENV_VAR = "VTSEARCH_TIMING_RECORD"

#: ``(media_type, embedder)`` pairs whose encoder this process has already
#: needed. The first run for a pair pays the cold load — downloading the weights
#: and instantiating the model — and every later one finds it resident, which is
#: why a warm text sort skips its ``load_model`` step entirely.
#:
#: Self-detecting residency is more robust than a caller-supplied flag, and lets
#: one driven sweep measure both branches in a single process. The key pairs the
#: media type in rather than keying on the embedder alone, because the name can
#: legitimately arrive blank: two blanks from different media types would
#: otherwise collide, and the second one's genuinely cold load would be written
#: ``cold_model: false`` — the mislabel #3345 measured on the other recorder.
_seen_models: set[tuple[str, str]] = set()
_seen_lock = threading.Lock()


#: Recorder bound to the current worker thread, so code far below the task's
#: entry point can name the branch a step took without threading a recorder
#: argument through every stage. Mirrors ``_load_profiler``'s ``_active``, which
#: exists for the same reason and is bound at the same place.
_active = threading.local()


def _active_recorder() -> Optional["TaskTimingRecorder"]:
    return getattr(_active, "recorder", None)


def note_branch(step: str, branch: str) -> None:
    """Record which path *step* took on the run recording this thread.

    A no-op when nothing is recording, which is the usual case — the call sites
    are ordinary product code (``load_demo_dataset``'s cache hit, the dataset
    open's atlas restore) and must not care whether a sweep is running.

    *branch* is one of the names in :data:`vtscore.timing.tasks.CHEAP_BRANCHES`
    or :data:`~vtscore.timing.tasks.DEAR_BRANCHES`. The vocabulary is shared
    rather than free-text because the fitter reads it: an unrecognised name is
    recorded faithfully and then treated as unclassifiable, which is the safe
    direction — it neither licenses a fit nor silently blocks one.
    """
    rec = _active_recorder()
    if rec is not None:
        rec.mark_branch(step, branch)


def note_no_encoder_load() -> None:
    """Declare that the run recording this thread instantiated no encoder.

    A no-op when nothing is recording. See
    :meth:`TaskTimingRecorder.disclaim_encoder` for why a run that skipped the
    load must not claim the residency key.
    """
    rec = _active_recorder()
    if rec is not None:
        rec.disclaim_encoder()


def reset_seen_models_for_tests() -> None:
    """Forget every encoder this process has recorded a load for.

    Process-global, so one test's recorded run would otherwise decide whether
    the next test's run is written cold or warm.
    """
    with _seen_lock:
        _seen_models.clear()


def recording_enabled() -> bool:
    """True when the ``VTSEARCH_TIMING_RECORD`` sink is armed."""
    return bool(os.environ.get(RECORD_ENV_VAR, "").strip())


class TaskTimingRecorder:
    """Measures per-step durations for one run of one task.

    Subscribes to the task's :class:`~vtscore.concurrency.progress.ProgressTracker`
    and stamps the first moment each step becomes active; a step's duration is
    the gap to the next step's start, and the last step runs to :meth:`finish`.
    That gives the same boundaries the user sees on the bar, which is the point —
    the model being fit predicts *displayed* phases, not internal function calls.

    Use it as a context manager; ``n`` is usually only known partway through, so
    set it whenever you learn it::

        with record_task(tracker, "text_sort", media_type=mt) as rec:
            ...
            rec.set_scale(n=len(medias))
    """

    #: Statuses that mean "this task is over". A task whose tracker is a
    #: long-lived singleton (``sort_progress``, ``find_progress``) ends by
    #: setting one of these, and every early-abort path does the same — which
    #: makes them a far more reliable end-of-task signal than a ``finally``
    #: bolted onto a route handler with a dozen ``abort()`` exits.
    TERMINAL_STATUSES = frozenset({"idle", "error", "cancelled"})

    def __init__(
        self,
        tracker: Any,
        task: str,
        *,
        media_type: str = "",
        embedder: str = "",
        status_phases: Optional[dict[str, str]] = None,
        auto_finish: bool = False,
        only_phases: Optional[tuple[str, ...]] = None,
    ) -> None:
        self._tracker = tracker
        self._task = task
        self._spec = task_spec(task)
        # A run that performs only part of the task's step structure — the
        # on-demand coverage-atlas rebuild is a ``dataset_open``'s second step
        # and nothing else. Without this the emit below would write a 0.0 row
        # for every step the run never had the chance to take, and those zeros
        # are indistinguishable from measurements of a step that really is free.
        self._only_phases = tuple(only_phases) if only_phases else ()
        self._media_type = media_type or ""
        self._embedder = embedder or ""
        # Maps a status string onto a phase name, for tasks where one tracker
        # step covers several cost phases and only the status tells them apart
        # (the dataset load's step 1 is both "downloading" and "extracting").
        self._status_phases = dict(status_phases or {})
        self._auto_finish = auto_finish
        self._path = os.environ.get(RECORD_ENV_VAR, "").strip()
        self._lock = threading.Lock()
        self._phase_start: dict[str, float] = {}
        self._phase_order: list[str] = []
        self._branches: dict[str, str] = {}
        self._loaded_encoder = True
        self._last_seen: Any = None
        self._n = 0.0
        self._size_mb = 0.0
        self._ok = True
        self._subscribed = False
        self._bound = False

    # -- lifecycle ----------------------------------------------------------
    def __enter__(self) -> "TaskTimingRecorder":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        # A run that raised measured a failure path, not a cost: mark it so the
        # fitter drops it rather than fitting a slope to an exception's timing.
        self.finish(ok=exc_type is None)
        return False

    def start(self) -> None:
        """Subscribe to the tracker. Idempotent."""
        if self._subscribed:
            return
        try:
            self._tracker.subscribe(self._on_update)
        except Exception:  # pragma: no cover - a tracker without subscribe()
            logger.debug("timing recorder: tracker for %s has no subscribe()", self._task)
            return
        self._subscribed = True

    def bind_thread(self) -> None:
        """Make this recorder the target of :func:`note_branch` on this thread.

        Called from inside the worker, not from :meth:`start`: a task is armed
        on the request thread and then runs on a background one, and it is the
        background thread whose stages make the branch decisions.
        """
        _active.recorder = self
        self._bound = True

    def _unbind_thread(self) -> None:
        if getattr(_active, "recorder", None) is self:
            _active.recorder = None
        self._bound = False

    def mark_branch(self, step: str, branch: str) -> None:
        """Name the path *step* took. Last call wins; safe from any thread."""
        if not step or not branch:
            return
        with self._lock:
            self._branches[str(step)] = str(branch)

    def disclaim_encoder(self) -> None:
        """Declare that this run never instantiated an encoder.

        A task marked ``loads_encoder`` normally claims its ``(media_type,
        embedder)`` key in the residency ledger when it emits. A run that
        satisfied itself from a cache did no such thing — a demo import that
        reads the embeddings pkl embeds nothing and so never touches the model —
        and letting it claim the key would stamp the *next* run warm when that
        run is the one that genuinely pays the load. That is exactly the
        mislabel #3345 measured on the other recorder, arriving by a different
        route: there two blank embedder names collided, here a cache hit stands
        in front of a real load.
        """
        with self._lock:
            self._loaded_encoder = False

    def set_scale(
        self,
        n: Optional[float] = None,
        size_mb: Optional[float] = None,
        embedder: Optional[str] = None,
    ) -> None:
        """Record the job's scale variables once they are known.

        Safe to call repeatedly and from any thread; the last value wins.

        *embedder* is here for the same reason *n* is: a task can start without
        knowing which encoder it will use. An import whose caller named no
        embedder still runs one — the media-type default, resolved deep in the
        embed stage — and recording the blank instead files every such run under
        the media rollup, leaving the exact ``(device, media, embedder)`` cell
        permanently empty (#3345). An empty string is ignored rather than
        written, so learning nothing cannot erase what the caller did know.
        """
        with self._lock:
            if n is not None:
                self._n = float(n)
            if size_mb is not None:
                self._size_mb = float(size_mb)
            if embedder:
                self._embedder = str(embedder)

    def finish(
        self,
        n: Optional[float] = None,
        size_mb: Optional[float] = None,
        ok: bool = True,
        embedder: Optional[str] = None,
    ) -> None:
        """Unsubscribe and append this run's rows. Idempotent."""
        self.set_scale(n, size_mb, embedder)
        with self._lock:
            self._ok = self._ok and ok
        if self._subscribed:
            try:
                self._tracker.unsubscribe(self._on_update)
            except Exception:  # pragma: no cover
                pass
            self._subscribed = False
            self._emit()

    # -- capture ------------------------------------------------------------
    def _on_update(self, snapshot: dict[str, Any]) -> None:
        """Tracker subscriber: stamp the first time each phase becomes active."""
        step = snapshot.get("step")
        status = snapshot.get("status")
        seen = (step, status)
        if seen == self._last_seen:
            return
        self._last_seen = seen
        if self._auto_finish and status in self.TERMINAL_STATUSES:
            # Unsubscribing from inside a callback is safe: the tracker notifies
            # over a copy of its subscriber list.
            self.finish(ok=status == "idle" and not snapshot.get("error"))
            return
        phase = self._resolve_phase(step, status)
        if phase is None:
            return
        now = time.monotonic()
        with self._lock:
            if phase not in self._phase_start:
                self._phase_start[phase] = now
                self._phase_order.append(phase)

    def _resolve_phase(self, step: Any, status: Any) -> Optional[str]:
        """Map a progress snapshot onto one of the task's declared phase names."""
        mapped = self._status_phases.get(str(status)) if status else None
        if mapped:
            return mapped
        if self._spec is None or not isinstance(step, int):
            return None
        for name, index in zip(self._spec.steps, self._spec.step_index):
            if index == step:
                return name
        return None

    # -- emit ---------------------------------------------------------------
    def _claim_encoder(self) -> Optional[bool]:
        """Whether this run paid the cold encoder load, claiming the key if so.

        ``None`` for a task that loads no encoder: it neither carries the field
        nor claims a key, because the ledger is process-wide and a task that
        never instantiates a model would otherwise mark one resident on behalf
        of the task that actually loads it.

        Read where the rows are written rather than at construction, for two
        reasons: a run that learns its embedder partway through is then filed
        under the encoder it really used, and a run that recorded nothing does
        not claim a key on behalf of the next one.
        """
        if self._spec is None or not self._spec.loads_encoder:
            return None
        if not self._loaded_encoder:
            # Ran, but never instantiated a model (see ``disclaim_encoder``).
            # Neither cold nor warm: the field would be a claim about a load
            # that did not happen, and claiming the key would mislabel the run
            # that does happen next.
            return None
        key = (self._media_type, self._embedder)
        with _seen_lock:
            cold = key not in _seen_models
            _seen_models.add(key)
        return cold

    def _emit(self) -> None:
        end = time.monotonic()
        if self._bound:
            self._unbind_thread()
        if not self._path:
            return
        with self._lock:
            order = list(self._phase_order)
            starts = dict(self._phase_start)
            branches = dict(self._branches)
            n, size_mb, ok = self._n, self._size_mb, self._ok
        if not order:
            return
        # "Complete" means the job reached its *last* declared step, not that it
        # visited every one. Steps legitimately skip: a warm text sort never
        # enters ``load_model`` because the encoder is already resident, and a
        # cache-backed dataset re-add never downloads. Those runs are perfectly
        # good cost samples. What is *not* a sample is a run that gave up
        # partway, because its final recorded step's duration runs to whenever
        # the task bailed rather than to a real phase boundary.
        declared = self._only_phases or (self._spec.steps if self._spec is not None else tuple(order))
        complete = bool(declared) and declared[-1] in starts
        base = {
            "task": self._task,
            "device": resolve_device_name(),
            "cuml": cuml_active(),
            "media_type": self._media_type,
            "embedder": self._embedder,
            "n": n,
            "size_mb": size_mb,
            "ok": ok,
            "complete": complete,
        }
        cold = self._claim_encoder()
        if cold is not None:
            base["cold_model"] = cold
        durations = {
            phase: max(0.0, (starts[order[i + 1]] if i + 1 < len(order) else end) - starts[phase])
            for i, phase in enumerate(order)
        }
        # A skipped step on an otherwise-complete run really did cost nothing, so
        # it is recorded as a zero rather than left out: the fit should learn
        # that this deployment's warm model loads are free, not silently drop
        # every warm run's evidence and over-budget the phase forever.
        rows = [
            {
                **base,
                "step": phase,
                "seconds": round(durations.get(phase, 0.0), 4),
                **({"branch": branches[phase]} if phase in branches else {}),
            }
            for phase in (declared if complete else order)
        ]
        try:
            with open(self._path, "a", encoding="utf-8") as fh:
                fh.write("".join(json.dumps(r) + "\n" for r in rows))
        except OSError as exc:
            logger.warning("timing recorder: cannot append to %s (%s)", self._path, exc)


class _NullRecorder:
    """No-op stand-in returned when recording is off.

    Exposes the same surface so call sites never branch on ``None`` — the
    disarmed path is a few method calls that do nothing.
    """

    def __enter__(self) -> "_NullRecorder":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def start(self) -> None:
        pass

    def bind_thread(self) -> None:
        pass

    def mark_branch(self, step: str, branch: str) -> None:
        pass

    def disclaim_encoder(self) -> None:
        pass

    def set_scale(
        self,
        n: Optional[float] = None,
        size_mb: Optional[float] = None,
        embedder: Optional[str] = None,
    ) -> None:
        pass

    def finish(
        self,
        n: Optional[float] = None,
        size_mb: Optional[float] = None,
        ok: bool = True,
        embedder: Optional[str] = None,
    ) -> None:
        pass


_NULL_RECORDER = _NullRecorder()


def record_task(
    tracker: Any,
    task: str,
    *,
    media_type: str = "",
    embedder: str = "",
    status_phases: Optional[dict[str, str]] = None,
    auto_finish: bool = False,
    only_phases: Optional[tuple[str, ...]] = None,
):
    """Return a recorder for one run of *task*, or a no-op when disarmed.

    Always returns an object supporting ``with``/``set_scale``/``finish``, so
    callers use it unconditionally and pay nothing when the env var is unset.

    Set *auto_finish* when the task reports through a long-lived singleton
    tracker that it parks at ``"idle"`` on the way out (every exit path,
    including aborts). The recorder then closes itself on that signal, so a
    route handler with a dozen early ``abort()``s needs no ``finally``.

    Set *only_phases* for an entry point that runs part of a task's step
    structure rather than all of it — the on-demand coverage-atlas rebuild is
    a ``dataset_open``'s coverage step arriving on its own. The run is then
    recorded against only those phases instead of writing a zero for every step
    it was never in a position to take.
    """
    if not recording_enabled():
        return _NULL_RECORDER
    return TaskTimingRecorder(
        tracker,
        task,
        media_type=media_type,
        embedder=embedder,
        status_phases=status_phases,
        auto_finish=auto_finish,
        only_phases=only_phases,
    )
