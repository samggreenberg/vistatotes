"""Progress tracking for long-running operations."""

import inspect
import math
import time
import threading
from typing import Any, Callable, Optional

_UNSET = object()  # sentinel for "caller did not provide this argument"


# ---------------------------------------------------------------------------
# ETA humility
# ---------------------------------------------------------------------------
# A remaining-time estimate is an extrapolation from a rate that is still
# changing. Reported to the second it is not just wrong, it is *visibly* wrong:
# "9 min left … 10 min … 9.5 min … 11 min" reads as a system that has no idea,
# because a number that precise invites you to check it. The same underlying
# estimate reported as "about 10 min" is right the whole time.
#
# So the tracker never publishes its raw estimate. It publishes the nearest rung
# of a coarse ladder, and it *stays* on that rung until the estimate has moved
# decisively past a neighbour. Two separate mechanisms, both needed:
#
#   - The **ladder** removes false precision. Rungs step by roughly 1.5×, and
#     every rung is a round number in the unit it will be displayed in (45 s,
#     7.5 min, 1.5 hr), so no renderer has to round again and re-introduce the
#     jitter this is here to remove.
#   - The **hysteresis** removes the flip-flop. Snapping alone still oscillates
#     when the estimate sits near a boundary, which is exactly where a
#     converging estimate spends its time. A rung is only abandoned once the raw
#     value clears the boundary to its neighbour by :data:`_ETA_HYSTERESIS`.
#
# The estimate can still *rise*: a job that genuinely slowed down should say so,
# and pinning the display to a never-increasing value would just move the lie
# from the number to its trend. What it can't do any more is twitch.
#
# The smoothed internal estimate stays raw — only the published value is
# snapped. Feeding a quantized value back into the EMA would let the ladder
# capture the estimate and stop it converging at all.
# fmt: off
_ETA_LADDER: tuple[float, ...] = (
    10, 15, 20, 30, 45,                                  # 10 sec – 45 sec
    60, 90, 120, 180, 300, 450, 600, 900, 1200, 1800, 2700,   # 1 min – 45 min
    3600, 5400, 7200, 10800, 14400, 21600, 28800, 43200, 86400,  # 1 hr – 24 hr
)
# fmt: on

#: How far past the boundary between two rungs the raw estimate must travel
#: before the displayed rung changes. 0.15 means a 15% overshoot: enough that
#: ordinary convergence noise never crosses it, small enough that a real
#: slowdown is reported within a couple of updates.
_ETA_HYSTERESIS = 0.15


def _nearest_rung(seconds: float) -> float:
    """Return the ladder rung closest to *seconds* in ratio terms.

    Ratio rather than absolute distance, because the ladder is geometric: 400 s
    is closer to 300 than to 450 on an absolute scale, but proportionally it sits
    almost exactly between them, and proportional error is what a reader of an
    ETA actually perceives.
    """
    if seconds <= _ETA_LADDER[0]:
        return _ETA_LADDER[0]
    if seconds >= _ETA_LADDER[-1]:
        return _ETA_LADDER[-1]
    return min(_ETA_LADDER, key=lambda rung: abs(math.log(seconds / rung)))


def _adjacent_rung(rung: float, upward: bool) -> float:
    """The next rung above (or below) *rung*, or *rung* itself at the ends."""
    index = _ETA_LADDER.index(rung)
    neighbour = index + 1 if upward else index - 1
    if 0 <= neighbour < len(_ETA_LADDER):
        return _ETA_LADDER[neighbour]
    return rung


class CancelledError(Exception):
    """Raised when an operation is cancelled via :meth:`ProgressTracker.cancel`."""


class ProgressTracker:
    """Thread-safe progress tracker for long-running operations.

    Each instance manages its own lock and data dict. The *extra_fields*
    parameter lets callers declare additional keys (e.g. ``"error"``,
    ``"staging_result"``) that are tracked alongside the base fields.

    A :class:`threading.Event` is used for cooperative cancellation: call
    :meth:`cancel` to set the flag and :meth:`check_cancelled` from inside
    the background thread to raise :class:`CancelledError` when the flag is
    set.

    Subscribers (registered via :meth:`subscribe`) are invoked with a
    snapshot of the data dict on every :meth:`update`. The SSE event
    endpoint uses this to push progress to connected clients without
    polling.

    Args:
        extra_fields: Mapping of extra field names to their default values.
            These fields can be set via keyword arguments in :meth:`update`
            and are returned by :meth:`get`.
    """

    #: Minimum elapsed time (seconds) before an ETA is computed. Below this we
    #: don't have enough samples to extrapolate reliably and the number jitters
    #: wildly, so the snapshot's ``eta_seconds`` stays ``None``.
    _ETA_MIN_ELAPSED = 5.0

    #: Smoothing factor for the EMA over the raw ETA. ``0.3`` weights the new
    #: sample lightly enough to dampen noise while still tracking real slowdowns.
    _ETA_SMOOTHING_ALPHA = 0.3

    def __init__(self, extra_fields: Optional[dict[str, Any]] = None) -> None:
        self._lock = threading.Lock()
        self._extra_defaults = dict(extra_fields) if extra_fields else {}
        self._cancel_event = threading.Event()
        self._data: dict[str, Any] = {
            "status": "idle",
            "message": "",
            "current": 0,
            "total": 0,
            **{k: v for k, v in self._extra_defaults.items()},
        }
        self._subscribers: list[Callable[[dict[str, Any]], None]] = []
        self._subscribers_lock = threading.Lock()
        self._phase_key: tuple[str, int] | None = None
        self._phase_start: float | None = None
        self._phase_current_start: int = 0
        self._smoothed_eta: float | None = None
        # Ladder rung currently being published as ``eta_seconds``. Held across
        # updates so the displayed estimate is sticky; see :meth:`_humble_eta`
        # and the "ETA humility" notes at the top of this module.
        self._eta_rung: float | None = None
        # Overall (whole-job) progress state, used when the caller reports a
        # ``step``/``total_steps`` structure.  See :meth:`_compute_overall`.
        self._overall_start_time: float | None = None
        self._overall_start_frac: float = 0.0
        self._overall_max: float = 0.0
        self._overall_smoothed_eta: float | None = None
        self._overall_last_step: int | None = None
        self._overall_total_steps: int | None = None
        # Optional per-step weights (one per step) reflecting each phase's
        # typical share of total wall-clock time. ``None`` => equal weight.
        self._step_weights: list[float] | None = None

    def _compute_eta(self, status: str, current: int, total: int) -> Optional[float]:
        """Compute the smoothed ETA in seconds for the current bar.

        Resets the phase clock when ``status`` changes, ``total`` changes, or
        ``current`` resets backwards (a new bar is starting). Returns ``None``
        until at least :data:`_ETA_MIN_ELAPSED` seconds of work have elapsed
        with a known total and ``current > 0``.
        """
        now = time.monotonic()
        phase_key = (status, total)
        if self._phase_key != phase_key or self._phase_start is None or current < self._phase_current_start:
            self._phase_key = phase_key
            self._phase_start = now
            self._phase_current_start = current
            self._smoothed_eta = None
            return None

        if total <= 0 or current <= 0 or current >= total:
            return None
        elapsed = now - self._phase_start
        if elapsed < self._ETA_MIN_ELAPSED:
            return None
        completed = current - self._phase_current_start
        if completed <= 0:
            return None
        raw_eta = (elapsed / completed) * (total - current)
        if self._smoothed_eta is None:
            self._smoothed_eta = raw_eta
        else:
            alpha = self._ETA_SMOOTHING_ALPHA
            self._smoothed_eta = alpha * raw_eta + (1.0 - alpha) * self._smoothed_eta
        return self._smoothed_eta

    def _humble_eta(self, raw: Optional[float]) -> Optional[float]:
        """Snap *raw* onto the coarse ETA ladder, sticking to the current rung.

        This is the only place a remaining-time estimate becomes user-visible,
        and it is deliberately the last step: everything upstream — the EMA, the
        per-phase rate windows, the pacer's re-weighting — keeps working at full
        precision, and only the published number is coarsened.

        Returns ``None`` (and forgets the held rung) whenever there is no
        estimate to show, so the next job starts from a clean slate rather than
        inheriting the last one's rung.
        """
        if raw is None or not math.isfinite(raw) or raw <= 0:
            self._eta_rung = None
            return None

        target = _nearest_rung(raw)
        current = self._eta_rung
        if current is None or current == target:
            self._eta_rung = target
            return target

        # Leaving a rung costs an overshoot. The boundary between two rungs is
        # their geometric mean (the ladder is geometric); the estimate must
        # clear it by ``_ETA_HYSTERESIS`` in the direction of travel. A large
        # move jumps straight to ``target`` rather than walking rung by rung, so
        # a genuinely mistaken estimate is corrected at once while a wobbling
        # one is not.
        upward = target > current
        boundary = math.sqrt(current * _adjacent_rung(current, upward))
        if upward and raw < boundary * (1.0 + _ETA_HYSTERESIS):
            return current
        if not upward and raw > boundary / (1.0 + _ETA_HYSTERESIS):
            return current
        self._eta_rung = target
        return target

    def _overall_raw_fraction(self, within: float, s: int, total_steps: int) -> float:
        """Map a within-step fraction to a whole-job fraction in ``[0, 1]``.

        Uses per-step weights when :meth:`set_step_weights` has supplied a list
        matching ``total_steps`` (so phases known to dominate wall-clock time —
        embedding, MLP training — get a proportionally larger slice of the
        bar), otherwise falls back to equal weight per step::

            weighted: (Σ weights[:s-1] + weights[s-1] * within) / Σ weights
            equal:    ((s - 1) + within) / total_steps

        The weights only shape how the bar *paces*; the overall ETA is derived
        from the actual elapsed-vs-fraction rate, so it self-corrects no matter
        how rough the weights are.
        """
        weights = self._step_weights
        if weights and len(weights) == total_steps:
            total_w = sum(weights)
            if total_w > 0:
                completed = sum(weights[: s - 1])
                return min(max((completed + weights[s - 1] * within) / total_w, 0.0), 1.0)
        return min(max(((s - 1) + within) / total_steps, 0.0), 1.0)

    def set_step_weights(self, weights: Optional[list[float]]) -> None:
        """Declare per-step weights for the whole-job ``overall`` fraction.

        *weights* must hold one non-negative value per step (its length should
        equal the job's ``total_steps``); a length mismatch is ignored and the
        bar falls back to equal weighting. Pass ``None`` to clear. Set this once
        at the start of a job (before the first :meth:`update`).
        """
        with self._lock:
            self._step_weights = [float(w) for w in weights] if weights else None

    def _compute_overall(
        self, current: int, total: int, step: Any, total_steps: Any, within_override: Optional[float] = None
    ) -> tuple[Optional[float], Optional[float], Optional[float]]:
        """Compute the whole-job completion fraction and a true overall ETA.

        When a caller declares a ``step``/``total_steps`` structure, the bar
        should advance once across the *entire* job instead of resetting at
        every phase. The within-step ``current``/``total`` fraction is mapped
        into the job's overall span by :meth:`_overall_raw_fraction` (weighted
        per step when weights were supplied, else equal weight).

        Returns ``(overall, overall_step_end, eta_seconds)``.
        ``overall_step_end`` is the whole-job fraction at which the *current*
        step's slice ends — the bar's guaranteed position once this step
        completes. A count-less step parks ``overall`` at its slice's floor, so
        the pair bounds the job's true position (somewhere in
        ``[overall, overall_step_end]``) and the frontend can render that span
        as a bounded indeterminate zone instead of a bare parked fill.

        All three are ``None`` when no step structure is present (the caller
        falls back to the per-phase
        :meth:`_compute_eta`). The fraction is clamped to be monotonic
        non-decreasing within a job so the bar never visibly retreats; a step
        going *backwards* (or ``total_steps`` changing) is read as a brand-new
        job and resets the overall clock. The ETA's rate window rebases at
        every forward step boundary so each step's extrapolation reflects the
        pace the job is sustaining *now*, not an average polluted by phases
        that completed instantly (cached downloads) or crawled.
        """
        if not step or not total_steps or total_steps <= 0:
            self._overall_start_time = None
            self._overall_last_step = None
            self._overall_total_steps = None
            self._overall_smoothed_eta = None
            return None, None, None

        s = min(max(int(step), 1), int(total_steps))
        within = 0.0
        if within_override is not None:
            within = min(max(within_override, 0.0), 1.0)
        elif total and total > 0:
            within = min(max(current / total, 0.0), 1.0)
        raw = self._overall_raw_fraction(within, s, total_steps)
        step_end = self._overall_raw_fraction(1.0, s, total_steps)

        now = time.monotonic()
        new_job = (
            self._overall_start_time is None
            or self._overall_total_steps != total_steps
            or (self._overall_last_step is not None and s < self._overall_last_step)
        )
        if new_job:
            self._overall_start_time = now
            self._overall_start_frac = raw
            self._overall_max = raw
            self._overall_smoothed_eta = None
            self._overall_last_step = s
            self._overall_total_steps = total_steps
            return raw, step_end, None

        step_advanced = self._overall_last_step is not None and s > self._overall_last_step
        self._overall_last_step = s
        # Clamp to monotonic non-decreasing so within-step jitter never rewinds
        # the unified bar.
        if raw < self._overall_max:
            raw = self._overall_max
        else:
            self._overall_max = raw
        # The monotonic clamp can only hold values earned in this same step (a
        # later step would have advanced ``s``), so ``raw`` cannot legitimately
        # pass the slice end; clamp defensively so the pair always brackets.
        step_end = max(step_end, raw)

        # Past the new-job guard above, the overall clock is always running.
        assert self._overall_start_time is not None
        elapsed = now - self._overall_start_time
        progressed = raw - self._overall_start_frac
        if step_advanced:
            # Rebase the rate window at every forward step boundary. Phases
            # run at wildly different fraction-rates — a cached download banks
            # its whole bar span in milliseconds — and a job-global average
            # lets that instantly-earned span masquerade as sustained speed,
            # so the extrapolated ETA starts absurdly low on warm loads
            # (issue #2615) or absurdly high after a mispaced slow phase.
            # The boundary update itself still samples the *outgoing* step's
            # window (a full step observed over a real elapsed span is honest
            # signal, and the min-elapsed gate below already discards the
            # instantly-completed case); every later update extrapolates from
            # the rate the job sustains within the current step. The smoothed
            # EMA carries across so the displayed value stays continuous.
            self._overall_start_time = now
            self._overall_start_frac = raw
        if elapsed < self._ETA_MIN_ELAPSED or progressed <= 0 or raw >= 1.0:
            # Not enough signal yet (or done): hold the last smoothed estimate.
            return raw, step_end, self._overall_smoothed_eta
        raw_eta = elapsed * (1.0 - raw) / progressed
        if self._overall_smoothed_eta is None:
            self._overall_smoothed_eta = raw_eta
        else:
            alpha = self._ETA_SMOOTHING_ALPHA
            self._overall_smoothed_eta = alpha * raw_eta + (1.0 - alpha) * self._overall_smoothed_eta
        return raw, step_end, self._overall_smoothed_eta

    def update(
        self,
        status: str,
        message: str = "",
        current: int = 0,
        total: int = 0,
        **kwargs: Any,
    ) -> None:
        """Update progress in a thread-safe manner.

        Args:
            status: Current operation phase (e.g. ``"idle"``, ``"loading"``).
            message: Human-readable description of what is happening.
            current: Number of units completed so far.
            total: Total number of units expected (0 if unknown).
            **kwargs: Values for any extra fields declared at construction.
                Unrecognised keys are silently ignored. The reserved key
                ``within`` (float 0..1) overrides the within-step fraction
                used for the whole-job ``overall`` math without touching the
                displayed ``current``/``total`` — pacing layers use it to
                composite sub-phases while byte/item counts stay visible.
        """
        within_override = kwargs.pop("within", None)
        with self._lock:
            self._data["status"] = status
            self._data["message"] = message
            self._data["current"] = current
            self._data["total"] = total
            for key in self._extra_defaults:
                if key in kwargs:
                    self._data[key] = kwargs[key]
            # When the caller reports a multi-step structure, surface a single
            # whole-job ``overall`` fraction (0..1) and a true overall ETA so
            # the bar fills once across the entire job instead of resetting at
            # each phase. Otherwise fall back to the per-phase ETA.
            overall = None
            overall_step_end = None
            overall_eta = None
            if "overall" in self._extra_defaults or "eta_seconds" in self._extra_defaults:
                overall, overall_step_end, overall_eta = self._compute_overall(
                    current, total, self._data.get("step"), self._data.get("total_steps"), within_override
                )
            if "overall" in self._extra_defaults:
                self._data["overall"] = overall
            if "overall_step_end" in self._extra_defaults:
                self._data["overall_step_end"] = overall_step_end
            if "eta_seconds" in self._extra_defaults:
                raw_eta = overall_eta if overall is not None else self._compute_eta(status, current, total)
                # Published coarse and sticky — see :meth:`_humble_eta`. Every
                # consumer (SSE, the CLI bars, the frontend chips) reads this
                # one field, so humility applied here applies everywhere.
                self._data["eta_seconds"] = self._humble_eta(raw_eta)
            snapshot = dict(self._data)
        self._notify(snapshot)

    def subscribe(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """Register a callback fired with a snapshot after every update.

        The callback runs synchronously on the thread that called
        :meth:`update`, *outside* the tracker's lock. Subscribers must be
        non-blocking and exception-safe; any exception they raise is
        swallowed so a misbehaving subscriber cannot break the producer.
        """
        with self._subscribers_lock:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """Remove a previously-registered subscriber. No-op if not present."""
        with self._subscribers_lock:
            try:
                self._subscribers.remove(callback)
            except ValueError:
                pass

    def _notify(self, snapshot: dict[str, Any]) -> None:
        with self._subscribers_lock:
            subs = list(self._subscribers)
        for cb in subs:
            try:
                cb(snapshot)
            except Exception:
                pass

    def cancel(self) -> None:
        """Signal the background operation to stop.

        This sets the internal cancel event.  The background thread must
        cooperatively check it via :meth:`check_cancelled`.
        """
        self._cancel_event.set()

    def check_cancelled(self) -> None:
        """Raise :class:`CancelledError` if :meth:`cancel` has been called.

        Call this periodically from inside the background thread (e.g. once
        per loop iteration) to allow cooperative cancellation.
        """
        if self._cancel_event.is_set():
            raise CancelledError("Operation cancelled by user")

    @property
    def is_cancelled(self) -> bool:
        """Return ``True`` if cancellation has been requested."""
        return self._cancel_event.is_set()

    @property
    def cancel_event(self) -> threading.Event:
        """The cooperative-cancellation flag itself.

        Exposed so a holder that must publish *one* cancel signal (see
        :class:`~vtscore.concurrency.async_jobs.AsyncJob`, whose ``cancel_event``
        is this object) can hand out the tracker's event rather than keeping a
        second one that has to be kept in sync.  Prefer :meth:`cancel` /
        :meth:`check_cancelled` / :attr:`is_cancelled` for ordinary use; reach
        for the event only to ``wait()`` on it.
        """
        return self._cancel_event

    def reset_cancel(self) -> None:
        """Clear the cancellation flag.

        Called at the beginning of a new operation so that a previous
        cancellation does not immediately abort the next run.
        """
        self._cancel_event.clear()

    def get(self) -> dict[str, Any]:
        """Return a snapshot of the current progress data.

        Returns:
            A shallow copy of the internal data dict.
        """
        with self._lock:
            return dict(self._data)


# ---------------------------------------------------------------------------
# Thread-local progress callback
# ---------------------------------------------------------------------------
# Background loading threads set a per-thread progress callback via
# set_thread_progress(); every library module that reports progress without an
# explicit callback resolves one through resolve_progress_callback().  This
# avoids monkey-patching module-level defaults and allows parallel loads to
# each report to their own ProgressTracker.
#
# There is deliberately no process-wide fallback sink.  One used to exist (the
# ``dataset_progress`` singleton behind the SSE ``dataset`` channel), and it
# was a standing source of phantom progress bars: work that finished, died, or
# never had a watcher still left the channel parked on its last message, and
# nothing downstream could tell that apart from a wedged import (#3167).  Work
# that wants to be seen binds a tracker for its thread; work that binds nothing
# is, by construction, work nobody is watching, so it reports into a no-op.

#: The shape every progress sink in the library implements:
#: ``(status, message, current, total) -> None``.
ProgressCallback = Callable[[str, str, int, int], None]

_thread_progress = threading.local()


def noop_progress(status: str, message: str = "", current: int = 0, total: int = 0) -> None:
    """Progress sink that discards everything.  The default when none is bound."""
    return None


def set_thread_progress(callback) -> None:
    """Set the progress callback for the current thread."""
    _thread_progress.callback = callback


def get_thread_progress():
    """Return the per-thread progress callback, or ``None``."""
    return getattr(_thread_progress, "callback", None)


def clear_thread_progress() -> None:
    """Remove the per-thread progress callback."""
    _thread_progress.callback = None


def resolve_progress_callback() -> ProgressCallback:
    """Return the progress sink the calling thread should report into.

    The per-thread callback installed by :func:`set_thread_progress` when one
    is bound, else :func:`noop_progress`.  Library code that takes an optional
    ``on_progress`` argument calls this to fill in the ``None`` case, so a
    caller that supplied nothing still reaches whichever tracker the enclosing
    load bound for its thread.
    """
    cb = get_thread_progress()
    return cb if cb is not None else noop_progress


# ---------------------------------------------------------------------------
# Shared progress extras
# ---------------------------------------------------------------------------
#: Extras shared by every long-running operation: an optional sub-step counter
#: (``step``/``total_steps`` - used when a single operation has multiple phases
#: like load→embed→stage), an ``error`` string, and a smoothed ``eta_seconds``
#: filled in automatically by :meth:`ProgressTracker._compute_eta`. Every
#: singleton tracker - and every per-task tracker created by
#: :class:`LoadingTasksTracker`, and every per-job tracker on an
#: :class:`~vtscore.concurrency.async_jobs.AsyncJob` - exposes these so the
#: frontend can render any progress payload with the same ``ProgressEvent``
#: interface (see ``frontend/src/app/models/api.models.ts``).
PROGRESS_COMMON_EXTRAS: dict[str, Any] = {
    "step": None,
    "total_steps": None,
    "error": None,
    "eta_seconds": None,
    # Whole-job completion fraction (0..1) for multi-step operations, computed
    # by :meth:`ProgressTracker._compute_overall`. ``None`` for single-phase
    # operations (the frontend then falls back to ``current``/``total``).
    "overall": None,
    # Whole-job fraction (0..1) at which the current step's slice ends. Paired
    # with ``overall`` it brackets the job's true position when the current
    # step is count-less (``overall`` parks at the slice floor); the frontend
    # renders the span as a bounded indeterminate zone. ``None`` whenever
    # ``overall`` is ``None``.
    "overall_step_end": None,
}


# ---------------------------------------------------------------------------
# Loading tasks tracker - manages multiple concurrent loading operations
# ---------------------------------------------------------------------------


class LoadingTasksTracker:
    """Manages multiple concurrent dataset loading tasks.

    Each task has its own :class:`ProgressTracker`, a display name, and a
    creation timestamp.  The dashboard polls :meth:`list_tasks` to show
    one progress row per loading dataset.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tasks: dict[str, dict[str, Any]] = {}
        self._subscribers: list[Callable[[list[dict[str, Any]]], None]] = []
        self._subscribers_lock = threading.Lock()

    def subscribe(self, callback: Callable[[list[dict[str, Any]]], None]) -> None:
        """Register a callback fired with the task list after every change.

        Same semantics as :meth:`ProgressTracker.subscribe`: invoked
        synchronously, outside locks, exceptions swallowed.
        """
        with self._subscribers_lock:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[list[dict[str, Any]]], None]) -> None:
        """Remove a previously-registered subscriber. No-op if not present."""
        with self._subscribers_lock:
            try:
                self._subscribers.remove(callback)
            except ValueError:
                pass

    def _notify(self) -> None:
        snapshot = self.list_tasks()
        with self._subscribers_lock:
            subs = list(self._subscribers)
        for cb in subs:
            try:
                cb(snapshot)
            except Exception:
                pass

    def create_task(
        self,
        task_id: str,
        name: str = "",
        dataset_id: str = "",
        media_type: str = "",
        detector_id: str = "",
        embedder: str = "",
        step_weights: list[float] | None = None,
        extra_fields: dict[str, Any] | None = None,
    ) -> ProgressTracker:
        """Create and register a new loading task.

        *step_weights* (one weight per step) tunes how the whole-job ``overall``
        bar paces across phases; omit for equal weighting. *extra_fields* adds
        task-specific tracked keys (e.g. ``staging_result``) on top of the
        shared progress extras. Returns the per-task :class:`ProgressTracker`
        instance.
        """
        fields = dict(PROGRESS_COMMON_EXTRAS)
        if extra_fields:
            fields.update(extra_fields)
        tracker = ProgressTracker(extra_fields=fields)
        if step_weights is not None:
            tracker.set_step_weights(step_weights)
        tracker.subscribe(lambda _snapshot: self._notify())
        with self._lock:
            self._tasks[task_id] = {
                "tracker": tracker,
                "name": name,
                "created_at": time.time(),
                "finished_at": None,
                # Filled in by :meth:`set_worker` once the caller has its thread.
                "worker": None,
                "dataset_id": dataset_id,
                "media_type": media_type,
                "detector_id": detector_id,
                "embedder": embedder,
            }
        self._notify()
        return tracker

    def mark_finished(self, task_id: str) -> None:
        """Record the time a task finished (for deferred cleanup).

        ``list_tasks()`` prunes stale finished entries the next time it is
        called; SSE streams call it on every heartbeat tick so a
        background timer here is unnecessary.
        """
        with self._lock:
            entry = self._tasks.get(task_id)
            if entry:
                entry["finished_at"] = time.time()
        self._notify()

    def is_finished(self, task_id: str) -> bool:
        """Return ``True`` once :meth:`mark_finished` ran for *task_id*.

        The task body calls ``mark_finished`` from its outermost ``finally``,
        so a ``True`` here means the worker thread has fully unwound (gates
        released, contexts restored).  Tests use this to wait for a
        cancelled/finished task deterministically instead of sleeping.
        ``False`` for unknown (never created or already removed) task ids.
        """
        with self._lock:
            entry = self._tasks.get(task_id)
        return bool(entry and entry.get("finished_at") is not None)

    def get_tracker(self, task_id: str) -> ProgressTracker | None:
        """Return the ProgressTracker for *task_id*, or ``None``."""
        with self._lock:
            entry = self._tasks.get(task_id)
        return entry["tracker"] if entry else None

    def cancel_task(self, task_id: str) -> bool:
        """Signal a specific task to cancel.  Returns ``True`` if found."""
        tracker = self.get_tracker(task_id)
        if tracker is not None:
            tracker.cancel()
            return True
        return False

    def cancel_all(self) -> None:
        """Signal all active tasks to cancel."""
        with self._lock:
            tasks = list(self._tasks.values())
        for entry in tasks:
            entry["tracker"].cancel()

    def set_worker(self, task_id: str, thread: threading.Thread) -> None:
        """Record the thread running *task_id*, so cancellation can be honest.

        Cancellation is cooperative: :meth:`ProgressTracker.cancel` only sets a
        flag, and *something has to be running* to observe it.  Knowing whether
        a worker is still alive is what lets a cancel distinguish "it will stop
        shortly" from "there is nobody here to stop" instead of answering both
        with the same cheerful ``ok`` (#3167).

        Register the thread *before* starting it; a task with no registered
        worker reads as not running.
        """
        with self._lock:
            entry = self._tasks.get(task_id)
            if entry:
                entry["worker"] = thread

    def worker_alive(self, task_id: str) -> bool:
        """Return ``True`` if *task_id* has a registered worker still running."""
        with self._lock:
            entry = self._tasks.get(task_id)
        worker = entry.get("worker") if entry else None
        return bool(worker is not None and worker.is_alive())

    def active_task_ids(self) -> list[str]:
        """Return the ids of tasks whose tracker still claims to be working.

        "Claims" is deliberate: a stale tracker looks exactly like a running
        one from here, which is why :func:`cancel_dataset_progress` cross-checks
        each id against :meth:`worker_alive` before reporting what it did.
        """
        with self._lock:
            entries = list(self._tasks.items())
        return [tid for tid, e in entries if e["tracker"].get()["status"] != "idle"]

    def set_dataset_id(self, task_id: str, dataset_id: str) -> None:
        """Associate a loading task with its final registry dataset ID."""
        with self._lock:
            entry = self._tasks.get(task_id)
            if entry:
                entry["dataset_id"] = dataset_id
        self._notify()

    def remove_task(self, task_id: str) -> None:
        """Remove a completed/cancelled task from the tracker."""
        with self._lock:
            self._tasks.pop(task_id, None)
        self._notify()

    @staticmethod
    def _build_snapshot(task_id: str, entry: dict[str, Any]) -> dict[str, Any]:
        """Build the public snapshot dict for one task entry.

        Reads the task's :class:`ProgressTracker` and stamps on the
        ``task_id``, ``name``, ``created_at`` identity plus any optional
        association fields (``dataset_id``, ``detector_id``, ``media_type``,
        ``embedder``) that are set.
        """
        snapshot = entry["tracker"].get()
        snapshot["task_id"] = task_id
        snapshot["name"] = entry["name"]
        snapshot["created_at"] = entry["created_at"]
        for key in ("dataset_id", "detector_id", "media_type", "embedder"):
            if entry.get(key):
                snapshot[key] = entry[key]
        return snapshot

    def list_tasks(self) -> list[dict[str, Any]]:
        """Return a snapshot of all active loading tasks.

        Each entry includes: ``task_id``, ``name``, ``created_at``, and
        all fields from the task's :class:`ProgressTracker`.

        Finished tasks without errors are removed after 5 seconds.
        Finished tasks *with* errors are kept for 30 seconds so that
        the polling frontend has time to display them.
        """
        now = time.time()
        stale: list[str] = []
        with self._lock:
            entries = list(self._tasks.items())
        result = []
        for task_id, entry in entries:
            finished = entry.get("finished_at")
            if finished is not None:
                has_error = bool(entry["tracker"].get().get("error"))
                max_age = 30 if has_error else 5
                if (now - finished) > max_age:
                    stale.append(task_id)
                    continue
            result.append(self._build_snapshot(task_id, entry))
        if stale:
            with self._lock:
                for tid in stale:
                    self._tasks.pop(tid, None)
        return result

    def has_active_tasks(self) -> bool:
        """Return ``True`` if any loading task is still running (not idle)."""
        with self._lock:
            entries = list(self._tasks.values())
        return any(e["tracker"].get()["status"] != "idle" for e in entries)

    def reset_for_tests(self, join_timeout: float = 5.0) -> None:
        """Cancel every task, wait for its worker, then clear the registry.

        Clearing alone is not isolation: the entry it drops is the only handle
        anything has on the worker thread, so a load still running at the end of
        a test became an *unreachable* thread that carried on mutating shared
        state — contexts, progress, the load concurrency gates — underneath the
        next test (issue #3613).  Cancelling first gives the worker its
        cooperative exit; the join then makes "the previous test's threads are
        gone" true rather than hoped-for.

        *join_timeout* is a budget for the whole sweep, not per worker, so a
        thread parked on something no test will ever set costs a bounded delay
        once instead of stalling the run.  It is spent only when a worker is
        actually alive, which is the exceptional case.
        """
        with self._lock:
            entries = list(self._tasks.values())

        for entry in entries:
            entry["tracker"].cancel()

        deadline = time.monotonic() + join_timeout
        for entry in entries:
            worker = entry.get("worker")
            if worker is None or not worker.is_alive():
                continue
            worker.join(timeout=max(0.0, deadline - time.monotonic()))

        with self._lock:
            self._tasks.clear()


#: Application-wide loading tasks tracker (for datasets).
loading_tasks = LoadingTasksTracker()

#: Application-wide loading tasks tracker (for detectors).
detector_loading_tasks = LoadingTasksTracker()


# ---------------------------------------------------------------------------
# Application-wide singleton trackers
# ---------------------------------------------------------------------------

#: Sort-specific progress (used by text-sort operations).
sort_progress = ProgressTracker(extra_fields=dict(PROGRESS_COMMON_EXTRAS))

#: Eval progress (used by train-and-score / voting-iterations analysis).
eval_progress = ProgressTracker(extra_fields=dict(PROGRESS_COMMON_EXTRAS))

#: Find progress (used by the /api/find multi-dataset×model scoring operation).
find_progress = ProgressTracker(extra_fields=dict(PROGRESS_COMMON_EXTRAS))


# ---------------------------------------------------------------------------
# Backward-compatible free-function API
# ---------------------------------------------------------------------------


def _common_extras_kwargs(
    step: Any = _UNSET,
    total_steps: Any = _UNSET,
    error: Any = _UNSET,
) -> dict[str, Any]:
    """Build the kwargs dict for the shared ``step``/``total_steps``/``error`` extras.

    Only fields explicitly supplied by the caller are forwarded so omitted
    fields are left unchanged (true update/merge semantics).
    """
    kwargs: dict[str, Any] = {}
    if step is not _UNSET:
        kwargs["step"] = step
    if total_steps is not _UNSET:
        kwargs["total_steps"] = total_steps
    if error is not _UNSET:
        kwargs["error"] = error
    return kwargs


def _accepts_extras(cb: Callable[..., None], kwargs: dict[str, Any]) -> bool:
    """Return ``True`` when *cb* can be called with every key in *kwargs*.

    A sink that declares ``**kwargs`` takes anything; otherwise every key must
    name a real keyword parameter.  A callable whose signature cannot be read
    (a C builtin, an exotic wrapper) is assumed not to accept them, so the
    four-argument call is what runs.
    """
    try:
        params = inspect.signature(cb).parameters
    except (TypeError, ValueError):
        return False
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return True
    return all(key in params and params[key].kind is not inspect.Parameter.POSITIONAL_ONLY for key in kwargs)


def update_progress(
    status: str,
    message: str = "",
    current: int = 0,
    total: int = 0,
    error: Any = _UNSET,
    step: Any = _UNSET,
    total_steps: Any = _UNSET,
) -> None:
    """Report dataset/import progress into the calling thread's sink.

    The public free-function spelling of :func:`resolve_progress_callback` for
    plugin authors (see ``vtscore/docs/extending/dataset-importers.md``): an
    importer that calls this from inside a running load reports onto that
    load's own :class:`ProgressTracker`, exactly as if it had accepted an
    ``on_progress`` argument.  Called with no tracker bound to the thread it is
    a no-op.

    ``error``, ``step`` and ``total_steps`` are forwarded only when supplied,
    and only when the bound sink accepts them (a :class:`ProgressTracker`'s
    ``update`` does; the plain four-argument callbacks the load pipeline
    installs do not, and would raise on them).  Acceptance is decided by
    inspecting the sink's signature rather than by catching :class:`TypeError`
    from the call, so a ``TypeError`` raised *inside* a sink still propagates
    instead of being retried as an arity mismatch.
    """
    cb = resolve_progress_callback()
    kwargs = _common_extras_kwargs(step, total_steps, error)
    if kwargs and _accepts_extras(cb, kwargs):
        cb(status, message, current, total, **kwargs)  # type: ignore[call-arg]
        return
    cb(status, message, current, total)


#: How long :func:`cancel_dataset_progress` waits for a worker to act on the
#: flag before reporting what it saw. Cooperative cancellation normally lands
#: within one progress tick (well under a second); a worker that has not
#: finished by then is classified by whether its thread is still alive, not by
#: the clock, so this only has to be long enough to avoid calling a prompt
#: cancel "pending".
CANCEL_ACK_GRACE_SECONDS = 2.0

#: Poll interval while waiting for acknowledgement.
_CANCEL_POLL_SECONDS = 0.02


def _cancel_acknowledged(target: str) -> bool:
    """Return ``True`` once *target* has reached a terminal state."""
    if loading_tasks.is_finished(target):
        return True
    tracker = loading_tasks.get_tracker(target)
    # A task that has been pruned out of the tracker is as stopped as it gets.
    return tracker is None or tracker.get()["status"] == "idle"


def _park_unresponsive(target: str) -> None:
    """Clear a progress entry that claims work no living thread is doing.

    Refusing to lie about the cancel is only half the fix: the phantom that
    made the refusal necessary is still on screen, and it will still be there
    tomorrow. Nothing is going to clear it — that is what "no worker" means —
    so the request that proved it stale clears it.
    """
    tracker = loading_tasks.get_tracker(target)
    if tracker is not None:
        tracker.update(
            "idle",
            "",
            0,
            0,
            error="Abandoned: no worker was running this operation.",
        )
    loading_tasks.mark_finished(target)


def _cancel_report(targets: list[str], grace_seconds: float) -> dict[str, Any]:
    """Wait out the grace period and classify every target in *targets*."""
    report: dict[str, Any] = {
        "ok": False,
        "targets": list(targets),
        "acknowledged": [],
        "pending": [],
        "unresponsive": [],
        "message": "",
    }
    if not targets:
        report["message"] = "Nothing to cancel: no dataset operation is running."
        return report

    deadline = time.monotonic() + max(0.0, grace_seconds)
    outstanding = list(targets)
    while True:
        outstanding = [t for t in outstanding if not _cancel_acknowledged(t)]
        if not outstanding or time.monotonic() >= deadline:
            break
        time.sleep(_CANCEL_POLL_SECONDS)

    still = set(outstanding)
    report["acknowledged"] = [t for t in targets if t not in still]
    for target in outstanding:
        alive = loading_tasks.worker_alive(target)
        report["pending" if alive else "unresponsive"].append(target)
    for target in report["unresponsive"]:
        _park_unresponsive(target)

    report["ok"] = bool(report["acknowledged"] or report["pending"])
    stopped = len(report["acknowledged"])
    stopping = len(report["pending"])
    stale = len(report["unresponsive"])
    if report["ok"]:
        parts = []
        if stopped:
            parts.append(f"{stopped} stopped")
        if stopping:
            parts.append(f"{stopping} still stopping")
        if stale:
            parts.append(f"{stale} stale entr{'y' if stale == 1 else 'ies'} cleared")
        report["message"] = "Cancelled: " + ", ".join(parts) + "."
    else:
        report["message"] = (
            f"Nothing acknowledged the cancel: {stale} progress "
            f"entr{'y' if stale == 1 else 'ies'} claimed work with no worker running it. "
            "The operation had already finished or died; the stale progress has been cleared."
        )
    return report


def cancel_dataset_progress(grace_seconds: float = CANCEL_ACK_GRACE_SECONDS) -> dict[str, Any]:
    """Signal the current dataset operation(s) to cancel, and report the outcome.

    Cancels every active per-task loading tracker, then waits up to
    *grace_seconds* for someone to act on the flag.

    Cancellation is cooperative — :meth:`ProgressTracker.cancel` sets an event
    and :meth:`ProgressTracker.check_cancelled` has to be *reached* by a
    running thread — so setting the flag says nothing about whether anything
    will happen.  Reporting success unconditionally therefore answered the one
    question a stuck-looking import raises ("is anything actually running?")
    with the same word either way (#3167).

    Returns a report with:

    ``targets``
        every task id that claimed to be working when the cancel arrived.
    ``acknowledged``
        targets that reached a terminal state within the grace period.
    ``pending``
        targets still running, with a live worker thread that will observe the
        flag; the cancel was delivered, it just has not landed yet.
    ``unresponsive``
        targets whose progress claimed work no live thread was doing.  These
        are stale trackers, not running jobs; they are parked at ``idle`` so
        the phantom does not outlive the request that exposed it.
    ``ok``
        ``True`` when at least one target acknowledged or is pending — i.e.
        when the cancel actually reached something.
    """
    targets = loading_tasks.active_task_ids()
    loading_tasks.cancel_all()
    return _cancel_report(targets, grace_seconds)


def cancel_dataset_task(task_id: str, grace_seconds: float = CANCEL_ACK_GRACE_SECONDS) -> dict[str, Any] | None:
    """Cancel one loading task and report the outcome, or ``None`` if unknown.

    Same contract as :func:`cancel_dataset_progress`, narrowed to a single
    task: a task whose tracker still claims work but whose worker thread has
    exited is reported as ``unresponsive`` rather than cancelled.
    """
    tracker = loading_tasks.get_tracker(task_id)
    if tracker is None:
        return None
    targets = [task_id] if tracker.get()["status"] != "idle" else []
    tracker.cancel()
    return _cancel_report(targets, grace_seconds)


def update_sort_progress(
    status: str,
    message: str = "",
    current: int = 0,
    total: int = 0,
    step: Any = _UNSET,
    total_steps: Any = _UNSET,
    error: Any = _UNSET,
) -> None:
    """Update the sort progress tracker in a thread-safe manner."""
    sort_progress.update(status, message, current, total, **_common_extras_kwargs(step, total_steps, error))


def get_sort_progress() -> dict[str, Any]:
    """Return a snapshot of the current sort progress data."""
    return sort_progress.get()


def update_eval_progress(
    status: str,
    message: str = "",
    current: int = 0,
    total: int = 0,
    step: Any = _UNSET,
    total_steps: Any = _UNSET,
    error: Any = _UNSET,
) -> None:
    """Update the eval progress tracker in a thread-safe manner."""
    eval_progress.update(status, message, current, total, **_common_extras_kwargs(step, total_steps, error))


def get_eval_progress() -> dict[str, Any]:
    """Return a snapshot of the current eval progress data."""
    return eval_progress.get()


def update_find_progress(
    status: str,
    message: str = "",
    current: int = 0,
    total: int = 0,
    step: Any = _UNSET,
    total_steps: Any = _UNSET,
    error: Any = _UNSET,
) -> None:
    """Update the find progress tracker in a thread-safe manner."""
    find_progress.update(status, message, current, total, **_common_extras_kwargs(step, total_steps, error))


def get_find_progress() -> dict[str, Any]:
    """Return a snapshot of the current find progress data."""
    return find_progress.get()
