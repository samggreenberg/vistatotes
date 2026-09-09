"""Per-task timing model: how long each step of a long-running task will take.

Every long-running VTSearch operation — a dataset load, a detector load, a text
sort, a Find, a train-and-score, a promote — reports progress as
``step``/``total_steps`` and paces its unified bar with a per-step **weight
vector** (see :meth:`vtscore.concurrency.progress.ProgressTracker.set_step_weights`).
Those vectors used to be hand-guessed constants sitting next to each task's
code. A guess that is wrong in the same direction for the whole job is exactly
what makes a progress bar race one phase, crawl the next, and walk its ETA
*upward* while the user watches.

This package replaces the guesses with a **measurable, per-environment cost
model**. Each task declares its ordered step names (:mod:`vtscore.timing.tasks`);
each step gets an affine cost::

    T_step ≈ a + b · n + per_mb · archive_mb

where ``n`` is the task's natural scale variable (items to embed, labels to
train on, medias to score) and ``archive_mb`` covers the byte-scaled phases of a
download. Coefficients are keyed by a **cell** — ``(device, media_type,
embedder)`` — because the same step costs wildly different amounts on a V100
versus a laptop CPU, and on 200-character texts versus 30-second videos.

Three layers resolve a cell, most specific first:

1. **The admin profile.** A JSON file named by ``VTSEARCH_TIMING_PROFILE``,
   produced by ``scripts/profiling/tune_timing_profile.py`` on the hardware that
   will actually serve the app. This is the whole point of the package: a
   deployment measures itself once and every instance in that environment
   predicts its own timings thereafter.
2. **The shipped defaults** in :mod:`vtscore.timing.tasks` (and, for
   ``dataset_load``, the calibrated table in
   :mod:`vtscore.datasets.stages._load_cost_model`). These reproduce the
   pre-profile hand-tuned weights exactly, so an instance with no profile
   behaves as it always did.
3. **Equal weighting**, if a task is unknown entirely.

Nothing here persists to disk at runtime and nothing is cached across
processes: the profile is read once per process at first use and can be
re-read with :func:`reload_profile`.

See ``../docs/packages/timing.md`` for the package reference.
"""

from __future__ import annotations

from vtscore.timing.profile import (
    StepCoeffs,
    TimingProfile,
    active_profile,
    cell_keys,
    known_tasks,
    normalize_device,
    profile_covers,
    reload_profile,
    slot_shares,
    step_terms,
    step_weights,
)
from vtscore.timing.recorder import note_branch, note_no_encoder_load, record_task, recording_enabled
from vtscore.timing.tasks import CHEAP_BRANCHES, TASKS, DEAR_BRANCHES, TaskSpec, task_spec

__all__ = [
    "CHEAP_BRANCHES",
    "DEAR_BRANCHES",
    "TASKS",
    "StepCoeffs",
    "TaskSpec",
    "TimingProfile",
    "active_profile",
    "cell_keys",
    "known_tasks",
    "normalize_device",
    "note_branch",
    "note_no_encoder_load",
    "profile_covers",
    "record_task",
    "recording_enabled",
    "reload_profile",
    "slot_shares",
    "step_terms",
    "step_weights",
    "task_spec",
]
