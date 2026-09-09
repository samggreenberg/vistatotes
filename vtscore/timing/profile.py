"""Load, resolve, and apply the per-environment timing profile.

The profile is a JSON file whose path comes from ``VTSEARCH_TIMING_PROFILE``.
It is produced by ``scripts/profiling/tune_timing_profile.py`` on the hardware
that will serve the app, and read once per process at first use::

    {
      "schema": "vtsearch-timing-profile",
      "version": 1,
      "generated_at": "2026-07-30T12:00:00Z",
      "host": "prod-gpu-01",
      "notes": "measured on 4 exemplar datasets, 3 reps each",
      "tasks": {
        "text_sort": {
          "cells": {
            "cuda|image|siglip": {
              "samples": 12,
              "steps": {
                "load_model":  {"a": 8.2},
                "embed_query": {"a": 0.04},
                "score":       {"a": 0.1, "b": 0.00012}
              }
            }
          }
        }
      }
    }

A **cell key** is ``"<device>|<media_type>|<embedder>"``. Empty (or ``"*"``)
components are wildcards, so ``"cuda||"`` is "any media, any embedder, on CUDA"
and ``"||"`` is the task's environment-wide default. Lookup walks from the most
specific key to the least (see :func:`cell_keys`) and takes the first hit, so a
profile can carry one broad row plus a handful of precise overrides.

A step whose cost **forks** on a cache carries a second set of coefficients per
branch, nested inside the step under ``"branches"``::

    "coverage": {
      "a": 0.2, "b": 0.0026,
      "branches": {
        "restored": {"a": 0.009},
        "rebuilt":  {"a": 0.2, "b": 0.0026}
      }
    }

A restore of the cached coverage atlas and a rebuild of it are two code paths,
not two samples of one, and #3521 measured them 110-700x apart on the same
datasets — far too much for one number to pace both. The top-level coefficients
stay what they have always been (the branch a user waits on, when the fit saw
one), so a reader that knows nothing of branches — an older build, a
hand-written profile — is unaffected; ``branches`` is consulted only when the
caller says which path this run is taking. That is why adding it did not bump
:data:`SCHEMA_VERSION`: it is additive and safely ignorable, and bumping would
have made every new profile unreadable to the builds that can still use it.

Every coefficient is optional and defaults to zero:

``a``
    Fixed seconds for the step, however big the job is (loading an encoder).
``b``
    Seconds per unit of the task's scale variable ``n`` (per item embedded, per
    label trained on, per media scored).
``per_mb``
    Seconds per megabyte of downloaded archive, for the byte-scaled phases.

Nothing in this module raises on a malformed profile. A deployment that ships a
broken JSON file gets a warning in the log and the built-in defaults — a bad
timing table must never take the app down, because all it can ever affect is how
smoothly a progress bar moves.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Optional, Union

from vtscore.timing.tasks import TASKS, task_spec

logger = logging.getLogger(__name__)

#: Environment variable naming the profile JSON. Unset (or empty) means "use the
#: shipped defaults", which is the correct behaviour for a fresh install: the
#: defaults reproduce the pre-profile pacing exactly.
PROFILE_ENV_VAR = "VTSEARCH_TIMING_PROFILE"

#: Schema marker + highest version this build understands. A profile with a
#: higher version is rejected wholesale rather than half-read, so a newer
#: tuning script's output can't be silently misinterpreted by an older app.
SCHEMA_NAME = "vtsearch-timing-profile"
SCHEMA_VERSION = 1

#: Wildcard spellings accepted in a cell key component.
_WILDCARDS = ("", "*")


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------


def normalize_device(device: str) -> str:
    """Collapse a resolved device string to the coarse profile key.

    ``resolve_device()`` returns things like ``"cuda:0"``, ``"cuda"``, ``"cpu"``,
    or ``"mps"``; timing only distinguishes "has a GPU doing the tensor work"
    from "does not", because that is the split that moves phase costs by an
    order of magnitude.
    """
    return "cuda" if device.startswith("cuda") else "cpu"


def resolve_device_name() -> str:
    """Return ``"cuda"``/``"cpu"`` for the active device.

    Defaults to ``"cpu"`` when torch can't be resolved, so a library-only caller
    (or a machine mid-driver-upgrade) never crashes inside progress pacing.
    """
    try:
        from vtscore.config import resolve_device  # noqa: PLC0415

        return normalize_device(resolve_device())
    except Exception:
        return "cpu"


def cuml_active() -> bool:
    """Whether cuML will serve this process's clustering (never raises).

    cuML moves the coverage-atlas k-means and the UMAP projection onto the GPU,
    which materially changes the cost of any step that clusters — enough that
    CUDA cells are measured as two variants, ``"cuda+cuml"`` and ``"cuda"``.
    """
    try:
        from vtscore.gpu_backends import cuml_enabled  # noqa: PLC0415

        return cuml_enabled()
    except Exception:
        return False


def device_candidates(device: Optional[str] = None) -> tuple[str, ...]:
    """Device keys to try, best match first.

    A CPU host has exactly one key. A CUDA host has two — with and without cuML
    — and the live cuML state decides which is tried first; the other still
    beats falling back to a generic row, because a same-device measurement with
    a different clustering backend is much closer than no measurement at all.
    """
    dev = normalize_device(device) if device else resolve_device_name()
    if dev != "cuda":
        return (dev,)
    return ("cuda+cuml", "cuda") if cuml_active() else ("cuda", "cuda+cuml")


def cell_keys(device: Optional[str], media_type: str = "", embedder: str = "") -> tuple[str, ...]:
    """Cell keys to look up, most specific first.

    Specificity of ``(media_type, embedder)`` outranks the CUDA cuML variant: a
    row measured for exactly this media type and encoder on the other clustering
    backend predicts far better than a media-agnostic row on the right one.
    """
    devices = device_candidates(device)
    specificities = ((media_type, embedder), (media_type, ""), ("", ""))
    keys: list[str] = []
    for media, emb in specificities:
        for dev in devices:
            key = f"{dev}|{media}|{emb}"
            if key not in keys:
                keys.append(key)
    return tuple(keys)


def _canonical_cell_key(raw: str) -> str:
    """Normalize a cell key as written in JSON into its lookup form.

    Accepts ``"*"`` for a wildcard component (friendlier to hand-edit than an
    empty string) and tolerates surrounding whitespace, so a hand-tweaked
    profile matches what :func:`cell_keys` generates.
    """
    parts = [p.strip() for p in raw.split("|")]
    while len(parts) < 3:
        parts.append("")
    parts = ["" if p in _WILDCARDS else p for p in parts[:3]]
    return "|".join(parts)


# ---------------------------------------------------------------------------
# Profile data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StepCoeffs:
    """Affine cost of one step: ``a + b · n + per_mb · archive_mb`` seconds."""

    a: float = 0.0
    b: float = 0.0
    per_mb: float = 0.0
    #: Goodness of the affine fit these coefficients came from, or NaN when the
    #: step was not fitted that way (a byte-scaled step, a median fallback, or a
    #: hand-written profile).  `affine_fit` has always computed this and
    #: `fit_step` threw it away at the call site, which made this the one place
    #: in the tree that measured a fit's quality and then discarded it (#3329).
    #: Kept so a profile can be read for whether its cost model describes the
    #: deployment it was measured on.
    r2: float = float("nan")

    def seconds(self, n: float = 0.0, size_mb: float = 0.0) -> float:
        """Predicted wall-clock seconds for this step, never negative.

        A least-squares fit over noisy timings can land a negative intercept
        (a steep slope overshooting at small ``n``); clamping here keeps a
        pathological row from handing a step a negative slice of the bar.
        """
        return max(0.0, self.a + self.b * max(0.0, n) + self.per_mb * max(0.0, size_mb))

    @classmethod
    def from_json(cls, raw: Any) -> Optional["StepCoeffs"]:
        """Parse one step's coefficients, or ``None`` if unusable.

        A bare number is accepted as shorthand for a fixed cost, so a
        hand-written profile can say ``"embed_query": 0.04``.
        """
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            return cls(a=float(raw))
        if not isinstance(raw, dict):
            return None
        try:
            return cls(
                r2=float(raw.get("r2", float("nan"))),
                a=float(raw.get("a", 0.0)),
                b=float(raw.get("b", 0.0)),
                per_mb=float(raw.get("per_mb", 0.0)),
            )
        except (TypeError, ValueError):
            return None

    def to_json(self) -> dict[str, float]:
        """Serialize, omitting zero terms so a profile stays readable."""
        out: dict[str, float] = {"a": round(self.a, 6)}
        if self.b:
            out["b"] = round(self.b, 9)
        if self.per_mb:
            out["per_mb"] = round(self.per_mb, 6)
        # NaN means "this step was not fitted as a line", which is a different
        # statement from a bad fit, so it is omitted rather than written as
        # null: `from_json` defaults it back to NaN either way.
        if not math.isnan(self.r2):
            out["r2"] = round(self.r2, 4)
        return out


@dataclass(frozen=True)
class TimingProfile:
    """A parsed profile: per-task, per-cell step coefficients and slot shares.

    ``steps`` maps ``task -> cell_key -> step_name -> StepCoeffs``.
    ``branches`` maps ``task -> cell_key -> step_name -> branch -> StepCoeffs``
    and refines ``steps`` for the steps that fork on a cache; a lookup that
    names no branch never reads it.
    ``slots`` maps ``task -> cell_key -> step_name -> slot_name -> share`` and
    covers steps that are themselves split into ordered sub-stages (the dataset
    load's finalize phase is the only one today).

    An empty profile — the default when no ``VTSEARCH_TIMING_PROFILE`` is set —
    is falsy, so callers can cheaply skip the lookup entirely.
    """

    steps: dict[str, dict[str, dict[str, StepCoeffs]]] = field(default_factory=dict)
    branches: dict[str, dict[str, dict[str, dict[str, StepCoeffs]]]] = field(default_factory=dict)
    slots: dict[str, dict[str, dict[str, dict[str, float]]]] = field(default_factory=dict)
    source: str = ""
    generated_at: str = ""
    host: str = ""
    notes: str = ""

    def __bool__(self) -> bool:
        return bool(self.steps or self.slots)

    def describe(self) -> str:
        """One-line human summary, for startup logs and the tuning script."""
        if not self:
            return "timing profile: none (built-in defaults)"
        cells = sum(len(c) for c in self.steps.values())
        where = f" from {self.source}" if self.source else ""
        when = f", generated {self.generated_at}" if self.generated_at else ""
        on = f" on {self.host}" if self.host else ""
        return f"timing profile: {len(self.steps)} tasks / {cells} cells{where}{when}{on}"


EMPTY_PROFILE = TimingProfile()


def _header_ok(raw: dict, source: str) -> bool:
    """Whether the document's schema marker and version are ones we understand."""
    schema = raw.get("schema", SCHEMA_NAME)
    if schema != SCHEMA_NAME:
        logger.warning("timing profile %s: unknown schema %r; ignoring", source, schema)
        return False
    try:
        version = int(raw.get("version", SCHEMA_VERSION))
    except (TypeError, ValueError):
        version = -1
    if version > SCHEMA_VERSION or version < 1:
        logger.warning(
            "timing profile %s: version %r not supported by this build (max %d); ignoring",
            source,
            raw.get("version"),
            SCHEMA_VERSION,
        )
        return False
    return True


_TaskTables = tuple[
    dict[str, dict[str, StepCoeffs]],
    dict[str, dict[str, dict[str, StepCoeffs]]],
    dict[str, dict[str, dict[str, float]]],
]


def _parse_task(task: str, task_raw: Any, source: str) -> _TaskTables:
    """Parse one task's cells into ``(steps, branches, slots)``, each by cell."""
    steps: dict[str, dict[str, StepCoeffs]] = {}
    branches: dict[str, dict[str, dict[str, StepCoeffs]]] = {}
    slots: dict[str, dict[str, dict[str, float]]] = {}
    if not isinstance(task_raw, dict):
        return steps, branches, slots
    spec = task_spec(task)
    if spec is None:
        # Forward-compatible: a profile generated by a newer build may name
        # tasks this one has never heard of. Keeping them out of the tables
        # (rather than erroring) lets one profile serve a mixed fleet.
        logger.info("timing profile %s: skipping unknown task %r", source, task)
        return steps, branches, slots
    cells_raw = task_raw.get("cells")
    if not isinstance(cells_raw, dict):
        return steps, branches, slots
    known_steps = set(spec.steps)
    for cell, cell_raw in cells_raw.items():
        if not isinstance(cell_raw, dict):
            continue
        key = _canonical_cell_key(str(cell))
        parsed = _parse_cell_steps(cell_raw.get("steps"), known_steps)
        if parsed:
            steps[key] = parsed
        parsed_branches = _parse_cell_branches(cell_raw.get("steps"), known_steps)
        if parsed_branches:
            branches[key] = parsed_branches
        parsed_slots = _parse_cell_slots(cell_raw.get("slots"), known_steps)
        if parsed_slots:
            slots[key] = parsed_slots
    return steps, branches, slots


def parse_profile(raw: Any, source: str = "") -> TimingProfile:
    """Parse a decoded profile document into a :class:`TimingProfile`.

    Structurally invalid pieces are dropped with a warning rather than raising:
    one malformed cell must not cost the deployment every other measured cell.
    Returns :data:`EMPTY_PROFILE` when the document itself is unusable.
    """
    where = source or "<inline>"
    if not isinstance(raw, dict):
        logger.warning("timing profile %s: top level is not an object; ignoring", where)
        return EMPTY_PROFILE
    if not _header_ok(raw, where):
        return EMPTY_PROFILE
    tasks_raw = raw.get("tasks")
    if not isinstance(tasks_raw, dict):
        logger.warning("timing profile %s: missing 'tasks' object; ignoring", where)
        return EMPTY_PROFILE

    steps: dict[str, dict[str, dict[str, StepCoeffs]]] = {}
    branches: dict[str, dict[str, dict[str, dict[str, StepCoeffs]]]] = {}
    slots: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for task, task_raw in tasks_raw.items():
        task_steps, task_branches, task_slots = _parse_task(task, task_raw, where)
        if task_steps:
            steps[task] = task_steps
        if task_branches:
            branches[task] = task_branches
        if task_slots:
            slots[task] = task_slots

    return TimingProfile(
        steps=steps,
        branches=branches,
        slots=slots,
        source=source,
        generated_at=str(raw.get("generated_at", "")),
        host=str(raw.get("host", "")),
        notes=str(raw.get("notes", "")),
    )


def _parse_cell_steps(raw: Any, known_steps: set[str]) -> dict[str, StepCoeffs]:
    """Parse one cell's ``steps`` map, dropping unknown or unparseable steps."""
    if not isinstance(raw, dict):
        return {}
    out: dict[str, StepCoeffs] = {}
    for step, coeff_raw in raw.items():
        if step not in known_steps:
            continue
        coeffs = StepCoeffs.from_json(coeff_raw)
        if coeffs is not None:
            out[step] = coeffs
    return out


def _parse_cell_branches(raw: Any, known_steps: set[str]) -> dict[str, dict[str, StepCoeffs]]:
    """Parse the per-branch coefficients nested in one cell's ``steps`` map.

    Read from the same JSON object :func:`_parse_cell_steps` reads, because a
    branch refines a step and belongs beside it in the document. A step with no
    ``branches`` key — every step of every profile written before this existed —
    simply contributes nothing here.
    """
    if not isinstance(raw, dict):
        return {}
    out: dict[str, dict[str, StepCoeffs]] = {}
    for step, coeff_raw in raw.items():
        if step not in known_steps or not isinstance(coeff_raw, dict):
            continue
        branches_raw = coeff_raw.get("branches")
        if not isinstance(branches_raw, dict):
            continue
        parsed: dict[str, StepCoeffs] = {}
        for branch, branch_raw in branches_raw.items():
            coeffs = StepCoeffs.from_json(branch_raw)
            if coeffs is not None:
                parsed[str(branch)] = coeffs
        if parsed:
            out[step] = parsed
    return out


def _parse_cell_slots(raw: Any, known_steps: set[str]) -> dict[str, dict[str, float]]:
    """Parse one cell's ``slots`` map (``step -> slot -> share``)."""
    if not isinstance(raw, dict):
        return {}
    out: dict[str, dict[str, float]] = {}
    for step, slot_raw in raw.items():
        if step not in known_steps or not isinstance(slot_raw, dict):
            continue
        shares: dict[str, float] = {}
        for slot, value in slot_raw.items():
            try:
                share = float(value)
            except (TypeError, ValueError):
                continue
            if share > 0:
                shares[str(slot)] = share
        if shares:
            out[step] = shares
    return out


# ---------------------------------------------------------------------------
# Process-wide active profile
# ---------------------------------------------------------------------------

_profile_lock = threading.Lock()
_profile: Optional[TimingProfile] = None


def _read_profile(path: str) -> TimingProfile:
    """Read and parse the profile at *path*, or :data:`EMPTY_PROFILE` on any error."""
    try:
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
    except OSError as exc:
        logger.warning("timing profile %s: cannot read (%s); using built-in defaults", path, exc)
        return EMPTY_PROFILE
    except json.JSONDecodeError as exc:
        logger.warning("timing profile %s: invalid JSON (%s); using built-in defaults", path, exc)
        return EMPTY_PROFILE
    return parse_profile(raw, source=path)


def active_profile() -> TimingProfile:
    """Return the process's profile, reading it on first use.

    Thread-safe and read-once: every long-running task consults this on its way
    into the work, so it must not re-stat a file per progress update.
    """
    global _profile
    if _profile is not None:
        return _profile
    with _profile_lock:
        if _profile is None:
            path = os.environ.get(PROFILE_ENV_VAR, "").strip()
            _profile = _read_profile(path) if path else EMPTY_PROFILE
            if _profile:
                logger.info("%s", _profile.describe())
    return _profile


def reload_profile(path: Optional[str] = None) -> TimingProfile:
    """Re-read the profile, optionally from an explicit *path*.

    Passing ``path=""`` clears the profile back to the built-in defaults. Used
    by the tuning script (to verify what it just wrote) and by tests; the app
    itself reads the profile once at startup and never reloads it.
    """
    global _profile
    with _profile_lock:
        target = path if path is not None else os.environ.get(PROFILE_ENV_VAR, "").strip()
        _profile = _read_profile(target) if target else EMPTY_PROFILE
    return _profile


def _lookup_cell(
    task: str, device: Optional[str], media_type: str, embedder: str
) -> tuple[dict[str, StepCoeffs], dict[str, dict[str, StepCoeffs]]]:
    """Return the best-matching cell's ``(step coefficients, branch refinements)``.

    Both come from the *same* winning key. Resolving them separately would let a
    cell's measured ``coverage`` be refined by another cell's branch split — two
    different machines' measurements multiplied together.
    """
    profile = active_profile()
    cells = profile.steps.get(task)
    if not cells:
        return {}, {}
    for key in cell_keys(device, media_type, embedder):
        row = cells.get(key)
        if row:
            return row, profile.branches.get(task, {}).get(key, {})
    return {}, {}


# ---------------------------------------------------------------------------
# Public lookup API
# ---------------------------------------------------------------------------


def _branch_name(step: str, branch: Optional[Union[str, Mapping[str, str]]]) -> str:
    """Which branch the caller says *step* is taking (``""`` when it doesn't).

    A bare string names the branch for whichever steps recorded one — the branch
    vocabulary is global (``restored`` belongs to exactly one step of exactly one
    task), so a caller with one decision to report need not spell out which step
    it belongs to. A mapping is for the caller that knows about several.
    """
    if branch is None:
        return ""
    if isinstance(branch, str):
        return branch
    return str(branch.get(step, ""))


def step_terms(
    task: str,
    *,
    device: Optional[str] = None,
    media_type: str = "",
    embedder: str = "",
    n: float = 0.0,
    size_mb: float = 0.0,
    skip_steps: Iterable[str] = (),
    branch: Optional[Union[str, Mapping[str, str]]] = None,
) -> Optional[dict[str, float]]:
    """Predicted seconds per step for *task*, keyed by step name.

    Returns ``None`` when neither a profile cell nor a shipped default covers
    the task, so the caller can keep its own fallback. When a profile cell
    exists but only names *some* of the task's steps, the unnamed steps fall
    back to their shipped default term — a partial measurement improves the
    steps it covers without blanking the others.

    ``n`` is the task's scale variable (see :attr:`TaskSpec.scale`); ``size_mb``
    is the archive size for byte-scaled phases. Both may be zero, which simply
    collapses their terms.

    *skip_steps* names steps **this invocation will not enter at all**, and
    prices them at zero regardless of what the profile or the defaults say. It
    is not a cost model and needs no measurement: a step that does not run
    costs nothing, and that is knowable in advance where a step's cost is not.
    See :func:`step_weights` for why a task reaches for it.

    ``branch`` names the path a forking step is taking on *this* run — a branch
    name from :data:`~vtscore.timing.tasks.CHEAP_BRANCHES` /
    :data:`~vtscore.timing.tasks.DEAR_BRANCHES`, or a ``{step: branch}`` mapping.
    A step with per-branch coefficients for that name is priced from them; every
    other step, and every profile that carries no branch split, prices exactly as
    it does with ``branch=None``. So passing a branch can only ever *sharpen* the
    answer, never blank it: a caller that knows its branch should always say so.
    It answers the question *skip_steps* cannot — a step that runs either way but
    costs two different things — so a run that skips a step names it there and a
    run that takes the cheap path of one names it here.
    """
    spec = task_spec(task)
    if spec is None:
        return None
    measured, forks = _lookup_cell(task, device, media_type, embedder)
    defaults = dict(zip(spec.steps, spec.default_terms)) if spec.default_terms else {}
    if not measured and not defaults:
        return None
    skipped = frozenset(skip_steps)
    terms: dict[str, float] = {}
    for step in spec.steps:
        if step in skipped:
            terms[step] = 0.0
            continue
        coeffs = forks.get(step, {}).get(_branch_name(step, branch)) or measured.get(step)
        terms[step] = coeffs.seconds(n, size_mb) if coeffs is not None else max(0.0, defaults.get(step, 0.0))
    if sum(terms.values()) <= 0:
        # An all-zero prediction carries no pacing information and would make
        # the weight vector a division by zero; the caller's fallback is better.
        return None
    return terms


def step_weights(
    task: str,
    *,
    device: Optional[str] = None,
    media_type: str = "",
    embedder: str = "",
    n: float = 0.0,
    size_mb: float = 0.0,
    skip_steps: Iterable[str] = (),
    branch: Optional[Union[str, Mapping[str, str]]] = None,
    fallback: Optional[list[float]] = None,
) -> Optional[list[float]]:
    """Normalized per-tracker-step weights for *task*, or *fallback*.

    The returned vector has one entry per tracker step (``TaskSpec.tracker_steps``)
    and sums to 1, ready for
    :meth:`~vtscore.concurrency.progress.ProgressTracker.set_step_weights`.
    Phases that share a tracker step have their predicted seconds summed into
    that step's slot.

    **Steps this run will skip.** A cost model answers "how long does this step
    take"; it cannot answer "does this step happen at all", and for a step that
    forks on process state the second question decides the bar. A text sort's
    ``load_model`` is seconds on the first sort of a process and *exactly zero*
    on every later one, so no single coefficient paces both: fitted from the
    warm runs it is free (and gets floored back up, see
    :data:`vtscore.timing.fit._ONCE_PER_PROCESS_FLOOR_S`), fitted from the cold
    one it eats a bar that will not move. #3596 measured every arm of #3521's
    study — two fitted profiles and the shipped defaults alike — at 0.80–0.85
    bar error on ``text_sort`` for exactly this reason.

    The caller usually *knows*: the sort route can ask whether the encoder is
    already resident before it starts. Naming the step in *skip_steps* prices it
    at zero for this run, which needs no measurement and no branch axis in the
    profile format — a step that does not run costs nothing.

    **Steps whose cost merely varies by branch** are the other half, and they do
    need the profile's branch axis: ``branch`` is passed through to
    :func:`step_terms`, which prices such a step from that branch's own
    coefficients. A task whose expensive step forks that way should call this
    **twice**: once on the way in with whatever it can guess, and again the
    moment it knows — the two vectors differ by up to the whole bar (#3594).
    """
    spec = task_spec(task)
    terms = step_terms(
        task,
        device=device,
        media_type=media_type,
        embedder=embedder,
        n=n,
        size_mb=size_mb,
        skip_steps=skip_steps,
        branch=branch,
    )
    if spec is None or terms is None:
        return fallback
    weights = [0.0] * spec.tracker_steps
    for step, index in zip(spec.steps, spec.step_index):
        weights[index - 1] += terms[step]
    total = sum(weights)
    if total <= 0:
        return fallback
    return [w / total for w in weights]


def slot_shares(
    task: str,
    step: str,
    *,
    device: Optional[str] = None,
    media_type: str = "",
    embedder: str = "",
) -> Optional[dict[str, float]]:
    """Measured sub-stage shares within one step, or ``None`` when unprofiled.

    Only steps that pace several ordered sub-stages behind a single step number
    use this — today just the dataset load's ``finalize``. Shares are raw
    weights; the consumer normalizes them.
    """
    profile = active_profile()
    cells = profile.slots.get(task)
    if not cells:
        return None
    for key in cell_keys(device, media_type, embedder):
        row = cells.get(key)
        if row and step in row:
            return dict(row[step])
    return None


def profile_covers(task: str) -> bool:
    """Whether the active profile carries any measured cell for *task*.

    Public API with **no caller inside this repository**: the tuning script's
    coverage report goes through :func:`vtscore.timing.fit.coverage_report`
    (which reads the profile dict directly), and the dataset-load path always
    consults :func:`step_weights` with a fallback rather than gating on
    coverage first.  It is exported for out-of-tree callers that want to
    branch on whether a profile has anything to say about a task before
    asking for weights - and is deliberately kept rather than deleted, since
    an in-repo grep cannot see those callers.
    """
    return bool(active_profile().steps.get(task))


def known_tasks() -> tuple[str, ...]:
    """Names of every registered task family, in registry order."""
    return tuple(TASKS)
