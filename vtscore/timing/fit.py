"""Turn recorded step timings into a timing-profile document.

This is the writer for the format :mod:`vtscore.timing.profile` reads. It lives
here, next to the reader, so the two cannot drift: a change to the schema has to
be made in one directory or it will not round-trip.

Input is JSONL from either recorder — the generic one
(:mod:`vtscore.timing.recorder`) or the older dataset-load profiler
(:mod:`vtscore.datasets.stages._load_profiler`) — because an admin should be
able to fold an existing calibration sweep into a new profile rather than
re-measuring what has already been measured. :func:`normalize_row` flattens both
shapes into one.

The fit itself is deliberately plain. Per ``(task, cell, step)``:

- **Byte-scaled steps** (a dataset download and its unpack) get a per-MB rate:
  the median of ``seconds / archive_mb``. Regressing them against item count
  would ask ``n`` to explain something it cannot see — 500 videos and 500 text
  files are the same ``n`` and two orders of magnitude apart in bytes.
- **Everything else** gets ordinary least squares against ``n``, giving the
  intercept (what the step costs at all — loading an encoder, opening a file)
  and the slope (what each additional item adds).
- A fit with no spread in ``n``, or one that comes back with a *negative* slope
  (noise beating signal on a short step), collapses to ``median seconds`` with
  no slope. A confidently wrong slope extrapolates badly at sizes the sweep
  never visited, and "we only know the average" is the honest answer there.
- **Cold runs are held out** when the warm ones can carry the regression alone.
  A cold run pays once-per-process costs no later run repeats, and it always
  lands at whichever ``n`` happened to go first, so it has enormous leverage on
  the slope. See :func:`fit_step`.
- **A step that forked on a cache is priced from the runs that did the work**,
  and a step whose runs *all* read a cache is not priced at all — the whole cell
  is withheld and the task keeps its shipped defaults. This is not the same
  judgement as the cold/warm holdout above. A warm run is a cheap sample of the
  same code path; a cached run is a measurement of a different one, and the two
  populations have no common mean worth reporting. See :func:`cheap_branch_only`
  and :data:`vtscore.timing.tasks.CHEAP_BRANCHES` (#3521).
- **A step measured on both paths is additionally priced per branch**, under the
  step's own ``branches`` key, so a caller that knows which path it is taking
  gets that branch's coefficients instead of the other's. The step's top-level
  coefficients are unchanged — still the dear branch — because that is what a
  caller who cannot say gets, and what every existing reader sees. See
  :func:`fit_branches` (#3594).

Cells are emitted at three specificities — exact ``(device, media, embedder)``,
then ``(device, media, *)``, then ``(device, *, *)``. The rollups are what make a
small sweep worth running: an admin who measures three exemplar datasets still
improves the pacing of every task on every dataset that host will ever see,
because the least-specific cell always matches.

That guarantee has a price, and it is not small. #3345 measured the three levels
against the same rows with the same fitter: exact cells at median r² 1.00 and
3 % prediction error, ``(device, media, *)`` at 0.99 and 9 %, and
``(device, *, *)`` at **0.29 and 50 %** — 162 % on one arm. So two things temper
the rollups here:

- A rollup step whose pooled groups disagree by more than
  :data:`_MAX_ROLLUP_SPREAD` is **not emitted at all**, and falls through to the
  shipped default. A rollup is only ever *reached* for a combination the sweep
  never measured (see :func:`_rollup_is_contradicted`), so a cell built by
  averaging things measured to be unlike is extrapolating from a number it has
  already been told is wrong for both of them.
- :func:`coverage_report` breaks its fit-quality line down **by specificity**,
  so an admin reading "5 cells" sees how many are exact measurements and how
  many are the fallbacks that will actually pace an unmeasured dataset.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any, Iterable, Optional

from vtscore.timing.profile import SCHEMA_NAME, SCHEMA_VERSION, StepCoeffs
from vtscore.timing.tasks import CHEAP_BRANCHES, DEAR_BRANCHES, TASKS, task_spec

#: Phase names the older dataset-load profiler writes, mapped onto the task
#: registry's step names for ``dataset_load``.
_LEGACY_PHASE_TO_STEP = {
    "download": "download",
    "extract": "extract",
    "model_load": "load",
    "embed": "embed",
    "finalize": "finalize",
}

#: A byte-scaled step under this many seconds is dominated by setup overhead
#: rather than transfer, so it makes a poor per-MB rate sample.
_MIN_BYTE_STEP_SECONDS = 0.1

#: Distinct ``n`` values a fit needs before its r2 means anything. Two points
#: define a line exactly, so r2 is 1.0 whatever they are.
_MIN_R2_POINTS = 3


def load_rows(paths: Iterable[str]) -> list[dict]:
    """Read JSONL from every path, skipping blank and unparseable lines.

    A recorder appends from live worker threads, so a file can end mid-line if
    the process died; one truncated row must not cost the sweep every good row
    before it.
    """
    import json  # noqa: PLC0415 - only needed on the file path

    rows: list[dict] = []
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
    return rows


def device_key(device: str, cuml: bool) -> str:
    """Collapse a recorded device onto the profile's device key.

    CUDA splits in two: cuML moves the clustering work onto the GPU, which
    changes what the finalize-shaped steps cost enough that pooling the two
    would average away the thing being measured.
    """
    if not str(device).startswith("cuda"):
        return "cpu"
    return "cuda+cuml" if cuml else "cuda"


def normalize_row(raw: dict) -> Optional[dict]:
    """Flatten one recorded row into ``{task, device, media_type, embedder, n,
    size_mb, step, slot, seconds}``, or ``None`` if it is not a usable sample.

    Handles both recorder shapes. Rejects rows that describe something other
    than a completed unit of work: a failed or partial run, or a legacy row with
    ``n <= 0`` (which is how the old profiler marks a load that died before
    producing any items).
    """
    seconds = raw.get("seconds")
    if seconds is None:
        return None
    try:
        seconds = float(seconds)
    except (TypeError, ValueError):
        return None

    task = raw.get("task")
    phase = raw.get("phase")
    slot = ""
    if task:
        step = raw.get("step")
        if not raw.get("ok", True) or not raw.get("complete", True):
            return None
        size_mb = float(raw.get("size_mb") or 0.0)
    elif phase:
        # Legacy dataset-load profiler row.
        task = "dataset_load"
        if float(raw.get("n") or 0) <= 0:
            return None
        if str(phase).startswith("finalize:"):
            step, slot = "finalize", str(phase).split(":", 1)[1]
        else:
            step = _LEGACY_PHASE_TO_STEP.get(str(phase))
        size_mb = float(raw.get("download_size_mb") or 0.0)
    else:
        return None

    spec = task_spec(str(task))
    if spec is None or not step or (not slot and step not in spec.steps):
        return None

    return {
        "task": str(task),
        "device": device_key(str(raw.get("device", "cpu")), bool(raw.get("cuml"))),
        "media_type": str(raw.get("media_type", "")),
        "embedder": str(raw.get("embedder", "")),
        "n": float(raw.get("n") or 0.0),
        "size_mb": size_mb,
        "step": str(step),
        "slot": slot,
        "seconds": max(0.0, seconds),
        # Both recorders spell this ``cold_model``. Absent means warm, which is
        # what a row recorded before the marker existed — or one from a task
        # that loads no encoder — should be treated as: it keeps every such
        # sample in a single population and fits exactly as it did before.
        "cold": bool(raw.get("cold_model", False)),
        # Which path this step took, when the code that chose it said so (see
        # ``vtscore.timing.note_branch``). Absent means "not a forking step, or
        # a row recorded before the marker existed"; such rows keep fitting
        # exactly as they did, because an unmarked population is not evidence
        # that a cheap branch monopolised it.
        "branch": str(raw.get("branch", "")),
    }


def affine_fit(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Ordinary least squares ``y ≈ a + b·x``, returning ``(a, b, r2)``.

    Falls back to ``(mean, 0, 0)`` when ``x`` has no spread — one dataset size
    can tell you what a step costs, but nothing about how it scales.
    """
    count = len(xs)
    if count == 0:
        return 0.0, 0.0, 0.0
    mean_x = sum(xs) / count
    mean_y = sum(ys) / count
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 1e-9 or count < 2:
        return mean_y, 0.0, 0.0
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else 1.0
    return intercept, slope, r2


def _r2_of(xs: list[float], ys: list[float], a: float, b: float) -> float:
    """Goodness of the *given* coefficients against the data, not of a best fit.

    :func:`affine_fit` scores the line it computed. When the coefficients that
    get stored differ from that line — the intercept is clamped at zero below —
    that score describes a model the profile does not contain, so it has to be
    recomputed against the one it does.
    """
    count = len(ys)
    if count == 0:
        return float("nan")
    mean_y = sum(ys) / count
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    if ss_tot <= 1e-9:
        return 1.0
    ss_res = sum((y - max(0.0, a + b * x)) ** 2 for x, y in zip(xs, ys))
    return 1.0 - ss_res / ss_tot


#: Seconds a step keeps when its warm runs measured it as free but a cold run in
#: the same cell measured it as real. Mirrors ``_WARM_MODEL_FLOOR_S`` in
#: ``scripts/profiling/fit_load_weights.py``, whose warm model load gets the same
#: value for the same reason; the two fitters have to agree, and ``vtscore``
#: cannot import from ``scripts/``.
_ONCE_PER_PROCESS_FLOOR_S = 0.5


def _deferred_cost_floor(median_seconds: float, cold: list[dict]) -> float:
    """Keep a skipped-because-warm step visible on the bar rather than free.

    The generic recorder writes an explicit ``0.0`` for a step a run skipped, so
    a cost paid once per process fits to a warm median of exactly zero — a true
    statement about 47 of 48 text sorts and a useless one about the 48th, which
    is the run somebody is watching. When a cold run in the same cell measured
    the step as real, the step is not free here, it is *deferred*, and the bar
    should keep a slice for it.

    Deliberately a floor and not the cold cost. Pacing every run at the cold
    price is as wrong as pacing it at zero, just in the other direction, and
    most runs are warm — which is why ``fit_load_weights.py`` floors its warm
    model load at the same value and carries the cold figure as a note. The
    floor is a guard against a confident zero, not a cost model; measuring the
    cold branch properly is the tuning driver's job (#3521).
    """
    if median_seconds > 0 or not any(s["seconds"] > 0 for s in cold):
        return max(0.0, median_seconds)
    return _ONCE_PER_PROCESS_FLOOR_S


def branch_split(samples: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """Partition *samples* into ``(dear, cheap, unmarked)`` by their branch.

    Unmarked rows are their own population rather than being folded into either
    named one: an unmarked row is a step that never forks, or a row written
    before the marker existed, and neither is evidence about a fork.
    """
    dear = [s for s in samples if s.get("branch") in DEAR_BRANCHES]
    cheap = [s for s in samples if s.get("branch") in CHEAP_BRANCHES]
    unmarked = [s for s in samples if not s.get("branch")]
    return dear, cheap, unmarked


def cheap_branch_only(samples: list[dict]) -> bool:
    """True when every run of this step took a cached path and none did the work.

    The step is then **unpriceable from these rows**, and that is a different
    state from "measured as cheap". #3345's sweep opened 16 datasets and every
    one of them restored the coverage atlas cached in its pickle; the honest
    reading of 0.008-0.016 s is not "the atlas costs 10 ms here", it is "this
    sweep never saw one built". Both are correct measurements; only one of them
    is a cost model, and ``tasks.py`` weights that step at 0.85 of the bar
    precisely because the branch nobody measured is the one that does the work:
    a rebuild costs 0.0026 s/item, ~700x the restore at n = 2954 (#3595).
    """
    dear, cheap, unmarked = branch_split(samples)
    return bool(cheap) and not dear and not unmarked


def fit_branches(samples: list[dict], byte_scaled: bool) -> dict[str, StepCoeffs]:
    """Fit one step's coefficients **per branch**, when its runs took more than one.

    :func:`fit_step` answers "what does this step cost?" with one number, which
    it can only do by choosing a branch — it prices from the runs that did the
    work and lets the cheap ones go. That is the right single answer, and #3521
    measured how far it is from the other one: a coverage atlas restores in
    0.011 s and rebuilds in 7.7 s on the same 2954-item dataset, and a bar paced
    by either number is ~0.5-0.94 of a bar wrong on the other branch.

    So the branches are also fitted separately and stored beside the step, for
    the lookup that knows which path it is about to take. Each branch is fitted
    by :func:`fit_step` over its own rows alone — the cold/warm holdout and the
    median fallback apply within a branch exactly as they do without one.

    Returns ``{}`` for a byte-scaled step (a per-MB rate has no fork to split)
    and for a step whose runs all took one branch, where a per-branch entry
    would only restate the step's own coefficients. A single branch measured is
    not evidence about the branch nobody ran, and writing it as though it were a
    split would suggest the profile knows something it does not.
    """
    if byte_scaled:
        return {}
    named: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        if sample.get("branch"):
            named[sample["branch"]].append(sample)
    if len(named) < 2:
        return {}
    out: dict[str, StepCoeffs] = {}
    for branch, group in named.items():
        coeffs = fit_step(group, byte_scaled=False)
        if coeffs is not None:
            out[branch] = coeffs
    return out


def fit_step(samples: list[dict], byte_scaled: bool) -> Optional[StepCoeffs]:
    """Fit one step's coefficients from its samples, or ``None`` if unusable."""
    if not samples:
        return None
    if byte_scaled:
        rates = [
            s["seconds"] / s["size_mb"] for s in samples if s["size_mb"] > 0 and s["seconds"] >= _MIN_BYTE_STEP_SECONDS
        ]
        if not rates:
            # Every sample was a cached/absent archive: this deployment's loads
            # genuinely pay nothing here, which is a real (zero) measurement.
            return StepCoeffs()
        return StepCoeffs(per_mb=statistics.median(rates))

    # Fit the warm population. A cold run folds in costs paid once per *process*
    # rather than once per job — the encoder download, the CUDA context, the
    # first forward pass — and it always lands at whichever ``n`` ran first, so
    # it has enormous leverage on the slope. #3062 measured a single cold row
    # pulling a load's finalize slope to 0.0018 s/item from the warm 0.0040 and
    # collapsing its r² from 0.999 to 0.08.
    #
    # The holdout stops short of costing a cell its only line. Below two
    # distinct sizes no slope is estimable at all, and a minimal sweep is
    # exactly that shape — its first run is always the cold one — so a strict
    # holdout would turn every two-run sweep into a table of flat medians. There
    # (and in a cell that only ever ran cold, which is all the legacy profiler
    # ever writes for a model load) the cold rows stay in the regression and the
    # floor below does what it can.
    # A step that forked on a cache is priced from the runs that did the work.
    # Unlike the cold/warm holdout below this is not leverage control: a cached
    # run is not a cheap sample of the same cost, it is a measurement of a
    # different code path, and averaging the two describes neither. Callers
    # reach here only when at least one such run exists — ``cheap_branch_only``
    # withholds the cell otherwise — so this never empties the population.
    worked = [s for s in samples if s.get("branch") in DEAR_BRANCHES]
    if worked and len(worked) < len(samples):
        samples = worked

    cold = [s for s in samples if s.get("cold")]
    warm = [s for s in samples if not s.get("cold")]
    fittable = warm if len({s["n"] for s in warm}) >= 2 else samples

    xs = [s["n"] for s in fittable]
    ys = [s["seconds"] for s in fittable]
    intercept, slope, r2 = affine_fit(xs, ys)
    if len({round(x, 6) for x in xs}) < _MIN_R2_POINTS:
        # A line through two points has ss_res == 0, so its r2 is 1.0 by
        # construction and says nothing (#3345). The coefficients are still the
        # best line available and are kept; only the goodness claim is withheld.
        # The clamp below deliberately overrides this: once the stored model is
        # no longer the interpolant, its residuals are real at any point count.
        r2 = float("nan")
    if slope <= 0:
        # No credible scaling signal (a flat step, or noise swamping a short
        # one). Report the typical cost and claim nothing about growth. The r2
        # is deliberately NOT carried here: it describes a line this branch has
        # just declined to use, so reporting it would attach a goodness score to
        # coefficients that are a median.
        return StepCoeffs(a=_deferred_cost_floor(statistics.median(ys), cold))

    a = max(0.0, intercept)
    if a != intercept:
        # A steep slope through a short step lands a negative intercept, and
        # `seconds()` would hand the bar a negative slice, so it is clamped. The
        # r2 above then scores a line nobody stores: #3345 measured 58 of 195
        # affine cells in this state, one of them annotating coefficients 52%
        # out with an r2 of 0.98. Rescore against what is actually kept -- which
        # can come back negative, meaning "worse than a constant", and that is
        # a true statement worth persisting rather than hiding.
        r2 = _r2_of(xs, ys, a, slope)
    return StepCoeffs(a=a, b=slope, r2=r2)


#: Ratio between the cheapest and dearest group a rollup pools, above which the
#: pooled fit is treated as contradicted by its own samples rather than merely
#: noisy. #3345 measured the harmful case at 7.3x — ``(cuda+cuml, *, *)`` fitting
#: one slope through a 0.014 s/item image import and a 0.102 s/item audio one,
#: for a median prediction error of 50% against 3% for the exact cells. The
#: threshold sits well above the spread a well-behaved rollup shows (that study's
#: media rollups ran at 0.09 error) so it fires on disagreement, not on scatter.
_MAX_ROLLUP_SPREAD = 3.0

#: A predicted step cost at or below this is "free" for pacing purposes. Ratios
#: between such numbers are arithmetic noise, so they are compared against this
#: floor instead: two negligible groups agree, and one negligible beside one
#: material group is maximal disagreement however the division lands.
_NEGLIGIBLE_SECONDS = 0.01


def _cell_variants(row: dict) -> tuple[tuple[str, str, str], ...]:
    """The cell keys *row* contributes to: exact, then the two rollups."""
    device, media, embedder = row["device"], row["media_type"], row["embedder"]
    variants = [(device, media, embedder), (device, media, ""), (device, "", "")]
    seen: list[tuple[str, str, str]] = []
    for variant in variants:
        if variant not in seen:
            seen.append(variant)
    return tuple(seen)


def _run_count(step_samples: dict[str, list[dict]]) -> int:
    """How many distinct runs fed a cell (its best-covered step's sample count)."""
    return max((len(v) for v in step_samples.values()), default=0)


#: ``task -> cell -> step -> samples``, plus the same shape one level deeper for
#: sub-slot durations (``… -> step -> slot -> seconds``).
_CellSamples = dict[tuple[str, str, str], dict[str, list[dict]]]
_CellSlots = dict[tuple[str, str, str], dict[str, dict[str, list[float]]]]


def _bucket_rows(rows: Iterable[dict]) -> tuple[dict[str, _CellSamples], dict[str, _CellSlots]]:
    """Group normalized rows by task and cell, splitting out sub-slot durations.

    Each row lands in *every* cell it qualifies for — exact and both rollups —
    so the broader cells are fit from strictly more evidence than the narrow
    ones they back up.
    """
    by_cell: dict[str, _CellSamples] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    slot_secs: dict[str, _CellSlots] = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    for raw in rows:
        row = normalize_row(raw)
        if row is None:
            continue
        for cell in _cell_variants(row):
            if row["slot"]:
                slot_secs[row["task"]][cell][row["step"]][row["slot"]].append(row["seconds"])
            else:
                by_cell[row["task"]][cell][row["step"]].append(row)
    return by_cell, slot_secs


def _wildcard_axes(cell: tuple[str, str, str]) -> tuple[str, ...]:
    """Which of ``("media_type", "embedder")`` this cell key wildcards.

    An empty component in a stored key is a wildcard at lookup time, so it is
    also the axis along which that cell pools unlike rows. The exact cell
    wildcards nothing and is returned as an empty tuple.
    """
    return tuple(axis for axis, value in (("media_type", cell[1]), ("embedder", cell[2])) if not value)


def _rollup_is_contradicted(samples: list[dict], cell: tuple[str, str, str]) -> bool:
    """Whether *cell* pools groups whose own fits disagree about this step.

    **A rollup cell is only ever reached for a combination the sweep never
    measured.** :func:`vtscore.timing.profile.cell_keys` tries every more
    specific key first, and this fitter emits one for every combination it saw,
    so ``(device, media, *)`` serves only encoders that media type was never
    measured with, and ``(device, *, *)`` only media types the sweep never
    touched at all. The rollup's whole job is extrapolation — which is exactly
    why it must not be built by averaging things it has measured to be unlike.

    The test is the one the profile is actually used for: fit each pooled group
    on its own, ask each what the step costs at a size all of them cover, and
    compare. That works the same for a sloped step and a flat one, where the raw
    ``seconds / n`` ratio does not — two constant-cost groups sampled at
    different ``n`` look wildly divergent per item and are not.

    Groups closer than :data:`_MAX_ROLLUP_SPREAD` keep their pooled fit: a
    rollup that is merely imprecise still beats the shipped default, and dropping
    it would throw away the coverage the rollups exist to provide.
    """
    axes = _wildcard_axes(cell)
    if not axes:
        return False
    groups: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for sample in samples:
        groups[tuple(sample[axis] for axis in axes)].append(sample)
    if len(groups) < 2:
        # One group behind the rollup: it is a rename of the specific cell it
        # backs up, and repeats its claim rather than blending anything.
        return False

    probe_n = statistics.median([s["n"] for s in samples])
    predictions: list[float] = []
    for group in groups.values():
        coeffs = fit_step(group, byte_scaled=False)
        if coeffs is None:
            return False
        predictions.append(coeffs.seconds(probe_n))

    hi, lo = max(predictions), min(predictions)
    if hi <= _NEGLIGIBLE_SECONDS:
        # Every group says this step is free; there is nothing to be wrong about.
        return False
    if lo <= _NEGLIGIBLE_SECONDS:
        # One group free beside one that is not: the pooled number is wrong for
        # whichever of them the lookup lands on.
        return True
    return hi / lo > _MAX_ROLLUP_SPREAD


def _fit_task_cells(spec, cells: _CellSamples, slots: _CellSlots, min_samples: int) -> dict[str, Any]:
    """Fit every sufficiently-sampled cell of one task into its JSON entry."""
    cells_out: dict[str, Any] = {}
    for cell, step_samples in cells.items():
        runs = _run_count(step_samples)
        if runs < min_samples:
            continue
        # Withhold the **whole cell**, not just the offending step, when a step
        # only ever measured a cached path. Dropping one step looks like the
        # rollup case above but behaves quite differently: ``step_terms`` fills
        # a missing step from ``TaskSpec.default_terms``, which are documented
        # pseudo-seconds whose ratios alone are meaningful. Pairing one measured
        # step in real seconds with another in pseudo-seconds produces a weight
        # vector in no units at all — for ``dataset_open`` a measured 2.5 s of
        # ``items`` beside a 0.85 pseudo-second ``coverage`` prices the atlas at
        # a quarter of the bar by arithmetic accident. Falling back to the whole
        # shipped vector keeps the units consistent and the ratios considered.
        if any(cheap_branch_only(samples) for step, samples in step_samples.items() if step not in spec.byte_scaled):
            continue
        steps_out: dict[str, Any] = {}
        for step, samples in step_samples.items():
            byte_scaled = step in spec.byte_scaled
            if not byte_scaled and _rollup_is_contradicted(samples, cell):
                # Omitting the step is what "no measurement here" looks like to
                # the reader: `step_terms` falls that one step through to its
                # shipped default while the rest of this cell still applies. A
                # confidently wrong number cannot be fallen through to (#3522).
                continue
            coeffs = fit_step(samples, byte_scaled)
            if coeffs is not None:
                entry_json: dict[str, Any] = dict(coeffs.to_json())
                forks = fit_branches(samples, byte_scaled)
                if forks:
                    entry_json["branches"] = {name: c.to_json() for name, c in forks.items()}
                steps_out[step] = entry_json
        if not steps_out:
            continue
        entry: dict[str, Any] = {"samples": runs, "steps": steps_out}
        slots_out = _fit_slots(slots.get(cell, {}))
        if slots_out:
            entry["slots"] = slots_out
        cells_out["|".join(cell)] = entry
    return cells_out


def fit_profile(
    rows: Iterable[dict],
    *,
    min_samples: int = 2,
    generated_at: str = "",
    host: str = "",
    notes: str = "",
) -> dict[str, Any]:
    """Fit every ``(task, cell, step)`` present in *rows* into a profile document.

    Cells with fewer than *min_samples* runs are dropped: a single timing is an
    anecdote, and shipping it as a coefficient would replace a considered
    default with one machine's one bad afternoon. Dropped cells simply fall
    through to the next-broader cell — or, if none survives, to the built-in
    defaults, which is the correct outcome for a thin sweep.

    Returns a JSON-serializable dict ready to write and hand to
    ``VTSEARCH_TIMING_PROFILE``.
    """
    by_cell, slot_secs = _bucket_rows(rows)

    tasks_out: dict[str, Any] = {}
    for task, cells in by_cell.items():
        cells_out = _fit_task_cells(TASKS[task], cells, slot_secs.get(task, {}), min_samples)
        if cells_out:
            tasks_out[task] = {"cells": cells_out}

    doc: dict[str, Any] = {
        "schema": SCHEMA_NAME,
        "version": SCHEMA_VERSION,
        "tasks": tasks_out,
    }
    if generated_at:
        doc["generated_at"] = generated_at
    if host:
        doc["host"] = host
    if notes:
        doc["notes"] = notes
    return doc


def _fit_slots(step_slots: dict[str, dict[str, list[float]]]) -> dict[str, dict[str, float]]:
    """Normalize per-step sub-slot durations into shares of that step.

    Median seconds per slot (robust to the one run where the disk stalled), then
    normalized across the step. A slot that measured no time at all is dropped
    rather than shipped as a zero weight — the consumer merges what is here over
    its own defaults, so an unmeasured slot keeps a sane share instead of being
    told it is free.
    """
    out: dict[str, dict[str, float]] = {}
    for step, slots in step_slots.items():
        medians = {slot: statistics.median(v) for slot, v in slots.items() if v}
        total = sum(medians.values())
        if total <= 0:
            continue
        shares = {slot: round(value / total, 4) for slot, value in medians.items() if value > 0}
        if shares:
            out[step] = shares
    return out


#: An affine fit at or above this describes its own samples well enough that a
#: reader need not look further. Below it, the line is worth a second look.
_GOOD_R2 = 0.90

#: How a stored cell key is described in the coverage report, keyed by which of
#: ``(media_type, embedder)`` it wildcards. Written in the key's own ``|`` syntax
#: so a reader can match a report line against the profile JSON by eye.
_SPECIFICITY_LABELS: tuple[tuple[tuple[str, ...], str], ...] = (
    ((), "exact  (device|media|embedder)"),
    (("embedder",), "rollup (device|media|*)"),
    (("media_type",), "rollup (device|*|embedder)"),
    (("media_type", "embedder"), "rollup (device|*|*)"),
)


def _specificity_label(cell_key: str) -> str:
    """Describe one stored cell key by how much of its identity is wildcarded."""
    parts = (cell_key.split("|") + ["", ""])[:3]
    axes = _wildcard_axes((parts[0], parts[1], parts[2]))
    for candidate, label in _SPECIFICITY_LABELS:
        if candidate == axes:
            return label
    return "rollup (device|*|*)"  # pragma: no cover - _wildcard_axes is exhaustive


def _fit_quality(spec, cells: dict[str, Any]) -> str:
    """One line describing *how* a task's cells were fitted, and how well.

    Three outcomes, and the difference between them is the thing readers get
    wrong. A step with **no r²** is not a badly fitted line — it is not a line:
    either the step is byte-scaled (a per-MB rate, never regressed against
    ``n``) or ``fit_step`` found no credible slope and reported a median. Only
    the affine count has a goodness attached, so only it gets one here.

    A count of the affine fits *below* :data:`_GOOD_R2` is what makes the line
    actionable: a median of 0.99 over a hundred cells hides the six that the
    line does not describe, and those six are where a progress bar drifts.
    """
    affine: list[float] = []
    fallback = 0
    byte_rate = 0
    for cell in cells.values():
        for step, coeffs in cell.get("steps", {}).items():
            if step in spec.byte_scaled:
                byte_rate += 1
            elif "r2" in coeffs:
                affine.append(float(coeffs["r2"]))
            else:
                fallback += 1
    parts: list[str] = []
    if affine:
        poor = sum(1 for r2 in affine if r2 < _GOOD_R2)
        detail = f"median r² {statistics.median(affine):.2f}"
        if poor:
            detail += f", {poor} below {_GOOD_R2:.2f}"
        parts.append(f"{len(affine)} affine ({detail})")
    if fallback:
        parts.append(f"{fallback} median-fallback (no credible slope)")
    if byte_rate:
        parts.append(f"{byte_rate} byte-rate")
    return ", ".join(parts)


def _specificity_lines(spec, cells: dict[str, Any], buckets: _CellSamples) -> list[str]:
    """One :func:`_fit_quality` line per specificity level, most specific first.

    Ordered to match :func:`vtscore.timing.profile.cell_keys`, so the lines read
    down in the order a lookup tries them: the first level with a cell for a
    given media type and encoder is the one that will actually pace that job.

    A level with no surviving cell still gets a line when steps were *withheld*
    there, since "this rollup was refused" is exactly the fact a bare cell count
    cannot carry.
    """
    grouped: dict[str, dict[str, Any]] = defaultdict(dict)
    for key, cell in cells.items():
        grouped[_specificity_label(key)][key] = cell
    withheld = _withheld_by_specificity(spec, buckets)

    width = max(len(label) for _, label in _SPECIFICITY_LABELS)
    lines: list[str] = []
    for _, label in _SPECIFICITY_LABELS:
        level = grouped.get(label, {})
        dropped = withheld.get(label, 0)
        if not level and not dropped:
            continue
        parts = [f"{len(level)} cell{'' if len(level) == 1 else 's'}"]
        quality = _fit_quality(spec, level)
        if quality:
            parts.append(quality)
        if dropped:
            parts.append(f"{dropped} step{'' if dropped == 1 else 's'} withheld (pooled groups disagree)")
        lines.append(f"  {'':<16} {label:<{width}}  {', '.join(parts)}")
    return lines


def _withheld_by_specificity(spec, cells: _CellSamples) -> dict[str, int]:
    """Count the steps :func:`_rollup_is_contradicted` kept out, per specificity.

    Recomputed from the rows rather than recorded in the profile: a withheld
    step is precisely one the document does *not* contain, and inventing a
    schema field to say so would make every reader parse a negative claim. The
    tuning script runs this once at the end of a sweep, so the second pass over
    the buckets costs nothing anyone waits on.
    """
    out: dict[str, int] = defaultdict(int)
    for cell, step_samples in cells.items():
        for step, samples in step_samples.items():
            if step not in spec.byte_scaled and _rollup_is_contradicted(samples, cell):
                out[_specificity_label("|".join(cell))] += 1
    return dict(out)


#: A task whose whole run is shorter than this, at the sizes the sweep visited,
#: cannot be usefully paced. The bar is then decided by the ranking of a handful
#: of tiny numbers, and an absolute error far too small to fit reorders them —
#: which is the state #3596 measured on ``text_sort``: three steps totalling
#: 0.9 s, all three profiles ranking them differently, and every arm sitting at
#: 0.80-0.85 bar error while its *step* error stayed near 0.2.
#:
#: One second is where a bar stops being a bar: below it the job is a flash, and
#: whatever slice each step drew never rendered at a size anybody read. It is a
#: reporting threshold only — nothing branches on it — so it is deliberately a
#: round number rather than a measured one.
_UNPACEABLE_TOTAL_SECONDS = 1.0


def _typical_seconds(samples: list[dict]) -> float:
    """Median seconds this step took, over the population a fit would price.

    Warm runs when there are any, mirroring :func:`fit_step`'s holdout, because
    the warm population is what a served process spends nearly all of its runs
    in. Raw: the deferred floor is reported separately, since conflating "this
    step measured zero" with "the profile stores 0.5 for it" is the confusion
    these lines exist to undo.
    """
    warm = [s["seconds"] for s in samples if not s.get("cold")]
    seconds = warm or [s["seconds"] for s in samples]
    return statistics.median(seconds) if seconds else 0.0


def _pace_lines(spec, samples_by_step: dict[str, list[dict]]) -> list[str]:
    """Say when a task's own measurements cannot pace its bar.

    Two things a sample count actively misreports, both measured on
    ``text_sort`` in #3521's sweep, where 288 step-samples read as the
    best-covered task in the report:

    **The whole task is too short.** Its three steps total 0.9 s warm, so the
    bar is settled by which of three sub-second numbers comes out largest, and
    the fitted profiles disagreed with each other about that ordering while
    predicting each step to within ~20 %. Coefficients cannot fix an ordering
    that noise decides; the line says so rather than letting a full sample count
    imply the opposite.

    **A step measured free is not priced free.** When the warm runs measure a
    step at exactly zero and a cold run measured it as real,
    :func:`_deferred_cost_floor` keeps a floor for it — right for a step that
    might yet be paid, and on a short task it is most of the predicted total,
    so it redistributes the bar for every warm run. Naming the floor beside the
    measurement is what makes that visible; a task whose caller can tell the
    two branches apart should pass the step to
    :func:`vtscore.timing.step_weights` as skipped instead (``text_sort`` now
    does).
    """
    steps = {step: samples for step, samples in samples_by_step.items() if step in spec.steps}
    if not steps:
        return []
    lines: list[str] = []
    typical = {step: _typical_seconds(samples) for step, samples in steps.items()}
    total = sum(typical.values())
    if 0 < total < _UNPACEABLE_TOTAL_SECONDS:
        breakdown = ", ".join(f"{step} {typical[step]:.2f}" for step in spec.steps if step in typical)
        lines.append(
            f"  {'':<16} TOO SHORT TO PACE: a typical run totals {total:.2f} s at the swept "
            f"sizes ({breakdown}) — the bar is decided by which of these is largest, which "
            f"is below the error any fit of them carries"
        )
    for step in spec.steps:
        samples = steps.get(step)
        if not samples or step in spec.byte_scaled:
            continue
        cold = [s for s in samples if s.get("cold")]
        floor = _deferred_cost_floor(typical[step], cold)
        if floor <= typical[step]:
            continue
        warm_zero = sum(1 for s in samples if not s.get("cold") and s["seconds"] <= 0)
        share = floor / (total - typical[step] + floor) if total - typical[step] + floor > 0 else 1.0
        lines.append(
            f"  {'':<16} {step}: measured 0.00 s on {warm_zero} of {len(samples)} runs and real on "
            f"{len(cold)} — deferred, so it is priced at the {floor:.2f} s floor, {share:.0%} of "
            f"the predicted total; a caller that knows the step will be skipped should say so"
        )
    return lines


def _branch_lines(spec, samples_by_step: dict[str, list[dict]]) -> list[str]:
    """Name every step whose runs only ever took a cached path.

    The line the issue that prompted this asked for: *"coverage_report should be
    able to say 'this cell only ever measured the warm path', because a cell
    count cannot"* (#3521). A step measured 16 times on the cheap branch and
    never once on the dear one reads, in every other column of this report, as
    the best-covered thing in the sweep.

    Reported per ``(step, branch)`` across the cells rather than per cell: an
    admin who has just been told a whole task fell back to its defaults needs to
    know *what to drive differently*, and the branch name is the actionable
    half. The per-cell breakdown is in the rows.

    Counts come from the **normalized rows**, not from the cell buckets, because
    every row is bucketed into three cells at three specificities and counting
    the buckets would report each run three times.
    """
    census: dict[tuple[str, str], int] = defaultdict(int)
    for step, samples in samples_by_step.items():
        if step in spec.byte_scaled or not cheap_branch_only(samples):
            continue
        for sample in samples:
            census[(step, sample["branch"])] += 1
    lines: list[str] = []
    for (step, branch), count in sorted(census.items()):
        lines.append(
            f"  {'':<16} {step}: {count} run{'' if count == 1 else 's'}, all '{branch}' — "
            f"the branch a user waits on was never measured, so this cell keeps its defaults"
        )
    return lines


def _priced_branch_lines(cells: dict[str, Any]) -> list[str]:
    """Name every step this task priced **per branch**, and with which branches.

    The counterpart to :func:`_branch_lines`. That one reports the step nobody
    measured on the dear path; this one reports the step measured on *both*, and
    so the step whose pacing now depends on the caller naming its branch. An
    admin reading a sweep needs to be able to tell the two apart, since the
    remedy for one (drive the other branch) is exactly what produces the other.
    """
    priced: dict[str, set[str]] = defaultdict(set)
    counts: dict[str, int] = defaultdict(int)
    for cell in cells.values():
        for step, coeffs in cell.get("steps", {}).items():
            forks = coeffs.get("branches") if isinstance(coeffs, dict) else None
            if isinstance(forks, dict) and forks:
                priced[step].update(str(b) for b in forks)
                counts[step] += 1
    return [
        f"  {'':<16} {step}: priced per branch in {counts[step]} cell{'' if counts[step] == 1 else 's'} "
        f"({', '.join(sorted(branches))}) — a caller that names its branch is paced from that branch alone"
        for step, branches in sorted(priced.items())
    ]


def coverage_report(rows: Iterable[dict], profile: dict[str, Any]) -> list[str]:
    """Human-readable lines describing what the sweep did and did not cover.

    Printed by the tuning script so a thin sweep is *visible* rather than
    quietly shipping a profile that improves two cells and leaves the rest to
    the defaults. Silent partial coverage is how a tuning run gets mistaken for
    a tuned system.

    Cell counts alone cannot say that, though: a task can be measured in fifty
    cells and still have every one of them fall back to a median, which reads
    as full coverage and paces like none. So each measured task also reports
    :func:`_fit_quality` — the r² ``StepCoeffs`` has carried since #3334 and
    which, until #3345, nothing in the tree ever read.

    That quality line is then **split by specificity**, because pooling the
    levels hides the one an admin most needs to see. #3345 measured, on one
    sweep's own rows, exact cells at r² 1.00 / 3 % error against
    ``(device, *, *)`` cells at 0.29 / 50 %; a single median over both reads
    like the exact number and is used like the rollup one, since the rollup is
    the cell guaranteed to match. Split, "5 cells" resolves into how many are
    measurements and how many are fallbacks (#3522).

    A third thing a count cannot say is *which branch* the runs took. A step
    whose every run read a cache is not a well-covered step; it is an unmeasured
    one wearing a full sample count, and it is the state that made #3345's sweep
    price a 7.7 s atlas rebuild at 2 % of the bar. :func:`_branch_lines`
    names those explicitly (#3521), and :func:`_priced_branch_lines` names the
    steps measured on *both* paths, whose pacing now depends on the caller
    saying which one it is taking (#3594).

    A fourth is whether the task is long enough for *any* coefficients to pace.
    ``text_sort``'s 288 step-samples are the largest count in #3521's report and
    describe a 0.9 s job whose bar every arm of that study got 0.80-0.85 wrong;
    :func:`_pace_lines` says that outright, along with the deferred floor that
    causes most of it (#3596).
    """
    rows = list(rows)
    normalized = [n for n in (normalize_row(r) for r in rows) if n is not None]
    seen_tasks = {n["task"] for n in normalized}
    by_step: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for sample in normalized:
        if not sample["slot"]:
            by_step[sample["task"]][sample["step"]].append(sample)
    by_cell, _ = _bucket_rows(rows)
    lines: list[str] = []
    for task in TASKS:
        cells = profile.get("tasks", {}).get(task, {}).get("cells", {})
        if cells:
            samples = sum(int(c.get("samples", 0)) for c in cells.values())
            lines.append(f"  {task:<16} {len(cells)} cells, {samples} step-samples")
            lines.extend(_specificity_lines(TASKS[task], cells, by_cell.get(task, {})))
            lines.extend(_branch_lines(TASKS[task], by_step.get(task, {})))
            lines.extend(_priced_branch_lines(cells))
            lines.extend(_pace_lines(TASKS[task], by_step.get(task, {})))
        elif task in seen_tasks:
            branch_lines = _branch_lines(TASKS[task], by_step.get(task, {}))
            if branch_lines:
                lines.append(f"  {task:<16} measured on the cached path only — using built-in defaults")
                lines.extend(branch_lines)
            else:
                lines.append(f"  {task:<16} measured but too few runs to fit — using built-in defaults")
        else:
            lines.append(f"  {task:<16} NOT MEASURED — using built-in defaults")
    return lines
