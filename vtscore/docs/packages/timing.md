# `vtscore.timing`

How long each step of a long-running task will take, measured rather
than guessed.

Every long-running VTSearch operation - a dataset load, a detector load,
a text sort, a Find, a train-and-score, a promote - reports progress as
`step` / `total_steps` and paces its unified bar with a per-step
**weight vector** (`ProgressTracker.set_step_weights`). Those vectors
used to be hand-guessed constants sitting next to each task's code. A
guess that is wrong in the same direction for a whole job is exactly
what makes a progress bar race one phase, crawl the next, and walk its
ETA *upward* while the user watches.

This package replaces the guesses with a per-environment cost model:
a deployment measures itself once, and every instance in that
environment predicts its own timings thereafter.

Related docs: [`concurrency.md`](concurrency.md) for `ProgressTracker`
and the bar these weights drive.

## Contents

| Module | Concern |
|--------|---------|
| `vtscore/timing/tasks.py` | `TASKS` / `TaskSpec` - the canonical registry of task families and their ordered steps |
| `vtscore/timing/profile.py` | Load, resolve and apply a profile: `step_weights`, `step_terms`, `slot_shares`, `active_profile` |
| `vtscore/timing/recorder.py` | Env-gated recorder that measures what each step really took |
| `vtscore/timing/fit.py` | Turn recorded timings into a profile document (the writer for the format `profile.py` reads) |

---

## The cost model

Each step gets an affine cost:

```
T_step ≈ a + b · n + per_mb · archive_mb
```

`n` is the task's natural scale variable - items to embed, labels to
train on, medias to score. `archive_mb` covers the byte-scaled phases of
a download.

Coefficients are keyed by a **cell**: `(device, media_type, embedder)`.
That granularity is the point. The same step costs wildly different
amounts on a V100 versus a laptop CPU, and on 200-character texts versus
30-second videos; a single global constant cannot be right for both.

A step that **forks** carries a second set of coefficients per branch,
nested under the step's own `branches` key:

```json
"coverage": {
  "a": 0.2, "b": 0.0026,
  "branches": {
    "restored": {"a": 0.009},
    "rebuilt":  {"a": 0.2, "b": 0.0026}
  }
}
```

The cell is not keyed by branch, because the branch is not a property of
the environment - it is a property of the run, decided while the job is
already under way. So the split lives inside the step, and a caller that
knows which path it is on passes `branch=` to `step_weights` /
`step_terms` to be priced from it. #3521 measured a coverage atlas
restoring in 0.011 s and rebuilding in 7.7 s on the same 2954-item
dataset; held out against each other, a profile fitted from restores
alone put **0.94 of the bar** in the wrong step on a rebuild, and one
fitted from rebuilds put 0.49 in the wrong step on a restore. No single
number wins both columns, which is why there are two (#3594).

The step's top-level coefficients are unchanged - still the dear branch
- so a caller that cannot say which path it is on, a hand-written
profile, and an older build all behave exactly as before. That is also
why the schema version did not move: `branches` is additive and safely
ignorable, and bumping it would have made every new profile unreadable
to the builds that can still use it.

## The three-layer resolution

A cell resolves most-specific-first:

1. **The admin profile** - a JSON file named by
   `VTSEARCH_TIMING_PROFILE`, produced by
   `scripts/profiling/tune_timing_profile.py` on the hardware that will
   actually serve the app.
2. **The shipped defaults** in `vtscore/timing/tasks.py` (and, for
   `dataset_load`, the calibrated table in
   `vtscore/datasets/stages/_load_cost_model.py`). These reproduce the
   pre-profile hand-tuned weights *exactly*, so an instance with no
   profile paces as it always did.
3. **Equal weighting**, if the task is unknown entirely.

Default terms are **pseudo-seconds**: only their ratios are meaningful,
because nobody measured them. A profile replaces them with real seconds,
which is what makes the ETA stop drifting.

Nothing persists at runtime and nothing is cached across processes: the
profile is read once per process at first use, and `reload_profile()`
re-reads it.

---

## Task registry

Every task driving a `step`/`total_steps` bar registers a `TaskSpec` in
`TASKS`. The registry is the shared vocabulary between three parties
that would otherwise drift apart: the **task code**, which needs one
weight per tracker step; the **recorder**, which must label a measured
duration with a step name; and the **tuning script**, which fits per
step and writes the profile JSON keyed by those same names.

| Field | Meaning |
|-------|---------|
| `name` | Stable identifier - the profile JSON's task key, the label in recorded rows, and the `--tasks` selector. **Never rename one** without migrating the profiles admins have already generated |
| `steps` | Ordered cost-*phase* names; profile coefficients are keyed by these |
| `step_index` | 1-based tracker step each phase reports against, parallel to `steps` |
| `tracker_steps` | How many step numbers the task reports - the length of the weight vector |
| `scale` | Human description of what `n` counts |
| `byte_scaled` | Which phases get a per-MB rate instead of a per-item slope |

Registered today: `dataset_load`, `dataset_open`, `dataset_promote`,
`dataset_stage`, `detector_load`, `text_sort`, `find`,
`train_and_score`.

**Phases versus tracker steps.** Usually they are the same and
`step_index` is just `(1, 2, 3, …)`. A task may model one step as
several phases that scale differently - `dataset_load`'s step 1 covers
both the network transfer and the archive unpack, both byte-scaled but
at very different rates - in which case the phases share a tracker step
and `step_weights` sums their predicted seconds back into that slot.

`dataset_load` deliberately carries **no** default terms: its shipped
model is the measured affine table in `_load_cost_model`, which is
already `n`-aware per cell and better than any flat vector.

Adding a long-running task means adding a `TaskSpec` here, then calling
`step_weights(...)` at the task's entry point instead of writing a
literal vector.

---

## Using it

```python
from vtscore.timing import step_weights

weights = step_weights(
    "text_sort",
    device=device, media_type="image", embedder="siglip",
    n=len(medias),
    fallback=[0.2, 0.8],
)
if weights:
    tracker.set_step_weights(weights)
```

The vector has one entry per tracker step and sums to 1, ready for
`set_step_weights`. Pass a `fallback` - `step_weights` returns it when
the task is unknown or nothing resolves.

### Steps this run will skip

A cost model answers "how long does this step take". It cannot answer
"does this step happen at all", and where a step forks on process state
the second question is the one that decides the bar. A text sort's
`load_model` is seconds on a process's first sort and **exactly zero**
on every later one, so no single coefficient paces both branches: fitted
from the warm runs the step is free (and gets floored back up, below),
fitted from the cold one it eats a bar that will not move. #3596
measured every arm of #3521's study - two fitted profiles and the
shipped defaults alike - at 0.80-0.85 bar error on `text_sort` for
exactly this reason, while their *step* error stayed near 0.2.

The caller usually knows which branch it is on before it starts. Name
the steps that will not run and they are priced at zero for this run:

```python
weights = step_weights(
    "text_sort",
    media_type="image", embedder="siglip", n=len(medias),
    skip_steps=() if encoder_is_cold else ("load_model",),
)
```

This needs no measurement and no branch axis in the profile format - a
step that does not run costs nothing, and that is knowable in advance
where a step's *cost* is not. Steps whose cost merely varies by branch
(a coverage atlas restored versus rebuilt) are the other half of the
problem, and they do need that axis - see below.

### Steps whose cost varies by branch

A step that runs either way but costs two different things cannot be
skipped, and no coefficient describes both. Name the branch instead and
it is priced from that branch's own
[coefficients](#the-cost-model):

```python
weights = step_weights(
    "dataset_open",
    media_type=media_type, embedder=embedder, n=len(medias),
    branch="restored",          # or a {step: branch} mapping
)
```

**A task whose expensive step forks this way should call `step_weights`
twice**: once on the way in with whatever it can guess, and again the
moment it knows, before the expensive part runs. The dataset-open route
is the worked example - it weights the bar from the branch the *last*
open of that dataset took (remembered on the registry entry, since
whether a pickle carries a restorable atlas is a durable fact about the
file), then re-weights the moment `restore_coverage_atlas_from_cache`
has answered. Re-weighting mid-job only ever moves the bar forward; the
tracker clamps its overall fraction to be monotonic.

| Function | Description |
|----------|-------------|
| `step_weights(task, *, device, media_type, embedder, n, size_mb, skip_steps, branch, fallback)` | Normalised per-tracker-step weights, or *fallback* |
| `step_terms(...)` | The same prediction before normalisation - raw predicted seconds per phase (takes `skip_steps` and `branch` too) |
| `branch=` (on both) | Which path a forking step is taking on *this* run: a branch name, or a `{step: branch}` mapping. Sharpens the answer where the profile has that branch and changes nothing where it does not, so a caller that knows should always say |
| `slot_shares(task, step, ...)` | Measured sub-stage shares *within* one step, for steps that pace several ordered sub-stages behind one number (today only the dataset load's `finalize`). Raw weights; the consumer normalises |
| `profile_covers(task)` | Whether the active profile has any measured cell for *task*. Public API with no in-repo caller - for out-of-tree callers that want to branch on coverage before asking for weights |
| `active_profile()` / `reload_profile(path=None)` | The parsed profile; re-read it |
| `known_tasks()` / `task_spec(name)` | Registry lookups |
| `cell_keys(device, media_type, embedder)` / `normalize_device(device)` | Cell-key resolution, most specific first |
| `note_branch(step, branch)` | Name the path *step* took on the run recording this thread. A no-op when nothing is recording, so ordinary product code calls it unconditionally |
| `note_no_encoder_load()` | Declare that this run instantiated no encoder, so it does not claim the residency key the next run needs |

---

## Recording

Arm the recorder by pointing `VTSEARCH_TIMING_RECORD` at a JSONL path.
Each task wrapped in `record_task` then appends one row per step:

```json
{"task": "text_sort", "device": "cuda", "cuml": true, "media_type": "image",
 "embedder": "siglip", "n": 12403, "size_mb": 0.0, "step": "score",
 "seconds": 1.83, "ok": true, "cold_model": false}
```

`cold_model` says whether this run was the first in the process to need
its `(media_type, embedder)` encoder, and so the one that paid to
download and instantiate it. Without it a once-per-process cost is
unfittable: a text sort's model load measures 15 s once and 0 s on the
next 47, and a fitter that cannot separate the two populations medians
them into "free". Only tasks whose `TaskSpec` declares `loads_encoder`
take part - a `dataset_open` reads a pkl and touches no encoder, so it
neither carries the field nor claims a key that the genuinely cold sort
behind it needs.

### Which branch a step took

`cold_model` is a property of the **run**. A step's cost can also fork on
a cache that is neither the encoder nor scoped to the process, and those
forks are recorded per **step**, in a `branch` field, by the code that
chooses them (`note_branch`):

| step | cheap branch | dear branch |
|---|---|---|
| `dataset_open` · `coverage` | `restored` (the atlas cached in the pickle), `deferred` (past the auto-build threshold) | `rebuilt` |
| `dataset_load` / `dataset_stage` · `embed` | `cached` (the demo embeddings pkl) | `fresh` |

The vocabulary is `CHEAP_BRANCHES` / `DEAR_BRANCHES` in `tasks.py`, and
both the fitter and the lookup read it: a forked step is priced from the
runs that did the work, a step whose runs *all* read a cache withholds
its whole cell so the task keeps its shipped defaults, and a step
measured on both paths additionally carries
[per-branch coefficients](#the-cost-model) for a caller that can name
its own branch. A row without the field is not a claim that the step
never forks - unmarked rows fit as they always did.

This exists because #3345's sweep opened 16 datasets and restored the
cached atlas on every one, recording 0.008-0.016 s at every `n` from
245 to 2954. That is a correct measurement of a branch nobody waits on,
and the profile fitted from it gave 2 % of the bar to a step whose
shipped default is 0.85 because a rebuild takes minutes. A run count
cannot say that, which is why `coverage_report` now does (#3521).

A run that satisfied itself from a cache also calls
`note_no_encoder_load()`: it instantiated no model, so it must not claim
the residency key that the next run - the one that really pays the load
- needs in order to be written cold.

Because the recorder sits behind an env var, an admin has two ways to
gather data and both produce the same file:

- **Drive it.** Run the tuning script, which exercises each task family
  against exemplar datasets with the recorder armed. It also *arranges*
  the dear branch where that is cheap and non-destructive: `--cold-embed`
  (on by default) clears the demo embeddings cache before each measured
  import, and `--cold-atlas` rebuilds each dataset's coverage atlas
  through the on-demand endpoint rather than editing anybody's pickle.
- **Watch it.** Set `VTSEARCH_TIMING_RECORD` on the real server and let
  real users generate the timings. This measures the production mix
  directly - the datasets people actually load, at the sizes they
  actually are - which no synthetic sweep reproduces.

When disarmed the cost is one `os.environ` lookup per task and a couple
of no-op method calls: no tracker subscription, no file handle.

### Two recorders, side by side

The dataset-load pipeline carries an older, richer recorder
(`vtscore/datasets/stages/_load_profiler.py`) that additionally
distinguishes cold from cached downloads and splits finalize into its
sub-slots. Both run on the same load. They answer different questions,
are armed by different env vars, and write different files - and the
fitter reads both row shapes, so a pre-existing dataset-load calibration
sweep folds into a new profile rather than being re-measured.

---

## Fitting

`vtscore/timing/fit.py` is the writer for the format `profile.py` reads.
It lives next to the reader so the two cannot drift: a schema change has
to be made in one directory or it will not round-trip.
`normalize_row` flattens both recorder shapes into one.

The fit is deliberately plain. Per `(task, cell, step)`:

- **Byte-scaled steps** (a download and its unpack) get a per-MB rate:
  the median of `seconds / archive_mb`. Regressing these against item
  count would ask `n` to explain something it cannot see - 500 videos
  and 500 text files are the same `n` and two orders of magnitude apart
  in bytes.
- **Everything else** gets ordinary least squares against `n`: the
  intercept is what the step costs at all (loading an encoder, opening a
  file) and the slope is what each additional item adds.
- **Cold runs are held out** when the warm ones can carry the regression
  alone. A cold run pays once-per-process costs no later run repeats -
  the encoder download, the CUDA context, the first forward pass - and it
  always lands at whichever `n` ran first, so it has enormous leverage on
  the slope. The holdout stops short of costing a cell its only line:
  below two distinct warm sizes no slope is estimable, and a two-run
  sweep's first run is always the cold one. When the warm runs then
  measure a step as *exactly* free while a cold one measured it as real,
  the step is not free here but deferred, and it keeps a small floor
  rather than 0 so the bar still shows a slice for it. The floor is a
  guard against a confident zero, not a cost model, and on a short task
  it is most of the predicted total - which is why a caller that can
  tell the branches apart should pass the step as
  [skipped](#steps-this-run-will-skip) rather than lean on it.
- A fit with no spread in `n`, or one that comes back with a **negative**
  slope (noise beating signal on a short step), collapses to the median
  seconds with no slope. A confidently wrong slope extrapolates badly at
  sizes the sweep never visited; a flat median merely stops improving.
- **A step measured on both paths is also fitted per branch** and stored
  under the step's `branches` key, each branch fitted from its own rows
  by the same `fit_step` (so the cold/warm holdout and the median
  fallback apply within a branch as they do without one). A step whose
  runs all took *one* branch gets no split: one branch measured is not
  evidence about the branch nobody ran, and writing it as a split would
  suggest the profile knows something it does not. `--cold-atlas` is
  what produces both branches for `dataset_open` in a single sweep.

### Is the fit any good?

`affine_fit` returns an OLS r² and `StepCoeffs` keeps it, so a profile can
be read for whether its cost model describes the deployment it was measured
on. `tune_timing_profile.py`'s coverage report prints it per task:

```
  dataset_load     5 cells, 24 step-samples
                   exact  (device|media|embedder)  2 cells, 6 affine (median r² 1.00)
                   rollup (device|media|*)         2 cells, 4 affine (median r² 0.98, 1 below 0.90), 4 byte-rate
                   rollup (device|*|*)             1 cell, 2 affine (median r² 0.29, 2 below 0.90), 1 step withheld (pooled groups disagree)
```

The counts are **split by specificity**, in the order
[`cell_keys`](#the-three-layer-resolution) tries them, because pooling the levels hides the
one that matters most: `(device, *, *)` is the cell guaranteed to match, and
it is the weakest. Read down the block and the first level with a cell for
your media type and encoder is the one that will pace that job. A bare
"5 cells" cannot tell you whether a sweep bought five measurements or one
measurement and four fallbacks.

Read the three counts before the r². **A missing r² is not a bad fit** - it
means the step was not fitted as a line at all, which happens two ways: a
byte-scaled step is a per-MB rate by design, and the median fallback above
declined to draw one. `to_json` omits the key rather than writing a
misleading zero, and `from_json` restores it as `NaN`.

**A sweep at one dataset size produces no r² anywhere.** One size means one
`n` per cell, `affine_fit` finds no x-variance and returns `(mean, 0, 0)`,
and every step lands in the median fallback. Drive several sizes per
(media type, embedder) - four is comfortable - or the profile is a table of
averages with no scaling term and nothing to judge it by.

Two cautions from the measurements in
[#3345](../../../docs/experiments/2026-09-02-timing-r2-3345/REPORT.md):

- **A high r² is not a well-paced step.** r² asks whether the points lie on
  the line; a bar wants to know how far off the prediction is. They can
  disagree, and the second is the one that shows.
- **The rollup cells are much weaker than the exact ones**, which matters
  because the least-specific cell is the one that always matches. Same
  sweep, same rows: exact cells fit to r² 1.00 with 3 % prediction error,
  and the `(device, *, *)` rollup to r² 0.29 with 50 % (162 % on one arm).

### Contradicted rollups are not emitted

A rollup is only ever *reached* for a combination the sweep never measured:
`cell_keys` tries every more specific key first, and the fitter emits a cell
for everything it saw. So `(device, media, *)` serves only encoders that
media type was never measured with, and `(device, *, *)` only media types the
sweep never touched at all. Extrapolation is the rollup's whole job - which
is why it must not be built by averaging rows measured to be unlike.

Before fitting a rollup step, `fit.py` fits each pooled group on its own and
asks what the step costs at a size all of them cover. If the cheapest and
dearest answers differ by more than `_MAX_ROLLUP_SPREAD` (3x), that step is
**left out of the cell**, and `step_terms` falls it through to the shipped
default while the rest of the cell still applies. #3345's measured case is
the one this catches: `(cuda+cuml, *, *)` fitting a single slope through an
image import at 0.014 s/item and an audio one at 0.102.

The threshold sits well above the spread a healthy rollup shows - that same
study's media rollups ran at 9 % error - so it fires on disagreement rather
than on scatter. A merely imprecise rollup still beats the shipped default
and is kept; a rollup with one group behind it is a rename of the cell it
backs up and is never suppressed. The withheld count is printed in the
coverage report, because a step the profile does *not* contain is invisible
to anything that reads the profile.

### When the task is too short to pace at all

A sample count is also silent about whether there is anything to pace. The
coverage report says it directly:

```
  text_sort        3 cells, 288 step-samples
                   TOO SHORT TO PACE: a typical run totals 0.90 s at the swept sizes (load_model 0.00, embed_query 0.05, score 0.85) — the bar is decided by which of these is largest, which is below the error any fit of them carries
                   load_model: measured 0.00 s on 47 of 48 runs and real on 1 — deferred, so it is priced at the 0.50 s floor, 36% of the predicted total; a caller that knows the step will be skipped should say so
```

Those 288 step-samples were the largest count in #3521's report and describe
a job whose bar every arm of that study got 0.80-0.85 wrong. The first line
is why coefficients cannot fix it: when the whole run is under a second, the
ranking of three tiny numbers decides the bar and an absolute error far too
small to fit reorders them. The second names the mechanism behind most of it
and the remedy - the deferred floor, and the
[skip](#steps-this-run-will-skip) that makes it moot on the warm branch.
