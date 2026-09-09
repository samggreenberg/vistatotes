# `vtscore.eval` - Offline evaluation

Reproducible evaluation of text-sort and learned-sort quality on demo
datasets, plus a voting-iteration simulator that tracks classification
cost as a function of how many labels have been cast. Everything in
this package is computation only - the numbers come out as dataclasses
and DataFrames. Rendering them to PNGs lives in `vtscore.eval.visualize`,
which is the one module here that imports matplotlib.

The package wraps the demo datasets registered under
`vtscore.datasets` with text descriptions (the queries a user would
type in the Text Sort box) so a single CLI invocation can sweep across
audio, image, video, and paragraph datasets and report comparable
metrics. The full user-facing guide is at
[`docs/EVAL.md`](../../../docs/EVAL.md); this doc covers the library
surface a programmatic consumer calls into.

## Contents

**Core harness** - the parts every eval run touches.

| Module                                  | Concern                                                  |
|-----------------------------------------|----------------------------------------------------------|
| `vtscore/eval/config.py`                | `EvalQuery` dataclass and `EVAL_DATASETS` registry       |
| `vtscore/eval/metrics.py`               | `QueryMetrics`, `LearnedSortMetrics`, `DatasetResult`, metric functions |
| `vtscore/eval/labels.py`                | The ground-truth membership test shared by every harness |
| `vtscore/eval/runner.py`                | `eval_text_sort`, `eval_learned_sort`, `run_eval`, `format_results_json` |
| `vtscore/eval/visualize.py`             | `plot_eval_results`, `plot_voting_iterations` (matplotlib) |
| `vtscore/eval/__main__.py`              | CLI for `python -m vtscore.eval`                         |

**Voting-iterations simulation** - the autopilot cost model.

| Module                                  | Concern                                                  |
|-----------------------------------------|----------------------------------------------------------|
| `vtscore/eval/voting_iterations.py`     | Per-step cost simulator and multi-dataset sweep          |
| `vtscore/eval/al_strategies.py`         | The vote-order simulation: `STRATEGIES`, `ALContext`, `select_next` |
| `vtscore/eval/al_benchmark.py`          | Hermetic harness around the voting-iterations eval       |
| `vtscore/eval/autopilot_flow.py`        | The app's Autopilot phase machine, **ported** from TypeScript |
| `vtscore/eval/seed_scores.py`           | Text-sort seed scores that start the simulation          |

**Experiment arms** - each answers one study; none is the default arm.

| Module                                  | Concern                                                  |
|-----------------------------------------|----------------------------------------------------------|
| `vtscore/eval/patch_styles.py`          | Detection-style abstraction for the Max-Patch experiment |
| `vtscore/eval/cut_rules.py`             | Score-cut rules and their oracle decomposition           |
| `vtscore/eval/evt_mixture.py`           | Gumbel + Normal score mixture behind the `gumbel_*` cut rules |
| `vtscore/eval/calibration_metrics.py`   | Pure-numpy calibration metrics and pooling variants      |
| `vtscore/eval/sweep_trainers.py`        | Standalone-estimator registry for the label-curve and timing sweeps |
| `vtscore/eval/label_curve.py`           | MLP-vs-SVM label-curve sweep (`run_label_curve_eval`)    |
| `vtscore/eval/label_curve_main.py`      | CLI for the label-curve sweep                            |
| `vtscore/eval/timing_benchmark.py`      | GPU microbenchmark: MLP (torch) vs SVM (cuML)            |

The package `__init__.py` re-exports the main entry points:

```python
from vtscore.eval import (
    EVAL_DATASETS, EvalQuery,
    STRATEGIES, ALContext, available_strategies, select_next,
    compute_metrics, run_eval,
    simulate_voting_iterations, run_voting_iterations_eval,
    run_voting_iterations_eval_from_pickles,
    plot_eval_results, plot_voting_iterations,
)
```

> **The default arm is the shipped algorithm.** `vtscore.eval` measures
> *deviations* from what the app does, which only means anything if the
> no-explicit-arm path is what the app actually runs. Most of the
> harness stays honest by **delegating** into app code
> (`MaxPatchStyle` calls `pool_box_from_media` / `media_score_rows`
> rather than re-deriving them). Two things can't delegate and so can
> drift: `autopilot_flow.py`, which is ported because the original is
> TypeScript, and the places where "no arm" resolves to the app's
> current default. `scripts/check-eval-app-sync.py` (a `./run-tests.sh`
> gate) pins a digest of each mirrored surface **on both sides** - the app
> code and the harness copy of it - and fails when either moves.
> See the "Eval Default Arm IS the App" rule in `CLAUDE.md`.

---

## Data classes

### `EvalQuery`

`vtscore/eval/config.py`. One natural-language query targeting one
ground-truth category.

```python
@dataclass
class EvalQuery:
    text: str               # what a user would type, e.g. "a dog barking"
    target_category: str    # the ground-truth category, e.g. "dog"
```

`EVAL_DATASETS` is a `dict[str, dict]` keyed by demo dataset id
(`esc50_s`, `caltech101_m`, `20newsgroups_l`, `ucf101_s`, ...). Each
value is `{"demo_dataset": "...", "queries": list[EvalQuery]}`. The
registry covers all 50 ESC-50 categories, 25 Caltech-101 / Caltech-256
categories, 15 of the 20-Newsgroups categories, and 10 UCF-101
categories - see `vtscore/eval/config.py` for the full lists.

### `QueryMetrics`

`vtscore/eval/metrics.py`. One text-sort query's results.

```python
@dataclass
class QueryMetrics:
    query_text: str
    target_category: str
    average_precision: float
    precision_at_k: dict[int, float]   # default factory: {}
    recall_at_k:    dict[int, float]   # default factory: {}
    num_relevant: int = 0
    num_total: int = 0
    elapsed_seconds: float = 0.0
```

### `LearnedSortMetrics`

`vtscore/eval/metrics.py`. One learned-sort fold's results.

```python
@dataclass
class LearnedSortMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    num_train: int
    num_test: int
    target_category: str = ""
    elapsed_seconds: float = 0.0
```

### `DatasetResult`

`vtscore/eval/metrics.py`. Aggregated results for one eval dataset.

```python
@dataclass
class DatasetResult:
    dataset_id: str
    media_type: str
    text_sort: list[QueryMetrics]            # default []
    learned_sort: list[LearnedSortMetrics]   # default []

    @property
    def mean_average_precision(self) -> float: ...   # mAP across text_sort
    @property
    def mean_learned_f1(self) -> float: ...          # mean F1 across folds

    def to_dict(self) -> dict[str, Any]: ...
```

`to_dict()` produces the JSON shape `format_results_json` emits.

---

## Metrics

`vtscore/eval/metrics.py` provides four pure functions on lists of ids
and labels. None of them depend on a dataset context, embedder, or
sort - they take what the runner produces and return numbers.

| Function                                                          | Behaviour                                                 |
|-------------------------------------------------------------------|-----------------------------------------------------------|
| `compute_average_precision(ranked_ids, relevant_ids)` (line 107)  | AP = Σ(precision@k) / num_relevant over relevant positions; 0 when `relevant_ids` is empty |
| `compute_precision_recall_at_k(ranked_ids, relevant_ids, k_values=None)` (line 132) | Tuple of `(precision_at_k, recall_at_k)` dicts keyed by k. Defaults to `[5, 10, 20]` |
| `compute_metrics(ranked_ids, relevant_ids, query_text, target_category, k_values=None)` (line 163) | Bundle: returns a populated `QueryMetrics` |
| `compute_binary_classification_metrics(predictions, labels)` (line 196) | Returns `(accuracy, precision, recall, f1)` from 0/1 lists |

```python
from vtscore.eval.metrics import compute_metrics

ranked = [3, 1, 2, 5, 4]            # cids sorted descending by score
relevant = {1, 2, 4}                # cids in the target category

qm = compute_metrics(ranked, relevant,
                     query_text="a dog barking",
                     target_category="dog",
                     k_values=[3, 5])
print(qm.average_precision, qm.precision_at_k, qm.recall_at_k)
```

---

## Runners

### `eval_text_sort(medias, queries, media_type, ...)`

`vtscore/eval/runner.py`. For each query: embed the query text via
`vtscore.embedding.helpers.embed_text_query`, score every media by
cosine similarity, sort descending, and compute metrics treating
medias whose `"category"` matches `query.target_category` as relevant.
Returns a list of `QueryMetrics`. Pass `enrich=True` to use wrapper-
averaged text embeddings; pass `start_time` (a `time.monotonic()`
baseline) to populate `elapsed_seconds` on each result.

### `eval_learned_sort(medias, queries, train_fraction=0.5, seed=42, ...)`

`vtscore/eval/runner.py`. For each query/category: split target-
category vs. other medias, take `train_fraction` of each as training
data, build the synthetic `good_votes` / `bad_votes` dicts, call
`train_and_score` from [`vtscore.detectors`](detectors.md), and
measure accuracy / precision / recall / F1 on the held-out test set
using the cross-calibrated threshold. Returns a list of
`LearnedSortMetrics`. Honours `calibrate_count` and
`calibration_fraction` exactly the way the production training path
does, and takes its threshold from the same shipped estimator.

```python
from vtscore.eval.runner import eval_text_sort, eval_learned_sort
from vtscore.eval.config import EvalQuery
from vtscore.datasets.loader import load_demo_dataset

medias = {}
load_demo_dataset("esc50_s", medias)

queries = [EvalQuery("a dog barking", "dog"),
           EvalQuery("rain falling", "rain")]

text_results = eval_text_sort(medias, queries, media_type="audio")
learned_results = eval_learned_sort(medias, queries,
                                    train_fraction=0.5, seed=42)
```

### `run_eval(dataset_ids=None, mode="both", ...)`

`vtscore/eval/runner.py`. The full pipeline, written for the CLI
but usable from Python directly. Iterates over `dataset_ids` (or every
key of `EVAL_DATASETS` when `None`), loads each demo dataset into a
fresh `medias` dict via `load_demo_dataset`, runs `eval_text_sort`
when `mode in ("text", "both")` and `eval_learned_sort` when
`mode in ("learned", "both")`, prints progress to stdout, and returns
a list of `DatasetResult`.

Args:

| Arg                       | Meaning                                                       |
|---------------------------|---------------------------------------------------------------|
| `dataset_ids`             | List of eval dataset ids, or `None` for all                   |
| `mode`                    | `"text"`, `"learned"`, or `"both"`                            |
| `k_values`                | k values for P@k / R@k (default `[5, 10, 20]`)                |
| `train_fraction`          | Train/test split for learned-sort (default 0.5)               |
| `seed`                    | Random seed (default 42)                                      |
| `enrich`                  | Use wrapper-averaged text embeddings (default False)          |
| `calibrate_count`         | Cross-cal folds (default 2)                                   |
| `calibration_fraction`    | Cross-cal calibrate split (`None` = the app's per-space default: 0.3 single-vector / 0.5 patch) |

### `format_results_json(results)`

`vtscore/eval/runner.py`. Serialise a list of `DatasetResult` to a
JSON string by calling `r.to_dict()` on each and round-tripping
through `json.dumps(indent=2)`.

---

## Voting iterations

`vtscore/eval/voting_iterations.py` simulates an interactive labelling
session - votes are cast one at a time in the order the app's
**Autopilot** would present them, and at each step (once both
polarities have at least one vote) a fresh head is trained, a threshold
is computed, the held-out test set is scored, and the inclusion-weighted
cost is recorded. This is how the team answers "how does cost drop as
the user labels?" without spinning up a UI session.

The only vote-order strategy is `autopilot` (see
`vtscore/eval/al_strategies.py`): the eval reproduces the real user flow
rather than any academic active-learning heuristic. Autopilot seeds the
first few positives from text sort when a `seed_scores` ranking is
supplied, else from a handful of random known-good examples ("3 random
examples pulled from the Good"), then gathers the initial negatives and
cycles the standard Good / Bad / Hard / New phases.

### `simulate_voting_iterations(clips_dict, target_category, seed, ...)`

Run one `(dataset, category, seed)` simulation. Splits `clips_dict` into
`D_sim` (used to draw votes) and `D_test` (held out for cost evaluation)
by `sim_fraction`, then iterates:

1. Pick the next vote with the autopilot selector (seed → Good → Bad →
   Hard → New) and apply it; mirror it onto the coverage atlas that
   drives the New phase.
2. When both polarities have at least one vote, train the head
   (`train_model`; the default arm passes `LINEAR_SVM_HEAD`, matching the app)
   and pick a threshold
   (`calculate_cross_calibration_threshold`, with optional
   `calculate_safe_threshold` blend).
3. Score `D_test`, compute FPR / FNR, weight by inclusion, record the
   cost.

Pass `seed_scores={media_id: similarity}` (each item's cosine to the
typed query) to route the seed through text sort; omit it for the
random-known-good seed.

Returns a list of row dicts:

```python
{
    "seed": 0, "dataset": "esc50_s", "category": "dog",
    "strategy": "autopilot",
    "t": 7, "n_good": 3, "n_bad": 4,
    "cost": 0.124, "fpr": 0.05, "fnr": 0.21,
    "elapsed_seconds": 12.4,
}
```

`n_good`/`n_bad` are the good/bad vote counts the row's model was trained
on (they sum to `t`). Autopilot seeds goods before bads, so the first
scored step carries the initial goods plus a single bad; carry these
counts through so analysis can weight rows by sample size.

Honours the same threshold knobs as the runner. The population
estimator is fitted over the simulation set only (not the test set) so
test scores can't leak into calibration.  `safe_thresholds` defaults to
`True` here - matching the app, which has no switch for it - and
`False` runs the no-fusion control arm.

### `run_voting_iterations_eval(dataset_clips, seeds, categories=None, ...)`

Sweep `simulate_voting_iterations` over `(seed × dataset × category)` and
return a `pandas.DataFrame` with columns `seed, dataset, category,
strategy, t, n_good, n_bad, cost, fpr, fnr, elapsed_seconds`. When
`categories` is `None` or a dataset is missing from the dict, every
unique category in that dataset is used. Pass
`seed_scores={dataset: {category: {media_id: similarity}}}` to route the
autopilot seed through text sort per (dataset, category).

```python
from vtscore.eval.voting_iterations import run_voting_iterations_eval

df = run_voting_iterations_eval(
    dataset_clips={"esc50_s": medias},
    seeds=[0, 1, 2, 3, 4],
    inclusion=0,
    sim_fraction=0.5,
)
```

### `run_voting_iterations_eval_from_pickles(dataset_paths, seeds, ...)`

`vtscore/eval/voting_iterations.py`. Convenience wrapper that
loads each dataset from a pickle path (via
`vtscore.datasets.loader.load_dataset_from_pickle`) and then calls
`run_voting_iterations_eval`. The returned DataFrame has the same
columns.

---

## Label curve

`vtscore/eval/label_curve.py` is a separate sweep that compares the
MLP and SVM estimators head-to-head as a function of training-set size.
The sweep iterates over `dataset × target_category × trainer ×
label_count × seed` and writes one row per cell to a tidy DataFrame.

The headline metrics are rank-based on purpose: VTSearch never trusts
the raw score as a probability - it derives the operating threshold
via cross-calibration. So `AUROC`, `AP`, and the production-path
`f1_at_xcal` (which uses
`conformal_threshold` on a held-out calibration slice) are what
matter; Brier and F1@0.5 are kept as diagnostics. See
`SWEEP_TRAINERS` for the plug-in registry of estimator functions.

`label_curve_main.py` is the CLI:

```bash
python -m vtscore.eval.label_curve \
    --datasets esc50_s flowers102_s \
    --trainers mlp svm_linear svm_rbf \
    --label-counts 5 10 20 50 100 200 \
    --seeds 0 1 2 3 4 \
    --output label_curve.csv
```

---

## CLI

`python -m vtscore.eval` is the main entry point.
`vtscore/eval/__main__.py` calls `initialize_models()` to set up the
torch runtime, parses argparse args, runs `run_eval`, and optionally
writes a JSON dump and matplotlib plots.

```bash
# Default: text-sort + learned-sort on every registered eval dataset
python -m vtscore.eval

# Subset
python -m vtscore.eval --datasets esc50_s caltech101_s --mode both

# Custom split + JSON output
python -m vtscore.eval --mode learned --train-fraction 0.6 --output results.json

# Generate visualisations
python -m vtscore.eval --plot-dir eval_plots

# List available eval datasets
python -m vtscore.eval --list
```

Notable flags:

| Flag                     | Meaning                                                      |
|--------------------------|--------------------------------------------------------------|
| `--datasets ID [ID ...]` | Restrict to these eval dataset ids                           |
| `--mode {text,learned,both}` | Which evaluation to run                                  |
| `--k K [K ...]`          | k values for P@k / R@k                                       |
| `--train-fraction F`     | Learned-sort split ratio                                     |
| `--seed N`               | Random seed                                                  |
| `--enrich-descriptions`  | Wrapper-averaged text embeddings                             |
| `--calibrate-count K`    | Cross-cal folds                                              |
| `--calibration-fraction F` | Cross-cal calibration split                                |
| `--output FILE`          | Write JSON results to `FILE`                                 |
| `--plot-dir DIR`         | Generate visualisation PNGs in `DIR`                         |
| `--no-plot`              | Disable plots even when `--plot-dir` is set                  |
| `--list`                 | Print the eval-dataset registry and exit                     |

The full user guide - including how the demo datasets are sourced,
what each metric means in practice, and how to interpret the output -
is at [`docs/EVAL.md`](../../../docs/EVAL.md).

---

## Visualisation

`vtscore/eval/visualize.py` is the one module in `vtscore.eval` that
imports matplotlib. It is the presentation layer; library callers that
want raw numbers should consume the `DatasetResult` / DataFrame
return values directly and skip this module.

| Function                                                   | Output                                                          |
|------------------------------------------------------------|-----------------------------------------------------------------|
| `plot_eval_results(results, output_dir="eval_output")`     | PNGs for mAP-by-dataset, AP-by-query, P@k curves, R@k curves, learned-sort F1, learned-sort metrics breakdown |
| `plot_voting_iterations(df, output_dir="voting_output")`   | Cost-over-iterations and FPR/FNR-over-iterations line charts, one line per (dataset, category), with shaded ±1σ band over seeds |

Both functions create `output_dir` if missing and return a list of the
generated `Path`s. They apply a clean default matplotlib style
(`white` facecolor, grid on, top/right spines off) before plotting.
matplotlib is not in the library's core dependencies - installing
`matplotlib` separately is required to use these helpers.

---

## `trainer` vs `head`: two knobs, two registries

The eval framework carries two things called a *trainer*. They answer different
questions, and until issue #3764 they also shared the string `"mlp"`, which named
an MLP in one of them and the app's pipeline in the other. Read a `trainer`
column against the sweep that produced it:

| | Voting simulation | Label curve / timing |
|---|---|---|
| Entry point | `simulate_voting_iterations` | `run_label_curve_eval`, `run_timing_benchmark` |
| Registry | `vtscore/eval/step_trainers.py` | `vtscore/eval/sweep_trainers.py` |
| A "trainer" is | a whole **pipeline**: fit + threshold calibration | a bare **estimator**: `(X, y, seed) -> predict_fn` |
| Values | `app` (VTSearch's own pipeline; the default), `svm_linear`, `svm_rbf`, `svm_<kernel>@<params>` | `mlp`, `svm_linear`, `svm_rbf`, `mlp_ens<N>`, `svm_<kernel>@<params>` |
| Is `"mlp"` an MLP? | n/a — the arm is spelled `app`, and `"mlp"` is accepted only as its retired alias | yes, `train_model` with an auto-sized hidden layer |

The voting simulation's `head` knob is the one that picks a *model*, and it
applies to the `app` trainer alone, because only that arm fits one of VTSearch's
heads:

| `head` | What the app pipeline fits |
|---|---|
| `linear_svm` | `Linear(d, 1)` fitted by liblinear. **The shipped detector head**; `head=None` resolves here. |
| `linear` | The same `Linear(d, 1)` fitted by balanced BCE — the logistic head the SVM replaced. |
| `mlp` | An auto-sized hidden layer, BCE — the head VTSearch shipped before #2790. |

So `trainer="app", head="linear_svm"` is the shipped detector; `trainer="app",
head="mlp"` is the app's pipeline around a legacy head; and `trainer="svm_rbf"`
is a standalone estimator that has no head at all (its rows carry an empty
`head` column). Passing `head=` with any `svm_*` trainer is an error.

`svm_linear` and `linear_svm` are also easy to swap by eye and are not the same
thing: the first is a standalone sklearn/cuML SVM scored through its own
`predict_proba` and thresholded by the harness's trainer-agnostic
cross-calibration port; the second is the app's `Linear(d, 1)` whose weights come
from liblinear, scored and thresholded exactly as production does.

---

## Invariants worth restating

- **No persisted weights.** `eval_learned_sort` and
  `simulate_voting_iterations` train heads in memory and discard them
  after scoring; no detector files are written.
- **Deterministic.** Every function that produces randomness takes a
  `seed` argument; `np.random.RandomState(seed)` controls splits and
  vote order, `train_model` uses its own thread-safe RNG (see
  [`training.md`](training.md)).
- **Pure computation.** The runner and voting-iterations modules take
  pre-loaded medias dicts as input. The CLI is the only entry point
  that calls `load_demo_dataset` - programmatic consumers are
  expected to load their own data.
- **No Flask, no settings.** Every threshold knob (`inclusion`,
  `calibrate_count`, `calibration_fraction`, `sim_fraction`) is a
  function argument, not a global lookup.
