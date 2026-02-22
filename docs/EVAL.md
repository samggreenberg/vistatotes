# Evaluation Guide

VTSearch includes a built-in evaluation framework that measures how well its sorting methods work on demo datasets. This guide covers how to run evaluations, write custom eval scripts, and interpret the results.

## Quick start

Run the full evaluation across all demo datasets:

```bash
python -m vtsearch.eval --plot-dir eval_output
```

This will:

1. Download each demo dataset (cached after first run).
2. Run **text sort** (embedding-based ranking) and **learned sort** (neural net trained on simulated votes) evaluations.
3. Print a summary table to the terminal.
4. Save visualisation charts as PNGs in `eval_output/`.

## Prerequisites

Install dependencies (if you haven't already):

```bash
pip install -r requirements-cpu.txt   # or requirements-gpu.txt
```

Matplotlib and pandas are required for plot generation and are included in the requirements file.

## CLI reference

```
python -m vtsearch.eval [OPTIONS]
```

| Flag | Description | Default |
|------|-------------|---------|
| `--datasets ID [ID ...]` | Only evaluate these dataset IDs | all |
| `--mode {text,learned,both}` | Which evaluation to run | `both` |
| `--k K [K ...]` | k values for P@k / R@k | `5 10 20` |
| `--train-fraction F` | Fraction of clips used for training in learned-sort | `0.5` |
| `--seed N` | Random seed for reproducible splits | `42` |
| `--output FILE` | Write JSON results to FILE | none |
| `--plot-dir DIR` | Save visualisation PNGs to DIR | none |
| `--no-plot` | Disable plot generation | off |
| `--list` | List available eval datasets and exit | — |

### Examples

```bash
# Text sort only, on image datasets, save JSON
python -m vtsearch.eval --mode text --datasets animals_images vehicles_images --output results.json --plot-dir eval_output

# Learned sort with a different train/test split
python -m vtsearch.eval --mode learned --train-fraction 0.7 --seed 123 --plot-dir eval_output

# List available eval datasets
python -m vtsearch.eval --list
```

## Available eval datasets

Each eval dataset wraps a demo dataset and defines text queries targeting specific categories.

| Eval dataset ID | Media type | Demo dataset | Categories |
|----------------|-----------|--------------|------------|
| `nature_sounds` | Audio | nature_sounds | chirping_birds, crow, frog, insects, rain, sea_waves, thunderstorm, wind, water_drops, crickets |
| `city_sounds` | Audio | city_sounds | car_horn, siren, engine, train, helicopter, vacuum_cleaner, washing_machine, clock_alarm, keyboard_typing, door_wood_knock |
| `animals_images` | Image | animals_images | bird, cat, deer, dog, frog, horse |
| `vehicles_images` | Image | vehicles_images | airplane, automobile, ship, truck |
| `world_news` | Text | world_news | world, business |
| `sports_science_news` | Text | sports_science_news | sports, science |
| `activities_video` | Video | activities_video | ApplyEyeMakeup, ApplyLipstick, BrushingTeeth, Drumming, YoYo |
| `sports_video` | Video | sports_video | CliffDiving, HandstandWalking, JumpRope, PushUps, TaiChi |

## Understanding the metrics

### Text sort metrics

Text sort evaluation measures how well embedding-based search ranks clips. For each query:

- **Average Precision (AP)** — How well all relevant items are ranked near the top. 1.0 means every relevant item appeared before every irrelevant one.
- **Precision@k (P@k)** — Of the top-k results, what fraction is relevant.
- **Recall@k (R@k)** — Of all relevant items, what fraction appears in the top-k.
- **Mean Average Precision (mAP)** — AP averaged across all queries for a dataset.

### Learned sort metrics

Learned sort evaluation simulates voting, trains a binary classifier, then measures on a held-out test set:

- **Accuracy** — Fraction of correct predictions.
- **Precision** — Of items predicted positive, fraction that is actually positive.
- **Recall** — Of actual positives, fraction predicted positive.
- **F1** — Harmonic mean of precision and recall.

## Visualisations

When `--plot-dir` is set, the following charts are generated:

### Text sort plots

| File | Description |
|------|-------------|
| `text_sort_map_by_dataset.png` | Horizontal bar chart of mAP per dataset |
| `text_sort_ap_by_query.png` | Bar chart of AP for each individual query |
| `text_sort_precision_at_k.png` | Line chart of Precision@k curves |
| `text_sort_recall_at_k.png` | Line chart of Recall@k curves |

### Learned sort plots

| File | Description |
|------|-------------|
| `learned_sort_f1_by_category.png` | Bar chart of F1 score per category |
| `learned_sort_metrics_breakdown.png` | Grouped bar chart comparing accuracy, precision, recall, and F1 |

### Voting iterations plots (from custom scripts)

| File | Description |
|------|-------------|
| `voting_iterations_cost.png` | Cost curve over voting iterations (mean ± std across seeds) |
| `voting_iterations_fpr_fnr.png` | FPR and FNR curves over voting iterations |

## Writing a custom evaluation script

For more control — looping over parameter values, running voting-iteration simulations, or combining results across experiments — write a Python script that uses the eval API directly.

### Example: sweep over train fractions

This script evaluates learned-sort quality at different train/test split ratios:

```python
#!/usr/bin/env python
"""Sweep train_fraction and plot learned-sort F1."""

from vtsearch.models import initialize_models
initialize_models()

from vtsearch.eval.runner import run_eval
from vtsearch.eval.visualize import plot_eval_results

train_fractions = [0.3, 0.5, 0.7, 0.9]
datasets = ["animals_images"]

for frac in train_fractions:
    print(f"\n=== train_fraction={frac} ===")
    results = run_eval(
        dataset_ids=datasets,
        mode="learned",
        train_fraction=frac,
        seed=42,
    )

    # Generate plots for each setting
    plot_eval_results(results, output_dir=f"eval_sweep/frac_{frac}")

    for r in results:
        print(f"  {r.dataset_id}: mean_F1={r.mean_learned_f1:.4f}")
```

Run it with:

```bash
python my_eval_sweep.py
```

### Example: sweep over seeds

Evaluate multiple random seeds to measure variance:

```python
#!/usr/bin/env python
"""Run evaluation across multiple seeds to measure stability."""

from vtsearch.models import initialize_models
initialize_models()

from vtsearch.eval.runner import run_eval
from vtsearch.eval.visualize import plot_eval_results

seeds = [1, 2, 3, 42, 100]
datasets = ["animals_images", "nature_sounds"]
all_results = []

for seed in seeds:
    results = run_eval(
        dataset_ids=datasets,
        mode="both",
        seed=seed,
    )
    all_results.extend(results)

    for r in results:
        line = f"  seed={seed}  {r.dataset_id}"
        if r.text_sort:
            line += f"  mAP={r.mean_average_precision:.4f}"
        if r.learned_sort:
            line += f"  F1={r.mean_learned_f1:.4f}"
        print(line)

# Plot the last seed's results as a representative sample
plot_eval_results(results, output_dir="eval_seeds")
```

### Example: voting iterations evaluation

The voting-iterations evaluation measures how classification quality improves as more votes are cast. This is useful for understanding how many labels a user needs to provide before the model converges.

```python
#!/usr/bin/env python
"""Evaluate learned-sort cost over simulated voting iterations."""

from pathlib import Path

from vtsearch.models import initialize_models
initialize_models()

from vtsearch.datasets.loader import load_demo_dataset
from vtsearch.eval.visualize import plot_voting_iterations
from vtsearch.eval.voting_iterations import run_voting_iterations_eval

# Load datasets
datasets_to_eval = ["nature_sounds", "animals_images"]
dataset_clips = {}
for name in datasets_to_eval:
    clips = {}
    load_demo_dataset(name, clips)
    dataset_clips[name] = clips

# Run the voting iterations eval
df = run_voting_iterations_eval(
    dataset_clips=dataset_clips,
    seeds=[1, 2, 3, 42, 100],           # multiple seeds for averaging
    categories={                          # specific categories to test
        "nature_sounds": ["rain", "frog"],
        "animals_images": ["cat", "dog"],
    },
    inclusion=0,                          # inclusion bias setting
    sim_fraction=0.5,                     # fraction used for simulated voting
)

# Save raw data
df.to_csv("eval_output/voting_iterations.csv", index=False)
print(df.groupby(["dataset", "category"])["cost"].agg(["mean", "std"]))

# Generate plots
paths = plot_voting_iterations(df, output_dir="eval_output")
for p in paths:
    print(f"Saved: {p}")
```

### Example: voting iterations from pickle files

If you have pre-exported dataset pickle files, load them directly:

```python
#!/usr/bin/env python
"""Run voting iterations eval from pre-exported pickle files."""

from vtsearch.models import initialize_models
initialize_models()

from vtsearch.eval.visualize import plot_voting_iterations
from vtsearch.eval.voting_iterations import run_voting_iterations_eval_from_pickles

df = run_voting_iterations_eval_from_pickles(
    dataset_paths={
        "my_audio": "data/embeddings/my_audio_dataset.pkl",
        "my_images": "data/embeddings/my_image_dataset.pkl",
    },
    seeds=[1, 2, 3],
)

plot_voting_iterations(df, output_dir="eval_output")
```

## Using the visualisation API directly

The `plot_eval_results` and `plot_voting_iterations` functions can be called from any Python code:

```python
from vtsearch.eval.visualize import plot_eval_results, plot_voting_iterations

# Standard eval results -> list of PNG paths
paths = plot_eval_results(results, output_dir="my_plots")

# Voting iterations DataFrame -> list of PNG paths
paths = plot_voting_iterations(df, output_dir="my_plots")
```

Both functions:
- Create the output directory if it doesn't exist.
- Return a list of `Path` objects pointing to the generated PNGs.
- Skip chart types that don't apply (e.g., no learned-sort plots if only text-sort was run).
