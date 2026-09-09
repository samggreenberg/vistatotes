"""The pre-registered experiment grid for the MLP-vs-SVM study.

Kept in one place so Stage B (the definitive run), the SLURM array indexer, and
the report generator all agree on exactly which cells exist.  Sizing knobs are
env-overridable so the grid can be trimmed if the cluster is busy without editing
code (documented in the report's "decision points").
"""

from __future__ import annotations

import os

# --- Datasets (image, SigLIP) ---
DATASETS = os.environ.get("MLPSVM_DATASETS", "caltech101_m,caltech256_a,visual_genome_m").split(",")

# --- Sizing knobs (env-overridable) ---
N_CATEGORIES = int(os.environ.get("MLPSVM_N_CATEGORIES", "6"))
SEEDS = list(range(int(os.environ.get("MLPSVM_N_SEEDS", "8"))))
MAX_STEPS = int(os.environ.get("MLPSVM_MAX_STEPS", "200"))

# --- Prevalence arms: natural + 1% rare (None = natural) ---
RARE_PREVALENCE = float(os.environ.get("MLPSVM_RARE_PREVALENCE", "0.01"))
ARMS: list[float | None] = [None, RARE_PREVALENCE]

# --- Trainers: the definitive run.  Overridable so Stage A's winners feed in. ---
STAGE_B_TRAINERS = os.environ.get("MLPSVM_TRAINERS", "app,svm_linear,svm_rbf").split(",")

# --- Production-faithful fixed choices (pre-registered) ---
INCLUSION = 0
SIM_FRACTION = 0.5
CALIBRATE_COUNT = 2
CALIBRATION_FRACTION = 0.5
SAFE_THRESHOLDS = False
EMBEDDER = "siglip"
MEDIA_TYPE = "image"

# --- Minimum positives a category needs to be usable at natural prevalence ---
_MIN_CATEGORY_COUNT = int(os.environ.get("MLPSVM_MIN_CAT_COUNT", "20"))


def select_categories(category_counts: dict[str, int], n: int = N_CATEGORIES) -> list[str]:
    """Pick *n* categories spanning common→rare, deterministically.

    Categories with fewer than ``_MIN_CATEGORY_COUNT`` positives are dropped
    (their held-out test sets would be too small to estimate FNR).  The rest are
    sorted by count and sampled at even rank intervals, so the chosen set spans
    the prevalence range present in the dataset rather than clustering at one end.
    """
    usable = sorted(
        ((c, n_) for c, n_ in category_counts.items() if n_ >= _MIN_CATEGORY_COUNT),
        key=lambda kv: kv[1],
        reverse=True,
    )
    if len(usable) <= n:
        return [c for c, _ in usable]
    # Even rank sampling across the sorted-by-count list.
    idx = [round(i * (len(usable) - 1) / (n - 1)) for i in range(n)]
    return [usable[i][0] for i in sorted(set(idx))]


def array_cells(categories_by_dataset: dict[str, list[str]]) -> list[dict]:
    """Enumerate the (dataset, category, arm, seed) cells for the SLURM array.

    Each cell runs all trainers inside one task (they share the loaded dataset).
    Deterministic order so a task index maps to a stable cell across submissions.
    """
    cells: list[dict] = []
    for ds in DATASETS:
        for cat in categories_by_dataset.get(ds, []):
            for arm in ARMS:
                for seed in SEEDS:
                    cells.append({"dataset": ds, "category": cat, "arm": arm, "seed": seed})
    return cells
