"""Evaluate learned-sort cost over simulated voting iterations.

For each combination of seed *s*, dataset *d*, and target category *c*:

1. Load the dataset and split clips into **D_sim** (simulation) and
   **D_test** (held-out) using *s* to control the random split.
2. Assign ground-truth labels based on *c*: clips whose ``"category"``
   matches *c* are positive (``good``), others are negative (``bad``).
3. Create a shuffled voting sequence from D_sim (order controlled by *s*).
4. Iterate through the voting sequence.  At each step *t* (once at least
   one good **and** one bad vote exist), train a model on votes so far,
   find a threshold, score D_test, and record the inclusion-weighted cost
   (``fpr_weight * FPR + fnr_weight * FNR``).

The result is a :class:`pandas.DataFrame` with columns
``seed, dataset, category, t, cost, fpr, fnr``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch

from vtsearch.models.training import (
    calculate_cross_calibration_threshold,
    train_model,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _inclusion_weights(inclusion: int) -> tuple[float, float]:
    """Return ``(fpr_weight, fnr_weight)`` for a given inclusion value."""
    if inclusion >= 0:
        return 1.0, 2.0**inclusion
    return 2.0 ** (-inclusion), 1.0


def _split_clip_ids(
    clips_dict: dict[int, dict[str, Any]],
    sim_fraction: float,
    rng: np.random.RandomState,
) -> tuple[list[int], list[int]]:
    """Randomly partition clip IDs into simulation and test sets."""
    all_ids = sorted(clips_dict.keys())
    shuffled = rng.permutation(all_ids).tolist()
    n_sim = max(1, int(len(shuffled) * sim_fraction))
    return shuffled[:n_sim], shuffled[n_sim:]


def _make_vote_sequence(
    sim_ids: list[int],
    clips_dict: dict[int, dict[str, Any]],
    target_category: str,
    rng: np.random.RandomState,
) -> list[tuple[int, str]]:
    """Build a shuffled list of ``(clip_id, label)`` pairs from simulation IDs."""
    votes = [
        (cid, "good" if clips_dict[cid]["category"] == target_category else "bad")
        for cid in sim_ids
    ]
    order = rng.permutation(len(votes))
    return [votes[i] for i in order]


def _evaluate_on_test(
    model: torch.nn.Sequential,
    threshold: float,
    clips_dict: dict[int, dict[str, Any]],
    test_ids: list[int],
    target_category: str,
    inclusion: int,
) -> dict[str, float]:
    """Score *test_ids* with *model* and return inclusion-weighted cost, FPR, FNR."""
    if not test_ids:
        return {"cost": float("nan"), "fpr": float("nan"), "fnr": float("nan")}

    embs = np.array([clips_dict[cid]["embedding"] for cid in test_ids])
    X = torch.tensor(embs, dtype=torch.float32)

    with torch.no_grad():
        scores = model(X).squeeze(1).tolist()

    true_labels = [
        1.0 if clips_dict[cid]["category"] == target_category else 0.0
        for cid in test_ids
    ]

    total_pos = sum(1 for lbl in true_labels if lbl == 1.0)
    total_neg = len(true_labels) - total_pos

    fp = fn = 0
    for score, label in zip(scores, true_labels):
        predicted = 1 if score >= threshold else 0
        if predicted == 1 and label == 0.0:
            fp += 1
        elif predicted == 0 and label == 1.0:
            fn += 1

    fpr = fp / total_neg if total_neg > 0 else 0.0
    fnr = fn / total_pos if total_pos > 0 else 0.0

    fpr_weight, fnr_weight = _inclusion_weights(inclusion)
    cost = fpr_weight * fpr + fnr_weight * fnr

    return {"cost": round(cost, 6), "fpr": round(fpr, 6), "fnr": round(fnr, 6)}


# ------------------------------------------------------------------
# Single (seed, dataset, category) evaluation
# ------------------------------------------------------------------


def simulate_voting_iterations(
    clips_dict: dict[int, dict[str, Any]],
    target_category: str,
    seed: int,
    dataset_name: str = "",
    inclusion: int = 0,
    sim_fraction: float = 0.5,
) -> list[dict[str, Any]]:
    """Simulate voting on *clips_dict* and evaluate at every step.

    Args:
        clips_dict: Pre-loaded clip dict (``{id: clip_data}``).
        target_category: Category treated as the positive class.
        seed: Random seed for splitting and vote ordering.
        dataset_name: Label included in result rows.
        inclusion: Inclusion setting in ``[-10, 10]``.
        sim_fraction: Fraction of clips used for simulated voting.

    Returns:
        List of row dicts with keys
        ``seed, dataset, category, t, cost, fpr, fnr``.
    """
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

    sim_ids, test_ids = _split_clip_ids(clips_dict, sim_fraction, rng)

    # Ensure the test set has both positive and negative clips
    test_pos = [cid for cid in test_ids if clips_dict[cid]["category"] == target_category]
    test_neg = [cid for cid in test_ids if clips_dict[cid]["category"] != target_category]
    if not test_pos or not test_neg:
        return []

    vote_seq = _make_vote_sequence(sim_ids, clips_dict, target_category, rng)

    good_votes: dict[int, None] = {}
    bad_votes: dict[int, None] = {}
    rows: list[dict[str, Any]] = []

    for t, (cid, label) in enumerate(vote_seq, start=1):
        if label == "good":
            good_votes[cid] = None
        else:
            bad_votes[cid] = None

        # Need at least 1 good and 1 bad to train
        if not good_votes or not bad_votes:
            continue

        # Build training data
        X_list: list[np.ndarray] = []
        y_list: list[float] = []
        for vid in good_votes:
            X_list.append(clips_dict[vid]["embedding"])
            y_list.append(1.0)
        for vid in bad_votes:
            X_list.append(clips_dict[vid]["embedding"])
            y_list.append(0.0)

        X = torch.tensor(np.array(X_list), dtype=torch.float32)
        y = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1)
        input_dim = X.shape[1]

        # Train and find threshold (mirrors train_and_score)
        threshold = calculate_cross_calibration_threshold(
            X_list, y_list, input_dim, inclusion
        )
        model = train_model(X, y, input_dim, inclusion)

        # Evaluate on held-out test set
        metrics = _evaluate_on_test(
            model, threshold, clips_dict, test_ids, target_category, inclusion
        )

        rows.append(
            {
                "seed": seed,
                "dataset": dataset_name,
                "category": target_category,
                "t": t,
                **metrics,
            }
        )

    return rows


# ------------------------------------------------------------------
# Full evaluation across seeds x datasets x categories
# ------------------------------------------------------------------


def run_voting_iterations_eval(
    dataset_clips: dict[str, dict[int, dict[str, Any]]],
    seeds: list[int],
    categories: Optional[dict[str, list[str]]] = None,
    inclusion: int = 0,
    sim_fraction: float = 0.5,
) -> pd.DataFrame:
    """Run the voting-iterations evaluation over multiple seeds/datasets/categories.

    Args:
        dataset_clips: Mapping of dataset name to a pre-loaded clips dict.
            Each clips dict maps ``int`` clip IDs to clip data dicts
            (must include ``"embedding"`` and ``"category"`` keys).
        seeds: List of random seeds to iterate over.
        categories: Optional mapping of dataset name to list of target
            categories.  If ``None`` or a dataset is missing from the dict,
            all unique categories in that dataset are used.
        inclusion: Inclusion setting in ``[-10, 10]``.
        sim_fraction: Fraction of clips reserved for simulated voting.

    Returns:
        A :class:`~pandas.DataFrame` with columns
        ``seed, dataset, category, t, cost, fpr, fnr``.
    """
    all_rows: list[dict[str, Any]] = []

    for ds_name, clips_dict in dataset_clips.items():
        # Determine target categories
        if categories and ds_name in categories:
            target_cats = categories[ds_name]
        else:
            target_cats = sorted({clips_dict[cid]["category"] for cid in clips_dict})

        for seed in seeds:
            for cat in target_cats:
                rows = simulate_voting_iterations(
                    clips_dict,
                    target_category=cat,
                    seed=seed,
                    dataset_name=ds_name,
                    inclusion=inclusion,
                    sim_fraction=sim_fraction,
                )
                all_rows.extend(rows)

    return pd.DataFrame(all_rows, columns=["seed", "dataset", "category", "t", "cost", "fpr", "fnr"])


def run_voting_iterations_eval_from_pickles(
    dataset_paths: dict[str, str],
    seeds: list[int],
    categories: Optional[dict[str, list[str]]] = None,
    inclusion: int = 0,
    sim_fraction: float = 0.5,
) -> pd.DataFrame:
    """Convenience wrapper that loads datasets from pickle files.

    Args:
        dataset_paths: Mapping of dataset name to pickle file path.
        seeds: List of random seeds.
        categories: Optional category filter (see :func:`run_voting_iterations_eval`).
        inclusion: Inclusion setting in ``[-10, 10]``.
        sim_fraction: Fraction of clips for simulation.

    Returns:
        A :class:`~pandas.DataFrame` identical to :func:`run_voting_iterations_eval`.
    """
    from vtsearch.datasets.loader import load_dataset_from_pickle

    dataset_clips: dict[str, dict[int, dict[str, Any]]] = {}
    for name, path in dataset_paths.items():
        clips: dict[int, dict[str, Any]] = {}
        load_dataset_from_pickle(Path(path), clips)
        dataset_clips[name] = clips

    return run_voting_iterations_eval(
        dataset_clips,
        seeds=seeds,
        categories=categories,
        inclusion=inclusion,
        sim_fraction=sim_fraction,
    )
