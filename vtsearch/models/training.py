"""ML training utilities for learned sorting."""

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.mixture import GaussianMixture

from config import TRAIN_EPOCHS


def calculate_gmm_threshold(scores: list[float]) -> float:
    """Use a Gaussian Mixture Model to find a threshold between two score distributions.

    Fits a 2-component GMM to the provided scores, assuming a bimodal distribution
    representing Bad (low) and Good (high) classes. Returns the midpoint between the
    two component means as the decision threshold.

    Args:
        scores: List of model confidence scores, expected to follow a bimodal distribution.

    Returns:
        A float threshold. Scores at or above this value are classified as Good.
        Falls back to the median of scores if GMM fitting fails or fewer than 2 scores
        are provided.
    """
    if len(scores) < 2:
        return 0.5

    # Reshape for sklearn
    X = np.array(scores).reshape(-1, 1)

    try:
        # Fit a 2-component GMM
        gmm: GaussianMixture = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(X)

        # Get the means of the two components
        means = np.ravel(gmm.means_)

        # Identify which component is "low" (Bad) and which is "high" (Good)
        low_idx = 0 if means[0] < means[1] else 1
        high_idx = 1 - low_idx

        # Threshold is at the intersection of the two Gaussians
        # For simplicity, use the midpoint between means
        threshold = (means[low_idx] + means[high_idx]) / 2.0

        return float(threshold)
    except Exception:
        # If GMM fails, return median
        return float(np.median(scores))


def train_model(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    input_dim: int,
    inclusion_value: int = 0,
) -> nn.Sequential:
    """Train a small MLP classifier and return the trained model.

    Trains a two-layer MLP (input -> 64 -> 1) using weighted binary cross-entropy
    loss. Class weights are adjusted based on ``inclusion_value`` to bias the
    classifier toward including more (positive) or fewer (positive) items.

    Args:
        X_train: Float tensor of shape ``(N, input_dim)`` containing training embeddings.
        y_train: Float tensor of shape ``(N, 1)`` containing binary labels
            (1.0 for good, 0.0 for bad).
        input_dim: Dimensionality of the input embeddings.
        inclusion_value: Integer in ``[-10, 10]`` controlling class-weight bias.
            - 0: balance classes equally (weight_true = num_false / num_true).
            - Positive: increase weight for True samples by ``2 ** inclusion_value``,
              causing the model to include more items.
            - Negative: increase weight for False samples by ``2 ** (-inclusion_value)``,
              causing the model to exclude more items.

    Returns:
        A trained ``nn.Sequential`` model in eval mode with layers:
        ``Linear(input_dim, 64) -> ReLU -> Linear(64, 1) -> Sigmoid``.
    """
    model = nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
        nn.Sigmoid(),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Calculate class weights based on inclusion
    num_true = y_train.sum().item()
    num_false = len(y_train) - num_true

    # Base weights for balanced classes
    if num_true > 0 and num_false > 0:
        weight_true = num_false / num_true
        weight_false = 1.0
    else:
        weight_true = 1.0
        weight_false = 1.0

    # Adjust weights based on inclusion
    if inclusion_value >= 0:
        # Increase weight for True samples
        weight_true *= 2.0**inclusion_value
    else:
        # Increase weight for False samples
        weight_false *= 2.0 ** (-inclusion_value)

    # Create sample weights
    weights = torch.where(y_train == 1, weight_true, weight_false).squeeze()
    loss_fn = nn.BCELoss(reduction="none")

    model.train()
    for _ in range(TRAIN_EPOCHS):
        optimizer.zero_grad()
        predictions = model(X_train)
        losses = loss_fn(predictions, y_train)
        weighted_loss = (losses.squeeze() * weights).mean()
        weighted_loss.backward()
        optimizer.step()

    model.eval()
    return model


def find_optimal_threshold(
    scores: list[float],
    labels: list[float],
    inclusion_value: int = 0,
) -> float:
    """Find the score threshold that best separates good (1) from bad (0) examples.

    Iterates over all candidate thresholds (each unique score value) and picks the
    one that minimises a weighted combination of false-positive rate (FPR) and
    false-negative rate (FNR). The relative weight of FPR vs. FNR is governed by
    ``inclusion_value``.

    Args:
        scores: List of model output scores, one per example.
        labels: List of true binary labels (1.0 for good, 0.0 for bad),
            corresponding to ``scores``.
        inclusion_value: Integer in ``[-10, 10]`` controlling the FPR/FNR trade-off.
            - 0: minimise ``fpr + fnr`` (equal weight).
            - Positive: minimise ``fpr + 2^inclusion_value * fnr`` (prefer recall,
              i.e., include more items).
            - Negative: minimise ``2^(-inclusion_value) * fpr + fnr`` (prefer
              precision, i.e., exclude more items).

    Returns:
        The float threshold that achieves the lowest weighted cost.
        Defaults to 0.5 if the score list is empty.
    """
    sorted_pairs = sorted(zip(scores, labels), reverse=True)
    best_threshold = 0.5
    best_cost = float("inf")

    # Calculate weights based on inclusion
    if inclusion_value >= 0:
        fpr_weight = 1.0
        fnr_weight = 2.0**inclusion_value
    else:
        fpr_weight = 2.0 ** (-inclusion_value)
        fnr_weight = 1.0

    for i in range(len(sorted_pairs)):
        threshold = sorted_pairs[i][0]

        # Calculate FPR and FNR at this threshold
        fp = 0  # false positives
        fn = 0  # false negatives
        tp = 0  # true positives
        tn = 0  # true negatives

        for score, label in sorted_pairs:
            predicted = 1 if score >= threshold else 0
            if predicted == 1 and label == 0:
                fp += 1
            elif predicted == 0 and label == 1:
                fn += 1
            elif predicted == 1 and label == 1:
                tp += 1
            else:  # predicted == 0 and label == 0
                tn += 1

        # Calculate rates
        total_positives = sum(1 for _, label in sorted_pairs if label == 1)
        total_negatives = len(sorted_pairs) - total_positives

        fpr = fp / total_negatives if total_negatives > 0 else 0
        fnr = fn / total_positives if total_positives > 0 else 0

        # Calculate weighted cost
        cost = fpr_weight * fpr + fnr_weight * fnr

        if cost < best_cost:
            best_cost = cost
            best_threshold = threshold

    return best_threshold


def calculate_cross_calibration_threshold(
    X_list: list[np.ndarray],
    y_list: list[float],
    input_dim: int,
    inclusion_value: int = 0,
) -> float:
    """Estimate a decision threshold using cross-calibration.

    Splits the labelled data into two halves (D1, D2), trains one model on each
    half, and uses each model to find an optimal threshold on the *other* half.
    The returned threshold is the mean of the two per-half thresholds, which
    reduces overfitting to the training split.

    Algorithm:
        1. Randomly split data into D1 and D2.
        2. Train M1 on D1; find threshold t1 by evaluating M1 on D2.
        3. Train M2 on D2; find threshold t2 by evaluating M2 on D1.
        4. Return ``(t1 + t2) / 2``.

    Args:
        X_list: List of embedding arrays (one per labelled example).
        y_list: List of binary labels (1.0 for good, 0.0 for bad),
            aligned with ``X_list``.
        input_dim: Dimensionality of the embeddings.
        inclusion_value: Integer in ``[-10, 10]`` passed to :func:`train_model`
            and :func:`find_optimal_threshold` to control the FPR/FNR trade-off.

    Returns:
        A float threshold. Returns 0.5 if fewer than 4 examples are provided
        (insufficient data for cross-calibration).
    """
    n = len(X_list)
    if n < 4:
        # Not enough data for cross-calibration
        return 0.5

    # Split data in half
    mid = n // 2
    indices = np.random.permutation(n)
    idx1 = indices[:mid]
    idx2 = indices[mid:]

    X_np = np.array(X_list)
    y_np = np.array(y_list)

    # Train M1 on D1
    X1 = torch.tensor(X_np[idx1], dtype=torch.float32)
    y1 = torch.tensor(y_np[idx1], dtype=torch.float32).unsqueeze(1)
    M1 = train_model(X1, y1, input_dim, inclusion_value)

    # Train M2 on D2
    X2 = torch.tensor(X_np[idx2], dtype=torch.float32)
    y2 = torch.tensor(y_np[idx2], dtype=torch.float32).unsqueeze(1)
    M2 = train_model(X2, y2, input_dim, inclusion_value)

    # Find t1: use M1 on D2
    with torch.no_grad():
        scores1_on_2 = M1(X2).squeeze(1).tolist()
    t1 = find_optimal_threshold(scores1_on_2, y_np[idx2].tolist(), inclusion_value)

    # Find t2: use M2 on D1
    with torch.no_grad():
        scores2_on_1 = M2(X1).squeeze(1).tolist()
    t2 = find_optimal_threshold(scores2_on_1, y_np[idx1].tolist(), inclusion_value)

    # Return mean
    return (t1 + t2) / 2.0


def train_and_score(
    clips_dict: dict[int, dict[str, Any]],
    good_votes: dict[int, None],
    bad_votes: dict[int, None],
    inclusion_value: int = 0,
) -> tuple[list[dict[str, Any]], float]:
    """Train a small MLP on voted clip embeddings and score every clip.

    Uses cross-calibration to determine an appropriate decision threshold, then
    trains a final model on all labelled data and scores every clip in
    ``clips_dict``.

    Args:
        clips_dict: Mapping of clip ID to clip data dict. Each value must contain
            an ``"embedding"`` key with a ``numpy.ndarray`` embedding vector.
        good_votes: Dict whose keys are clip IDs labelled as good (values are ``None``).
        bad_votes: Dict whose keys are clip IDs labelled as bad (values are ``None``).
        inclusion_value: Integer in ``[-10, 10]`` passed to the training and
            threshold-finding functions to control the inclusion/exclusion bias.

    Returns:
        A tuple ``(results, threshold)`` where:

        - ``results`` is a list of ``{"id": int, "score": float}`` dicts, sorted
          by score in descending order (highest confidence first).
        - ``threshold`` is the cross-calibrated decision boundary as a float.
    """
    X_list = []
    y_list = []
    for cid in good_votes:
        X_list.append(clips_dict[cid]["embedding"])
        y_list.append(1.0)
    for cid in bad_votes:
        X_list.append(clips_dict[cid]["embedding"])
        y_list.append(0.0)

    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    y = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1)

    input_dim = X.shape[1]

    # Calculate threshold using cross-calibration
    threshold = calculate_cross_calibration_threshold(X_list, y_list, input_dim, inclusion_value)

    # Train final model on all data
    model = train_model(X, y, input_dim, inclusion_value)

    # Score every clip
    all_ids = sorted(clips_dict.keys())
    all_embs = np.array([clips_dict[cid]["embedding"] for cid in all_ids])
    X_all = torch.tensor(all_embs, dtype=torch.float32)
    with torch.no_grad():
        scores = model(X_all).squeeze(1).tolist()

    # Sort by raw scores (full precision) so that tiny differences still
    # affect ordering.  Round only for the JSON response values.
    paired = sorted(zip(all_ids, scores), key=lambda x: x[1], reverse=True)
    results = [{"id": cid, "score": round(s, 4)} for cid, s in paired]
    return results, threshold
