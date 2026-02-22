"""ML training utilities for learned sorting."""

import math
from concurrent.futures import ThreadPoolExecutor
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


def build_model(input_dim: int, generator: torch.Generator | None = None) -> nn.Sequential:
    """Construct the MLP architecture (untrained).

    The model outputs raw logits (no sigmoid).  Apply ``torch.sigmoid``
    to the output at inference time to obtain probabilities in [0, 1].

    Args:
        input_dim: Dimensionality of the input embeddings.
        generator: Optional local RNG for weight initialisation.  When
            provided the weights are re-initialised using this generator
            instead of PyTorch's global RNG, making construction
            thread-safe and deterministic.

    Returns:
        An ``nn.Sequential`` model with layers:
        ``Linear(input_dim, 64) -> ReLU -> Linear(64, 1)``.
    """
    model = nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )
    if generator is not None:
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5), generator=generator)
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(module.bias, -bound, bound, generator=generator)
    return model


def train_model(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    input_dim: int,
    inclusion_value: int = 0,
    seed: int = 42,
) -> nn.Sequential:
    """Train a small MLP classifier and return the trained model.

    Trains a two-layer MLP (input -> 64 -> 1) using weighted binary
    cross-entropy loss with logits (``BCEWithLogitsLoss``).  Class weights
    are adjusted based on ``inclusion_value`` to bias the classifier toward
    including more (positive) or fewer (positive) items.

    A local ``torch.Generator`` seeded with *seed* is used for model-weight
    initialisation, so the same inputs always produce the same trained model
    without mutating PyTorch's global RNG (thread-safe).

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
        seed: Seed for the local RNG used for weight initialisation (default 42).

    Returns:
        A trained ``nn.Sequential`` model in eval mode with layers:
        ``Linear(input_dim, 64) -> ReLU -> Linear(64, 1)``.
        The model outputs raw logits — apply ``torch.sigmoid`` at inference.
    """
    g = torch.Generator()
    g.manual_seed(seed)

    model = build_model(input_dim, generator=g)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

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
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    model.train()
    with torch.enable_grad():
        for _ in range(TRAIN_EPOCHS):
            optimizer.zero_grad()
            logits = model(X_train)
            losses = loss_fn(logits, y_train)
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
    if not scores:
        return 0.5

    # Vectorized O(n log n) threshold search using cumulative sums
    scores_arr = np.array(scores)
    labels_arr = np.array(labels)

    # Sort by score descending
    order = np.argsort(-scores_arr)
    sorted_scores = scores_arr[order]
    sorted_labels = labels_arr[order]

    total_positives = int(np.sum(sorted_labels == 1))
    total_negatives = len(sorted_labels) - total_positives

    if total_positives == 0 or total_negatives == 0:
        return 0.5

    # Calculate weights based on inclusion
    if inclusion_value >= 0:
        fpr_weight = 1.0
        fnr_weight = 2.0**inclusion_value
    else:
        fpr_weight = 2.0 ** (-inclusion_value)
        fnr_weight = 1.0

    # Cumulative counts as we move the threshold down the sorted list.
    # At position i, threshold = sorted_scores[i], so items 0..i are predicted positive.
    cum_positives = np.cumsum(sorted_labels == 1)  # TP at each threshold
    cum_negatives = np.cumsum(sorted_labels == 0)  # FP at each threshold

    # FP = cum_negatives, FN = total_positives - cum_positives
    fp = cum_negatives
    fn = total_positives - cum_positives

    fpr = fp / total_negatives
    fnr = fn / total_positives

    costs = fpr_weight * fpr + fnr_weight * fnr

    best_idx = int(np.argmin(costs))
    return float(sorted_scores[best_idx])


def calculate_cross_calibration_threshold(
    X_list: list[np.ndarray],
    y_list: list[float],
    input_dim: int,
    inclusion_value: int = 0,
    rng: np.random.RandomState | None = None,
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
        rng: Optional seeded RandomState for reproducible splits. Falls back
            to the global ``np.random`` state when ``None``.

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
    _rng = rng if rng is not None else np.random
    indices = _rng.permutation(n)
    idx1 = indices[:mid]
    idx2 = indices[mid:]

    X_np = np.array(X_list)
    y_np = np.array(y_list)

    X1 = torch.tensor(X_np[idx1], dtype=torch.float32)
    y1 = torch.tensor(y_np[idx1], dtype=torch.float32).unsqueeze(1)
    X2 = torch.tensor(X_np[idx2], dtype=torch.float32)
    y2 = torch.tensor(y_np[idx2], dtype=torch.float32).unsqueeze(1)

    # Train M1 and M2 in parallel (PyTorch releases the GIL during tensor ops)
    with ThreadPoolExecutor(max_workers=2) as pool:
        future_m1 = pool.submit(train_model, X1, y1, input_dim, inclusion_value)
        future_m2 = pool.submit(train_model, X2, y2, input_dim, inclusion_value)
        M1 = future_m1.result()
        M2 = future_m2.result()

    # Find thresholds: use each model on the opposite split
    with torch.no_grad():
        scores1_on_2 = torch.sigmoid(M1(X2)).squeeze(1).tolist()
        scores2_on_1 = torch.sigmoid(M2(X1)).squeeze(1).tolist()
    t1 = find_optimal_threshold(scores1_on_2, y_np[idx2].tolist(), inclusion_value)
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
        scores = torch.sigmoid(model(X_all)).squeeze(1).tolist()

    # Sort by raw scores (full precision) so that tiny differences still
    # affect ordering.  Round only for the JSON response values.
    paired = sorted(zip(all_ids, scores), key=lambda x: x[1], reverse=True)
    results = [{"id": cid, "score": round(s, 4)} for cid, s in paired]
    return results, threshold
