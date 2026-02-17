"""ML training utilities for learned sorting."""

import numpy as np
import torch
import torch.nn as nn
from sklearn.mixture import GaussianMixture


def calculate_gmm_threshold(scores):
    """
    Use Gaussian Mixture Model to find threshold between two distributions.
    Assumes scores come from a bimodal distribution (Bad + Good).
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


def train_model(X_train, y_train, input_dim, inclusion_value=0):
    """
    Train a small MLP and return the trained model.

    Args:
        X_train: Training data
        y_train: Training labels (1 for good, 0 for bad)
        input_dim: Input dimension
        inclusion_value: Inclusion setting (-10 to +10)
            - If 0: balance classes equally
            - If positive: weight True samples more (effectively more Trues)
            - If negative: weight False samples more (effectively more Falses)
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
    for _ in range(200):
        optimizer.zero_grad()
        predictions = model(X_train)
        losses = loss_fn(predictions, y_train)
        weighted_loss = (losses.squeeze() * weights).mean()
        weighted_loss.backward()
        optimizer.step()

    model.eval()
    return model


def find_optimal_threshold(scores, labels, inclusion_value=0):
    """
    Find the best threshold that separates good (1) from bad (0).

    Args:
        scores: List of model scores
        labels: List of true labels (1 for good, 0 for bad)
        inclusion_value: Inclusion setting (-10 to +10)
            - If 0: minimize fpr + fnr
            - If positive: minimize fpr + 2^inclusion * fnr (include more)
            - If negative: minimize 2^inclusion * fpr + fnr (exclude more)
    """
    sorted_pairs = sorted(zip(scores, labels), reverse=True)
    best_threshold = 0.5
    best_cost = float("inf")

    # Calculate weights based on inclusion
    if inclusion_value >= 0:
        fpr_weight = 1.0
        fnr_weight = 2.0**inclusion_value
    else:
        fpr_weight = 2.0**inclusion_value
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


def calculate_cross_calibration_threshold(X_list, y_list, input_dim, inclusion_value=0):
    """
    Calculate threshold using cross-calibration:
    - Split data into two halves D1 and D2
    - Train M1 on D1, find threshold t1 using M1 on D2
    - Train M2 on D2, find threshold t2 using M2 on D1
    - Return mean(t1, t2)
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


def train_and_score(clips_dict, good_votes, bad_votes, inclusion_value=0):
    """Train a small MLP on voted clip embeddings and score every clip.

    Args:
        clips_dict: Dictionary of clips
        good_votes: Dict of good vote clip IDs
        bad_votes: Dict of bad vote clip IDs
        inclusion_value: Inclusion setting

    Returns:
        (results, threshold) where results is a list of {id, score} dicts
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
    threshold = calculate_cross_calibration_threshold(
        X_list, y_list, input_dim, inclusion_value
    )

    # Train final model on all data
    model = train_model(X, y, input_dim, inclusion_value)

    # Score every clip
    all_ids = sorted(clips_dict.keys())
    all_embs = np.array([clips_dict[cid]["embedding"] for cid in all_ids])
    X_all = torch.tensor(all_embs, dtype=torch.float32)
    with torch.no_grad():
        scores = model(X_all).squeeze(1).tolist()

    results = [{"id": cid, "score": round(s, 4)} for cid, s in zip(all_ids, scores)]
    results.sort(key=lambda x: x["score"], reverse=True)
    return results, threshold
