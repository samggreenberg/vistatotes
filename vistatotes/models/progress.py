"""Progress tracking and stopping condition analysis."""

import numpy as np
import torch

from vistatotes.models.training import find_optimal_threshold, train_model


def recreate_model_at_time(clips_dict, label_history, time_index, inclusion_value=0):
    """
    Recreate model M_t using only labels up to time_index.

    Args:
        clips_dict: Dictionary of all clips
        label_history: List of (clip_id, label, timestamp) tuples
        time_index: Index into label_history (use labels up to this index)
        inclusion_value: Inclusion setting

    Returns:
        (model, threshold, good_ids, bad_ids) or (None, None, [], []) if insufficient data
    """
    if time_index < 0 or time_index >= len(label_history):
        return None, None, [], []

    # Collect labels up to time_index
    good_ids = set()
    bad_ids = set()

    for i in range(time_index + 1):
        clip_id, label, _ = label_history[i]
        if label == "good":
            bad_ids.discard(clip_id)
            good_ids.add(clip_id)
        else:
            good_ids.discard(clip_id)
            bad_ids.add(clip_id)

    # Need at least one of each to train
    if not good_ids or not bad_ids:
        return None, None, list(good_ids), list(bad_ids)

    # Build training data
    X_list = []
    y_list = []
    for cid in good_ids:
        if cid in clips_dict:
            X_list.append(clips_dict[cid]["embedding"])
            y_list.append(1.0)
    for cid in bad_ids:
        if cid in clips_dict:
            X_list.append(clips_dict[cid]["embedding"])
            y_list.append(0.0)

    if len(X_list) < 2:
        return None, None, list(good_ids), list(bad_ids)

    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    y = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1)
    input_dim = X.shape[1]

    # Train model
    model = train_model(X, y, input_dim, inclusion_value)

    # Calculate threshold on training data (not ideal, but matches current approach)
    with torch.no_grad():
        scores = model(X).squeeze(1).tolist()
    threshold = find_optimal_threshold(scores, y_list, inclusion_value)

    return model, threshold, list(good_ids), list(bad_ids)


def calculate_error_cost_over_time(
    clips_dict, label_history, current_good_votes, current_bad_votes, inclusion_value=0
):
    """
    Calculate error cost (FPR + FNR weighted by inclusion) for each M_t against current labelset.

    Args:
        clips_dict: Dictionary of all clips
        label_history: List of (clip_id, label, timestamp) tuples
        current_good_votes: Current good votes dict
        current_bad_votes: Current bad votes dict
        inclusion_value: Inclusion setting

    Returns:
        List of {time_index, num_labels, error_cost, fpr, fnr}
    """
    results = []

    # Determine weights based on inclusion
    if inclusion_value >= 0:
        fpr_weight = 1.0
        fnr_weight = 2.0**inclusion_value
    else:
        fpr_weight = 2.0 ** (-inclusion_value)
        fnr_weight = 1.0

    # Build current truth labels
    current_labels = {}
    for cid in current_good_votes:
        current_labels[cid] = 1.0
    for cid in current_bad_votes:
        current_labels[cid] = 0.0

    if not current_labels:
        return []

    # Only evaluate on currently labeled clips
    eval_ids = list(current_labels.keys())
    eval_embs = []
    eval_labels = []
    for cid in eval_ids:
        if cid in clips_dict:
            eval_embs.append(clips_dict[cid]["embedding"])
            eval_labels.append(current_labels[cid])

    if not eval_embs:
        return []

    X_eval = torch.tensor(np.array(eval_embs), dtype=torch.float32)

    # Iterate through time, training models and evaluating
    for t in range(len(label_history)):
        model, threshold, good_ids, bad_ids = recreate_model_at_time(
            clips_dict, label_history, t, inclusion_value
        )

        if model is None:
            # Not enough data yet
            continue

        # Score the current labelset
        with torch.no_grad():
            scores = model(X_eval).squeeze(1).tolist()

        # Calculate FPR and FNR
        fp = 0
        fn = 0
        tp = 0
        tn = 0

        for score, true_label in zip(scores, eval_labels):
            predicted = 1 if score >= threshold else 0
            if predicted == 1 and true_label == 0:
                fp += 1
            elif predicted == 0 and true_label == 1:
                fn += 1
            elif predicted == 1 and true_label == 1:
                tp += 1
            else:
                tn += 1

        total_positives = sum(1 for lbl in eval_labels if lbl == 1)
        total_negatives = len(eval_labels) - total_positives

        fpr = fp / total_negatives if total_negatives > 0 else 0
        fnr = fn / total_positives if total_positives > 0 else 0

        error_cost = fpr_weight * fpr + fnr_weight * fnr

        results.append(
            {
                "time_index": t,
                "num_labels": len(good_ids) + len(bad_ids),
                "error_cost": round(error_cost, 4),
                "fpr": round(fpr, 4),
                "fnr": round(fnr, 4),
            }
        )

    return results


def calculate_prediction_stability_over_time(
    clips_dict, label_history, inclusion_value=0
):
    """
    Calculate how many unlabeled clips change their predicted label at each time step.

    Args:
        clips_dict: Dictionary of all clips
        label_history: List of (clip_id, label, timestamp) tuples
        inclusion_value: Inclusion setting

    Returns:
        List of {time_index, num_labels, num_flips, num_unlabeled}
    """
    results = []
    previous_predictions = None

    # Get all clip IDs
    all_clip_ids = sorted(clips_dict.keys())

    for t in range(len(label_history)):
        model, threshold, good_ids, bad_ids = recreate_model_at_time(
            clips_dict, label_history, t, inclusion_value
        )

        if model is None:
            continue

        # Get IDs of currently labeled clips
        labeled_ids = set(good_ids) | set(bad_ids)

        # Get unlabeled clips
        unlabeled_ids = [cid for cid in all_clip_ids if cid not in labeled_ids]

        if not unlabeled_ids:
            results.append(
                {
                    "time_index": t,
                    "num_labels": len(good_ids) + len(bad_ids),
                    "num_flips": 0,
                    "num_unlabeled": 0,
                }
            )
            continue

        # Score unlabeled clips
        unlabeled_embs = np.array([clips_dict[cid]["embedding"] for cid in unlabeled_ids])
        X_unlabeled = torch.tensor(unlabeled_embs, dtype=torch.float32)

        with torch.no_grad():
            scores = model(X_unlabeled).squeeze(1).tolist()

        # Convert scores to binary predictions
        predictions = {
            cid: 1 if score >= threshold else 0
            for cid, score in zip(unlabeled_ids, scores)
        }

        # Count flips from previous time step
        num_flips = 0
        if previous_predictions is not None:
            # Only count flips for clips that were unlabeled in both time steps
            common_unlabeled = set(predictions.keys()) & set(previous_predictions.keys())
            for cid in common_unlabeled:
                if predictions[cid] != previous_predictions[cid]:
                    num_flips += 1

        results.append(
            {
                "time_index": t,
                "num_labels": len(good_ids) + len(bad_ids),
                "num_flips": num_flips,
                "num_unlabeled": len(unlabeled_ids),
            }
        )

        previous_predictions = predictions

    return results


def analyze_labeling_progress(
    clips_dict, label_history, current_good_votes, current_bad_votes, inclusion_value=0
):
    """
    Comprehensive analysis of labeling progress.

    Returns:
        {
            "error_cost_over_time": [...],
            "stability_over_time": [...],
            "total_labels": int,
            "total_clips": int,
        }
    """
    error_cost = calculate_error_cost_over_time(
        clips_dict, label_history, current_good_votes, current_bad_votes, inclusion_value
    )

    stability = calculate_prediction_stability_over_time(
        clips_dict, label_history, inclusion_value
    )

    return {
        "error_cost_over_time": error_cost,
        "stability_over_time": stability,
        "total_labels": len(current_good_votes) + len(current_bad_votes),
        "total_clips": len(clips_dict),
    }
