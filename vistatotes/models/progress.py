"""Progress tracking and stopping condition analysis."""

from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from vistatotes.models.training import find_optimal_threshold, train_model


def recreate_model_at_time(
    clips_dict: dict[int, dict[str, Any]],
    label_history: list[tuple[int, str, float]],
    time_index: int,
    inclusion_value: int = 0,
) -> tuple[Optional[nn.Sequential], Optional[float], list[int], list[int]]:
    """Recreate the classifier that would have been trained after a given labelling step.

    Builds training data from all labels in ``label_history`` up to and including
    ``time_index``, then trains an MLP and computes a threshold using those labels.
    Later labels for the same clip override earlier ones (last label wins).

    Args:
        clips_dict: Mapping of clip ID to clip data dict. Each value must contain
            an ``"embedding"`` key with a ``numpy.ndarray`` embedding vector.
        label_history: Ordered list of ``(clip_id, label, timestamp)`` tuples
            representing all labelling events, where ``label`` is ``"good"`` or
            ``"bad"``.
        time_index: Index into ``label_history``. Labels from index 0 through
            ``time_index`` (inclusive) are used. Must be in
            ``[0, len(label_history) - 1]``.
        inclusion_value: Integer in ``[-10, 10]`` controlling the FPR/FNR trade-off
            passed to :func:`~vistatotes.models.training.train_model` and
            :func:`~vistatotes.models.training.find_optimal_threshold`.

    Returns:
        A 4-tuple ``(model, threshold, good_ids, bad_ids)`` where:

        - ``model`` is the trained ``nn.Sequential`` model, or ``None`` if there
          is insufficient labelled data (fewer than 2 examples, or all labels are
          the same class).
        - ``threshold`` is the float decision boundary, or ``None`` when ``model``
          is ``None``.
        - ``good_ids`` is a list of clip IDs currently labelled good (may be
          non-empty even when ``model`` is ``None``).
        - ``bad_ids`` is a list of clip IDs currently labelled bad (may be
          non-empty even when ``model`` is ``None``).
    """
    if time_index < 0 or time_index >= len(label_history):
        return None, None, [], []

    # Collect labels up to time_index
    good_ids: set[int] = set()
    bad_ids: set[int] = set()

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
    X_list: list[np.ndarray] = []
    y_list: list[float] = []
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
    clips_dict: dict[int, dict[str, Any]],
    label_history: list[tuple[int, str, float]],
    current_good_votes: dict[int, None],
    current_bad_votes: dict[int, None],
    inclusion_value: int = 0,
) -> list[dict[str, Any]]:
    """Calculate classification error cost at each labelling step against the current labelset.

    For each time step ``t`` in ``label_history``, recreates the model trained on
    labels up to ``t`` and evaluates it against the *current* full labelset. This
    shows how the model's quality improved as more labels were added.

    Args:
        clips_dict: Mapping of clip ID to clip data dict. Each value must contain
            an ``"embedding"`` key with a ``numpy.ndarray`` embedding vector.
        label_history: Ordered list of ``(clip_id, label, timestamp)`` tuples
            representing all labelling events.
        current_good_votes: Dict whose keys are clip IDs currently labelled good.
        current_bad_votes: Dict whose keys are clip IDs currently labelled bad.
        inclusion_value: Integer in ``[-10, 10]`` controlling the FPR/FNR trade-off.
            - 0: cost = ``fpr + fnr``.
            - Positive: cost = ``fpr + 2^inclusion_value * fnr`` (penalise misses more).
            - Negative: cost = ``2^(-inclusion_value) * fpr + fnr`` (penalise false
              alarms more).

    Returns:
        A list of dicts, one per time step where a model could be trained (i.e.
        where both classes were present). Each dict has the keys:

        - ``"time_index"`` (``int``): Index into ``label_history``.
        - ``"num_labels"`` (``int``): Total number of labelled clips at this step.
        - ``"error_cost"`` (``float``): Weighted ``fpr + fnr``, rounded to 4 dp.
        - ``"fpr"`` (``float``): False-positive rate, rounded to 4 dp.
        - ``"fnr"`` (``float``): False-negative rate, rounded to 4 dp.

        Returns an empty list if ``current_good_votes`` and ``current_bad_votes``
        are both empty.
    """
    results: list[dict[str, Any]] = []

    # Determine weights based on inclusion
    if inclusion_value >= 0:
        fpr_weight = 1.0
        fnr_weight = 2.0**inclusion_value
    else:
        fpr_weight = 2.0 ** (-inclusion_value)
        fnr_weight = 1.0

    # Build current truth labels
    current_labels: dict[int, float] = {}
    for cid in current_good_votes:
        current_labels[cid] = 1.0
    for cid in current_bad_votes:
        current_labels[cid] = 0.0

    if not current_labels:
        return []

    # Only evaluate on currently labeled clips
    eval_ids = list(current_labels.keys())
    eval_embs: list[np.ndarray] = []
    eval_labels: list[float] = []
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
    clips_dict: dict[int, dict[str, Any]],
    label_history: list[tuple[int, str, float]],
    inclusion_value: int = 0,
) -> list[dict[str, Any]]:
    """Count how many unlabelled clips change their predicted label at each step.

    At each time step ``t``, recreates the model trained on labels up to ``t``,
    scores all *unlabelled* clips, and counts how many changed their predicted
    label (flipped) compared to the previous time step. A decreasing flip count
    is a sign that the model's predictions are stabilising.

    Args:
        clips_dict: Mapping of clip ID to clip data dict. Each value must contain
            an ``"embedding"`` key with a ``numpy.ndarray`` embedding vector.
        label_history: Ordered list of ``(clip_id, label, timestamp)`` tuples
            representing all labelling events.
        inclusion_value: Integer in ``[-10, 10]`` controlling the decision
            threshold used to binarise model scores. Passed to
            :func:`recreate_model_at_time`.

    Returns:
        A list of dicts, one per time step where a model could be trained. Each
        dict has the keys:

        - ``"time_index"`` (``int``): Index into ``label_history``.
        - ``"num_labels"`` (``int``): Total number of labelled clips at this step.
        - ``"num_flips"`` (``int``): Number of unlabelled clips whose predicted
          label changed since the previous time step (0 for the first valid step).
        - ``"num_unlabeled"`` (``int``): Number of unlabelled clips at this step.
    """
    results: list[dict[str, Any]] = []
    previous_predictions: Optional[dict[int, int]] = None

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
        predictions: dict[int, int] = {
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


def compute_labeling_status(
    clips_dict: dict[int, dict[str, Any]],
    label_history: list[tuple[int, str, float]],
    current_good_votes: dict[int, None],
    current_bad_votes: dict[int, None],
    inclusion_value: int = 0,
) -> dict[str, Any]:
    """Compute a lightweight red/yellow/green labeling status from the last 10 steps.

    Evaluates only the most recent 10 labeling steps (instead of the full history)
    to quickly determine whether continuing to label is worthwhile.

    Status meanings:

    - ``"red"``: Fewer than 20 total labels, or fewer than 5 good or 5 bad labels.
      The metric is not yet reliable.
    - ``"yellow"``: Minimum counts met, but error cost over the last 10 steps is
      still sloping downward, meaning labeling is still improving the model.
    - ``"green"``: Minimum counts met and error cost is flat over the last 10 steps.
      You can likely stop labeling.

    Args:
        clips_dict: Mapping of clip ID to clip data dict (must include ``"embedding"``).
        label_history: Ordered list of ``(clip_id, label, timestamp)`` tuples.
        current_good_votes: Dict whose keys are clip IDs labelled good.
        current_bad_votes: Dict whose keys are clip IDs labelled bad.
        inclusion_value: Integer in ``[-10, 10]`` controlling the FPR/FNR trade-off.

    Returns:
        A dict with keys ``"status"`` (``"red"``, ``"yellow"``, or ``"green"``),
        ``"reason"`` (human-readable explanation), ``"good_count"``, ``"bad_count"``,
        and ``"total_count"``. Yellow/green results also include a ``"slope"`` value
        (relative slope of error cost over the last 10 steps).
    """
    good = len(current_good_votes)
    bad = len(current_bad_votes)
    total = good + bad

    base = {"good_count": good, "bad_count": bad, "total_count": total}

    # Red: not enough data for a reliable metric
    if total < 20 or good < 5 or bad < 5:
        return {
            **base,
            "status": "red",
            "reason": (
                f"Need at least 20 labels with 5 good and 5 bad. "
                f"Currently {total} total ({good} good, {bad} bad)."
            ),
        }

    n = len(label_history)
    if n < 3:
        return {
            **base,
            "status": "yellow",
            "reason": "Not enough label history steps to assess trend.",
        }

    # Determine FPR/FNR weights from inclusion
    if inclusion_value >= 0:
        fpr_weight = 1.0
        fnr_weight = 2.0**inclusion_value
    else:
        fpr_weight = 2.0 ** (-inclusion_value)
        fnr_weight = 1.0

    # Build the current evaluation set (all labeled clips)
    current_labels: dict[int, float] = {}
    for cid in current_good_votes:
        current_labels[cid] = 1.0
    for cid in current_bad_votes:
        current_labels[cid] = 0.0

    eval_pairs = [(cid, lbl) for cid, lbl in current_labels.items() if cid in clips_dict]
    if not eval_pairs:
        return {**base, "status": "red", "reason": "No clip embeddings available."}

    eval_ids, eval_labels_list = zip(*eval_pairs)
    eval_embs = [clips_dict[cid]["embedding"] for cid in eval_ids]
    X_eval = torch.tensor(np.array(eval_embs), dtype=torch.float32)

    total_positives = sum(1 for lbl in eval_labels_list if lbl == 1.0)
    total_negatives = len(eval_labels_list) - total_positives

    # Evaluate only the last 10 time steps
    start_idx = max(0, n - 10)
    recent_error_costs: list[float] = []

    for t in range(start_idx, n):
        model, threshold, _, _ = recreate_model_at_time(
            clips_dict, label_history, t, inclusion_value
        )
        if model is None:
            continue

        with torch.no_grad():
            scores = model(X_eval).squeeze(1).tolist()

        fp = fn = 0
        for score, true_label in zip(scores, eval_labels_list):
            predicted = 1 if score >= threshold else 0
            if predicted == 1 and true_label == 0.0:
                fp += 1
            elif predicted == 0 and true_label == 1.0:
                fn += 1

        fpr = fp / total_negatives if total_negatives > 0 else 0.0
        fnr = fn / total_positives if total_positives > 0 else 0.0
        recent_error_costs.append(fpr_weight * fpr + fnr_weight * fnr)

    if len(recent_error_costs) < 3:
        return {
            **base,
            "status": "yellow",
            "reason": "Not enough valid model steps in recent history to assess trend.",
        }

    # Linear regression slope over the last 10 error-cost values
    n_pts = len(recent_error_costs)
    x_vals = list(range(n_pts))
    x_mean = sum(x_vals) / n_pts
    y_mean = sum(recent_error_costs) / n_pts

    numer = sum((x_vals[i] - x_mean) * (recent_error_costs[i] - y_mean) for i in range(n_pts))
    denom = sum((x_vals[i] - x_mean) ** 2 for i in range(n_pts))
    slope = numer / denom if denom != 0 else 0.0

    # Relative slope: normalise by mean error cost so the threshold is scale-independent
    relative_slope = slope / y_mean if y_mean > 0 else slope

    # If still dropping by more than 1.5 % of mean per step, labeling is still helping
    FLAT_THRESHOLD = -0.015

    if relative_slope < FLAT_THRESHOLD:
        return {
            **base,
            "status": "yellow",
            "reason": (
                "Error cost is still declining over the last 10 labels. "
                "Keep labeling to improve the model."
            ),
            "slope": round(relative_slope, 4),
        }
    else:
        return {
            **base,
            "status": "green",
            "reason": (
                "Error cost has leveled off over the last 10 labels. "
                "You can likely stop labeling."
            ),
            "slope": round(relative_slope, 4),
        }


def analyze_labeling_progress(
    clips_dict: dict[int, dict[str, Any]],
    label_history: list[tuple[int, str, float]],
    current_good_votes: dict[int, None],
    current_bad_votes: dict[int, None],
    inclusion_value: int = 0,
) -> dict[str, Any]:
    """Run a comprehensive analysis of labelling progress.

    Combines error-cost tracking (how well the model fits the current labelset over
    time) with prediction-stability tracking (how often unlabelled clips change
    their predicted label over time). Together these metrics can help determine
    when enough labels have been collected.

    Args:
        clips_dict: Mapping of clip ID to clip data dict. Each value must contain
            an ``"embedding"`` key with a ``numpy.ndarray`` embedding vector.
        label_history: Ordered list of ``(clip_id, label, timestamp)`` tuples
            representing all labelling events.
        current_good_votes: Dict whose keys are clip IDs currently labelled good.
        current_bad_votes: Dict whose keys are clip IDs currently labelled bad.
        inclusion_value: Integer in ``[-10, 10]`` controlling the FPR/FNR trade-off
            passed to both sub-analyses.

    Returns:
        A dict with the following keys:

        - ``"error_cost_over_time"`` (``list[dict]``): Output of
          :func:`calculate_error_cost_over_time`.
        - ``"stability_over_time"`` (``list[dict]``): Output of
          :func:`calculate_prediction_stability_over_time`.
        - ``"total_labels"`` (``int``): Combined count of good and bad votes.
        - ``"total_clips"`` (``int``): Total number of clips in ``clips_dict``.
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
