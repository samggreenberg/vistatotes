"""Ranking and classification metrics for evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class QueryMetrics:
    """Metrics for a single text-sort query against one target category."""

    query_text: str
    target_category: str
    average_precision: float
    precision_at_k: dict[int, float] = field(default_factory=dict)
    recall_at_k: dict[int, float] = field(default_factory=dict)
    num_relevant: int = 0
    num_total: int = 0


@dataclass
class LearnedSortMetrics:
    """Metrics for a single learned-sort evaluation fold."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    num_train: int
    num_test: int
    target_category: str = ""


@dataclass
class DatasetResult:
    """Aggregated results for one eval dataset."""

    dataset_id: str
    media_type: str
    text_sort: list[QueryMetrics] = field(default_factory=list)
    learned_sort: list[LearnedSortMetrics] = field(default_factory=list)

    @property
    def mean_average_precision(self) -> float:
        """Mean AP across all text-sort queries (mAP)."""
        if not self.text_sort:
            return 0.0
        return float(np.mean([q.average_precision for q in self.text_sort]))

    @property
    def mean_learned_f1(self) -> float:
        """Mean F1 across all learned-sort folds."""
        if not self.learned_sort:
            return 0.0
        return float(np.mean([f.f1 for f in self.learned_sort]))

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict for JSON output."""
        result: dict[str, Any] = {
            "dataset_id": self.dataset_id,
            "media_type": self.media_type,
        }

        if self.text_sort:
            result["text_sort"] = {
                "mAP": round(self.mean_average_precision, 4),
                "per_query": [
                    {
                        "query": q.query_text,
                        "target_category": q.target_category,
                        "AP": round(q.average_precision, 4),
                        "num_relevant": q.num_relevant,
                        "precision_at_k": {str(k): round(v, 4) for k, v in sorted(q.precision_at_k.items())},
                        "recall_at_k": {str(k): round(v, 4) for k, v in sorted(q.recall_at_k.items())},
                    }
                    for q in self.text_sort
                ],
            }

        if self.learned_sort:
            result["learned_sort"] = {
                "mean_f1": round(self.mean_learned_f1, 4),
                "per_category": [
                    {
                        "target_category": f.target_category,
                        "accuracy": round(f.accuracy, 4),
                        "precision": round(f.precision, 4),
                        "recall": round(f.recall, 4),
                        "f1": round(f.f1, 4),
                        "num_train": f.num_train,
                        "num_test": f.num_test,
                    }
                    for f in self.learned_sort
                ],
            }

        return result


def compute_average_precision(ranked_ids: list[int], relevant_ids: set[int]) -> float:
    """Compute Average Precision for a ranked list of clip IDs.

    AP = sum over relevant positions of (precision@k) / num_relevant.

    Args:
        ranked_ids: Clip IDs in descending order of predicted relevance.
        relevant_ids: Set of clip IDs that are actually relevant (match target category).

    Returns:
        Average Precision score in [0, 1].  Returns 0 if no relevant items exist.
    """
    if not relevant_ids:
        return 0.0

    hits = 0
    sum_precisions = 0.0
    for i, cid in enumerate(ranked_ids):
        if cid in relevant_ids:
            hits += 1
            sum_precisions += hits / (i + 1)

    return sum_precisions / len(relevant_ids)


def compute_precision_recall_at_k(
    ranked_ids: list[int],
    relevant_ids: set[int],
    k_values: list[int] | None = None,
) -> tuple[dict[int, float], dict[int, float]]:
    """Compute precision@k and recall@k for several values of k.

    Args:
        ranked_ids: Clip IDs in descending order of predicted relevance.
        relevant_ids: Set of clip IDs that are actually relevant.
        k_values: List of k values to evaluate.  Defaults to [5, 10, 20].

    Returns:
        Tuple of (precision_at_k, recall_at_k) dicts keyed by k.
    """
    if k_values is None:
        k_values = [5, 10, 20]

    num_relevant = len(relevant_ids)
    precision_at_k: dict[int, float] = {}
    recall_at_k: dict[int, float] = {}

    for k in k_values:
        top_k = set(ranked_ids[:k])
        hits = len(top_k & relevant_ids)
        precision_at_k[k] = hits / k if k > 0 else 0.0
        recall_at_k[k] = hits / num_relevant if num_relevant > 0 else 0.0

    return precision_at_k, recall_at_k


def compute_metrics(
    ranked_ids: list[int],
    relevant_ids: set[int],
    query_text: str,
    target_category: str,
    k_values: list[int] | None = None,
) -> QueryMetrics:
    """Compute all text-sort metrics for a single query.

    Args:
        ranked_ids: Clip IDs sorted by descending similarity.
        relevant_ids: Clip IDs belonging to the target category.
        query_text: The text query used.
        target_category: The category name targeted.
        k_values: Optional list of k values for P@k / R@k.

    Returns:
        A :class:`QueryMetrics` with AP, P@k, and R@k populated.
    """
    ap = compute_average_precision(ranked_ids, relevant_ids)
    p_at_k, r_at_k = compute_precision_recall_at_k(ranked_ids, relevant_ids, k_values)

    return QueryMetrics(
        query_text=query_text,
        target_category=target_category,
        average_precision=ap,
        precision_at_k=p_at_k,
        recall_at_k=r_at_k,
        num_relevant=len(relevant_ids),
        num_total=len(ranked_ids),
    )


def compute_binary_classification_metrics(
    predictions: list[int],
    labels: list[int],
) -> tuple[float, float, float, float]:
    """Compute accuracy, precision, recall, F1 for binary predictions.

    Args:
        predictions: List of 0/1 predicted labels.
        labels: List of 0/1 ground-truth labels.

    Returns:
        Tuple of (accuracy, precision, recall, f1).
    """
    tp = sum(p == 1 and gt == 1 for p, gt in zip(predictions, labels))
    fp = sum(p == 1 and gt == 0 for p, gt in zip(predictions, labels))
    fn = sum(p == 0 and gt == 1 for p, gt in zip(predictions, labels))
    tn = sum(p == 0 and gt == 0 for p, gt in zip(predictions, labels))

    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return accuracy, precision, recall, f1
