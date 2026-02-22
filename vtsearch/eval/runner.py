"""Eval runner — loads datasets and measures sorting quality.

Two evaluation modes:

1. **Text sort**: For each query, embed the text, rank all clips by cosine
   similarity to the query embedding, and measure how well clips of the
   target category float to the top (AP, P@k, R@k).

2. **Learned sort**: For each category, randomly split its clips into
   train/test.  Simulate votes (target = good, rest = bad) on the
   train set, run ``train_and_score``, and measure classification
   quality on the held-out test set.
"""

from __future__ import annotations

import json
import sys
from typing import Any

import numpy as np

from vtsearch.eval.config import EVAL_DATASETS, EvalQuery
from vtsearch.eval.metrics import (
    DatasetResult,
    LearnedSortMetrics,
    compute_binary_classification_metrics,
    compute_metrics,
)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(np.dot(a, b) / norm)


def _run_text_sort_query(
    query: EvalQuery,
    clips: dict[int, dict[str, Any]],
    media_type: str,
    enrich: bool = False,
) -> list[dict[str, Any]]:
    """Embed the query text and rank clips by cosine similarity.

    Returns a list of ``{"id": int, "similarity": float}`` sorted descending.
    """
    from vtsearch.models.embeddings import embed_text_query

    text_vec = embed_text_query(query.text, media_type, enrich=enrich)
    if text_vec is None:
        raise RuntimeError(f"Could not embed query {query.text!r} for media type {media_type}")

    results = []
    for clip_id, clip in clips.items():
        sim = _cosine_similarity(clip["embedding"], text_vec)
        results.append({"id": clip_id, "similarity": sim})

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results


def eval_text_sort(
    clips: dict[int, dict[str, Any]],
    queries: list[EvalQuery],
    media_type: str,
    k_values: list[int] | None = None,
    enrich: bool = False,
) -> list:
    """Run text-sort evaluation for a list of queries.

    For each query, computes AP, P@k, and R@k treating clips whose
    ``"category"`` matches ``query.target_category`` as relevant.

    Args:
        clips: Loaded clip dict (``{id: clip_data}``).
        queries: List of :class:`EvalQuery` to evaluate.
        media_type: The media type string for embedding dispatch.
        k_values: Optional k values for P@k/R@k.
        enrich: If ``True``, use enriched (wrapper-averaged) text embeddings.

    Returns:
        List of :class:`~vtsearch.eval.metrics.QueryMetrics`.
    """
    from vtsearch.eval.metrics import QueryMetrics

    results: list[QueryMetrics] = []
    for query in queries:
        ranked = _run_text_sort_query(query, clips, media_type, enrich=enrich)
        ranked_ids = [r["id"] for r in ranked]
        relevant_ids = {cid for cid, c in clips.items() if c.get("category") == query.target_category}

        qm = compute_metrics(ranked_ids, relevant_ids, query.text, query.target_category, k_values)
        results.append(qm)

    return results


def eval_learned_sort(
    clips: dict[int, dict[str, Any]],
    queries: list[EvalQuery],
    train_fraction: float = 0.5,
    seed: int = 42,
) -> list[LearnedSortMetrics]:
    """Run learned-sort evaluation via simulated voting.

    For each query/category:
      1. Partition clips into target-category (positive) and others (negative).
      2. Randomly split both pools into train and test by ``train_fraction``.
      3. Build ``good_votes`` from train positives, ``bad_votes`` from train
         negatives.
      4. Call ``train_and_score`` on the full clip set.
      5. Measure accuracy/precision/recall/F1 on the test set using the
         cross-calibrated threshold.

    Args:
        clips: Loaded clip dict.
        queries: List of :class:`EvalQuery` (one per category to test).
        train_fraction: Fraction of clips to use for training (rest for test).
        seed: Random seed for reproducible splits.

    Returns:
        List of :class:`LearnedSortMetrics`, one per query.
    """
    from vtsearch.models.training import train_and_score

    rng = np.random.RandomState(seed)
    results: list[LearnedSortMetrics] = []

    for query in queries:
        # Split clips into target vs. other
        target_ids = [cid for cid, c in clips.items() if c.get("category") == query.target_category]
        other_ids = [cid for cid, c in clips.items() if c.get("category") != query.target_category]

        if len(target_ids) < 2 or len(other_ids) < 2:
            continue  # not enough data

        # Shuffle and split
        rng.shuffle(target_ids)
        rng.shuffle(other_ids)

        n_target_train = max(1, int(len(target_ids) * train_fraction))
        n_other_train = max(1, int(len(other_ids) * train_fraction))

        train_good = target_ids[:n_target_train]
        test_good = target_ids[n_target_train:]
        train_bad = other_ids[:n_other_train]
        test_bad = other_ids[n_other_train:]

        if not test_good or not test_bad:
            continue  # empty test set

        # Build vote dicts
        good_votes: dict[int, None] = {cid: None for cid in train_good}
        bad_votes: dict[int, None] = {cid: None for cid in train_bad}

        # Run train_and_score
        scored, threshold = train_and_score(clips, good_votes, bad_votes)

        # Evaluate on test set
        score_map = {r["id"]: r["score"] for r in scored}
        test_ids = test_good + test_bad
        predictions = [1 if score_map.get(cid, 0) >= threshold else 0 for cid in test_ids]
        labels = [1] * len(test_good) + [0] * len(test_bad)

        acc, prec, rec, f1 = compute_binary_classification_metrics(predictions, labels)

        results.append(
            LearnedSortMetrics(
                accuracy=acc,
                precision=prec,
                recall=rec,
                f1=f1,
                num_train=len(train_good) + len(train_bad),
                num_test=len(test_ids),
                target_category=query.target_category,
            )
        )

    return results


def run_eval(
    dataset_ids: list[str] | None = None,
    mode: str = "both",
    k_values: list[int] | None = None,
    train_fraction: float = 0.5,
    seed: int = 42,
    enrich: bool = False,
) -> list[DatasetResult]:
    """Run evaluation on one or more eval datasets.

    This is the main entry point.  It loads the demo dataset (downloading
    and embedding if needed), then runs text-sort and/or learned-sort
    evaluation.

    Args:
        dataset_ids: Which eval datasets to run.  ``None`` means all.
        mode: ``"text"`` for text-sort only, ``"learned"`` for learned-sort
            only, ``"both"`` for both.
        k_values: k values for P@k/R@k.
        train_fraction: Train/test split ratio for learned-sort.
        seed: Random seed.
        enrich: If ``True``, use enriched (wrapper-averaged) text embeddings
            for text-sort evaluation.

    Returns:
        List of :class:`DatasetResult`, one per evaluated dataset.
    """
    from vtsearch.datasets.config import DEMO_DATASETS
    from vtsearch.datasets.loader import load_demo_dataset

    if dataset_ids is None:
        dataset_ids = list(EVAL_DATASETS.keys())

    all_results: list[DatasetResult] = []

    for ds_id in dataset_ids:
        if ds_id not in EVAL_DATASETS:
            print(f"WARNING: unknown eval dataset {ds_id!r}, skipping", file=sys.stderr)
            continue

        eval_cfg = EVAL_DATASETS[ds_id]
        demo_id = eval_cfg["demo_dataset"]

        if demo_id not in DEMO_DATASETS:
            print(f"WARNING: demo dataset {demo_id!r} not found, skipping {ds_id!r}", file=sys.stderr)
            continue

        demo_info = DEMO_DATASETS[demo_id]
        media_type = demo_info.get("media_type", "audio")

        print(f"\n{'=' * 60}")
        print(f"Evaluating: {ds_id}  (media_type={media_type})")
        print(f"{'=' * 60}")

        # Load the demo dataset into a fresh clips dict
        clips: dict[int, dict] = {}
        try:
            load_demo_dataset(demo_id, clips)
        except Exception as e:
            print(f"ERROR loading dataset {demo_id}: {e}", file=sys.stderr)
            continue

        print(f"Loaded {len(clips)} clips across categories: ", end="")
        categories = sorted({c.get("category", "?") for c in clips.values()})
        print(", ".join(categories))

        queries = eval_cfg["queries"]
        ds_result = DatasetResult(dataset_id=ds_id, media_type=media_type)

        # --- Text sort ---
        if mode in ("text", "both"):
            print(f"\n--- Text Sort Evaluation ({len(queries)} queries) ---")
            text_results = eval_text_sort(clips, queries, media_type, k_values, enrich=enrich)
            ds_result.text_sort = text_results

            for qm in text_results:
                p5 = qm.precision_at_k.get(5, 0)
                p10 = qm.precision_at_k.get(10, 0)
                print(
                    f"  [{qm.target_category:20s}] AP={qm.average_precision:.3f}  "
                    f"P@5={p5:.2f}  P@10={p10:.2f}  "
                    f"({qm.num_relevant} relevant / {qm.num_total} total)"
                )
            print(f"  mAP = {ds_result.mean_average_precision:.4f}")

        # --- Learned sort ---
        if mode in ("learned", "both"):
            print(f"\n--- Learned Sort Evaluation ({len(queries)} categories) ---")
            learned_results = eval_learned_sort(clips, queries, train_fraction, seed)
            ds_result.learned_sort = learned_results

            for lm in learned_results:
                print(
                    f"  [{lm.target_category:20s}] Acc={lm.accuracy:.3f}  "
                    f"P={lm.precision:.3f}  R={lm.recall:.3f}  F1={lm.f1:.3f}  "
                    f"(train={lm.num_train}, test={lm.num_test})"
                )
            print(f"  Mean F1 = {ds_result.mean_learned_f1:.4f}")

        all_results.append(ds_result)

    return all_results


def format_results_json(results: list[DatasetResult]) -> str:
    """Serialise a list of :class:`DatasetResult` to a JSON string."""
    return json.dumps([r.to_dict() for r in results], indent=2)
