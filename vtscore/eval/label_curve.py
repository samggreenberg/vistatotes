"""Label-curve sweep: MLP vs SVM as a function of training-set size.

Answers the question "should we switch from MLPs to SVMs?" by training
each candidate classifier on a growing number of labels (5, 10, 20, ...)
drawn from a demo dataset's simulation split, scoring a held-out test
split, and reporting rank-based metrics plus a production-path F1
across seeds.

The headline metrics are rank-based on purpose: VTSearch never trusts
the model's raw score as a probability - it derives the operating
threshold via :func:`vtscore.training.thresholds.calculate_cross_calibration_threshold`
and then applies it at inference.  So the relevant comparison is "how
good is the ranking" (AUROC, AP, best-F1) plus "what F1 does the
production cross-calibration path actually achieve" (``f1_at_xcal``,
which uses :func:`vtscore.training.thresholds.conformal_threshold` on a
held-out cal slice of the training labels).  Brier score and F1@0.5 are
kept on every row as diagnostics but excluded from the default
summary.

The sweep iterates over::

    dataset x target_category x trainer x label_count x seed

and writes one row per cell to a tidy DataFrame, mirroring the layout
of :mod:`vtscore.eval.voting_iterations` so the results compose with
existing tooling.

Example::

    from vtscore.eval.label_curve import run_label_curve_eval
    df = run_label_curve_eval(
        dataset_clips={"esc50_s": medias},
        trainers=("mlp", "svm_linear"),
        label_counts=(5, 10, 20, 50, 100),
        seeds=(0, 1, 2, 3, 4),
    )

See ``python -m vtscore.eval.label_curve --help`` for the CLI.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from vtscore.embedding.media_vectors import media_embedding

if TYPE_CHECKING:
    import pandas as pd


# ---------------------------------------------------------------------------
# Sweep-trainer registry
# ---------------------------------------------------------------------------
#
# The registry, its adapter helpers, and the trainer-agnostic cross-calibration
# threshold live in :mod:`vtscore.eval.sweep_trainers` so the voting-iterations
# sweep can share the parameterised-SVM parser and the threshold without
# importing this whole module.  They are re-exported here unchanged.
#
# Every arm in that registry is a **standalone estimator**, so this sweep's
# ``trainer`` column is not the voting simulation's: there, ``trainer="app"``
# names the shipped detector pipeline (issue #3764).

from vtscore.eval.sweep_trainers import (  # noqa: E402
    SWEEP_TRAINERS,  # noqa: F401  - re-exported: label_curve_main's CLI imports it from here
    _as_scores,  # noqa: F401  - re-exported for tests / backward compatibility
    _cross_calibrated_threshold,
    resolve_trainer,
)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve via the Mann-Whitney U formulation.

    Returns ``nan`` if either class is empty (undefined).
    """
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    # Average over ties: rank all scores, sum of positive ranks, etc.
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average ranks for ties so AUROC handles flat predictions correctly.
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i + 1
        while j < len(s_sorted) and s_sorted[j] == s_sorted[i]:
            j += 1
        if j - i > 1:
            avg = ranks[order[i:j]].mean()
            ranks[order[i:j]] = avg
        i = j
    pos_rank_sum = ranks[labels == 1].sum()
    n_pos = float(pos.size)
    n_neg = float(neg.size)
    u = pos_rank_sum - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def _auroc_std_err(scores: np.ndarray, labels: np.ndarray, auroc: float) -> float:
    """Hanley-McNeil analytic standard error of the AUROC.

    Uses the closed-form variance estimate from Hanley & McNeil (1982),
    ``Var = [A(1-A) + (n_pos-1)(Q1 - A²) + (n_neg-1)(Q2 - A²)] /
    (n_pos·n_neg)`` with ``Q1 = A/(2-A)`` and ``Q2 = 2A²/(1+A)``.  It is a
    deterministic function of the AUROC and the class counts, so it needs
    neither bootstrapping nor ensemble members - it is emitted for every
    trainer as the ``std_err_auroc`` diagnostic, giving an error bar on
    the ranking metric that complements the ensemble's ``std_mean``.

    Returns ``nan`` when either class is empty or *auroc* is not finite
    (the AUROC itself is undefined there).
    """
    pos = int((labels == 1).sum())
    neg = int((labels == 0).sum())
    if pos == 0 or neg == 0 or not np.isfinite(auroc):
        return float("nan")
    a = float(auroc)
    q1 = a / (2.0 - a)
    q2 = 2.0 * a * a / (1.0 + a)
    var = (a * (1.0 - a) + (pos - 1) * (q1 - a * a) + (neg - 1) * (q2 - a * a)) / (pos * neg)
    return float(np.sqrt(max(0.0, var)))


def _average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Average precision (area under PR curve), matching sklearn semantics."""
    if labels.sum() == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    cum_tp = np.cumsum(sorted_labels == 1)
    precisions = cum_tp / np.arange(1, len(sorted_labels) + 1)
    # AP = mean precision over positive positions.
    pos_mask = sorted_labels == 1
    return float(precisions[pos_mask].mean())


def _brier(scores: np.ndarray, labels: np.ndarray) -> float:
    """Brier score: mean squared error between probabilities and 0/1 labels."""
    return float(((scores - labels) ** 2).mean())


def _f1_at(scores: np.ndarray, labels: np.ndarray, threshold: float) -> float:
    preds = (scores >= threshold).astype(np.int32)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else 2.0 * tp / denom


def _best_f1(scores: np.ndarray, labels: np.ndarray) -> float:
    """Maximum F1 achievable over any threshold on these scores."""
    if labels.sum() == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order].astype(np.int64)
    cum_tp = np.cumsum(sorted_labels == 1)
    positions = np.arange(1, len(sorted_labels) + 1)
    total_pos = int(sorted_labels.sum())
    precisions = cum_tp / positions
    recalls = cum_tp / total_pos
    denom = precisions + recalls
    f1 = np.divide(
        2 * precisions * recalls,
        denom,
        out=np.zeros_like(denom, dtype=np.float64),
        where=denom > 0,
    )
    return float(f1.max())


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SplitPool:
    """Pre-built positive/negative pools for one (dataset, category, seed).

    Decoupling the split from the label-count loop ensures every label
    count under a fixed seed draws from the *same* sim pool, so the curve
    measures "what changes as we add labels" rather than "what changes as
    we reshuffle".
    """

    sim_pos: np.ndarray
    sim_neg: np.ndarray
    test_X: np.ndarray
    test_y: np.ndarray


def _build_split_pool(
    clips: dict[int, dict[str, Any]],
    target_category: str,
    seed: int,
    sim_fraction: float,
) -> _SplitPool | None:
    """Partition medias into sim/test pools, separated by ground-truth class.

    Returns ``None`` when the dataset is too small to form a usable test
    set with at least one positive and one negative example.
    """
    rng = np.random.default_rng(seed)
    pos_ids = np.array(
        [cid for cid, m in clips.items() if m.get("category") == target_category],
        dtype=np.int64,
    )
    neg_ids = np.array(
        [cid for cid, m in clips.items() if m.get("category") != target_category],
        dtype=np.int64,
    )
    if pos_ids.size < 2 or neg_ids.size < 2:
        return None
    rng.shuffle(pos_ids)
    rng.shuffle(neg_ids)

    n_pos_sim = max(1, int(pos_ids.size * sim_fraction))
    n_neg_sim = max(1, int(neg_ids.size * sim_fraction))
    sim_pos_ids = pos_ids[:n_pos_sim]
    sim_neg_ids = neg_ids[:n_neg_sim]
    test_pos_ids = pos_ids[n_pos_sim:]
    test_neg_ids = neg_ids[n_neg_sim:]
    if test_pos_ids.size == 0 or test_neg_ids.size == 0:
        return None

    def _stack(ids: np.ndarray) -> np.ndarray:
        return np.stack([np.asarray(media_embedding(clips[int(cid)]), dtype=np.float32) for cid in ids])

    sim_pos = _stack(sim_pos_ids)
    sim_neg = _stack(sim_neg_ids)
    test_X = np.concatenate([_stack(test_pos_ids), _stack(test_neg_ids)], axis=0)
    test_y = np.concatenate([np.ones(test_pos_ids.size, dtype=np.int32), np.zeros(test_neg_ids.size, dtype=np.int32)])
    return _SplitPool(sim_pos=sim_pos, sim_neg=sim_neg, test_X=test_X, test_y=test_y)


def _sample_labels(
    pool: _SplitPool,
    n_labels: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Draw a balanced label budget from the sim pools.

    Splits the label budget roughly 50/50 across classes; clips to the
    pool size when one class is smaller than ``n_labels // 2``.  Returns
    ``None`` when the request can't be satisfied with at least one of
    each class.
    """
    rng = np.random.default_rng(seed)
    n_pos_want = n_labels // 2
    n_neg_want = n_labels - n_pos_want
    n_pos = min(n_pos_want, pool.sim_pos.shape[0])
    n_neg = min(n_neg_want, pool.sim_neg.shape[0])
    if n_pos < 1 or n_neg < 1:
        return None
    pos_idx = rng.choice(pool.sim_pos.shape[0], size=n_pos, replace=False)
    neg_idx = rng.choice(pool.sim_neg.shape[0], size=n_neg, replace=False)
    X = np.concatenate([pool.sim_pos[pos_idx], pool.sim_neg[neg_idx]], axis=0)
    y = np.concatenate([np.ones(n_pos, dtype=np.int32), np.zeros(n_neg, dtype=np.int32)])
    # Shuffle so trainers can't exploit ordering.
    order = rng.permutation(X.shape[0])
    return X[order], y[order]


# ---------------------------------------------------------------------------
# Sweep driver
# ---------------------------------------------------------------------------


_HEADLINE_METRICS: tuple[str, ...] = (
    "auroc",
    "average_precision",
    "best_f1",
    "xcal_threshold",
    "f1_at_xcal",
    "train_seconds",
)
"""Rank-based metrics plus the production-path threshold metric.

These are the columns ``summarise()`` aggregates by default and the CLI
leads its summary with.  ``best_f1`` represents "what F1 is achievable
with the right threshold" and ``f1_at_xcal`` represents "what F1
production would actually get, using cross-calibration to pick that
threshold from the labels alone."
"""

_DIAGNOSTIC_METRICS: tuple[str, ...] = (
    "brier",
    "f1_at_0.5",
    "std_err_auroc",
    "std_mean",
    "predict_seconds",
)
"""Kept on every row but excluded from the default summary.

Brier and F1@0.5 only mean something if the score is a calibrated
probability with 0.5 as the operating point - neither holds in VTSearch
(the MLP's sigmoid is uncalibrated and the operating point is the
cross-calibrated threshold).  They stay available for anyone debugging
score-distribution shapes.

``std_err_auroc`` is the Hanley-McNeil analytic standard error of the
AUROC - an error bar on the ranking metric, computable for every
trainer.  ``std_mean`` is the mean per-item ensemble uncertainty (the
member-to-member sigmoid std, averaged over the test set); it is
``nan`` for non-ensemble trainers, which report a single score with no
spread.  Both are diagnostics: they characterise confidence, not
ranking quality, so they stay out of the headline summary.
"""

_ROW_COLUMNS: tuple[str, ...] = (
    "dataset",
    "category",
    "trainer",
    "n_labels",
    "n_pos",
    "n_neg",
    "seed",
    *_HEADLINE_METRICS,
    *_DIAGNOSTIC_METRICS,
)


def evaluate_one(
    pool: _SplitPool,
    trainer_name: str,
    n_labels: int,
    seed: int,
    *,
    inclusion_value: int = 0,
    calibrate_count: int = 2,
    cal_fraction: float = 0.5,
) -> dict[str, Any] | None:
    """Run a single cell of the sweep, returning a result row or ``None``.

    ``None`` means the cell was skipped (not enough labels in the pool to
    satisfy the request with both classes present).
    """
    trainer_fn = resolve_trainer(trainer_name)
    sample = _sample_labels(pool, n_labels, seed)
    if sample is None:
        return None
    X_train, y_train = sample

    t0 = time.monotonic()
    try:
        predict = trainer_fn(X_train, y_train, seed)
    except ValueError:
        # SVM trainer raises on single-class data - guard already covered
        # by the balanced sampler, but the trainer also defends itself so
        # we treat unexpected refusals as skipped cells.
        return None
    train_seconds = time.monotonic() - t0

    t0 = time.monotonic()
    prediction = predict(pool.test_X)
    predict_seconds = time.monotonic() - t0

    # Ensemble trainers return ``(scores, per_item_std)``; plain trainers
    # return just ``scores``.  ``std_mean`` averages the per-item spread over
    # the test set (mean epistemic uncertainty) and is ``nan`` when the trainer
    # reports no spread.
    if isinstance(prediction, tuple):
        scores = np.asarray(prediction[0], dtype=np.float64)
        std_mean = float(np.mean(np.asarray(prediction[1], dtype=np.float64)))
    else:
        scores = np.asarray(prediction, dtype=np.float64)
        std_mean = float("nan")

    labels = pool.test_y
    auroc = _auroc(scores, labels)

    # Cross-calibrated threshold mirrors the production path: split the
    # *training* labels (no test info leaks in), find the optimal
    # threshold on the cal halves, average.  ``f1_at_xcal`` then measures
    # the F1 we'd actually achieve at inference time, given only the
    # labels the model was trained on.
    xcal_thr = _cross_calibrated_threshold(
        X_train,
        y_train,
        trainer_fn,
        seed,
        inclusion_value=inclusion_value,
        calibrate_count=calibrate_count,
        cal_fraction=cal_fraction,
    )
    return {
        "trainer": trainer_name,
        "n_labels": int(y_train.size),
        "n_pos": int((y_train == 1).sum()),
        "n_neg": int((y_train == 0).sum()),
        "seed": int(seed),
        "auroc": auroc,
        "average_precision": _average_precision(scores, labels),
        "best_f1": _best_f1(scores, labels),
        "xcal_threshold": float(xcal_thr),
        "f1_at_xcal": _f1_at(scores, labels, xcal_thr),
        "train_seconds": round(train_seconds, 4),
        "brier": _brier(np.clip(scores, 0.0, 1.0), labels),
        "f1_at_0.5": _f1_at(scores, labels, 0.5),
        "std_err_auroc": _auroc_std_err(scores, labels, auroc),
        "std_mean": std_mean,
        "predict_seconds": round(predict_seconds, 4),
    }


def run_label_curve_eval(  # noqa: C901
    dataset_clips: dict[str, dict[int, dict[str, Any]]],
    trainers: Sequence[str] = ("mlp", "svm_linear"),
    label_counts: Sequence[int] = (5, 10, 20, 50, 100),
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    categories: dict[str, Sequence[str]] | None = None,
    sim_fraction: float = 0.5,
    inclusion_value: int = 0,
    calibrate_count: int = 2,
    cal_fraction: float = 0.5,
    progress: bool = False,
) -> pd.DataFrame:
    """Sweep trainers × label counts × seeds × (dataset, category).

    Args:
        dataset_clips: Mapping of dataset name to a preloaded medias dict.
            Each media must carry a resolvable embedding in the per-embedder
            ``"embeddings"`` store and a ``"category"`` (str).
        trainers: Sweep-trainer names to compare (keys of
            :data:`~vtscore.eval.sweep_trainers.SWEEP_TRAINERS`).
        label_counts: How many training labels to feed each trainer.
        seeds: Random seeds.  The split/sample is fully determined by the
            (dataset, category, seed) triple, so different trainers see
            identical training data at each (n_labels, seed).
        categories: Optional restriction of target categories per dataset.
            ``None`` means "every unique category in the dataset".
        sim_fraction: Fraction of medias placed in the sim pool (the rest
            become the held-out test pool).
        inclusion_value: Passed to the cross-calibration threshold finder
            (FPR/FNR tradeoff).  Doesn't affect the trainer's ranking; only
            the ``xcal_threshold`` / ``f1_at_xcal`` columns.
        calibrate_count: Number of train/cal folds for cross-calibration
            (mirrors the production default of 2).
        cal_fraction: Fraction of training labels held out as the cal
            portion within each fold.
        progress: When ``True``, print a one-line status update at the
            start of each (dataset, category, seed) outer cell.

    Returns:
        A tidy :class:`pandas.DataFrame` with one row per evaluated cell.
        Columns: ``dataset, category, trainer, n_labels, n_pos, n_neg,
        seed, auroc, average_precision, brier, f1_at_0.5, best_f1,
        train_seconds, predict_seconds``.
    """
    import pandas as pd  # noqa: PLC0415

    for name in trainers:
        resolve_trainer(name)  # validate up front; raises KeyError on an unknown name

    rows: list[dict[str, Any]] = []
    for ds_name, clips in dataset_clips.items():
        if categories and ds_name in categories:
            target_cats: Sequence[str] = list(categories[ds_name])
        else:
            target_cats = sorted({m["category"] for m in clips.values() if m.get("category")})

        for cat in target_cats:
            for seed in seeds:
                pool = _build_split_pool(clips, cat, int(seed), sim_fraction)
                if pool is None:
                    continue
                if progress:
                    print(
                        f"[{ds_name}] category={cat!r} seed={seed} "
                        f"sim_pos={pool.sim_pos.shape[0]} sim_neg={pool.sim_neg.shape[0]} "
                        f"test={pool.test_y.size}",
                        flush=True,
                    )
                for n in label_counts:
                    for trainer in trainers:
                        row = evaluate_one(
                            pool,
                            trainer,
                            int(n),
                            int(seed),
                            inclusion_value=inclusion_value,
                            calibrate_count=calibrate_count,
                            cal_fraction=cal_fraction,
                        )
                        if row is None:
                            continue
                        row["dataset"] = ds_name
                        row["category"] = cat
                        rows.append(row)

    return pd.DataFrame(rows, columns=pd.Index(list(_ROW_COLUMNS)))


def summarise(df: pd.DataFrame, *, include_diagnostics: bool = False) -> pd.DataFrame:
    """Collapse seeds into per-(dataset, category, trainer, n_labels) means.

    Useful when you just want the headline numbers for a plot.  Returns
    mean and stddev for each metric so error bars are still computable.

    By default only the rank-based metrics, ``best_f1``, the cross-
    calibrated threshold, and ``f1_at_xcal`` are aggregated - those are
    the columns that matter when downstream consumes ranks and learns
    its own threshold.  Pass ``include_diagnostics=True`` to also get
    Brier and F1@0.5 (useful if you specifically want to inspect score
    distribution shapes).
    """
    if df.empty:
        return df
    metric_cols = list(_HEADLINE_METRICS)
    if include_diagnostics:
        metric_cols += list(_DIAGNOSTIC_METRICS)
    # Keep only the metrics that are actually present in *df* - lets older
    # callers / cached frames flow through ``summarise()`` without crashing.
    metric_cols = [c for c in metric_cols if c in df.columns]
    grouped = df.groupby(["dataset", "category", "trainer", "n_labels"], sort=False)
    agg = grouped[metric_cols].agg(["mean", "std"]).reset_index()
    # Flatten MultiIndex columns: ("auroc", "mean") -> "auroc_mean".
    agg.columns = [f"{a}_{b}" if isinstance(b, str) and b else a for a, b in agg.columns.to_flat_index()]
    return agg
