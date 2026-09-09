"""Tests for the label-curve sweep (vtscore.eval.label_curve)."""

from __future__ import annotations

import numpy as np
import pytest

from vtscore.eval.label_curve import (
    SWEEP_TRAINERS,
    _as_scores,
    _auroc,
    _auroc_std_err,
    _average_precision,
    _best_f1,
    _brier,
    _build_split_pool,
    _cross_calibrated_threshold,
    _f1_at,
    _sample_labels,
    evaluate_one,
    run_label_curve_eval,
    summarise,
)


def _synth_dataset(
    n_pos: int = 30,
    n_neg: int = 30,
    dim: int = 8,
    seed: int = 0,
    category: str = "target",
) -> dict[int, dict]:
    """Build a tiny synthetic medias dict with two clearly-separable categories.

    Mirrors the shape of a real VTSearch medias dict: each entry has an
    ``embedding`` (np.ndarray) and a ``category`` (str), keyed by an
    integer media ID.
    """
    rng = np.random.default_rng(seed)
    clips: dict[int, dict] = {}
    next_id = 1
    pos = rng.standard_normal((n_pos, dim)).astype(np.float32) * 0.2
    pos[:, 0] += 1.0
    neg = rng.standard_normal((n_neg, dim)).astype(np.float32) * 0.2
    neg[:, 0] -= 1.0
    for vec in pos:
        clips[next_id] = {"embeddings": {"emb": vec}, "category": category}
        next_id += 1
    for vec in neg:
        clips[next_id] = {"embeddings": {"emb": vec}, "category": "other"}
        next_id += 1
    return clips


class TestMetricHelpers:
    def test_auroc_perfect_separation(self):
        scores = np.array([0.9, 0.8, 0.1, 0.2])
        labels = np.array([1, 1, 0, 0])
        assert _auroc(scores, labels) == pytest.approx(1.0)

    def test_auroc_reversed_predictions(self):
        scores = np.array([0.1, 0.2, 0.9, 0.8])
        labels = np.array([1, 1, 0, 0])
        assert _auroc(scores, labels) == pytest.approx(0.0)

    def test_auroc_random_is_half(self):
        # Constant scores → AUROC should average to 0.5 over ties.
        scores = np.array([0.5, 0.5, 0.5, 0.5])
        labels = np.array([1, 0, 1, 0])
        assert _auroc(scores, labels) == pytest.approx(0.5)

    def test_average_precision_perfect_ranking(self):
        scores = np.array([0.9, 0.8, 0.1])
        labels = np.array([1, 1, 0])
        assert _average_precision(scores, labels) == pytest.approx(1.0)

    def test_brier_zero_for_perfect_calibration(self):
        scores = np.array([1.0, 0.0, 1.0])
        labels = np.array([1, 0, 1])
        assert _brier(scores, labels) == pytest.approx(0.0)

    def test_brier_max_for_inverted_predictions(self):
        scores = np.array([0.0, 1.0])
        labels = np.array([1, 0])
        assert _brier(scores, labels) == pytest.approx(1.0)

    def test_f1_at_threshold(self):
        scores = np.array([0.9, 0.6, 0.4, 0.1])
        labels = np.array([1, 1, 0, 0])
        assert _f1_at(scores, labels, 0.5) == pytest.approx(1.0)
        # Threshold too high → all predicted negative → recall=0, F1=0
        assert _f1_at(scores, labels, 0.95) == pytest.approx(0.0)

    def test_best_f1_finds_optimum(self):
        scores = np.array([0.9, 0.6, 0.4, 0.1])
        labels = np.array([1, 1, 0, 0])
        assert _best_f1(scores, labels) == pytest.approx(1.0)

    def test_auroc_std_err_finite_and_nonneg(self):
        # Non-degenerate AUROC → a finite, non-negative Hanley-McNeil SE.
        scores = np.array([0.9, 0.7, 0.6, 0.3, 0.2, 0.1])
        labels = np.array([1, 1, 0, 1, 0, 0])
        auroc = _auroc(scores, labels)
        se = _auroc_std_err(scores, labels, auroc)
        assert np.isfinite(se)
        assert se >= 0.0

    def test_auroc_std_err_zero_for_perfect_separation(self):
        # A=1.0 makes every variance term vanish → SE == 0.
        scores = np.array([0.9, 0.8, 0.2, 0.1])
        labels = np.array([1, 1, 0, 0])
        assert _auroc_std_err(scores, labels, 1.0) == pytest.approx(0.0)

    def test_auroc_std_err_nan_when_single_class(self):
        scores = np.array([0.9, 0.8, 0.7])
        labels = np.array([1, 1, 1])
        assert np.isnan(_auroc_std_err(scores, labels, float("nan")))


class TestAsScores:
    def test_passes_through_plain_array(self):
        arr = np.array([0.1, 0.9, 0.5])
        np.testing.assert_array_equal(_as_scores(arr), arr)

    def test_extracts_scores_from_tuple(self):
        scores = np.array([0.1, 0.9])
        std = np.array([0.01, 0.02])
        np.testing.assert_array_equal(_as_scores((scores, std)), scores)


class TestSplitPool:
    def test_partitions_categories(self):
        clips = _synth_dataset(n_pos=20, n_neg=20)
        pool = _build_split_pool(clips, "target", seed=0, sim_fraction=0.5)
        assert pool is not None
        assert pool.sim_pos.shape[0] == 10
        assert pool.sim_neg.shape[0] == 10
        # Test set holds the remaining 10 positives + 10 negatives.
        assert pool.test_X.shape[0] == 20
        assert int(pool.test_y.sum()) == 10

    def test_returns_none_when_too_small(self):
        clips = _synth_dataset(n_pos=1, n_neg=10)
        assert _build_split_pool(clips, "target", seed=0, sim_fraction=0.5) is None

    def test_returns_none_when_unknown_category(self):
        clips = _synth_dataset(n_pos=10, n_neg=10)
        assert _build_split_pool(clips, "missing", seed=0, sim_fraction=0.5) is None

    def test_seed_is_reproducible(self):
        clips = _synth_dataset(n_pos=20, n_neg=20)
        pool_a = _build_split_pool(clips, "target", seed=0, sim_fraction=0.5)
        pool_b = _build_split_pool(clips, "target", seed=0, sim_fraction=0.5)
        assert pool_a is not None and pool_b is not None
        np.testing.assert_array_equal(pool_a.sim_pos, pool_b.sim_pos)
        np.testing.assert_array_equal(pool_a.test_y, pool_b.test_y)


class TestSampleLabels:
    def test_balanced_split(self):
        clips = _synth_dataset(n_pos=20, n_neg=20)
        pool = _build_split_pool(clips, "target", seed=0, sim_fraction=0.5)
        assert pool is not None
        X, y = _sample_labels(pool, n_labels=10, seed=0)  # type: ignore[misc]
        assert X.shape == (10, 8)
        assert int(y.sum()) == 5  # n_labels // 2

    def test_caps_to_pool_size_when_imbalanced(self):
        # Pool with only 3 positives in sim; asking for 50 labels should
        # cap at 3 positives.
        clips = _synth_dataset(n_pos=6, n_neg=200)
        pool = _build_split_pool(clips, "target", seed=0, sim_fraction=0.5)
        assert pool is not None
        out = _sample_labels(pool, n_labels=50, seed=0)
        assert out is not None
        X, y = out
        assert int(y.sum()) <= pool.sim_pos.shape[0]
        assert int((y == 0).sum()) <= pool.sim_neg.shape[0]

    def test_returns_none_when_pool_is_single_class(self):
        # An empty positive sim pool can't satisfy any balanced request.
        clips = _synth_dataset(n_pos=2, n_neg=20)
        pool = _build_split_pool(clips, "target", seed=0, sim_fraction=0.5)
        assert pool is not None
        # Hand-strip the positive pool to force the corner case.
        bare = pool.__class__(
            sim_pos=np.empty((0, pool.sim_pos.shape[1]), dtype=np.float32),
            sim_neg=pool.sim_neg,
            test_X=pool.test_X,
            test_y=pool.test_y,
        )
        assert _sample_labels(bare, n_labels=4, seed=0) is None


class TestCrossCalibratedThreshold:
    def test_returns_finite_threshold_with_balanced_labels(self):
        # 10 labels, balanced; every fold should produce a valid threshold.
        from vtscore.eval.label_curve import SWEEP_TRAINERS as T

        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 4)).astype(np.float32)
        X[:10, 0] += 1.5  # positives in upper half
        X[10:, 0] -= 1.5
        y = np.array([1] * 10 + [0] * 10, dtype=np.int32)
        t = _cross_calibrated_threshold(X, y, T["svm_linear"], seed=0)
        assert np.isfinite(t)

    def test_too_few_labels_returns_default(self):
        from vtscore.eval.label_curve import SWEEP_TRAINERS as T

        X = np.zeros((3, 4), dtype=np.float32)
        y = np.array([1, 0, 1], dtype=np.int32)
        # n=3 hits the n < 4 guard → returns 0.5 fallback.
        assert _cross_calibrated_threshold(X, y, T["svm_linear"], seed=0) == 0.5


class TestEvaluateOne:
    _EXPECTED_KEYS = (
        "trainer",
        "n_labels",
        "n_pos",
        "n_neg",
        "seed",
        "auroc",
        "average_precision",
        "best_f1",
        "xcal_threshold",
        "f1_at_xcal",
        "train_seconds",
        "brier",
        "f1_at_0.5",
        "std_err_auroc",
        "std_mean",
        "predict_seconds",
    )

    def test_returns_row_schema_for_svm(self):
        clips = _synth_dataset(n_pos=20, n_neg=20)
        pool = _build_split_pool(clips, "target", seed=0, sim_fraction=0.5)
        assert pool is not None
        row = evaluate_one(pool, trainer_name="svm_linear", n_labels=10, seed=0)
        assert row is not None
        for key in self._EXPECTED_KEYS:
            assert key in row, f"missing key {key!r}"
        assert row["trainer"] == "svm_linear"
        # Trivially-separable synthetic data should score high.
        assert row["auroc"] > 0.85
        # Cross-calibrated threshold is a finite float.
        assert np.isfinite(row["xcal_threshold"])
        # f1_at_xcal should be high too on this trivially-separable data.
        assert row["f1_at_xcal"] > 0.7
        # A non-ensemble trainer reports no per-item spread.
        assert np.isnan(row["std_mean"])
        # But the analytic AUROC standard error is always available.
        assert np.isfinite(row["std_err_auroc"])

    def test_returns_row_schema_for_mlp(self):
        clips = _synth_dataset(n_pos=20, n_neg=20)
        pool = _build_split_pool(clips, "target", seed=0, sim_fraction=0.5)
        assert pool is not None
        row = evaluate_one(pool, trainer_name="mlp", n_labels=10, seed=0)
        assert row is not None
        assert row["trainer"] == "mlp"
        assert 0.0 <= row["auroc"] <= 1.0
        for key in self._EXPECTED_KEYS:
            assert key in row, f"missing key {key!r}"

    def test_unknown_trainer_raises(self):
        clips = _synth_dataset()
        pool = _build_split_pool(clips, "target", seed=0, sim_fraction=0.5)
        assert pool is not None
        with pytest.raises(KeyError):
            evaluate_one(pool, trainer_name="random_forest", n_labels=10, seed=0)


class TestEnsembleTrainers:
    """The ``mlp_ens{N}`` factories: mean sigmoid + per-item disagreement."""

    @pytest.mark.parametrize("name", ["mlp_ens3", "mlp_ens5", "mlp_ens7", "mlp_ens10"])
    def test_registered(self, name):
        assert name in SWEEP_TRAINERS

    def test_predict_returns_scores_and_std(self):
        # A 5-member ensemble's predict() returns (mean_sigmoid, per_item_std)
        # with member disagreement strictly positive on real (noisy) inits.
        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 6)).astype(np.float32)
        X[:10, 0] += 1.5
        X[10:, 0] -= 1.5
        y = np.array([1] * 10 + [0] * 10, dtype=np.int32)
        predict = SWEEP_TRAINERS["mlp_ens5"](X, y, 0)
        out = predict(X)
        assert isinstance(out, tuple)
        scores, std = out
        assert scores.shape == (20,)
        assert std.shape == (20,)
        assert np.all(scores >= 0.0) and np.all(scores <= 1.0)
        assert np.all(std >= 0.0)
        # Distinct member seeds ⇒ some genuine disagreement somewhere.
        assert float(std.max()) > 0.0

    def test_evaluate_one_emits_std_mean(self):
        clips = _synth_dataset(n_pos=20, n_neg=20)
        pool = _build_split_pool(clips, "target", seed=0, sim_fraction=0.5)
        assert pool is not None
        row = evaluate_one(pool, trainer_name="mlp_ens3", n_labels=10, seed=0)
        assert row is not None
        assert row["trainer"] == "mlp_ens3"
        # Ensemble trainers report a finite, non-negative mean uncertainty
        # (unlike single-model trainers, whose std_mean is nan).
        assert np.isfinite(row["std_mean"])
        assert row["std_mean"] >= 0.0
        assert np.isfinite(row["std_err_auroc"])
        assert 0.0 <= row["auroc"] <= 1.0

    def test_ensemble_is_deterministic_for_fixed_seed(self):
        clips = _synth_dataset(n_pos=20, n_neg=20)
        pool = _build_split_pool(clips, "target", seed=0, sim_fraction=0.5)
        assert pool is not None
        row_a = evaluate_one(pool, trainer_name="mlp_ens3", n_labels=10, seed=1)
        row_b = evaluate_one(pool, trainer_name="mlp_ens3", n_labels=10, seed=1)
        assert row_a is not None and row_b is not None
        assert row_a["auroc"] == row_b["auroc"]
        assert row_a["std_mean"] == row_b["std_mean"]

    def test_sweep_diagnostic_columns_present(self):
        clips = _synth_dataset(n_pos=20, n_neg=20)
        df = run_label_curve_eval(
            dataset_clips={"synth": clips},
            trainers=("mlp_ens3",),
            label_counts=(10,),
            seeds=(0,),
            categories={"synth": ["target"]},
            progress=False,
        )
        assert not df.empty
        assert {"std_err_auroc", "std_mean"} <= set(df.columns)
        assert bool(df["std_mean"].notna().all())

    def test_summary_include_diagnostics_has_std_mean(self):
        clips = _synth_dataset(n_pos=20, n_neg=20)
        df = run_label_curve_eval(
            dataset_clips={"synth": clips},
            trainers=("mlp_ens3",),
            label_counts=(10,),
            seeds=(0, 1),
            categories={"synth": ["target"]},
            progress=False,
        )
        summary = summarise(df, include_diagnostics=True)
        assert "std_mean_mean" in summary.columns
        assert "std_err_auroc_mean" in summary.columns


class TestRunLabelCurveEval:
    def test_full_sweep_schema(self):
        clips = _synth_dataset(n_pos=20, n_neg=20, seed=1)
        df = run_label_curve_eval(
            dataset_clips={"synth": clips},
            trainers=("svm_linear", "mlp"),
            label_counts=(6, 10),
            seeds=(0, 1),
            progress=False,
        )
        # 1 dataset * 1 category (sim_fraction filters out 'other' as target? no;
        # 'other' has 20 entries too so it's also a valid target).
        # Categories: {target, other} → 2 categories.
        # trainers x label_counts x seeds = 2 * 2 * 2 = 8 per category → 16 rows total.
        assert len(df) == 2 * 2 * 2 * 2
        assert set(df.columns) >= {
            "dataset",
            "category",
            "trainer",
            "n_labels",
            "seed",
            "auroc",
            "average_precision",
            "best_f1",
            "xcal_threshold",
            "f1_at_xcal",
            "train_seconds",
            "brier",
            "f1_at_0.5",
        }
        assert set(df["trainer"]) == {"svm_linear", "mlp"}
        assert set(df["dataset"]) == {"synth"}

    def test_category_filter(self):
        clips = _synth_dataset(n_pos=20, n_neg=20)
        df = run_label_curve_eval(
            dataset_clips={"synth": clips},
            trainers=("svm_linear",),
            label_counts=(10,),
            seeds=(0,),
            categories={"synth": ["target"]},
            progress=False,
        )
        assert set(df["category"]) == {"target"}

    def test_unknown_trainer_raises(self):
        clips = _synth_dataset(n_pos=10, n_neg=10)
        with pytest.raises(KeyError):
            run_label_curve_eval(
                dataset_clips={"synth": clips},
                trainers=("does_not_exist",),
            )

    def test_empty_when_dataset_too_small(self):
        clips = _synth_dataset(n_pos=1, n_neg=1)
        df = run_label_curve_eval(
            dataset_clips={"tiny": clips},
            trainers=("svm_linear",),
            label_counts=(10,),
            seeds=(0,),
            progress=False,
        )
        assert df.empty
        assert set(df.columns) >= {"dataset", "trainer", "auroc"}


class TestSummarise:
    def test_summary_collapses_seeds(self):
        clips = _synth_dataset(n_pos=20, n_neg=20)
        df = run_label_curve_eval(
            dataset_clips={"synth": clips},
            trainers=("svm_linear",),
            label_counts=(10,),
            seeds=(0, 1, 2),
            categories={"synth": ["target"]},
            progress=False,
        )
        summary = summarise(df)
        # 1 dataset x 1 category x 1 trainer x 1 n_labels = 1 row
        assert len(summary) == 1
        for col in ("auroc_mean", "auroc_std", "f1_at_xcal_mean", "xcal_threshold_mean"):
            assert col in summary.columns, f"missing summary column {col!r}"

    def test_default_summary_drops_diagnostics(self):
        clips = _synth_dataset(n_pos=20, n_neg=20)
        df = run_label_curve_eval(
            dataset_clips={"synth": clips},
            trainers=("svm_linear",),
            label_counts=(10,),
            seeds=(0,),
            categories={"synth": ["target"]},
            progress=False,
        )
        summary = summarise(df)
        # Brier and F1@0.5 are diagnostics; excluded from the default
        # summary because neither is meaningful for an uncalibrated
        # ranking model with a learnable threshold.
        for col in summary.columns:
            assert "brier" not in col, f"unexpected diagnostic column {col!r}"
            assert "f1_at_0.5" not in col, f"unexpected diagnostic column {col!r}"

    def test_include_diagnostics_brings_them_back(self):
        clips = _synth_dataset(n_pos=20, n_neg=20)
        df = run_label_curve_eval(
            dataset_clips={"synth": clips},
            trainers=("svm_linear",),
            label_counts=(10,),
            seeds=(0,),
            categories={"synth": ["target"]},
            progress=False,
        )
        summary = summarise(df, include_diagnostics=True)
        assert "brier_mean" in summary.columns
        assert "f1_at_0.5_mean" in summary.columns

    def test_summary_handles_empty(self):
        clips = _synth_dataset(n_pos=1, n_neg=1)
        df = run_label_curve_eval(
            dataset_clips={"tiny": clips},
            trainers=("svm_linear",),
            label_counts=(10,),
            seeds=(0,),
            progress=False,
        )
        # summarise on empty df just returns it unchanged
        assert summarise(df).empty


class TestRegistryIntrospection:
    def test_mlp_and_svm_trainers_registered(self):
        assert "mlp" in SWEEP_TRAINERS
        assert "svm_linear" in SWEEP_TRAINERS
        assert "svm_rbf" in SWEEP_TRAINERS

    def test_ensemble_trainers_registered(self):
        for n in (3, 5, 7, 10):
            assert f"mlp_ens{n}" in SWEEP_TRAINERS
