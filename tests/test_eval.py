"""Tests for the vtsearch.eval evaluation framework.

These tests exercise the metrics, config, and runner modules using
synthetic data — no real model downloads or embeddings required.
"""

import json
from unittest.mock import patch

import numpy as np
import pytest

from vtsearch.eval.config import EVAL_DATASETS, EvalQuery
from vtsearch.eval.metrics import (
    DatasetResult,
    LearnedSortMetrics,
    QueryMetrics,
    compute_average_precision,
    compute_binary_classification_metrics,
    compute_metrics,
    compute_precision_recall_at_k,
)
from vtsearch.eval.runner import (
    _cosine_similarity,
    eval_learned_sort,
    eval_text_sort,
    format_results_json,
)


# =====================================================================
# Metrics: compute_average_precision
# =====================================================================


class TestAveragePrecision:
    def test_perfect_ranking(self):
        """All relevant items at the top -> AP = 1.0."""
        ranked = [1, 2, 3, 4, 5]
        relevant = {1, 2, 3}
        assert compute_average_precision(ranked, relevant) == pytest.approx(1.0)

    def test_worst_ranking(self):
        """All relevant items at the bottom -> AP < 1.0."""
        ranked = [4, 5, 1, 2, 3]
        relevant = {1, 2, 3}
        ap = compute_average_precision(ranked, relevant)
        assert ap < 1.0
        assert ap > 0.0

    def test_single_relevant(self):
        """One relevant item at position k -> AP = 1/k."""
        ranked = [10, 20, 30, 1, 40]
        relevant = {1}
        assert compute_average_precision(ranked, relevant) == pytest.approx(1 / 4)

    def test_no_relevant(self):
        ranked = [1, 2, 3]
        relevant: set[int] = set()
        assert compute_average_precision(ranked, relevant) == 0.0

    def test_empty_ranking(self):
        ranked: list[int] = []
        relevant = {1, 2}
        assert compute_average_precision(ranked, relevant) == 0.0

    def test_interleaved(self):
        """Relevant at positions 1, 3, 5 out of 6 items."""
        ranked = [1, 10, 2, 20, 3, 30]
        relevant = {1, 2, 3}
        # P@1 = 1/1, P@3 = 2/3, P@5 = 3/5
        expected = (1 / 1 + 2 / 3 + 3 / 5) / 3
        assert compute_average_precision(ranked, relevant) == pytest.approx(expected)


# =====================================================================
# Metrics: precision / recall at k
# =====================================================================


class TestPrecisionRecallAtK:
    def test_default_k_values(self):
        ranked = list(range(1, 101))
        relevant = set(range(1, 11))  # first 10
        p, r = compute_precision_recall_at_k(ranked, relevant)
        assert set(p.keys()) == {5, 10, 20}
        assert p[5] == pytest.approx(1.0)
        assert p[10] == pytest.approx(1.0)
        assert p[20] == pytest.approx(0.5)
        assert r[5] == pytest.approx(0.5)
        assert r[10] == pytest.approx(1.0)
        assert r[20] == pytest.approx(1.0)

    def test_custom_k(self):
        ranked = [1, 2, 3, 4, 5]
        relevant = {1, 3, 5}
        p, r = compute_precision_recall_at_k(ranked, relevant, k_values=[2, 4])
        assert p[2] == pytest.approx(1 / 2)  # {1, 2} & {1, 3, 5} = {1}
        assert p[4] == pytest.approx(2 / 4)  # {1, 2, 3, 4} & {1, 3, 5} = {1, 3}

    def test_no_relevant(self):
        ranked = [1, 2, 3]
        relevant: set[int] = set()
        p, r = compute_precision_recall_at_k(ranked, relevant, k_values=[2])
        assert p[2] == 0.0
        assert r[2] == 0.0


# =====================================================================
# Metrics: binary classification
# =====================================================================


class TestBinaryClassification:
    def test_perfect(self):
        preds = [1, 1, 0, 0]
        labels = [1, 1, 0, 0]
        acc, prec, rec, f1 = compute_binary_classification_metrics(preds, labels)
        assert acc == 1.0
        assert prec == 1.0
        assert rec == 1.0
        assert f1 == 1.0

    def test_all_wrong(self):
        preds = [0, 0, 1, 1]
        labels = [1, 1, 0, 0]
        acc, prec, rec, f1 = compute_binary_classification_metrics(preds, labels)
        assert acc == 0.0
        assert prec == 0.0
        assert rec == 0.0
        assert f1 == 0.0

    def test_mixed(self):
        preds = [1, 0, 1, 0]
        labels = [1, 1, 0, 0]
        acc, prec, rec, f1 = compute_binary_classification_metrics(preds, labels)
        assert acc == pytest.approx(0.5)
        # tp=1, fp=1, fn=1, tn=1
        assert prec == pytest.approx(0.5)
        assert rec == pytest.approx(0.5)
        assert f1 == pytest.approx(0.5)

    def test_empty(self):
        acc, prec, rec, f1 = compute_binary_classification_metrics([], [])
        assert acc == 0.0


# =====================================================================
# Metrics: compute_metrics (integration)
# =====================================================================


class TestComputeMetrics:
    def test_returns_query_metrics(self):
        ranked = [1, 2, 3, 4, 5]
        relevant = {1, 2}
        qm = compute_metrics(ranked, relevant, "test query", "test_cat", k_values=[2, 3])
        assert isinstance(qm, QueryMetrics)
        assert qm.query_text == "test query"
        assert qm.target_category == "test_cat"
        assert qm.average_precision == pytest.approx(1.0)
        assert qm.num_relevant == 2
        assert qm.num_total == 5
        assert 2 in qm.precision_at_k
        assert 3 in qm.recall_at_k


# =====================================================================
# Metrics: DatasetResult
# =====================================================================


class TestDatasetResult:
    def test_mean_average_precision(self):
        dr = DatasetResult(dataset_id="test", media_type="audio")
        dr.text_sort = [
            QueryMetrics("q1", "cat1", average_precision=0.8, num_relevant=5, num_total=20),
            QueryMetrics("q2", "cat2", average_precision=0.6, num_relevant=5, num_total=20),
        ]
        assert dr.mean_average_precision == pytest.approx(0.7)

    def test_mean_learned_f1(self):
        dr = DatasetResult(dataset_id="test", media_type="image")
        dr.learned_sort = [
            LearnedSortMetrics(accuracy=0.9, precision=0.8, recall=0.7, f1=0.75, num_train=10, num_test=10),
            LearnedSortMetrics(accuracy=0.85, precision=0.8, recall=0.9, f1=0.85, num_train=10, num_test=10),
        ]
        assert dr.mean_learned_f1 == pytest.approx(0.8)

    def test_empty_results(self):
        dr = DatasetResult(dataset_id="empty", media_type="audio")
        assert dr.mean_average_precision == 0.0
        assert dr.mean_learned_f1 == 0.0

    def test_to_dict_has_expected_keys(self):
        dr = DatasetResult(dataset_id="test", media_type="audio")
        dr.text_sort = [
            QueryMetrics(
                "q1",
                "cat1",
                average_precision=0.8,
                num_relevant=5,
                num_total=20,
                precision_at_k={5: 0.6},
                recall_at_k={5: 0.3},
            ),
        ]
        d = dr.to_dict()
        assert d["dataset_id"] == "test"
        assert d["media_type"] == "audio"
        assert "text_sort" in d
        assert d["text_sort"]["mAP"] == 0.8
        assert len(d["text_sort"]["per_query"]) == 1

    def test_to_dict_learned(self):
        dr = DatasetResult(dataset_id="test", media_type="image")
        dr.learned_sort = [
            LearnedSortMetrics(
                accuracy=0.9, precision=0.8, recall=0.7, f1=0.75, num_train=10, num_test=10, target_category="cat1"
            ),
        ]
        d = dr.to_dict()
        assert "learned_sort" in d
        assert d["learned_sort"]["mean_f1"] == 0.75


# =====================================================================
# Config
# =====================================================================


class TestEvalConfig:
    def test_all_eval_datasets_have_queries(self):
        for ds_id, ds_cfg in EVAL_DATASETS.items():
            assert "queries" in ds_cfg, f"{ds_id} missing queries"
            assert len(ds_cfg["queries"]) > 0, f"{ds_id} has no queries"
            for q in ds_cfg["queries"]:
                assert isinstance(q, EvalQuery)
                assert q.text.strip(), f"{ds_id}: empty query text"
                assert q.target_category.strip(), f"{ds_id}: empty target_category"

    def test_all_eval_datasets_reference_demo_datasets(self):
        from vtsearch.datasets.config import DEMO_DATASETS

        for ds_id, ds_cfg in EVAL_DATASETS.items():
            demo_id = ds_cfg["demo_dataset"]
            assert demo_id in DEMO_DATASETS, f"eval {ds_id} references missing demo dataset {demo_id}"

    def test_query_categories_match_demo_categories(self):
        """Every query's target_category must appear in the demo dataset's category list."""
        from vtsearch.datasets.config import DEMO_DATASETS

        for ds_id, ds_cfg in EVAL_DATASETS.items():
            demo_id = ds_cfg["demo_dataset"]
            demo_cats = set(DEMO_DATASETS[demo_id]["categories"])
            for q in ds_cfg["queries"]:
                assert q.target_category in demo_cats, (
                    f"eval {ds_id}: query target {q.target_category!r} not in demo categories {demo_cats}"
                )


# =====================================================================
# Runner: _cosine_similarity
# =====================================================================


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = np.array([1.0, 0.0, 0.0])
        assert _cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert _cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert _cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector(self):
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 0.0])
        assert _cosine_similarity(a, b) == 0.0


# =====================================================================
# Runner: eval_text_sort with synthetic clips
# =====================================================================


class TestEvalTextSort:
    """Test eval_text_sort using mocked embeddings."""

    def _make_synthetic_clips(self):
        """Create clips with known embeddings: cats point one way, dogs another."""
        rng = np.random.RandomState(0)
        clips = {}
        clip_id = 1
        # "cat" clips cluster around [1, 0, 0, ...]
        cat_dir = np.zeros(16)
        cat_dir[0] = 1.0
        for _ in range(10):
            emb = cat_dir + rng.normal(0, 0.05, 16)
            emb /= np.linalg.norm(emb)
            clips[clip_id] = {"id": clip_id, "embedding": emb, "category": "cat", "type": "image"}
            clip_id += 1
        # "dog" clips cluster around [0, 1, 0, ...]
        dog_dir = np.zeros(16)
        dog_dir[1] = 1.0
        for _ in range(10):
            emb = dog_dir + rng.normal(0, 0.05, 16)
            emb /= np.linalg.norm(emb)
            clips[clip_id] = {"id": clip_id, "embedding": emb, "category": "dog", "type": "image"}
            clip_id += 1
        return clips, cat_dir, dog_dir

    def test_text_sort_separates_categories(self):
        clips, cat_dir, dog_dir = self._make_synthetic_clips()

        queries = [
            EvalQuery("a cat", "cat"),
            EvalQuery("a dog", "dog"),
        ]

        # Mock embed_text_query to return the cluster centre
        def mock_embed(text, media_type, enrich=False):
            if "cat" in text:
                return cat_dir.copy()
            return dog_dir.copy()

        with patch("vtsearch.models.embeddings.embed_text_query", side_effect=mock_embed):
            results = eval_text_sort(clips, queries, "image", k_values=[5, 10])

        assert len(results) == 2
        # With clean clusters, AP should be very high
        for qm in results:
            assert qm.average_precision > 0.9
            assert qm.num_relevant == 10

    def test_text_sort_returns_correct_fields(self):
        clips, cat_dir, _ = self._make_synthetic_clips()
        queries = [EvalQuery("a cat", "cat")]

        with patch("vtsearch.models.embeddings.embed_text_query", return_value=cat_dir.copy()):
            results = eval_text_sort(clips, queries, "image", k_values=[5])

        qm = results[0]
        assert qm.query_text == "a cat"
        assert qm.target_category == "cat"
        assert 5 in qm.precision_at_k
        assert 5 in qm.recall_at_k
        assert qm.num_total == 20


# =====================================================================
# Runner: eval_learned_sort with synthetic clips
# =====================================================================


class TestEvalLearnedSort:
    def _make_synthetic_clips(self, dim=16, n_per_cat=20):
        """Two categories with separable embeddings."""
        rng = np.random.RandomState(42)
        clips = {}
        clip_id = 1
        for _ in range(n_per_cat):
            emb = rng.normal(1.0, 0.3, dim).astype(np.float32)
            clips[clip_id] = {"id": clip_id, "embedding": emb, "category": "cat_a", "type": "image"}
            clip_id += 1
        for _ in range(n_per_cat):
            emb = rng.normal(-1.0, 0.3, dim).astype(np.float32)
            clips[clip_id] = {"id": clip_id, "embedding": emb, "category": "cat_b", "type": "image"}
            clip_id += 1
        return clips

    def test_learned_sort_returns_metrics(self):
        clips = self._make_synthetic_clips()
        queries = [EvalQuery("category a stuff", "cat_a")]
        results = eval_learned_sort(clips, queries, train_fraction=0.5, seed=42)
        assert len(results) == 1
        lm = results[0]
        assert lm.target_category == "cat_a"
        assert 0.0 <= lm.accuracy <= 1.0
        assert 0.0 <= lm.f1 <= 1.0
        assert lm.num_train > 0
        assert lm.num_test > 0

    def test_learned_sort_well_separated_categories(self):
        """With well-separated embeddings, learned sort should get high F1."""
        clips = self._make_synthetic_clips(n_per_cat=30)
        queries = [EvalQuery("a", "cat_a")]
        results = eval_learned_sort(clips, queries, train_fraction=0.5, seed=42)
        assert results[0].f1 > 0.7  # generous threshold for small synthetic data

    def test_learned_sort_skips_tiny_categories(self):
        """Categories with < 2 clips should be skipped."""
        clips = {
            1: {"id": 1, "embedding": np.ones(8, dtype=np.float32), "category": "rare", "type": "image"},
            2: {"id": 2, "embedding": -np.ones(8, dtype=np.float32), "category": "common", "type": "image"},
        }
        queries = [EvalQuery("rare stuff", "rare")]
        results = eval_learned_sort(clips, queries, train_fraction=0.5)
        assert len(results) == 0  # skipped due to too few clips


# =====================================================================
# Runner: format_results_json
# =====================================================================


class TestFormatResults:
    def test_valid_json(self):
        dr = DatasetResult(dataset_id="test", media_type="audio")
        dr.text_sort = [
            QueryMetrics(
                "q1",
                "cat1",
                average_precision=0.9,
                num_relevant=5,
                num_total=20,
                precision_at_k={5: 0.8},
                recall_at_k={5: 0.4},
            ),
        ]
        result = format_results_json([dr])
        parsed = json.loads(result)
        assert isinstance(parsed, list)
        assert len(parsed) == 1
        assert parsed[0]["dataset_id"] == "test"
