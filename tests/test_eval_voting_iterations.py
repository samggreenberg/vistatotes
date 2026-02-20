"""Tests for the voting-iterations evaluation framework.

All tests use small synthetic datasets with known, well-separated
embeddings so no real model downloads are needed.
"""

import numpy as np
import pandas as pd

from vtsearch.eval.voting_iterations import (
    _inclusion_weights,
    _make_vote_sequence,
    _split_clip_ids,
    run_voting_iterations_eval,
    simulate_voting_iterations,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


def _make_separable_clips(dim=16, n_per_cat=20, seed=0):
    """Two categories with well-separated embeddings.

    Category "alpha" clusters around [+1, 0, 0, ...],
    category "beta"  clusters around [-1, 0, 0, ...].
    """
    rng = np.random.RandomState(seed)
    clips = {}
    clip_id = 1
    for _ in range(n_per_cat):
        emb = rng.normal(1.0, 0.2, dim).astype(np.float32)
        clips[clip_id] = {"id": clip_id, "embedding": emb, "category": "alpha"}
        clip_id += 1
    for _ in range(n_per_cat):
        emb = rng.normal(-1.0, 0.2, dim).astype(np.float32)
        clips[clip_id] = {"id": clip_id, "embedding": emb, "category": "beta"}
        clip_id += 1
    return clips


def _make_overlapping_clips(dim=16, n_per_cat=20, seed=0):
    """Two categories with overlapping embeddings (harder to classify).

    Category "alpha" centred at [+0.3, 0, 0, ...],
    category "beta"  centred at [-0.3, 0, 0, ...], with large noise.
    """
    rng = np.random.RandomState(seed)
    clips = {}
    clip_id = 1
    for _ in range(n_per_cat):
        emb = rng.normal(0.3, 1.0, dim).astype(np.float32)
        clips[clip_id] = {"id": clip_id, "embedding": emb, "category": "alpha"}
        clip_id += 1
    for _ in range(n_per_cat):
        emb = rng.normal(-0.3, 1.0, dim).astype(np.float32)
        clips[clip_id] = {"id": clip_id, "embedding": emb, "category": "beta"}
        clip_id += 1
    return clips


def _make_three_category_clips(dim=16, n_per_cat=15, seed=0):
    """Three categories: alpha, beta, gamma."""
    rng = np.random.RandomState(seed)
    clips = {}
    clip_id = 1
    centres = {"alpha": 1.0, "beta": -1.0, "gamma": 0.0}
    for cat, centre in centres.items():
        for _ in range(n_per_cat):
            emb = rng.normal(centre, 0.2, dim).astype(np.float32)
            clips[clip_id] = {"id": clip_id, "embedding": emb, "category": cat}
            clip_id += 1
    return clips


# ------------------------------------------------------------------
# Unit tests: helpers
# ------------------------------------------------------------------


class TestInclusionWeights:
    def test_zero_inclusion(self):
        fpr_w, fnr_w = _inclusion_weights(0)
        assert fpr_w == 1.0
        assert fnr_w == 1.0

    def test_positive_inclusion(self):
        fpr_w, fnr_w = _inclusion_weights(3)
        assert fpr_w == 1.0
        assert fnr_w == 8.0

    def test_negative_inclusion(self):
        fpr_w, fnr_w = _inclusion_weights(-2)
        assert fpr_w == 4.0
        assert fnr_w == 1.0


class TestSplitClipIds:
    def test_split_sizes(self):
        clips = _make_separable_clips(n_per_cat=10)
        rng = np.random.RandomState(42)
        sim, test = _split_clip_ids(clips, 0.5, rng)
        assert len(sim) + len(test) == len(clips)
        assert len(sim) == 10
        assert len(test) == 10

    def test_no_overlap(self):
        clips = _make_separable_clips(n_per_cat=10)
        rng = np.random.RandomState(42)
        sim, test = _split_clip_ids(clips, 0.5, rng)
        assert set(sim).isdisjoint(set(test))

    def test_deterministic(self):
        clips = _make_separable_clips(n_per_cat=10)
        rng1 = np.random.RandomState(42)
        sim1, test1 = _split_clip_ids(clips, 0.5, rng1)
        rng2 = np.random.RandomState(42)
        sim2, test2 = _split_clip_ids(clips, 0.5, rng2)
        assert sim1 == sim2
        assert test1 == test2


class TestMakeVoteSequence:
    def test_all_clips_voted(self):
        clips = _make_separable_clips(n_per_cat=5)
        sim_ids = list(clips.keys())[:5]
        rng = np.random.RandomState(42)
        seq = _make_vote_sequence(sim_ids, clips, "alpha", rng)
        assert len(seq) == 5
        assert {cid for cid, _ in seq} == set(sim_ids)

    def test_labels_match_category(self):
        clips = _make_separable_clips(n_per_cat=5)
        sim_ids = list(clips.keys())
        rng = np.random.RandomState(42)
        seq = _make_vote_sequence(sim_ids, clips, "alpha", rng)
        for cid, label in seq:
            expected = "good" if clips[cid]["category"] == "alpha" else "bad"
            assert label == expected


# ------------------------------------------------------------------
# Unit tests: simulate_voting_iterations
# ------------------------------------------------------------------


class TestSimulateVotingIterations:
    def test_returns_rows(self):
        clips = _make_separable_clips(n_per_cat=10)
        rows = simulate_voting_iterations(
            clips, target_category="alpha", seed=42,
            dataset_name="test_ds", inclusion=0, sim_fraction=0.5,
        )
        assert len(rows) > 0

    def test_row_schema(self):
        clips = _make_separable_clips(n_per_cat=10)
        rows = simulate_voting_iterations(
            clips, target_category="alpha", seed=42,
            dataset_name="test_ds",
        )
        expected_keys = {"seed", "dataset", "category", "t", "cost", "fpr", "fnr"}
        for row in rows:
            assert set(row.keys()) == expected_keys

    def test_seed_determinism(self):
        clips = _make_separable_clips(n_per_cat=10)
        rows1 = simulate_voting_iterations(clips, "alpha", seed=42)
        rows2 = simulate_voting_iterations(clips, "alpha", seed=42)
        assert len(rows1) == len(rows2)
        for r1, r2 in zip(rows1, rows2):
            assert r1 == r2

    def test_different_seeds_differ(self):
        clips = _make_separable_clips(n_per_cat=10)
        rows1 = simulate_voting_iterations(clips, "alpha", seed=42)
        rows2 = simulate_voting_iterations(clips, "alpha", seed=99)
        # Different seeds should produce different vote orderings / splits,
        # so the t-indexed costs should differ (not guaranteed for every row,
        # but at least the full sequence should differ).
        costs1 = [r["cost"] for r in rows1]
        costs2 = [r["cost"] for r in rows2]
        assert costs1 != costs2

    def test_t_values_monotonically_increase(self):
        clips = _make_separable_clips(n_per_cat=10)
        rows = simulate_voting_iterations(clips, "alpha", seed=42)
        t_vals = [r["t"] for r in rows]
        assert t_vals == sorted(t_vals)
        # t starts >=2 because we need at least 1 good + 1 bad
        assert all(t >= 2 for t in t_vals)

    def test_cost_decreases_over_time_for_overlapping_data(self):
        """With overlapping data, cost should generally decrease as more votes come in."""
        clips = _make_overlapping_clips(n_per_cat=30, dim=16)
        rows = simulate_voting_iterations(
            clips, "alpha", seed=42, sim_fraction=0.5,
        )
        costs = [r["cost"] for r in rows]
        # Compare average of first quarter vs last quarter
        n = len(costs)
        q = max(1, n // 4)
        early_avg = sum(costs[:q]) / q
        late_avg = sum(costs[-q:]) / q
        assert late_avg <= early_avg

    def test_empty_when_no_test_positives(self):
        """If all clips of target category land in sim, test set has no positives -> empty."""
        # Only 1 clip of target category — likely all end up in sim with 50% split
        clips = {
            1: {"id": 1, "embedding": np.ones(8, dtype=np.float32), "category": "rare"},
            2: {"id": 2, "embedding": -np.ones(8, dtype=np.float32), "category": "common"},
            3: {"id": 3, "embedding": -np.ones(8, dtype=np.float32) * 0.9, "category": "common"},
            4: {"id": 4, "embedding": -np.ones(8, dtype=np.float32) * 0.8, "category": "common"},
        }
        rows = simulate_voting_iterations(clips, "rare", seed=42, sim_fraction=0.5)
        # Might be empty or not depending on split — just shouldn't crash
        assert isinstance(rows, list)

    def test_inclusion_affects_cost(self):
        """With overlapping data, different inclusion values produce different costs."""
        clips = _make_overlapping_clips(n_per_cat=20)
        rows_inc0 = simulate_voting_iterations(clips, "alpha", seed=42, inclusion=0)
        rows_inc5 = simulate_voting_iterations(clips, "alpha", seed=42, inclusion=5)
        # Same splits but different inclusion -> costs should differ
        costs0 = [r["cost"] for r in rows_inc0]
        costs5 = [r["cost"] for r in rows_inc5]
        assert costs0 != costs5


# ------------------------------------------------------------------
# Integration test: run_voting_iterations_eval
# ------------------------------------------------------------------


class TestRunVotingIterationsEval:
    def test_returns_dataframe(self):
        clips = _make_separable_clips(n_per_cat=10)
        df = run_voting_iterations_eval(
            dataset_clips={"ds1": clips},
            seeds=[42],
            categories={"ds1": ["alpha"]},
        )
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["seed", "dataset", "category", "t", "cost", "fpr", "fnr"]

    def test_multiple_seeds(self):
        clips = _make_separable_clips(n_per_cat=10)
        df = run_voting_iterations_eval(
            dataset_clips={"ds1": clips},
            seeds=[1, 2, 3],
            categories={"ds1": ["alpha"]},
        )
        assert set(df["seed"].unique()) == {1, 2, 3}

    def test_multiple_categories(self):
        clips = _make_separable_clips(n_per_cat=10)
        df = run_voting_iterations_eval(
            dataset_clips={"ds1": clips},
            seeds=[42],
            categories={"ds1": ["alpha", "beta"]},
        )
        assert set(df["category"].unique()) == {"alpha", "beta"}

    def test_auto_categories(self):
        """When categories=None, all unique categories are used."""
        clips = _make_three_category_clips(n_per_cat=10)
        df = run_voting_iterations_eval(
            dataset_clips={"ds1": clips},
            seeds=[42],
        )
        assert set(df["category"].unique()) == {"alpha", "beta", "gamma"}

    def test_multiple_datasets(self):
        clips1 = _make_separable_clips(n_per_cat=10, seed=0)
        clips2 = _make_separable_clips(n_per_cat=10, seed=1)
        df = run_voting_iterations_eval(
            dataset_clips={"ds1": clips1, "ds2": clips2},
            seeds=[42],
            categories={"ds1": ["alpha"], "ds2": ["beta"]},
        )
        assert set(df["dataset"].unique()) == {"ds1", "ds2"}

    def test_cost_column_numeric(self):
        clips = _make_separable_clips(n_per_cat=10)
        df = run_voting_iterations_eval(
            dataset_clips={"ds1": clips},
            seeds=[42],
            categories={"ds1": ["alpha"]},
        )
        assert df["cost"].dtype == np.float64
        assert df["fpr"].dtype == np.float64
        assert df["fnr"].dtype == np.float64

    def test_full_cross_product_shape(self):
        """2 seeds x 1 dataset x 2 categories -> each combo produces rows."""
        clips = _make_separable_clips(n_per_cat=10)
        df = run_voting_iterations_eval(
            dataset_clips={"ds1": clips},
            seeds=[1, 2],
            categories={"ds1": ["alpha", "beta"]},
        )
        combos = df.groupby(["seed", "dataset", "category"]).ngroups
        assert combos == 4  # 2 seeds x 1 dataset x 2 categories
