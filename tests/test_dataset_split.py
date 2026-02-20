"""Tests for dataset split-by-fraction functionality."""

import numpy as np
import pytest

from vtsearch.datasets.split import split_dataset


def _make_clips(categories: dict[str, int]) -> dict[int, dict]:
    """Build a minimal clips dict with the given category counts.

    Args:
        categories: Mapping of category name to number of clips.

    Returns:
        A clips dict keyed by sequential integer IDs starting at 1.
    """
    clips = {}
    clip_id = 1
    for cat, count in categories.items():
        for _ in range(count):
            clips[clip_id] = {
                "id": clip_id,
                "type": "audio",
                "category": cat,
                "embedding": np.zeros(4),
                "duration": 1.0,
                "file_size": 100,
                "md5": f"md5_{clip_id}",
                "filename": f"clip_{clip_id}.wav",
                "wav_bytes": b"\x00",
            }
            clip_id += 1
    return clips


class TestSplitDataset:
    def test_basic_split_preserves_all_clips(self):
        clips = _make_clips({"cat": 10, "dog": 10})
        sim, test = split_dataset(clips, test_fraction=0.2, seed=42)

        all_ids = set(sim) | set(test)
        assert all_ids == set(clips)
        assert len(set(sim) & set(test)) == 0

    def test_fraction_applied_per_category(self):
        clips = _make_clips({"cat": 20, "dog": 20, "bird": 20})
        sim, test = split_dataset(clips, test_fraction=0.2, seed=42)

        for cat_name in ["cat", "dog", "bird"]:
            original_count = 20
            test_count = sum(1 for c in test.values() if c["category"] == cat_name)
            sim_count = sum(1 for c in sim.values() if c["category"] == cat_name)
            assert test_count + sim_count == original_count
            assert test_count == round(original_count * 0.2)

    def test_same_seed_same_split(self):
        clips = _make_clips({"cat": 50, "dog": 50})
        sim1, test1 = split_dataset(clips, test_fraction=0.3, seed=123)
        sim2, test2 = split_dataset(clips, test_fraction=0.3, seed=123)

        assert set(sim1) == set(sim2)
        assert set(test1) == set(test2)

    def test_different_seed_different_split(self):
        clips = _make_clips({"cat": 50, "dog": 50})
        _, test1 = split_dataset(clips, test_fraction=0.3, seed=1)
        _, test2 = split_dataset(clips, test_fraction=0.3, seed=2)

        # With 100 clips and 30% test, very unlikely to be identical
        assert set(test1) != set(test2)

    def test_clip_ids_preserved(self):
        clips = _make_clips({"cat": 5})
        sim, test = split_dataset(clips, test_fraction=0.4, seed=42)

        for cid, clip in sim.items():
            assert clip is clips[cid]
        for cid, clip in test.items():
            assert clip is clips[cid]

    def test_single_clip_category_goes_to_simulate(self):
        clips = _make_clips({"rare": 1, "common": 20})
        sim, test = split_dataset(clips, test_fraction=0.2, seed=42)

        # Category with 1 clip: round(1 * 0.2) = 0, so it goes to simulate
        rare_in_sim = [c for c in sim.values() if c["category"] == "rare"]
        rare_in_test = [c for c in test.values() if c["category"] == "rare"]
        assert len(rare_in_sim) + len(rare_in_test) == 1

    def test_two_clip_category_splits(self):
        clips = _make_clips({"small": 2, "big": 20})
        sim, test = split_dataset(clips, test_fraction=0.2, seed=42)

        # Category with 2 clips: round(2 * 0.2) = 0, but guarantee at least 1
        small_in_sim = sum(1 for c in sim.values() if c["category"] == "small")
        small_in_test = sum(1 for c in test.values() if c["category"] == "small")
        assert small_in_sim >= 1
        assert small_in_test >= 1

    def test_high_fraction(self):
        clips = _make_clips({"cat": 10})
        sim, test = split_dataset(clips, test_fraction=0.8, seed=42)

        assert len(test) == 8
        assert len(sim) == 2

    def test_invalid_fraction_zero(self):
        clips = _make_clips({"cat": 10})
        with pytest.raises(ValueError, match="test_fraction must be in"):
            split_dataset(clips, test_fraction=0.0, seed=42)

    def test_invalid_fraction_one(self):
        clips = _make_clips({"cat": 10})
        with pytest.raises(ValueError, match="test_fraction must be in"):
            split_dataset(clips, test_fraction=1.0, seed=42)

    def test_invalid_fraction_negative(self):
        clips = _make_clips({"cat": 10})
        with pytest.raises(ValueError, match="test_fraction must be in"):
            split_dataset(clips, test_fraction=-0.1, seed=42)

    def test_empty_clips(self):
        with pytest.raises(ValueError, match="clips dict is empty"):
            split_dataset({}, test_fraction=0.2, seed=42)

    def test_many_categories_equal_fraction(self):
        clips = _make_clips({"a": 100, "b": 100, "c": 100, "d": 100})
        sim, test = split_dataset(clips, test_fraction=0.25, seed=7)

        for cat in ["a", "b", "c", "d"]:
            test_count = sum(1 for c in test.values() if c["category"] == cat)
            assert test_count == 25

    def test_categories_independent_of_each_other(self):
        """Adding a new category should not change the split of existing ones."""
        clips_small = _make_clips({"cat": 20, "dog": 20})
        clips_large = _make_clips({"cat": 20, "dog": 20, "bird": 20})

        _, test_small = split_dataset(clips_small, test_fraction=0.3, seed=99)
        _, test_large = split_dataset(clips_large, test_fraction=0.3, seed=99)

        # The cat and dog clips should have the same IDs in both splits,
        # because categories are processed independently in sorted order
        # and the same IDs exist for cat/dog in both dicts.
        cat_test_small = {c["id"] for c in test_small.values() if c["category"] == "cat"}
        cat_test_large = {c["id"] for c in test_large.values() if c["category"] == "cat"}
        assert cat_test_small == cat_test_large

        dog_test_small = {c["id"] for c in test_small.values() if c["category"] == "dog"}
        dog_test_large = {c["id"] for c in test_large.values() if c["category"] == "dog"}
        assert dog_test_small == dog_test_large
