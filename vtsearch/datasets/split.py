"""Dataset splitting utilities for evaluation."""

import hashlib
import random
from collections import defaultdict
from typing import Any


def _category_seed(seed: int, category: str) -> int:
    """Derive a per-category seed so each category is shuffled independently."""
    h = hashlib.sha256(f"{seed}:{category}".encode()).digest()
    return int.from_bytes(h[:8], "big")


def split_dataset(
    clips: dict[int, dict[str, Any]],
    test_fraction: float,
    seed: int,
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    """Split a clips dict into simulation and test sets, stratified by category.

    Each category is split independently using the same fraction, so if
    ``test_fraction=0.2`` then roughly 20% of each category ends up in the
    test set and 80% in the simulation set.  Each category uses its own
    deterministic RNG derived from ``seed`` and the category name, so
    adding or removing categories does not change the split of other
    categories.

    Clip IDs are preserved (not renumbered) in both output dicts.

    Args:
        clips: Mapping of clip ID to clip data dict.  Every clip must have a
            ``"category"`` key.
        test_fraction: Fraction of each category to allocate to the test set.
            Must be in ``(0, 1)``.
        seed: Integer seed used to derive per-category random states for
            reproducible shuffling.

    Returns:
        A 2-tuple ``(simulate_clips, test_clips)`` where each element is a
        dict with the same structure as ``clips``.

    Raises:
        ValueError: If ``test_fraction`` is not in ``(0, 1)`` or if ``clips``
            is empty.
    """
    if not 0 < test_fraction < 1:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")
    if not clips:
        raise ValueError("clips dict is empty")

    # Group clip IDs by category
    by_category: dict[str, list[int]] = defaultdict(list)
    for clip_id, clip in clips.items():
        by_category[clip["category"]].append(clip_id)

    simulate_clips: dict[int, dict[str, Any]] = {}
    test_clips: dict[int, dict[str, Any]] = {}

    for category in sorted(by_category):
        ids = by_category[category]
        ids.sort()  # deterministic order before shuffle

        rng = random.Random(_category_seed(seed, category))
        rng.shuffle(ids)

        n_test = round(len(ids) * test_fraction)
        # Ensure at least 1 in each split when the category is large enough
        if n_test == 0 and len(ids) >= 2:
            n_test = 1
        if n_test == len(ids) and len(ids) >= 2:
            n_test = len(ids) - 1

        test_ids = ids[:n_test]
        simulate_ids = ids[n_test:]

        for cid in test_ids:
            test_clips[cid] = clips[cid]
        for cid in simulate_ids:
            simulate_clips[cid] = clips[cid]

    return simulate_clips, test_clips
