"""Tests for the read-only indicator-history path and its per-step cache costs.

The progress-plot modal used to load its series through the
``calculate_*_over_time`` helpers, which call ``_ensure_cache`` and therefore
retrain one MLP per uncached label step - plus, on a cold cache, a full
hierarchical-k-means coverage-atlas build - on the calling thread while holding
``_progress_lock``.  ``/api/labeling-status`` deliberately defers exactly that
work to a background worker (issue #2397), so the cache is behind for most of a
labeling session and the modal absorbed the whole deferred build.

These tests pin the three pieces that fixed it:

* :func:`cached_indicator_history` reads the cache and **never advances it**,
  reporting ``complete=False`` instead, and never blocks on ``_progress_lock``.
* The fallback coverage-atlas build applies the same depth cap as every other
  build site.

``complete`` means "the cache is caught up", never "every label step has a
point": since #3757 only the steps the app trained a detector for carry one, so
these tests inject live models where they need a series to be non-empty, the
same way a learned sort does.
"""

from __future__ import annotations

import threading

import numpy as np

import vtscore.detectors.labeling_progress as lp
from vtscore.embedding.media_vectors import EMBEDDINGS_KEY


def _active_cache() -> lp._ProgressCache:
    """The cache for the pair these tests run under, as an entry point would resolve it."""
    with lp._progress_lock:
        return lp._active_cache()


def _inject_models(history: list[tuple[int, str, float]]) -> None:
    """Give every label set in *history* a live model, as a sort per vote would.

    Only steps whose label set had a sort run against it carry a model (#3757),
    so a test that wants a point per step has to say so.
    """
    good: dict[int, None] = {}
    bad: dict[int, None] = {}
    for mid, label, _ in history:
        (good if label == "good" else bad)[mid] = None
        if good and bad:
            lp.inject_live_model(dict(good), dict(bad), _linear_model(), 0.5)


def _linear_model(dim: int = 8):
    """A tiny fixed linear model producing ``(n, 1)`` logits."""
    import torch
    import torch.nn as nn

    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(dim, 1))


def _clips(n: int, dim: int = 8, seed: int = 0) -> dict[int, dict]:
    rng = np.random.default_rng(seed)
    return {
        cid: {EMBEDDINGS_KEY: {"test": rng.standard_normal(dim).astype(np.float32)}, "embedder": "test"}
        for cid in range(n)
    }


def _history(n_votes: int) -> list[tuple[int, str, float]]:
    return [(k, "good" if k % 2 == 0 else "bad", float(k)) for k in range(n_votes)]


def _votes(n_votes: int) -> tuple[dict[int, None], dict[int, None]]:
    good = {k: None for k in range(n_votes) if k % 2 == 0}
    bad = {k: None for k in range(n_votes) if k % 2 == 1}
    return good, bad


class TestCachedIndicatorHistory:
    def test_cold_cache_reports_incomplete_without_advancing(self):
        """The whole point: a cold read must not trigger any per-step training."""
        clips = _clips(60)
        history = _history(8)
        good, bad = _votes(8)
        lp.clear_progress_cache()

        data, complete = lp.cached_indicator_history("smart", clips, history, good, bad, 0)

        assert complete is False
        assert data == []
        # No steps were built: the cache is exactly as cold as we left it.
        assert _active_cache().steps == []

    def test_partially_advanced_cache_reports_incomplete(self):
        """A cache covering only a prefix must not yield a truncated plot."""
        clips = _clips(60)
        good, bad = _votes(8)
        lp.clear_progress_cache()

        lp.calculate_error_cost_over_time(clips, _history(4), good, bad, 0)
        # More votes have landed since the last background refresh.
        data, complete = lp.cached_indicator_history("smart", clips, _history(8), good, bad, 0)

        assert complete is False
        assert data == []

    def test_inclusion_change_reports_incomplete(self):
        """A cache built for another inclusion value would be rebuilt on read."""
        clips = _clips(60)
        history = _history(8)
        good, bad = _votes(8)
        lp.clear_progress_cache()

        lp.calculate_error_cost_over_time(clips, history, good, bad, 0)
        _, complete = lp.cached_indicator_history("smart", clips, history, good, bad, 5)

        assert complete is False
        # The read must not have rebuilt the cache under the new inclusion.
        assert _active_cache().inclusion == 0

    def test_does_not_block_on_an_in_flight_cache_build(self, monkeypatch):
        """A click landing mid-refresh falls through instead of hanging.

        The background worker holds ``_progress_lock`` for the entire build, so
        a blocking read would reintroduce exactly the hang this path exists to
        avoid.
        """
        monkeypatch.setattr(lp, "_CACHE_READ_LOCK_TIMEOUT", 0.05)
        clips = _clips(60)
        history = _history(8)
        good, bad = _votes(8)
        lp.clear_progress_cache()
        lp.calculate_error_cost_over_time(clips, history, good, bad, 0)

        holding = threading.Event()
        release = threading.Event()

        def _hold_lock():
            with lp._progress_lock:
                holding.set()
                release.wait(timeout=5)

        holder = threading.Thread(target=_hold_lock, daemon=True)
        holder.start()
        try:
            assert holding.wait(timeout=5)
            # The cache is complete, but the lock is held: report unavailable
            # rather than waiting for the holder.
            data, complete = lp.cached_indicator_history("smart", clips, history, good, bad, 0)
            assert complete is False
            assert data == []
        finally:
            release.set()
            holder.join(timeout=5)

        # Once the lock frees up the same read succeeds.
        _, complete = lp.cached_indicator_history("smart", clips, history, good, bad, 0)
        assert complete is True


class TestSeriesCoverTrainedStepsOnly:
    def test_every_metric_has_a_point_when_every_step_was_sorted(self):
        """With a sort behind each label set, Smart and Stable are dense again."""
        clips = _clips(60)
        history = _history(8)
        good, bad = _votes(8)
        lp.clear_progress_cache()
        _inject_models(history)

        lp.calculate_error_cost_over_time(clips, history, good, bad, 0)

        for metric in ("smart", "stable", "diverse"):
            data, complete = lp.cached_indicator_history(metric, clips, history, good, bad, 0)
            assert complete is True, metric
            assert len(data) > 0, metric

    def test_complete_but_empty_when_nothing_was_ever_sorted(self):
        """A caught-up cache with no trained detector reports an empty Smart series.

        Not ``complete=False``: there is nothing left to compute, and sending the
        modal to the async job would only have it recompute the same emptiness.
        Diversity is unaffected - it measures the votes, not a detector.
        """
        clips = _clips(60)
        history = _history(8)
        good, bad = _votes(8)
        lp.clear_progress_cache()

        lp.calculate_error_cost_over_time(clips, history, good, bad, 0)

        for metric in ("smart", "stable"):
            data, complete = lp.cached_indicator_history(metric, clips, history, good, bad, 0)
            assert complete is True, metric
            assert data == [], metric
        diverse, complete = lp.cached_indicator_history("diverse", clips, history, good, bad, 0)
        assert complete is True
        assert len(diverse) > 0


class TestFallbackAtlasDepth:
    def test_fallback_build_applies_the_depth_cap(self, monkeypatch):
        """The fallback atlas must not be deeper than the context atlas it stands in for.

        Omitting ``max_depth`` left this build on ``COVERAGE_ATLAS_MAX_DEPTH``
        while every other build site passes ``auto_max_depth``, so the throwaway
        atlas cost many more k-means fits than the real one.
        """
        import vtscore.coverage.atlas as ca

        monkeypatch.setattr(lp, "_active_context_atlas", lambda: None)
        monkeypatch.setattr(ca, "auto_max_depth", lambda n, k=3, **kw: 2)

        atlas = lp._build_coverage_atlas(_clips(120))

        assert atlas is not None
        assert atlas.max_depth == 2

    def test_reuses_the_context_atlas_structure_when_ids_match(self, monkeypatch):
        """The clone path skips the hierarchical k-means entirely."""
        from vtscore.coverage.atlas import CoverageAtlas

        clips = _clips(120)
        vectors = {cid: media[EMBEDDINGS_KEY]["test"] for cid, media in clips.items()}
        ctx_atlas = CoverageAtlas(vectors, k=3)
        monkeypatch.setattr(lp, "_active_context_atlas", lambda: ctx_atlas)

        atlas = lp._build_coverage_atlas(clips)

        assert atlas is not ctx_atlas
        # Structure shared by reference; only the label overlay is fresh.
        assert atlas.nodes is ctx_atlas.nodes
