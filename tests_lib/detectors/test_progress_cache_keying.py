"""Tests for the ``(dataset, detector)`` keying of the labeling-progress cache.

All per-step progress state lives in ``_ProgressCache`` objects held in
``_caches``.  Two things about that map are load-bearing and easy to break:

* Every entry point resolves its cache through :func:`_active_cache`, which is
  what makes the pair identity structural.  It used to be a module-global slot
  plus a ``_bind_cache_identity()`` call every entry point had to remember, and
  forgetting it served one detector's models as another's (issue #2914).
* The map is LRU-bounded, and the stability pools beside it are keyed by
  dataset alone so several warm pairs do not multiply memory.  A pool must
  survive while any cache still refers to its dataset, and go when none does.
"""

from __future__ import annotations

import vtscore.detectors.labeling_progress as lp


def _seed(key: tuple[str, str]) -> lp._ProgressCache:
    """Put a cache for *key* into the map without going through the resolver."""
    cache = lp._ProgressCache(key=key)
    lp._caches[key] = cache
    return cache


class TestActiveCacheResolution:
    def test_repeated_resolution_returns_the_same_object(self):
        """Entry points must all land on one cache, not rebuild it per call."""
        lp.clear_progress_cache()
        with lp._progress_lock:
            first = lp._active_cache()
            first.steps.append({"model": None, "threshold": None, "good_ids": [], "bad_ids": [], "stability": None})
            assert lp._active_cache() is first
            assert len(lp._active_cache().steps) == 1

    def test_a_foreign_pair_gets_its_own_cache(self):
        """Another pair's accumulated steps are invisible to the active one."""
        lp.clear_progress_cache()
        with lp._progress_lock:
            active = lp._active_cache()
            other = _seed((active.key[0], "det_foreign"))
            other.steps.extend([{"stability": None}] * 10)
            other.good_ids.update({1, 2, 3})

            resolved = lp._active_cache()
            assert resolved is active
            assert resolved.steps == []
            assert resolved.good_ids == set()

    def test_clear_drops_every_pair(self):
        """``clear_progress_cache`` is deliberately global, not active-pair-scoped."""
        with lp._progress_lock:
            lp._active_cache()
            _seed(("ds_other", "det_other"))
            assert len(lp._caches) >= 2

        lp.clear_progress_cache()

        assert lp._caches == {}


class TestCacheBound:
    def test_lru_evicts_the_least_recently_used_pair(self):
        lp.clear_progress_cache()
        with lp._progress_lock:
            # Fill to capacity with synthetic pairs, least-recently-used first.
            for i in range(lp._MAX_CACHED_PAIRS):
                _seed(("ds_bound", f"det{i}"))

            # Resolving the active pair pushes the count over the cap.
            active = lp._active_cache()

            assert len(lp._caches) == lp._MAX_CACHED_PAIRS
            assert ("ds_bound", "det0") not in lp._caches, "the LRU victim must be the one evicted"
            assert lp._caches[active.key] is active

    def test_touching_a_pair_saves_it_from_eviction(self):
        """The point of the bound is to keep an A-to-B-and-back toggle warm."""
        lp.clear_progress_cache()
        with lp._progress_lock:
            active = lp._active_cache()
            for i in range(lp._MAX_CACHED_PAIRS - 1):
                _seed(("ds_bound", f"det{i}"))

            # Re-resolving the active pair marks it most-recently-used, so the
            # next arrival evicts an older entry instead of it.
            lp._active_cache()
            _seed(("ds_bound", "det_newcomer"))
            lp._active_cache()

            assert lp._caches[active.key] is active
            assert len(lp._caches) == lp._MAX_CACHED_PAIRS
