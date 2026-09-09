"""Regression tests for the embedding-matrix builders.

Covers audit M11: a media item with ``embedding=None`` must NOT silently
poison the matrix with a NaN row (numpy 2.x behaviour of
``matrix[i] = None``).  The builder raises ``ValueError`` naming the
offending cid instead.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from vtscore.datasets import registry
from vtscore.embedding import matrix as matrix_mod
from vtscore.embedding.matrix import (
    get_embedding_matrix,
    get_embedding_matrix_for_snap,
    get_embedding_submatrix,
    invalidate_embedding_matrix,
    media_score_rows,
)
from vtscore.embedding.precomputed import MismatchedVectorError
from vtscore.state.core import DatasetContext, _state_lock, set_thread_dataset_context


class TestGetEmbeddingMatrixRaisesOnNoneEmbedding:
    def test_first_media_none_embedding(self):
        ctx = DatasetContext("test_first_none")
        ctx.medias[1] = {"id": 1, "embeddings": {}}
        ctx.medias[2] = {"id": 2, "embedder": "e5", "embeddings": {"e5": np.ones(4, dtype=np.float32)}}

        with pytest.raises(ValueError, match=r"media 1.*has no embedding"):
            get_embedding_matrix(ctx)

    def test_later_media_none_embedding(self):
        """Without the guard, numpy 2.x silently fills row i with NaN."""
        ctx = DatasetContext("test_later_none")
        ctx.medias[1] = {"id": 1, "embedder": "e5", "embeddings": {"e5": np.ones(4, dtype=np.float32)}}
        ctx.medias[2] = {"id": 2, "embedder": "e5", "embeddings": {"e5": np.full(4, 2.0, dtype=np.float32)}}
        ctx.medias[3] = {"id": 3, "embeddings": {}}

        with pytest.raises(ValueError, match=r"media 3.*has no embedding"):
            get_embedding_matrix(ctx)

    def test_empty_medias_ok(self):
        ctx = DatasetContext("test_empty")
        ids, mat = get_embedding_matrix(ctx)
        assert ids == []
        assert mat.shape == (0, 0)

    def test_clean_medias_ok(self):
        ctx = DatasetContext("test_clean")
        for cid in (1, 2, 3):
            ctx.medias[cid] = {
                "id": cid,
                "embedder": "e5",
                "embeddings": {"e5": np.full(4, float(cid), dtype=np.float32)},
            }
        ids, mat = get_embedding_matrix(ctx)
        assert ids == [1, 2, 3]
        assert mat.shape == (3, 4)
        # Row order matches sorted IDs.
        assert mat[0, 0] == 1.0
        assert mat[1, 0] == 2.0
        assert mat[2, 0] == 3.0


class TestGetEmbeddingSubmatrix:
    """Subset matrix builder for VTSBrowse subset projections."""

    def _ctx(self) -> DatasetContext:
        ctx = DatasetContext("test_submatrix")
        for cid in (1, 2, 3, 4):
            ctx.medias[cid] = {
                "id": cid,
                "embedder": "e5",
                "embeddings": {"e5": np.full(4, float(cid), dtype=np.float32)},
            }
        return ctx

    def test_returns_only_requested_ids_sorted(self):
        ctx = self._ctx()
        ids, mat = get_embedding_submatrix(ctx, [3, 1])
        assert ids == [1, 3]
        assert mat.shape == (2, 4)
        assert mat[0, 0] == 1.0
        assert mat[1, 0] == 3.0

    def test_dedups_and_drops_unknown_ids(self):
        ctx = self._ctx()
        ids, mat = get_embedding_submatrix(ctx, [2, 2, 99])
        assert ids == [2]
        assert mat.shape == (1, 4)

    def test_empty_when_no_match(self):
        ctx = self._ctx()
        ids, mat = get_embedding_submatrix(ctx, [99, 100])
        assert ids == []
        assert mat.shape == (0, 0)

    def test_raises_on_none_embedding(self):
        ctx = self._ctx()
        ctx.medias[2]["embeddings"] = {}
        with pytest.raises(ValueError, match=r"media 2.*has no embedding"):
            get_embedding_submatrix(ctx, [1, 2])

    def test_does_not_populate_cache(self):
        """Subset builds must not poison the context-wide matrix cache."""
        ctx = self._ctx()
        get_embedding_submatrix(ctx, [1, 2])
        assert ctx._emb_matrix is None
        assert ctx._emb_matrix_ids is None


class TestEmbedderAwareMatrix:
    """The matrix builders can source rows from a specific bound embedder.

    A multi-embedder dataset carries ``media["embeddings"]`` (dict keyed by
    embedder name); requesting an explicit name builds that embedder's matrix,
    while the default (no name) follows the primary mirror.
    """

    def _ctx(self) -> DatasetContext:
        ctx = DatasetContext("test_embedder_aware")
        for cid in (1, 2, 3):
            ctx.medias[cid] = {
                "id": cid,
                "embedder": "siglip",
                "embeddings": {
                    "siglip": np.full(4, float(cid), dtype=np.float32),
                    "dinov3_patch": np.full(4, float(cid) + 100.0, dtype=np.float32),
                },
            }
        return ctx

    def test_named_embedder_selects_that_matrix(self):
        ctx = self._ctx()
        ids, mat = get_embedding_matrix(ctx, "dinov3_patch")
        assert ids == [1, 2, 3]
        assert mat[0, 0] == 101.0
        assert mat[2, 0] == 103.0

    def test_default_follows_primary(self):
        ctx = self._ctx()
        ids, mat = get_embedding_matrix(ctx)
        assert mat[0, 0] == 1.0
        assert mat[2, 0] == 3.0

    def test_named_path_does_not_populate_cache(self):
        """The named path is uncached; the primary cache stays untouched."""
        ctx = self._ctx()
        get_embedding_matrix(ctx, "dinov3_patch")
        assert ctx._emb_matrix is None
        assert ctx._emb_matrix_ids is None

    def test_named_missing_vector_raises_with_embedder_name(self):
        ctx = self._ctx()
        del ctx.medias[2]["embeddings"]["dinov3_patch"]
        ctx.medias[2]["embedder"] = "siglip"  # primary is siglip's, not the requested embedder
        with pytest.raises(ValueError, match=r"media 2.*has no embedding for embedder 'dinov3_patch'"):
            get_embedding_matrix(ctx, "dinov3_patch")

    def test_primary_name_collapses_to_cache(self):
        """Routing hands a name even for single-embedder datasets; a name equal
        to the primary must collapse to the cached primary path, not the
        uncached named path - keeping the hot path byte-for-byte unchanged."""
        ctx = self._ctx()
        # "siglip" is the primary (the recorded ``embedder``).
        ids, mat = get_embedding_matrix(ctx, "siglip")
        assert ids == [1, 2, 3]
        # Primary's vectors (1,2,3), not dinov3's (101,...).
        assert mat[0, 0] == 1.0
        # ...and the cache was populated (the named path would not cache).
        assert ctx._emb_matrix is not None
        assert ctx._emb_matrix_ids == [1, 2, 3]

    def test_primary_name_snap_collapses_to_cache(self):
        ctx = self._ctx()
        set_thread_dataset_context(ctx)
        invalidate_embedding_matrix(ctx)
        snap = {cid: ctx.medias[cid] for cid in ctx.medias}
        ids, mat = get_embedding_matrix_for_snap(snap, "siglip")
        assert ids == [1, 2, 3]
        assert mat[0, 0] == 1.0
        assert ctx._emb_matrix is not None

    def test_submatrix_named_embedder(self):
        ctx = self._ctx()
        ids, mat = get_embedding_submatrix(ctx, [3, 1], "dinov3_patch")
        assert ids == [1, 3]
        assert mat[0, 0] == 101.0
        assert mat[1, 0] == 103.0

    def test_snap_named_embedder_matching_active_ctx(self):
        ctx = self._ctx()
        set_thread_dataset_context(ctx)
        invalidate_embedding_matrix(ctx)
        snap = {cid: ctx.medias[cid] for cid in ctx.medias}
        ids, mat = get_embedding_matrix_for_snap(snap, "dinov3_patch")
        assert ids == [1, 2, 3]
        assert mat[0, 0] == 101.0
        # Delegated to the context builder, but the named path must not cache.
        assert ctx._emb_matrix is None

    def test_snap_named_embedder_temp_dict(self):
        # Active ctx empty → snap can't match → fresh-build path.
        ctx = DatasetContext("test_snap_named_temp")
        set_thread_dataset_context(ctx)
        snap = {
            10: {"embedder": "siglip", "embeddings": {"dinov3_patch": np.full(4, 5.0, dtype=np.float32)}},
            11: {"embedder": "siglip", "embeddings": {"dinov3_patch": np.full(4, 6.0, dtype=np.float32)}},
        }
        ids, mat = get_embedding_matrix_for_snap(snap, "dinov3_patch")
        assert ids == [10, 11]
        assert mat[0, 0] == 5.0
        assert mat[1, 0] == 6.0


class TestGetEmbeddingMatrixForSnapRaisesOnNoneEmbedding:
    """Same guarantee for the snap helper, including the cross-dataset
    'temp dict' path that does NOT hit the cached active-ctx branch."""

    def test_snap_with_none_in_temp_dict(self):
        # Active ctx is empty so the snap can't match its key set; this
        # forces the fresh-build (uncached) path.
        ctx = DatasetContext("test_snap_temp")
        set_thread_dataset_context(ctx)

        snap = {
            10: {"embedder": "e5", "embeddings": {"e5": np.ones(4, dtype=np.float32)}},
            11: {"embeddings": {}},
        }
        with pytest.raises(ValueError, match=r"media 11.*has no embedding"):
            get_embedding_matrix_for_snap(snap)

    def test_snap_matching_active_ctx_with_none(self):
        """The matching-keys fast path delegates to ``get_embedding_matrix``
        on the active ctx; the same guard fires there too."""
        ctx = DatasetContext("test_snap_match")
        ctx.medias[1] = {"id": 1, "embedder": "e5", "embeddings": {"e5": np.ones(4, dtype=np.float32)}}
        ctx.medias[2] = {"id": 2, "embeddings": {}}
        set_thread_dataset_context(ctx)
        invalidate_embedding_matrix(ctx)

        snap = {cid: ctx.medias[cid] for cid in ctx.medias}
        with pytest.raises(ValueError, match=r"media 2.*has no embedding"):
            get_embedding_matrix_for_snap(snap)


class TestEmbeddingMatrixLockScoping:
    """The contiguous (N, D) build must run OUTSIDE ``_state_lock``.

    Holding the global state lock across ``_stack_embeddings`` lets a large
    (or multi-embedder named-path) build stall every other request's
    ``before_request`` state-sync, which also takes ``_state_lock``, on the
    single worker. These guard against re-introducing the in-lock build.
    """

    def _ctx(self) -> DatasetContext:
        ctx = DatasetContext("test_lock_scoping")
        for cid in (1, 2, 3, 4):
            ctx.medias[cid] = {
                "id": cid,
                "embedder": "e5",
                "embeddings": {"e5": np.full(4, float(cid), dtype=np.float32)},
            }
        return ctx

    def _assert_lock_free_during_build(self, monkeypatch, call) -> None:
        real_stack = matrix_mod._stack_embeddings
        lock_was_free = threading.Event()

        def probing_stack(sorted_ids, source, embedder_name):
            def grab():
                if _state_lock.acquire(timeout=2.0):
                    lock_was_free.set()
                    _state_lock.release()

            t = threading.Thread(target=grab)
            t.start()
            t.join(3.0)
            return real_stack(sorted_ids, source, embedder_name)

        monkeypatch.setattr(matrix_mod, "_stack_embeddings", probing_stack)
        call()
        assert lock_was_free.is_set(), "_state_lock was held across the matrix build"

    def test_get_embedding_matrix_builds_outside_state_lock(self, monkeypatch):
        ctx = self._ctx()  # fresh context -> cache miss -> reaches the build
        self._assert_lock_free_during_build(monkeypatch, lambda: get_embedding_matrix(ctx))

    def test_get_embedding_submatrix_builds_outside_state_lock(self, monkeypatch):
        ctx = self._ctx()
        self._assert_lock_free_during_build(monkeypatch, lambda: get_embedding_submatrix(ctx, [1, 2, 3, 4]))

    def test_stale_matrix_not_cached_when_medias_change_during_build(self, monkeypatch):
        """Phase-3 double-check: a media-set change during the unlocked build
        must NOT populate the primary cache with the now-stale matrix.
        """
        ctx = self._ctx()
        real_stack = matrix_mod._stack_embeddings

        def mutating_stack(sorted_ids, source, embedder_name):
            built = real_stack(sorted_ids, source, embedder_name)
            # A concurrent media insert lands while we build outside the lock.
            ctx.medias[99] = {"id": 99, "embedder": "e5", "embeddings": {"e5": np.full(4, 9.0, dtype=np.float32)}}
            return built

        monkeypatch.setattr(matrix_mod, "_stack_embeddings", mutating_stack)
        ids, mat = get_embedding_matrix(ctx)

        assert ids == [1, 2, 3, 4]
        assert mat.shape == (4, 4)
        # Cache must not hold the stale [1,2,3,4] build now that medias changed.
        assert ctx._emb_matrix_ids != [1, 2, 3, 4]


class TestMediaRevisionCounter:
    """Root-cause Pattern #4: the matrix cache keys on ``media_revision``.

    Structural mutations bump the counter transparently through
    :class:`MediasDict`; an in-place vector rewrite (same id set, different
    vectors) is invisible to a dict subclass and must be signalled by
    ``invalidate_embedding_matrix`` / ``bump_media_revision`` — this is the
    exact C4 miscompute the counter neutralises.
    """

    def _ctx(self) -> DatasetContext:
        ctx = DatasetContext("test_media_revision")
        for cid in (1, 2, 3):
            ctx.medias[cid] = {
                "id": cid,
                "embedder": "e5",
                "embeddings": {"e5": np.full(4, float(cid), dtype=np.float32)},
            }
        return ctx

    def test_structural_mutations_bump_revision(self):
        ctx = DatasetContext("test_bump_structural")
        assert ctx.media_revision == 0
        ctx.medias[1] = {"id": 1, "embedder": "e5", "embeddings": {"e5": np.ones(4, dtype=np.float32)}}
        after_add = ctx.media_revision
        assert after_add > 0
        del ctx.medias[1]
        assert ctx.media_revision > after_add

    def test_wholesale_reassignment_bumps_revision(self):
        ctx = DatasetContext("test_bump_reassign")
        before = ctx.media_revision
        ctx.medias = {1: {"id": 1, "embedder": "e5", "embeddings": {"e5": np.ones(4, dtype=np.float32)}}}
        assert ctx.media_revision > before
        # The assigned mapping is wrapped so it keeps bumping on later edits.
        after_assign = ctx.media_revision
        ctx.medias[2] = {"id": 2, "embedder": "e5", "embeddings": {"e5": np.full(4, 2.0, dtype=np.float32)}}
        assert ctx.media_revision > after_assign

    def test_no_bump_without_mutation(self):
        ctx = self._ctx()
        rev = ctx.media_revision
        _ = ctx.medias[1]  # a read must not bump
        _ = list(ctx.medias.keys())
        assert ctx.media_revision == rev

    def test_cache_reused_across_calls_at_same_revision(self):
        ctx = self._ctx()
        ids1, mat1 = get_embedding_matrix(ctx)
        ids2, mat2 = get_embedding_matrix(ctx)
        assert ids1 == ids2 == [1, 2, 3]
        # Same underlying array object → served from cache, not rebuilt.
        assert mat1 is mat2
        assert ctx._emb_matrix_revision == ctx.media_revision

    def test_inplace_vector_rewrite_needs_explicit_bump(self):
        """A dict-subclass can't see ``medias[cid][...] = vec``; until the
        counter is bumped the cache keeps serving the pre-rewrite matrix."""
        ctx = self._ctx()
        _, mat_before = get_embedding_matrix(ctx)
        assert mat_before[0, 0] == 1.0

        # Rewrite media 1's vector in place — no structural change, no bump.
        ctx.medias[1]["embeddings"]["e5"] = np.full(4, 42.0, dtype=np.float32)
        _, mat_stale = get_embedding_matrix(ctx)
        assert mat_stale[0, 0] == 1.0  # still the cached pre-rewrite row

        # The embed/clip stages signal the in-place change via this call.
        invalidate_embedding_matrix(ctx)
        _, mat_fresh = get_embedding_matrix(ctx)
        assert mat_fresh[0, 0] == 42.0

    def test_same_id_set_new_content_not_served_stale(self):
        """Reassigning ``medias`` to a fresh dict with the *same ids* but new
        vectors must invalidate the cache (the id-list key could not tell)."""
        ctx = self._ctx()
        _, mat_before = get_embedding_matrix(ctx)
        assert mat_before[0, 0] == 1.0

        ctx.medias = {
            cid: {"id": cid, "embedder": "e5", "embeddings": {"e5": np.full(4, float(cid) + 100.0, dtype=np.float32)}}
            for cid in (1, 2, 3)
        }
        ids, mat_after = get_embedding_matrix(ctx)
        assert ids == [1, 2, 3]
        assert mat_after[0, 0] == 101.0


class TestEmbeddingMatrixSidecar:
    """S1 (docs/plans/scalability.md): the mmap embedding-matrix sidecar.

    A dataset backed by a registry entry gets its primary matrix persisted
    as a ``<pkl_stem>.embids.npy`` / ``<pkl_stem>.embmat.npy`` pair after the
    first build, so a fresh ``DatasetContext`` for the same dataset can mmap
    it instead of rebuilding from per-item embeddings.
    """

    def _register(self, tmp_path, num_items: int = 3) -> str:
        pkl_dir = tmp_path / "saved"
        pkl_dir.mkdir(parents=True, exist_ok=True)
        entry = registry.register_dataset(
            name="sidecar-test",
            media_type="audio",
            num_items=num_items,
            pkl_path=str(pkl_dir / "ds_sidecar.pkl"),
        )
        return entry["id"]

    def _medias(self, n: int = 3) -> dict:
        return {
            cid: {"id": cid, "embedder": "e5", "embeddings": {"e5": np.full(4, float(cid), dtype=np.float32)}}
            for cid in range(1, n + 1)
        }

    def _pkl_path(self, dataset_id: str) -> str:
        entry = registry.get_dataset(dataset_id)
        assert entry is not None
        return entry["pkl_path"]

    def test_sidecar_written_after_first_build(self, tmp_path):
        dataset_id = self._register(tmp_path)
        ctx = DatasetContext(dataset_id)
        ctx.medias = self._medias()

        get_embedding_matrix(ctx)

        ids_path, mat_path = matrix_mod._emb_sidecar_paths(self._pkl_path(dataset_id))
        assert ids_path.is_file()
        assert mat_path.is_file()
        assert np.array_equal(np.load(ids_path), np.array([1, 2, 3], dtype=np.int64))

    def test_fresh_context_mmaps_the_sidecar(self, tmp_path):
        dataset_id = self._register(tmp_path)
        ctx1 = DatasetContext(dataset_id)
        ctx1.medias = self._medias()
        get_embedding_matrix(ctx1)  # writes the sidecar

        # A second context for the same registered dataset stands in for a
        # fresh process re-loading the same pkl from disk.
        ctx2 = DatasetContext(dataset_id)
        ctx2.medias = self._medias()
        ids, mat = get_embedding_matrix(ctx2)

        assert ids == [1, 2, 3]
        assert mat[0, 0] == 1.0
        assert mat[2, 0] == 3.0
        assert isinstance(ctx2._emb_matrix, np.memmap), "expected the mmap sidecar path, not a fresh rebuild"

    def test_unregistered_dataset_never_writes_a_sidecar(self, tmp_path):
        """No registry entry -> no pkl_path -> the mmap cache stays opt-in only."""
        ctx = DatasetContext("not_registered")
        ctx.medias = self._medias()

        get_embedding_matrix(ctx)

        assert not any(tmp_path.iterdir())

    def test_id_mismatch_sidecar_falls_back_to_live_rebuild(self, tmp_path):
        """A sidecar written for a different id set must never be trusted."""
        dataset_id = self._register(tmp_path)
        ids_path, mat_path = matrix_mod._emb_sidecar_paths(self._pkl_path(dataset_id))
        matrix_mod._atomic_save_npy(ids_path, np.array([1, 2, 99], dtype=np.int64))
        matrix_mod._atomic_save_npy(mat_path, np.zeros((3, 4), dtype=np.float32))

        ctx = DatasetContext(dataset_id)
        ctx.medias = self._medias()
        ids, mat = get_embedding_matrix(ctx)

        assert ids == [1, 2, 3]
        assert mat[0, 0] == 1.0  # live value, not the bogus zero-filled sidecar

    def test_dim_mismatch_sidecar_falls_back_to_live_rebuild(self, tmp_path):
        """A same-id-count sidecar with the wrong dimension must never be trusted."""
        dataset_id = self._register(tmp_path)
        ids_path, mat_path = matrix_mod._emb_sidecar_paths(self._pkl_path(dataset_id))
        matrix_mod._atomic_save_npy(ids_path, np.array([1, 2, 3], dtype=np.int64))
        matrix_mod._atomic_save_npy(mat_path, np.zeros((3, 99), dtype=np.float32))

        ctx = DatasetContext(dataset_id)
        ctx.medias = self._medias()
        ids, mat = get_embedding_matrix(ctx)

        assert ids == [1, 2, 3]
        assert mat.shape == (3, 4)
        assert mat[0, 0] == 1.0

    def test_invalidate_disables_sidecar_for_rest_of_context_lifetime(self, tmp_path):
        """Root-cause Pattern #4 for the sidecar: a same-id in-place vector
        rewrite must never be served stale from the on-disk mmap cache."""
        dataset_id = self._register(tmp_path)
        ctx1 = DatasetContext(dataset_id)
        ctx1.medias = self._medias()
        get_embedding_matrix(ctx1)  # writes the sidecar with row 0 == 1.0

        ctx2 = DatasetContext(dataset_id)
        ctx2.medias = self._medias()
        ids, mat = get_embedding_matrix(ctx2)
        assert mat[0, 0] == 1.0
        assert isinstance(ctx2._emb_matrix, np.memmap)

        # An in-place rewrite (re-embed/clip) with the same id set - the
        # sidecar's id-list check alone cannot see this.
        ctx2.medias[1]["embeddings"]["e5"] = np.full(4, 42.0, dtype=np.float32)
        invalidate_embedding_matrix(ctx2)
        assert ctx2._emb_sidecar_disabled is True

        ids, mat = get_embedding_matrix(ctx2)
        assert mat[0, 0] == 42.0, "must reflect the in-place rewrite, not the stale mmap'd sidecar"
        assert not isinstance(ctx2._emb_matrix, np.memmap)

        # The on-disk sidecar itself must also be refreshed (not just this
        # context's in-memory view), or a third, later-loading context would
        # still mmap the stale pre-rewrite values.
        _, mat_path = matrix_mod._emb_sidecar_paths(self._pkl_path(dataset_id))
        assert np.load(mat_path)[0, 0] == 42.0


class TestMixedDimensionsRaiseALocatableError:
    """A dataset holding vectors of two widths must fail with a message that names them.

    Ingestion validation (:mod:`vtscore.embedding.precomputed`) is supposed to
    make this state unreachable, but a dataset can still arrive at it by another
    route - a pickle written before that validation existed, or a third-party
    importer writing ``media["embeddings"]`` directly.  When it does, the raw
    numpy failure is ``could not broadcast input array from shape (768,) into
    shape (1152,)``: no cid, no filename, no embedder, on whatever request
    happened to rebuild the matrix.  These tests pin the replacement.
    """

    @staticmethod
    def _mixed_ctx(name: str) -> DatasetContext:
        ctx = DatasetContext(name)
        ctx.medias[1] = {
            "id": 1,
            "embedder": "e5",
            "filename": "first.txt",
            "embeddings": {"e5": np.ones(4, dtype=np.float32)},
        }
        ctx.medias[2] = {
            "id": 2,
            "embedder": "e5",
            "filename": "odd-one-out.txt",
            "embeddings": {"e5": np.ones(7, dtype=np.float32)},
        }
        return ctx

    def test_matrix_build_names_the_offending_media_and_both_widths(self):
        ctx = self._mixed_ctx("test_mixed_dims_matrix")
        with pytest.raises(MismatchedVectorError) as exc:
            get_embedding_matrix(ctx)
        msg = str(exc.value)
        assert "media 2" in msg
        assert "odd-one-out.txt" in msg
        assert "7" in msg and "4" in msg

    def test_submatrix_build_raises_too(self):
        ctx = self._mixed_ctx("test_mixed_dims_submatrix")
        with pytest.raises(MismatchedVectorError, match="odd-one-out.txt"):
            get_embedding_submatrix(ctx, [1, 2])

    def test_prefers_origin_name_over_filename(self):
        ctx = self._mixed_ctx("test_mixed_dims_origin_name")
        ctx.medias[2]["origin_name"] = "shards.tar::odd.jpg"
        with pytest.raises(MismatchedVectorError, match=r"shards\.tar::odd\.jpg"):
            get_embedding_matrix(ctx)

    def test_uniform_widths_still_build_normally(self):
        ctx = DatasetContext("test_uniform_dims")
        ctx.medias[1] = {"id": 1, "embedder": "e5", "embeddings": {"e5": np.ones(4, dtype=np.float32)}}
        ctx.medias[2] = {"id": 2, "embedder": "e5", "embeddings": {"e5": np.zeros(4, dtype=np.float32)}}
        ids, mat = get_embedding_matrix(ctx)
        assert ids == [1, 2]
        assert mat.shape == (2, 4)

    def test_float16_stored_vector_is_still_accepted(self):
        """Width is what must match; a half-precision row is widened, not refused.

        This is the shape issue #3143's fp16 work would produce, and it must
        remain a non-event: the matrix is float32 and numpy widens on assignment.
        """
        ctx = DatasetContext("test_fp16_rows")
        ctx.medias[1] = {"id": 1, "embedder": "e5", "embeddings": {"e5": np.ones(4, dtype=np.float16)}}
        ctx.medias[2] = {"id": 2, "embedder": "e5", "embeddings": {"e5": np.zeros(4, dtype=np.float32)}}
        _ids, mat = get_embedding_matrix(ctx)
        assert mat.dtype == np.float32
        assert mat[0, 0] == 1.0


class TestPatchGridWidthMismatch:
    def test_cls_vector_and_patch_grid_must_share_a_space(self):
        """``media_score_rows`` max-pools the two together, so they must agree."""
        media = {
            "id": 5,
            "filename": "mixed.jpg",
            "embedder": "dinov3_patch",
            "embeddings": {"dinov3_patch": np.ones(4, dtype=np.float32)},
            "patch_grid": np.ones((2, 2, 9), dtype=np.float16),
        }
        with pytest.raises(MismatchedVectorError) as exc:
            media_score_rows(media, "dinov3_patch")
        msg = str(exc.value)
        assert "mixed.jpg" in msg
        assert "4" in msg and "9" in msg

    def test_matching_widths_stack_cls_row_then_patches(self):
        media = {
            "id": 5,
            "embedder": "dinov3_patch",
            "embeddings": {"dinov3_patch": np.ones(3, dtype=np.float32)},
            "patch_grid": np.zeros((2, 2, 3), dtype=np.float16),
        }
        rows = media_score_rows(media, "dinov3_patch")
        assert rows is not None
        assert rows.shape == (5, 3)  # 1 CLS row + 2x2 patches
        assert rows[0, 0] == 1.0


class TestScoreableSnapshot:
    """``scoreable_snapshot`` is the pre-filter that turns a fatal matrix build
    into a skipped item (issue #3179).  It answers exactly the two questions
    the builder raises on: is there a vector at all, and is it a 1-D row of the
    same width as the rest?
    """

    @staticmethod
    def _media(cid: int, vec, name: str = "e5") -> dict:
        return {
            "id": cid,
            "embedder": name,
            "embeddings": {} if vec is None else {name: vec},
            "filename": f"m{cid}.wav",
        }

    def test_media_without_a_vector_is_dropped(self):
        from vtscore.embedding.matrix import scoreable_snapshot

        snap = {
            1: self._media(1, np.ones(4, dtype=np.float32)),
            2: self._media(2, None),
            3: self._media(3, np.zeros(4, dtype=np.float32)),
        }
        scoreable, dropped = scoreable_snapshot(snap)
        assert dropped == [2]
        assert sorted(scoreable) == [1, 3]
        # And what survives builds a matrix without raising.
        ids, mat = get_embedding_matrix_for_snap(scoreable)
        assert ids == [1, 3]
        assert mat.shape == (2, 4)

    def test_wrong_width_vector_is_dropped(self):
        """The ``require_dim`` / ``MismatchedVectorError`` half of #3179."""
        from vtscore.embedding.matrix import scoreable_snapshot

        snap = {
            1: self._media(1, np.ones(4, dtype=np.float32)),
            2: self._media(2, np.ones(8, dtype=np.float32)),
            3: self._media(3, np.zeros(4, dtype=np.float32)),
        }
        # Unfiltered, this is the crash the issue reports.
        with pytest.raises(MismatchedVectorError):
            get_embedding_matrix_for_snap(snap)

        scoreable, dropped = scoreable_snapshot(snap)
        assert dropped == [2]
        ids, mat = get_embedding_matrix_for_snap(scoreable)
        assert ids == [1, 3]
        assert mat.shape == (2, 4)

    def test_non_row_vector_is_dropped(self):
        from vtscore.embedding.matrix import scoreable_snapshot

        snap = {
            1: self._media(1, np.ones(4, dtype=np.float32)),
            2: self._media(2, np.ones((2, 4), dtype=np.float32)),
        }
        scoreable, dropped = scoreable_snapshot(snap)
        assert dropped == [2]
        assert sorted(scoreable) == [1]

    def test_named_embedder_keys_the_check(self):
        """A media embedded by one bound embedder but not the other is dropped
        only from the space it is missing - the multi-embedder shape of the
        same failure."""
        from vtscore.embedding.matrix import scoreable_snapshot

        snap = {
            1: {
                "id": 1,
                "embedder": "siglip",
                "embeddings": {"siglip": np.ones(4, dtype=np.float32), "e5": np.ones(4, dtype=np.float32)},
            },
            2: {"id": 2, "embedder": "siglip", "embeddings": {"siglip": np.ones(4, dtype=np.float32)}},
        }
        kept_siglip, dropped_siglip = scoreable_snapshot(snap, "siglip")
        assert dropped_siglip == [] and sorted(kept_siglip) == [1, 2]

        kept_e5, dropped_e5 = scoreable_snapshot(snap, "e5")
        assert dropped_e5 == [2] and sorted(kept_e5) == [1]

    def test_everything_unusable_yields_an_empty_snapshot(self):
        from vtscore.embedding.matrix import scoreable_snapshot

        snap = {1: self._media(1, None), 2: self._media(2, None)}
        scoreable, dropped = scoreable_snapshot(snap)
        assert scoreable == {}
        assert dropped == [1, 2]

    def test_empty_snapshot(self):
        from vtscore.embedding.matrix import scoreable_snapshot

        assert scoreable_snapshot({}) == ({}, [])


class TestMixedSpaceSnapshotDoesNotCollapse:
    """Issue #3650: the primary-collapse quantifier is "every", not "the first".

    ``_collapse_to_primary`` used to decide whether a routed embedder name *is*
    the primary by looking at one media.  On a mixed-type snapshot that is a
    sampling error: media #1's primary picked the path for all N, so asking for
    space ``A`` on a snapshot led by an ``A`` media stacked the *other* media's
    ``B``-space vectors into an ``A``-space matrix - silently, whenever the two
    spaces share a width.  Reordering the same dict flipped the answer.
    """

    def _mixed(self) -> dict[int, dict]:
        """Two ``test``-space media and two ``vid``-space ones, same width."""
        return {
            1: {"id": 1, "embedder": "test", "embeddings": {"test": np.full(4, 1.0, dtype=np.float32)}},
            2: {"id": 2, "embedder": "test", "embeddings": {"test": np.full(4, 2.0, dtype=np.float32)}},
            3: {"id": 3, "embedder": "vid", "embeddings": {"vid": np.full(4, 30.0, dtype=np.float32)}},
            4: {"id": 4, "embedder": "vid", "embeddings": {"vid": np.full(4, 40.0, dtype=np.float32)}},
        }

    def test_collapse_requires_every_media_to_agree(self):
        from vtscore.embedding.matrix import _collapse_to_primary

        mixed = self._mixed()
        # Mixed: the name survives, so the named path (which reads only "test"
        # vectors) is taken instead of the primary path (which reads whatever
        # each media happens to carry).
        assert _collapse_to_primary(mixed, "test") == "test"
        assert _collapse_to_primary(mixed, "vid") == "vid"
        # Homogeneous: unchanged - this is the single-embedder hot path.
        homogeneous = {cid: m for cid, m in mixed.items() if m["embedder"] == "test"}
        assert _collapse_to_primary(homogeneous, "test") is None

    def test_collapse_is_independent_of_dict_order(self):
        from vtscore.embedding.matrix import _collapse_to_primary

        mixed = self._mixed()
        reversed_order = {cid: mixed[cid] for cid in sorted(mixed, reverse=True)}
        assert _collapse_to_primary(mixed, "test") == _collapse_to_primary(reversed_order, "test")

    def test_scoreable_snapshot_drops_the_foreign_space(self):
        """The `calibration haystack size: 60 of 60` of #3650, now 30 of 60."""
        from vtscore.embedding.matrix import scoreable_snapshot

        scoreable, dropped = scoreable_snapshot(self._mixed(), "test")
        assert sorted(scoreable) == [1, 2]
        assert dropped == [3, 4]

    def test_scoreable_snapshot_answer_is_order_independent(self):
        from vtscore.embedding.matrix import scoreable_snapshot

        mixed = self._mixed()
        reversed_order = {cid: mixed[cid] for cid in sorted(mixed, reverse=True)}
        assert scoreable_snapshot(mixed, "test") == scoreable_snapshot(reversed_order, "test")

    def test_snap_matrix_never_stacks_the_other_space(self):
        """Pre-fix this built a 4-row matrix holding two ``vid`` rows."""
        ctx = DatasetContext("test_mixed_space_snap")
        set_thread_dataset_context(ctx)
        mixed = self._mixed()

        with pytest.raises(ValueError, match=r"media 3.*has no embedding for embedder 'test'"):
            get_embedding_matrix_for_snap(mixed, "test")

        # The pre-filtered snapshot builds cleanly, in the requested space only.
        from vtscore.embedding.matrix import scoreable_snapshot

        scoreable, _dropped = scoreable_snapshot(mixed, "test")
        ids, mat = get_embedding_matrix_for_snap(scoreable, "test")
        assert ids == [1, 2]
        assert mat[:, 0].tolist() == [1.0, 2.0]

    def test_ctx_matrix_never_stacks_the_other_space(self):
        ctx = DatasetContext("test_mixed_space_ctx")
        ctx.medias.update(self._mixed())
        with pytest.raises(ValueError, match=r"media 3.*has no embedding for embedder 'test'"):
            get_embedding_matrix(ctx, "test")
        # ...and the named path left the primary cache alone.
        assert ctx._emb_matrix is None

    def test_unnamed_request_still_reads_each_medias_primary(self):
        """Only the *named* request changes: an unnamed one is by definition a
        request for whatever each media carries, and keeps doing that."""
        ctx = DatasetContext("test_mixed_space_unnamed")
        ctx.medias.update(self._mixed())
        ids, mat = get_embedding_matrix(ctx)
        assert ids == [1, 2, 3, 4]
        assert mat[:, 0].tolist() == [1.0, 2.0, 30.0, 40.0]


class TestUniformPrimaryMemo:
    """``_collapse_to_primary_for_ctx`` memoises the O(N) scan per revision.

    The scan runs under ``_state_lock`` *before* the matrix-cache check, so the
    hot path must not pay it per call - but the memo must not outlive the
    medias it answered for.
    """

    def _ctx(self) -> DatasetContext:
        ctx = DatasetContext("test_uniform_primary_memo")
        for cid in (1, 2):
            ctx.medias[cid] = {
                "id": cid,
                "embedder": "siglip",
                "embeddings": {"siglip": np.full(4, float(cid), dtype=np.float32)},
            }
        return ctx

    def test_memo_is_computed_once_per_revision(self, monkeypatch):
        from vtscore.embedding.matrix import _collapse_to_primary_for_ctx

        ctx = self._ctx()
        calls: list[int] = []
        real = matrix_mod._uniform_primary_embedder
        monkeypatch.setattr(
            matrix_mod,
            "_uniform_primary_embedder",
            lambda medias: (calls.append(len(medias)), real(medias))[1],
        )
        assert _collapse_to_primary_for_ctx(ctx, "siglip") is None
        assert _collapse_to_primary_for_ctx(ctx, "siglip") is None
        assert len(calls) == 1

    def test_a_structural_mutation_reopens_the_question(self):
        from vtscore.embedding.matrix import _collapse_to_primary_for_ctx

        ctx = self._ctx()
        assert _collapse_to_primary_for_ctx(ctx, "siglip") is None
        # Adding a media from another space bumps media_revision via MediasDict.
        ctx.medias[3] = {"id": 3, "embedder": "vid", "embeddings": {"vid": np.zeros(4, dtype=np.float32)}}
        assert _collapse_to_primary_for_ctx(ctx, "siglip") == "siglip"

    def test_matrix_cache_is_not_served_across_a_mixing_mutation(self):
        """The end-to-end shape of the memo going stale: the cached primary
        matrix must not be handed back for a *named* request once the dataset
        stops being homogeneous."""
        ctx = self._ctx()
        _ids, mat = get_embedding_matrix(ctx, "siglip")
        assert mat[:, 0].tolist() == [1.0, 2.0]
        assert ctx._emb_matrix is not None  # collapsed to the cached path

        ctx.medias[3] = {"id": 3, "embedder": "vid", "embeddings": {"vid": np.zeros(4, dtype=np.float32)}}
        with pytest.raises(ValueError, match=r"media 3.*has no embedding for embedder 'siglip'"):
            get_embedding_matrix(ctx, "siglip")

    def test_reset_derived_caches_drops_the_memo(self):
        from vtscore.embedding.matrix import _collapse_to_primary_for_ctx

        ctx = self._ctx()
        _collapse_to_primary_for_ctx(ctx, "siglip")
        assert ctx._uniform_primary == "siglip"
        ctx.reset_derived_caches()
        assert ctx._uniform_primary is None
        assert ctx._uniform_primary_revision is None


class TestMixedSpaceDropIsLogged:
    """A short haystack caused by space-mixing says so, once per call."""

    def _snap(self) -> dict[int, dict]:
        return {
            1: {"id": 1, "embedder": "test", "embeddings": {"test": np.ones(4, dtype=np.float32)}},
            2: {"id": 2, "embedder": "vid", "embeddings": {"vid": np.ones(4, dtype=np.float32)}},
            3: {"id": 3, "embedder": "vid", "embeddings": {"vid": np.ones(4, dtype=np.float32)}},
        }

    def test_warns_once_naming_the_other_space(self, caplog):
        from vtscore.embedding.matrix import scoreable_snapshot

        with caplog.at_level("WARNING", logger="vtscore.embedding.matrix"):
            scoreable_snapshot(self._snap(), "test")
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        assert "Dropped 2 of 3 media" in message
        assert "'vid'" in message and "'test'" in message

    def test_a_plain_failed_embed_is_not_reported_as_mixing(self, caplog):
        """A dropped media whose own primary *is* the requested embedder simply
        failed to embed - the pre-existing per-item failure the callers already
        report, not a space mismatch."""
        from vtscore.embedding.matrix import scoreable_snapshot

        snap = {
            1: {"id": 1, "embedder": "test", "embeddings": {"test": np.ones(4, dtype=np.float32)}},
            2: {"id": 2, "embedder": "test", "embeddings": {}},
        }
        with caplog.at_level("WARNING", logger="vtscore.embedding.matrix"):
            _scoreable, dropped = scoreable_snapshot(snap, "test")
        assert dropped == [2]
        assert [r for r in caplog.records if r.levelname == "WARNING"] == []

    def test_a_homogeneous_snapshot_is_silent(self, caplog):
        from vtscore.embedding.matrix import scoreable_snapshot

        snap = {cid: m for cid, m in self._snap().items() if m["embedder"] == "vid"}
        with caplog.at_level("WARNING", logger="vtscore.embedding.matrix"):
            scoreable_snapshot(snap, "vid")
        assert [r for r in caplog.records if r.levelname == "WARNING"] == []
