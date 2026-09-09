"""Core state types and lock shared by all state submodules.

Defines the per-dataset and per-detector context classes
(:class:`DatasetContext`, :class:`DetectorContext`), the resolution
functions that find the "active" context for the current request /
thread, and the reentrant lock that protects all mutable state.

Multi-dataset support
---------------------
Per-dataset state is bundled in :class:`DatasetContext` objects.  Per-
detector state lives in :class:`DetectorContext`.  Context stores map
each ID to its context, and per-request / thread-local resolvers determine
which one library helpers operate on.

The app-side facade
-------------------
Module-level convenience names (``medias``, ``good_votes``, …) used to
live here as proxy objects, but they belong to the app layer; the
library never imports them.  They now live in
:mod:`vtsearch.state_proxies` and are re-exported from
:mod:`vtsearch.state` so existing app-tier imports continue to work.
See Phase 3 of ``../docs/architecture.md``.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any


# Reentrant lock protecting all mutable state.
# RLock is used because some public functions call other public functions
# (e.g. clear_all -> clear_medias + clear_votes).
_state_lock = threading.RLock()


class DatasetNotLoadedError(LookupError):
    """The request explicitly named a dataset that is not loaded in memory.

    Raised by the request-scoped dataset resolver (and propagated through
    :func:`get_active_context`) when an ``X-Dataset-Id`` header (or
    ``?dataset_id=`` query param) was sent but no matching
    :class:`DatasetContext` is registered. Silent fallback to an empty
    context produced stale results that the client could not detect;
    see logical-bug-audit H16.
    """

    def __init__(self, dataset_id: str) -> None:
        super().__init__(f"dataset {dataset_id!r} is not loaded")
        self.dataset_id = dataset_id


class DetectorNotLoadedError(LookupError):
    """The request explicitly named a detector that is not loaded in memory.

    Detector counterpart of :class:`DatasetNotLoadedError`. See
    logical-bug-audit H16 / H34.
    """

    def __init__(self, detector_id: str) -> None:
        super().__init__(f"detector {detector_id!r} is not loaded")
        self.detector_id = detector_id


# ---------------------------------------------------------------------------
# Request-missing sentinel: frozen empty context returned when a Flask
# request didn't identify a dataset/detector (missing header or unloaded id).
# Reads see an empty context (so non-mutating endpoints continue working);
# any mutation raises ``RequestMissingContextError`` immediately so the
# silent-mistarget failure modes flagged by H13/H16 fail loudly instead.
# ---------------------------------------------------------------------------


class RequestMissingContextError(RuntimeError):
    """Raised when code tries to mutate the request-missing context sentinel.

    The sentinel is what :func:`get_active_context` /
    :func:`get_active_detector_context` return inside a Flask request when
    the client didn't identify a dataset/detector; either the
    ``X-Dataset-Id`` / ``X-Detector-Id`` header was missing, or it named an
    unloaded id.  Reads against the sentinel see an empty context (so
    listing/dashboard endpoints keep working); writes hit this exception so
    votes / labels / pile additions cannot silently land on the wrong
    target.
    """


def _frozen_mutation_error(kind: str) -> RequestMissingContextError:
    return RequestMissingContextError(
        f"Refusing to mutate the request-missing {kind} context. "
        f"This Flask request did not identify a {kind} (missing "
        f"X-{kind.capitalize()}-Id header / query param, or it named an "
        f"unloaded id). Mutation endpoints must identify the {kind} "
        f"explicitly."
    )


class _FrozenDict(dict):  # type: ignore[type-arg]
    """A ``dict`` that allows reads but raises on every mutation."""

    __slots__ = ("_kind",)

    def __init__(self, kind: str) -> None:
        super().__init__()
        # Bypass our own __setattr__ (dict subclasses don't get one by default,
        # but be explicit so adding one later doesn't break this).
        object.__setattr__(self, "_kind", kind)

    def __setitem__(self, key: Any, value: Any) -> None:
        raise _frozen_mutation_error(self._kind)

    def __delitem__(self, key: Any) -> None:
        raise _frozen_mutation_error(self._kind)

    def update(self, *a: Any, **k: Any) -> None:
        raise _frozen_mutation_error(self._kind)

    def setdefault(self, *a: Any, **k: Any) -> Any:
        raise _frozen_mutation_error(self._kind)

    def pop(self, *a: Any, **k: Any) -> Any:
        raise _frozen_mutation_error(self._kind)

    def popitem(self) -> Any:
        raise _frozen_mutation_error(self._kind)

    def clear(self) -> None:
        raise _frozen_mutation_error(self._kind)


class _FrozenList(list):  # type: ignore[type-arg]
    """A ``list`` that allows reads but raises on every mutation."""

    __slots__ = ("_kind",)

    def __init__(self, kind: str) -> None:
        super().__init__()
        object.__setattr__(self, "_kind", kind)

    def append(self, *a: Any, **k: Any) -> None:
        raise _frozen_mutation_error(self._kind)

    def extend(self, *a: Any, **k: Any) -> None:
        raise _frozen_mutation_error(self._kind)

    def insert(self, *a: Any, **k: Any) -> None:
        raise _frozen_mutation_error(self._kind)

    def remove(self, *a: Any, **k: Any) -> None:
        raise _frozen_mutation_error(self._kind)

    def pop(self, *a: Any, **k: Any) -> Any:
        raise _frozen_mutation_error(self._kind)

    def clear(self) -> None:
        raise _frozen_mutation_error(self._kind)

    def __setitem__(self, *a: Any, **k: Any) -> None:
        raise _frozen_mutation_error(self._kind)

    def __delitem__(self, *a: Any, **k: Any) -> None:
        raise _frozen_mutation_error(self._kind)

    def __iadd__(self, *a: Any, **k: Any) -> Any:
        raise _frozen_mutation_error(self._kind)

    def __imul__(self, *a: Any, **k: Any) -> Any:
        raise _frozen_mutation_error(self._kind)

    def sort(self, *a: Any, **k: Any) -> None:
        raise _frozen_mutation_error(self._kind)

    def reverse(self) -> None:
        raise _frozen_mutation_error(self._kind)


def _iter_slots(cls: type) -> Iterator[str]:
    """Yield every ``__slots__`` name declared anywhere in *cls*'s MRO, once."""
    seen: set[str] = set()
    for klass in cls.__mro__:
        for name in getattr(klass, "__slots__", ()):
            if name not in seen:
                seen.add(name)
                yield name


def _freeze_into(sentinel: Any, template: Any, kind: str) -> None:
    """Copy every slot of *template* onto *sentinel*, freezing its containers.

    The request-missing sentinels are documented to read as *empty*
    contexts, so they must carry the full slot set of the class they
    subclass - a subclass with ``__slots__ = ()`` and no ``__getattr__``
    turns any missed slot into an ``AttributeError`` (a 500 on the exact
    dropped-header path the sentinel exists to serve).  Hand-maintaining
    that list drifted twice; instead we build a fresh real context and copy
    it across, so a newly added slot can never be missed (issue #2933).

    Every ``dict`` / ``list`` value is replaced by a :class:`_FrozenDict` /
    :class:`_FrozenList` so container mutations raise
    :class:`RequestMissingContextError` instead of silently landing on the
    sentinel.  Scalar / ``None`` slots are copied verbatim; the whole
    sentinel refuses attribute assignment anyway (``__setattr__``), so a
    write to one of those raises too.
    """
    for name in _iter_slots(type(template)):
        try:
            value = getattr(template, name)
        except AttributeError:
            # A slot the class declares but never initialises: leave it
            # unset here too, so the sentinel reads exactly like a real
            # freshly-constructed context.
            continue
        if isinstance(value, dict):
            value = _FrozenDict(kind)
        elif isinstance(value, list):
            value = _FrozenList(kind)
        object.__setattr__(sentinel, name, value)


# ---------------------------------------------------------------------------
# Flask-request predicate hook
# ---------------------------------------------------------------------------
# Returns True when execution is inside a Flask request that should be
# refused if no explicit dataset/detector was identified.  The vtscore
# library stays Flask-free; the Flask shim registers
# ``flask.has_request_context`` here at app startup.  Outside Flask
# (CLI, library callers, background threads), the default ``lambda: False``
# stays in place so :func:`get_active_context` keeps falling back to the
# empty context as before.
def _default_request_context_predicate() -> bool:
    return False


_request_context_predicate: Callable[[], bool] = _default_request_context_predicate


def register_request_context_predicate(fn: Callable[[], bool]) -> None:
    """Install the predicate used to decide whether to return the
    request-missing sentinel instead of the empty fallback context.

    The Flask shim wires this to :func:`flask.has_request_context` at
    startup so that a Flask request without an identified dataset/detector
    sees the frozen sentinel (which fails loudly on mutation) rather than
    silently landing on the empty global fallback.
    """
    global _request_context_predicate
    _request_context_predicate = fn


# ---------------------------------------------------------------------------
# Pluggable per-request context resolvers
# ---------------------------------------------------------------------------
# In the Flask app, the ``before_request`` hook resolves an
# ``X-Dataset-Id`` / ``X-Detector-Id`` header to the matching context and
# stashes it on ``g``.  The proxy objects then need to read it back.
#
# To keep this module Flask-free (so it can move into ``vtscore`` later -
# see ``../docs/architecture.md``), the read side is exposed as a
# **pluggable resolver**: a callable that returns the current request's
# DatasetContext / DetectorContext, or ``None`` if there is no request.
#
# The Flask integration lives in ``vtsearch/shim/`` and registers Flask-
# aware resolvers at app startup.  By default both resolvers return
# ``None``; callers fall back to the thread-local context (set by
# ``set_thread_*_context`` for background threads and tests).
# ---------------------------------------------------------------------------


def _default_context_resolver() -> Any:
    return None


_dataset_context_resolver: Callable[[], Any] = _default_context_resolver
_detector_context_resolver: Callable[[], Any] = _default_context_resolver


def register_dataset_context_resolver(fn: Callable[[], Any]) -> None:
    """Install the function used to resolve the current request's dataset context.

    The resolver should return a ``DatasetContext`` or ``None``.  The Flask
    shim installs a resolver that reads from ``flask.g`` at app startup;
    library-only callers can leave the default in place.
    """
    global _dataset_context_resolver
    _dataset_context_resolver = fn


def register_detector_context_resolver(fn: Callable[[], Any]) -> None:
    """Install the function used to resolve the current request's detector context.

    Counterpart to :func:`register_dataset_context_resolver`.
    """
    global _detector_context_resolver
    _detector_context_resolver = fn


# ---------------------------------------------------------------------------
# DatasetContext: bundles all per-dataset mutable state
# ---------------------------------------------------------------------------


class MediasDict(dict):
    """A ``dict`` of media items that bumps a revision counter on mutation.

    Backs :attr:`DatasetContext.medias`.  The embedding-matrix and
    region-matrix caches key their validity on ``ctx.media_revision``
    (see :mod:`vtscore.embedding.matrix`); routing every add / remove /
    replace through this subclass means those *structural* mutations bump
    the revision automatically, so no call site has to remember to
    invalidate the derived caches after changing which media are loaded.

    Limitation (important): a ``dict`` subclass only observes changes to
    the *key → value* mapping.  An in-place edit of a value dict - e.g.
    ``medias[cid]["embedding"] = vec`` while re-embedding or clipping -
    never calls ``__setitem__`` on this container, so it does **not** bump
    the counter.  Code that rewrites a media's vector in place must bump it
    itself via :meth:`DatasetContext.bump_media_revision`; the embed / clip
    load stages do this through ``invalidate_embedding_matrix``.  This is
    the ``media_revision`` follow-up (logical-bug-audit root-cause
    Pattern #4).

    Over-bumping (bumping when nothing actually changed) is always safe: it
    only forces an unnecessary cache rebuild, never serves a stale one.
    """

    __slots__ = ("_on_mutate",)

    def __init__(self, on_mutate: Callable[[], None], *args: Any, **kwargs: Any) -> None:
        # dict's C-level constructor / update populate directly without
        # calling our Python __setitem__, so seeding from *args* does not
        # fire _on_mutate - the owner counts construction as one bump when it
        # assigns the dict, not once per seeded item.
        object.__setattr__(self, "_on_mutate", on_mutate)
        super().__init__(*args, **kwargs)

    def __setitem__(self, key: Any, value: Any) -> None:
        super().__setitem__(key, value)
        self._on_mutate()

    def __delitem__(self, key: Any) -> None:
        super().__delitem__(key)  # raises before the bump if key is absent
        self._on_mutate()

    def pop(self, *args: Any) -> Any:
        n = len(self)
        result = super().pop(*args)
        if len(self) != n:
            self._on_mutate()
        return result

    def popitem(self) -> Any:
        result = super().popitem()  # raises on empty before the bump
        self._on_mutate()
        return result

    def clear(self) -> None:
        had_items = bool(self)
        super().clear()
        if had_items:
            self._on_mutate()

    def update(self, *args: Any, **kwargs: Any) -> None:
        super().update(*args, **kwargs)
        self._on_mutate()

    def setdefault(self, key: Any, default: Any = None) -> Any:
        existed = key in self
        result = super().setdefault(key, default)
        if not existed:
            self._on_mutate()
        return result


class DatasetContext:
    """All mutable state that belongs to a single loaded dataset.

    Vote-related state (``good_votes``, ``bad_votes``, ``label_history``, etc.)
    lives in :class:`DetectorContext`, not here.  ``DatasetContext`` holds only
    dataset-intrinsic state: the media items, coverage atlas, and display name.
    """

    __slots__ = (
        "dataset_id",
        # Backing store for the ``medias`` property (a :class:`MediasDict`)
        # and the monotonic revision counter it bumps.  ``media_revision``
        # advances on every structural mutation of the medias (and on every
        # in-place vector rewrite that calls ``bump_media_revision``); the
        # embedding / region matrix caches key their validity on it instead
        # of comparing sorted id lists (logical-bug-audit root-cause
        # Pattern #4).
        "_medias",
        "_media_revision",
        "coverage_atlas",
        "dataset_display_name",
        # Cached contiguous (N, D) float32 embedding matrix, the sorted
        # media-id list it corresponds to, and the ``media_revision`` it was
        # built at.  Built lazily on first access by
        # ``vtscore.embedding.matrix.get_embedding_matrix`` and reused
        # across cosine sort, MLP scoring, and coverage-atlas construction so
        # we don't rebuild a 10k-row matrix per call.  Cache validity is the
        # revision match; the id list is kept for the returned tuple / region
        # snap-key match.
        "_emb_matrix_ids",
        "_emb_matrix",
        "_emb_matrix_revision",
        # Memoised answer to "do *all* these medias share one primary
        # embedder, and which?" (``None`` = they disagree, or some media has
        # no primary), plus the ``media_revision`` it was computed at.
        # ``vtscore.embedding.matrix._collapse_to_primary_for_ctx`` asks this
        # before the matrix-cache check to decide whether an explicitly routed
        # embedder name is the whole dataset's primary and so may reuse the
        # cached primary matrix (issue #3650).  Memoised because that question
        # is an O(N) Python scan on a path that otherwise touches no media.
        "_uniform_primary",
        "_uniform_primary_revision",
        # One-way latch: set the first time ``invalidate_embedding_matrix``
        # fires for this context (an in-place vector rewrite - re-embed /
        # clip). Once set, ``get_embedding_matrix`` never again considers the
        # on-disk mmap sidecar (see S1, docs/plans/scalability.md) for this
        # context's lifetime, even though the id list alone can't detect a
        # same-id in-place rewrite. A fresh load gets a fresh ``DatasetContext``
        # (and so a fresh, unset latch), so this never needs resetting.
        "_emb_sidecar_disabled",
        # Cached *flattened region matrix* for patch-region (e.g. DINOv3)
        # datasets, mirroring ``_emb_matrix`` but expanded to one row per
        # (media, region) pair.  ``_region_matrix`` is the ``(R, D)`` float32
        # matrix; ``_region_media_index`` / ``_region_index_per_row`` are the
        # parallel ``int64`` arrays mapping each row back to its media index
        # (into the sorted id list) and its region index within that media.
        # Built lazily by ``vtscore.embedding.matrix.get_region_matrix_for_snap``
        # and reused across votes so online retraining never rebuilds the
        # multi-hundred-thousand-row matrix per vote.  Keyed, like
        # ``_emb_matrix``, on the sorted media-id list in ``_region_matrix_ids``
        # plus the ``media_revision`` it was built at (``_region_matrix_revision``).
        "_region_matrix_ids",
        "_region_matrix",
        "_region_matrix_revision",
        "_region_media_index",
        "_region_index_per_row",
        # Cached secondary media lookups (S14): the ``(origin_key, md5, name)``
        # triple ``build_media_lookup`` produces, keyed - like the embedding
        # matrix above - on the ``media_revision`` it was built at
        # (``_lookup_index_revision``).  Many routes resolve label entries
        # against the active dataset (label import/export, find-stats,
        # add-to-pile, learned sort); rebuilding these O(N) tables (a
        # ``json.dumps`` per origin) on every request is what S14 removes.
        # Built + reused by ``vtscore.state.media_lookup.cached_media_lookups``;
        # ``None`` until first access, invalidated automatically because the
        # revision advances on every structural medias mutation.
        "_origin_key_index",
        "_md5_index",
        "_name_index",
        "_lookup_index_revision",
        # VTSBrowse: cached projection (frozen at ingest) + per-bin-shape
        # pyramids derived from it. The projection (UMAP coords) is shared
        # across bin shapes; ``_pyramids`` maps "hex"/"square" -> Pyramid so the
        # browse hex/square toggle can keep both binnings cached at once.
        # ``_full_job_id`` tracks the in-flight background UMAP build for status
        # polling — the projection runner is a single app-wide slot shared with
        # every other dataset's full build and every subset build, so a poll
        # must look up *this* dataset's own job by id rather than asking the
        # runner what it happens to be running right now (which may be a
        # different dataset's job while this one sits parked pending).
        "_projection",
        "_pyramids",
        "_full_job_id",
        # VTSBrowse region signposts (the "street sign" name layer; see
        # docs/plans/vtsbrowse-toponymy.md).  A RegionLabelSet computed by the
        # labeling pipeline for the *current* frozen layout, or None.  The set
        # carries the projection_id it was fit against, and the labels route
        # refuses to serve it over a different layout — so a stale set is
        # inert, never wrong.  Text + 2-D anchors only, no vectors.
        "_region_labels",
        # In-flight background relabel job id (issue #2404).  When the serve
        # path finds a persisted label set whose ``labeler_signature`` no
        # longer matches the active pipeline, it kicks a background rebuild so
        # the stale signs self-heal; this tracks that job so repeated polls
        # coalesce onto the one rebuild instead of queueing a fresh one each
        # time.  Shares the ``signpost_relabel_jobs`` runner across datasets.
        "_relabel_job_id",
        # VTSBrowse subset projection: an ephemeral UMAP fit over just a subset
        # of this dataset's media ids (e.g. the positives of a Find run),
        # computed on demand and never persisted.  Held alongside the full
        # projection so a user can browse the whole dataset and a subset
        # independently.  ``_subset_ids`` is the sorted id list the cached
        # subset layout was fit against (cache key); ``_subset_job_id`` tracks
        # the in-flight background UMAP build for status polling.
        "_subset_projection",
        "_subset_pyramids",
        "_subset_ids",
        "_subset_job_id",
        "_subset_content_version",
        # Region signposts for the subset layout (mirror of ``_region_labels``).
        "_subset_region_labels",
        # Role-typed embedder binding (v3 "three-slot" trio; see
        # docs/plans/patch-embedder.md).  A dataset binds up to one
        # text-capable embedder, up to one patch-capable embedder, and up to
        # one structural (geometric-verification) embedder.  By default the
        # binding is *derived* on demand from the dataset's medias (the
        # single embedder a pre-v3 dataset was loaded with);
        # ``bind_embedders`` overrides that with an explicit, validated
        # triple for genuinely multi-embedder datasets.  The values are
        # embedder *names*, never vectors - no embeddings live here.
        "_text_embedder",
        "_patch_embedder",
        "_structural_embedder",
        "_binding_explicit",
        # Transient create-time flag: when set, the finalize stage runs an
        # extra near-duplicate collapse (images + text) after exact MD5
        # dedup.  Never persisted - the *result* is baked into origins, so
        # reloads default this off.
        "merge_near_duplicates",
    )

    # ------------------------------------------------------------------
    # Derived-cache families (deliberately adjacent to ``__slots__``)
    # ------------------------------------------------------------------
    # Every slot above is either a *source of truth* for this dataset or a
    # *derived cache* - something rebuildable, at some cost, from the medias
    # or from the persisted layout.  The four families below name the caches
    # and :meth:`reset_derived_caches` drops them by family; every call site
    # that used to hand-clear a list of private slots goes through it now.
    #
    # ``_NON_DERIVED_SLOTS`` names the rest explicitly so the two sets can be
    # asserted to *partition* ``__slots__`` (see
    # ``tests_lib/core/test_derived_cache_reset.py``).  That partition is the
    # whole point: a slot added to ``__slots__`` fails the test until it is
    # filed as a cache or as state, instead of being quietly missed the way
    # every hand-written clear-list here had already missed eleven slots by
    # the time issue #3377 counted them.
    _DERIVED_CACHE_SLOTS: dict[str, frozenset[str]] = {
        # Contiguous embedding / flattened-region matrices and their keys.
        "matrices": frozenset(
            {
                "_emb_matrix_ids",
                "_emb_matrix",
                "_emb_matrix_revision",
                "_uniform_primary",
                "_uniform_primary_revision",
                "_region_matrix_ids",
                "_region_matrix",
                "_region_matrix_revision",
                "_region_media_index",
                "_region_index_per_row",
            }
        ),
        # Secondary media lookups (origin key / md5 / name) and their key.
        "lookups": frozenset(
            {
                "_origin_key_index",
                "_md5_index",
                "_name_index",
                "_lookup_index_revision",
            }
        ),
        # VTSBrowse full-dataset layout: projection, per-shape pyramids,
        # signposts, and the in-flight build / relabel job ids that would
        # otherwise keep pointing at a discarded layout.
        "projection": frozenset(
            {
                "_projection",
                "_pyramids",
                "_full_job_id",
                "_region_labels",
                "_relabel_job_id",
            }
        ),
        # VTSBrowse subset layout: the mirror of the above, plus the id list
        # the cached subset was fit against and its tile-cache token.
        "subset": frozenset(
            {
                "_subset_projection",
                "_subset_pyramids",
                "_subset_ids",
                "_subset_job_id",
                "_subset_content_version",
                "_subset_region_labels",
            }
        ),
    }

    #: Slots that are state, not cache - never touched by
    #: :meth:`reset_derived_caches`.  Each is here for a reason:
    #:
    #: * ``dataset_id`` / ``dataset_display_name`` / ``merge_near_duplicates``
    #:   and the four embedder-binding slots are dataset identity + config.
    #: * ``_medias`` / ``_media_revision`` are the source of truth the caches
    #:   are derived *from*.
    #: * ``coverage_atlas`` is derived, but it is persisted in the dataset
    #:   pickle and restored by the load pipeline, which owns its lifetime.
    #: * ``_emb_sidecar_disabled`` is a one-way latch, not a cache: it records
    #:   that an in-place vector rewrite happened, which no rebuild undoes.
    _NON_DERIVED_SLOTS: frozenset[str] = frozenset(
        {
            "dataset_id",
            "_medias",
            "_media_revision",
            "coverage_atlas",
            "dataset_display_name",
            "_emb_sidecar_disabled",
            "_text_embedder",
            "_patch_embedder",
            "_structural_embedder",
            "_binding_explicit",
            "merge_near_duplicates",
        }
    )

    def __init__(self, dataset_id: str = "") -> None:
        self.dataset_id: str = dataset_id
        # ``_media_revision`` must exist before ``_medias`` so the MediasDict's
        # mutate callback can safely bump it.  Assign the backing slot directly
        # (not via the ``medias`` setter) so construction leaves the revision
        # at 0 rather than counting itself as a mutation.
        self._media_revision: int = 0
        self._medias: MediasDict = MediasDict(self.bump_media_revision)
        self.coverage_atlas: Any = None  # CoverageAtlas | None
        self.dataset_display_name: str | None = None
        self._emb_matrix_ids: list[int] | None = None
        self._emb_matrix: Any = None  # np.ndarray | None
        self._emb_matrix_revision: int | None = None  # media_revision the matrix was built at
        self._uniform_primary: str | None = None  # primary embedder shared by every media, else None
        self._uniform_primary_revision: int | None = None  # media_revision that answer was computed at
        self._emb_sidecar_disabled: bool = False  # latched True by invalidate_embedding_matrix
        self._region_matrix_ids: list[int] | None = None
        self._region_matrix: Any = None  # np.ndarray | None, shape (R, D)
        self._region_matrix_revision: int | None = None  # media_revision the region matrix was built at
        self._region_media_index: Any = None  # np.ndarray | None, int64 (R,)
        self._region_index_per_row: Any = None  # np.ndarray | None, int64 (R,)
        # Cached secondary media lookups (S14); see the __slots__ comment.
        self._origin_key_index: dict[str, list[int]] | None = None
        self._md5_index: dict[str, list[int]] | None = None
        self._name_index: dict[str, list[int]] | None = None
        self._lookup_index_revision: int | None = None  # media_revision the lookups were built at
        self._projection: Any = None  # Projection | None
        self._pyramids: dict[str, Any] = {}  # bin_shape -> Pyramid
        self._full_job_id: str | None = None  # in-flight full-dataset build job id
        self._region_labels: Any = None  # RegionLabelSet | None (signposts, full layout)
        self._relabel_job_id: str | None = None  # in-flight signpost-relabel job id (#2404)
        self._subset_projection: Any = None  # Projection | None (ephemeral subset UMAP)
        self._subset_pyramids: dict[str, Any] = {}  # bin_shape -> Pyramid (subset)
        self._subset_ids: list[int] | None = None  # sorted ids the subset layout is fit on
        self._subset_job_id: str | None = None  # in-flight subset build job id
        self._subset_region_labels: Any = None  # RegionLabelSet | None (signposts, subset)
        # Bumped on each in-place edit of the subset layout (e.g. removing
        # false-positives from a Find browse).  The layout/``projection_id`` is
        # kept stable so the canvas preserves the viewport; this counter changes
        # only the tile cache key/URL so stale tiles aren't served (the tile URL
        # is otherwise cached ``immutable``).  Reset to 0 on a fresh subset fit.
        self._subset_content_version: int = 0
        # Role-typed embedder binding (see __slots__ comment).  ``None``
        # until either explicitly bound or derived from the medias.
        self._text_embedder: str | None = None
        self._patch_embedder: str | None = None
        self._structural_embedder: str | None = None
        self._binding_explicit: bool = False
        self.merge_near_duplicates: bool = False

    # ------------------------------------------------------------------
    # Medias + revision counter (root-cause Pattern #4)
    # ------------------------------------------------------------------

    @property
    def medias(self) -> MediasDict:
        """The dataset's ``{id: media}`` map, as a revision-tracking dict.

        Structural mutations (add / remove / replace an entry) bump
        :attr:`media_revision` automatically via :class:`MediasDict`.
        """
        return self._medias

    @medias.setter
    def medias(self, value: dict[int, dict[str, Any]]) -> None:
        # Wrap any assigned mapping in a fresh MediasDict bound to *this*
        # context, then count the wholesale replacement as one mutation so a
        # cached matrix built against the old contents is invalidated even
        # when the new id set happens to match the old one.
        self._medias = MediasDict(self.bump_media_revision, value)
        self.bump_media_revision()

    @property
    def media_revision(self) -> int:
        """Monotonic counter, advanced on every mutation of the medias.

        The embedding-matrix and region-matrix caches compare this single
        int instead of two sorted id lists, so a mutation that changes
        vectors without changing the id set still invalidates them
        (logical-bug-audit root-cause Pattern #4).
        """
        return self._media_revision

    def bump_media_revision(self) -> None:
        """Advance :attr:`media_revision` by one.

        Called automatically by :class:`MediasDict` on structural changes.
        Call it directly after an *in-place* rewrite of a media's embedding
        vector (which a dict subclass can't observe) so the derived matrix
        caches rebuild; the embed / clip stages do so via
        ``invalidate_embedding_matrix``.
        """
        self._media_revision += 1

    # ------------------------------------------------------------------
    # Derived-cache invalidation
    # ------------------------------------------------------------------

    def reset_derived_caches(
        self,
        *,
        matrices: bool = True,
        lookups: bool = True,
        projection: bool = True,
        subset: bool = True,
    ) -> None:
        """Drop this context's derived caches, by family.

        The single place any of the ``_DERIVED_CACHE_SLOTS`` above are
        cleared.  Before issue #3377 four call sites each hand-wrote their
        own list of private slots to ``None``, and all four had drifted from
        ``__slots__``: ``clear_medias`` alone was missing the four lookup
        slots, the two full-layout job ids, and all six subset slots, so a
        reload left a stale subset layout that
        ``POST /api/projection/subset`` would serve verbatim whenever the new
        id set happened to match the old one - the exact failure the
        ``clear_medias`` docstring already called out for ``_projection`` and
        ``_pyramids``.

        Each keyword drops one family; pass ``False`` to keep one. The
        default (everything) is what a wholesale medias clear wants.

        Does **not** bump :attr:`media_revision`. Dropping a cache is not a
        change to the medias, and the revision-keyed caches distinguish the
        two: a caller signalling an in-place vector rewrite (
        :func:`vtscore.embedding.matrix.invalidate_embedding_matrix`) bumps
        it itself.
        """
        with _state_lock:
            if matrices:
                self._emb_matrix_ids = None
                self._emb_matrix = None
                self._emb_matrix_revision = None
                self._uniform_primary = None
                self._uniform_primary_revision = None
                self._region_matrix_ids = None
                self._region_matrix = None
                self._region_matrix_revision = None
                self._region_media_index = None
                self._region_index_per_row = None
            if lookups:
                self._origin_key_index = None
                self._md5_index = None
                self._name_index = None
                self._lookup_index_revision = None
            if projection:
                self._projection = None
                self._pyramids = {}
                self._full_job_id = None
                self._region_labels = None
                self._relabel_job_id = None
            if subset:
                self._subset_projection = None
                self._subset_pyramids = {}
                self._subset_ids = None
                self._subset_job_id = None
                self._subset_content_version = 0
                self._subset_region_labels = None

    # ------------------------------------------------------------------
    # Role-typed embedder binding
    # ------------------------------------------------------------------

    def bind_embedders(
        self,
        *,
        text_embedder: str | None = None,
        patch_embedder: str | None = None,
        structural_embedder: str | None = None,
    ) -> None:
        """Explicitly bind role-typed embedders to this dataset.

        Validates that each slot points at an embedder with the matching
        capability (text / patch / structural) and then stores the triple,
        overriding the default media-derived binding.  Use this for genuinely
        multi-embedder datasets; a single-embedder dataset can rely on the
        derived binding instead.

        Stores embedder *names* only - never vectors or models.
        """
        from vtscore.embedding.binding import validate_binding  # noqa: PLC0415

        validate_binding(text_embedder, patch_embedder, structural_embedder)
        self._text_embedder = text_embedder
        self._patch_embedder = patch_embedder
        self._structural_embedder = structural_embedder
        self._binding_explicit = True

    def _resolve_binding(self) -> tuple[str | None, str | None, str | None]:
        """Return the ``(text_embedder, patch_embedder, structural_embedder)`` triple.

        An explicit binding (set via :meth:`bind_embedders`) wins; otherwise
        the triple is derived from the dataset's medias - the single embedder
        a pre-v3 dataset was loaded with, role-typed by its capabilities.
        """
        if self._binding_explicit:
            return (self._text_embedder, self._patch_embedder, self._structural_embedder)
        if not self.medias:
            return (None, None, None)
        from vtscore.embedding.binding import derive_binding_from_names  # noqa: PLC0415
        from vtscore.embedding.media_vectors import media_embedder_names  # noqa: PLC0415

        first = next(iter(self.medias.values()))
        return derive_binding_from_names(media_embedder_names(first))

    @property
    def text_embedder(self) -> str | None:
        """Name of the bound text-capable embedder, or ``None``."""
        return self._resolve_binding()[0]

    @property
    def patch_embedder(self) -> str | None:
        """Name of the bound patch-capable embedder, or ``None``."""
        return self._resolve_binding()[1]

    @property
    def structural_embedder(self) -> str | None:
        """Name of the bound structural (geometric-verification) embedder, or ``None``."""
        return self._resolve_binding()[2]

    def routed_embedder(self, role: str) -> str | None:
        """Resolve which bound embedder serves *role* (the v3 routing table).

        Roles (see "Routing rules" in ``docs/plans/patch-embedder.md``):

        * ``"text"`` - text queries (``POST /api/sort``): the text slot, or
          ``None`` (the caller 400s with ``supports_text=False``).
        * ``"patch"`` - region similarity / voting (``/api/find-label`` region
          overlays, region votes): the patch slot, or ``None`` (the caller 400s;
          region ops need a patch embedder).
        * ``"structural"`` - instance retrieval / geometric verification: the
          structural slot, or ``None`` (the caller 400s; structural ops need a
          structural embedder).
        * ``"score"`` - cosine example sort, the detector MLP (train + score),
          and the coverage atlas: the structural slot if bound, else the patch
          slot, else the text slot, else ``None``.  ``None`` means a slot-less
          single-vector dataset (e.g.  ``dinov2_single``); the matrix layer then
          reads each media's primary vector, so cosine sort / MLP scoring keep
          working rather than 400-ing.

        Returns an embedder *name* (or ``None``).  Pass the result straight to
        the embedder-aware matrix layer: a name equal to the dataset's primary
        embedder collapses to the cached primary path there, so the
        single-embedder hot path is unchanged byte-for-byte.
        """
        text, patch, structural = self._resolve_binding()
        if role == "text":
            return text
        if role == "patch":
            return patch
        if role == "structural":
            return structural
        if role == "score":
            return structural or patch or text
        raise ValueError(f"unknown embedder routing role: {role!r}")

    @property
    def supports_text(self) -> bool:
        """Whether this dataset can answer text queries (text slot is bound)."""
        return self.text_embedder is not None

    @property
    def supports_patch_regions(self) -> bool:
        """Whether this dataset has region overlays / voting (patch slot is bound)."""
        return self.patch_embedder is not None

    @property
    def supports_geometric_verification(self) -> bool:
        """Whether this dataset can do instance retrieval (structural slot is bound)."""
        return self.structural_embedder is not None


class DetectorContext:
    """All mutable state that belongs to a single loaded detector.

    Bundles per-detector vote state, training artifacts, and cached in-memory
    data (MLP, threshold, training media with embeddings).  Multiple detectors
    can be loaded simultaneously; one is "active" (feeding the labeling UI).
    """

    __slots__ = (
        "detector_id",
        "name",
        "media_type",
        # The concrete space the label cache / MLP is *currently* built in (an
        # adaptive marker, re-stamped when the active dataset's space changes).
        # Distinct from ``embedder_type``: it names whichever concrete embedder
        # of the detector's type the active dataset supplied, so the
        # cache-invalidation compare stays honest across same-type swaps.
        "embedder",
        # The detector's *locked* embedder type - ``"semantic"`` /
        # ``"patch_semantic"`` / ``"structural"`` - persisted on the detector
        # JSON and loaded here at detector load.  Immutable in memory (never
        # re-stamped).  The detector scores in whatever concrete embedder of
        # this type the active dataset binds.  Empty for a legacy detector,
        # where routing falls back to the dataset score precedence.  See
        # ``docs/plans/patch-embedder.md`` → "Per-detector embedder type".
        "embedder_type",
        # Vote state
        "good_votes",
        "bad_votes",
        "label_history",
        "vote_click_times",
        "vote_region_boxes",
        # Per-vote surfacing provenance (which flow / phase / sort / rank
        # surfaced the item).  Recorded at click time because none of it is
        # re-derivable later: the ranking is client-side ephemeral state and
        # the score's model is overwritten by the next retrain.  See
        # ``vtscore/datasets/vote_provenance.py``.
        "vote_provenance",
        "click_counter",
        # True when this detector's in-memory votes are find/scoring output
        # (set by /api/find-label) rather than genuine training labels.  While
        # set, ``sync_labels_to_loaded_detector`` refuses to persist the votes
        # so a scoring pass can't overwrite the detector's saved labelset.
        # Per-detector (not a process global) so a find pass on one detector
        # never blocks vote syncing on another.  Cleared whenever the votes are
        # re-derived from the on-disk labelset (detector load / dataset switch).
        "find_mode",
        # Training artifacts
        "last_learned_scores",
        "textsort_suggestions",
        "find_initial_labels",
        # Find-session verification state:
        # ``verified_ids`` are the ids the human has explicitly verified this Find
        # session (a dict used as an ordered set, like ``good_votes``);
        # ``find_scores`` is the frozen per-item detector score from the single
        # scoring pass, so an Inclusion (cutoff) change re-thresholds without
        # re-scoring.  Both are in-memory only and never persisted.
        "verified_ids",
        "find_scores",
        # True when the on-disk labelset has been changed (e.g. Find corrections
        # folded in + retrain) *since* this frozen Find evaluation was scored, so
        # the still-displayed find_initial_labels / find_scores reflect the
        # previous detector version.  Drives the "out of date" note in
        # ``GET /api/find/stats``.  Cleared on a fresh find-label scoring pass and
        # on any session reset (clear votes / dataset switch).
        "find_eval_stale",
        "inclusion",
        # Cached in-memory data (never exported)
        "training_medias",  # voted media items with embeddings
        "label_embeddings",  # str → np.ndarray, keyed by stable_element_id
        # Region box the cached ``label_embeddings`` entry was built against,
        # keyed by stable_element_id.  ``None`` means the cached vector is
        # image-level; a 4-tuple means it was pooled from that box.  Lets
        # ``populate_label_embeddings`` detect a region→none (or any region
        # edit) transition and re-resolve instead of returning a stale
        # region-pooled vector keyed to an element that no longer has a
        # region.  See logical-bug-audit finding M4.
        "label_embedding_regions",
        # Cross-dataset local features (StructuralFeatures) for the labelset's
        # elements, keyed by stable_element_id.  Re-derived from each element's
        # origin so a saved structural detector can build templates + train its
        # verification classifier against datasets that aren't currently loaded;
        # the full (unfiltered) features are cached and the region_box is applied
        # downstream at template-build time.  In-memory only, never persisted.
        # See docs/plans/structural-embedder.md.
        "label_local_features",  # str → StructuralFeatures
        # Region-flooded negatives for the labelset's Bad elements on patch
        # datasets: str → list[np.ndarray], keyed by stable_element_id, holding
        # the element's image-level vector + every raw patch so a saved detector
        # re-sorted cross-dataset floods the same rows the live vote path does.
        # In-memory only, re-derived from origins, never persisted; cleared on
        # embedder switch alongside ``label_embeddings``.
        "label_negative_regions",
        # Full score-row stacks for the labelset's elements on patch datasets:
        # str → list[np.ndarray], keyed by stable_element_id, holding the
        # image-level vector + every raw patch - the rows the scorer max-pools
        # that element's media over (identical to the flood above under
        # MaxPatch).  Lets threshold calibration collapse a Good bag and a Bad
        # bag the same way inference collapses any image, instead of comparing
        # a max-over-1 against a max-over-197.  In-memory only, re-derived from
        # origins, never persisted; cleared on embedder switch alongside
        # ``label_embeddings``.
        "label_score_regions",
        "model",  # nn.Sequential | None (current trained MLP)
        # Structural (SIFT/VLAD) detectors carry a *second* learned object next
        # to the retrieval MLP: the match-statistic verification classifier
        # (None until trained / for non-structural detectors).  In-memory only,
        # re-derived from votes on every retrain, never persisted.  See
        # docs/plans/structural-embedder.md.
        "verification_classifier",  # nn.Sequential | None
        "threshold",  # decision threshold
        # Cross-dataset training-corpus counts (from on-disk labelset).  These
        # are independent of ``good_votes``/``bad_votes``, which only count
        # labels for media in the *currently loaded* dataset.  They drive the
        # frontend's "Sort by Learned" gating so a detector trained on dataset
        # A stays trainable when the user switches to dataset B.
        "labelset_good_count",
        "labelset_bad_count",
        # Dataset ID for which the cid-keyed vote state above is valid.
        # Media IDs are only meaningful within a single dataset, so when the
        # active dataset changes for a loaded detector we must clear the cid
        # dicts and re-derive them from the on-disk labelset against the new
        # dataset's medias.  See ``ensure_votes_match_active_dataset``.
        "votes_dataset_id",
        # Cached parsed labelset + mtime of the on-disk detector JSON the cache
        # was derived from.  Lets ``ensure_votes_match_active_dataset`` skip the
        # rehydrate (read+parse) when neither the active dataset nor the file
        # has changed, and lets ``learned_sort`` reuse the parsed labelset
        # instead of re-reading the JSON from disk on every click.
        "cached_labelset",  # LabelSet | None
        "cached_labelset_mtime",  # float
        "cached_labelset_media_type",  # str
        # Sync source
        "labelset_source",  # dict | None: {"source_name": "...", "field_values": {...}}
        # Calibration folds cache.  Holds ``(key, CalibrationFolds)`` where
        # *key* is a deterministic fingerprint of the
        # **inclusion-independent** calibration inputs (training vectors,
        # labels, calibrate_count, calibration_fraction, hidden_dim) and the
        # payload carries the per-fold held-out ``(scores, labels)``, the
        # fallback sentinel, and the trained fold models.  Because
        # inclusion is deliberately absent from *key*, an Inclusion change hits
        # the cache and only re-runs the cheap quantile rule (no fold refit);
        # a label/embedder change rotates *key* and falls through to a fresh
        # calibration.
        "calibration_cache",  # tuple[Any, CalibrationFolds] | None
        # The fold-anchored population estimator behind the current threshold
        # (``FoldAnchoredCut``), or None when the estimator degenerated.  A
        # re-cut answers Inclusion: the shipped ``mid_tilt`` rule anchors the
        # measured midpoint cut at inclusion 0 and tilts monotonically away
        # from it (issue #2865).  Written on every retrain that computes a safe
        # threshold; read by ``recompute_detector_thresholds_for_inclusion``
        # and the Find Stats sweep so both re-cut the *shipped* estimator
        # instead of the raw cross-calibration one.  Holds fitted Gaussians and
        # sorted score samples - process-scoped, never serialised.
        "anchored_cut_cache",  # FoldAnchoredCut | None
    )

    def __init__(
        self,
        detector_id: str = "",
        *,
        name: str = "",
        media_type: str = "",
        embedder: str = "",
        embedder_type: str = "",
    ) -> None:
        self.detector_id: str = detector_id
        self.name: str = name
        self.media_type: str = media_type
        self.embedder: str = embedder
        self.embedder_type: str = embedder_type
        # Vote state
        self.good_votes: dict[int, None] = {}
        self.bad_votes: dict[int, None] = {}
        self.label_history: list[tuple[int, str, float]] = []
        self.vote_click_times: dict[int, int] = {}
        # Per-good-vote region boxes (normalised x0, y0, x1, y1).  Only set when
        # the user drew a region as part of a yes-vote; absent for image-level
        # yes-votes and for every no-vote.  Patch-embedder v2.
        self.vote_region_boxes: dict[int, tuple[float, float, float, float]] = {}
        # Per-vote surfacing provenance, keyed by media id.  See the slot
        # comment and :mod:`vtscore.datasets.vote_provenance`.
        self.vote_provenance: dict[int, dict[str, object]] = {}
        self.click_counter: int = 0
        # See the slot comment: True while these votes are find/scoring output.
        self.find_mode: bool = False
        # Training artifacts
        self.last_learned_scores: dict[int, float] = {}
        self.textsort_suggestions: list[str] = []
        self.find_initial_labels: dict[int, str] = {}
        # Find-session verification state.
        self.verified_ids: dict[int, None] = {}
        self.find_scores: dict[int, float] = {}
        # See the slot comment: the displayed Find evaluation is for the detector
        # version that scored this pass; flipped True when its labelset changes
        # underneath (corrections folded in + retrain).
        self.find_eval_stale: bool = False
        self.inclusion: int | None = None
        # Cached in-memory data (never exported)
        self.training_medias: dict[int, dict[str, Any]] = {}
        # Embeddings for every saved labelset element, keyed by
        # stable_element_id.  Populated at detector load (resolve_file +
        # embed_file) and topped up when new votes come in.  Lets MLP
        # training and learned-sort use *all* saved labels, including
        # those whose underlying media isn't part of the active dataset.
        self.label_embeddings: dict[str, Any] = {}
        self.label_embedding_regions: dict[str, tuple[float, float, float, float] | None] = {}
        self.label_local_features: dict[str, Any] = {}
        self.label_negative_regions: dict[str, list[Any]] = {}
        self.label_score_regions: dict[str, list[Any]] = {}
        self.model: Any = None  # nn.Sequential | None
        # Match-statistic verification classifier for structural detectors;
        # None for non-structural detectors and until first trained.
        self.verification_classifier: Any = None  # nn.Sequential | None
        self.threshold: float = 0.5
        self.labelset_good_count: int = 0
        self.labelset_bad_count: int = 0
        self.votes_dataset_id: str = ""
        self.cached_labelset: Any = None  # LabelSet | None
        self.cached_labelset_mtime: float = 0.0
        self.cached_labelset_media_type: str = ""
        # Sync source
        self.labelset_source: dict[str, Any] | None = None
        self.calibration_cache: tuple[Any, Any] | None = None
        self.anchored_cut_cache: Any = None  # FoldAnchoredCut | None


# ---------------------------------------------------------------------------
# Dataset context store and thread-local fallback
# ---------------------------------------------------------------------------

# Maps dataset_id -> DatasetContext for every in-memory dataset.
_contexts: dict[str, DatasetContext] = {}

# Thread-local storage for the fallback dataset/detector context.
# Used by background threads and tests that operate outside a Flask
# request context.  Each thread sets its own value; no global "active"
# pointer exists.
_thread_local = threading.local()

# Fallback context used when no dataset is set.  Proxies delegate to
# this so that code accessing ``medias`` when nothing is loaded sees empty
# containers rather than crashing.
_empty_dataset_context = DatasetContext("")


class _RequestMissingDatasetContext(DatasetContext):
    """Sentinel returned inside a Flask request when no dataset was identified.

    Behaves as an empty :class:`DatasetContext` for reads, but every
    container is a :class:`_FrozenDict` / :class:`_FrozenList` that raises
    :class:`RequestMissingContextError` on any mutation, and the context
    itself refuses attribute assignment.  This converts the "header was
    dropped and we silently fell back to the empty global context"
    failure mode (audit bugs H13 / H16) into a loud error at the actual
    write site.
    """

    __slots__ = ()

    def __init__(self) -> None:
        # Copy every slot from a fresh, real ``DatasetContext`` (containers
        # frozen on the way in) rather than hand-listing them, so a slot added
        # to ``DatasetContext`` can never go missing here.  ``medias`` is a
        # property on the base class; ``_freeze_into`` writes its backing slot
        # directly with a frozen dict, so every mutation raises.  The revision
        # counter is inert here (a frozen dict never mutates), but the slot
        # must exist for the property / bump machinery not to AttributeError.
        _freeze_into(self, DatasetContext(""), "dataset")
        # Use object.__setattr__ to bypass our own write guard.
        object.__setattr__(self, "dataset_id", "__request_missing__")

    def __setattr__(self, name: str, value: Any) -> None:
        raise _frozen_mutation_error("dataset")


_request_missing_dataset_context = _RequestMissingDatasetContext()


def is_request_missing_dataset_context(ctx: Any) -> bool:
    """Return True iff *ctx* is the request-missing dataset sentinel."""
    return ctx is _request_missing_dataset_context


def get_active_context() -> DatasetContext:
    """Return the ``DatasetContext`` for the current execution context.

    Resolution order:
    1. Request-scoped context (set by ``before_request`` from ``X-Dataset-Id`` header)
    2. Thread-local context (set by ``set_thread_dataset_context``, for
       background threads and tests)
    3. Request-missing sentinel, when inside a Flask request that didn't
       identify a dataset (registered via
       :func:`register_request_context_predicate`).  Reads see an empty
       context; writes raise :class:`RequestMissingContextError`.
    4. Empty fallback context for CLI / library callers outside any
       Flask request.
    """
    # 1. Per-request override (Flask shim or whatever the host app registered)
    req_ctx = _dataset_context_resolver()
    if req_ctx is not None:
        return req_ctx
    # 2. Thread-local fallback
    ctx = getattr(_thread_local, "dataset_context", None)
    if ctx is not None:
        return ctx
    # 3. Inside a Flask request with no header and no thread-local → fail
    #    loudly on mutation instead of silently writing into the global
    #    empty context.
    if _request_context_predicate():
        return _request_missing_dataset_context
    return _empty_dataset_context


def set_thread_dataset_context(ctx: DatasetContext | None) -> None:
    """Set the thread-local dataset context for the current thread.

    Prefer :func:`thread_dataset_context` (a context manager) for new code
    (it saves and restores the prior value automatically).

    Called by test fixtures and background threads to direct proxy
    resolution without global state.
    """
    _thread_local.dataset_context = ctx


def get_thread_dataset_context() -> DatasetContext | None:
    """Return the thread-local dataset context, or ``None``."""
    return getattr(_thread_local, "dataset_context", None)


@contextmanager
def thread_dataset_context(ctx: DatasetContext | None) -> Iterator[None]:
    """Scope the thread-local dataset context to *ctx* for the ``with``-block.

    Snapshots the prior thread-local value on entry and restores it on
    exit, so nested scopes compose correctly and a reused / pooled
    thread cannot leak the wrong dataset context across jobs.

    Unlike :class:`with_dataset_context` (which looks the context up by
    ID from the registry and requires it to be registered), this helper
    takes a context object directly; useful for newly-created contexts
    that have not yet been registered.
    """
    prev = getattr(_thread_local, "dataset_context", None)
    _thread_local.dataset_context = ctx
    try:
        yield
    finally:
        _thread_local.dataset_context = prev


def register_context(ctx: DatasetContext) -> None:
    """Add *ctx* to the context store, keyed by its ``dataset_id``."""
    with _state_lock:
        _contexts[ctx.dataset_id] = ctx


def unregister_context(dataset_id: str) -> DatasetContext | None:
    """Remove and return the context for *dataset_id*, or ``None``."""
    with _state_lock:
        ctx = _contexts.pop(dataset_id, None)
        # Clear thread-local if it was pointing to the removed context.
        tl_ctx = getattr(_thread_local, "dataset_context", None)
        if tl_ctx is not None and tl_ctx.dataset_id == dataset_id:
            _thread_local.dataset_context = None
        return ctx


def get_context(dataset_id: str) -> DatasetContext | None:
    """Return the context for *dataset_id*, or ``None`` if not loaded."""
    with _state_lock:
        return _contexts.get(dataset_id)


def list_loaded_dataset_ids() -> list[str]:
    """Return all dataset IDs that have an in-memory context."""
    with _state_lock:
        return list(_contexts.keys())


def clear_all_contexts() -> None:
    """Remove all dataset contexts and clear the thread-local.  For tests."""
    with _state_lock:
        _contexts.clear()
        _thread_local.dataset_context = None
        # Also reset the empty context's state
        _empty_dataset_context.__init__("")  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Detector context store and thread-local fallback
# ---------------------------------------------------------------------------

# Maps detector_id -> DetectorContext for every in-memory detector.
_detector_contexts: dict[str, DetectorContext] = {}

# Fallback context used when no detector is set.  Vote proxies delegate
# to this so that code accessing ``good_votes``, ``bad_votes``, etc. when
# no detector is loaded sees empty containers rather than crashing.
_empty_detector_context = DetectorContext("")


class _RequestMissingDetectorContext(DetectorContext):
    """Sentinel returned inside a Flask request when no detector was identified.

    Counterpart of :class:`_RequestMissingDatasetContext`: every container
    is frozen and the context refuses attribute assignment.  Without this
    sentinel, vote-mutation endpoints called without ``X-Detector-Id``
    would silently accumulate votes on the global
    ``_empty_detector_context`` (audit bug H13 / H14).
    """

    __slots__ = ()

    def __init__(self) -> None:
        # Copy every slot from a fresh, real ``DetectorContext`` (containers
        # frozen on the way in) rather than hand-listing them, so a slot added
        # to ``DetectorContext`` can never go missing here.
        _freeze_into(self, DetectorContext(""), "detector")
        # Use object.__setattr__ to bypass our own write guard.
        object.__setattr__(self, "detector_id", "__request_missing__")

    def __setattr__(self, name: str, value: Any) -> None:
        raise _frozen_mutation_error("detector")


_request_missing_detector_context = _RequestMissingDetectorContext()


def is_request_missing_detector_context(ctx: Any) -> bool:
    """Return True iff *ctx* is the request-missing detector sentinel."""
    return ctx is _request_missing_detector_context


def is_request_missing_context(ctx: Any) -> bool:
    """Return True iff *ctx* is either request-missing sentinel."""
    return ctx is _request_missing_dataset_context or ctx is _request_missing_detector_context


def get_active_detector_context() -> DetectorContext:
    """Return the ``DetectorContext`` for the current execution context.

    Resolution order:
    1. Forced override (``override_detector_context`` context manager)
    2. Request-scoped context (set by ``before_request`` from ``X-Detector-Id`` header)
    3. Thread-local context (set by ``set_thread_detector_context``)
    4. Request-missing sentinel, inside a Flask request with no header
       and no thread-local; mutations raise
       :class:`RequestMissingContextError`.
    5. Empty fallback context for CLI / library callers outside Flask.
    """
    # 1. Forced override (set by override_detector_context context manager)
    forced = getattr(_thread_local, "forced_detector_context", None)
    if forced is not None:
        return forced
    # 2. Per-request override (Flask shim or whatever the host app registered)
    req_ctx = _detector_context_resolver()
    if req_ctx is not None:
        return req_ctx
    # 3. Thread-local fallback
    ctx = getattr(_thread_local, "detector_context", None)
    if ctx is not None:
        return ctx
    # 4. Inside a Flask request with no header and no thread-local → fail
    #    loudly on mutation instead of polluting _empty_detector_context.
    if _request_context_predicate():
        return _request_missing_detector_context
    return _empty_detector_context


@contextmanager
def override_detector_context(ctx: DetectorContext) -> Iterator[None]:
    """Force :func:`get_active_detector_context` to return *ctx* for the
    duration of the ``with`` block.

    Takes priority over the registered request resolver and the thread-local
    fallback.  Use this from call sites that need to swap the active detector
    inside their own body (typically when applying labels to a freshly-loaded
    detector that isn't the request's currently-active one) without having
    to know whether they're running inside a Flask request or a background
    thread.
    """
    prev = getattr(_thread_local, "forced_detector_context", None)
    _thread_local.forced_detector_context = ctx
    try:
        yield
    finally:
        _thread_local.forced_detector_context = prev


def set_thread_detector_context(ctx: DetectorContext | None) -> None:
    """Set the thread-local detector context for the current thread.

    Prefer :func:`thread_detector_context` (a context manager) for new
    code (it saves and restores the prior value automatically).
    """
    _thread_local.detector_context = ctx


def get_thread_detector_context() -> DetectorContext | None:
    """Return the thread-local detector context, or ``None``."""
    return getattr(_thread_local, "detector_context", None)


@contextmanager
def thread_detector_context(ctx: DetectorContext | None) -> Iterator[None]:
    """Scope the thread-local detector context to *ctx* for the ``with``-block.

    Snapshots the prior thread-local value on entry and restores it on
    exit, so nested scopes compose correctly and a reused / pooled
    thread cannot leak the wrong detector context across jobs.

    Unlike :class:`with_detector_context` (which looks the context up by
    ID from the registry and requires it to be registered), this helper
    takes a context object directly.
    """
    prev = getattr(_thread_local, "detector_context", None)
    _thread_local.detector_context = ctx
    try:
        yield
    finally:
        _thread_local.detector_context = prev


def register_detector_context(ctx: DetectorContext) -> None:
    """Add *ctx* to the detector context store, keyed by its ``detector_id``.

    Also clears the module-level progress cache so that stale training
    indicators from a previously-active detector are not reused.
    """
    from vtscore.detectors.labeling_progress import clear_progress_cache

    with _state_lock:
        _detector_contexts[ctx.detector_id] = ctx
    # ``_progress_lock`` is acquired strictly outside ``_state_lock`` so the
    # two locks never establish a cross-module ordering (audit M1).
    clear_progress_cache()


def unregister_detector_context(detector_id: str) -> DetectorContext | None:
    """Remove and return the detector context for *detector_id*, or ``None``.

    Also clears the progress cache so stale cached steps from the removed
    detector are not used by a subsequent detector.
    """
    from vtscore.detectors.labeling_progress import clear_progress_cache

    with _state_lock:
        ctx = _detector_contexts.pop(detector_id, None)
        tl_ctx = getattr(_thread_local, "detector_context", None)
        if tl_ctx is not None and tl_ctx.detector_id == detector_id:
            _thread_local.detector_context = None
    # ``_progress_lock`` is acquired strictly outside ``_state_lock`` so the
    # two locks never establish a cross-module ordering (audit M1).
    clear_progress_cache()
    return ctx


def get_detector_context(detector_id: str) -> DetectorContext | None:
    """Return the detector context for *detector_id*, or ``None`` if not loaded."""
    with _state_lock:
        return _detector_contexts.get(detector_id)


def list_loaded_detector_ids() -> list[str]:
    """Return all detector IDs that have an in-memory context."""
    with _state_lock:
        return list(_detector_contexts.keys())


def clear_all_detector_contexts() -> None:
    """Remove all detector contexts and clear the thread-local.  For tests."""
    with _state_lock:
        _detector_contexts.clear()
        _thread_local.detector_context = None
        _empty_detector_context.__init__("")  # type: ignore[misc]


def invalidate_loaded_detector_models() -> None:
    """Drop the cached MLP and threshold on every loaded detector context.

    Called by the setters of training-relevant settings (``inclusion``,
    ``calibrate_count``, ``calibration_fraction``) so
    the next consumer that would otherwise short-circuit on the cached
    ``det_ctx.model`` / ``det_ctx.threshold`` (``/api/find-label``,
    ``/api/find``, ``/api/auto-detect``) retrains under the new setting.

    Sort / vote paths already retrain every call, so this is purely about
    making the cached-MLP consumers honour live setting changes.
    """
    with _state_lock:
        for ctx in _detector_contexts.values():
            ctx.model = None
            ctx.threshold = 0.5


def detector_acquisition_threshold(ctx: "DetectorContext", inclusion_value: int) -> float:
    """The cut Autopilot's ``hard`` / ``new`` picks should sample around.

    **Not the decision line.**  ``ctx.threshold`` is what the user sees and what
    Find calls a match; this is a second cut taken from the *same* fitted
    estimator, :data:`~vtscore.training.thresholds.ACQUISITION_INCLUSION_OFFSET`
    inclusion steps below it, because the picks read a threshold as a rank
    position rather than a boundary and so want it further *up* the ranking.
    Decoupling the two buys 4.5x the positives per 100 votes at lower cost - see
    ``docs/experiments/2026-08-07-acquisition-inclusion/REPORT.md`` (PR #2876).

    Derived on demand rather than stored beside ``ctx.threshold``: there are
    four places that write a threshold onto a detector context, and a second
    field would be one more thing for each of them to forget.  Re-cutting is
    arithmetic on already-fitted Gaussians, so it is cheap enough to do per
    request.

    Falls back to ``ctx.threshold`` when there is no fold-anchored estimator to
    re-cut (safe thresholds off, or a degenerate fit that fell back to the
    schedule blend, which has no inclusion-aware form) - the two jobs coincide
    there, exactly as they did everywhere before #2876.
    """
    cut = ctx.anchored_cut_cache
    if cut is None:
        return ctx.threshold
    from vtscore.training.thresholds import acquisition_inclusion

    candidate = float(cut.threshold_at(acquisition_inclusion(inclusion_value)))
    return candidate if math.isfinite(candidate) else ctx.threshold


def recompute_detector_thresholds_for_inclusion(inclusion_value: int) -> None:
    """Re-derive each loaded detector's threshold at *inclusion_value* from its
    cached fold orderings, leaving the (inclusion-independent) MLP in place.

    Inclusion is a pure cutoff knob now: a change must not drop the model or
    re-score the haystack - only move the threshold over already-computed
    scores.  Detectors with no cached fold orderings yet are left untouched;
    the next training pass computes the threshold under the new inclusion.

    **The safe threshold is re-derived faithfully.**  With safe thresholds on,
    a fresh retrain stores the fold-anchored population cut
    (:func:`vtscore.training.thresholds.fold_anchored_gmm_threshold`), and the
    fitted estimator is parked on ``ctx.anchored_cut_cache``.  Re-cutting it at
    a new inclusion is arithmetic on the already-fitted Gaussians, so a slide
    reproduces exactly what a retrain at that inclusion would have stored -
    without touching the model or re-scoring the haystack.  Detectors with no
    anchored cut (safe thresholds off, or a degenerate fit that fell back to
    the blend) slide on the raw cross-calibration rule over the cached fold
    orderings, as they always have.
    """
    from vtscore.training.thresholds import threshold_from_fold_orderings

    with _state_lock:
        for ctx in _detector_contexts.values():
            cut = ctx.anchored_cut_cache
            if cut is not None:
                ctx.threshold = cut.threshold_at(inclusion_value)
                continue
            cache = ctx.calibration_cache
            if cache is None:
                continue
            folds = cache[1]
            if folds.fallback is not None:
                ctx.threshold = folds.fallback
            elif folds.orderings:
                ctx.threshold = threshold_from_fold_orderings(folds.orderings, inclusion_value)


# ---------------------------------------------------------------------------
# Scalar state accessors
# ---------------------------------------------------------------------------
# These thin helpers wrap "give me the X of whatever context is currently
# active" so callers that operate on the active context but don't need
# a full DatasetContext / DetectorContext reference can stay one-liners.
# Dataset-intrinsic scalars (coverage_atlas, dataset_display_name) delegate
# to the active DatasetContext.  Detector-related scalars (click_counter,
# inclusion) delegate to the active DetectorContext.
# ---------------------------------------------------------------------------


def _get_click_counter() -> int:
    return get_active_detector_context().click_counter


def _set_click_counter(value: int) -> None:
    get_active_detector_context().click_counter = value


def _get_dataset_display_name() -> str | None:
    return get_active_context().dataset_display_name


def _set_dataset_display_name(value: str | None) -> None:
    get_active_context().dataset_display_name = value


def _get_inclusion() -> int | None:
    return get_active_detector_context().inclusion


def _set_inclusion(value: int | None) -> None:
    ctx = get_active_detector_context()
    # ``inclusion`` is cached per-detector for fast reads, but its canonical
    # persisted home is the per-user settings store (written by the caller's
    # ``_persist_setting`` hook). When a Flask request identifies no detector
    # (e.g. the VTSBrowser, which has a dataset but no loaded detector), the
    # active context is the frozen request-missing sentinel; there is no
    # detector to cache the value on, so skip the cache write rather than
    # raising ``RequestMissingContextError``. The user-settings persist still
    # runs, so an inclusion value echoed back by a bulk settings save is a
    # harmless no-op instead of a 400.
    if is_request_missing_detector_context(ctx):
        return
    ctx.inclusion = value


# ---------------------------------------------------------------------------
# Context managers for explicit, scoped context switching
# ---------------------------------------------------------------------------


class with_dataset_context:
    """Context manager for temporarily switching the active dataset.

    Saves the current active dataset ID on entry, switches to the
    requested *dataset_id*, and restores the original on exit
    even if an exception occurs.

    Usage::

        with with_dataset_context("my_dataset"):
            # code here sees my_dataset's medias, coverage atlas, etc.
            print(len(medias))
        # original dataset is restored here

    .. warning::
        This is NOT thread-safe.  Only use from a single thread or
        protect with ``_state_lock`` externally.
    """

    __slots__ = ("_target_id", "_previous_ctx")

    def __init__(self, dataset_id: str) -> None:
        self._target_id = dataset_id
        self._previous_ctx: DatasetContext | None = None

    def __enter__(self) -> DatasetContext:
        self._previous_ctx = get_thread_dataset_context()
        ctx = get_context(self._target_id)
        if ctx is None:
            raise ValueError(f"No dataset context registered for {self._target_id!r}")
        set_thread_dataset_context(ctx)
        return ctx

    def __exit__(self, *exc_info: object) -> None:
        set_thread_dataset_context(self._previous_ctx)


class with_detector_context:
    """Context manager for temporarily switching the active detector.

    Saves the current active detector ID on entry, switches to the
    requested *detector_id*, and restores the original on exit.

    Usage::

        with with_detector_context("my_detector"):
            # code here sees my_detector's votes, scores, etc.
            print(len(good_votes))
        # original detector is restored here

    .. warning::
        This is NOT thread-safe.  Only use from a single thread or
        protect with ``_state_lock`` externally.
    """

    __slots__ = ("_target_id", "_previous_ctx")

    def __init__(self, detector_id: str) -> None:
        self._target_id = detector_id
        self._previous_ctx: DetectorContext | None = None

    def __enter__(self) -> DetectorContext:
        self._previous_ctx = get_thread_detector_context()
        ctx = get_detector_context(self._target_id)
        if ctx is None:
            raise ValueError(f"No detector context registered for {self._target_id!r}")
        set_thread_detector_context(ctx)
        return ctx

    def __exit__(self, *exc_info: object) -> None:
        set_thread_detector_context(self._previous_ctx)
