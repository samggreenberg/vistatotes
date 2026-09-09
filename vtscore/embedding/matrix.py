"""Lazy, cached contiguous embedding matrix on :class:`DatasetContext`.

Building ``np.array([media_embedding(medias[cid]) for cid in sorted(...)])``
per request copies 10k+ arrays twice (once into a list, once into the
``np.array(...)`` allocation) and another copy when wrapping with
``torch.tensor(...)``.  The matrix changes only when the set of loaded
media IDs changes, so we cache one contiguous ``(N, D)`` float32 array
on the active dataset context and hand callers a ``torch.from_numpy``
view (zero-copy) when they need a tensor.

Cache invalidation is keyed on ``ctx.media_revision``: when the counter
differs from the one the cached matrix was built at, the matrix is
rebuilt.  Structural mutations of ``ctx.medias`` (add / remove / replace
an entry) bump the counter automatically via
:class:`~vtscore.state.core.MediasDict`, so callers don't need to do
anything.  An *in-place* rewrite of an existing media's vector (re-embed /
clip) is invisible to a dict subclass, so those stages call
``invalidate_embedding_matrix`` (which bumps the counter) after the
rewrite - see the ``media_revision`` root-cause pattern (logical-bug-audit
Pattern #4).

Any media whose ``embedding`` is ``None`` causes the builder to raise
``ValueError`` instead of silently filling the row with NaN - the bug
described as logical-bug-audit M11.  On numpy 2.x
``matrix[i] = None`` quietly stores ``nan`` and the resulting score
propagates through every downstream consumer (always-False threshold
compares, NaN-poisoned sort, JSON ``NaN`` in the response).  Raising
turns that into a loud, locatable failure naming the offending cid.

That strictness is right for the *dataset's own* matrix, where the load
pipeline's ``_drop_none_embeddings_stage`` has already removed vector-less
media, but wrong for an arbitrary snapshot handed to the scorer: the CLI
scores importer output that never went through that stage, and one bound
embedder of a multi-embedder dataset can legitimately have failed on media
another succeeded on.  :func:`scoreable_snapshot` is what those callers use to
skip the unusable items *before* the build, so a single failed image costs one
skipped item and a log line instead of the whole run.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from vtscore.embedding.media_vectors import media_embedder_names, media_embedding, primary_embedder_name
from vtscore.embedding.precomputed import MismatchedVectorError, require_dim
from vtscore.state.core import _state_lock

if TYPE_CHECKING:
    from vtscore.state.core import DatasetContext

logger = logging.getLogger(__name__)


def _uniform_primary_embedder(medias: dict[int, dict[str, Any]]) -> str | None:
    """The primary embedder *every* media in *medias* shares, else ``None``.

    ``None`` means "no single answer": the medias disagree (a mixed-type
    snapshot binding more than one embedding space), some media has no
    recorded primary at all, or the mapping is empty.  Short-circuits on the
    first disagreement, so the mixed case is cheap and only the homogeneous
    case pays the full scan.

    Callers use this to decide whether an explicitly-named embedder can be
    treated as the primary for the whole snapshot - see
    :func:`_collapse_to_primary`.
    """
    shared: str | None = None
    for media in medias.values():
        name = primary_embedder_name(media)
        if name is None:
            return None
        if shared is None:
            shared = name
        elif name != shared:
            return None
    return shared


def _collapse_to_primary(medias: dict[int, dict[str, Any]], embedder_name: str | None) -> str | None:
    """Map a routed embedder name to ``None`` when it is *every* media's primary.

    The routing layer (``DatasetContext.routed_embedder``) hands callers an
    explicit embedder name even for the common single-embedder dataset, where
    that name *is* the primary.  The named matrix path builds fresh on every
    call; the primary path is cached on the context.  When every media's
    primary is that name the two paths read byte-for-byte the same vectors, so
    we collapse to ``None`` here and the cache is reused - keeping the
    single-embedder hot path unchanged after routing threads names through.

    A name that differs from the primary (a genuine second bound embedder)
    passes through unchanged and takes the uncached named path.

    **The quantifier is "every", not "the first" (issue #3650).**  Sampling one
    media is right on a homogeneous snapshot and a sampling error on a mixed
    one: media #1's primary would decide the path for all N.  Ask for space
    ``A`` on a snapshot whose first media is an ``A`` media and whose rest are
    ``B`` media, and the name collapses, every media contributes *its own*
    vector, and ``B``-space rows get stacked into an ``A``-space matrix and
    scored through an ``A``-space head - silently, whenever the two spaces
    share a width (512-d and 768-d encoders are common).  Reordering the same
    dict flipped the answer.  Requiring agreement costs one short-circuiting
    O(N) scan on a path that is about to stack N rows anyway (and the hot
    ``DatasetContext`` path memoises it - see
    :func:`_collapse_to_primary_for_ctx`), and it preserves the single-embedder
    optimisation exactly, which is the only case it was written for.
    """
    if embedder_name is None or not medias:
        return embedder_name
    if embedder_name == _uniform_primary_embedder(medias):
        return None
    return embedder_name


def _collapse_to_primary_for_ctx(ctx: "DatasetContext", embedder_name: str | None) -> str | None:
    """:func:`_collapse_to_primary` over *ctx*'s medias, memoised per revision.

    The scan is O(N) in Python, and :func:`get_embedding_matrix` runs it under
    ``_state_lock`` *before* the cache check - i.e. on the hot path that would
    otherwise do no per-media work at all.  Memoising the uniform-primary
    answer on the context keeps that path O(1) after the first call.

    Keyed on ``ctx.media_revision``, exactly like the matrix cache beside it:
    a structural mutation of ``ctx.medias`` bumps the counter automatically,
    and an in-place rewrite of a media's ``embedder`` field travels with the
    vector rewrite that ``invalidate_embedding_matrix`` already signals.  Call
    with ``_state_lock`` held (the callers do).
    """
    if embedder_name is None or not ctx.medias:
        return embedder_name
    if ctx._uniform_primary_revision != ctx.media_revision:
        ctx._uniform_primary = _uniform_primary_embedder(ctx.medias)
        ctx._uniform_primary_revision = ctx.media_revision
    return None if embedder_name == ctx._uniform_primary else embedder_name


def _require_embedding(cid: int, media: dict[str, Any], embedder_name: str | None = None) -> Any:
    emb = media_embedding(media, embedder_name)
    if emb is None:
        suffix = f" for embedder {embedder_name!r}" if embedder_name else ""
        raise ValueError(
            f"media {cid!r} has no embedding{suffix} (embedding=None); "
            "scoring/sorting require every media to have a vector. "
            "This usually means an importer or re-embed step silently failed."
        )
    return emb


def _row_width(vec: Any) -> int | None:
    """Return *vec*'s width when it is a 1-D row, else ``None``.

    Cheap by design (one ``.shape`` read, no copy): this is the per-media test
    :func:`scoreable_snapshot` runs over a whole dataset, so it must not touch
    the vector's contents.  A ``None``, a scalar, or a 2-D block all answer
    ``None`` - none of them can be assigned into a matrix row.
    """
    if vec is None:
        return None
    shape = getattr(vec, "shape", None)
    if shape is None:
        try:
            shape = np.shape(vec)
        except Exception:
            return None
    return int(shape[0]) if len(shape) == 1 else None


def scoreable_snapshot(
    snap: dict[int, dict[str, Any]],
    embedder_name: str | None = None,
    *,
    region_rows: bool = False,
) -> tuple[dict[int, dict[str, Any]], list[int]]:
    """Split *snap* into the media that can be scored and the ids that cannot.

    Returns ``(scoreable, dropped_ids)``.  A media is scoreable when it carries
    a vector under *embedder_name* (its primary when unnamed) **and** that
    vector is a 1-D row whose width matches the first scoreable media's - the
    two conditions the matrix builder enforces by raising, via
    :func:`_require_embedding` and :func:`require_dim` respectively.

    Filtering *before* the build is what turns "one image failed to embed" from
    an aborted run into a skipped item.  The builders stay strict on purpose:
    on the app's own dataset the load pipeline's ``_drop_none_embeddings_stage``
    has already removed vector-less media, so a raise there is a real invariant
    break worth hearing about.  A *snapshot* handed to the scorer carries no
    such guarantee - the CLI scores importer output that was never run through
    that stage, and a multi-embedder dataset can legitimately hold media that
    one bound embedder failed on while another succeeded - so the scoring paths
    filter first and score what is left, mirroring the drop-and-log policy of
    ``_drop_none_embeddings_stage`` and
    :func:`~vtscore.detectors.converter_routing.route_and_embed`.

    With *region_rows* set the check is run against the snapshot's **patch-slot**
    embedder instead, matching what :func:`_build_region_arrays` reads for the
    image-level row of every media in a region-row matrix.

    Drops caused by *snap* mixing embedding spaces are logged once per call by
    :func:`_log_mixed_space_drops`; the drop itself is the policy, but a
    silently short haystack is what made issue #3650 invisible.
    """
    key = _patch_embedder_for_region_snap(snap) if region_rows else _collapse_to_primary(snap, embedder_name)
    scoreable: dict[int, dict[str, Any]] = {}
    dropped: list[int] = []
    width: int | None = None
    for cid in sorted(snap.keys()):
        media = snap[cid]
        row_width = _row_width(media_embedding(media, key))
        if row_width is None:
            dropped.append(cid)
            continue
        if width is None:
            width = row_width
        if row_width != width:
            dropped.append(cid)
            continue
        scoreable[cid] = media
    if not region_rows:
        _log_mixed_space_drops(snap, embedder_name, dropped)
    return scoreable, dropped


def _log_mixed_space_drops(
    snap: dict[int, dict[str, Any]],
    embedder_name: str | None,
    dropped: list[int],
) -> None:
    """Warn once when *dropped* media were left out because *snap* mixes spaces.

    Dropping is the right policy - a mixed-type dataset scored for one space
    *should* leave out the media that live in another, rather than abort a long
    run or (worse, pre-#3650) stack their vectors into the wrong matrix.  But a
    quietly short haystack is exactly what hid that bug: the callers that
    report skips (``vtscore.cli._emit_skipped_medias``) see a count, not a
    reason, and library-tier callers see nothing at all.  So the layer that
    knows *why* says so, once per call rather than once per media.

    Only space-mixing is worth a warning.  A dropped media whose own primary
    **is** *embedder_name* failed to embed (or carries an unusable vector) -
    that is the pre-existing per-item failure the drop policy was written for,
    already surfaced by the callers, and not this function's business.  A
    dropped media with no primary at all is vector-less for the same reason.
    """
    if not dropped or embedder_name is None:
        return
    others = {primary_embedder_name(snap[cid]) for cid in dropped}
    others -= {embedder_name, None}
    if not others:
        return
    logger.warning(
        "Dropped %d of %d media from the %r embedding matrix: this snapshot mixes embedding "
        "spaces, and those media are embedded in %s instead. They carry no %r vector, so they "
        "are excluded from scoring rather than scored against another space's head.",
        len(dropped),
        len(snap),
        embedder_name,
        ", ".join(sorted(repr(name) for name in others)),
        embedder_name,
    )


def _vector_label(cid: int, media: dict[str, Any], embedder_name: str | None) -> str:
    """Human-locatable name for one media's vector, for a mismatch message.

    Prefers the media's origin name / filename over the bare content id: the id
    is an internal integer the user has no way to look up, whereas the filename
    is the row they can go and find in the manifest that supplied it.
    """
    who = media.get("origin_name") or media.get("filename") or "?"
    suffix = f" under embedder {embedder_name!r}" if embedder_name else ""
    return f"media {cid} ({who}){suffix}"


def _stack_embeddings(
    sorted_ids: list[int],
    source: dict[int, dict[str, Any]],
    embedder_name: str | None,
) -> np.ndarray:
    """Build a contiguous ``(N, D)`` float32 matrix of *embedder_name*'s vectors.

    Rows follow *sorted_ids* order, pulling each media's vector via
    :func:`_require_embedding` (which routes through the dict-keyed accessor).
    Raises ``ValueError`` naming the first media that lacks a vector.

    Each row's width is checked against the first row's before it is assigned.
    Ingestion is supposed to make a mixed-width dataset impossible (see
    :mod:`vtscore.embedding.precomputed`), but a dataset can still acquire one
    by other routes - a pickle written before that validation existed, or a
    third-party importer that writes ``media["embeddings"]`` directly - and
    without the check numpy reports only

        could not broadcast input array from shape (768,) into shape (1152,)

    which names neither the media nor the embedder, on a request that has
    nothing to do with where the bad vector came from.  The check is one
    attribute read per media against an ``(N, D)`` allocation and a full copy,
    so it costs nothing measurable on the hot path.
    """
    first_emb = np.asarray(_require_embedding(sorted_ids[0], source[sorted_ids[0]], embedder_name), dtype=np.float32)
    dim = int(first_emb.shape[-1])
    matrix = np.empty((len(sorted_ids), dim), dtype=np.float32)
    for i, cid in enumerate(sorted_ids):
        vec = _require_embedding(cid, source[cid], embedder_name)
        require_dim(
            vec,
            dim,
            label=_vector_label(cid, source[cid], embedder_name),
            expected_source=f"matching {_vector_label(sorted_ids[0], source[sorted_ids[0]], embedder_name)}",
        )
        matrix[i] = vec
    return matrix


def _registered_pkl_path(dataset_id: str) -> str | None:
    """Return the on-disk pkl path registered for *dataset_id*, or ``None``.

    Datasets built purely in memory (tests, ephemeral browse contexts,
    positives-map previews) have no registry entry and get no sidecar - the
    mmap cache (S1, ``docs/plans/scalability.md``) is opportunistic only for
    datasets actually backed by a saved pickle file.
    """
    from vtscore.datasets.registry import get_dataset  # noqa: PLC0415

    try:
        entry = get_dataset(dataset_id)
    except Exception:
        return None
    return (entry or {}).get("pkl_path") or None


def _emb_sidecar_paths(pkl_path: str) -> tuple[Path, Path]:
    """Return ``(ids_path, matrix_path)`` sidecar paths for *pkl_path*.

    Both share the pkl's stem (``ds_<uuid>``) followed by a dot, so
    ``registry.unregister_dataset``'s stem-glob sweep deletes them alongside
    the pkl on both age-off expiry and manual delete - no separate cleanup
    bookkeeping needed.
    """
    p = Path(pkl_path)
    return p.parent / f"{p.stem}.embids.npy", p.parent / f"{p.stem}.embmat.npy"


def _try_load_matrix_sidecar(pkl_path: str, sorted_ids: list[int], probe_dim: int) -> np.ndarray | None:
    """Return the mmap'd primary embedding matrix for *pkl_path* if valid, else ``None``.

    Valid means: both sidecar files exist, the persisted id list matches
    *sorted_ids* exactly (order and content), and the persisted matrix's
    column count matches *probe_dim* (a live vector's dimension, guarding
    against a same-id-set re-embed to a different dimension - the drift the
    id-list check alone can't see). Any mismatch, missing file, or read error
    returns ``None``; the caller always has a safe, correct fallback: rebuild
    from live ``ctx.medias``.
    """
    ids_path, mat_path = _emb_sidecar_paths(pkl_path)
    if not (ids_path.is_file() and mat_path.is_file()):
        return None
    try:
        sidecar_ids = np.load(ids_path)
        if sidecar_ids.shape != (len(sorted_ids),) or not np.array_equal(
            sidecar_ids, np.asarray(sorted_ids, dtype=np.int64)
        ):
            return None
        matrix = np.load(mat_path, mmap_mode="r")
        if matrix.ndim != 2 or matrix.shape[0] != len(sorted_ids) or matrix.shape[1] != probe_dim:
            return None
        return matrix
    except Exception:
        logger.warning("Failed to load embedding-matrix sidecar for %s", pkl_path, exc_info=True)
        return None


def _atomic_save_npy(path: Path, arr: np.ndarray) -> None:
    """Write *arr* to *path* as ``.npy`` bytes via write-to-temp + atomic rename.

    Mirrors the tmp-file idiom used by ``vtscore.datasets.container.write_container``:
    a crash mid-write leaves an orphan ``.tmp`` file, never a truncated file at
    the real name, so a concurrent reader never observes a partial array.
    """
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            np.save(f, arr)
        os.replace(tmp_name, str(path))
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _maybe_persist_matrix_sidecar(pkl_path: str, sorted_ids: list[int], matrix: np.ndarray) -> None:
    """Best-effort write of the primary embedding matrix as a mmap-able sidecar.

    Lets a future cold load of the same pkl (a fresh process / DatasetContext)
    skip rebuilding the matrix from per-item embeddings - see S1,
    ``docs/plans/scalability.md``. A pure derived cache of data already
    durably persisted in the dataset pickle: always regenerable from
    ``ctx.medias``, deterministic, and swept alongside the pkl by
    ``registry.unregister_dataset``. Any failure (read-only filesystem, full
    disk, concurrent writer) is logged and swallowed - the sidecar is an
    optimization, never a dependency.

    Always writes (no "ids already match, skip" shortcut): the caller only
    reaches here on a genuine cache-miss rebuild, which happens either on the
    first-ever build for this context or after an explicit invalidation - and
    an in-place vector rewrite (re-embed/clip) leaves the id set unchanged
    while the *content* legitimately changed, so an ids-only check would skip
    a write that must happen and silently entrench a stale sidecar.
    """
    ids_path, mat_path = _emb_sidecar_paths(pkl_path)
    try:
        # Matrix first: if a crash lands between the two atomic renames, a
        # mismatched pair is caught by the shape/dim checks in
        # ``_try_load_matrix_sidecar`` on the next read, never served as-is.
        _atomic_save_npy(mat_path, matrix)
        _atomic_save_npy(ids_path, np.asarray(sorted_ids, dtype=np.int64))
    except OSError:
        logger.info("Could not persist embedding-matrix sidecar for %s (read-only filesystem?)", pkl_path)
    except Exception:
        logger.warning("Failed to persist embedding-matrix sidecar for %s", pkl_path, exc_info=True)


def _try_primary_sidecar(
    ctx: "DatasetContext", sorted_ids: list[int], medias_snapshot: dict[int, dict[str, Any]]
) -> tuple[str | None, np.ndarray | None]:
    """Return ``(pkl_path, matrix)`` for the primary-path on-disk mmap sidecar.

    *matrix* is ``None`` when there is no registered pkl, the sidecar-disabled
    latch is set (see ``invalidate_embedding_matrix``), or the sidecar is
    missing/stale - the caller always falls back to ``_stack_embeddings`` in
    that case. *pkl_path* is returned even on a sidecar miss (``None`` matrix)
    since the caller still needs it to persist a freshly-built matrix.
    """
    pkl_path = _registered_pkl_path(ctx.dataset_id)
    if not pkl_path or ctx._emb_sidecar_disabled:
        return pkl_path, None
    probe_dim = int(np.asarray(_require_embedding(sorted_ids[0], medias_snapshot[sorted_ids[0]])).shape[-1])
    return pkl_path, _try_load_matrix_sidecar(pkl_path, sorted_ids, probe_dim)


def get_embedding_matrix(ctx: "DatasetContext", embedder_name: str | None = None) -> tuple[list[int], np.ndarray]:
    """Return ``(sorted_ids, (N, D) float32 matrix)`` for *ctx*'s medias.

    With *embedder_name* unset the matrix is built from each media's *primary*
    embedder and cached on the context, rebuilt only when the set of media IDs
    changes.  Pass an explicit *embedder_name* (one of a multi-embedder
    dataset's bound slots) to build a matrix from that embedder's vectors
    instead; the named path builds fresh on every call and never touches the
    cache, since the cache is reserved for the hot primary path.  Convert to a
    tensor with ``torch.from_numpy(matrix)`` for a zero-copy view.

    Returns ``([], np.empty((0, 0), dtype=np.float32))`` when the dataset is
    empty.  Raises ``ValueError`` if any media lacks the requested vector.
    """
    # Phase 1 (locked): snapshot the media refs and serve a cache hit. The
    # expensive _stack_embeddings build runs OUTSIDE the lock (phase 2) so a
    # large primary or named-embedder build cannot hold _state_lock across the
    # numpy stack and stall every other request's before_request state-sync.
    with _state_lock:
        sorted_ids = sorted(ctx.medias.keys())
        # Snapshot the revision under the lock; the cache is valid iff it still
        # matches after the unlocked build (phase 3). Keying on the counter
        # rather than an id-list compare also catches an in-place vector
        # rewrite that leaves the id set unchanged (root-cause Pattern #4).
        revision = ctx.media_revision
        # A routed name equal to *every* media's primary collapses to the
        # cached path (memoised per media_revision: this runs before the cache
        # check, so the hot path must not pay an O(N) scan per call).
        embedder_name = _collapse_to_primary_for_ctx(ctx, embedder_name)
        if embedder_name is None:
            cached_matrix = ctx._emb_matrix
            # A revision match guarantees the id set is unchanged, so the live
            # sorted_ids equals the cached ids the matrix rows correspond to.
            if cached_matrix is not None and ctx._emb_matrix_revision == revision:
                return list(sorted_ids), cached_matrix

        if not sorted_ids:
            if embedder_name is None:
                ctx._emb_matrix_ids = []
                ctx._emb_matrix = np.empty((0, 0), dtype=np.float32)
                ctx._emb_matrix_revision = revision
                return [], ctx._emb_matrix
            return [], np.empty((0, 0), dtype=np.float32)

        # Shallow ref-copy so the build below reads a stable view even if
        # ctx.medias is reassigned concurrently (cheap: pointers, not vectors).
        medias_snapshot = dict(ctx.medias)

    # Phase 2 (unlocked): try a matching on-disk mmap sidecar first (S1,
    # docs/plans/scalability.md), else the heavy contiguous (N, D) build.
    # ``pkl_path`` is resolved even on a sidecar miss (used again in phase 4
    # to persist a fresh build).
    pkl_path: str | None = None
    matrix: np.ndarray | None = None
    if embedder_name is None:
        pkl_path, matrix = _try_primary_sidecar(ctx, sorted_ids, medias_snapshot)
    used_sidecar = matrix is not None
    if matrix is None:
        matrix = _stack_embeddings(sorted_ids, medias_snapshot, embedder_name)

    # Phase 3 (locked): repopulate the primary cache, double-checking the
    # revision still matches so a media mutation during the unlocked build
    # cannot cache a stale matrix. The named path never touches the cache.
    cache_populated = False
    if embedder_name is None:
        with _state_lock:
            if ctx.media_revision == revision:
                ctx._emb_matrix_ids = sorted_ids
                ctx._emb_matrix = matrix
                ctx._emb_matrix_revision = revision
                cache_populated = True

    # Phase 4 (unlocked, best-effort): persist a freshly-built primary matrix
    # as a sidecar so a future cold load of this pkl can mmap it instead of
    # rebuilding. Skipped when we just read a valid sidecar (already on disk,
    # nothing to refresh) or when phase 3 lost the race (that matrix no
    # longer matches the live id set and must not be written as this pkl's
    # cache).
    if cache_populated and not used_sidecar and pkl_path:
        _maybe_persist_matrix_sidecar(pkl_path, sorted_ids, matrix)

    return list(sorted_ids), matrix


def get_embedding_submatrix(
    ctx: "DatasetContext", ids: list[int], embedder_name: str | None = None
) -> tuple[list[int], np.ndarray]:
    """Return ``(sorted_ids, (N, D) float32 matrix)`` for a *subset* of *ctx*'s medias.

    Unlike :func:`get_embedding_matrix`, this builds a fresh matrix over only
    the requested *ids* (intersected with the dataset's current medias) and
    never populates the context-wide cache - subset projections (e.g. the
    positives of a Find run) are ephemeral.  The returned id list is sorted and
    de-duplicated; ids absent from the dataset are dropped silently.  Pass
    *embedder_name* to source the rows from a specific bound embedder.

    Returns ``([], np.empty((0, 0), dtype=np.float32))`` when nothing matches.
    Raises ``ValueError`` if any requested media lacks the requested vector.
    """
    # Snapshot just the requested rows under the lock, then build the matrix
    # outside it so a large subset stack does not hold _state_lock across the
    # numpy build (see get_embedding_matrix for the rationale).
    with _state_lock:
        medias = ctx.medias
        sorted_ids = sorted({cid for cid in ids if cid in medias})
        if not sorted_ids:
            return [], np.empty((0, 0), dtype=np.float32)
        medias_snapshot = {cid: medias[cid] for cid in sorted_ids}

    return sorted_ids, _stack_embeddings(sorted_ids, medias_snapshot, embedder_name)


def invalidate_embedding_matrix(ctx: "DatasetContext") -> None:
    """Drop the cached matrices on *ctx*; next access rebuilds them.

    Clears both the per-media embedding matrix and the flattened
    per-score-row matrix (used by patch scoring), since both are keyed
    on ``media_revision`` and become stale together when the dataset's media
    change.  Also bumps ``media_revision`` so this stands in as the explicit
    "vectors changed in place" signal at the embed / clip stages: an
    in-place rewrite of existing media dicts is invisible to
    :class:`~vtscore.state.core.MediasDict`, so those stages call this to
    advance the counter (and free the cached arrays' RAM immediately).
    """
    with _state_lock:
        # Only the matrix family: the lookups and the browse layouts are
        # revision-keyed too and invalidate themselves off the bump below,
        # so dropping them here would throw away work for nothing.
        ctx.reset_derived_caches(matrices=True, lookups=False, projection=False, subset=False)
        # An in-place rewrite can leave the id set (and dimension) unchanged,
        # which the sidecar's validity check alone cannot detect - permanently
        # stop trusting the on-disk mmap sidecar for this context so every
        # later rebuild reads live ``ctx.medias``, never a stale cached file.
        # A latch, not a cache, which is why ``reset_derived_caches`` leaves
        # it alone and this line stays here.
        ctx._emb_sidecar_disabled = True
        ctx.bump_media_revision()


def _patch_embedder_for_region_snap(snap: dict[int, dict[str, Any]]) -> str | None:
    """Return the patch-slot embedder name that produced *snap*'s patch grids.

    Derived from a media that actually carries a ``patch_grid`` (rather than
    just the first media in the dict, which may be a grid-less item that
    never had a vector for the patch embedder at all - the mixed-media-type
    case).  That media's own bound embedder names are role-typed via
    :func:`~vtscore.embedding.binding.derive_binding_from_names`; the patch
    slot is the space the patch vectors live in.  Returns ``None`` when no
    media in *snap* has a grid, or when the grid-bearing media's embedders
    don't role-type to a patch slot (unexpected; the fallback path then keeps
    the pre-fix behaviour of reading the primary vector).
    """
    from vtscore.embedding.binding import derive_binding_from_names  # noqa: PLC0415

    for media in snap.values():
        if media.get("patch_grid") is not None:
            _text, patch, _structural = derive_binding_from_names(media_embedder_names(media))
            return patch
    return None


def media_score_rows(
    media: dict[str, Any],
    embedder_name: str | None = None,
    *,
    dtype: Any = np.float32,
) -> np.ndarray | None:
    """The row stack *media* is max-pooled over at inference, or ``None``.

    **The single definition of MaxPatch scoring geometry.**  Every path that
    needs "the candidate vectors of this image" goes through here - the
    flattened scoring matrix (:func:`_build_region_arrays`), a Bad vote's
    negative flood (:func:`vtscore.detectors.training.bad_negative_vecs`), and
    the per-bag stacks threshold calibration collapses
    (:func:`vtscore.detectors.training.inference_score_rows`) - so the
    train/score invariant holds by construction rather than by three
    implementations agreeing:

        Every vector a vote can train on must also be a row that is scored.

    Layout, for a patch media carrying an ``(H, W, D)`` ``patch_grid``:

    * **row 0** - the image-level (CLS) vector, box ``(0, 0, 1, 1)``.  It is
      load-bearing, not decoration: a *boxless* Good vote trains on the
      image-level vector (:func:`~vtscore.detectors.training._training_vec_for_vote`),
      so without this row that vote would train on a vector nothing ever
      scores.  Dropping it produced perfect ranking with catastrophic FNR when
      the Max-Patch style was first prototyped - see
      :mod:`vtscore.eval.patch_styles`.
    * **rows 1 .. H*W** - every raw patch vector, row-major
      (``1 + r*W + c`` is grid cell ``(r, c)``).  :func:`patch_row_box` is the
      inverse map back to a box.

    A media with no ``patch_grid`` contributes its single image-level vector,
    so a legacy single-vector dataset is one row per media exactly as before.

    *embedder_name* selects which bound embedder supplies the image-level row;
    on a patch dataset that must be the **patch-slot** embedder (the space the
    grid lives in), which callers resolve once per snapshot via
    :func:`_patch_embedder_for_region_snap`.  Returns ``None`` only when the
    media has neither a grid nor a resolvable vector.
    """
    grid = media.get("patch_grid")
    emb = media_embedding(media, embedder_name)
    if grid is None:
        if emb is None:
            return None
        return np.asarray(emb, dtype=dtype).reshape(1, -1)
    arr = np.asarray(grid, dtype=dtype)
    flat = arr.reshape(-1, arr.shape[-1])
    if emb is None:
        return flat
    cls_row = np.asarray(emb, dtype=dtype).reshape(1, -1)
    if cls_row.shape[1] != flat.shape[1]:
        # The image-level vector and the patch grid must live in the same space
        # for a max-pool over them to mean anything.  np.concatenate would report
        # only an axis-1 dimension mismatch, so say which two things disagree.
        raise MismatchedVectorError(
            f"media {media.get('id', '?')} ({media.get('origin_name') or media.get('filename') or '?'}): "
            f"its image-level vector is {cls_row.shape[1]}-dimensional but its patch grid is "
            f"{flat.shape[1]}-dimensional. MaxPatch scoring max-pools the two together, so they must come "
            f"from the same embedder"
            + (f" (expected {embedder_name!r})" if embedder_name else "")
            + ". Re-embed this dataset so the image-level and patch vectors are produced by one model."
        )
    return np.concatenate([cls_row, flat], axis=0)


def media_row_box(media: dict[str, Any], row_index: int) -> list[float] | None:
    """The box of *media*'s score row *row_index*, or ``None`` for a grid-less media.

    Turns the winner of a segmented max-pool back into the rectangle the
    gallery / image viewer outlines.  ``None`` (no overlay) for a media with no
    ``patch_grid``: its only row is the whole image, which the viewer suppresses
    anyway, and a single-vector dataset never showed a best-region box.
    """
    from vtscore.media.patch_embed import patch_row_box  # noqa: PLC0415

    grid = media.get("patch_grid")
    if grid is None:
        return None
    arr = np.asarray(grid)
    if arr.ndim != 3:
        return None
    height, width = int(arr.shape[0]), int(arr.shape[1])
    if not 0 <= row_index <= height * width:
        return None
    return list(patch_row_box(int(row_index), height, width))


def _build_region_arrays(
    snap: dict,
    sorted_ids: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten every media's :func:`media_score_rows` stack into one ``(R, D)`` matrix.

    Returns ``(region_matrix, media_index_per_row, region_index_per_row)``:

    * ``region_matrix`` - ``(R, D)`` **float16**, one row per (media, row) pair.
    * ``media_index_per_row`` - ``int64 (R,)``, the index into *sorted_ids*
      that each row belongs to.  Non-decreasing and contiguous per media.
    * ``region_index_per_row`` - ``int64 (R,)``, the row's index within its
      media's :func:`media_score_rows` stack: ``0`` = whole image, ``1..H*W`` =
      raw patch cells row-major.  The winning value surfaces as the UI's
      best-match overlay via :func:`media_row_box`.

    Media that expose no ``patch_grid`` contribute a single row (index 0) so
    every media has at least one row - keeping the downstream segmented
    max-pool free of empty groups.  That fallback row is read from the
    *patch-slot* embedder shared by the rest of the snapshot's patch rows
    (:func:`_patch_embedder_for_region_snap`), not unconditionally the primary
    vector: on a dataset that mixes patch-capable and patch-less media (e.g. a
    combined dataset, or a media type the patch embedder can't process), the
    primary can be a different embedder than the one that produced the patch
    vectors, and stacking its vector alongside them would silently mix
    embedding spaces in one matrix.  If the grid-less media has no vector under
    that patch embedder either, :func:`_require_embedding` raises rather than
    falling back further - a loud, locatable failure instead of a silently
    meaningless score.

    **Dtype is float16, not float32.**  MaxPatch scores ~197 rows per image
    where the old HAC tree scored ~24, so a float32 matrix would be ~8x the
    bytes of the tree's.  The grid is already stored float16, so keeping the
    flattened stack in that dtype holds the blow-up to the row count alone;
    consumers upcast chunk-wise (:func:`chunked_row_scores`,
    :func:`vtscore.detectors.training._forward_sigmoid_chunked`) so peak
    float32 memory stays bounded regardless of dataset size.

    The build is two-pass - count rows from the grid *shapes*, allocate once,
    then fill - rather than stacking per-media blocks and concatenating.  A
    concatenate holds the blocks and the result alive at the same moment, i.e.
    2x the matrix at peak, which at MaxPatch's row count is gigabytes on a
    large collection.  The per-media temporary is one image's rows (~300 KB).
    """
    patch_embedder_name = _patch_embedder_for_region_snap(snap)

    # Pass 1: row counts, read off the grid shape without materialising rows.
    # ``_require_embedding`` here is what makes the count trustworthy - it
    # guarantees every media contributes its image-level row 0, so a grid-bearing
    # media is exactly ``1 + H*W`` rows.  A media with no resolvable vector
    # raises loudly, naming the cid, instead of silently short-stacking.
    counts = np.empty(len(sorted_ids), dtype=np.int64)
    for mi, cid in enumerate(sorted_ids):
        media = snap[cid]
        _require_embedding(cid, media, patch_embedder_name)
        grid = media.get("patch_grid")
        if grid is None:
            counts[mi] = 1
        else:
            shape = np.asarray(grid).shape
            counts[mi] = 1 + int(shape[0]) * int(shape[1])
    starts = np.zeros(len(sorted_ids), dtype=np.int64)
    np.cumsum(counts[:-1], out=starts[1:])
    total = int(counts.sum())

    # Pass 2: allocate once and fill each media's slice in place.  Each media's
    # rows are width-checked against the first media's before assignment, for
    # the same reason as in ``_stack_embeddings``: the raw failure here is a
    # bare numpy broadcast error naming neither the media nor the embedder.
    first = media_score_rows(snap[sorted_ids[0]], patch_embedder_name, dtype=np.float16)
    assert first is not None  # guaranteed by the _require_embedding pass above
    dim = int(first.shape[1])
    region_matrix = np.empty((total, dim), dtype=np.float16)
    region_matrix[: first.shape[0]] = first
    for mi, cid in enumerate(sorted_ids[1:], start=1):
        rows = media_score_rows(snap[cid], patch_embedder_name, dtype=np.float16)
        assert rows is not None
        if int(rows.shape[1]) != dim:
            raise MismatchedVectorError(
                f"{_vector_label(cid, snap[cid], patch_embedder_name)}: contributes "
                f"{int(rows.shape[1])}-dimensional score rows, but "
                f"{_vector_label(sorted_ids[0], snap[sorted_ids[0]], patch_embedder_name)} is {dim}-dimensional. "
                "Every media in a dataset must be scored in one embedding space; this dataset mixes two. "
                "Re-import the odd media with vectors from the same embedder, or rebuild the dataset."
            )
        start = int(starts[mi])
        region_matrix[start : start + rows.shape[0]] = rows

    media_index_per_row = np.repeat(np.arange(len(sorted_ids), dtype=np.int64), counts)
    region_index_per_row = np.arange(total, dtype=np.int64) - np.repeat(starts, counts)
    return region_matrix, media_index_per_row, region_index_per_row


def get_region_matrix_for_snap(
    snap: dict,
) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray]:
    """Return the cached flattened score-row matrix for *snap*.

    Returns ``(sorted_ids, region_matrix, media_index_per_row,
    region_index_per_row)`` - see :func:`_build_region_arrays` for the
    array shapes and dtype.  When *snap*'s key set matches the active
    :class:`DatasetContext`'s medias (the common per-vote case), the matrix
    is built once and cached on the context, then reused across subsequent
    votes; only the MLP weights change between votes, never the patch
    vectors, so the cache is valid until the media-id set changes.  A
    cross-dataset / subset *snap* builds fresh without populating the cache.

    Returns empty arrays when *snap* is empty.  Raises ``ValueError`` if a
    grid-less media has ``embedding=None``.
    """
    from vtscore.state.core import get_active_context

    sorted_ids = sorted(snap.keys())
    if not sorted_ids:
        empty_vecs = np.empty((0, 0), dtype=np.float32)
        empty_idx = np.empty((0,), dtype=np.int64)
        return [], empty_vecs, empty_idx, empty_idx

    ctx = get_active_context()
    with _state_lock:
        # Snapshot the revision so a mutation during the unlocked build below
        # can't cache a stale matrix. The id-list compare still guards against
        # a *different* snap with a coincidentally equal cached id set; the
        # revision compare additionally catches an in-place vector rewrite
        # under the same id set (root-cause Pattern #4).
        revision = ctx.media_revision
        if (
            ctx._region_matrix is not None
            and ctx._region_matrix_ids == sorted_ids
            and ctx._region_matrix_revision == revision
            and ctx._region_media_index is not None
            and ctx._region_index_per_row is not None
        ):
            return (
                list(sorted_ids),
                ctx._region_matrix,
                ctx._region_media_index,
                ctx._region_index_per_row,
            )

    region_matrix, media_index, region_index = _build_region_arrays(snap, sorted_ids)

    # Populate the cache only when *snap* matches the active dataset's medias
    # (the common case: ``snap = snapshot_medias()``) and no mutation landed
    # during the build.  Subset / cross-dataset dicts are ephemeral and must
    # not clobber the active cache.
    if sorted_ids == sorted(ctx.medias.keys()):
        with _state_lock:
            if ctx.media_revision == revision:
                ctx._region_matrix_ids = sorted_ids
                ctx._region_matrix = region_matrix
                ctx._region_matrix_revision = revision
                ctx._region_media_index = media_index
                ctx._region_index_per_row = region_index
    return list(sorted_ids), region_matrix, media_index, region_index


#: Rows per chunk when upcasting a float16 score matrix for numpy/torch math.
#: Bounds peak float32 memory at ~``ROW_CHUNK * D * 4`` bytes regardless of how
#: many rows the dataset has.
ROW_CHUNK = 65_536


def chunked_row_scores(matrix: np.ndarray, query_vec: np.ndarray) -> np.ndarray:
    """``matrix @ query_vec`` as float64, upcasting a float16 matrix chunk-wise.

    The region matrix is stored float16 (see :func:`_build_region_arrays`).
    numpy's float16 BLAS path is both slow and lossy, and a whole-matrix
    ``.astype(np.float32)`` would allocate a second copy twice the size of the
    matrix itself - on a large patch dataset that is gigabytes.  Chunking keeps
    the upcast bounded while producing exactly the float32 dot product.
    """
    out = np.empty(matrix.shape[0], dtype=np.float64)
    q = np.asarray(query_vec, dtype=np.float32)
    for start in range(0, matrix.shape[0], ROW_CHUNK):
        chunk = np.asarray(matrix[start : start + ROW_CHUNK], dtype=np.float32)
        out[start : start + chunk.shape[0]] = chunk @ q
    return out


def segmented_max_pool(
    flat_scores: np.ndarray,
    media_index_per_row: np.ndarray,
    region_index_per_row: np.ndarray,
    n_media: int,
) -> tuple[list[float], list[int]]:
    """Max-pool per-row scores down to one score + winning region per media.

    Shared by the MLP scoring path (:func:`vtscore.detectors.training._score_all_media`)
    and the region-aware cosine sort (:func:`vtscore.training.region_similarity.cosine_sort_with_boxes`),
    both of which flatten every ``(media, region)`` pair into one ``(R,)`` score
    vector (via :func:`get_region_matrix_for_snap`) and need to reduce it back to
    one score + winning region index per media.

    *media_index_per_row* is non-decreasing and contiguous (every media owns
    a single run of rows), and every media has at least one row, so each
    media's rows form one ``reduceat`` segment.  Returns ``(scores,
    best_region)`` as plain Python lists, where ``best_region[m]`` is the
    region index of the *first* row achieving media ``m``'s max - matching
    the strict-``>`` "first wins" tie-break of the original scalar loop.

    Fully vectorised so the scoring tail holds the GIL for microseconds
    rather than iterating hundreds of thousands of rows in Python.
    """
    # Start of each media's contiguous run of rows.
    seg_starts = np.searchsorted(media_index_per_row, np.arange(n_media))
    seg_max = np.maximum.reduceat(flat_scores, seg_starts)

    # First row per media that reaches its segment max (region 0 - the
    # CLS/full-image node - is always row 0 of a segment, so an all-sentinel
    # media resolves to region 0, exactly as the old -1.0-seeded loop did).
    is_max = flat_scores >= seg_max[media_index_per_row]
    cand_rows = np.flatnonzero(is_max)
    cand_media = media_index_per_row[cand_rows]
    first_cand = np.searchsorted(cand_media, np.arange(n_media))
    winning_rows = cand_rows[first_cand]
    best_region = region_index_per_row[winning_rows]

    return seg_max.tolist(), best_region.tolist()


def get_embedding_matrix_for_snap(
    snap: dict,
    embedder_name: str | None = None,
) -> tuple[list[int], np.ndarray]:
    """Return ``(sorted_ids, matrix)`` for *snap*.

    With *embedder_name* unset and *snap*'s key set matching the active
    :class:`DatasetContext`'s medias, the cached primary matrix is reused.
    Otherwise (a different snapshot, a temp dict from cross-dataset Find, or an
    explicit non-primary embedder) the matrix is built fresh without populating
    the cache.  Raises ``ValueError`` if any entry in *snap* lacks the
    requested vector.
    """
    from vtscore.state.core import get_active_context

    sorted_ids = sorted(snap.keys())
    if not sorted_ids:
        return [], np.empty((0, 0), dtype=np.float32)

    # A routed name equal to *every* media's primary collapses to the cached
    # primary path (matching get_embedding_matrix), so single-embedder
    # snapshots keep reusing the context cache after routing names through.
    # A snapshot whose medias disagree keeps the name and takes the named
    # path, which raises on a media lacking that vector rather than stacking
    # another space's vector into this matrix (issue #3650); callers that
    # would rather drop than raise pre-filter with scoreable_snapshot.
    embedder_name = _collapse_to_primary(snap, embedder_name)

    ctx = get_active_context()
    matches_active = sorted_ids == sorted(ctx.medias.keys())

    if embedder_name is None:
        with _state_lock:
            cached_ids = ctx._emb_matrix_ids
            cached_matrix = ctx._emb_matrix
            cache_current = ctx._emb_matrix_revision == ctx.media_revision
        # The id-list compare guards against a *different* snap whose ids
        # happen to equal the cached set; the revision compare additionally
        # rejects a cache stale from an in-place vector rewrite under the same
        # id set (root-cause Pattern #4).
        if cached_matrix is not None and cached_ids == sorted_ids and cache_current:
            return sorted_ids, cached_matrix

    # When *snap* matches the active dataset's medias (the common case:
    # `snap = snapshot_medias()`), delegate to the context builder so the
    # primary path populates / reuses the cache; the named path builds fresh
    # there too.
    if matches_active:
        return get_embedding_matrix(ctx, embedder_name)

    # Temp dict / cross-dataset case: build fresh, don't cache.
    return sorted_ids, _stack_embeddings(sorted_ids, snap, embedder_name)
