"""Train and score using a detector's full labelset, not just current votes.

The detector's saved labelset on disk is origin-keyed and dataset-agnostic.
At load time we resolve every element to a file via its origin importer,
embed it, and cache the resulting vector on
:attr:`~vtscore.state.core.DetectorContext.label_embeddings`.  MLP
training (load-time and during interactive learned-sort) then iterates the
labelset directly, so labels from datasets that aren't currently loaded
still contribute.

This module is the single place that knows how to (re-)build the
``label_embeddings`` cache, build ``(X_list, y_list)`` from it, and run
:func:`~vtscore.detectors.training.train_and_threshold`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, NamedTuple

import numpy as np

from vtscore.datasets.labelset import LabeledElement, LabelSet
from vtscore.embedding.binding import keying_embedder_for_snap
from vtscore.embedding.media_vectors import media_embedding


log = logging.getLogger(__name__)

ProgressCallback = Callable[[str, int, int], None]


def _lookups_for_snap(
    snap: dict[int, dict[str, Any]] | None,
) -> tuple[dict[str, list[int]], dict[str, list[int]], dict[str, list[int]]] | None:
    """Build the ``(origin, md5, name)`` media-lookup triple for *snap*, once.

    Returns ``None`` when *snap* is empty/absent (the in-dataset resolution
    paths are all guarded on ``if snap:`` anyway).  Loop callers build this a
    single time and thread it into :func:`resolve_current_dataset_cid` so the
    per-element cid resolution is O(1) instead of rebuilding the tables (which
    ``json.dumps`` every origin) for each of ``labels`` elements.
    """
    if not snap:
        return None
    from vtscore.state import build_media_lookup

    return build_media_lookup(snap)


def _patch_output_from_file(
    file_path: Path,
    *,
    media_type: str,
    embedder_name: str,
):
    """Re-derive a media's :class:`PatchEmbedOutput` from its file, or ``None``.

    Resolves the embedder (the detector's, else the media type's default) and
    runs :meth:`MediaEmbedder.patch_forward`.  Returns ``None`` when the
    embedder doesn't support patch regions or the forward pass produces no
    output.  Shared by the Good-vote nearest-patch path and the Bad-vote flood
    path so both see the identical grid cross-dataset.
    """
    from vtscore.media import embedders_for_type, get_embedder
    from vtscore.media.embedder import media_from_path

    embedder = None
    if embedder_name:
        try:
            embedder = get_embedder(embedder_name)
        except (KeyError, ValueError):
            embedder = None
    if embedder is None:
        avail = embedders_for_type(media_type)
        if not avail:
            return None
        embedder = avail[0]

    if not getattr(embedder, "supports_patch_regions", False):
        return None

    try:
        output = embedder.patch_forward(media_from_path(file_path))
    except Exception:
        log.warning(
            "labelset_training: patch_forward(%s) raised; region vote will fall back to image-level embedding",
            file_path,
            exc_info=True,
        )
        return None
    return output


def _embedder_supports_patch_regions(embedder_name: str) -> bool:
    """Whether *embedder_name* produces a patch grid at all.

    This is the labelset tier's patch gate - the counterpart of
    :func:`~vtscore.detectors.training._scores_in_patch_space`, phrased against
    the detector's own embedder because the cross-dataset branch has no snap to
    resolve a patch slot from.  It answers "does this detector score in a patch
    space?", and *every* patch behaviour on this path hangs off it: the raw-patch
    pool under a Good element's ``region_box``, a Bad element's flooded
    negatives, and the calibration score-row stacks.

    Two things it buys:

    * On a legacy single-vector detector there is no grid to re-derive, so
      probing every element's origin file would be pure I/O for a guaranteed
      ``None``.
    * On a **multi-embedder** dataset it keeps a detector locked to the text or
      structural space off the patch grid entirely.  That detector is scored
      against its own space's full-image vectors
      (:func:`~vtscore.detectors.training._score_all_media`), so reading the
      media's stored ``patch_grid`` - which lives in the *patch* embedder's
      space - would train it on vectors from an unrelated space.
    """
    if not embedder_name:
        return False
    from vtscore.media import get_embedder

    try:
        embedder = get_embedder(embedder_name)
    except (KeyError, ValueError):
        return False
    return bool(getattr(embedder, "supports_patch_regions", False))


def _patch_pooled_from_file(
    file_path: Path,
    *,
    media_type: str,
    embedder_name: str,
    region_box: tuple[float, float, float, float],
) -> np.ndarray | None:
    """Run ``patch_forward`` on *file_path* and take the patch nearest *region_box*.

    Mirrors the in-dataset path (:func:`~vtscore.detectors.training.pool_box_from_media`
    → :func:`nearest_patch_to_box`) for the cross-dataset case: re-derive the
    patch grid via :func:`_patch_output_from_file` and pick the raw patch under
    the user-drawn box - so a Good region-vote trains on one of the very rows
    the MLP will max-pool over on the new dataset.  Returns ``None`` when the
    grid can't be rebuilt, so the caller can fall back to an image-level
    embedding.
    """
    from vtscore.media.patch_embed import nearest_patch_to_box

    output = _patch_output_from_file(file_path, media_type=media_type, embedder_name=embedder_name)
    if output is None:
        return None
    return nearest_patch_to_box(np.asarray(output.patch_grid), region_box)


def _embed_one(elem: LabeledElement, *, media_type: str, embedder_name: str) -> np.ndarray | None:
    """Resolve *elem*'s origin file and embed it.  Returns ``None`` on failure.

    When *elem* carries a ``region_box`` and the active embedder supports
    patch regions, the resolved file is patch-forwarded and the raw patch under
    the box is taken via :func:`nearest_patch_to_box` so the
    user's region-level training intent survives a dataset switch.  Logs a
    warning and falls
    back to a full-file embedding when the patch path is unavailable -
    legacy single-vector embedders, an origin carrying a clipper we'd
    have to replay against an unknown patch grid, or a failed forward
    pass.
    """
    from vtscore.detectors.resolver import (
        _apply_clip_and_embed,
        embed_file,
        resolve_file_context,
    )

    with resolve_file_context(elem.origin, elem.origin_name, elem.filename) as file_path:
        if file_path is None:
            return None

        origin = elem.origin or {}
        params = origin.get("params", {}) if isinstance(origin, dict) else {}
        has_clipper = isinstance(params, dict) and bool(params.get("clipper"))

        if elem.region_box is not None and not has_clipper:
            pooled = _patch_pooled_from_file(
                file_path,
                media_type=media_type,
                embedder_name=embedder_name,
                region_box=elem.region_box,
            )
            if pooled is not None:
                return pooled
            log.warning(
                "labelset_training: region_box on %r cannot be honored cross-dataset "
                "(embedder=%r does not support patch regions or patch_forward "
                "produced no output); falling back to image-level embedding",
                elem.origin_name or elem.filename or "<unknown>",
                embedder_name or "<default>",
            )
        elif elem.region_box is not None and has_clipper:
            log.warning(
                "labelset_training: region_box on %r cannot be honored cross-dataset "
                "because the origin carries a clipper; falling back to image-level "
                "embedding",
                elem.origin_name or elem.filename or "<unknown>",
            )

        if has_clipper:
            result = _apply_clip_and_embed(file_path, media_type, origin, embedder_name)
            if result is None:
                return None
            embedding, _clip_bytes = result
            return embedding
        return embed_file(file_path, media_type, embedder_name)


def _maybe_clear_cache_on_embedder_switch(det_ctx, embedder_name: str) -> None:
    """Drop the label-embedding cache when the active dataset's embedder changed.

    Mixing vectors from two embedders into one MLP produces garbage.  When
    ``det_ctx.embedder`` is empty (fresh load or legacy state) we keep the
    cache; otherwise a mismatch with the active embedder forces a rebuild.
    """
    if det_ctx.embedder and embedder_name and det_ctx.embedder != embedder_name:
        det_ctx.label_embeddings.clear()
        det_ctx.label_embedding_regions.clear()
        # Flooded patch negatives and the region scoring stacks live in the old
        # embedder's space too.
        det_ctx.label_negative_regions.clear()
        det_ctx.label_score_regions.clear()
        # Local features are descriptor sets in the old embedder's feature space;
        # a switch invalidates them too (a SIFT template can't verify against a
        # learned-feature candidate, nor vice versa).
        det_ctx.label_local_features.clear()


def _resolve_uncached_embedding(
    elem: LabeledElement,
    snap: dict[int, dict[str, Any]] | None,
    *,
    media_type: str,
    embedder_name: str,
    patch_capable: bool,
    lookups: tuple[dict[str, list[int]], dict[str, list[int]], dict[str, list[int]]] | None = None,
) -> np.ndarray | None:
    """Produce a training vector for *elem*, not consulting the cache.

    Tries the in-dataset path first: when *elem* resolves to a cid in the
    active *snap*, reuse the stored embedding (taking the raw patch under the
    ``region_box`` when the element carries one).
    Falls back to the cross-dataset path - resolve via the importer and embed
    freshly.  Returns ``None`` when neither path produces a vector.

    *patch_capable* (:func:`_embedder_supports_patch_regions`) gates the
    raw-patch pool: a detector that doesn't score in a patch space trains on the
    image-level vector of its own space even when the media carries a grid,
    because the grid's vectors belong to the dataset's *patch* embedder.  The
    cross-dataset branch is already gated the same way inside
    :func:`_patch_output_from_file`.

    *lookups* is the pre-built ``build_media_lookup`` triple for *snap*; loop
    callers pass it so the cid resolution doesn't rebuild the tables per element.
    """
    from vtscore.detectors.labelset_elements import resolve_current_dataset_cid
    from vtscore.detectors.training import pool_box_from_media

    if snap:
        cid = resolve_current_dataset_cid(elem, lookups)
        if cid is not None and cid in snap:
            media = snap[cid]
            pooled = pool_box_from_media(media, elem.region_box) if patch_capable else None
            # Read the in-dataset vector from the detector's primary space (the
            # same space the cross-dataset path embeds into), not the media's
            # generic primary - they diverge on a multi-embedder dataset.
            emb = pooled if pooled is not None else media_embedding(media, embedder_name or None)
            if emb is not None:
                return np.asarray(emb)

    # Cross-dataset path: ``_embed_one`` re-derives the patch grid on the
    # resolved file when ``elem.region_box`` is set and the embedder
    # supports patch regions, then takes the nearest raw patch so
    # region votes survive a dataset switch.  When the patch path isn't
    # available (legacy single-vector embedder, clipper-bearing origin,
    # failed forward pass) it logs a warning and returns the image-level
    # embedding - the only signal we have left to offer training.
    emb = _embed_one(elem, media_type=media_type, embedder_name=embedder_name)
    return np.asarray(emb) if emb is not None else None


def _resolve_score_rows(
    elem: LabeledElement,
    snap: dict[int, dict[str, Any]] | None,
    *,
    media_type: str,
    embedder_name: str,
    lookups: tuple[dict[str, list[int]], dict[str, list[int]], dict[str, list[int]]] | None = None,
) -> list[np.ndarray] | None:
    """*elem*'s MaxPatch score-row stack (image-level vector + every raw patch), or ``None``.

    In-dataset: read the resolved media's stored ``patch_grid`` (no I/O), via
    the same :func:`~vtscore.embedding.matrix.media_score_rows` the scorer
    uses.  Cross-dataset: re-derive the grid from the origin file via
    :func:`_patch_output_from_file`.  Returns ``None`` for non-patch
    datasets/elements (and clipper-bearing origins, whose patch grid we can't
    reconstruct) so the callers fall back to their image-level behaviour - a
    single negative for a Bad element, the training row as the scoring row.

    One resolution answers both questions the caller has, because under
    MaxPatch the rows a Bad vote floods and the rows the scorer max-pools are
    *the same rows*.  (Under the old HAC tree they differed: the flood skipped
    the internal merge nodes - see
    :func:`~vtscore.detectors.training.bad_negative_vecs`.)

    *lookups* is the pre-built ``build_media_lookup`` triple for *snap*; loop
    callers pass it so the cid resolution doesn't rebuild the tables per element.
    """
    from vtscore.detectors.labelset_elements import resolve_current_dataset_cid
    from vtscore.detectors.resolver import resolve_file_context
    from vtscore.embedding.matrix import media_score_rows

    if snap:
        cid = resolve_current_dataset_cid(elem, lookups)
        if cid is not None and cid in snap:
            media = snap[cid]
            if media.get("patch_grid") is None:
                return None
            rows = media_score_rows(media, embedder_name or None)
            return list(rows) if rows is not None else None

    origin = elem.origin or {}
    params = origin.get("params", {}) if isinstance(origin, dict) else {}
    if isinstance(params, dict) and params.get("clipper"):
        return None

    with resolve_file_context(elem.origin, elem.origin_name, elem.filename) as file_path:
        if file_path is None:
            return None
        output = _patch_output_from_file(file_path, media_type=media_type, embedder_name=embedder_name)
    if output is None:
        return None
    grid = np.asarray(output.patch_grid, dtype=np.float32)
    cls_row = np.asarray(output.cls_vec, dtype=np.float32).reshape(1, -1)
    return list(np.concatenate([cls_row, grid.reshape(-1, grid.shape[-1])], axis=0))


def _cache_region_vectors(
    elem: LabeledElement,
    eid: str,
    snap: dict[int, dict[str, Any]] | None,
    *,
    media_type: str,
    embedder_name: str,
    lookups: tuple[dict[str, list[int]], dict[str, list[int]], dict[str, list[int]]] | None,
    neg_cache: dict[str, list[np.ndarray]],
    score_cache: dict[str, list[np.ndarray]],
    patch_capable: bool,
) -> None:
    """Top up *elem*'s two region caches from a single grid resolution.

    A patch element's score-row stack feeds both: the negatives a Bad element
    floods (the cross-dataset counterpart of
    :func:`~vtscore.detectors.training.bad_negative_vecs`), and the rows the
    *scorer* max-pools that image over - which threshold calibration collapses
    every bag, Good and Bad alike, over.  Under MaxPatch those are the same
    rows, so the two caches are fed from one resolution, cached so re-sorts
    don't re-resolve.

    The resolution is skipped entirely unless the detector scores in a patch
    space (*patch_capable*; see :func:`_embedder_supports_patch_regions`).  On a
    legacy single-vector detector it would return ``None`` anyway, so no origin
    file is probed for a grid that cannot exist.  On a multi-embedder dataset
    the skip is load-bearing: a semantic- or structural-locked detector is
    scored against its own space's full-image vectors, so flooding it with the
    dataset's patch grid would train it on a different embedding space (and
    raise :class:`~vtscore.embedding.precomputed.MismatchedVectorError` outright
    when the two dimensions differ).  Both
    caches stay empty, and :func:`build_xy_from_labelset` falls back to the
    single image-level vector per element.
    """
    if not patch_capable:
        return
    wants_negatives = elem.label == "bad" and eid not in neg_cache
    wants_score_rows = elem.label in ("good", "bad") and eid not in score_cache
    if not (wants_negatives or wants_score_rows):
        return
    rows = _resolve_score_rows(elem, snap, media_type=media_type, embedder_name=embedder_name, lookups=lookups)
    if rows is None:
        return
    if wants_negatives:
        neg_cache[eid] = rows
    if wants_score_rows:
        score_cache[eid] = rows


def populate_label_embeddings(
    det_ctx,
    labelset: LabelSet,
    *,
    media_type: str,
    snap: dict[int, dict[str, Any]] | None,
    on_progress: ProgressCallback | None = None,
) -> int:
    """Ensure every labelset element has a cached embedding on *det_ctx*.

    Resolution per element (skipping if already cached):

    1. Element resolves to a cid in the active dataset → reuse
       ``snap[cid]``'s primary vector (no I/O).
    2. Element's origin can be resolved to a file via its importer → embed
       the file with the active dataset's embedder (or the media type's
       default).
    3. Otherwise the element is skipped - it won't contribute to training
       this session.

    Returns the number of elements that have a cached vector after this
    pass.
    """
    from vtscore.detectors.labelset_elements import stable_element_id

    # Labels are resolved and embedded in the same space the model-invalidation
    # and re-embed checks key on, so the label cache stays valid across an
    # active-dataset switch that keeps binding the same concrete embedder of the
    # detector's type.  See ``docs/plans/patch-embedder.md`` → "Per-detector
    # embedder type".
    embedder_name = keying_embedder_for_snap(det_ctx, snap)
    _maybe_clear_cache_on_embedder_switch(det_ctx, embedder_name)
    cache: dict[str, np.ndarray] = det_ctx.label_embeddings
    region_cache: dict[str, tuple[float, float, float, float] | None] = det_ctx.label_embedding_regions
    total = len(labelset.elements)
    cached = 0

    neg_cache: dict[str, list[np.ndarray]] = det_ctx.label_negative_regions
    score_cache: dict[str, list[np.ndarray]] = det_ctx.label_score_regions
    patch_capable = _embedder_supports_patch_regions(embedder_name)

    # Build the origin/md5/name lookup tables once from *snap* and thread them
    # through every element's cid resolution, so the pass is O(N + labels)
    # instead of O(labels × N).
    lookups = _lookups_for_snap(snap)

    for idx, elem in enumerate(labelset.elements):
        eid = stable_element_id(elem)
        _cache_region_vectors(
            elem,
            eid,
            snap,
            media_type=media_type,
            embedder_name=embedder_name,
            lookups=lookups,
            neg_cache=neg_cache,
            score_cache=score_cache,
            patch_capable=patch_capable,
        )

        # Cache hit only when the cached vector was built against the same
        # ``region_box`` the element currently carries.  Region-voted
        # elements (``region_box is not None``) always fall through so the
        # patch grid is re-pooled with the latest box.  Image-level
        # elements use the cache only when the cached vector was *also*
        # built image-level - otherwise we'd return a stale region-pooled
        # vector after a region→none transition (e.g. good→bad on a
        # previously region-voted media; or un-vote / re-vote without a
        # region).  See logical-bug-audit finding M4.
        if eid in cache and elem.region_box is None and region_cache.get(eid) is None:
            cached += 1
            continue

        emb = _resolve_uncached_embedding(
            elem,
            snap,
            media_type=media_type,
            embedder_name=embedder_name,
            patch_capable=patch_capable,
            lookups=lookups,
        )
        if emb is not None:
            cache[eid] = emb
            region_cache[eid] = elem.region_box
            cached += 1
        if on_progress:
            on_progress(elem.origin_name or elem.filename or eid, idx + 1, total)

    # Stamp the embedder the cache is now built against so the next call can
    # detect a switch and invalidate.  Also persist to the detector registry
    # so the smart preload predictor warms the right model next session
    # instead of the media type's default.
    if embedder_name:
        det_ctx.embedder = embedder_name
        from vtscore.detectors.registry import record_detector_embedder

        record_detector_embedder(det_ctx.detector_id, embedder_name)
    return cached


def build_xy_from_labelset(
    det_ctx,
    labelset: LabelSet,
) -> tuple[list[np.ndarray], list[float], list, dict]:
    """Build ``(X_list, y_list, groups, score_rows)`` from *det_ctx*'s caches.

    A Good element contributes its single cached vector.  A Bad element on a
    patch dataset contributes its flooded image-level + raw-patch negatives (cached in
    ``label_negative_regions`` by :func:`populate_label_embeddings`); on a
    legacy dataset it contributes its single image-level vector, as before.
    ``groups`` carries one bag id per row - ``("g"/"b", element_id)`` - so the
    trainer/calibrator balance and split by element (image), not by row.  When
    nothing floods, every bag holds one row and the downstream path is
    byte-for-byte the pre-flood behaviour.

    ``score_rows`` maps each bag id to the full score-row stack that element's
    media is *scored* over at inference (``label_score_regions``), so threshold
    calibration collapses a Good bag exactly the way it collapses a Bad bag and
    the way the scorer collapses any image - see
    :func:`vtscore.detectors.training._calibration_score_rows`.  Empty on a
    legacy dataset, where there is nothing to correct.
    """
    from vtscore.detectors.labelset_elements import stable_element_id

    cache: dict[str, np.ndarray] = det_ctx.label_embeddings
    neg_cache: dict[str, list[np.ndarray]] = det_ctx.label_negative_regions
    score_cache: dict[str, list[np.ndarray]] = det_ctx.label_score_regions
    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    groups: list = []
    score_rows: dict = {}

    def _record(group: tuple, eid: str) -> None:
        rows = score_cache.get(eid)
        if rows:
            score_rows[group] = np.stack(rows)

    for elem in labelset.elements:
        if elem.label not in ("good", "bad"):
            continue
        eid = stable_element_id(elem)
        if elem.label == "bad":
            flooded = neg_cache.get(eid)
            if flooded:
                for vec in flooded:
                    X_list.append(vec)
                    y_list.append(0.0)
                    groups.append(("b", eid))
                _record(("b", eid), eid)
                continue
        emb = cache.get(eid)
        if emb is None:
            continue
        X_list.append(emb)
        y_list.append(1.0 if elem.label == "good" else 0.0)
        group = ("g" if elem.label == "good" else "b", eid)
        groups.append(group)
        _record(group, eid)
    return X_list, y_list, groups, score_rows


def labeled_media_ids(labelset: LabelSet, snap: dict[int, dict[str, Any]] | None) -> set[int]:
    """The media ids in *snap* that carry a good/bad label in *labelset*.

    These are the in-dataset media the detector trains on, and therefore the
    ids the fold-anchored threshold must drop from its haystacks (issue #3308;
    see :func:`vtscore.detectors.training._fused_threshold`).  Cross-dataset
    elements that resolve to nothing in *snap* are absent from the haystack
    already, so they contribute nothing here.
    """
    if not snap:
        return set()
    from vtscore.detectors.labelset_elements import resolve_current_dataset_cid  # noqa: PLC0415
    from vtscore.state import build_media_lookup  # noqa: PLC0415

    # ``build_media_lookup`` reads ``media["id"]``; every Flask-loaded snap
    # carries it equal to the dict key, but minimal snaps (tests, embedded
    # callers) may not - default it to the key rather than requiring it.
    lookups = build_media_lookup({cid: {**m, "id": m.get("id", cid)} for cid, m in snap.items()})

    ids: set[int] = set()
    for elem in labelset.elements:
        if elem.label not in ("good", "bad"):
            continue
        cid = resolve_current_dataset_cid(elem, lookups)
        if cid is not None and cid in snap:
            ids.add(cid)
    return ids


def labelset_resolution_report(
    det_ctx,
    labelset: LabelSet,
    *,
    media_type: str,
    snap: dict[int, dict[str, Any]] | None,
) -> dict[str, Any]:
    """Why *labelset* produced no trainable head, as a UI-facing diagnostic.

    :func:`train_from_labelset` and :func:`labelset_train_and_score` simply
    decline to train when the labels don't resolve into at least one Good and
    one Bad vector - which is the right shape for a *training* entry point but
    loses the "why did this detector produce nothing?" answer the find-label
    and portable-export routes show the user.  This rebuilds that answer from
    what the training pass already left on *det_ctx*: the elements
    :func:`build_xy_from_labelset` can assemble a row for are exactly the ones
    that resolved, so the failures are the complement.

    Call it **after** :func:`populate_label_embeddings` (i.e. after the training
    attempt), never before - on a cold context it would report every element as
    a failure.  It is a failure-path-only report, so the extra
    :func:`~vtscore.state.media_lookup.build_media_lookup` pass it costs is
    paid only by a detector that is already not going to score.

    ``dataset_matched`` counts elements resolving to a media in *snap* by the
    full origin ▸ md5 ▸ name ladder :func:`resolve_current_dataset_cid` walks -
    not md5 alone, which is all the hand-rolled predecessor of this path could
    match on.
    """
    from vtscore.detectors.labelset_elements import resolve_current_dataset_cid, stable_element_id

    # Ask the trainer's own assembly which elements contributed, rather than
    # re-deriving "did this resolve?" from the embedding cache: a Bad element on
    # a patch dataset trains from ``label_negative_regions`` and can contribute
    # rows with nothing in ``label_embeddings``, so the cache alone would report
    # a label that trained fine as a failure.
    _x, y_list, groups, _score_rows = build_xy_from_labelset(det_ctx, labelset)
    contributed = {eid for _kind, eid in groups}
    has_good = any(y == 1.0 for y in y_list)
    has_bad = any(y == 0.0 for y in y_list)
    lookups = _lookups_for_snap(snap)

    total = 0
    dataset_matched = 0
    resolved_from_origin = 0
    failures: list[LabeledElement] = []
    for elem in labelset.elements:
        if elem.label not in ("good", "bad"):
            continue
        total += 1
        in_dataset = False
        if snap:
            cid = resolve_current_dataset_cid(elem, lookups)
            in_dataset = cid is not None and cid in snap
        if in_dataset:
            dataset_matched += 1
        if stable_element_id(elem) not in contributed:
            failures.append(elem)
        elif not in_dataset:
            resolved_from_origin += 1

    diagnostic: dict[str, Any] = {
        "total_labels": total,
        "dataset_matched": dataset_matched,
        "needed_resolution": total - dataset_matched,
        "resolved_from_origin": resolved_from_origin,
        "failed_resolution": len(failures),
        "has_good": has_good,
        "has_bad": has_bad,
        "media_type": media_type,
    }
    if failures:
        diagnostic["sample_failures"] = [
            {
                "origin": elem.origin,
                "origin_name": elem.origin_name,
                "filename": elem.filename,
                "md5": elem.md5[:12],
                "label": elem.label,
            }
            for elem in failures[:3]
        ]
    elif not has_good or not has_bad:
        diagnostic["hint"] = "Every label resolved, but they are all the same class (need both good and bad)"
    return diagnostic


# ---------------------------------------------------------------------------
# Cross-dataset local features (structural / SIFT-VLAD detectors)
# ---------------------------------------------------------------------------


def _resolve_uncached_local_features(
    elem: LabeledElement,
    snap: dict[int, dict[str, Any]] | None,
    *,
    embedder,
    lookups: tuple[dict[str, list[int]], dict[str, list[int]], dict[str, list[int]]] | None = None,
) -> Any | None:
    """Re-derive *elem*'s :class:`StructuralFeatures`, not consulting the cache.

    Tries the in-dataset path first: when *elem* resolves to a cid in the
    active *snap* that already carries ``local_features``, reuse them (no I/O).
    Otherwise resolve the origin to a file via its importer and run the
    embedder's ``local_features_forward`` to detect features freshly.  The
    **full** (unfiltered) feature set is returned; any ``region_box`` is applied
    downstream at template-build time.  Returns ``None`` when neither path
    yields features.

    *lookups* is the pre-built ``build_media_lookup`` triple for *snap*; loop
    callers pass it so the cid resolution doesn't rebuild the tables per element.
    """
    from vtscore.detectors.labelset_elements import resolve_current_dataset_cid
    from vtscore.detectors.resolver import resolve_file_context
    from vtscore.media.embedder import media_from_path
    from vtscore.media.structural import StructuralFeatures

    if snap:
        cid = resolve_current_dataset_cid(elem, lookups)
        if cid is not None and cid in snap:
            feats = snap[cid].get("local_features")
            if isinstance(feats, StructuralFeatures) and feats.count > 0:
                return feats

    with resolve_file_context(elem.origin, elem.origin_name, elem.filename) as file_path:
        if file_path is None:
            return None
        try:
            feats = embedder.local_features_forward(media_from_path(file_path))
        except Exception:
            log.warning(
                "labelset_training: local_features_forward(%s) raised; "
                "this label won't contribute a structural template",
                elem.origin_name or elem.filename or "<unknown>",
                exc_info=True,
            )
            return None
    if feats is None or getattr(feats, "count", 0) == 0:
        return None
    return feats


def populate_label_local_features(
    det_ctx,
    labelset: LabelSet,
    *,
    snap: dict[int, dict[str, Any]] | None,
) -> int:
    """Ensure every labelled element has cached local features on *det_ctx*.

    A no-op (returns 0) unless the active dataset's embedder is structural
    (``supports_geometric_verification``) - non-structural detectors never need
    local features.  For a structural detector it re-derives the
    :class:`~vtscore.media.structural.StructuralFeatures` for each good/bad
    element (reusing the active dataset's stored features when the element is
    loaded, resolving the origin file otherwise) and caches them on
    ``det_ctx.label_local_features`` keyed by ``stable_element_id``.  The
    embedder-switch invalidation in :func:`_maybe_clear_cache_on_embedder_switch`
    clears this cache alongside the embedding cache.

    Returns the number of elements that have cached features after this pass.
    """
    from vtscore.detectors.labelset_elements import stable_element_id

    # Labels are resolved and embedded in the same space the model-invalidation
    # and re-embed checks key on, so the label cache stays valid across an
    # active-dataset switch that keeps binding the same concrete embedder of the
    # detector's type.  See ``docs/plans/patch-embedder.md`` → "Per-detector
    # embedder type".
    embedder_name = keying_embedder_for_snap(det_ctx, snap)
    embedder = None
    if embedder_name:
        from vtscore.media import get_embedder

        try:
            embedder = get_embedder(embedder_name)
        except (KeyError, ValueError):
            embedder = None
    if embedder is None or not getattr(embedder, "supports_geometric_verification", False):
        return 0

    cache: dict[str, Any] = det_ctx.label_local_features
    # Build the origin/md5/name lookup tables once and thread them through,
    # so this pass is O(N + labels) instead of O(labels × N).
    lookups = _lookups_for_snap(snap)
    for elem in labelset.elements:
        if elem.label not in ("good", "bad"):
            continue
        eid = stable_element_id(elem)
        if eid in cache:
            continue
        feats = _resolve_uncached_local_features(elem, snap, embedder=embedder, lookups=lookups)
        if feats is not None:
            cache[eid] = feats
    return len(cache)


def _labelset_feature_snapshot(
    det_ctx,
    labelset: LabelSet,
) -> tuple[dict[str, dict[str, Any]], dict[str, None], dict[str, None], dict[str, tuple[float, float, float, float]]]:
    """Project the cached local features into the chokepoint's vote/snap shape.

    Builds a synthetic ``feature_snap`` (``element_id -> {"local_features": ...}``)
    plus ``good_votes`` / ``bad_votes`` / ``region_boxes`` keyed by the same
    ``stable_element_id`` so :func:`maybe_structural_rerank` can build templates
    and train the verification classifier against the cross-dataset labelset
    exactly as it does against in-dataset votes.
    """
    from vtscore.detectors.labelset_elements import stable_element_id

    cache: dict[str, Any] = det_ctx.label_local_features
    feature_snap: dict[str, dict[str, Any]] = {}
    good_votes: dict[str, None] = {}
    bad_votes: dict[str, None] = {}
    region_boxes: dict[str, tuple[float, float, float, float]] = {}
    for elem in labelset.elements:
        if elem.label not in ("good", "bad"):
            continue
        eid = stable_element_id(elem)
        feats = cache.get(eid)
        if feats is None:
            continue
        feature_snap[eid] = {"local_features": feats}
        if elem.label == "good":
            good_votes[eid] = None
            if elem.region_box is not None:
                region_boxes[eid] = elem.region_box
        else:
            bad_votes[eid] = None
    return feature_snap, good_votes, bad_votes, region_boxes


def maybe_labelset_structural_rerank(
    det_ctx,
    labelset: LabelSet,
    results: list[dict[str, Any]],
    threshold: float,
    snap: dict[int, dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], float]:
    """Stage-2 structural re-rank for the saved-detector (labelset) sort path.

    The counterpart to the vote-driven re-rank wired into
    :func:`~vtscore.detectors.training.train_and_score`: when a saved structural
    detector is sorted against a (possibly different) loaded dataset, this
    re-derives the labelset's local features, builds the RegionYes templates and
    verification classifier from them, and geometrically re-ranks the active
    dataset's Stage-1 shortlist.  A no-op for non-structural datasets (gated on
    the active snapshot carrying ``local_features``) and when no labelled element
    yields a usable template.
    """
    from vtscore.training.structural_similarity import maybe_structural_rerank, snapshot_is_structural

    if not snap or not snapshot_is_structural(snap):
        return results, threshold
    populate_label_local_features(det_ctx, labelset, snap=snap)
    feature_snap, good_votes, bad_votes, region_boxes = _labelset_feature_snapshot(det_ctx, labelset)
    if not good_votes:
        return results, threshold
    return maybe_structural_rerank(
        results,
        threshold,
        snap,
        good_votes,
        bad_votes,
        region_boxes,
        det_ctx,
        feature_snap=feature_snap,
    )


class Haystack(NamedTuple):
    """A calibration population that is not the label-resolution snapshot.

    ``medias`` is the snapshot the fold-anchored cut is realized on, and
    ``to_source`` maps each of its ids back to the id in *snap* it descends
    from - the identity for a snapshot that is just *snap* renumbered, the
    parent media for every converter output or re-clip sub-item.  The mapping
    is what lets the vote exclusion (#3308) still name the labelled items after
    routing has thrown their ids away.
    """

    medias: dict[int, dict[str, Any]]
    to_source: dict[int, int]


def train_from_labelset(
    det_ctx,
    labelset: LabelSet,
    *,
    media_type: str,
    snap: dict[int, dict[str, Any]] | None,
    haystack_for: "Callable[[str], Haystack | None] | None" = None,
    on_progress: ProgressCallback | None = None,
) -> bool:
    """Populate the embedding cache, build (X, y), train, and store on *det_ctx*.

    Returns ``True`` when an MLP was trained (need ≥1 good and ≥1 bad cached
    vector); otherwise leaves ``det_ctx.model`` untouched.

    *snap* does two jobs, and a caller that scores something other than what it
    loaded needs them separated.  It is the snapshot the labelset's elements
    are resolved against, and it is the haystack the threshold is calibrated
    on.  For the GUI those are the same set - it scores the dataset it loaded -
    but the CLI converts, re-clips and re-embeds before scoring, so calibrating
    on *snap* would realize the cut's quantile on a population inference never
    sees (issue #3647).  Such a caller passes *haystack_for*, which is invoked
    **after** the labels are embedded - so it is handed ``det_ctx.embedder``,
    the space they turned out to land in, which is the very thing a routing
    decision needs and which does not exist before this call.  Returning
    ``None`` from it keeps *snap* as the haystack.
    """
    populate_label_embeddings(
        det_ctx,
        labelset,
        media_type=media_type,
        snap=snap,
        on_progress=on_progress,
    )
    X_list, y_list, groups, score_rows = build_xy_from_labelset(det_ctx, labelset)
    if len(X_list) < 2:
        return False
    if not any(y == 1.0 for y in y_list) or not any(y == 0.0 for y in y_list):
        return False

    from vtscore.detectors.training import train_and_threshold

    # populate_label_embeddings stamped det_ctx.embedder with the space the
    # labels were embedded in; score the safe-threshold pass in that same space.
    # Pass det_ctx so the fold orderings are cached for a no-retrain Inclusion
    # slide (otherwise the slide can't move the cutoff — see train_and_threshold).
    haystack = haystack_for(det_ctx.embedder or "") if haystack_for is not None else None
    voted_ids = labeled_media_ids(labelset, snap)
    if haystack is not None:
        # The haystack's ids are its own; name the labelled items in them.
        voted_ids = {hid for hid, src in haystack.to_source.items() if src in voted_ids}

    mlp, threshold = train_and_threshold(
        X_list,
        y_list,
        snap=snap,
        embedder_name=det_ctx.embedder or None,
        det_ctx=det_ctx,
        groups=groups,
        score_rows=score_rows,
        voted_ids=voted_ids,
        haystack=haystack.medias if haystack is not None else None,
    )
    det_ctx.model = mlp
    det_ctx.threshold = threshold
    return True


def labelset_train_and_score(
    det_ctx,
    labelset: LabelSet,
    *,
    media_type: str,
    clips_dict: dict[int, dict[str, Any]],
    inclusion_value: int = 0,
    calibrate_count: int = 2,
    calibration_fraction: float | None = None,
    rows: Any = None,
    on_progress: ProgressCallback | None = None,
) -> tuple[list[dict[str, Any]], float, Any | None]:
    """Train an MLP on the full labelset, then score every media in *clips_dict*.

    Counterpart to :func:`~vtscore.detectors.training.train_and_score` that
    trains on cross-dataset labels.  It assembles ``(X_list, y_list)`` from
    the resolved labelset (populating the embedding cache on the way) and
    then defers the threshold → train → score → format tail to the shared
    :func:`~vtscore.detectors.training._train_and_score_xy` core, so the two
    pipelines stay in lock-step (region-aware scoring, NaN sanitisation,
    population-fused thresholding).  Scoring is still scoped to the active
    dataset's media, since that is what the user is sorting in the UI.

    ``calibration_fraction=None`` (no explicit user setting) resolves inside
    the shared core to the per-space production split for the detector's
    embedder (see
    :func:`~vtscore.detectors.training.resolve_calibration_fraction`).

    *rows* is an optional pre-built
    :class:`~vtscore.detectors.training.ScoringRows` for *clips_dict*, handed
    straight to the shared core so a caller scoring several detectors over one
    uncached snapshot (cross-dataset Find) stacks the corpus once - see
    :func:`~vtscore.detectors.training._train_and_score_xy`.

    *on_progress* is forwarded to :func:`populate_label_embeddings`, which is
    the expensive half whenever the labelset does not resolve into
    *clips_dict*: each unresolved element costs one origin fetch (and, on a
    patch detector, one ``patch_forward``).  A caller driving a progress bar -
    or wanting a cancellation checkpoint - passes it.
    """
    from vtscore.detectors.training import _train_and_score_xy

    populate_label_embeddings(det_ctx, labelset, media_type=media_type, snap=clips_dict, on_progress=on_progress)
    X_list, y_list, groups, score_rows = build_xy_from_labelset(det_ctx, labelset)
    results, threshold, model = _train_and_score_xy(
        X_list,
        y_list,
        clips_dict,
        inclusion_value=inclusion_value,
        calibrate_count=calibrate_count,
        calibration_fraction=calibration_fraction,
        det_ctx=det_ctx,
        groups=groups,
        score_rows=score_rows,
        voted_ids=labeled_media_ids(labelset, clips_dict),
        rows=rows,
    )

    # Stage-2 structural re-rank for a saved structural detector reloaded
    # cross-dataset (the labelset counterpart to the vote-driven re-rank in
    # ``train_and_score``).  A no-op for every non-structural dataset.
    results, threshold = maybe_labelset_structural_rerank(det_ctx, labelset, results, threshold, clips_dict)
    return results, threshold, model
