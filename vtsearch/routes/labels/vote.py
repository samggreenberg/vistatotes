"""Blueprint for label management routes (export, import, fill-from-sort).

Migrated to ``flask_smorest`` so the routes are described in
``/api/openapi.json``.
"""

from __future__ import annotations

import logging

from flask_smorest import Blueprint, abort

from vtsearch.schemas.labels import (
    FillFromSortRequestSchema,
    FillFromSortResponseSchema,
    LabelsExportQuerySchema,
    LabelsExportResponseSchema,
    LabelsImportRequestSchema,
    LabelsImportResponseSchema,
)
from vtscore.detectors.dataset_sync import validated_vote_snapshot
from vtsearch.routes._context import require_dataset_header, require_detector_header
from vtsearch.state import (
    apply_label,
    apply_label_with_click_time,
    cached_md5_lookup,
    cached_media_lookups,
    get_active_detector_context,
    resolve_media_ids,
)
from vtscore.datasets.vote_provenance import read_provenance
from vtscore.utils.hits import build_media_hit, hit_custom_metadata

logger = logging.getLogger(__name__)

labels_bp = Blueprint(
    "labels",
    __name__,
    description="Export, import, and bulk-fill label assignments.",
)


def _select_vote_pools(
    label_filter: str,
    goods_only: bool,
    good_votes: dict[int, None],
    bad_votes: dict[int, None],
    verified_ids: dict[int, None] | None = None,
) -> tuple[dict, dict]:
    """Pick the (goods, bads) dicts to feed into ``LabelSet.from_clips_and_votes``.

    ``label_filter == "corrections"`` returns both pools; the corrections
    filtering step happens after annotation, since "correction" depends on
    the find-initial labels, not on good vs bad.

    ``unverified`` / ``verified`` partition the pools by ``verified_ids`` (the
    Find-mode set of human-touched items): ``unverified`` is the left-panel
    work queue (the detector's calls the human hasn't acted on), ``verified``
    is the right-panel confirmed set.

    Takes the vote dicts as parameters (rather than reading the module-level
    proxies) so the caller can pass an atomic snapshot from
    :func:`validated_vote_snapshot`, guaranteed to be keyed in the same
    dataset's cid space as the medias being composed with.
    """
    verified = verified_ids or {}
    if label_filter == "good":
        return good_votes, {}
    if label_filter == "bad":
        return {}, bad_votes
    if label_filter == "unverified":
        return (
            {cid: None for cid in good_votes if cid not in verified},
            {cid: None for cid in bad_votes if cid not in verified},
        )
    if label_filter == "verified":
        return (
            {cid: None for cid in good_votes if cid in verified},
            {cid: None for cid in bad_votes if cid in verified},
        )
    if label_filter == "corrections":
        return good_votes, bad_votes
    if label_filter:
        return good_votes, bad_votes
    return good_votes, ({} if goods_only else bad_votes)


def _make_correction_annotator(all_medias: dict):
    """Return a per-entry ``is_correction`` annotator, or ``None`` when no find-initial state exists.

    A label is a correction when the detector's pre-vote label
    (``find_initial_labels``) differs from the current label. When there is
    no find-initial state (no detector was run, or the vote came from
    outside Find), we return ``None`` so callers skip annotation entirely.

    The ``md5 -> media_id`` map is built once here so the returned closure is
    O(1) per entry, letting both the buffered and the streaming export paths
    annotate one element at a time without re-scanning ``all_medias``.
    """
    from vtsearch.state import get_find_initial_labels

    find_initial = get_find_initial_labels()
    if not find_initial:
        return None

    md5_to_id: dict[str, int] = {}
    for mid, m in all_medias.items():
        md5_val = m.get("md5")
        if md5_val and md5_val not in md5_to_id:
            md5_to_id[md5_val] = mid

    def annotate(entry: dict) -> None:
        media_id = md5_to_id.get(entry.get("md5", ""))
        if media_id is not None and media_id in find_initial:
            entry["is_correction"] = entry.get("label") != find_initial[media_id]
        else:
            entry["is_correction"] = False

    return annotate


def _annotate_corrections(result: dict, all_medias: dict) -> None:
    """Add ``is_correction`` to every label entry in *result* (mutates in place)."""
    annotate = _make_correction_annotator(all_medias)
    if annotate is None:
        return
    for entry in result["labels"]:
        annotate(entry)


def _build_entry_metadata(media: dict) -> dict:
    """Return the metadata blob for one labelled media (display, origin, and custom)."""
    from vtscore.media import get as get_media_type  # noqa: PLC0415

    try:
        meta = get_media_type(media.get("media_type", "audio")).display_metadata(media)
    except KeyError:
        meta = {}

    origin = media.get("origin")
    if isinstance(origin, dict):
        for k, v in origin.get("params", {}).items():
            meta.setdefault(k, v)

    # Sanitised rather than read straight off the media: an importer may ship
    # a pre-computed vector nested inside ``custom_metadata`` via
    # ``custom_metadata_map``, and this blob becomes both an export column and
    # a JSON response field.
    importer_custom = hit_custom_metadata(media)
    if importer_custom:
        meta.update(importer_custom)
    # Humanize "File Size" for export so rows read "8.0 KB" instead of raw
    # bytes, matching what the focus-view UI shows. ``display_metadata`` keeps
    # the raw int (the media-list API returns it and the UI formats it
    # client-side); only this export copy is stringified. Mirrors the frontend
    # formula in center-panel.component.ts (`formatMetadataValue`).
    fs = meta.get("File Size")
    if isinstance(fs, (int, float)) and not isinstance(fs, bool):
        meta["File Size"] = f"{fs / 1024:.1f} KB"
    return meta


_BASE_EXPORT_COLUMNS = ["label", "md5", "origin_name", "filename", "category", "origin"]

#: Find-mode partitions of the *session's* vote state.  ``verified`` /
#: ``unverified`` split the current Find run's work queue by ``verified_ids``
#: and ``corrections`` compares against the session's ``find_initial_labels``;
#: none of these are properties of the persisted labelset, so labelset
#: elements that never resolved into the active dataset (and therefore were
#: never part of the session) are excluded from these exports by design.
_VOTE_SCOPED_FILTERS = frozenset({"corrections", "unverified", "verified"})


def _fallback_passes_filter(label: str, label_filter: str, goods_only: bool) -> bool:
    """Apply the good/bad export filters to an origin-only labelset element.

    Mirrors :func:`_select_vote_pools` for the labelset-shaped filters
    (``""`` / ``good`` / ``bad`` / ``both``); the vote-scoped filters never
    reach this function (see ``_VOTE_SCOPED_FILTERS``).
    """
    if label_filter == "good":
        return label == "good"
    if label_filter == "bad":
        return label == "bad"
    if not label_filter and goods_only:
        return label == "good"
    return True


def _persisted_labelset_for_active_detector():
    """Return the active detector's on-disk :class:`LabelSet`, or ``None``.

    Reads the detector JSON fresh so the fallback sees writes made through
    the labelset-element vote route (which bypasses the in-memory vote
    dicts).  Falls back to the context's ``cached_labelset`` when the
    registry entry or file is missing (e.g. a test-constructed context).
    """
    from vtscore.datasets.labelset import LabelSet
    from vtscore.detectors.registry import get_detector
    from vtscore.detectors.store import _detector_path, _read_detector

    det_ctx = get_active_detector_context()
    if not det_ctx.detector_id:
        return None
    entry = get_detector(det_ctx.detector_id)
    if entry and entry.get("name"):
        data = _read_detector(_detector_path(entry["name"]))
        if data is not None:
            return LabelSet.from_dict(data.get("labelset") or {})
    return det_ctx.cached_labelset


def _origin_only_fallback_entries(
    labelset,
    all_medias: dict,
    label_filter: str,
    goods_only: bool,
) -> list[dict]:
    """Serialise persisted labelset elements the vote path cannot represent.

    ``GET /api/labels/export`` composes its payload from cid-keyed votes
    intersected with the active dataset's medias, but the detector's
    labelset is origin-keyed and dataset-agnostic: elements whose media were
    never ingested into any loaded dataset (imported detectors whose origins
    were unreachable, registry detectors from another host, partial dataset
    overlap) have no cid and would silently vanish from the export.  This
    fallback appends those elements straight from the on-disk labelset so an
    export is always a faithful rendering of the labelset.

    Elements that *do* resolve into the active dataset are never added here:
    for them the session's vote state is the source of truth (an element
    that resolves but has no vote was unlabelled by the user this session,
    and resurrecting it from disk would undo that).  Each returned entry is
    marked ``origin_only: true`` so consumers can tell a resolved label from
    one rendered purely from provenance.

    Vote-scoped filters (``corrections`` / ``unverified`` / ``verified``)
    return no fallback entries - they partition the current session's vote
    state, which origin-only elements were never part of.
    """
    if label_filter in _VOTE_SCOPED_FILTERS:
        return []

    persisted = _persisted_labelset_for_active_detector()
    if persisted is None or not persisted.elements:
        return []

    from vtscore.datasets.labelset import element_key

    seen = {element_key(el) for el in labelset.elements}
    seen.discard(None)

    return _serialise_persisted_elements(
        persisted.elements,
        all_medias,
        label_filter,
        goods_only,
        seen=seen,
        skip_resolved=True,
    )


def _serialise_persisted_elements(
    elements,
    all_medias: dict,
    label_filter: str,
    goods_only: bool,
    *,
    seen: set | None = None,
    skip_resolved: bool = False,
) -> list[dict]:
    """Serialise on-disk labelset elements into export entries.

    Shared by the two paths that read straight from a detector's persisted
    labelset, which differ only in what they skip:

    * :func:`_origin_only_fallback_entries` tops up a *vote-derived* export,
      so it passes the vote-derived keys as *seen* and sets *skip_resolved*:
      an element the active dataset can see is the session's to represent.
    * :func:`_export_persisted_labelset` renders the labelset on its own, so
      it skips nothing and every element is emitted.

    ``origin_only`` marks an entry whose media does not resolve into the
    active dataset, in both paths - a resolved entry from the second path is
    a normal label, not a provenance-only rendering.
    """
    from vtscore.datasets.labelset import element_key

    if all_medias:
        origin_lookup, md5_lookup, name_lookup = cached_media_lookups()
    else:
        origin_lookup, md5_lookup, name_lookup = {}, {}, {}

    entries: list[dict] = []
    for el in elements:
        if el.label not in ("good", "bad"):
            continue
        if not _fallback_passes_filter(el.label, label_filter, goods_only):
            continue
        if seen:
            key = element_key(el)
            if key is not None and key in seen:
                continue
        entry = el.to_dict()
        if not resolve_media_ids(entry, origin_lookup, md5_lookup, name_lookup):
            entry["origin_only"] = True
        elif skip_resolved:
            continue
        entries.append(entry)
    return entries


def _build_unresolved_entry_metadata(entry: dict) -> dict:
    """Return the metadata blob for a label entry with no media to enrich from.

    Degraded counterpart of :func:`_build_entry_metadata` for entries whose
    MD5 doesn't resolve into the active dataset (origin-only labelset
    elements, dupe-set members collapsed out of the dataset): there is no
    media dict to take display metadata from, so the blob is composed from
    what the entry itself carries - flattened origin params plus the
    element's stored ``metadata``.
    """
    meta: dict = {}
    origin = entry.get("origin")
    if isinstance(origin, dict):
        for k, v in (origin.get("params") or {}).items():
            meta.setdefault(k, v)
    stored = entry.get("metadata")
    if isinstance(stored, dict):
        meta.update(stored)
    return meta


def _make_enricher(all_medias: dict):
    """Return a per-entry ``custom_metadata`` enricher.

    The returned callable mutates one label entry in place (attaching
    ``custom_metadata`` when the media has any) and returns the set of
    metadata keys it added, so a streaming caller can annotate elements one
    at a time.  Resolution reuses the active dataset's cached MD5 → media-ID
    lookup (S14) instead of building a fresh ``md5 -> media`` map on every
    export; media are then resolved by id against the caller's snapshot.

    Flattens origin params so fields like ``contentID`` / ``mediaID`` /
    ``media_url`` surface as selectable export columns alongside the
    importer's own ``custom_metadata``.  Entries with no media in the active
    dataset degrade to :func:`_build_unresolved_entry_metadata`.
    """
    # No medias means no resolution to do, and the lookup itself reads the
    # active dataset context - which the detector-scoped export may not have.
    md5_lookup = cached_md5_lookup() if all_medias else {}

    def enrich(entry: dict) -> set[str]:
        cids = md5_lookup.get(entry.get("md5") or "")
        media = all_medias.get(cids[0]) if cids else None
        meta = _build_entry_metadata(media) if media else _build_unresolved_entry_metadata(entry)
        if not meta:
            return set()
        entry["custom_metadata"] = meta
        return set(meta.keys())

    return enrich


def _enrich_with_metadata(result: dict, all_medias: dict) -> None:
    """Attach ``custom_metadata`` per entry and the ``available_columns`` list."""
    enrich = _make_enricher(all_medias)
    all_meta_keys: set[str] = set()
    for entry in result["labels"]:
        all_meta_keys.update(enrich(entry))

    base_lower = {c.lower() for c in _BASE_EXPORT_COLUMNS}
    extra_keys = sorted(k for k in all_meta_keys if k.lower() not in base_lower)
    result["available_columns"] = _BASE_EXPORT_COLUMNS + extra_keys


def _export_persisted_labelset(name: str, query: dict):
    """Export a *named* detector's on-disk labelset, ignoring the live session.

    The session-composed path below answers "what has the human labelled in
    the active (dataset, detector) pair right now"; this one answers "what is
    detector X's labelset", which is a different question and the one the
    Dashboard's row action asks - it names a detector in a list, and the
    answer must not depend on which pair the top-bar pulldown happens to be
    pointing at (issue #3639).  Reading the detector JSON is what makes that
    true: the labelset on disk is kept in step with every vote by
    :func:`~vtscore.detectors.label_sync.sync_labels_to_loaded_detector`, so
    it is current, and it is the exact artefact that re-imports as the
    detector.

    It is also the only reading that survives a live Find session.  Find
    replaces the detector's in-memory votes with its own call for *every*
    item in the dataset, flagged ``find_mode`` so the sync above keeps those
    presumptions out of the labelset (see ``end_find_session``); composing
    from votes would export the whole collection as labels.

    The vote-scoped filters partition that live session, so they have no
    meaning here and are refused rather than silently ignored.
    """
    from vtscore.datasets.labelset import LabelSet
    from vtscore.detectors.store import _detector_path, _read_detector
    from vtscore.state.core import DatasetNotLoadedError
    from vtsearch.state import medias

    label_filter = query["label_filter"]
    if label_filter in _VOTE_SCOPED_FILTERS:
        abort(
            400,
            message=(
                f"label_filter='{label_filter}' partitions the live vote session and "
                "cannot be combined with detector_name, which exports a persisted labelset."
            ),
        )

    data = _read_detector(_detector_path(name))
    if data is None:
        abort(404, message=f"Detector '{name}' not found")

    labelset = LabelSet.from_dict(data.get("labelset") or {})
    # The active dataset is only ever *enrichment* here, so an absent or
    # unloaded one degrades to an unenriched export rather than failing it:
    # the whole point of this path is that the labels do not depend on which
    # pair the app happens to be pointed at.
    try:
        all_medias = dict(medias)
    except DatasetNotLoadedError:
        all_medias = {}
    entries = _serialise_persisted_elements(labelset.elements, all_medias, label_filter, query["goods_only"])

    if query["format"] == "ndjson":
        return _stream_labels_ndjson(LabelSet(), all_medias, label_filter, query["enrich"], entries, annotate=False)

    result: dict = {"labels": entries}
    if query["enrich"]:
        _enrich_with_metadata(result, all_medias)
    return result


@labels_bp.route("/api/labels/export")
@labels_bp.arguments(LabelsExportQuerySchema, location="query")
@labels_bp.response(200, LabelsExportResponseSchema)
@labels_bp.alt_response(400, description="A vote-scoped label_filter was combined with detector_name.")
@labels_bp.alt_response(404, description="detector_name names a detector that does not exist.")
def export_labels(query: dict):
    """Export labels as a :class:`~vtscore.datasets.labelset.LabelSet`.

    Each label entry includes the element's ``origin`` and ``origin_name``
    so consumers know exactly where each labeled element came from. The
    format is a superset of the legacy export format; old consumers
    that only read ``md5`` and ``label`` keys continue to work unchanged.

    Without ``detector_name`` the payload is composed from the session's
    votes intersected with the active dataset, then topped up with persisted
    labelset elements that don't resolve into the active dataset (marked
    ``origin_only: true``) so the export is always a faithful rendering of
    the detector's labelset - see :func:`_origin_only_fallback_entries`.

    With ``detector_name`` it is that detector's persisted labelset instead,
    read from disk and independent of the request's active pair - see
    :func:`_export_persisted_labelset`.
    """
    from vtscore.datasets.labelset import LabelSet

    if query["detector_name"]:
        return _export_persisted_labelset(query["detector_name"], query)

    label_filter = query["label_filter"]
    # Atomic (medias, good_votes, bad_votes, vote_region_boxes) snapshot so
    # the votes we compose with ``all_medias`` are guaranteed to be keyed in
    # the same dataset's cid space; even if a concurrent request rehydrates
    # the detector against a different dataset before this route finishes.
    snap = validated_vote_snapshot()
    goods, bads = _select_vote_pools(
        label_filter, query["goods_only"], snap.good_votes, snap.bad_votes, snap.verified_ids
    )

    all_medias = snap.medias
    labelset = LabelSet.from_clips_and_votes(
        all_medias,
        goods,
        bads,
        vote_region_boxes=snap.vote_region_boxes,
        vote_provenance=snap.vote_provenance,
    )

    fallback_entries = _origin_only_fallback_entries(labelset, all_medias, label_filter, query["goods_only"])

    if query["format"] == "ndjson":
        return _stream_labels_ndjson(labelset, all_medias, label_filter, query["enrich"], fallback_entries)

    result: dict = labelset.to_dict()
    result["labels"].extend(fallback_entries)

    _annotate_corrections(result, all_medias)
    if label_filter == "corrections":
        result["labels"] = [e for e in result["labels"] if e.get("is_correction")]

    if query["enrich"]:
        _enrich_with_metadata(result, all_medias)

    return result


def _stream_labels_ndjson(
    labelset,
    all_medias: dict,
    label_filter: str,
    enrich: bool,
    fallback_entries: list[dict] | None = None,
    *,
    annotate: bool = True,
):
    """Stream the labelset as newline-delimited JSON, one label entry per line (S13).

    Encodes one :class:`~vtscore.datasets.labelset.LabeledElement` at a time
    via :meth:`LabelSet.iter_dicts`, wrapped in ``flask.stream_with_context``
    so the buffered ``[e.to_dict() for e in elements]`` list (~50 MB at 100 k
    labels) is never held in memory.  Each line carries the same
    ``is_correction`` / ``custom_metadata`` annotations the buffered response
    would attach, applied in the same order (annotate corrections, drop
    non-corrections under ``label_filter=corrections``, then enrich).
    *fallback_entries* (origin-only labelset elements, already serialised)
    stream after the vote-derived rows through the same annotation pipeline.
    *annotate* is off for the detector-scoped export, whose rows belong to a
    detector the request's live Find session says nothing about.

    The top-level ``available_columns`` list has no place in NDJSON: it's a
    whole-set aggregate that can't be emitted before the last row is seen, and
    consumers of a streamed export derive columns from the rows themselves.
    """
    import json
    from itertools import chain

    from flask import Response, stream_with_context

    annotator = _make_correction_annotator(all_medias) if annotate else None
    enricher = _make_enricher(all_medias) if enrich else None
    corrections_only = label_filter == "corrections"

    def generate():
        for entry in chain(labelset.iter_dicts(), fallback_entries or ()):
            if annotator is not None:
                annotator(entry)
            if corrections_only and not entry.get("is_correction"):
                continue
            if enricher is not None:
                enricher(entry)
            yield json.dumps(entry) + "\n"

    return Response(
        stream_with_context(generate()),
        mimetype="application/x-ndjson",
    )


@labels_bp.route("/api/labels/import", methods=["POST"])
@labels_bp.arguments(LabelsImportRequestSchema)
@labels_bp.response(200, LabelsImportResponseSchema)
@require_dataset_header
@require_detector_header
def import_labels(body: dict):
    """Import labels from JSON, matching medias by origin+origin_name (MD5 fallback)."""
    labels = body["labels"]

    origin_lookup, md5_lookup, _ = cached_media_lookups()

    applied = 0
    skipped = 0
    for entry in labels:
        label = entry.get("label")
        if label not in ("good", "bad"):
            skipped += 1
            continue
        cids = resolve_media_ids(entry, origin_lookup, md5_lookup)
        if not cids:
            skipped += 1
            continue

        # Round-trip region_box on good-vote imports.  ``LabeledElement.from_dict``
        # already coerces list↔tuple, so we just check shape and pass through.
        rb_raw = entry.get("region_box") if label == "good" else None
        region_box: tuple[float, float, float, float] | None = None
        if isinstance(rb_raw, (list, tuple)) and len(rb_raw) == 4 and all(isinstance(v, (int, float)) for v in rb_raw):
            region_box = (float(rb_raw[0]), float(rb_raw[1]), float(rb_raw[2]), float(rb_raw[3]))

        for cid in cids:
            # Importing a labelset is a bulk action, not consecutive individual
            # hand-clicks: credit the other vote achievements but not the
            # Marathoner streak.
            apply_label(
                cid,
                label,
                region_box=region_box,
                count_streak=False,
                provenance=read_provenance(entry.get("metadata")) or {"flow": "import"},
            )
        applied += 1

    from vtscore.detectors.label_sync import sync_labels_to_loaded_detector

    sync_labels_to_loaded_detector()

    from vtscore.labels.sync import sync_to_labelset_source

    sync_to_labelset_source()

    return {"applied": applied, "skipped": skipped}


def _partition_candidates(
    sort_results: list[dict],
    thresh: float,
    sides: str,
    snap_good: dict[int, None],
    snap_bad: dict[int, None],
    snap_medias: dict,
) -> tuple[list[dict], list[dict]]:
    """Split sort results into (good, bad) candidate lists by threshold.

    Skips entries that are missing an id/score, carry a non-numeric score,
    are already voted (in *snap_good* / *snap_bad*), or are absent from
    *snap_medias*.  An entry scoring at or above *thresh* becomes a good
    candidate; below it, a bad candidate.  The *sides* selector then clears
    whichever list isn't wanted (``"good"`` drops bads, ``"bad"`` drops
    goods, ``"both"`` keeps both).
    """
    good_candidates = []
    bad_candidates = []
    for entry in sort_results:
        cid = entry.get("id")
        score = entry.get("score", entry.get("similarity"))
        if cid is None or score is None:
            continue
        if not isinstance(score, (int, float)):
            continue
        if cid in snap_good or cid in snap_bad:
            continue
        if cid not in snap_medias:
            continue
        if score >= thresh:
            good_candidates.append({"id": cid, "score": float(score)})
        else:
            bad_candidates.append({"id": cid, "score": float(score)})

    if sides == "good":
        bad_candidates = []
    elif sides == "bad":
        good_candidates = []
    # "both" keeps both lists
    return good_candidates, bad_candidates


@labels_bp.route("/api/labels/fill-from-sort", methods=["POST"])
@labels_bp.arguments(FillFromSortRequestSchema)
@labels_bp.response(200, FillFromSortResponseSchema)
@require_dataset_header
@require_detector_header
def fill_labels_from_sort(body: dict):
    """Fill labels from the current sort results.

    Assigns Good/Bad labels to currently-unlabeled medias based on their
    position relative to the sort threshold. With ``confirm=false`` (the
    default), returns counts only; with ``confirm=true``, applies the
    labels and returns the resulting data as a results dict suitable for
    any exporter.
    """
    sort_results = body["sort_results"]
    thresh = body["threshold"]
    sides = body["sides"]
    confirm = body["confirm"]

    # Atomic snapshot so the membership checks below use the same dataset's
    # cid space as the medias dict; a concurrent rehydrate on the detector
    # against a different dataset can't make us think an A-cid is "already
    # voted" when in fact we're scoring against B's medias.
    vote_snap = validated_vote_snapshot()
    snap_good = vote_snap.good_votes
    snap_bad = vote_snap.bad_votes
    snap_medias = vote_snap.medias

    # Find unlabeled medias above/below threshold
    good_candidates, bad_candidates = _partition_candidates(
        sort_results, thresh, sides, snap_good, snap_bad, snap_medias
    )

    if not confirm:
        return {
            "good_count": len(good_candidates),
            "bad_count": len(bad_candidates),
        }

    # Snapshot the vote state we're about to mutate so a failed persistence
    # pass can roll it back (mirroring ``apply_and_retrain``, audit H30):
    # without the rollback the 500 below tells the user the labels were NOT
    # committed while they stay live in memory and get silently persisted by
    # the next vote-triggered sync.
    from vtscore.state.core import _state_lock, get_active_detector_context

    det_ctx = get_active_detector_context()
    saved_good_votes = dict(det_ctx.good_votes)
    saved_bad_votes = dict(det_ctx.bad_votes)
    saved_region_boxes = dict(det_ctx.vote_region_boxes)
    saved_provenance = dict(det_ctx.vote_provenance)
    saved_history = list(det_ctx.label_history)
    saved_click_times = dict(det_ctx.vote_click_times)
    saved_click_counter = det_ctx.click_counter

    # Apply labels
    # Fill-from-sort takes a whole window off the head of the current sort -
    # the top-of-list draw the selection-bias study measured as the unsafe
    # one - so it is recorded as such, with each item's own sort score.
    for entry in good_candidates:
        apply_label_with_click_time(
            entry["id"],
            "good",
            provenance={"flow": "bulk", "select_mode": "top", "score_at_vote": entry["score"]},
        )

    for entry in bad_candidates:
        apply_label_with_click_time(
            entry["id"],
            "bad",
            provenance={"flow": "bulk", "select_mode": "top", "score_at_vote": entry["score"]},
        )

    # Persist labels to disk BEFORE building the response.  Letting a
    # silent disk-write failure here fall through to ``return {...}`` is
    # the C11 bug; the UI would treat the labels as committed while
    # ``detectors/<name>.json`` never received them.  ``sync_to_labelset_source``
    # is fire-and-forget by design (debounced background timer), so we
    # only guard against the unlikely synchronous scheduling failure.
    from vtscore.detectors.label_sync import sync_labels_to_loaded_detector
    from vtscore.labels.sync import sync_to_labelset_source

    try:
        sync_labels_to_loaded_detector()
    except Exception as exc:
        with _state_lock:
            det_ctx.good_votes.clear()
            det_ctx.good_votes.update(saved_good_votes)
            det_ctx.bad_votes.clear()
            det_ctx.bad_votes.update(saved_bad_votes)
            det_ctx.vote_region_boxes.clear()
            det_ctx.vote_region_boxes.update(saved_region_boxes)
            det_ctx.vote_provenance.clear()
            det_ctx.vote_provenance.update(saved_provenance)
            det_ctx.label_history.clear()
            det_ctx.label_history.extend(saved_history)
            det_ctx.vote_click_times.clear()
            det_ctx.vote_click_times.update(saved_click_times)
            det_ctx.click_counter = saved_click_counter
        logger.exception("fill_labels_from_sort: detector label sync failed")
        abort(500, message=f"Failed to persist labels to detector store: {exc}")

    try:
        sync_to_labelset_source()
    except Exception:
        logger.exception("fill_labels_from_sort: labelset source scheduling failed")

    # Build a results dict compatible with exporters.  Reuse the snapshot
    # taken at the top so the hit dicts reference the same dataset's media
    # entries we used for membership checks.
    good_hits = [
        build_media_hit(e["id"], snap_medias.get(e["id"], {}), e["score"], label="good") for e in good_candidates
    ]
    bad_hits = [build_media_hit(e["id"], snap_medias.get(e["id"], {}), e["score"], label="bad") for e in bad_candidates]

    media_type = "unknown"
    for media in snap_medias.values():
        media_type = media.get("media_type", "unknown")
        break

    results_dict = {
        "media_type": media_type,
        "detectors_run": 1,
        "results": {
            "fill_from_sort": {
                "detector_name": "fill_from_sort",
                "threshold": round(thresh, 4),
                "total_hits": len(good_hits),
                "hits": good_hits,
                "negative_hits": bad_hits,
            },
        },
    }

    return {
        "good_applied": len(good_candidates),
        "bad_applied": len(bad_candidates),
        "results": results_dict,
    }
