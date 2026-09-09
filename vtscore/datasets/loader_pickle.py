"""Pickle-based dataset loaders.

Loads datasets from a ZIP container ``.pkl`` file (with optional companion
media-file directory), plus the embedder/clipper metadata readers.  Split
out from :mod:`vtscore.datasets.loader` for navigability.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any, Iterator

from vtscore.datasets.loader_common import ProgressCallback
from vtscore.embedding.normalize import l2_normalize
from vtscore.utils.hashing import content_md5

logger = logging.getLogger(__name__)


def _load_embeddings_dict(media_info: dict[str, Any]) -> dict[str, Any] | None:
    """Build a media's per-embedder ``{name: vector}`` dict, L2-normalising each.

    A v3 pickle carries the dict under ``embeddings`` (one entry per bound
    embedder); a legacy single-vector pickle carries only the singular
    ``embedding`` plus the recorded ``embedder`` name, which is re-keyed into a
    one-entry dict here (the Phase 2c migration — the singular key never enters
    the in-memory media).  Returns ``None`` when no usable vector is present.
    """
    embs = media_info.get("embeddings")
    if isinstance(embs, dict) and embs:
        out: dict[str, Any] = {}
        for name, vec in embs.items():
            if name and vec is not None:
                out[name] = l2_normalize(vec)
        return out or None
    # Legacy single-vector pickle: re-key the singular vector under its
    # recorded embedder name.
    vec = media_info.get("embedding")
    name = media_info.get("embedder")
    if vec is not None and name:
        return {name: l2_normalize(vec)}
    return None


#: Share of the pickle load's wall clock spent streaming + deserialising the
#: container, before the per-item conversion loop starts.  Measured across two
#: dataset shapes (a 243 MB / 20k-item image container and a 328 MB / 105k-item
#: text container): the read phase ran 56% and 43% of the total respectively,
#: so half is the honest split.  It only sets where the read hands the bar over
#: to the item loop; both sub-phases advance the same fraction either way, so a
#: drift of a few points costs pacing smoothness, never correctness.
_READ_SHARE = 0.5


class _PickleLoadProgress:
    """Composites a pickle load's two sub-phases onto one monotone bar.

    Streaming/deserialising the container and converting each media measure
    different things — bytes consumed, then items built — so reporting each on
    its native scale would rewind the fraction to zero at the handover.  Every
    consumer clamps monotonically (``ProgressTracker._compute_overall`` pins
    ``overall``, ``AdaptiveLoadPacer.update`` pins ``_frac``), so that rewind
    never shows as a retreat; it shows as the bar *freezing* for the whole item
    loop.  Lighting up the read at the cost of deadening the conversion is not
    a trade worth making, so both report against one denominator — the
    ``medias.pkl`` byte size — with the read owning ``[0, _READ_SHARE]`` and
    the item loop the rest.  The human-readable byte and item counts stay in
    the message, where they are not load-bearing for the bar.
    """

    def __init__(self, on_progress: ProgressCallback, name: str) -> None:
        self._on_progress = on_progress
        self._name = name
        self._total = 0

    def on_read(self, read_bytes: int, total_bytes: int) -> None:
        """Report the streaming read, and latch its denominator for the loop."""
        self._total = total_bytes
        if total_bytes > 0:
            self._on_progress("loading", f"Reading {self._name}…", int(_READ_SHARE * read_bytes), total_bytes)

    def on_item(self, loaded: int, total_count: int) -> None:
        """Report the conversion loop, continuing the read's scale."""
        message = f"Processing {loaded} of {total_count} items…"
        if self._total <= 0 or total_count <= 0:
            # Nothing to continue from (an unreadable entry size, or an empty
            # dataset): fall back to the item scale on its own.
            self._on_progress("loading", message, loaded, total_count)
            return
        done = _READ_SHARE + (1.0 - _READ_SHARE) * (loaded / total_count)
        self._on_progress("loading", message, int(self._total * done), self._total)


def _read_pickle_dataset(file_path: Path, on_progress: Any = None) -> dict[str, Any]:
    """Load a dataset ZIP container and assert the ``"medias"`` envelope.

    Translates :class:`MemoryError` into a contextual message and raises
    :class:`ValueError` when the file does not contain a dict with a
    ``"medias"`` key.  *on_progress* is forwarded to
    :func:`~vtscore.datasets.container.read_container` as
    ``on_progress(bytes_read, total_bytes)``.
    """
    from vtscore.datasets.container import read_container

    try:
        data, _meta = read_container(file_path, on_progress=on_progress)
    except MemoryError:
        gc.collect()
        raise MemoryError(
            f"Out of memory while reading {file_path.name}. The pickle file is too large for available RAM."
        )
    return data


def _build_pickle_dir_maps() -> tuple[dict[str, str], dict[str, list[str]]]:
    """Build (dir_keys, extra_fields) maps keyed by media type id."""
    from vtscore.media import all_types  # noqa: PLC0415

    dir_keys: dict[str, str] = {}
    extra_fields: dict[str, list[str]] = {}
    for mt in all_types():
        dir_keys[mt.type_id] = mt.dir_key
        extra_fields[mt.type_id] = mt.pickle_extra_fields
    return dir_keys, extra_fields


def _resolve_thin_media_path(
    media_type: str,
    media_info: dict[str, Any],
    data: dict[str, Any],
    dir_keys: dict[str, str],
) -> str | None:
    """Resolve ``media_path`` for thin-mode pickle loads.

    Returns the path stored on the media (if any) or a probe of the
    pickle's external directory entry for the media's type.
    """
    media_path = media_info.get("media_path")
    if media_path:
        return media_path
    dir_key = dir_keys.get(media_type)
    if not dir_key or dir_key not in data or "filename" not in media_info:
        return None
    candidate = Path(data[dir_key]) / media_info["filename"]
    if candidate.exists():
        return str(candidate.resolve())
    return None


def _has_external_byte_source(media_info: dict[str, Any]) -> bool:
    """Return ``True`` when a media's bytes re-derive from outside the pickle.

    A full-mode load resolves bytes up front from the pickle's inline payload
    or a file on disk, but some media carry neither and are still perfectly
    serveable: an *archive member* streams its bytes from a byte range inside a
    tar/zip shard we deliberately never extract (``local_archive_member``, e.g.
    audio tiles cut from tar shards), and a *URL-backed* media fetches them
    from ``media_url`` (an importer's thin mode).  Both
    re-resolve on demand in
    :meth:`~vtscore.media.base.MediaType._resolve_media_bytes`, so such an
    entry must be kept lazily rather than dropped as unresolvable.
    """
    from vtscore.datasets.archive_stream import archive_member_ref  # noqa: PLC0415

    if archive_member_ref(media_info) is not None:
        return True
    return bool(media_info.get("media_url"))


def _load_pickle_media_payload(
    media_type: str,
    media_info: dict[str, Any],
    data: dict[str, Any],
    dir_keys: dict[str, str],
) -> tuple[bytes | None, str | None, str | None, bool]:
    """Resolve ``(media_bytes, media_string, media_path, missing)`` for full mode.

    Returns ``missing=True`` when the media references an external file
    that does not exist on disk.  ``media_bytes`` may still be ``None``
    when neither inline nor external content was found.
    """
    bytes_val = media_info.get("media_bytes")
    string_val = media_info.get("media_string")
    if bytes_val is not None:
        return bytes_val, None, None, False
    if string_val is not None:
        return string_val.encode("utf-8"), string_val, None, False

    dir_key = dir_keys.get(media_type)
    if not dir_key or dir_key not in data or "filename" not in media_info:
        return None, None, None, False
    ext_path = Path(data[dir_key]) / media_info["filename"]
    if not ext_path.exists():
        return None, None, None, True
    if ext_path.suffix in (".txt", ".md"):
        raw = ext_path.read_bytes()
        try:
            txt = raw.decode("utf-8")
        except UnicodeDecodeError:
            logger.warning(
                "%s is not valid UTF-8; falling back to latin-1 encoding. Non-Latin characters may be corrupted.",
                ext_path,
            )
            txt = raw.decode("latin-1")
        return txt.encode("utf-8"), txt, str(ext_path.resolve()), False
    with open(ext_path, "rb") as f:
        return f.read(), None, str(ext_path.resolve()), False


def _restore_signpost_text(media_data: dict[str, Any], media_info: dict[str, Any]) -> None:
    """Carry a pickled signpost text (+ its signature and kind) onto the media.

    Written at ingest by the signpost-texts stage; restoring it is what makes
    a later browse (or Find→Browse re-fit) skip the caption/tag models, and
    what keeps the item's "AI Caption" / "AI Tags" metadata row across a
    reload.  Absent fields are left off entirely rather than set to ``None``,
    so an unprepped dataset's medias keep the shape they had before.
    """
    from vtscore.projection.signpost_texts import PERSISTED_FIELDS  # noqa: PLC0415

    for field in PERSISTED_FIELDS:
        value = media_info.get(field)
        if value:
            media_data[field] = value


def _restore_original_payload(media_data: dict[str, Any], media_info: dict[str, Any]) -> None:
    """Carry a pickled pre-clean payload snapshot onto the media.

    Written by the chain runner when a :class:`~vtscore.media.cleaner.MediaCleaner`
    rewrote the item at load time (``docs/plans/media-cleaners.md``); it is the
    only copy of what the user imported, so it has to survive the round-trip or
    the item silently loses its Clean/Original toggle.  Absent fields are left
    off entirely rather than set to ``None``, so an uncleaned dataset's medias
    keep the shape they had before.
    """
    from vtscore.datasets.clipper_chain import ORIGINAL_PAYLOAD_KEYS  # noqa: PLC0415

    for field in ORIGINAL_PAYLOAD_KEYS:
        value = media_info.get(field)
        if value is not None:
            media_data[field] = value


def _restore_media_url(media_data: dict[str, Any], media_info: dict[str, Any]) -> None:
    """Carry a pickled ``media_url`` onto the media when one is recorded.

    URL-backed media keep the remote URL as their
    byte source; it is what lets a reloaded item still serve content when the
    pickle holds no inline bytes and no local file.  Set only when present, so
    the far more common file-backed media keep the shape they had before.
    """
    url = media_info.get("media_url")
    if url:
        media_data["media_url"] = url


def _restore_clip_window(media_data: dict[str, Any], media_info: dict[str, Any]) -> None:
    """Carry a pickled clip window onto the media when one is recorded.

    ``clip_start`` / ``clip_end`` / ``clip_index`` / ``clip_box`` are the
    playback window of a clipped or windowed item: the players seek to
    ``clip_start`` and loop within ``[clip_start, clip_end]``, the audio
    waveform is sliced by the same pair, and ``display_metadata`` renders them
    as the item's "Clip …" rows.  Restoring them is what keeps each window of a
    shared source distinct after a reload; without it a manifest that windows
    one tar member N times reloads as N items that all play (and draw) the
    whole member from 0.  Set only when present, so an unclipped media keeps
    the shape it had before.
    """
    from vtscore.media.provenance import CLIP_WINDOW_FIELDS  # noqa: PLC0415

    for field in CLIP_WINDOW_FIELDS:
        value = media_info.get(field)
        if value is not None:
            media_data[field] = value


def _build_pickle_thin_media(
    new_id: int,
    media_info: dict[str, Any],
    media_type: str,
    media_path: str | None,
    extra_fields: list[str],
) -> dict[str, Any]:
    """Build a thin-mode media dict from a pickle entry (no bytes loaded)."""
    fname = media_info.get("filename", f"media_{new_id}.{media_type}")
    media_data: dict[str, Any] = {
        "id": new_id,
        "media_type": media_type,
        "embedder": media_info.get("embedder", ""),
        "duration": media_info.get("duration", 0),
        "file_size": media_info.get("file_size", 0),
        "md5": media_info.get("md5", ""),
        "media_bytes": None,
        "media_string": None,
        "embeddings": _load_embeddings_dict(media_info),
        "media_path": media_path,
        "filename": fname,
        "category": media_info.get("category", "unknown"),
        "origin": media_info.get("origin"),
        "origin_name": media_info.get("origin_name", fname),
    }
    for field in extra_fields:
        media_data[field] = media_info.get(field)
    _restore_media_url(media_data, media_info)
    cm = media_info.get("custom_metadata")
    if cm:
        media_data["custom_metadata"] = cm
    _restore_signpost_text(media_data, media_info)
    _restore_original_payload(media_data, media_info)
    _restore_clip_window(media_data, media_info)
    return media_data


def _build_pickle_full_media(
    new_id: int,
    media_info: dict[str, Any],
    media_type: str,
    media_bytes: bytes,
    media_string: str | None,
    media_path: str | None,
    extra_fields: list[str],
) -> dict[str, Any]:
    """Build a full-mode media dict from a pickle entry."""
    fname = media_info.get("filename", f"media_{new_id}.{media_type}")
    media_data = {
        "id": new_id,
        "media_type": media_type,
        "embedder": media_info.get("embedder", ""),
        "duration": media_info.get("duration", 0),
        "file_size": media_info.get("file_size", len(media_bytes)),
        "md5": media_info.get("md5") or content_md5(media_bytes),
        "embeddings": _load_embeddings_dict(media_info),
        "media_bytes": media_bytes,
        "media_string": media_string,
        "media_path": media_path or media_info.get("media_path"),
        "filename": fname,
        "category": media_info.get("category", "unknown"),
        "origin": media_info.get("origin"),
        "origin_name": media_info.get("origin_name", fname),
    }
    for field in extra_fields:
        media_data[field] = media_info.get(field)
    _restore_media_url(media_data, media_info)
    cm = media_info.get("custom_metadata")
    if cm:
        media_data["custom_metadata"] = cm
    _restore_signpost_text(media_data, media_info)
    _restore_original_payload(media_data, media_info)
    _restore_clip_window(media_data, media_info)
    return media_data


def _convert_one_pickle_media(
    new_id: int,
    media_info: dict[str, Any],
    thin: bool,
    data: dict[str, Any],
    dir_keys: dict[str, str],
    extra_fields_map: dict[str, list[str]],
) -> tuple[dict[str, Any] | None, bool]:
    """Convert one pickle media entry to the app's media format.

    Returns ``(media_data, missing)``.  ``media_data`` is ``None`` when
    the entry is unusable (no usable embedding, or full mode without
    bytes); ``missing`` is ``True`` when the entry was skipped because
    its embedding or external-file reference could not be resolved
    (used to bump the "missing media" warning counter).
    """
    media_type = media_info.get("media_type", "audio")
    extra_fields = extra_fields_map.get(media_type, [])

    # No usable vector → skip the entry.  A v3 pickle carries the per-embedder
    # ``embeddings`` dict; a legacy pickle carries the singular ``embedding`` +
    # ``embedder`` name (re-keyed by :func:`_load_embeddings_dict`).  Resolving
    # the dict once here also guards against an explicit ``None`` vector that
    # would otherwise become a 0-d ``dtype=object`` array and poison every
    # embedding-matrix consumer downstream.
    if _load_embeddings_dict(media_info) is None:
        return None, True

    if thin:
        media_path = _resolve_thin_media_path(media_type, media_info, data, dir_keys)
        if media_path or _has_external_byte_source(media_info):
            return _build_pickle_thin_media(new_id, media_info, media_type, media_path, extra_fields), False
        # Nothing outside the pickle can reproduce this entry's bytes: no file
        # on disk, no archive member, no URL.  Thin's contract is "hold a
        # reference instead of the payload", and here there is no reference to
        # hold - dropping ``media_bytes`` would not defer the read, it would
        # destroy the only copy, leaving the media permanently unembeddable.
        # Anything that needs pixels later (a converter route, a re-clip, an
        # embedder the pickle has no vector for) then fails, and the media is
        # silently skipped at scoring - which also shrinks the haystack the
        # threshold is fitted on, so the cut moves too (issue #3556).  Keep the
        # inline payload for these entries; the memory thin would have saved
        # here was never available without losing data.
        media_bytes, media_string, _path, _missing = _load_pickle_media_payload(media_type, media_info, data, dir_keys)
        if media_bytes is None:
            return _build_pickle_thin_media(new_id, media_info, media_type, media_path, extra_fields), False
        return (
            _build_pickle_full_media(new_id, media_info, media_type, media_bytes, media_string, None, extra_fields),
            False,
        )

    media_bytes, media_string, media_path, missing = _load_pickle_media_payload(
        media_type,
        media_info,
        data,
        dir_keys,
    )
    if media_bytes is None:
        # Reference dataset (imported with reference_files / thin): the pickle
        # carries no inline bytes, but a stored ``media_path`` may still point
        # at the original file on disk.  Keep the media lazy (load it thin)
        # rather than dropping it, so a reference dataset survives the
        # registry save → full-mode reopen round-trip.  ``_resolve_media_bytes``
        # reads the bytes on demand from ``media_path`` at serve/embed time.
        ref_path = _resolve_thin_media_path(media_type, media_info, data, dir_keys)
        if ref_path and Path(ref_path).exists():
            return _build_pickle_thin_media(new_id, media_info, media_type, ref_path, extra_fields), False
        # Bytes that live outside the pickle *and* outside the filesystem - an
        # archive member inside an unextracted tar/zip shard, or a URL-backed
        # media - re-resolve on demand at serve time.  Keep the entry lazy too,
        # so an archive-member/clip dataset (whose items never have a
        # standalone file) survives the registry save -> full-mode reopen
        # instead of reloading as an empty dataset.
        if _has_external_byte_source(media_info):
            return _build_pickle_thin_media(new_id, media_info, media_type, ref_path, extra_fields), False
        # A reference whose file has gone counts as missing so the load-time
        # warning fires, even when there is no companion dir for
        # ``_load_pickle_media_payload`` to flag it against.
        if media_info.get("media_path"):
            return None, True
        return None, missing
    return (
        _build_pickle_full_media(
            new_id,
            media_info,
            media_type,
            media_bytes,
            media_string,
            media_path,
            extra_fields,
        ),
        False,
    )


def load_dataset_from_pickle(
    file_path: Path,
    medias: dict[int, dict[str, Any]],
    thin: bool = False,
    on_progress: ProgressCallback | None = None,
) -> dict[str, Any] | None:
    """Load a dataset from a pickle file into the medias dict in-place.

    The pickle must contain a dict with a ``"medias"`` key mapping media IDs
    to media data dicts.  It may also include ``"audio_dir"``, ``"video_dir"``,
    ``"image_dir"``, or ``"text_dir"`` keys pointing to directories containing
    raw media files when the bytes are not stored inline.

    If media bytes are not stored inline in the pickle, the function attempts to
    load them from the companion directory entry in the pickle. Medias for which
    no bytes can be resolved - or whose ``embedding`` field is missing or
    ``None`` - are silently skipped (a warning is printed to stdout after
    loading).  Skipping ``None`` embeddings is important: ``np.array(None)``
    yields a 0-d ``dtype=object`` array that downstream consumers cannot
    distinguish from a real vector until they crash.

    The ``medias`` dict is cleared before loading begins.

    Args:
        file_path: Path to a ``.pkl`` file previously created by
            :func:`export_dataset_to_file` or :func:`load_demo_dataset`.
        medias: Dict to populate in-place. Existing entries are removed before
            loading. Keys are media IDs (int); values are media data dicts.
        thin: When ``True``, skip loading media bytes into memory.  Inline
            bytes from the pickle are discarded and external-dir files are
            referenced by ``media_path`` instead of read.  Useful for CLI
            workflows that only need embeddings for scoring.

    Returns:
        The cached ``"coverage_atlas"`` payload (a plain-dict snapshot written
        by :func:`export_dataset_to_file`) when the pickle carries one, else
        ``None``.  Callers can hand it to
        :func:`vtscore.state.coverage.restore_coverage_atlas_from_cache` to
        skip rebuilding the coverage atlas.
    """
    reporter = _PickleLoadProgress(on_progress, file_path.name) if on_progress is not None else None
    if on_progress is not None:
        on_progress("loading", f"Reading {file_path.name}…", 0, 0)

    data = _read_pickle_dataset(file_path, reporter.on_read if reporter is not None else None)
    medias.clear()
    cached_coverage_atlas = data.get("coverage_atlas")
    medias_data = data["medias"]
    dir_keys, extra_fields_map = _build_pickle_dir_maps()

    missing_media = 0
    loaded_count = 0
    total_count = len(medias_data)
    # ~50 updates for the whole loop, whatever the item count.  The old
    # ``min(50, total // 50)`` capped the *interval* at 50 instead of the
    # update count, so it inverted above 2500 items: a 300k-item dataset
    # emitted 6,000 progress events, each one a full re-serialisation of the
    # task list pushed to every open SSE stream (enough to back up the
    # per-client 1024-deep queue and start dropping frames).
    _progress_interval = max(1, total_count // 50) if total_count > 0 else 1

    if reporter is not None:
        reporter.on_item(0, total_count)

    try:
        for media_id, media_info in medias_data.items():
            media_data, missing = _convert_one_pickle_media(
                media_id,
                media_info,
                thin,
                data,
                dir_keys,
                extra_fields_map,
            )
            if media_data is None:
                if missing:
                    missing_media += 1
                continue
            medias[media_id] = media_data
            loaded_count += 1
            if reporter is not None and loaded_count % _progress_interval == 0:
                reporter.on_item(loaded_count, total_count)
    except MemoryError:
        medias.clear()
        del data
        gc.collect()
        raise MemoryError(
            f"Out of memory after loading {loaded_count} of {total_count} medias from "
            f"{file_path.name}. Try a smaller dataset or free up system RAM."
        )

    # Release the raw pickle data now that medias are built
    del data  # noqa: F821 - ruff cannot see past `del data` in the except branch (which always re-raises)
    gc.collect()

    if missing_media > 0:
        print(f"WARNING: {missing_media} media files missing from {file_path}", flush=True)

    return cached_coverage_atlas


def load_dataset_from_pickle_chunked(
    file_path: Path,
    chunk_size: int,
    thin: bool = False,
) -> Iterator[dict[int, dict[str, Any]]]:
    """Yield chunks of medias from a pickle dataset file.

    Works identically to :func:`load_dataset_from_pickle` but yields the
    medias in groups of at most *chunk_size*.  Each yielded dict is a
    self-contained medias dict with IDs starting at 1.

    The entire pickle is deserialized once (unavoidable for ``.pkl``
    format), but media bytes are dropped or skipped per-chunk so that
    only one chunk's worth of media data is alive at a time.

    Args:
        file_path: Path to a ``.pkl`` dataset file.
        chunk_size: Maximum number of medias per yielded chunk.
        thin: When ``True``, skip loading media bytes into memory.

    Yields:
        A dict mapping int media IDs (starting at 1) to media data dicts.
    """
    data = _read_pickle_dataset(file_path)
    medias_data = data["medias"]
    dir_keys, extra_fields_map = _build_pickle_dir_maps()

    all_media_ids = sorted(medias_data.keys())

    for start in range(0, len(all_media_ids), chunk_size):
        batch_ids = all_media_ids[start : start + chunk_size]
        chunk_medias: dict[int, dict[str, Any]] = {}
        new_id = 1

        for media_id in batch_ids:
            media_info = medias_data[media_id]
            media_data, _missing = _convert_one_pickle_media(
                new_id,
                media_info,
                thin,
                data,
                dir_keys,
                extra_fields_map,
            )
            if media_data is None:
                continue
            chunk_medias[new_id] = media_data
            new_id += 1

        if chunk_medias:
            yield chunk_medias


def _read_pkl_meta_safe(pkl_path: Path) -> dict[str, Any]:
    """Read a container's ``meta.json``, returning ``{}`` if it can't be read.

    A cached ``.pkl`` may be corrupt, truncated, or a legacy non-zip pickle.
    The demo catalog reads metadata from every cached file to report each
    demo's embedder/clipper, so one unreadable file must not raise and take
    down the whole listing — it degrades to "metadata unknown" instead.
    """
    from vtscore.datasets.container import read_meta

    try:
        return read_meta(pkl_path)
    except Exception:
        logger.warning("Could not read container metadata from %s; treating as unknown", pkl_path, exc_info=True)
        return {}


def read_pkl_clipper(pkl_path: Path) -> str | None:
    """Return the clipper name from the container's ``meta.json``, or ``None`` if unreadable."""
    return _read_pkl_meta_safe(pkl_path).get("clipper") or None


def read_pkl_embedder(pkl_path: Path) -> str | None:
    """Return the embedder name from the container's ``meta.json``, or ``None`` if unreadable."""
    return _read_pkl_meta_safe(pkl_path).get("embedder") or None
