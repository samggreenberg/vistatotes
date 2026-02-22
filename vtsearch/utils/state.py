"""Global state management for clips and votes."""

import json
from typing import Any

# Clips storage: id -> {id, type, duration, file_size, embedding, wav_bytes, video_bytes}
clips: dict[int, dict[str, Any]] = {}

# Voting storage (OrderedDict behavior via dict in Python 3.7+)
good_votes: dict[int, None] = {}
bad_votes: dict[int, None] = {}

# Combined label history: [(clip_id, label, timestamp), ...]
# Tracks the order of all labels across both categories
label_history: list[tuple[int, str, float]] = []

# Inclusion setting: -10 to +10, default 0.
# ``None`` means "not yet loaded"; on first access the value is read from the
# persisted settings file so that it survives restarts.
inclusion: int | None = None

# Favorite detectors: name -> {name, media_type, weights, threshold, created_at}
favorite_detectors: dict[str, dict[str, Any]] = {}

# Favorite extractors: name -> {name, extractor_type, media_type, config, created_at}
favorite_extractors: dict[str, dict[str, Any]] = {}

# Dataset creation info: records how the current dataset was created
# (importer name, field values, CLI args).  ``None`` when no dataset is loaded.
dataset_creation_info: dict[str, Any] | None = None


def clear_votes() -> None:
    """Clear all votes and the full label history.

    Removes all entries from ``good_votes``, ``bad_votes``, and
    ``label_history`` in place. Does not affect the ``clips`` dict.
    Also clears the progress model cache.
    """
    from vtsearch.models.progress import clear_progress_cache

    good_votes.clear()
    bad_votes.clear()
    label_history.clear()
    clear_progress_cache()


def clear_clips() -> None:
    """Clear all loaded clips from memory.

    Removes all entries from the ``clips`` dict in place. Does not affect
    votes or label history. Also clears the progress model cache since
    cached models reference clip embeddings.  Also clears
    :data:`dataset_creation_info`.
    """
    global dataset_creation_info
    from vtsearch.models.progress import clear_progress_cache

    clips.clear()
    dataset_creation_info = None
    clear_progress_cache()


def clear_all() -> None:
    """Clear all clips, votes, and label history.

    Convenience wrapper that calls :func:`clear_clips` followed by
    :func:`clear_votes`.
    """
    clear_clips()
    clear_votes()


def get_inclusion() -> int:
    """Return the current inclusion setting.

    On first call the value is loaded from the persisted settings file so
    that it survives app restarts.

    Returns:
        An integer in ``[-10, 10]`` representing the inclusion bias. Positive
        values cause the learned sort model to include more items (higher
        recall); negative values cause it to include fewer (higher precision).
    """
    global inclusion
    if inclusion is None:
        from vtsearch import settings

        inclusion = settings.get_inclusion()
    return inclusion


def set_inclusion(value: int) -> None:
    """Set the global inclusion value and persist it to the settings file.

    Also clears the progress model cache since cached models were trained
    with the old inclusion value.

    Args:
        value: New inclusion setting. Should be an integer in ``[-10, 10]``.
            Values outside this range are accepted but may produce unexpected
            results in model training weight calculations.
    """
    global inclusion
    if value != inclusion:
        from vtsearch.models.progress import clear_progress_cache

        clear_progress_cache()
    inclusion = value

    from vtsearch import settings

    settings.set_inclusion(value)


def add_label_to_history(clip_id: int, label: str) -> None:
    """Append a labelling event to the global label history with a timestamp.

    Args:
        clip_id: Integer ID of the clip that was labelled.
        label: The assigned label; should be ``"good"`` or ``"bad"``.
    """
    import time

    label_history.append((clip_id, label, time.time()))


def add_favorite_detector(name: str, media_type: str, weights: dict[str, Any], threshold: float) -> None:
    """Add or overwrite a named favorite detector in the global store.

    If a detector with the same ``name`` already exists it is replaced.

    Args:
        name: Unique human-readable name for the detector (e.g. ``"dog barks"``).
        media_type: The media type the detector was trained on (``"audio"``,
            ``"video"``, ``"image"``, or ``"paragraph"``).
        weights: Dict mapping layer-parameter names (e.g. ``"0.weight"``) to
            lists of float values, representing the serialised MLP state dict.
        threshold: Decision boundary score in ``[0, 1]``. Clips scoring at or
            above this value are classified as positive.
    """
    import time

    favorite_detectors[name] = {
        "name": name,
        "media_type": media_type,
        "weights": weights,
        "threshold": threshold,
        "created_at": time.time(),
    }


def remove_favorite_detector(name: str) -> bool:
    """Remove a named favorite detector from the global store.

    Args:
        name: Name of the detector to remove.

    Returns:
        ``True`` if the detector was found and removed; ``False`` if no
        detector with that name exists.
    """
    if name in favorite_detectors:
        del favorite_detectors[name]
        return True
    return False


def rename_favorite_detector(old_name: str, new_name: str) -> bool:
    """Rename a favorite detector, updating its internal ``"name"`` field.

    The operation is atomic with respect to the dict: the old entry is removed
    and a new entry is created in a single step (no window where neither exists).

    Args:
        old_name: Current name of the detector to rename.
        new_name: Desired new name for the detector.

    Returns:
        ``True`` if the rename succeeded (old name existed and new name was not
        already taken); ``False`` otherwise (no changes are made).
    """
    if old_name in favorite_detectors and new_name not in favorite_detectors:
        favorite_detectors[new_name] = favorite_detectors[old_name].copy()
        favorite_detectors[new_name]["name"] = new_name
        del favorite_detectors[old_name]
        return True
    return False


def get_favorite_detectors() -> dict[str, dict[str, Any]]:
    """Return a shallow copy of all favorite detectors.

    Returns:
        A dict mapping detector name to its data dict (with keys ``"name"``,
        ``"media_type"``, ``"weights"``, ``"threshold"``, ``"created_at"``).
        The returned dict is a copy; mutations to it do not affect the global store.
    """
    return favorite_detectors.copy()


def get_favorite_detectors_by_media(media_type: str) -> dict[str, dict[str, Any]]:
    """Return all favorite detectors matching a given media type.

    Args:
        media_type: Media type to filter by (``"audio"``, ``"video"``,
            ``"image"``, or ``"paragraph"``).

    Returns:
        A dict mapping detector name to its data dict, containing only
        detectors whose ``"media_type"`` field equals ``media_type``.
        The returned dict is a new dict object; mutations do not affect the
        global store.
    """
    return {name: det for name, det in favorite_detectors.items() if det["media_type"] == media_type}


# ---------------------------------------------------------------------------
# Favorite Extractors
# ---------------------------------------------------------------------------


def add_favorite_extractor(name: str, extractor_type: str, media_type: str, config: dict[str, Any]) -> None:
    """Add or overwrite a named favorite extractor in the global store.

    Args:
        name: Unique human-readable name for the extractor (e.g. ``"license plates"``).
        extractor_type: The extractor class identifier (e.g. ``"image_class"``).
        media_type: The media type the extractor operates on (``"image"``, etc.).
        config: Extractor-specific configuration dict (class name, threshold, etc.).
    """
    import time

    favorite_extractors[name] = {
        "name": name,
        "extractor_type": extractor_type,
        "media_type": media_type,
        "config": config,
        "created_at": time.time(),
    }


def remove_favorite_extractor(name: str) -> bool:
    """Remove a named favorite extractor from the global store.

    Returns:
        ``True`` if the extractor was found and removed; ``False`` otherwise.
    """
    if name in favorite_extractors:
        del favorite_extractors[name]
        return True
    return False


def rename_favorite_extractor(old_name: str, new_name: str) -> bool:
    """Rename a favorite extractor.

    Returns:
        ``True`` if the rename succeeded; ``False`` otherwise.
    """
    if old_name in favorite_extractors and new_name not in favorite_extractors:
        favorite_extractors[new_name] = favorite_extractors[old_name].copy()
        favorite_extractors[new_name]["name"] = new_name
        del favorite_extractors[old_name]
        return True
    return False


def get_favorite_extractors() -> dict[str, dict[str, Any]]:
    """Return a shallow copy of all favorite extractors."""
    return favorite_extractors.copy()


def get_favorite_extractors_by_media(media_type: str) -> dict[str, dict[str, Any]]:
    """Return all favorite extractors matching a given media type."""
    return {name: ext for name, ext in favorite_extractors.items() if ext["media_type"] == media_type}


# ---------------------------------------------------------------------------
# Dataset Creation Info
# ---------------------------------------------------------------------------


def set_dataset_creation_info(info: dict[str, Any] | None) -> None:
    """Set the creation info for the currently loaded dataset.

    Args:
        info: A dict with keys ``"importer"``, ``"display_name"``,
            ``"field_values"``, and ``"cli_args"``, or ``None`` to clear.
    """
    global dataset_creation_info
    dataset_creation_info = info


def get_dataset_creation_info() -> dict[str, Any] | None:
    """Return the creation info for the currently loaded dataset, or ``None``."""
    return dataset_creation_info


# ---------------------------------------------------------------------------
# Clip matching helpers (origin+origin_name union with MD5)
# ---------------------------------------------------------------------------


def _origin_key(origin: dict[str, Any], origin_name: str) -> str:
    """Return a hashable string key for an (origin, origin_name) pair."""
    return json.dumps(origin, sort_keys=True) + "\0" + origin_name


def build_clip_lookup(
    clip_dict: dict[int, dict[str, Any]],
) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    """Build lookup tables for matching label entries to clips.

    Returns ``(origin_lookup, md5_lookup)`` where:

    * **origin_lookup** maps ``_origin_key(origin, origin_name)`` to a list of
      clip IDs that share that origin+name pair.
    * **md5_lookup** maps an MD5 hex string to a list of clip IDs whose
      content hash matches.

    Both lookups map to *lists* because the same key can match multiple clips
    (e.g. duplicate files with the same MD5).
    """
    origin_lookup: dict[str, list[int]] = {}
    md5_lookup: dict[str, list[int]] = {}

    for clip in clip_dict.values():
        cid = clip["id"]

        origin = clip.get("origin")
        origin_name = clip.get("origin_name", "")
        if origin is not None and origin_name:
            key = _origin_key(origin, origin_name)
            origin_lookup.setdefault(key, []).append(cid)

        md5 = clip.get("md5", "")
        if md5:
            md5_lookup.setdefault(md5, []).append(cid)

    return origin_lookup, md5_lookup


def resolve_clip_ids(
    entry: dict[str, Any],
    origin_lookup: dict[str, list[int]],
    md5_lookup: dict[str, list[int]],
) -> list[int]:
    """Resolve a label entry to matching clip ID(s).

    Returns the **union** of clips matched by ``origin`` + ``origin_name``
    and clips matched by ``md5``.  Both lookups are always attempted so that
    a label is applied to every element in the dataset that corresponds to
    the entry, regardless of whether it was matched by provenance or by
    content hash.  Duplicate IDs are removed.
    """
    matched: dict[int, None] = {}

    origin = entry.get("origin")
    origin_name = entry.get("origin_name", "")

    if origin is not None and origin_name:
        key = _origin_key(origin, origin_name)
        for cid in origin_lookup.get(key, []):
            matched[cid] = None

    md5 = entry.get("md5", "")
    if md5:
        for cid in md5_lookup.get(md5, []):
            matched[cid] = None

    return list(matched)


def find_missing_entries(
    label_entries: list[dict[str, Any]],
    origin_lookup: dict[str, list[int]],
    md5_lookup: dict[str, list[int]],
) -> list[dict[str, Any]]:
    """Return label entries that do not match any clip by origin+name or md5.

    Only entries with a valid label (``"good"`` or ``"bad"``) are considered;
    entries with invalid labels are silently excluded (they are already counted
    as "skipped" by the caller).
    """
    missing: list[dict[str, Any]] = []
    for entry in label_entries:
        label = entry.get("label", "")
        if label not in ("good", "bad"):
            continue
        cids = resolve_clip_ids(entry, origin_lookup, md5_lookup)
        if not cids:
            missing.append(entry)
    return missing


def next_clip_id(clip_dict: dict[int, dict[str, Any]]) -> int:
    """Return the next available clip ID (one past the current maximum)."""
    if not clip_dict:
        return 1
    return max(clip_dict) + 1
