"""Global state management for clips and votes."""

from typing import Any

# Clips storage: id -> {id, type, duration, file_size, embedding, wav_bytes, video_bytes}
clips: dict[int, dict[str, Any]] = {}

# Voting storage (OrderedDict behavior via dict in Python 3.7+)
good_votes: dict[int, None] = {}
bad_votes: dict[int, None] = {}

# Combined label history: [(clip_id, label, timestamp), ...]
# Tracks the order of all labels across both categories
label_history: list[tuple[int, str, float]] = []

# Inclusion setting: -10 to +10, default 0
inclusion: int = 0

# Favorite detectors: name -> {name, media_type, weights, threshold, created_at}
favorite_detectors: dict[str, dict[str, Any]] = {}


def clear_votes() -> None:
    """Clear all votes and the full label history.

    Removes all entries from ``good_votes``, ``bad_votes``, and
    ``label_history`` in place. Does not affect the ``clips`` dict.
    Also clears the progress model cache.
    """
    from vistatotes.models.progress import clear_progress_cache

    good_votes.clear()
    bad_votes.clear()
    label_history.clear()
    clear_progress_cache()


def clear_clips() -> None:
    """Clear all loaded clips from memory.

    Removes all entries from the ``clips`` dict in place. Does not affect
    votes or label history. Also clears the progress model cache since
    cached models reference clip embeddings.
    """
    from vistatotes.models.progress import clear_progress_cache

    clips.clear()
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

    Returns:
        An integer in ``[-10, 10]`` representing the inclusion bias. Positive
        values cause the learned sort model to include more items (higher
        recall); negative values cause it to include fewer (higher precision).
    """
    return inclusion


def set_inclusion(value: int) -> None:
    """Set the global inclusion value.

    Also clears the progress model cache since cached models were trained
    with the old inclusion value.

    Args:
        value: New inclusion setting. Should be an integer in ``[-10, 10]``.
            Values outside this range are accepted but may produce unexpected
            results in model training weight calculations.
    """
    global inclusion
    if value != inclusion:
        from vistatotes.models.progress import clear_progress_cache

        clear_progress_cache()
    inclusion = value


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
