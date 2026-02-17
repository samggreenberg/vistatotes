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
    """Clear all votes."""
    good_votes.clear()
    bad_votes.clear()
    label_history.clear()


def clear_clips() -> None:
    """Clear all clips."""
    clips.clear()


def clear_all() -> None:
    """Clear all clips and votes."""
    clear_clips()
    clear_votes()


def get_inclusion() -> int:
    """Get the current inclusion value."""
    return inclusion


def set_inclusion(value: int) -> None:
    """Set the inclusion value."""
    global inclusion
    inclusion = value


def add_label_to_history(clip_id: int, label: str) -> None:
    """Add a label event to the history."""
    import time

    label_history.append((clip_id, label, time.time()))


def add_favorite_detector(name: str, media_type: str, weights: dict, threshold: float) -> None:
    """Add or update a favorite detector."""
    import time

    favorite_detectors[name] = {
        "name": name,
        "media_type": media_type,
        "weights": weights,
        "threshold": threshold,
        "created_at": time.time(),
    }


def remove_favorite_detector(name: str) -> bool:
    """Remove a favorite detector. Returns True if found and removed."""
    if name in favorite_detectors:
        del favorite_detectors[name]
        return True
    return False


def rename_favorite_detector(old_name: str, new_name: str) -> bool:
    """Rename a favorite detector. Returns True if successful."""
    if old_name in favorite_detectors and new_name not in favorite_detectors:
        favorite_detectors[new_name] = favorite_detectors[old_name].copy()
        favorite_detectors[new_name]["name"] = new_name
        del favorite_detectors[old_name]
        return True
    return False


def get_favorite_detectors() -> dict[str, dict[str, Any]]:
    """Get all favorite detectors."""
    return favorite_detectors.copy()


def get_favorite_detectors_by_media(media_type: str) -> dict[str, dict[str, Any]]:
    """Get favorite detectors filtered by media type."""
    return {name: det for name, det in favorite_detectors.items() if det["media_type"] == media_type}
