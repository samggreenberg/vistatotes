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
