"""Global state management for clips and votes."""

from typing import Any

# Clips storage: id -> {id, type, duration, file_size, embedding, wav_bytes, video_bytes}
clips: dict[int, dict[str, Any]] = {}

# Voting storage (OrderedDict behavior via dict in Python 3.7+)
good_votes: dict[int, None] = {}
bad_votes: dict[int, None] = {}

# Inclusion setting: -10 to +10, default 0
inclusion: int = 0


def clear_votes() -> None:
    """Clear all votes."""
    good_votes.clear()
    bad_votes.clear()


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
