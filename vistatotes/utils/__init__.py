"""Utility modules for progress tracking and state management."""

from vistatotes.utils.progress import (get_progress, get_sort_progress,
                                       update_progress, update_sort_progress)
from vistatotes.utils.state import (add_favorite_detector,
                                    add_label_to_history, bad_votes, clear_all,
                                    clear_clips, clear_votes, clips,
                                    favorite_detectors, get_favorite_detectors,
                                    get_favorite_detectors_by_media,
                                    get_inclusion, good_votes, inclusion,
                                    label_history, remove_favorite_detector,
                                    rename_favorite_detector, set_inclusion)

__all__ = [
    # Progress
    "update_progress",
    "get_progress",
    "update_sort_progress",
    "get_sort_progress",
    # State
    "clips",
    "good_votes",
    "bad_votes",
    "label_history",
    "inclusion",
    "favorite_detectors",
    "clear_votes",
    "clear_clips",
    "clear_all",
    "get_inclusion",
    "set_inclusion",
    "add_label_to_history",
    "add_favorite_detector",
    "remove_favorite_detector",
    "rename_favorite_detector",
    "get_favorite_detectors",
    "get_favorite_detectors_by_media",
]
