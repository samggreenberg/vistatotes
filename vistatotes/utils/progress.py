"""Progress tracking for long-running operations."""

import threading
from typing import Any, Optional

# Progress tracking for long-running operations
progress_lock = threading.Lock()
progress_data = {
    "status": "idle",  # idle, loading, downloading, embedding
    "message": "",
    "current": 0,
    "total": 0,
    "error": None,
}


def update_progress(
    status: str,
    message: str = "",
    current: int = 0,
    total: int = 0,
    error: Optional[str] = None,
) -> None:
    """Update the global progress tracker in a thread-safe manner.

    All write access to ``progress_data`` is serialised with ``progress_lock``
    so that background threads can safely report progress while the Flask
    request thread polls :func:`get_progress`.

    Args:
        status: Current operation phase. Recognised values:
            ``"idle"``, ``"loading"``, ``"downloading"``, ``"embedding"``.
        message: Human-readable description of what is currently happening
            (e.g. ``"Embedding clip 12/200..."``). Defaults to ``""``.
        current: Number of units completed so far (e.g. bytes downloaded,
            clips embedded). Defaults to 0.
        total: Total number of units expected (e.g. total bytes, total clips).
            A value of 0 indicates the total is unknown. Defaults to 0.
        error: If the operation failed, a string describing the error;
            otherwise ``None``. Defaults to ``None``.
    """
    with progress_lock:
        progress_data["status"] = status
        progress_data["message"] = message
        progress_data["current"] = current
        progress_data["total"] = total
        progress_data["error"] = error


def get_progress() -> dict[str, Any]:
    """Return a snapshot of the current progress data.

    Acquires ``progress_lock`` before reading to ensure a consistent view
    across threads.

    Returns:
        A shallow copy of ``progress_data`` as a plain dict with the keys:

        - ``"status"`` (``str``): Current operation phase
          (``"idle"``, ``"loading"``, ``"downloading"``, or ``"embedding"``).
        - ``"message"`` (``str``): Human-readable status description.
        - ``"current"`` (``int``): Units completed so far.
        - ``"total"`` (``int``): Total units expected (0 if unknown).
        - ``"error"`` (``Optional[str]``): Error message if the last operation
          failed, otherwise ``None``.
    """
    with progress_lock:
        return dict(progress_data)


# ---------------------------------------------------------------------------
# Sort-specific progress tracking
# ---------------------------------------------------------------------------

sort_progress_lock = threading.Lock()
sort_progress_data: dict[str, Any] = {
    "status": "idle",  # idle, sorting
    "message": "",
    "current": 0,
    "total": 0,
}


def update_sort_progress(
    status: str,
    message: str = "",
    current: int = 0,
    total: int = 0,
) -> None:
    """Update the sort progress tracker in a thread-safe manner."""
    with sort_progress_lock:
        sort_progress_data["status"] = status
        sort_progress_data["message"] = message
        sort_progress_data["current"] = current
        sort_progress_data["total"] = total


def get_sort_progress() -> dict[str, Any]:
    """Return a snapshot of the current sort progress data."""
    with sort_progress_lock:
        return dict(sort_progress_data)
