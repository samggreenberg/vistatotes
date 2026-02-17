"""Progress tracking for long-running operations."""

import threading
from typing import Optional

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
    """Update the global progress tracker."""
    with progress_lock:
        progress_data["status"] = status
        progress_data["message"] = message
        progress_data["current"] = current
        progress_data["total"] = total
        progress_data["error"] = error


def get_progress() -> dict:
    """Get the current progress data."""
    with progress_lock:
        return dict(progress_data)
