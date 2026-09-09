"""Persistent detector registry.

Maintains a JSON manifest at ``data/detector_registry.json`` that tracks every
detector the user has created.  Each entry stores enough metadata to display
the detector in the dashboard grid.

Every entry is backed by a labelset file under ``data/detectors/``, named
after a slug of the detector name rather than the name itself (see
:func:`vtscore.detectors.store._slug`).
The MLP that scores the detector is trained on demand from the labelset and
lives only in RAM (see :class:`~vtsearch.state.DetectorContext`).

Multiple detectors can be *loaded* into memory simultaneously.  Which detector
the UI interacts with is determined per-request via the ``X-Detector-Id``
header.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable
from typing import Any, TypeVar

from vtscore.config import DATA_DIR
from vtscore.io import atomic_write_json, file_lock

logger = logging.getLogger(__name__)

REGISTRY_PATH = DATA_DIR / "detector_registry.json"

_T = TypeVar("_T")

# Guards the process-local state below: the ``_entries`` cache and the
# ``_loaded_ids`` / ``_loading_ids`` sets.  Cross-process durability of the
# on-disk manifest is handled by :func:`_read_modify_write` under a
# ``file_lock``; this lock only serialises threads within one process.
_lock = threading.RLock()

_entries: list[dict[str, Any]] | None = None
#: ``(mtime_ns, size)`` of the registry file the cache was filled from.
_entries_stamp: tuple[int, int] | None = None

# Set of detector IDs currently loaded in memory (each has a DetectorContext).
_loaded_ids: set[str] = set()

# Detector IDs whose background load is currently in flight.  Gates the
# ``.../load`` handler's check-then-act (the loaded flag is only set at the end
# of the loader, so two concurrent loads would otherwise both start).
_loading_ids: set[str] = set()


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _load() -> list[dict[str, Any]]:
    if REGISTRY_PATH.exists():
        try:
            text = REGISTRY_PATH.read_text(encoding="utf-8")
            data = json.loads(text)
            if isinstance(data, list):
                return data
        except Exception as exc:
            logger.warning("Failed to read detector registry: %s", exc)
    return []


def _save(entries: list[dict[str, Any]]) -> None:
    atomic_write_json(REGISTRY_PATH, entries)


def _manifest_stamp() -> tuple[int, int] | None:
    """Return ``(mtime_ns, size)`` for the registry file, or ``None`` if absent."""
    try:
        stat = REGISTRY_PATH.stat()
    except OSError:
        return None
    return (stat.st_mtime_ns, stat.st_size)


def _ensure_loaded() -> list[dict[str, Any]]:
    """Return the in-memory cache, re-reading it whenever disk has moved on.

    The dataset registry was fixed for this in #3167 and its twin here was not,
    so every *read* stayed blind to a write by anyone else. A detector could be
    unregistered on disk while ``GET /api/detectors/registry`` went on listing
    it until the process restarted -- which is what happened when six finished
    slates were cleared from a running app: the file said nine, that endpoint
    said fifteen, and ``GET /api/detectors`` (which reads the detector files
    rather than the registry) said nine, so two views of one dashboard
    disagreed.

    Mutations were never at risk -- :func:`_read_modify_write` re-reads under
    the lock before mutating -- so this is a read-path staleness, but a
    convincing one: the stale view is the one a person is looking at.

    A stat per read is cheap next to the JSON parse it usually skips, and the
    stamp makes the re-read happen exactly when the file has actually changed.
    """
    global _entries, _entries_stamp
    stamp = _manifest_stamp()
    if _entries is None or stamp != _entries_stamp:
        _entries = _load()
        _entries_stamp = stamp
    return _entries


def _read_modify_write(mutator: Callable[[list[dict[str, Any]]], _T]) -> _T:
    """Run *mutator* over a fresh-from-disk registry under a cross-process lock.

    Mirrors :func:`vtscore.datasets.registry._read_modify_write`.  Holding the
    ``file_lock`` while re-reading closes the multi-process read-modify-write
    race so a mutation merges into the current on-disk state instead of
    clobbering entries a sibling process committed since this process last read.
    ``_load`` starts from disk truth (not the possibly-stale cache); the result
    is always persisted and swapped into ``_entries``.
    """
    global _entries
    with file_lock(REGISTRY_PATH):
        entries = _load()
        result = mutator(entries)
        _save(entries)
        with _lock:
            _entries = entries
            # Stamp what we just wrote, so the next read does not re-parse it.
            global _entries_stamp
            _entries_stamp = _manifest_stamp()
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def list_detectors() -> list[dict[str, Any]]:
    """Return summary info for all registered detectors."""
    with _lock:
        return [dict(e) for e in _ensure_loaded()]


def get_detector(detector_id: str) -> dict[str, Any] | None:
    """Return a single registry entry by *detector_id*, or ``None``."""
    with _lock:
        for entry in _ensure_loaded():
            if entry["id"] == detector_id:
                return dict(entry)
    return None


def register_detector(
    *,
    name: str,
    media_type: str,
    num_training: int = 0,
    text_query: str = "",
    media_example: str = "",
    examples: list[dict[str, Any]] | None = None,
    created_by: str = "default",
    embedder: str = "",
    embedder_type: str = "",
    readers: list[str] | None = None,
) -> dict[str, Any]:
    """Add a new detector to the registry and persist.

    Args:
        name: Display name for the dashboard.  Also the key the on-disk
            labelset file is looked up by; the file itself is
            ``data/detectors/<slug-of-name>.json``.
        media_type: ``"audio"``, ``"image"``, ``"video"``, ``"text"``, etc.
        num_training: Number of training examples (label count).
        text_query: Text-sort query associated with the detector.
        media_example: Optional path to an example media file (the first
            media example; kept alongside ``examples`` for quick display).
        examples: Full seed-example list (``[{"type": "text"|"media",
            "value": ...}, ...]``), mirroring the detector JSON.  The label
            session reads it so Autopilot can sort by every media example.
        created_by: Username of the user who created this detector.
        embedder: Name of the embedder used for this detector's labels.
            Defaults to ``""`` for newly created detectors that haven't been
            trained yet; stamped automatically the first time training runs
            (see :func:`record_detector_embedder`).  Read by the smart
            preload predictor so the right model is warmed at startup
            instead of the media type's default.
        embedder_type: The detector's locked embedder *type* ("semantic" /
            "patch_semantic" / "structural"), resolved at create time.  Drives
            detector/dataset compatibility gating without loading the detector.

    Returns:
        The newly created entry dict.
    """
    import uuid

    entry: dict[str, Any] = {
        "id": uuid.uuid4().hex,
        "name": name,
        "media_type": media_type,
        "num_training": num_training,
        "text_query": text_query,
        "media_example": media_example,
        "examples": list(examples) if examples else [],
        "created_by": created_by,
        "created_at": time.time(),
        "embedder": embedder,
        "embedder_type": embedder_type,
        # Access list (mirrors datasets): usernames allowed to see/load this
        # detector besides the creator. Empty = private to the creator;
        # ``["*"]`` = visible to everyone. See ``can_user_access_detector``.
        "readers": list(readers) if readers else [],
    }
    _read_modify_write(lambda entries: entries.append(entry))
    return entry


def unregister_detector(detector_id: str) -> bool:
    """Remove a detector from the registry. Returns ``True`` if found."""

    def mutate(entries: list[dict[str, Any]]) -> bool:
        for i, entry in enumerate(entries):
            if entry["id"] == detector_id:
                entries.pop(i)
                return True
        return False

    removed = _read_modify_write(mutate)
    with _lock:
        _loaded_ids.discard(detector_id)
    return removed


def rename_detector(detector_id: str, new_name: str) -> bool:
    """Rename a registered detector. Returns ``True`` on success."""

    def mutate(entries: list[dict[str, Any]]) -> bool:
        for entry in entries:
            if entry["id"] == detector_id:
                entry["name"] = new_name
                return True
        return False

    return _read_modify_write(mutate)


def update_detector(detector_id: str, **fields: Any) -> bool:
    """Update arbitrary fields on a registered detector."""

    def mutate(entries: list[dict[str, Any]]) -> bool:
        for entry in entries:
            if entry["id"] == detector_id:
                entry.update(fields)
                return True
        return False

    return _read_modify_write(mutate)


def record_detector_embedder(detector_id: str, embedder_name: str) -> None:
    """Persist the embedder a detector's labels are currently embedded with.

    Called from the training paths that stamp ``DetectorContext.embedder``
    so the smart preload predictor knows which model to warm on the next
    process start.  No-ops on empty inputs or unknown detector ids; swallows
    registry write failures because losing the optimization is preferable
    to crashing a training cycle.
    """
    if not detector_id or not embedder_name:
        return
    global _entries, _entries_stamp
    try:
        # Inline read-modify-write (not the shared helper) so an already-stamped
        # embedder skips the disk write - this runs on every training cycle.
        with file_lock(REGISTRY_PATH):
            entries = _load()
            changed = False
            for entry in entries:
                if entry["id"] == detector_id:
                    if entry.get("embedder") != embedder_name:
                        entry["embedder"] = embedder_name
                        changed = True
                    break
            if changed:
                _save(entries)
            with _lock:
                _entries = entries
                # Same reason as in `_read_modify_write`: stamp what we just
                # read/wrote so the next read does not re-parse it.
                _entries_stamp = _manifest_stamp()
    except Exception as exc:
        logger.warning("Failed to persist embedder for detector %s: %s", detector_id, exc)


def find_by_name(name: str) -> dict[str, Any] | None:
    """Return the entry whose ``name`` matches, or ``None``."""
    with _lock:
        for entry in _ensure_loaded():
            if entry.get("name") == name:
                return dict(entry)
    return None


# ---------------------------------------------------------------------------
# Access control (mirrors vtscore.datasets.registry)
#
# Detectors are user-shared just like datasets: the creator always has access,
# plus anyone listed in ``readers`` (or everyone when ``readers`` contains the
# wildcard ``"*"``). Entries created before this feature have no ``readers``
# key; ``.get("readers", [])`` treats them as private to their creator.
# ---------------------------------------------------------------------------


def can_user_access_detector(detector_id: str, username: str) -> bool:
    """Return ``True`` if *username* may view/load the detector.

    Granted when the user is the creator, is listed in ``readers``, or
    ``readers`` contains the wildcard ``"*"``.
    """
    with _lock:
        for entry in _ensure_loaded():
            if entry["id"] == detector_id:
                if entry.get("created_by", "default") == username:
                    return True
                readers = entry.get("readers", [])
                return username in readers or "*" in readers
    return False


def is_detector_owner(detector_id: str, username: str) -> bool:
    """Return ``True`` if *username* is the creator of the detector."""
    with _lock:
        for entry in _ensure_loaded():
            if entry["id"] == detector_id:
                return entry.get("created_by", "default") == username
    return False


def list_detectors_for_user(username: str) -> list[dict[str, Any]]:
    """Return only detectors *username* is allowed to see.

    Visible when the user is the creator, is listed in ``readers``, or
    ``"*"`` is in ``readers``.
    """
    with _lock:
        result = []
        for entry in _ensure_loaded():
            creator = entry.get("created_by", "default")
            readers = entry.get("readers", [])
            if creator == username or username in readers or "*" in readers:
                result.append(dict(entry))
        return result


def set_detector_readers(detector_id: str, readers: list[str], requesting_user: str) -> tuple[bool, str]:
    """Update a detector's ``readers`` list. Only the creator may call this.

    Returns ``(success, error_message)``.
    """

    def mutate(entries: list[dict[str, Any]]) -> tuple[bool, str]:
        for entry in entries:
            if entry["id"] == detector_id:
                if entry.get("created_by", "default") != requesting_user:
                    return False, "Only the detector creator can modify readers"
                entry["readers"] = readers
                return True, ""
        return False, "Detector not found"

    return _read_modify_write(mutate)


def add_loaded_detector_id(detector_id: str) -> None:
    """Add *detector_id* to the set of loaded detectors (without changing active)."""
    with _lock:
        _loaded_ids.add(detector_id)


def remove_loaded_detector_id(detector_id: str) -> None:
    """Remove *detector_id* from the loaded set."""
    with _lock:
        _loaded_ids.discard(detector_id)


def is_detector_loaded(detector_id: str) -> bool:
    """Return ``True`` if *detector_id* is in the loaded set."""
    with _lock:
        return detector_id in _loaded_ids


def begin_detector_load(detector_id: str) -> str:
    """Atomically claim the right to load *detector_id*.

    Mirrors :func:`vtscore.datasets.registry.begin_load`.  Returns ``"loaded"``
    if the detector is already resident, ``"in_progress"`` if another loader is
    already running (attach to its task), or ``"reserved"`` if the caller won
    the race and must run the load then call :func:`end_detector_load`.
    """
    with _lock:
        if detector_id in _loaded_ids:
            return "loaded"
        if detector_id in _loading_ids:
            return "in_progress"
        _loading_ids.add(detector_id)
        return "reserved"


def end_detector_load(detector_id: str) -> None:
    """Release the load reservation taken by :func:`begin_detector_load`."""
    with _lock:
        _loading_ids.discard(detector_id)


def get_loaded_detector_ids() -> set[str]:
    """Return a copy of all loaded detector IDs."""
    with _lock:
        return set(_loaded_ids)


def is_find_mode() -> bool:
    """Return ``True`` if the active detector's votes are find/scoring output.

    Find mode is per-detector state (``DetectorContext.find_mode``), not a
    process global: a scoring pass on one detector must never block vote
    syncing on another, and switching to a different detector must not inherit
    the previous detector's find state.
    """
    from vtscore.state.core import _state_lock, get_active_detector_context

    with _state_lock:
        return get_active_detector_context().find_mode


def set_find_mode(enabled: bool = True) -> None:
    """Set or clear find mode on the active detector context.

    No-op when no real detector is active (the empty / request-missing
    sentinel contexts have no labelset to protect).
    """
    from vtscore.state.core import (
        _state_lock,
        get_active_detector_context,
        is_request_missing_detector_context,
    )

    with _state_lock:
        det_ctx = get_active_detector_context()
        # The request-missing sentinel carries a truthy placeholder id
        # ("__request_missing__"), so identify it explicitly rather than
        # leaning on ``detector_id`` truthiness (which only screens the
        # empty fallback context).
        if is_request_missing_detector_context(det_ctx):
            return
        if det_ctx.detector_id:
            det_ctx.find_mode = enabled


def reset_for_tests() -> None:
    """Reset the in-memory cache (for test isolation)."""
    global _entries, _entries_stamp
    with _lock:
        _entries = None
        _entries_stamp = None
        _loaded_ids.clear()
        _loading_ids.clear()
