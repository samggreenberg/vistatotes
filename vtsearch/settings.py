"""Persistent settings for VTSearch.

Settings are stored as a JSON file in ``data/settings.json``.  The module
exposes simple get/set helpers and auto-saves on every mutation.

Schema (all keys optional, missing keys use defaults)::

    {
        "volume": 1.0,
        "inclusion": 0,
        "favorite_processors": [
            {
                "processor_name": "my detector",
                "processor_importer": "detector_file",
                "field_values": {"file": "/path/to/detector.json"}
            }
        ]
    }

Favorite processors store the *recipe* for importing a processor (the importer
name, field values, and desired detector name).  They are only materialised
into favorite detectors on demand — during autodetect.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from config import DATA_DIR

logger = logging.getLogger(__name__)

SETTINGS_PATH: Path = DATA_DIR / "settings.json"

_DEFAULTS: dict[str, Any] = {
    "volume": 1.0,
    "inclusion": 0,
    "theme": "dark",
    "enrich_descriptions": False,
    "favorite_processors": [],
}

# In-memory cache — loaded once, written on every mutation.
_settings: dict[str, Any] | None = None


def _load() -> dict[str, Any]:
    """Read settings from disk, returning defaults on any failure."""
    if SETTINGS_PATH.exists():
        try:
            text = SETTINGS_PATH.read_text(encoding="utf-8")
            data = json.loads(text)
            if isinstance(data, dict):
                return data
        except Exception as exc:
            logger.warning("Failed to read settings file: %s", exc)
    return {}


def _save(data: dict[str, Any]) -> None:
    """Write *data* to the settings file (creating parent dirs if needed)."""
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _ensure_loaded() -> dict[str, Any]:
    global _settings
    if _settings is None:
        _settings = _load()
    return _settings


# -------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------


def get_all() -> dict[str, Any]:
    """Return the full settings dict (with defaults filled in)."""
    s = _ensure_loaded()
    result = dict(_DEFAULTS)
    result.update(s)
    return result


def get_volume() -> float:
    """Return the persisted playback volume (0.0–1.0)."""
    return float(_ensure_loaded().get("volume", _DEFAULTS["volume"]))


def set_volume(value: float) -> None:
    """Set and persist the playback volume."""
    s = _ensure_loaded()
    s["volume"] = max(0.0, min(1.0, float(value)))
    _save(s)


def get_inclusion() -> int:
    """Return the persisted inclusion setting (-10 to +10)."""
    return int(_ensure_loaded().get("inclusion", _DEFAULTS["inclusion"]))


def set_inclusion(value: int) -> None:
    """Set and persist the inclusion setting."""
    s = _ensure_loaded()
    s["inclusion"] = int(max(-10, min(10, int(value))))
    _save(s)


def get_theme() -> str:
    """Return the persisted theme ('dark' or 'light')."""
    return str(_ensure_loaded().get("theme", _DEFAULTS["theme"]))


def set_theme(value: str) -> None:
    """Set and persist the theme.  Must be 'dark' or 'light'."""
    if value not in ("dark", "light"):
        raise ValueError(f"Invalid theme: {value!r}")
    s = _ensure_loaded()
    s["theme"] = value
    _save(s)


def get_enrich_descriptions() -> bool:
    """Return whether enriched description embedding is enabled."""
    return bool(_ensure_loaded().get("enrich_descriptions", _DEFAULTS["enrich_descriptions"]))


def set_enrich_descriptions(value: bool) -> None:
    """Set and persist the enrich_descriptions flag."""
    s = _ensure_loaded()
    s["enrich_descriptions"] = bool(value)
    _save(s)


def get_favorite_processors() -> list[dict[str, Any]]:
    """Return the list of favorite processor recipes."""
    return list(_ensure_loaded().get("favorite_processors", []))


def add_favorite_processor(
    processor_name: str,
    processor_importer: str,
    field_values: dict[str, Any],
) -> None:
    """Add a favorite processor recipe (or overwrite one with the same name)."""
    s = _ensure_loaded()
    procs: list[dict[str, Any]] = s.setdefault("favorite_processors", [])
    # Remove existing entry with same name
    procs[:] = [p for p in procs if p.get("processor_name") != processor_name]
    procs.append(
        {
            "processor_name": processor_name,
            "processor_importer": processor_importer,
            "field_values": field_values,
        }
    )
    _save(s)


def remove_favorite_processor(processor_name: str) -> bool:
    """Remove a favorite processor by name.  Returns True if found."""
    s = _ensure_loaded()
    procs: list[dict[str, Any]] = s.get("favorite_processors", [])
    before = len(procs)
    procs[:] = [p for p in procs if p.get("processor_name") != processor_name]
    if len(procs) < before:
        s["favorite_processors"] = procs
        _save(s)
        return True
    return False


def to_settings_json(entry: dict[str, Any]) -> str:
    """Build the JSON snippet for a favorite processor entry.

    Returns the JSON object that would appear inside the
    ``favorite_processors`` array in a settings file.  Useful for showing
    users how to recreate this processor configuration.

    Example output::

        {"processor_name": "my detector", "processor_importer": "detector_file",
         "field_values": {"file": "detector.json"}}
    """
    import json

    snippet = {
        "processor_name": entry["processor_name"],
        "processor_importer": entry["processor_importer"],
        "field_values": entry.get("field_values", {}),
    }
    return json.dumps(snippet)


def ensure_favorite_processors_imported() -> list[str]:
    """Import any favorite processors that are not already loaded as favorite detectors.

    This is the lazy-load mechanism: favorite processor recipes are materialised
    into real favorite detectors only when this function is called (typically
    right before autodetect).

    Returns:
        A list of processor names that were newly imported.
    """
    from vtsearch.processors.importers import get_processor_importer
    from vtsearch.utils import add_favorite_detector, get_favorite_detectors

    existing = get_favorite_detectors()
    imported: list[str] = []

    for entry in get_favorite_processors():
        name = entry.get("processor_name", "")
        if not name or name in existing:
            continue

        importer_name = entry.get("processor_importer", "")
        importer = get_processor_importer(importer_name)
        if importer is None:
            logger.warning("Favorite processor '%s': unknown importer '%s'", name, importer_name)
            continue

        field_values = dict(entry.get("field_values", {}))

        try:
            importer.validate_cli_field_values(field_values)
            result = importer.run_cli(field_values)

            if not isinstance(result, dict) or not result.get("weights"):
                logger.warning("Favorite processor '%s': importer returned invalid result", name)
                continue

            add_favorite_detector(
                name,
                result.get("media_type", "audio"),
                result["weights"],
                result.get("threshold", 0.5),
            )
            imported.append(name)
        except Exception as exc:
            logger.warning("Favorite processor '%s': import failed: %s", name, exc)

    return imported


def set_settings_path(path: str | Path) -> None:
    """Override the settings file path and reset the in-memory cache.

    Call this before :func:`ensure_favorite_processors_imported` to load
    favorite processors from a custom settings file (e.g. the ``--settings``
    CLI flag).
    """
    global SETTINGS_PATH, _settings
    SETTINGS_PATH = Path(path)
    _settings = None  # force reload on next access


def reset() -> None:
    """Reset the in-memory cache (for testing)."""
    global _settings
    _settings = None
