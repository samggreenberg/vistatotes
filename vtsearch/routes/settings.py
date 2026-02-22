"""Flask routes for the Settings API.

Endpoints
---------
GET  /api/settings
    Return all persisted settings (volume, favorite_processors).

PUT  /api/settings
    Update one or more settings fields.  Only supplied keys are changed.

GET  /api/settings/favorite-processors
    List all favorite processor recipes.

POST /api/settings/favorite-processors
    Add (or overwrite) a favorite processor recipe.

DELETE /api/settings/favorite-processors/<name>
    Remove a favorite processor recipe by name.
"""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from vtsearch import settings

settings_bp = Blueprint("settings", __name__)


@settings_bp.route("/api/settings", methods=["GET"])
def get_settings():
    """Return all settings."""
    data = settings.get_all()
    # Include CLI command strings for each favorite processor
    for proc in data.get("favorite_processors", []):
        proc["cli_command"] = settings.to_cli_command(proc)
    return jsonify(data)


@settings_bp.route("/api/settings", methods=["PUT"])
def update_settings():
    """Update settings.  Only supplied keys are changed."""
    body = request.get_json(force=True, silent=True)
    if not body or not isinstance(body, dict):
        return jsonify({"error": "Invalid request body"}), 400

    if "volume" in body:
        try:
            settings.set_volume(float(body["volume"]))
        except (TypeError, ValueError):
            return jsonify({"error": "volume must be a number"}), 400

    if "theme" in body:
        try:
            settings.set_theme(str(body["theme"]))
        except ValueError:
            return jsonify({"error": "theme must be 'dark' or 'light'"}), 400

    if "inclusion" in body:
        try:
            val = body["inclusion"]
            if not isinstance(val, (int, float)):
                return jsonify({"error": "inclusion must be a number"}), 400
            clamped = int(max(-10, min(10, int(val))))
            # Update runtime state (which also persists to settings file)
            from vtsearch.utils import set_inclusion

            set_inclusion(clamped)
        except (TypeError, ValueError):
            return jsonify({"error": "inclusion must be a number"}), 400

    if "enrich_descriptions" in body:
        settings.set_enrich_descriptions(bool(body["enrich_descriptions"]))

    return jsonify(settings.get_all())


@settings_bp.route("/api/settings/favorite-processors", methods=["GET"])
def get_favorite_processors():
    """List all favorite processor recipes."""
    procs = settings.get_favorite_processors()
    for proc in procs:
        proc["cli_command"] = settings.to_cli_command(proc)
    return jsonify({"favorite_processors": procs})


@settings_bp.route("/api/settings/favorite-processors", methods=["POST"])
def add_favorite_processor():
    """Add or overwrite a favorite processor recipe."""
    body = request.get_json(force=True, silent=True)
    if not body or not isinstance(body, dict):
        return jsonify({"error": "Invalid request body"}), 400

    processor_name = (body.get("processor_name") or "").strip()
    processor_importer = (body.get("processor_importer") or "").strip()
    field_values = body.get("field_values", {})

    if not processor_name:
        return jsonify({"error": "processor_name is required"}), 400
    if not processor_importer:
        return jsonify({"error": "processor_importer is required"}), 400

    settings.add_favorite_processor(processor_name, processor_importer, field_values)
    entry = {
        "processor_name": processor_name,
        "processor_importer": processor_importer,
        "field_values": field_values,
    }
    entry["cli_command"] = settings.to_cli_command(entry)
    return jsonify({"success": True, **entry})


@settings_bp.route("/api/settings/favorite-processors/<name>", methods=["DELETE"])
def delete_favorite_processor(name: str):
    """Remove a favorite processor recipe by name."""
    if settings.remove_favorite_processor(name):
        return jsonify({"success": True})
    return jsonify({"error": "Favorite processor not found"}), 404
