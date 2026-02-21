"""Flask routes for the Processor Importer API.

Endpoints
---------
GET  /api/processor-importers
    List all registered processor importers with their metadata and field definitions.

POST /api/processor-importers/import/<importer_name>
    Run the named processor importer.  Accepts ``multipart/form-data`` (when the
    importer has a ``"file"`` field) or JSON (for text-only importers).

    The importer returns processor data (weights, threshold, media_type).
    A ``name`` field is required (from form data or JSON body) to save the
    result as a favorite detector.

    Returns::

        {"success": true, "name": "<str>", "media_type": "<str>", ...}
"""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from vtsearch.processors.importers import get_processor_importer, list_processor_importers
from vtsearch.utils import add_favorite_detector

processor_importers_bp = Blueprint("processor_importers", __name__)


# ---------------------------------------------------------------------------
# GET /api/processor-importers
# ---------------------------------------------------------------------------


@processor_importers_bp.route("/api/processor-importers", methods=["GET"])
def get_processor_importers():
    """Return a list of all registered processor importers."""
    return jsonify([imp.to_dict() for imp in list_processor_importers()])


# ---------------------------------------------------------------------------
# POST /api/processor-importers/import/<importer_name>
# ---------------------------------------------------------------------------


@processor_importers_bp.route("/api/processor-importers/import/<importer_name>", methods=["POST"])
def run_processor_import(importer_name: str):
    """Run the named processor importer and save the result as a favorite detector.

    Accepts ``multipart/form-data`` when the importer has a ``"file"`` field,
    or ``application/json`` for text-only importers.  In both cases the route
    builds a ``field_values`` dict and passes it to
    :meth:`~vtsearch.processors.importers.base.ProcessorImporter.run`.

    A ``name`` field is required to identify the saved detector.

    Returns JSON with ``success``, ``name``, ``media_type``, and any extra
    keys returned by the importer.
    """
    importer = get_processor_importer(importer_name)
    if importer is None:
        known = [imp.name for imp in list_processor_importers()]
        return (
            jsonify({"error": f"Unknown processor importer '{importer_name}'. Available: {known}"}),
            404,
        )

    # Build field_values from either multipart or JSON body
    has_file_fields = any(f.field_type == "file" for f in importer.fields)
    field_values: dict = {}

    if has_file_fields:
        for f in importer.fields:
            if f.field_type == "file":
                field_values[f.key] = request.files.get(f.key)
            else:
                field_values[f.key] = request.form.get(f.key, f.default or "")
        # name comes from form data
        name = request.form.get("name", "").strip()
    else:
        body = request.get_json(force=True, silent=True) or {}
        for f in importer.fields:
            field_values[f.key] = body.get(f.key, f.default or "")
        name = body.get("name", "").strip()

    if not name:
        return jsonify({"error": "name is required"}), 400

    # Validate required fields (skip file fields — presence checked by importer)
    missing = [
        f.key
        for f in importer.fields
        if f.required and f.field_type != "file" and not field_values.get(f.key, "").strip()
    ]
    if missing:
        return (
            jsonify({"error": f"Missing required field(s): {missing}", "missing_fields": missing}),
            400,
        )

    try:
        result = importer.run(field_values)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover
        return jsonify({"error": f"Import failed: {exc}"}), 500

    if not isinstance(result, dict):
        return jsonify({"error": "Importer did not return a dict."}), 500

    media_type = result.get("media_type", "audio")
    weights = result.get("weights")
    threshold = result.get("threshold", 0.5)

    if not weights:
        return jsonify({"error": "Importer result missing 'weights'."}), 500

    # Use suggested name from the importer if the user didn't provide one
    # (already checked above that name is non-empty, but importer may suggest)
    add_favorite_detector(name, media_type, weights, threshold)

    response: dict = {
        "success": True,
        "name": name,
        "media_type": media_type,
    }
    # Forward extra keys from the importer (loaded, skipped, etc.)
    for key in ("loaded", "skipped"):
        if key in result:
            response[key] = result[key]

    return jsonify(response)
