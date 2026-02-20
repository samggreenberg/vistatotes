"""Flask routes for the Results Exporter API.

Endpoints
---------
GET  /api/exporters
    List all registered exporters with their metadata and field definitions.

POST /api/exporters/export
    Run a specific exporter on auto-detect results supplied in the request
    body.  Body (JSON)::

        {
            "exporter_name": "file",
            "field_values":  {"filepath": "/home/user/results.json"},
            "results":       { ...auto-detect results dict... }
        }

    Returns::

        {"success": true, "message": "...", ...exporter-specific keys...}
"""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from vistatotes.exporters import get_exporter, list_exporters

exporters_bp = Blueprint("exporters", __name__)


# ---------------------------------------------------------------------------
# GET /api/exporters
# ---------------------------------------------------------------------------


@exporters_bp.route("/api/exporters", methods=["GET"])
def get_exporters():
    """Return a list of all registered results exporters."""
    return jsonify([exp.to_dict() for exp in list_exporters()])


# ---------------------------------------------------------------------------
# POST /api/exporters/export
# ---------------------------------------------------------------------------


@exporters_bp.route("/api/exporters/export", methods=["POST"])
def run_export():
    """Run the named exporter on the supplied auto-detect results.

    Request body (JSON):

    .. code-block:: json

        {
            "exporter_name": "file",
            "field_values":  {"filepath": "/home/user/results.json"},
            "results":       {}
        }

    ``field_values`` and ``results`` are both optional – they default to
    empty dicts – but a valid ``exporter_name`` is required.
    """
    data = request.get_json(force=True, silent=True) or {}

    exporter_name = data.get("exporter_name", "").strip()
    if not exporter_name:
        return jsonify({"error": "exporter_name is required"}), 400

    exporter = get_exporter(exporter_name)
    if exporter is None:
        known = [exp.name for exp in list_exporters()]
        return (
            jsonify({"error": f"Unknown exporter '{exporter_name}'. Available: {known}"}),
            404,
        )

    field_values: dict = data.get("field_values", {}) or {}
    results: dict = data.get("results", {}) or {}

    # Validate required fields
    missing = [f.key for f in exporter.fields if f.required and not field_values.get(f.key, "").strip()]
    if missing:
        return (
            jsonify(
                {
                    "error": f"Missing required field(s): {missing}",
                    "missing_fields": missing,
                }
            ),
            400,
        )

    try:
        outcome = exporter.export(results, field_values)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover
        return jsonify({"error": f"Export failed: {exc}"}), 500

    return jsonify({"success": True, **outcome})
