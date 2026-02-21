"""Flask routes for the Label Importer API.

Endpoints
---------
GET  /api/label-importers
    List all registered label importers with their metadata and field definitions.

POST /api/label-importers/import/<importer_name>
    Run the named label importer.  Accepts ``multipart/form-data`` (when the
    importer has a ``"file"`` field) or JSON (for text-only importers).

    Returns::

        {"applied": <int>, "skipped": <int>, "message": "<str>"}
"""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from vtsearch.labels.importers import get_label_importer, list_label_importers
from vtsearch.utils import add_label_to_history, bad_votes, clips, good_votes

label_importers_bp = Blueprint("label_importers", __name__)


# ---------------------------------------------------------------------------
# GET /api/label-importers
# ---------------------------------------------------------------------------


@label_importers_bp.route("/api/label-importers", methods=["GET"])
def get_label_importers():
    """Return a list of all registered label importers."""
    return jsonify([imp.to_dict() for imp in list_label_importers()])


# ---------------------------------------------------------------------------
# POST /api/label-importers/import/<importer_name>
# ---------------------------------------------------------------------------


@label_importers_bp.route("/api/label-importers/import/<importer_name>", methods=["POST"])
def run_label_import(importer_name: str):
    """Run the named label importer and apply the resulting labels.

    Accepts ``multipart/form-data`` when the importer has a ``"file"`` field,
    or ``application/json`` for text-only importers.  In both cases the route
    builds a ``field_values`` dict and passes it to
    :meth:`~vtsearch.labels.importers.base.LabelImporter.run`.

    Returns JSON with ``applied``, ``skipped``, and ``message`` keys.
    """
    importer = get_label_importer(importer_name)
    if importer is None:
        known = [imp.name for imp in list_label_importers()]
        return (
            jsonify({"error": f"Unknown label importer '{importer_name}'. Available: {known}"}),
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
    else:
        body = request.get_json(force=True, silent=True) or {}
        for f in importer.fields:
            field_values[f.key] = body.get(f.key, f.default or "")

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
        label_entries = importer.run(field_values)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover
        return jsonify({"error": f"Import failed: {exc}"}), 500

    if not isinstance(label_entries, list):
        return jsonify({"error": "Importer did not return a list of label dicts."}), 500

    # Apply labels to global vote state
    md5_to_id = {clip["md5"]: clip["id"] for clip in clips.values()}
    applied = 0
    skipped = 0

    for entry in label_entries:
        md5 = entry.get("md5", "")
        label = entry.get("label", "")
        if label not in ("good", "bad"):
            skipped += 1
            continue
        cid = md5_to_id.get(md5)
        if cid is None:
            skipped += 1
            continue

        if label == "good":
            bad_votes.pop(cid, None)
            good_votes[cid] = None
            add_label_to_history(cid, "good")
        else:
            good_votes.pop(cid, None)
            bad_votes[cid] = None
            add_label_to_history(cid, "bad")
        applied += 1

    return jsonify(
        {
            "applied": applied,
            "skipped": skipped,
            "message": f"Applied {applied} label(s), skipped {skipped}.",
        }
    )
