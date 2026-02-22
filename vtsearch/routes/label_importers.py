"""Flask routes for the Label Importer API.

Endpoints
---------
GET  /api/label-importers
    List all registered label importers with their metadata and field definitions.

POST /api/label-importers/import/<importer_name>
    Run the named label importer.  Accepts ``multipart/form-data`` (when the
    importer has a ``"file"`` field) or JSON (for text-only importers).

    Returns::

        {
          "applied": <int>,
          "skipped": <int>,
          "missing_count": <int>,
          "missing": [<entry>, ...],
          "message": "<str>"
        }

    When ``missing_count`` is non-zero the response contains the label entries
    that could not be matched to any clip in the current dataset (neither by
    ``origin`` + ``origin_name`` nor by ``md5``).  The frontend can prompt the
    user and then call ``POST /api/label-importers/ingest-missing`` to pull
    those clips from their origins.

POST /api/label-importers/ingest-missing
    Accept a list of missing label entries, re-ingest them from their origins,
    and apply the labels.
"""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from vtsearch.labels.importers import get_label_importer, list_label_importers
from vtsearch.utils import (
    add_label_to_history,
    bad_votes,
    build_clip_lookup,
    clips,
    find_missing_entries,
    good_votes,
    resolve_clip_ids,
)

label_importers_bp = Blueprint("label_importers", __name__)


def _apply_labels(
    label_entries: list[dict],
    origin_lookup: dict[str, list[int]],
    md5_lookup: dict[str, list[int]],
) -> tuple[int, int]:
    """Apply label entries to the global vote state.

    Returns ``(applied, skipped)`` counts.
    """
    applied = 0
    skipped = 0

    for entry in label_entries:
        label = entry.get("label", "")
        if label not in ("good", "bad"):
            skipped += 1
            continue
        cids = resolve_clip_ids(entry, origin_lookup, md5_lookup)
        if not cids:
            skipped += 1
            continue

        for cid in cids:
            if label == "good":
                bad_votes.pop(cid, None)
                good_votes[cid] = None
                add_label_to_history(cid, "good")
            else:
                good_votes.pop(cid, None)
                bad_votes[cid] = None
                add_label_to_history(cid, "bad")
        applied += 1

    return applied, skipped


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

    Returns JSON with ``applied``, ``skipped``, ``missing_count``,
    ``missing``, and ``message`` keys.  When ``missing_count > 0`` the
    client should prompt the user and optionally call
    ``POST /api/label-importers/ingest-missing`` with the ``missing`` list.
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
                field_values[f.key] = request.form.get(f.key, f.default if f.default is not None else "")
    else:
        body = request.get_json(force=True, silent=True) or {}
        for f in importer.fields:
            field_values[f.key] = body.get(f.key, f.default if f.default is not None else "")

    # Validate required fields (skip file fields — presence checked by importer)
    missing_fields = [
        f.key
        for f in importer.fields
        if f.required and f.field_type != "file" and not field_values.get(f.key, "").strip()
    ]
    if missing_fields:
        return (
            jsonify({"error": f"Missing required field(s): {missing_fields}", "missing_fields": missing_fields}),
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
    origin_lookup, md5_lookup = build_clip_lookup(clips)
    applied, skipped = _apply_labels(label_entries, origin_lookup, md5_lookup)

    # Detect entries that could not be matched at all
    missing = find_missing_entries(label_entries, origin_lookup, md5_lookup)
    # Adjust skipped count: missing entries were already counted as skipped
    # by _apply_labels, but we report them separately now.
    skipped -= len(missing)

    msg = f"Applied {applied} label(s), skipped {skipped}."
    if missing:
        msg += f" {len(missing)} element(s) not found in dataset."

    return jsonify(
        {
            "applied": applied,
            "skipped": skipped,
            "missing_count": len(missing),
            "missing": missing,
            "message": msg,
        }
    )


# ---------------------------------------------------------------------------
# POST /api/label-importers/ingest-missing
# ---------------------------------------------------------------------------


@label_importers_bp.route("/api/label-importers/ingest-missing", methods=["POST"])
def ingest_missing():
    """Re-ingest missing clips from their origins, then apply their labels.

    Expects a JSON body::

        {"entries": [<label-entry>, ...]}

    Groups the entries by origin, runs each origin's dataset importer to
    recover the full clip data (media bytes + embedding), appends the
    matched clips to the live dataset, and applies the labels.

    Returns::

        {"ingested": <int>, "applied": <int>, "message": "<str>"}
    """
    body = request.get_json(force=True, silent=True) or {}
    entries = body.get("entries", [])

    if not isinstance(entries, list) or not entries:
        return jsonify({"error": "Request must contain a non-empty 'entries' list."}), 400

    from vtsearch.datasets.ingest import ingest_missing_clips

    ingested = ingest_missing_clips(entries, clips)

    # Now apply labels to the newly ingested clips
    origin_lookup, md5_lookup = build_clip_lookup(clips)
    applied, _ = _apply_labels(entries, origin_lookup, md5_lookup)

    return jsonify(
        {
            "ingested": ingested,
            "applied": applied,
            "message": f"Ingested {ingested} clip(s), applied {applied} label(s).",
        }
    )
