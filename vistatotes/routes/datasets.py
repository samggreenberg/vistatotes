"""Blueprint for dataset management routes."""

import io
import threading
from pathlib import Path

from flask import Blueprint, jsonify, request, send_file

from config import (CIFAR10_DOWNLOAD_SIZE_MB, CLIPS_PER_CATEGORY,
                    CLIPS_PER_VIDEO_CATEGORY, EMBEDDINGS_DIR,
                    ESC50_DOWNLOAD_SIZE_MB, IMAGES_PER_CIFAR10_CATEGORY,
                    SAMPLE_VIDEOS_DOWNLOAD_SIZE_MB, VIDEO_DIR)
from vistatotes.datasets import (DEMO_DATASETS, export_dataset_to_file,
                                 get_importer, list_importers,
                                 load_demo_dataset)
from vistatotes.models import get_e5_model
from vistatotes.models.progress import clear_progress_cache
from vistatotes.utils import (bad_votes, clips, get_progress, good_votes,
                              label_history, update_progress)

datasets_bp = Blueprint("datasets", __name__)

# Names of importers that have dedicated, hand-crafted UI sections in the
# frontend.  They are excluded from the generic /api/dataset/importers list
# so the frontend doesn't render a duplicate panel for them.
_BUILTIN_IMPORTER_NAMES = {"pickle"}


def clear_dataset():
    """Clear the current dataset."""
    clips.clear()
    good_votes.clear()
    bad_votes.clear()
    label_history.clear()
    clear_progress_cache()


def _run_importer_in_background(importer, field_values: dict) -> None:
    """Start *importer*.run() in a daemon thread after clearing the dataset."""

    def load_task():
        try:
            clear_dataset()
            importer.run(field_values, clips)
        except Exception as e:
            update_progress("idle", "", 0, 0, str(e))

    thread = threading.Thread(target=load_task, daemon=True)
    thread.start()


# ---------------------------------------------------------------------------
# Status / progress
# ---------------------------------------------------------------------------

@datasets_bp.route("/api/dataset/status")
def dataset_status():
    """Return the current dataset status."""
    media_type = None
    if clips:
        media_type = next(iter(clips.values())).get("type", "audio")
    return jsonify(
        {
            "loaded": len(clips) > 0,
            "num_clips": len(clips),
            "has_votes": len(good_votes) + len(bad_votes) > 0,
            "media_type": media_type,
        }
    )


@datasets_bp.route("/api/dataset/progress")
def dataset_progress():
    """Return the current progress of long-running operations."""
    return jsonify(get_progress())


# ---------------------------------------------------------------------------
# Importer discovery
# ---------------------------------------------------------------------------

@datasets_bp.route("/api/dataset/importers")
def dataset_importers():
    """List all registered importers (excluding those with dedicated UI).

    The frontend uses this endpoint to auto-render any importer that isn't
    already handled by a hard-coded UI panel (i.e. anything beyond the
    built-in pickle/folder/demo importers).

    Returns a JSON object::

        {
          "importers": [
            {
              "name": "sftp",
              "display_name": "SFTP Server",
              "description": "...",
              "fields": [ { "key": ..., "label": ..., "field_type": ..., ... }, ... ]
            },
            ...
          ]
        }
    """
    extended = [
        imp.to_dict()
        for imp in list_importers()
        if imp.name not in _BUILTIN_IMPORTER_NAMES
    ]
    return jsonify({"importers": extended})


# ---------------------------------------------------------------------------
# Generic import endpoint
# ---------------------------------------------------------------------------

@datasets_bp.route("/api/dataset/import/<importer_name>", methods=["POST"])
def import_dataset(importer_name: str):
    """Run a registered importer by name in a background thread.

    For importers that have a field with ``field_type="file"``, the request
    must be ``multipart/form-data`` with the file stored under the field's
    ``key``.  All other field values are read from the form data (multipart)
    or from the JSON body.

    Returns ``{"ok": true, "message": "Loading started"}`` immediately; poll
    ``/api/dataset/progress`` to track progress.
    """
    importer = get_importer(importer_name)
    if importer is None:
        return jsonify({"error": f"Unknown importer: {importer_name!r}"}), 404

    file_keys = {f.key for f in importer.fields if f.field_type == "file"}

    # Build field_values from either multipart or JSON body.
    field_values: dict = {}
    if file_keys:
        for key in file_keys:
            if key not in request.files:
                return jsonify({"error": f"Missing file field: {key!r}"}), 400
            field_values[key] = request.files[key]
        # Non-file fields come from form data when using multipart.
        for f in importer.fields:
            if f.field_type != "file":
                field_values[f.key] = request.form.get(f.key, f.default)
    else:
        body = request.get_json(force=True) or {}
        for f in importer.fields:
            if f.key not in body and f.required:
                return jsonify({"error": f"Missing required field: {f.key!r}"}), 400
            field_values[f.key] = body.get(f.key, f.default)

    _run_importer_in_background(importer, field_values)
    return jsonify({"ok": True, "message": "Loading started"})


# ---------------------------------------------------------------------------
# Demo datasets  (special-cased: their own discovery + load endpoints)
# ---------------------------------------------------------------------------

@datasets_bp.route("/api/dataset/demo-list")
def demo_dataset_list():
    """List available demo datasets."""
    demos = []
    for name, dataset_info in DEMO_DATASETS.items():
        pkl_file = EMBEDDINGS_DIR / f"{name}.pkl"
        is_ready = pkl_file.exists()

        media_type = dataset_info.get("media_type", "audio")

        # Some pkl files reference external media directories rather than
        # inlining bytes.  If that directory has been removed since the pkl
        # was created, the dataset can't actually be loaded — don't show it
        # as ready.  Each demo dataset declares its own required_folder so
        # this check stays generic as new demo datasets are added.
        if is_ready:
            required_folder = dataset_info.get("required_folder")
            if required_folder is not None and not required_folder.exists():
                is_ready = False

        # Calculate number of files
        num_categories = len(dataset_info["categories"])
        if media_type == "video":
            num_files = num_categories * CLIPS_PER_VIDEO_CATEGORY
        elif media_type == "image":
            num_files = num_categories * IMAGES_PER_CIFAR10_CATEGORY
        elif media_type == "paragraph":
            # 20 Newsgroups: up to 50 texts per category
            num_files = num_categories * 50
        else:
            # Audio datasets (ESC-50 has 40 clips per category)
            num_files = num_categories * CLIPS_PER_CATEGORY

        # Calculate download size
        if is_ready:
            # If ready, show the actual .pkl file size
            download_size_mb = pkl_file.stat().st_size / (1024 * 1024)
        else:
            # If not ready, estimate download size
            if media_type == "video":
                # Check if video files exist
                video_source = dataset_info.get("source", "ucf101")
                if video_source == "ucf101":
                    video_dir = VIDEO_DIR / "ucf101"
                    if video_dir.exists():
                        # Videos are present, just need to embed
                        download_size_mb = 0
                        is_ready = False  # Not embedded yet, but videos available
                    else:
                        # Need to download/obtain videos (manual process for UCF-101)
                        download_size_mb = 0  # Manual download required
                else:
                    download_size_mb = SAMPLE_VIDEOS_DOWNLOAD_SIZE_MB
            elif media_type == "image":
                # CIFAR-10 dataset
                download_size_mb = CIFAR10_DOWNLOAD_SIZE_MB
            elif media_type == "paragraph":
                # 20 Newsgroups is small (scikit-learn downloads automatically)
                download_size_mb = 15  # Approximate size
            else:
                # Audio dataset - ESC-50 download
                download_size_mb = ESC50_DOWNLOAD_SIZE_MB

        demos.append(
            {
                "name": name,
                "label": dataset_info.get("label", name),
                "ready": is_ready,
                "num_files": num_files,
                "download_size_mb": round(download_size_mb, 1),
                "description": dataset_info.get("description", ""),
                "media_type": dataset_info.get("media_type", "audio"),
            }
        )
    return jsonify({"datasets": demos})


@datasets_bp.route("/api/dataset/load-demo", methods=["POST"])
def load_demo_dataset_route():
    """Load a demo dataset in a background thread."""
    data = request.get_json(force=True)
    if data is None:
        return jsonify({"error": "Invalid request body"}), 400

    dataset_name = data.get("name")

    if not dataset_name or dataset_name not in DEMO_DATASETS:
        return jsonify({"error": "Invalid dataset name"}), 400

    def load_task():
        try:
            clear_dataset()
            e5_model = get_e5_model()
            load_demo_dataset(dataset_name, clips, e5_model)
        except Exception as e:
            update_progress("idle", "", 0, 0, str(e))

    thread = threading.Thread(target=load_task, daemon=True)
    thread.start()

    return jsonify({"ok": True, "message": "Loading started"})


# ---------------------------------------------------------------------------
# Legacy endpoints – kept for backward compatibility.
# These now delegate to the appropriate importer internally.
# ---------------------------------------------------------------------------

@datasets_bp.route("/api/dataset/load-file", methods=["POST"])
def load_dataset_file():
    """Load a dataset from an uploaded pickle file.

    Delegates to the ``pickle`` importer.  Kept for backward compatibility
    with existing frontends and scripts.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    importer = get_importer("pickle")
    _run_importer_in_background(importer, {"file": file})
    return jsonify({"ok": True, "message": "Loading started"})


@datasets_bp.route("/api/dataset/load-folder", methods=["POST"])
def load_dataset_folder():
    """Generate dataset from a folder of media files.

    Delegates to the ``folder`` importer.  Kept for backward compatibility
    with existing frontends and scripts.
    """
    data = request.get_json(force=True)
    if data is None:
        return jsonify({"error": "Invalid request body"}), 400

    folder_path = data.get("path")
    media_type = data.get(
        "media_type", "sounds"
    )  # Default to sounds for backward compatibility

    if not folder_path:
        return jsonify({"error": "No folder path provided"}), 400

    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        return jsonify({"error": "Invalid folder path"}), 400

    importer = get_importer("folder")
    _run_importer_in_background(importer, {"path": str(folder), "media_type": media_type})
    return jsonify({"ok": True, "message": "Loading started"})


# ---------------------------------------------------------------------------
# Export / clear
# ---------------------------------------------------------------------------

@datasets_bp.route("/api/dataset/export")
def export_dataset():
    """Export the current dataset to a pickle file."""
    if not clips:
        return jsonify({"error": "No dataset loaded"}), 400

    try:
        dataset_bytes = export_dataset_to_file(clips)
        return send_file(
            io.BytesIO(dataset_bytes),
            mimetype="application/octet-stream",
            download_name="vectorytones_dataset.pkl",
            as_attachment=True,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@datasets_bp.route("/api/dataset/clear", methods=["POST"])
def clear_dataset_route():
    """Clear the current dataset."""
    clear_dataset()
    return jsonify({"ok": True})
