"""Blueprint for dataset management routes."""

import io
import threading
from pathlib import Path

from flask import Blueprint, jsonify, request, send_file

from config import (
    CIFAR10_DOWNLOAD_SIZE_MB,
    CLIPS_PER_CATEGORY,
    CLIPS_PER_VIDEO_CATEGORY,
    DATA_DIR,
    EMBEDDINGS_DIR,
    ESC50_DOWNLOAD_SIZE_MB,
    SAMPLE_VIDEOS_DOWNLOAD_SIZE_MB,
    VIDEO_DIR,
)
from vistatotes.datasets import (
    DEMO_DATASETS,
    export_dataset_to_file,
    load_dataset_from_folder,
    load_dataset_from_pickle,
    load_demo_dataset,
)
from vistatotes.models import get_e5_model
from vistatotes.utils import bad_votes, clips, good_votes, get_progress, update_progress

datasets_bp = Blueprint("datasets", __name__)


def clear_dataset():
    """Clear the current dataset."""
    clips.clear()
    good_votes.clear()
    bad_votes.clear()


@datasets_bp.route("/api/dataset/status")
def dataset_status():
    """Return the current dataset status."""
    return jsonify(
        {
            "loaded": len(clips) > 0,
            "num_clips": len(clips),
            "has_votes": len(good_votes) + len(bad_votes) > 0,
        }
    )


@datasets_bp.route("/api/dataset/progress")
def dataset_progress():
    """Return the current progress of long-running operations."""
    return jsonify(get_progress())


@datasets_bp.route("/api/dataset/demo-list")
def demo_dataset_list():
    """List available demo datasets."""
    demos = []
    for name, dataset_info in DEMO_DATASETS.items():
        pkl_file = EMBEDDINGS_DIR / f"{name}.pkl"
        is_ready = pkl_file.exists()

        media_type = dataset_info.get("media_type", "audio")

        # Calculate number of files
        num_categories = len(dataset_info["categories"])
        if media_type == "video":
            # For video datasets, estimate based on available files or use default
            num_files = num_categories * CLIPS_PER_VIDEO_CATEGORY
        else:
            # For audio datasets (ESC-50 has 40 clips per category)
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
                "ready": is_ready,
                "num_categories": num_categories,
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


@datasets_bp.route("/api/dataset/load-file", methods=["POST"])
def load_dataset_file():
    """Load a dataset from an uploaded pickle file."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    def load_task():
        try:
            update_progress("loading", "Loading dataset from file...", 0, 0)
            # Save to temp file
            temp_path = DATA_DIR / "temp_upload.pkl"
            DATA_DIR.mkdir(exist_ok=True)
            file.save(temp_path)

            clear_dataset()
            load_dataset_from_pickle(temp_path, clips)

            # Clean up
            temp_path.unlink()
            update_progress("idle", f"Loaded {len(clips)} clips from file")
        except Exception as e:
            update_progress("idle", "", 0, 0, str(e))

    thread = threading.Thread(target=load_task, daemon=True)
    thread.start()

    return jsonify({"ok": True, "message": "Loading started"})


@datasets_bp.route("/api/dataset/load-folder", methods=["POST"])
def load_dataset_folder():
    """Generate dataset from a folder of media files."""
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

    def load_task():
        try:
            clear_dataset()
            load_dataset_from_folder(folder, media_type, clips)
        except Exception as e:
            update_progress("idle", "", 0, 0, str(e))

    thread = threading.Thread(target=load_task, daemon=True)
    thread.start()

    return jsonify({"ok": True, "message": "Loading started"})


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
