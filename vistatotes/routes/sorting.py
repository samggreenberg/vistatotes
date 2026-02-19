"""Blueprint for sorting and voting routes."""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from flask import Blueprint, jsonify, request

from config import DATA_DIR
from vistatotes.models import (
    analyze_labeling_progress,
    compute_labeling_status,
    calculate_cross_calibration_threshold,
    calculate_gmm_threshold,
    embed_audio_file,
    embed_image_file,
    embed_paragraph_file,
    embed_video_file,
    embed_text_query,
    get_clap_model,
    train_and_score,
    train_model,
)
from vistatotes.utils import (
    add_favorite_detector,
    add_favorite_extractor,
    add_label_to_history,
    bad_votes,
    clips,
    get_favorite_detectors,
    get_favorite_detectors_by_media,
    get_favorite_extractors,
    get_favorite_extractors_by_media,
    get_inclusion,
    get_sort_progress,
    good_votes,
    label_history,
    remove_favorite_detector,
    remove_favorite_extractor,
    rename_favorite_detector,
    rename_favorite_extractor,
    set_inclusion,
    update_sort_progress,
)

sorting_bp = Blueprint("sorting", __name__)


@sorting_bp.route("/api/sort/progress")
def sort_progress():
    """Return the current progress of a text sort operation."""
    return jsonify(get_sort_progress())


@sorting_bp.route("/api/sort", methods=["POST"])
def sort_clips():
    """Return clips sorted by cosine similarity to a text query."""
    try:
        data = request.get_json(force=True)
    except Exception as e:
        print(f"DEBUG: Sort failed - JSON error: {e}", flush=True)
        update_sort_progress("idle")
        return jsonify({"error": "Invalid request body"}), 400

    if data is None:
        update_sort_progress("idle")
        return jsonify({"error": "Invalid request body"}), 400

    text = data.get("text", "").strip()
    print(f"DEBUG: Sort request for '{text}'", flush=True)

    if not text:
        update_sort_progress("idle")
        return jsonify({"error": "text is required"}), 400

    # Determine media type from current clips
    if not clips:
        print("DEBUG: Sort failed - No clips loaded", flush=True)
        update_sort_progress("idle")
        return jsonify({"error": "No clips loaded"}), 400

    media_type = next(iter(clips.values())).get("type", "audio")

    # Total steps: 1 (embed) + len(clips) (similarities) + 1 (threshold)
    total_steps = 1 + len(clips) + 1
    update_sort_progress("sorting", "Embedding text query…", 0, total_steps)

    # Embed text query using refactored module
    text_vec = embed_text_query(text, media_type)
    if text_vec is None:
        print(
            f"DEBUG: Sort failed - Could not embed text for media type {media_type}",
            flush=True,
        )
        update_sort_progress("idle")
        return (
            jsonify({"error": f"Could not embed text for media type {media_type}"}),
            500,
        )

    update_sort_progress("sorting", "Computing similarities…", 1, total_steps)

    results = []
    scores = []
    for i, (clip_id, clip) in enumerate(clips.items()):
        media_vec = clip["embedding"]
        norm_product = np.linalg.norm(media_vec) * np.linalg.norm(text_vec)
        if norm_product == 0:
            similarity = 0.0
        else:
            similarity = float(np.dot(media_vec, text_vec) / norm_product)
        results.append({"id": clip_id, "similarity": round(similarity, 4)})
        scores.append(similarity)
        if (i + 1) % 50 == 0 or i + 1 == len(clips):
            update_sort_progress("sorting", "Computing similarities…", 1 + i + 1, total_steps)

    # Calculate GMM-based threshold
    update_sort_progress("sorting", "Calculating threshold…", total_steps - 1, total_steps)
    threshold = calculate_gmm_threshold(scores)

    results.sort(key=lambda x: x["similarity"], reverse=True)
    update_sort_progress("idle")
    return jsonify({"results": results, "threshold": round(threshold, 4)})


@sorting_bp.route("/api/learned-sort", methods=["POST"])
def learned_sort():
    """Train MLP on voted clips, return all clips sorted by predicted score."""
    if not good_votes or not bad_votes:
        return jsonify({"error": "need at least one good and one bad vote"}), 400
    results, threshold = train_and_score(clips, good_votes, bad_votes, get_inclusion())
    return jsonify({"results": results, "threshold": round(threshold, 4)})


@sorting_bp.route("/api/votes")
def get_votes():
    return jsonify(
        {
            "good": list(good_votes),  # Maintains insertion order (dict keys)
            "bad": list(bad_votes),  # Maintains insertion order (dict keys)
        }
    )


@sorting_bp.route("/api/labels/export")
def export_labels():
    """Export labels as JSON keyed by clip MD5 hash."""
    labels = []
    for cid in good_votes:  # Maintains insertion order
        clip = clips.get(cid)
        if clip:
            labels.append({"md5": clip["md5"], "label": "good"})
    for cid in bad_votes:  # Maintains insertion order
        clip = clips.get(cid)
        if clip:
            labels.append({"md5": clip["md5"], "label": "bad"})
    return jsonify({"labels": labels})


@sorting_bp.route("/api/labels/import", methods=["POST"])
def import_labels():
    """Import labels from JSON, matching clips by MD5 hash."""
    data = request.get_json(force=True)
    if data is None:
        return jsonify({"error": "Invalid request body"}), 400

    labels = data.get("labels")
    if not isinstance(labels, list):
        return jsonify({"error": "labels must be a list"}), 400

    # Build MD5 -> clip ID lookup
    md5_to_id = {clip["md5"]: clip["id"] for clip in clips.values()}

    applied = 0
    skipped = 0
    for entry in labels:
        md5 = entry.get("md5")
        label = entry.get("label")
        if label not in ("good", "bad"):
            skipped += 1
            continue
        cid = md5_to_id.get(md5)
        if cid is None:
            skipped += 1
            continue

        # Apply the label, overriding any existing vote
        if label == "good":
            bad_votes.pop(cid, None)
            good_votes[cid] = None
            add_label_to_history(cid, "good")
        else:
            good_votes.pop(cid, None)
            bad_votes[cid] = None
            add_label_to_history(cid, "bad")
        applied += 1

    return jsonify({"applied": applied, "skipped": skipped})


@sorting_bp.route("/api/inclusion")
def get_inclusion_route():
    """Get the current Inclusion setting."""
    return jsonify({"inclusion": get_inclusion()})


@sorting_bp.route("/api/inclusion", methods=["POST"])
def set_inclusion_route():
    """Set the Inclusion setting."""
    data = request.get_json(force=True)
    if data is None:
        return jsonify({"error": "Invalid request body"}), 400

    new_inclusion = data.get("inclusion")

    if not isinstance(new_inclusion, (int, float)):
        return jsonify({"error": "inclusion must be a number"}), 400

    # Clamp to -10 to +10 range
    new_inclusion = int(max(-10, min(10, new_inclusion)))
    set_inclusion(new_inclusion)

    return jsonify({"inclusion": get_inclusion()})


@sorting_bp.route("/api/detector/export", methods=["POST"])
def export_detector():
    """Train MLP on current votes and export the model weights."""
    if not good_votes or not bad_votes:
        return jsonify({"error": "need at least one good and one bad vote"}), 400

    # Train the model
    X_list = []
    y_list = []
    for cid in good_votes:
        X_list.append(clips[cid]["embedding"])
        y_list.append(1.0)
    for cid in bad_votes:
        X_list.append(clips[cid]["embedding"])
        y_list.append(0.0)

    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    y = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1)

    input_dim = X.shape[1]

    # Calculate threshold using cross-calibration with inclusion
    threshold = calculate_cross_calibration_threshold(X_list, y_list, input_dim, get_inclusion())

    # Train final model on all data with inclusion
    model = train_model(X, y, input_dim, get_inclusion())

    # Extract model weights
    state_dict = model.state_dict()
    weights = {}
    for key, value in state_dict.items():
        weights[key] = value.tolist()

    return jsonify({"weights": weights, "threshold": round(threshold, 4)})


@sorting_bp.route("/api/detector-sort", methods=["POST"])
def detector_sort():
    """Score all clips using a loaded detector model."""
    data = request.get_json(force=True)
    if data is None:
        return jsonify({"error": "Invalid request body"}), 400

    detector = data.get("detector")
    if not detector:
        return jsonify({"error": "detector is required"}), 400

    weights = detector.get("weights")
    threshold = detector.get("threshold", 0.5)

    if not weights:
        return jsonify({"error": "detector weights are required"}), 400

    # Reconstruct the model from weights
    # Determine input_dim from the first layer weights
    input_dim = len(weights["0.weight"][0])

    model = nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
        nn.Sigmoid(),
    )

    # Load weights
    state_dict = {}
    for key, value in weights.items():
        state_dict[key] = torch.tensor(value, dtype=torch.float32)
    model.load_state_dict(state_dict)
    model.eval()

    # Score every clip
    all_ids = sorted(clips.keys())
    all_embs = np.array([clips[cid]["embedding"] for cid in all_ids])
    X_all = torch.tensor(all_embs, dtype=torch.float32)
    with torch.no_grad():
        scores = model(X_all).squeeze(1).tolist()

    results = [{"id": cid, "score": round(s, 4)} for cid, s in zip(all_ids, scores)]
    results.sort(key=lambda x: x["score"], reverse=True)
    return jsonify({"results": results, "threshold": round(threshold, 4)})


@sorting_bp.route("/api/example-sort", methods=["POST"])
def example_sort():
    """Sort clips by similarity to an uploaded example audio file."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    clap_model, clap_processor = get_clap_model()
    if clap_model is None or clap_processor is None:
        return jsonify({"error": "CLAP model not loaded"}), 500

    try:
        # Save uploaded file to temp location
        temp_path = DATA_DIR / "temp_example.wav"
        DATA_DIR.mkdir(exist_ok=True)
        file.save(temp_path)

        # Embed the example audio file
        example_embedding = embed_audio_file(temp_path)

        # Clean up temp file
        temp_path.unlink()

        if example_embedding is None:
            return jsonify({"error": "Failed to embed audio file"}), 500

        # Calculate cosine similarity with all clips
        results = []
        scores = []
        for clip_id, clip in clips.items():
            audio_vec = clip["embedding"]
            norm_product = np.linalg.norm(audio_vec) * np.linalg.norm(example_embedding)
            if norm_product == 0:
                similarity = 0.0
            else:
                similarity = float(np.dot(audio_vec, example_embedding) / norm_product)
            results.append({"id": clip_id, "similarity": round(similarity, 4)})
            scores.append(similarity)

        # Calculate GMM-based threshold
        threshold = calculate_gmm_threshold(scores)

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return jsonify({"results": results, "threshold": round(threshold, 4)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@sorting_bp.route("/api/label-file-sort", methods=["POST"])
def label_file_sort():
    """Train MLP on external audio files from a label file, then sort all clips."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    clap_model, clap_processor = get_clap_model()
    if clap_model is None or clap_processor is None:
        return jsonify({"error": "CLAP model not loaded"}), 500

    try:
        # Parse the label file
        text = file.read().decode("utf-8")
        try:
            label_data = json.loads(text)
        except Exception:
            return jsonify({"error": "Invalid label file format"}), 400

        # Extract labels list
        labels = label_data.get("labels", [])
        if not labels:
            return jsonify({"error": "No labels found in file"}), 400

        # Load and embed each labeled audio file
        X_list = []
        y_list = []
        loaded_count = 0
        skipped_count = 0

        for entry in labels:
            label = entry.get("label")
            if label not in ("good", "bad"):
                skipped_count += 1
                continue

            # Try to get audio file path
            audio_path = entry.get("path") or entry.get("file") or entry.get("filename")
            if not audio_path:
                skipped_count += 1
                continue

            audio_path = Path(audio_path)
            if not audio_path.exists():
                skipped_count += 1
                continue

            # Embed the audio file
            embedding = embed_audio_file(audio_path)
            if embedding is None:
                skipped_count += 1
                continue

            X_list.append(embedding)
            y_list.append(1.0 if label == "good" else 0.0)
            loaded_count += 1

        if loaded_count < 2:
            return (
                jsonify(
                    {"error": f"Need at least 2 valid labeled files (loaded {loaded_count}, skipped {skipped_count})"}
                ),
                400,
            )

        # Check if we have both good and bad examples
        num_good = sum(1 for y in y_list if y == 1.0)
        num_bad = len(y_list) - num_good
        if num_good == 0 or num_bad == 0:
            return (
                jsonify({"error": "Need at least one good and one bad labeled example"}),
                400,
            )

        # Train MLP using the same approach as learned sort
        X = torch.tensor(np.array(X_list), dtype=torch.float32)
        y = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1)

        input_dim = X.shape[1]

        # Calculate threshold using cross-calibration
        threshold = calculate_cross_calibration_threshold(X_list, y_list, input_dim, get_inclusion())

        # Train final model on all data
        model = train_model(X, y, input_dim, get_inclusion())

        # Score every clip in the dataset
        all_ids = sorted(clips.keys())
        all_embs = np.array([clips[cid]["embedding"] for cid in all_ids])
        X_all = torch.tensor(all_embs, dtype=torch.float32)
        with torch.no_grad():
            scores = model(X_all).squeeze(1).tolist()

        results = [{"id": cid, "score": round(s, 4)} for cid, s in zip(all_ids, scores)]
        results.sort(key=lambda x: x["score"], reverse=True)

        return jsonify(
            {
                "results": results,
                "threshold": round(threshold, 4),
                "loaded": loaded_count,
                "skipped": skipped_count,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@sorting_bp.route("/api/labeling-progress", methods=["POST"])
def labeling_progress():
    """Analyze labeling progress and calculate stopping condition metrics."""
    if not good_votes or not bad_votes:
        return jsonify({"error": "need at least one good and one bad vote"}), 400

    if not label_history:
        return jsonify({"error": "no label history available"}), 400

    try:
        analysis = analyze_labeling_progress(clips, label_history, good_votes, bad_votes, get_inclusion())
        return jsonify(analysis)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@sorting_bp.route("/api/labeling-status", methods=["GET"])
def labeling_status_indicator():
    """Return a lightweight red/yellow/green labeling status based on the last 10 steps.

    Red  – fewer than 20 labels, or fewer than 5 good or 5 bad.
    Yellow – minimum counts met but error cost is still declining (keep labeling).
    Green  – minimum counts met and error cost has leveled off (safe to stop).
    """
    try:
        status = compute_labeling_status(clips, label_history, good_votes, bad_votes, get_inclusion())
        return jsonify(status)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@sorting_bp.route("/api/favorite-detectors")
def get_favorite_detectors_route():
    """Get all favorite detectors."""
    detectors = get_favorite_detectors()
    return jsonify({"detectors": list(detectors.values())})


@sorting_bp.route("/api/favorite-detectors", methods=["POST"])
def add_favorite_detector_route():
    """Add a new favorite detector."""
    data = request.get_json(force=True)
    if data is None:
        return jsonify({"error": "Invalid request body"}), 400

    name = data.get("name", "").strip()
    media_type = data.get("media_type", "").strip()
    weights = data.get("weights")
    threshold = data.get("threshold", 0.5)

    if not name:
        return jsonify({"error": "name is required"}), 400
    if not media_type:
        return jsonify({"error": "media_type is required"}), 400
    if not weights:
        return jsonify({"error": "weights are required"}), 400

    add_favorite_detector(name, media_type, weights, threshold)
    return jsonify({"success": True, "name": name})


@sorting_bp.route("/api/favorite-detectors/<name>", methods=["DELETE"])
def delete_favorite_detector_route(name):
    """Delete a favorite detector."""
    if remove_favorite_detector(name):
        return jsonify({"success": True})
    return jsonify({"error": "Detector not found"}), 404


@sorting_bp.route("/api/favorite-detectors/<name>/rename", methods=["PUT"])
def rename_favorite_detector_route(name):
    """Rename a favorite detector."""
    data = request.get_json(force=True)
    if data is None:
        return jsonify({"error": "Invalid request body"}), 400

    new_name = data.get("new_name", "").strip()
    if not new_name:
        return jsonify({"error": "new_name is required"}), 400

    if rename_favorite_detector(name, new_name):
        return jsonify({"success": True, "new_name": new_name})
    return jsonify({"error": "Detector not found or new name already exists"}), 400


@sorting_bp.route("/api/favorite-detectors/import-pkl", methods=["POST"])
def import_detector_pkl():
    """Import a favorite detector from a PKL file."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    name = request.form.get("name", "").strip()
    if not name:
        # Use filename without extension as default name
        name = Path(file.filename).stem

    try:
        # Read the JSON detector file
        text = file.read().decode("utf-8")
        detector_data = json.loads(text)

        weights = detector_data.get("weights")
        threshold = detector_data.get("threshold", 0.5)

        if not weights:
            return jsonify({"error": "Invalid detector file format"}), 400

        # Prefer media_type stored in the file; fall back to current clips, then "audio"
        media_type = detector_data.get("media_type", "")
        if not media_type:
            if clips:
                media_type = next(iter(clips.values())).get("type", "audio")
            else:
                media_type = "audio"

        add_favorite_detector(name, media_type, weights, threshold)
        return jsonify({"success": True, "name": name, "media_type": media_type})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@sorting_bp.route("/api/favorite-detectors/import-labels", methods=["POST"])
def import_detector_labels():
    """Import a favorite detector by training on a label file.

    The label file is a JSON object with a ``"labels"`` list. Each entry has
    ``"path"`` (or ``"file"``/``"filename"``) and ``"label"`` (``"good"`` or
    ``"bad"``). Media type is inferred from file extensions; you may also pass
    ``media_type`` as a form field to force a specific type.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    name = request.form.get("name", "").strip()
    if not name:
        name = Path(file.filename).stem

    # Optional explicit media type override
    media_type_hint = request.form.get("media_type", "").strip()

    # Extension → media-type lookup tables
    _AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    _IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
    _VIDEO_EXTS = {".mp4", ".avi", ".mov", ".webm", ".mkv"}
    _TEXT_EXTS = {".txt", ".md"}

    def _media_type_for_path(p: Path) -> str | None:
        ext = p.suffix.lower()
        if ext in _AUDIO_EXTS:
            return "audio"
        if ext in _IMAGE_EXTS:
            return "image"
        if ext in _VIDEO_EXTS:
            return "video"
        if ext in _TEXT_EXTS:
            return "paragraph"
        return None

    def _embed(media_type: str, p: Path):
        if media_type == "audio":
            return embed_audio_file(p)
        if media_type == "image":
            return embed_image_file(p)
        if media_type == "video":
            return embed_video_file(p)
        if media_type == "paragraph":
            return embed_paragraph_file(p)
        return None

    try:
        text = file.read().decode("utf-8")
        try:
            label_data = json.loads(text)
        except Exception:
            return jsonify({"error": "Invalid label file format"}), 400

        labels = label_data.get("labels", [])
        if not labels:
            return jsonify({"error": "No labels found in file"}), 400

        X_list: list = []
        y_list: list = []
        loaded_count = 0
        skipped_count = 0
        detected_media_type: str | None = media_type_hint or None

        for entry in labels:
            label = entry.get("label")
            if label not in ("good", "bad"):
                skipped_count += 1
                continue

            file_path_str = entry.get("path") or entry.get("file") or entry.get("filename")
            if not file_path_str:
                skipped_count += 1
                continue

            file_path = Path(file_path_str)
            if not file_path.exists():
                skipped_count += 1
                continue

            # Resolve media type for this entry
            mt = media_type_hint or _media_type_for_path(file_path)
            if mt is None:
                skipped_count += 1
                continue

            # Enforce a single media type across all entries
            if detected_media_type is None:
                detected_media_type = mt
            elif detected_media_type != mt:
                skipped_count += 1
                continue

            embedding = _embed(mt, file_path)
            if embedding is None:
                skipped_count += 1
                continue

            X_list.append(embedding)
            y_list.append(1.0 if label == "good" else 0.0)
            loaded_count += 1

        if loaded_count < 2:
            return (
                jsonify(
                    {"error": (f"Need at least 2 valid labeled files (loaded {loaded_count}, skipped {skipped_count})")}
                ),
                400,
            )

        num_good = sum(1 for y in y_list if y == 1.0)
        num_bad = len(y_list) - num_good
        if num_good == 0 or num_bad == 0:
            return (
                jsonify({"error": "Need at least one good and one bad labeled example"}),
                400,
            )

        X = torch.tensor(np.array(X_list), dtype=torch.float32)
        y = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1)
        input_dim = X.shape[1]

        threshold = calculate_cross_calibration_threshold(X_list, y_list, input_dim, get_inclusion())
        model = train_model(X, y, input_dim, get_inclusion())

        state_dict = model.state_dict()
        weights = {}
        for key, value in state_dict.items():
            weights[key] = value.tolist()

        final_media_type = detected_media_type or "audio"
        add_favorite_detector(name, final_media_type, weights, threshold)
        return jsonify(
            {
                "success": True,
                "name": name,
                "media_type": final_media_type,
                "loaded": loaded_count,
                "skipped": skipped_count,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@sorting_bp.route("/api/auto-detect", methods=["POST"])
def auto_detect():
    """Run all favorite detectors for the current media type and return positive hits."""
    if not clips:
        return jsonify({"error": "No clips loaded"}), 400

    # Determine media type from current clips
    media_type = next(iter(clips.values())).get("type", "audio")

    # Get favorite detectors for this media type
    detectors = get_favorite_detectors_by_media(media_type)

    if not detectors:
        return jsonify({"error": f"No favorite detectors found for media type: {media_type}"}), 400

    # Run each detector and collect positive hits
    results = {}
    all_ids = sorted(clips.keys())
    all_embs = np.array([clips[cid]["embedding"] for cid in all_ids])
    X_all = torch.tensor(all_embs, dtype=torch.float32)

    for detector_name, detector_data in detectors.items():
        weights = detector_data["weights"]
        threshold = detector_data["threshold"]

        # Reconstruct the model from weights
        input_dim = len(weights["0.weight"][0])

        model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Load weights
        state_dict = {}
        for key, value in weights.items():
            state_dict[key] = torch.tensor(value, dtype=torch.float32)
        model.load_state_dict(state_dict)
        model.eval()

        # Score all clips
        with torch.no_grad():
            scores = model(X_all).squeeze(1).tolist()

        # Collect positive hits (score >= threshold)
        positive_hits = []
        for cid, score in zip(all_ids, scores):
            if score >= threshold:
                clip_info = clips[cid].copy()
                # Don't include embedding or raw media in response
                clip_info.pop("embedding", None)
                clip_info.pop("wav_bytes", None)
                clip_info.pop("video_bytes", None)
                clip_info.pop("image_bytes", None)
                clip_info.pop("text_content", None)
                clip_info["score"] = round(score, 4)
                positive_hits.append(clip_info)

        # Sort by score descending
        positive_hits.sort(key=lambda x: x["score"], reverse=True)

        results[detector_name] = {
            "detector_name": detector_name,
            "threshold": round(threshold, 4),
            "total_hits": len(positive_hits),
            "hits": positive_hits,
        }

    return jsonify(
        {
            "media_type": media_type,
            "detectors_run": len(detectors),
            "results": results,
        }
    )


# ---------------------------------------------------------------------------
# Extractor routes
# ---------------------------------------------------------------------------

# Registry of extractor type constructors.
# Each entry maps an extractor_type string to a callable(name, config) -> Extractor.
_EXTRACTOR_FACTORIES: dict = {}


def _ensure_extractor_factories():
    """Populate the factory registry on first use (lazy to avoid import cycles)."""
    if _EXTRACTOR_FACTORIES:
        return
    from vistatotes.media.image.extractor import ImageClassExtractor

    _EXTRACTOR_FACTORIES["image_class"] = ImageClassExtractor.from_config


def _build_extractor(name: str, extractor_type: str, config: dict):
    """Instantiate an Extractor from its serialised form."""
    _ensure_extractor_factories()
    factory = _EXTRACTOR_FACTORIES.get(extractor_type)
    if factory is None:
        raise ValueError(f"Unknown extractor_type: {extractor_type!r}")
    return factory(name, config)


@sorting_bp.route("/api/favorite-extractors")
def get_favorite_extractors_route():
    """Get all favorite extractors."""
    extractors = get_favorite_extractors()
    return jsonify({"extractors": list(extractors.values())})


@sorting_bp.route("/api/favorite-extractors", methods=["POST"])
def add_favorite_extractor_route():
    """Add a new favorite extractor."""
    data = request.get_json(force=True)
    if data is None:
        return jsonify({"error": "Invalid request body"}), 400

    name = data.get("name", "").strip()
    extractor_type = data.get("extractor_type", "").strip()
    media_type = data.get("media_type", "").strip()
    config = data.get("config")

    if not name:
        return jsonify({"error": "name is required"}), 400
    if not extractor_type:
        return jsonify({"error": "extractor_type is required"}), 400
    if not media_type:
        return jsonify({"error": "media_type is required"}), 400
    if not config or not isinstance(config, dict):
        return jsonify({"error": "config is required"}), 400

    # Validate that the extractor can be built from this config
    try:
        _build_extractor(name, extractor_type, config)
    except Exception as e:
        return jsonify({"error": f"Invalid extractor config: {e}"}), 400

    add_favorite_extractor(name, extractor_type, media_type, config)
    return jsonify({"success": True, "name": name})


@sorting_bp.route("/api/favorite-extractors/<name>", methods=["DELETE"])
def delete_favorite_extractor_route(name):
    """Delete a favorite extractor."""
    if remove_favorite_extractor(name):
        return jsonify({"success": True})
    return jsonify({"error": "Extractor not found"}), 404


@sorting_bp.route("/api/favorite-extractors/<name>/rename", methods=["PUT"])
def rename_favorite_extractor_route(name):
    """Rename a favorite extractor."""
    data = request.get_json(force=True)
    if data is None:
        return jsonify({"error": "Invalid request body"}), 400

    new_name = data.get("new_name", "").strip()
    if not new_name:
        return jsonify({"error": "new_name is required"}), 400

    if rename_favorite_extractor(name, new_name):
        return jsonify({"success": True, "new_name": new_name})
    return jsonify({"error": "Extractor not found or new name already exists"}), 400


@sorting_bp.route("/api/extract", methods=["POST"])
def run_extract():
    """Run a single extractor on all clips and return per-clip extraction results."""
    data = request.get_json(force=True)
    if data is None:
        return jsonify({"error": "Invalid request body"}), 400

    extractor_name = data.get("name", "").strip()
    extractor_type = data.get("extractor_type", "").strip()
    config = data.get("config")

    if not extractor_type:
        return jsonify({"error": "extractor_type is required"}), 400
    if not config or not isinstance(config, dict):
        return jsonify({"error": "config is required"}), 400

    if not clips:
        return jsonify({"error": "No clips loaded"}), 400

    try:
        extractor = _build_extractor(extractor_name or "adhoc", extractor_type, config)
    except Exception as e:
        return jsonify({"error": f"Invalid extractor config: {e}"}), 400

    media_type = next(iter(clips.values())).get("type", "")
    if extractor.media_type != media_type:
        return (
            jsonify({"error": f"Extractor media type '{extractor.media_type}' does not match clips '{media_type}'"}),
            400,
        )

    results = []
    for clip_id in sorted(clips.keys()):
        clip = clips[clip_id]
        extractions = extractor.extract(clip)
        if extractions:
            clip_info = {
                k: v
                for k, v in clip.items()
                if k not in ("embedding", "wav_bytes", "video_bytes", "image_bytes", "text_content")
            }
            clip_info["extractions"] = extractions
            results.append(clip_info)

    return jsonify(
        {
            "extractor_name": extractor.name,
            "media_type": media_type,
            "total_clips_with_hits": len(results),
            "results": results,
        }
    )


@sorting_bp.route("/api/auto-extract", methods=["POST"])
def auto_extract():
    """Run all favorite extractors for the current media type and return extraction results."""
    if not clips:
        return jsonify({"error": "No clips loaded"}), 400

    media_type = next(iter(clips.values())).get("type", "")
    extractors = get_favorite_extractors_by_media(media_type)

    if not extractors:
        return jsonify({"error": f"No favorite extractors found for media type: {media_type}"}), 400

    results = {}
    for ext_name, ext_data in extractors.items():
        try:
            extractor = _build_extractor(ext_name, ext_data["extractor_type"], ext_data["config"])
        except Exception:
            continue

        ext_results = []
        for clip_id in sorted(clips.keys()):
            clip = clips[clip_id]
            extractions = extractor.extract(clip)
            if extractions:
                clip_info = {
                    k: v
                    for k, v in clip.items()
                    if k not in ("embedding", "wav_bytes", "video_bytes", "image_bytes", "text_content")
                }
                clip_info["extractions"] = extractions
                ext_results.append(clip_info)

        results[ext_name] = {
            "extractor_name": ext_name,
            "total_clips_with_hits": len(ext_results),
            "results": ext_results,
        }

    return jsonify(
        {
            "media_type": media_type,
            "extractors_run": len(results),
            "results": results,
        }
    )
