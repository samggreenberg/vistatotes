"""Blueprint for sorting and voting routes."""

import json
from pathlib import Path

import numpy as np
import torch
from flask import Blueprint, jsonify, request

from config import DATA_DIR
from vtsearch.models import (
    analyze_labeling_progress,
    calculate_cross_calibration_threshold,
    calculate_gmm_threshold,
    compute_labeling_status,
    embed_audio_file,
    embed_text_query,
    get_clap_model,
    train_and_score,
    train_model,
)
from vtsearch.utils import (
    add_label_to_history,
    bad_votes,
    clips,
    get_inclusion,
    get_sort_progress,
    good_votes,
    label_history,
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
    except Exception:
        update_sort_progress("idle")
        return jsonify({"error": "Invalid request body"}), 400

    if data is None:
        update_sort_progress("idle")
        return jsonify({"error": "Invalid request body"}), 400

    text = data.get("text", "").strip()

    if not text:
        update_sort_progress("idle")
        return jsonify({"error": "text is required"}), 400

    # Determine media type from current clips
    if not clips:
        update_sort_progress("idle")
        return jsonify({"error": "No clips loaded"}), 400

    media_type = next(iter(clips.values())).get("type", "audio")

    # Total steps: 1 (embed) + len(clips) (similarities) + 1 (threshold)
    total_steps = 1 + len(clips) + 1

    # Check if the embedder needs loading (first use of this media type)
    from vtsearch.media import get as media_get

    try:
        mt = media_get(media_type)
        needs_loading = getattr(mt, "_model", None) is None
    except KeyError:
        needs_loading = False
    if needs_loading:
        update_sort_progress("sorting", "Loading embedder…", 0, total_steps)
    else:
        update_sort_progress("sorting", "Embedding text query…", 0, total_steps)

    # Embed text query using refactored module
    text_vec = embed_text_query(text, media_type)
    if text_vec is None:
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

        # Sort by raw scores (full precision) before rounding for display.
        paired = sorted(zip(all_ids, scores), key=lambda x: x[1], reverse=True)
        results = [{"id": cid, "score": round(s, 4)} for cid, s in paired]

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
