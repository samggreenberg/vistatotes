"""Blueprint for clip-related routes."""

import io

from flask import Blueprint, jsonify, request, send_file

from vistatotes.utils import add_label_to_history, bad_votes, clips, good_votes

clips_bp = Blueprint("clips", __name__)


@clips_bp.route("/api/clips")
def list_clips():
    result = []
    for c in clips.values():
        clip_data = {
            "id": c["id"],
            "type": c.get("type", "audio"),
            "duration": c["duration"],
            "file_size": c["file_size"],
            "filename": c.get("filename", f"clip_{c['id']}.wav"),
            "category": c.get("category", "unknown"),
            "md5": c["md5"],
        }
        # Only include frequency if it exists (for synthetic clips)
        if "frequency" in c:
            clip_data["frequency"] = c["frequency"]
        result.append(clip_data)
    return jsonify(result)


@clips_bp.route("/api/clips/<int:clip_id>/audio")
def clip_audio(clip_id):
    c = clips.get(clip_id)
    if not c:
        return jsonify({"error": "not found"}), 404
    return send_file(
        io.BytesIO(c["wav_bytes"]),
        mimetype="audio/wav",
        download_name=f"clip_{clip_id}.wav",
    )


@clips_bp.route("/api/clips/<int:clip_id>/video")
def clip_video(clip_id):
    c = clips.get(clip_id)
    if not c:
        return jsonify({"error": "not found"}), 404
    if c.get("type") != "video" or not c.get("video_bytes"):
        return jsonify({"error": "not a video clip"}), 400

    # Determine mimetype based on filename extension
    filename = c.get("filename", "")
    if filename.endswith(".webm"):
        mimetype = "video/webm"
    elif filename.endswith(".mov"):
        mimetype = "video/quicktime"
    elif filename.endswith(".avi"):
        mimetype = "video/x-msvideo"
    else:
        mimetype = "video/mp4"

    return send_file(
        io.BytesIO(c["video_bytes"]),
        mimetype=mimetype,
        download_name=f"clip_{clip_id}.mp4",
    )


@clips_bp.route("/api/clips/<int:clip_id>/image")
def clip_image(clip_id):
    c = clips.get(clip_id)
    if not c:
        return jsonify({"error": "not found"}), 404
    if c.get("type") != "image" or not c.get("image_bytes"):
        return jsonify({"error": "not an image clip"}), 400

    # Determine mimetype based on filename extension
    filename = c.get("filename", "")
    if filename.endswith(".png"):
        mimetype = "image/png"
    elif filename.endswith(".gif"):
        mimetype = "image/gif"
    elif filename.endswith(".webp"):
        mimetype = "image/webp"
    elif filename.endswith(".bmp"):
        mimetype = "image/bmp"
    else:
        mimetype = "image/jpeg"

    return send_file(
        io.BytesIO(c["image_bytes"]),
        mimetype=mimetype,
        download_name=f"clip_{clip_id}.jpg",
    )


@clips_bp.route("/api/clips/<int:clip_id>/paragraph")
def clip_paragraph(clip_id):
    c = clips.get(clip_id)
    if not c:
        return jsonify({"error": "not found"}), 404
    if c.get("type") != "paragraph" or not c.get("text_content"):
        return jsonify({"error": "not a paragraph clip"}), 400

    return jsonify(
        {
            "content": c.get("text_content", ""),
            "word_count": c.get("word_count", 0),
            "character_count": c.get("character_count", 0),
        }
    )


@clips_bp.route("/api/clips/<int:clip_id>/vote", methods=["POST"])
def vote_clip(clip_id):
    if clip_id not in clips:
        print(f"DEBUG: Vote failed - Clip {clip_id} not found", flush=True)
        return jsonify({"error": "not found"}), 404

    try:
        data = request.get_json(force=True)
    except Exception as e:
        print(f"DEBUG: Vote failed - JSON error: {e}", flush=True)
        return jsonify({"error": "Invalid request body"}), 400

    if data is None:
        return jsonify({"error": "Invalid request body"}), 400

    vote = data.get("vote")
    if vote not in ("good", "bad"):
        print(f"DEBUG: Vote failed - Invalid vote '{vote}'", flush=True)
        return jsonify({"error": "vote must be 'good' or 'bad'"}), 400

    if vote == "good":
        if clip_id in good_votes:
            good_votes.pop(clip_id, None)
        else:
            bad_votes.pop(clip_id, None)
            good_votes[clip_id] = None
            add_label_to_history(clip_id, "good")
    else:
        if clip_id in bad_votes:
            bad_votes.pop(clip_id, None)
        else:
            good_votes.pop(clip_id, None)
            bad_votes[clip_id] = None
            add_label_to_history(clip_id, "bad")

    print(f"DEBUG: Vote '{vote}' recorded for clip {clip_id}", flush=True)
    return jsonify({"ok": True})
