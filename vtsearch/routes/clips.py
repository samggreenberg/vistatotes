"""Blueprint for clip-related routes."""

import io
from pathlib import Path
from typing import Any

from flask import Blueprint, Response, jsonify, request, send_file

from vtsearch.media.base import MediaResponse
from vtsearch.utils import add_label_to_history, bad_votes, clips, good_votes

clips_bp = Blueprint("clips", __name__)


def _flask_response(mr: MediaResponse) -> Response:
    """Convert a framework-agnostic :class:`MediaResponse` to a Flask response."""
    if isinstance(mr.data, dict):
        return jsonify(mr.data)
    return send_file(io.BytesIO(mr.data), mimetype=mr.mimetype, download_name=mr.download_name)


@clips_bp.route("/api/clips")
def list_clips() -> Response:
    """Return metadata for all loaded clips as a JSON array.

    Excludes heavyweight fields (``embedding``, ``clip_bytes``,
    ``clip_string``) from the response. Only includes the
    ``frequency`` field when it is present (synthetic clips only).

    Returns:
        A JSON array of clip metadata dicts, each containing: ``id``, ``type``,
        ``duration``, ``file_size``, ``filename``, ``category``, ``md5``, and
        optionally ``frequency``.
    """
    result: list[dict[str, Any]] = []
    for c in clips.values():
        clip_data: dict[str, Any] = {
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
def clip_audio(clip_id: int) -> tuple[Response, int] | Response:
    """Stream the WAV audio bytes for a single clip.

    Args:
        clip_id: Integer clip ID from the URL path.

    Returns:
        A ``audio/wav`` file response on success (HTTP 200), or a JSON error
        response with HTTP 404 if the clip does not exist.
    """
    c = clips.get(clip_id)
    if not c:
        return jsonify({"error": "not found"}), 404
    return send_file(
        io.BytesIO(c["clip_bytes"]),
        mimetype="audio/wav",
        download_name=f"clip_{clip_id}.wav",
    )


@clips_bp.route("/api/clips/<int:clip_id>/video")
def clip_video(clip_id: int) -> tuple[Response, int] | Response:
    """Stream the video bytes for a single video clip.

    Determines the MIME type from the clip's filename extension, defaulting to
    ``video/mp4`` for unrecognised extensions.

    Args:
        clip_id: Integer clip ID from the URL path.

    Returns:
        A video file response with the appropriate MIME type on success
        (HTTP 200), a JSON 404 error if the clip does not exist, or a JSON 400
        error if the clip exists but is not of type ``"video"``.
    """
    c = clips.get(clip_id)
    if not c:
        return jsonify({"error": "not found"}), 404
    if c.get("type") != "video" or not c.get("clip_bytes"):
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

    ext = Path(filename).suffix if filename else ".mp4"
    return send_file(
        io.BytesIO(c["clip_bytes"]),
        mimetype=mimetype,
        download_name=f"clip_{clip_id}{ext}",
    )


@clips_bp.route("/api/clips/<int:clip_id>/image")
def clip_image(clip_id: int) -> tuple[Response, int] | Response:
    """Stream the image bytes for a single image clip.

    Determines the MIME type from the clip's filename extension, defaulting to
    ``image/jpeg`` for unrecognised extensions.

    Args:
        clip_id: Integer clip ID from the URL path.

    Returns:
        An image file response with the appropriate MIME type on success
        (HTTP 200), a JSON 404 error if the clip does not exist, or a JSON 400
        error if the clip exists but is not of type ``"image"``.
    """
    c = clips.get(clip_id)
    if not c:
        return jsonify({"error": "not found"}), 404
    if c.get("type") != "image" or not c.get("clip_bytes"):
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
        io.BytesIO(c["clip_bytes"]),
        mimetype=mimetype,
        download_name=f"clip_{clip_id}.jpg",
    )


@clips_bp.route("/api/clips/<int:clip_id>/paragraph")
def clip_paragraph(clip_id: int) -> tuple[Response, int] | Response:
    """Return the text content and statistics for a single paragraph clip.

    Args:
        clip_id: Integer clip ID from the URL path.

    Returns:
        A JSON object with keys ``"content"`` (str), ``"word_count"`` (int),
        and ``"character_count"`` (int) on success (HTTP 200), a JSON 404
        error if the clip does not exist, or a JSON 400 error if the clip
        exists but is not of type ``"paragraph"``.
    """
    c = clips.get(clip_id)
    if not c:
        return jsonify({"error": "not found"}), 404
    if c.get("type") != "paragraph" or not c.get("clip_string"):
        return jsonify({"error": "not a paragraph clip"}), 400

    return jsonify(
        {
            "content": c.get("clip_string", ""),
            "word_count": c.get("word_count", 0),
            "character_count": c.get("character_count", 0),
        }
    )


@clips_bp.route("/api/clips/<int:clip_id>/media")
def clip_media(clip_id: int) -> tuple[Response, int] | Response:
    """Serve the media content for any clip type via a single generic endpoint.

    Determines the media type from the clip's ``"type"`` field and delegates
    to the registered :class:`~vtsearch.media.base.MediaType`'s
    :meth:`~vtsearch.media.base.MediaType.clip_response` method.  This
    endpoint works for all current and future media types without modification.

    Args:
        clip_id: Integer clip ID from the URL path.

    Returns:
        The media content with the appropriate MIME type on success (HTTP 200),
        or a JSON error response for HTTP 404 (clip not found) or HTTP 400
        (unrecognised media type).
    """
    c = clips.get(clip_id)
    if not c:
        return jsonify({"error": "not found"}), 404

    from vtsearch.media import get as media_get

    try:
        mt = media_get(c.get("type", ""))
    except KeyError:
        return jsonify({"error": f"unsupported media type: {c.get('type')}"}), 400

    return _flask_response(mt.clip_response(c))


@clips_bp.route("/api/clips/<int:clip_id>/vote", methods=["POST"])
def vote_clip(clip_id: int) -> tuple[Response, int] | Response:
    """Record or toggle a good/bad vote for a single clip.

    Voting behaviour (toggle semantics):

    - If ``vote == "good"`` and the clip is already in ``good_votes``, the vote
      is *removed* (toggled off).
    - If ``vote == "good"`` and the clip is not yet in ``good_votes``, it is
      added to ``good_votes`` (removed from ``bad_votes`` if present) and the
      event is appended to ``label_history``.
    - The same toggle logic applies symmetrically for ``vote == "bad"``.

    Args:
        clip_id: Integer clip ID from the URL path.

    Request body (JSON):
        ``{"vote": "good"}`` or ``{"vote": "bad"}``.

    Returns:
        ``{"ok": True}`` (HTTP 200) on success, or a JSON error response for:

        - HTTP 404 – clip not found.
        - HTTP 400 – request body is missing, malformed, or ``vote`` is not
          ``"good"`` or ``"bad"``.
    """
    if clip_id not in clips:
        return jsonify({"error": "not found"}), 404

    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid request body"}), 400

    if data is None:
        return jsonify({"error": "Invalid request body"}), 400

    vote = data.get("vote")
    if vote not in ("good", "bad"):
        return jsonify({"error": "vote must be 'good' or 'bad'"}), 400

    if vote == "good":
        if clip_id in good_votes:
            good_votes.pop(clip_id, None)
            add_label_to_history(clip_id, "unlabel")
        else:
            bad_votes.pop(clip_id, None)
            good_votes[clip_id] = None
            add_label_to_history(clip_id, "good")
    else:
        if clip_id in bad_votes:
            bad_votes.pop(clip_id, None)
            add_label_to_history(clip_id, "unlabel")
        else:
            good_votes.pop(clip_id, None)
            bad_votes[clip_id] = None
            add_label_to_history(clip_id, "bad")

    return jsonify({"ok": True})
