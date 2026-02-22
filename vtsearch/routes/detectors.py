"""Blueprint for detector and extractor routes."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from flask import Blueprint, jsonify, request

from vtsearch.models import (
    calculate_cross_calibration_threshold,
    embed_audio_file,
    embed_image_file,
    embed_paragraph_file,
    embed_video_file,
    train_model,
)
from vtsearch.utils import (
    add_favorite_detector,
    add_favorite_extractor,
    clips,
    get_favorite_detectors,
    get_favorite_detectors_by_media,
    get_favorite_extractors,
    get_favorite_extractors_by_media,
    get_inclusion,
    remove_favorite_detector,
    remove_favorite_extractor,
    rename_favorite_detector,
    rename_favorite_extractor,
)

detectors_bp = Blueprint("detectors", __name__)


# ---------------------------------------------------------------------------
# Detector routes
# ---------------------------------------------------------------------------


@detectors_bp.route("/api/detector/export", methods=["POST"])
def export_detector():
    """Train MLP on current votes and export the model weights."""
    from vtsearch.utils import bad_votes, good_votes

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


@detectors_bp.route("/api/detector-sort", methods=["POST"])
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


# ---------------------------------------------------------------------------
# Favorite detectors
# ---------------------------------------------------------------------------


@detectors_bp.route("/api/favorite-detectors")
def get_favorite_detectors_route():
    """Get all favorite detectors."""
    detectors = get_favorite_detectors()
    return jsonify({"detectors": list(detectors.values())})


@detectors_bp.route("/api/favorite-detectors", methods=["POST"])
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


@detectors_bp.route("/api/favorite-detectors/<name>", methods=["DELETE"])
def delete_favorite_detector_route(name):
    """Delete a favorite detector."""
    if remove_favorite_detector(name):
        return jsonify({"success": True})
    return jsonify({"error": "Detector not found"}), 404


@detectors_bp.route("/api/favorite-detectors/<name>/rename", methods=["PUT"])
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


@detectors_bp.route("/api/favorite-detectors/import-pkl", methods=["POST"])
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


@detectors_bp.route("/api/favorite-detectors/import-labels", methods=["POST"])
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


@detectors_bp.route("/api/auto-detect", methods=["POST"])
def auto_detect():
    """Run all favorite detectors for the current media type and return positive hits."""
    if not clips:
        return jsonify({"error": "No clips loaded"}), 400

    # Import any favorite processors from settings that aren't already loaded
    from vtsearch.settings import ensure_favorite_processors_imported

    newly_imported = ensure_favorite_processors_imported()

    # Determine media type from current clips
    media_type = next(iter(clips.values())).get("type", "audio")

    # Get favorite detectors for this media type
    detectors = get_favorite_detectors_by_media(media_type)

    if not detectors:
        return jsonify({"error": f"No favorite detectors found for media type: {media_type}"}), 400

    # Prepare shared data for all detectors
    all_ids = sorted(clips.keys())
    all_embs = np.array([clips[cid]["embedding"] for cid in all_ids])
    X_all = torch.tensor(all_embs, dtype=torch.float32)

    def _run_single_detector(detector_name, detector_data):
        """Run a single detector and return (name, result_dict)."""
        weights = detector_data["weights"]
        threshold = detector_data["threshold"]

        input_dim = len(weights["0.weight"][0])
        model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        state_dict = {}
        for key, value in weights.items():
            state_dict[key] = torch.tensor(value, dtype=torch.float32)
        model.load_state_dict(state_dict)
        model.eval()

        with torch.no_grad():
            scores = model(X_all).squeeze(1).tolist()

        positive_hits = []
        for cid, score in zip(all_ids, scores):
            if score >= threshold:
                clip_info = clips[cid].copy()
                clip_info.pop("embedding", None)
                clip_info.pop("wav_bytes", None)
                clip_info.pop("video_bytes", None)
                clip_info.pop("image_bytes", None)
                clip_info.pop("text_content", None)
                clip_info["score"] = round(score, 4)
                positive_hits.append(clip_info)

        positive_hits.sort(key=lambda x: x["score"], reverse=True)

        return detector_name, {
            "detector_name": detector_name,
            "threshold": round(threshold, 4),
            "total_hits": len(positive_hits),
            "hits": positive_hits,
        }

    # Run all detectors in parallel (PyTorch releases GIL during tensor ops)
    results = {}
    with ThreadPoolExecutor(max_workers=len(detectors)) as pool:
        futures = [pool.submit(_run_single_detector, name, data) for name, data in detectors.items()]
        for future in futures:
            name, result = future.result()
            results[name] = result

    response: dict = {
        "media_type": media_type,
        "detectors_run": len(detectors),
        "results": results,
    }
    if newly_imported:
        response["newly_imported"] = newly_imported

    return jsonify(response)


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
    from vtsearch.media.image.extractor import ImageClassExtractor

    _EXTRACTOR_FACTORIES["image_class"] = ImageClassExtractor.from_config


def _build_extractor(name: str, extractor_type: str, config: dict):
    """Instantiate an Extractor from its serialised form."""
    _ensure_extractor_factories()
    factory = _EXTRACTOR_FACTORIES.get(extractor_type)
    if factory is None:
        raise ValueError(f"Unknown extractor_type: {extractor_type!r}")
    return factory(name, config)


@detectors_bp.route("/api/favorite-extractors")
def get_favorite_extractors_route():
    """Get all favorite extractors."""
    extractors = get_favorite_extractors()
    return jsonify({"extractors": list(extractors.values())})


@detectors_bp.route("/api/favorite-extractors", methods=["POST"])
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


@detectors_bp.route("/api/favorite-extractors/<name>", methods=["DELETE"])
def delete_favorite_extractor_route(name):
    """Delete a favorite extractor."""
    if remove_favorite_extractor(name):
        return jsonify({"success": True})
    return jsonify({"error": "Extractor not found"}), 404


@detectors_bp.route("/api/favorite-extractors/<name>/rename", methods=["PUT"])
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


@detectors_bp.route("/api/extract", methods=["POST"])
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


@detectors_bp.route("/api/auto-extract", methods=["POST"])
def auto_extract():
    """Run all favorite extractors for the current media type and return extraction results."""
    if not clips:
        return jsonify({"error": "No clips loaded"}), 400

    media_type = next(iter(clips.values())).get("type", "")
    extractors = get_favorite_extractors_by_media(media_type)

    if not extractors:
        return jsonify({"error": f"No favorite extractors found for media type: {media_type}"}), 400

    sorted_clip_ids = sorted(clips.keys())

    def _run_single_extractor(ext_name, ext_data):
        """Run a single extractor on all clips and return (name, result_dict) or None."""
        try:
            extractor = _build_extractor(ext_name, ext_data["extractor_type"], ext_data["config"])
        except Exception:
            return None

        ext_results = []
        for clip_id in sorted_clip_ids:
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

        return ext_name, {
            "extractor_name": ext_name,
            "total_clips_with_hits": len(ext_results),
            "results": ext_results,
        }

    # Run all extractors in parallel
    results = {}
    with ThreadPoolExecutor(max_workers=len(extractors)) as pool:
        futures = [pool.submit(_run_single_extractor, name, data) for name, data in extractors.items()]
        for future in futures:
            outcome = future.result()
            if outcome is not None:
                name, result = outcome
                results[name] = result

    return jsonify(
        {
            "media_type": media_type,
            "extractors_run": len(results),
            "results": results,
        }
    )
