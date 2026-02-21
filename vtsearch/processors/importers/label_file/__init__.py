"""Label-file processor importer -- trains a detector from labelled media files.

This importer reads a JSON file containing labelled media paths, embeds each
file using the appropriate embedding model, and trains an MLP detector on the
resulting vectors.  The JSON format is::

    {
        "labels": [
            {"path": "/data/audio/dog_bark.wav", "label": "good"},
            {"path": "/data/audio/silence.wav",  "label": "bad"},
            ...
        ]
    }

Media type is inferred from file extensions (or can be overridden via the
optional ``media_type`` field).  Requires the embedding models for the detected
media type to be loadable on the current machine.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vtsearch.processors.importers.base import ProcessorImporter, ProcessorImporterField


class LabelFileProcessorImporter(ProcessorImporter):
    """Train a detector from a JSON label file with media file paths.

    Each entry in the ``"labels"`` list must have a ``"path"`` (or ``"file"``
    or ``"filename"``) key pointing to a local media file, and a ``"label"``
    key set to ``"good"`` or ``"bad"``.  Media type is inferred from file
    extensions; supply the ``media_type`` field to override.

    The importer embeds all referenced files, trains an MLP detector using
    cross-calibrated thresholding, and returns the resulting weights and
    threshold.
    """

    name = "label_file"
    display_name = "Label File (.json)"
    description = "Train a new detector from a JSON file listing labelled media paths."
    icon = "\U0001f3f7\ufe0f"  # label
    fields = [
        ProcessorImporterField(
            key="file",
            label="Labels JSON File",
            field_type="file",
            accept=".json",
            description="A JSON file with a 'labels' list of {path, label} objects.",
        ),
        ProcessorImporterField(
            key="media_type",
            label="Media Type",
            field_type="select",
            options=["", "audio", "image", "video", "paragraph"],
            default="",
            required=False,
            description="Override auto-detected media type (leave blank to auto-detect).",
        ),
    ]

    def run(self, field_values: dict[str, Any]) -> dict[str, Any]:
        """Parse the uploaded label file, embed media, train a detector.

        In the GUI path ``field_values["file"]`` is a Werkzeug
        ``FileStorage`` object; in the CLI path it is a plain file-path
        string.  Use :meth:`run_cli` for the CLI path.
        """
        file_storage = field_values.get("file")
        if file_storage is None:
            raise ValueError("No file provided.")
        try:
            raw = file_storage.read()
        except AttributeError:
            raise ValueError("Expected a file upload, not a string. Use run_cli for CLI usage.")
        media_type_hint = (field_values.get("media_type") or "").strip()
        return _train_from_labels(raw, media_type_hint)

    def run_cli(self, field_values: dict[str, Any]) -> dict[str, Any]:
        """Train a detector from a file-path string (CLI usage)."""
        filepath = field_values.get("file", "").strip()
        if not filepath:
            raise ValueError("--file is required.")
        raw = Path(filepath).read_bytes()
        media_type_hint = (field_values.get("media_type") or "").strip()
        return _train_from_labels(raw, media_type_hint)

    def add_cli_arguments(self, parser: Any) -> None:
        parser.add_argument(
            "--file",
            dest="file",
            help="Path to a JSON file with labelled media paths.",
            required=False,
        )
        parser.add_argument(
            "--media-type",
            dest="media_type",
            help="Override auto-detected media type.",
            choices=["audio", "image", "video", "paragraph"],
            default="",
            required=False,
        )


# ---- Extension-to-media-type lookup tables ----

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
    from vtsearch.models import embed_audio_file, embed_image_file, embed_paragraph_file, embed_video_file

    if media_type == "audio":
        return embed_audio_file(p)
    if media_type == "image":
        return embed_image_file(p)
    if media_type == "video":
        return embed_video_file(p)
    if media_type == "paragraph":
        return embed_paragraph_file(p)
    return None


def _train_from_labels(raw: bytes, media_type_hint: str) -> dict[str, Any]:
    """Parse label JSON, embed referenced files, and train an MLP detector."""
    import numpy as np
    import torch

    from vtsearch.models import calculate_cross_calibration_threshold, train_model
    from vtsearch.utils import get_inclusion

    try:
        data = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc

    labels = data.get("labels", [])
    if not labels:
        raise ValueError("No labels found in file.")

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
        raise ValueError(f"Need at least 2 valid labeled files (loaded {loaded_count}, skipped {skipped_count})")

    num_good = sum(1 for y in y_list if y == 1.0)
    num_bad = len(y_list) - num_good
    if num_good == 0 or num_bad == 0:
        raise ValueError("Need at least one good and one bad labeled example")

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
    return {
        "media_type": final_media_type,
        "weights": weights,
        "threshold": round(threshold, 4),
        "loaded": loaded_count,
        "skipped": skipped_count,
    }


PROCESSOR_IMPORTER = LabelFileProcessorImporter()
