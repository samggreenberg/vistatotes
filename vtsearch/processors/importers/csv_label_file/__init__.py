"""CSV label-file processor importer -- trains a detector from a CSV of labelled media files.

This importer reads a CSV file containing labelled media paths, embeds each
file using the appropriate embedding model, and trains an MLP detector on the
resulting vectors.  The CSV format is::

    path,label
    /data/audio/dog_bark.wav,good
    /data/audio/silence.wav,bad

Media type is inferred from file extensions (or can be overridden via the
optional ``media_type`` field).  Requires the embedding models for the detected
media type to be loadable on the current machine.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

from vtsearch.processors.importers.base import ProcessorImporter, ProcessorImporterField


class CsvLabelFileProcessorImporter(ProcessorImporter):
    """Train a detector from a CSV label file with media file paths.

    Each row must have a ``"path"`` (or ``"file"`` or ``"filename"``) column
    pointing to a local media file, and a ``"label"`` column set to ``"good"``
    or ``"bad"``.  Media type is inferred from file extensions; supply the
    ``media_type`` field to override.

    The importer embeds all referenced files, trains an MLP detector using
    cross-calibrated thresholding, and returns the resulting weights and
    threshold.
    """

    name = "csv_label_file"
    display_name = "Label File (.csv)"
    description = "Train a new detector from a CSV file listing labelled media paths."
    icon = "\U0001f4ca"  # bar chart
    fields = [
        ProcessorImporterField(
            key="file",
            label="Labels CSV File",
            field_type="file",
            accept=".csv",
            description="A CSV file with 'path' and 'label' columns.",
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
        """Parse the uploaded CSV label file, embed media, train a detector."""
        file_storage = field_values.get("file")
        if file_storage is None:
            raise ValueError("No file provided.")
        try:
            raw = file_storage.read()
        except AttributeError:
            raise ValueError("Expected a file upload, not a string. Use run_cli for CLI usage.")
        media_type_hint = (field_values.get("media_type") or "").strip()
        return _train_from_csv_labels(raw, media_type_hint)

    def run_cli(self, field_values: dict[str, Any]) -> dict[str, Any]:
        """Train a detector from a file-path string (CLI usage)."""
        filepath = field_values.get("file", "").strip()
        if not filepath:
            raise ValueError("--file is required.")
        raw = Path(filepath).read_bytes()
        media_type_hint = (field_values.get("media_type") or "").strip()
        return _train_from_csv_labels(raw, media_type_hint)

    def add_cli_arguments(self, parser: Any) -> None:
        parser.add_argument(
            "--file",
            dest="file",
            help="Path to a CSV file with labelled media paths.",
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


def _parse_csv_bytes(raw: bytes) -> list[dict[str, str]]:
    """Decode *raw* bytes as CSV and extract path/label pairs."""
    try:
        text = raw.decode("utf-8-sig")  # strip BOM if present
    except UnicodeDecodeError:
        text = raw.decode("latin-1")

    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise ValueError("CSV file appears to be empty.")

    # Normalise header names to lower-case and strip whitespace
    normalised = {k.strip().lower(): k for k in reader.fieldnames if k}

    # Accept path/file/filename for the path column
    path_key = None
    for candidate in ("path", "file", "filename"):
        if candidate in normalised:
            path_key = normalised[candidate]
            break
    if path_key is None:
        raise ValueError("CSV must have a 'path' (or 'file' or 'filename') column header.")
    if "label" not in normalised:
        raise ValueError("CSV must have a 'label' column header.")

    label_key = normalised["label"]

    results = []
    for row in reader:
        path_val = row.get(path_key, "").strip()
        label_val = row.get(label_key, "").strip().lower()
        if path_val and label_val:
            results.append({"path": path_val, "label": label_val})
    return results


def _train_from_csv_labels(raw: bytes, media_type_hint: str) -> dict[str, Any]:
    """Parse CSV label file, embed referenced files, and train an MLP detector."""
    # Re-use the training logic from the JSON label_file importer
    from vtsearch.processors.importers.label_file import _media_type_for_path, _embed

    import numpy as np
    import torch

    from vtsearch.models import calculate_cross_calibration_threshold, train_model
    from vtsearch.utils import get_inclusion

    entries = _parse_csv_bytes(raw)
    if not entries:
        raise ValueError("No labels found in CSV file.")

    X_list: list = []
    y_list: list = []
    loaded_count = 0
    skipped_count = 0
    detected_media_type: str | None = media_type_hint or None

    for entry in entries:
        label = entry.get("label")
        if label not in ("good", "bad"):
            skipped_count += 1
            continue

        file_path = Path(entry["path"])
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


PROCESSOR_IMPORTER = CsvLabelFileProcessorImporter()
