"""Detector-file processor importer -- loads a detector from a ``.json`` file.

This importer reads a VTSearch detector JSON file containing pre-trained MLP
weights and a threshold::

    {
        "weights": {"0.weight": [...], "0.bias": [...], "2.weight": [...], "2.bias": [...]},
        "threshold": 0.5,
        "media_type": "audio",
        "name": "my detector"
    }

No additional pip packages are required; uses only Python's ``json`` and
``pathlib`` stdlib modules.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vtsearch.processors.importers.base import ProcessorImporter, ProcessorImporterField


class DetectorFileProcessorImporter(ProcessorImporter):
    """Import a processor (detector) from a JSON file.

    The file must contain ``"weights"`` (serialised MLP state dict) and
    ``"threshold"`` (float).  An optional ``"media_type"`` key specifies the
    media type; when absent it defaults to ``"audio"``.
    """

    name = "detector_file"
    display_name = "Detector File (.json)"
    description = "Import a pre-trained detector from a VTSearch detector JSON file."
    icon = "\U0001f4c4"  # page facing up
    fields = [
        ProcessorImporterField(
            key="file",
            label="Detector JSON File",
            field_type="file",
            accept=".json",
            description="A VTSearch detector JSON file with weights and threshold.",
        ),
    ]

    def run(self, field_values: dict[str, Any]) -> dict[str, Any]:
        """Parse the uploaded JSON file and return processor data.

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
        return _parse_detector_json(raw)

    def run_cli(self, field_values: dict[str, Any]) -> dict[str, Any]:
        """Load a detector from a file-path string (CLI usage)."""
        filepath = field_values.get("file", "").strip()
        if not filepath:
            raise ValueError("--file is required.")
        raw = Path(filepath).read_bytes()
        return _parse_detector_json(raw)

    def add_cli_arguments(self, parser: Any) -> None:
        parser.add_argument(
            "--file",
            dest="file",
            help="Path to a VTSearch detector JSON file.",
            required=False,
        )


def _parse_detector_json(raw: bytes) -> dict[str, Any]:
    """Decode *raw* bytes as JSON and extract detector data."""
    try:
        data = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc

    weights = data.get("weights")
    if not weights:
        raise ValueError("Detector file missing 'weights' field.")

    threshold = data.get("threshold", 0.5)
    media_type = data.get("media_type", "audio")
    suggested_name = data.get("name", "")

    result: dict[str, Any] = {
        "media_type": media_type,
        "weights": weights,
        "threshold": threshold,
    }
    if suggested_name:
        result["name"] = suggested_name
    return result


PROCESSOR_IMPORTER = DetectorFileProcessorImporter()
