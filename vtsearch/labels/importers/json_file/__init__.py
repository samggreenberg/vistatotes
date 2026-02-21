"""JSON label importer – loads labels from a ``.json`` file on disk.

This importer reads the standard VTSearch label format::

    {"labels": [{"md5": "...", "label": "good"}, ...]}

No additional pip packages are required; uses only Python's ``json`` and
``pathlib`` stdlib modules.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vtsearch.labels.importers.base import LabelImporter, LabelImporterField


class JsonLabelImporter(LabelImporter):
    """Import labels from a JSON file in the standard VTSearch label format.

    The file must be a JSON object with a top-level ``"labels"`` key whose
    value is a list of ``{"md5": "...", "label": "good"|"bad"}`` dicts.
    This is the same format produced by ``GET /api/labels/export``.
    """

    name = "json_file"
    display_name = "JSON File"
    description = "Import labels from a VTSearch-format JSON file (.json)."
    icon = "📄"
    fields = [
        LabelImporterField(
            key="file",
            label="Labels JSON File",
            field_type="file",
            accept=".json",
            description="A JSON file with a 'labels' list of {md5, label} objects.",
        ),
    ]

    def run(self, field_values: dict[str, Any]) -> list[dict[str, str]]:
        """Parse the uploaded JSON file and return label dicts.

        In the GUI path ``field_values["file"]`` is a Werkzeug
        ``FileStorage`` object; in the CLI path it is a plain file-path string.
        Use :meth:`run_cli` for the CLI path.
        """
        file_storage = field_values.get("file")
        if file_storage is None:
            raise ValueError("No file provided.")
        try:
            raw = file_storage.read()
        except AttributeError:
            raise ValueError("Expected a file upload, not a string. Use run_cli for CLI usage.")
        return _parse_json_bytes(raw)

    def run_cli(self, field_values: dict[str, Any]) -> list[dict[str, str]]:
        """Load labels from a file-path string (CLI usage)."""
        filepath = field_values.get("file", "").strip()
        if not filepath:
            raise ValueError("--file is required.")
        raw = Path(filepath).read_bytes()
        return _parse_json_bytes(raw)

    def add_cli_arguments(self, parser: Any) -> None:
        parser.add_argument(
            "--file",
            dest="file",
            help="Path to a VTSearch labels JSON file.",
            required=False,
        )


def _parse_json_bytes(raw: bytes) -> list[dict[str, str]]:
    """Decode *raw* bytes as JSON and extract the labels list."""
    try:
        data = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc
    labels = data.get("labels")
    if not isinstance(labels, list):
        raise ValueError("JSON must contain a top-level 'labels' list.")
    return [entry for entry in labels if isinstance(entry, dict)]


LABEL_IMPORTER = JsonLabelImporter()
