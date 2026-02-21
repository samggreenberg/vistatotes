"""CSV label importer – loads labels from a ``.csv`` file on disk.

The CSV file must have at least two columns: ``md5`` and ``label``.  A header
row is required.  Any additional columns are silently ignored.

Example CSV::

    md5,label
    d41d8cd98f00b204e9800998ecf8427e,good
    098f6bcd4621d373cade4e832627b4f6,bad

No additional pip packages are required; uses only Python's ``csv`` and
``io`` stdlib modules.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

from vtsearch.labels.importers.base import LabelImporter, LabelImporterField


class CsvLabelImporter(LabelImporter):
    """Import labels from a CSV file with ``md5`` and ``label`` columns.

    The file must have a header row.  Any row missing the ``md5`` or
    ``label`` column, or with a label that is not ``"good"`` or ``"bad"``,
    is silently skipped by the route handler.
    """

    name = "csv_file"
    display_name = "CSV File"
    description = "Import labels from a CSV file with md5 and label columns."
    icon = "📊"
    fields = [
        LabelImporterField(
            key="file",
            label="Labels CSV File",
            field_type="file",
            accept=".csv",
            description="A CSV file with header row containing 'md5' and 'label' columns.",
        ),
    ]

    def run(self, field_values: dict[str, Any]) -> list[dict[str, str]]:
        """Parse the uploaded CSV file and return label dicts.

        In the GUI path ``field_values["file"]`` is a Werkzeug
        ``FileStorage`` object.  Use :meth:`run_cli` for the CLI path.
        """
        file_storage = field_values.get("file")
        if file_storage is None:
            raise ValueError("No file provided.")
        try:
            raw = file_storage.read()
        except AttributeError:
            raise ValueError("Expected a file upload, not a string. Use run_cli for CLI usage.")
        return _parse_csv_bytes(raw)

    def run_cli(self, field_values: dict[str, Any]) -> list[dict[str, str]]:
        """Load labels from a file-path string (CLI usage)."""
        filepath = field_values.get("file", "").strip()
        if not filepath:
            raise ValueError("--file is required.")
        raw = Path(filepath).read_bytes()
        return _parse_csv_bytes(raw)

    def add_cli_arguments(self, parser: Any) -> None:
        parser.add_argument(
            "--file",
            dest="file",
            help="Path to a CSV file with md5 and label columns.",
            required=False,
        )


def _parse_csv_bytes(raw: bytes) -> list[dict[str, str]]:
    """Decode *raw* bytes as CSV and extract ``md5``/``label`` pairs."""
    try:
        text = raw.decode("utf-8-sig")  # strip BOM if present
    except UnicodeDecodeError:
        text = raw.decode("latin-1")

    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise ValueError("CSV file appears to be empty.")

    # Normalise header names to lower-case and strip whitespace
    normalised = {k.strip().lower(): k for k in reader.fieldnames if k}
    if "md5" not in normalised or "label" not in normalised:
        raise ValueError("CSV must have 'md5' and 'label' column headers.")

    results = []
    for row in reader:
        md5 = row.get(normalised["md5"], "").strip()
        label = row.get(normalised["label"], "").strip().lower()
        if md5 and label:
            results.append({"md5": md5, "label": label})
    return results


LABEL_IMPORTER = CsvLabelImporter()
