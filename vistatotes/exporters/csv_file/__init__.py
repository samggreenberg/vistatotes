"""CSV exporter – saves auto-detect results to a CSV file on disk.

No additional pip packages are required; uses only Python's ``csv`` and
``pathlib`` stdlib modules.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from vistatotes.exporters.base import ExporterField, ResultsExporter


class CsvExporter(ResultsExporter):
    """Save auto-detect results as a CSV file.

    Produces one row per hit across all detectors, with columns for the
    detector name, filename, category, and score.  Opens directly in
    Excel, Google Sheets, or any spreadsheet application.
    """

    name = "csv"
    display_name = "Save to CSV"
    description = "Write the results to a CSV file for spreadsheet analysis."
    icon = "\U0001f4ca"
    fields = [
        ExporterField(
            key="filepath",
            label="File Path",
            field_type="text",
            description=(
                "Absolute or relative path where the CSV results file will be "
                "written.  Parent directories are created automatically."
            ),
            placeholder="/home/user/autodetect_results.csv",
            default="autodetect_results.csv",
        ),
    ]

    def export(self, results: dict[str, Any], field_values: dict[str, Any]) -> dict[str, Any]:
        filepath_str = field_values.get("filepath", "").strip()
        if not filepath_str:
            raise ValueError("A file path is required.")

        filepath = Path(filepath_str)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        total_hits = 0
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["detector", "threshold", "filename", "category", "score"])

            for det_result in results.get("results", {}).values():
                detector_name = det_result.get("detector_name", "unknown")
                threshold = det_result.get("threshold", "")
                for hit in det_result.get("hits", []):
                    writer.writerow(
                        [
                            detector_name,
                            threshold,
                            hit.get("filename", ""),
                            hit.get("category", ""),
                            hit.get("score", ""),
                        ]
                    )
                    total_hits += 1

        return {
            "message": (
                f"Saved {total_hits} hit(s) across "
                f"{results.get('detectors_run', 0)} detector(s) "
                f"to {filepath.resolve()}."
            ),
            "filepath": str(filepath.resolve()),
        }


EXPORTER = CsvExporter()
