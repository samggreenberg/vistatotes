"""File exporter – saves auto-detect results to a JSON file on disk.

No additional pip packages are required; uses only Python's ``json`` and
``pathlib`` stdlib modules.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vistatotes.exporters.base import ExporterField, ResultsExporter


class FileExporter(ResultsExporter):
    """Save auto-detect results as a JSON file at a path chosen by the user.

    The user supplies the destination path (absolute or relative to the
    current working directory).  Parent directories are created automatically.
    """

    name = "file"
    display_name = "Save to File"
    description = "Write the results to a JSON file on the local filesystem."
    icon = "💾"
    fields = [
        ExporterField(
            key="filepath",
            label="File Path",
            field_type="text",
            description=(
                "Absolute or relative path where the JSON results file will be "
                "written.  Parent directories are created automatically."
            ),
            placeholder="/home/user/autodetect_results.json",
            default="autodetect_results.json",
        ),
    ]

    def export(self, results: dict[str, Any], field_values: dict[str, Any]) -> dict[str, Any]:
        filepath_str = field_values.get("filepath", "").strip()
        if not filepath_str:
            raise ValueError("A file path is required.")

        filepath = Path(filepath_str)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(json.dumps(results, indent=2), encoding="utf-8")

        total_hits = sum(
            r.get("total_hits", 0) for r in results.get("results", {}).values()
        )
        return {
            "message": (
                f"Saved {total_hits} hit(s) across "
                f"{results.get('detectors_run', 0)} detector(s) "
                f"to {filepath.resolve()}."
            ),
            "filepath": str(filepath.resolve()),
        }


EXPORTER = FileExporter()
