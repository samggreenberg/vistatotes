"""GUI exporter – returns results to the browser for display in a modal.

No additional pip packages are required; this exporter is handled entirely
by the frontend JavaScript.
"""

from __future__ import annotations

from typing import Any

from vistatotes.exporters.base import ResultsExporter


class GuiExporter(ResultsExporter):
    """Display auto-detect results in the browser's built-in results modal.

    This is the default exporter: it performs no server-side work and simply
    passes the results back to the frontend, which renders them in the
    Auto-Detect Results modal.  No configuration fields are needed.
    """

    name = "gui"
    display_name = "Show in Browser"
    description = "Display the results in a popup window in the browser."
    icon = "🖥️"
    fields = []  # no questions to ask

    def export(self, results: dict[str, Any], field_values: dict[str, Any]) -> dict[str, Any]:
        total_hits = sum(r.get("total_hits", 0) for r in results.get("results", {}).values())
        return {
            "message": (f"Showing {total_hits} hit(s) across {results.get('detectors_run', 0)} detector(s)."),
            "display_results": results,
        }

    def export_cli(self, results: dict[str, Any], field_values: dict[str, Any]) -> dict[str, Any]:
        """Print results to stdout (there is no browser in CLI mode)."""
        lines: list[str] = []
        for det_result in results.get("results", {}).values():
            hits = det_result.get("hits", [])
            if not hits:
                lines.append("No items predicted as Good.")
                continue
            lines.append(f"Predicted Good ({len(hits)} items):\n")
            for hit in hits:
                lines.append(
                    f"  {hit['filename']}  (score: {hit['score']}, category: {hit.get('category', 'unknown')})"
                )
        output = "\n".join(lines) if lines else "No results."
        print(output)
        total_hits = sum(r.get("total_hits", 0) for r in results.get("results", {}).values())
        return {
            "message": (f"Printed {total_hits} hit(s) across {results.get('detectors_run', 0)} detector(s) to stdout."),
        }


EXPORTER = GuiExporter()
