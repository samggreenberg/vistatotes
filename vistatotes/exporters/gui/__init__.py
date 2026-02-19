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
        total_hits = sum(
            r.get("total_hits", 0) for r in results.get("results", {}).values()
        )
        return {
            "message": (
                f"Showing {total_hits} hit(s) across "
                f"{results.get('detectors_run', 0)} detector(s)."
            ),
            "display_results": results,
        }


EXPORTER = GuiExporter()
