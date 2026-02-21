"""Display labelset exporter – shows results in the browser (GUI) or prints to console (CLI).

No additional pip packages are required; in GUI mode this exporter is handled
entirely by the frontend JavaScript, and in CLI mode it prints to stdout.
"""

from __future__ import annotations

from typing import Any

from vtsearch.exporters.base import LabelsetExporter


def _format_origin(hit: dict[str, Any]) -> str:
    """Return a human-readable origin string for a hit, or ``""``."""
    origin = hit.get("origin")
    if origin is None:
        return ""
    try:
        from vtsearch.datasets.origin import Origin

        return Origin.from_dict(origin).display()
    except Exception:
        return str(origin)


class DisplayLabelsetExporter(LabelsetExporter):
    """Display auto-detect results in the browser (GUI) or print to console (CLI).

    This is the default exporter: in GUI mode it performs no server-side work
    and simply passes the results back to the frontend, which renders them in
    the Auto-Detect Results modal.  In CLI mode it prints a summary to stdout.
    No configuration fields are needed.
    """

    name = "gui"
    display_name = "Display Results"
    description = "Display the results in the browser (GUI) or print to console (CLI)."
    icon = "🖥️"
    fields = []  # no questions to ask

    def export(self, results: dict[str, Any], field_values: dict[str, Any]) -> dict[str, Any]:
        total_hits = sum(r.get("total_hits", 0) for r in results.get("results", {}).values())
        return {
            "message": (f"Showing {total_hits} hit(s) across {results.get('detectors_run', 0)} detector(s)."),
            "display_results": results,
        }

    def export_cli(self, results: dict[str, Any], field_values: dict[str, Any]) -> dict[str, Any]:
        """Print origins and names of Good results to stdout (no categories, no scores)."""
        lines: list[str] = []
        total_hits = 0
        for det_result in results.get("results", {}).values():
            hits = det_result.get("hits", [])
            total_hits += len(hits)
            for hit in hits:
                origin_str = _format_origin(hit)
                name = hit.get("origin_name") or hit.get("filename", "")
                if origin_str:
                    lines.append(f"  {origin_str}  {name}")
                else:
                    lines.append(f"  {name}")
        if not lines:
            print("No items predicted as Good.")
        else:
            print(f"Predicted Good ({total_hits} items):\n")
            print("\n".join(lines))
        return {
            "message": (
                f"Printed {total_hits} hit(s) across "
                f"{results.get('detectors_run', 0)} detector(s) to stdout."
            ),
        }


EXPORTER = DisplayLabelsetExporter()
