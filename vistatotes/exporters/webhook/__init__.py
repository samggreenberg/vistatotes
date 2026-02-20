"""Webhook exporter – POSTs auto-detect results to an arbitrary URL.

Requires only ``requests``, which is already a core dependency.
"""

from __future__ import annotations

from typing import Any

import requests

from vistatotes.exporters.base import ExporterField, ResultsExporter


class WebhookExporter(ResultsExporter):
    """POST the results JSON to a user-specified URL.

    Enables integration with automation platforms (Zapier, n8n, Make,
    custom services) without writing a dedicated exporter.  The full
    results dict is sent as the JSON request body.
    """

    name = "webhook"
    display_name = "Webhook (HTTP POST)"
    description = "POST the results as JSON to a URL."
    icon = "\U0001f310"
    fields = [
        ExporterField(
            key="url",
            label="Webhook URL",
            field_type="text",
            description="The URL to POST the results JSON to.",
            placeholder="https://example.com/webhook",
        ),
        ExporterField(
            key="auth_header",
            label="Authorization Header",
            field_type="password",
            description="Optional Bearer token or API key sent as the Authorization header.",
            required=False,
        ),
    ]

    def export(self, results: dict[str, Any], field_values: dict[str, Any]) -> dict[str, Any]:
        url = field_values.get("url", "").strip()
        if not url:
            raise ValueError("A webhook URL is required.")

        headers: dict[str, str] = {"Content-Type": "application/json"}
        auth_header = field_values.get("auth_header", "").strip()
        if auth_header:
            headers["Authorization"] = auth_header

        resp = requests.post(url, json=results, headers=headers, timeout=30)
        resp.raise_for_status()

        total_hits = sum(r.get("total_hits", 0) for r in results.get("results", {}).values())
        return {
            "message": (
                f"Posted {total_hits} hit(s) across "
                f"{results.get('detectors_run', 0)} detector(s) "
                f"to {url} (HTTP {resp.status_code})."
            ),
            "status_code": resp.status_code,
            "url": url,
        }


EXPORTER = WebhookExporter()
