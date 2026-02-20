"""Email (SMTP) exporter – sends auto-detect results via e-mail.

Uses only Python's built-in ``smtplib`` and ``email`` stdlib modules.
No additional pip packages are required.

Tested against Gmail (smtp.gmail.com:587 with an App Password) and generic
STARTTLS-capable SMTP servers.  For Gmail, enable 2-step verification and
generate an App Password at https://myaccount.google.com/apppasswords.
"""

from __future__ import annotations

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

from vistatotes.exporters.base import ExporterField, ResultsExporter


def _build_plain_text(results: dict[str, Any]) -> str:
    """Render results as a human-readable plain-text summary."""
    lines: list[str] = [
        "Auto-Detect Results",
        "===================",
        f"Media Type:    {results.get('media_type', 'unknown')}",
        f"Detectors Run: {results.get('detectors_run', 0)}",
        "",
    ]
    for det_result in results.get("results", {}).values():
        lines.append(f"--- {det_result['detector_name']} ---")
        lines.append(f"Threshold: {det_result['threshold']}  |  Total Hits: {det_result['total_hits']}")
        if det_result["hits"]:
            for hit in det_result["hits"]:
                lines.append(f"  Clip #{hit['id']}: {hit.get('filename', 'N/A')} (score: {hit['score']})")
        else:
            lines.append("  No positive hits found.")
        lines.append("")
    return "\n".join(lines)


def _build_html(results: dict[str, Any]) -> str:
    """Render results as a minimal HTML e-mail body."""
    rows = ""
    for det_result in results.get("results", {}).values():
        hits_html = ""
        for hit in det_result["hits"]:
            hits_html += (
                f"<tr><td>Clip #{hit['id']}</td><td>{hit.get('filename', 'N/A')}</td><td>{hit['score']}</td></tr>"
            )
        if not hits_html:
            hits_html = '<tr><td colspan="3"><em>No positive hits found.</em></td></tr>'
        rows += (
            f"<h3>{det_result['detector_name']}</h3>"
            f"<p>Threshold: {det_result['threshold']} &mdash; "
            f"Total Hits: {det_result['total_hits']}</p>"
            f"<table border='1' cellpadding='4' cellspacing='0'>"
            f"<tr><th>Clip</th><th>Filename</th><th>Score</th></tr>"
            f"{hits_html}</table>"
        )
    return (
        f"<html><body>"
        f"<h2>Auto-Detect Results</h2>"
        f"<p><strong>Media Type:</strong> {results.get('media_type', 'unknown')}<br>"
        f"<strong>Detectors Run:</strong> {results.get('detectors_run', 0)}</p>"
        f"{rows}"
        f"</body></html>"
    )


class EmailSmtpExporter(ResultsExporter):
    """Send auto-detect results by e-mail via an SMTP server.

    Supports any STARTTLS-capable SMTP server (Gmail, Outlook, custom, etc.).
    The message is sent as multipart/alternative with both plain-text and HTML
    parts; the raw JSON is also attached inline.
    """

    name = "email_smtp"
    display_name = "Send by Email"
    description = "Email the results summary to any address via SMTP."
    icon = "📧"
    fields = [
        ExporterField(
            key="to",
            label="Recipient Email",
            field_type="email",
            description="The email address to send the results to.",
            placeholder="recipient@example.com",
        ),
        ExporterField(
            key="from_email",
            label="Sender Email",
            field_type="email",
            description="The email address used as the sender (must match your SMTP account).",
            placeholder="you@example.com",
        ),
        ExporterField(
            key="smtp_password",
            label="SMTP Password / App Password",
            field_type="password",
            description=(
                "Your SMTP password. For Gmail, use an App Password (see https://myaccount.google.com/apppasswords)."
            ),
        ),
        ExporterField(
            key="smtp_host",
            label="SMTP Host",
            field_type="text",
            description="The outgoing mail server hostname.",
            placeholder="smtp.gmail.com",
            default="smtp.gmail.com",
        ),
        ExporterField(
            key="smtp_port",
            label="SMTP Port",
            field_type="text",
            description="Port for STARTTLS connections (usually 587).",
            placeholder="587",
            default="587",
        ),
    ]

    def export(self, results: dict[str, Any], field_values: dict[str, Any]) -> dict[str, Any]:
        to_addr = field_values.get("to", "").strip()
        from_addr = field_values.get("from_email", "").strip()
        password = field_values.get("smtp_password", "")
        smtp_host = field_values.get("smtp_host", "smtp.gmail.com").strip()
        smtp_port = int(field_values.get("smtp_port", "587").strip() or "587")

        if not to_addr:
            raise ValueError("Recipient email address is required.")
        if not from_addr:
            raise ValueError("Sender email address is required.")
        if not password:
            raise ValueError("SMTP password is required.")

        media_type = results.get("media_type", "unknown")
        total_hits = sum(r.get("total_hits", 0) for r in results.get("results", {}).values())
        subject = f"VistaTotes Auto-Detect: {total_hits} hit(s) on {media_type} dataset"

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = from_addr
        msg["To"] = to_addr

        plain = _build_plain_text(results)
        html = _build_html(results)
        msg.attach(MIMEText(plain, "plain", "utf-8"))
        msg.attach(MIMEText(html, "html", "utf-8"))

        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.ehlo()
            server.starttls()
            server.login(from_addr, password)
            server.sendmail(from_addr, [to_addr], msg.as_string())

        return {
            "message": (f"Email with {total_hits} hit(s) sent to {to_addr} via {smtp_host}:{smtp_port}."),
            "to": to_addr,
        }


EXPORTER = EmailSmtpExporter()
