"""Base classes for Results Exporters.

To add a new exporter, subclass :class:`ResultsExporter`, define its class
attributes and :meth:`~ResultsExporter.export`, then expose a module-level
``EXPORTER`` instance from a package under this directory.  The registry will
discover it automatically.

Example – a minimal SFTP exporter skeleton::

    # vistatotes/exporters/sftp/__init__.py
    from vistatotes.exporters.base import ResultsExporter, ExporterField

    class SftpExporter(ResultsExporter):
        name         = "sftp"
        display_name = "SFTP Upload"
        description  = "Upload results JSON to a remote SFTP server."
        icon         = "📡"
        fields = [
            ExporterField("host",     "Hostname",    "text"),
            ExporterField("user",     "Username",    "text"),
            ExporterField("password", "Password",    "password"),
            ExporterField("path",     "Remote Path", "text",
                          default="/results/autodetect.json"),
        ]

        def export(self, results: dict, field_values: dict) -> dict:
            import paramiko
            ...  # connect, write JSON, disconnect
            return {"message": f"Uploaded to {field_values['host']}:{field_values['path']}"}

    EXPORTER = SftpExporter()

Then create ``vistatotes/exporters/sftp/requirements.txt`` containing
``paramiko>=2.0.0`` and add::

    -r vistatotes/exporters/sftp/requirements.txt

to ``requirements-exporters.txt``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

FieldType = Literal["text", "password", "email", "file", "folder", "select"]


@dataclass
class ExporterField:
    """Describes a single configurable input for an exporter.

    The ``field_type`` value drives how the frontend renders it:

    - ``"text"``     – Generic single-line text input.
    - ``"password"`` – Text input whose characters are masked.
    - ``"email"``    – Text input pre-validated as an e-mail address.
    - ``"file"``     – OS file-picker; value arrives as a path string from the
      browser (save-as path).
    - ``"folder"``   – Path text-input for a local directory.
    - ``"select"``   – Drop-down; ``options`` must be populated.
    """

    key: str
    label: str
    field_type: FieldType
    description: str = ""
    #: For ``"select"`` fields: the list of allowed values.
    options: list[str] = field(default_factory=list)
    #: Pre-filled default value shown in the UI.
    default: str = ""
    required: bool = True
    #: Hint shown as placeholder text inside the input widget.
    placeholder: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "field_type": self.field_type,
            "description": self.description,
            "options": self.options,
            "default": self.default,
            "required": self.required,
            "placeholder": self.placeholder,
        }


class ResultsExporter:
    """Abstract base class for results exporters.

    Subclass this, set the class-level attributes, implement :meth:`export`,
    and expose a module-level ``EXPORTER = YourExporter()`` – the registry
    picks it up automatically.

    The :meth:`export` method receives the full results dict returned by
    ``/api/auto-detect`` and a flat mapping of field values supplied by the
    user via the UI.  It should return a dict with at minimum a ``"message"``
    key describing what happened (shown to the user as confirmation).
    """

    #: Internal snake_case identifier used in API routes, e.g. ``"sftp"``.
    name: str
    #: Human-readable label shown in the UI, e.g. ``"SFTP Upload"``.
    display_name: str
    #: One-sentence description shown as a subtitle in the UI.
    description: str
    #: Emoji or icon string shown next to the display name in the UI.
    icon: str = "📤"
    #: Ordered list of fields the user must fill before exporting.
    #: Leave empty if the exporter needs no configuration.
    fields: list[ExporterField]

    def export(self, results: dict[str, Any], field_values: dict[str, Any]) -> dict[str, Any]:
        """Perform the export and return a status dict.

        Args:
            results: The full auto-detect results dict from ``/api/auto-detect``.
                     Shape::

                         {
                           "media_type": "audio",
                           "detectors_run": 2,
                           "results": {
                             "detector_name": {
                               "detector_name": "...",
                               "threshold": 0.5,
                               "total_hits": 15,
                               "hits": [{...}, ...]
                             }
                           }
                         }

            field_values: Mapping of :attr:`ExporterField.key` → value string
                supplied by the user.

        Returns:
            A dict that **must** contain a ``"message"`` key with a short
            human-readable confirmation string.  It may also carry arbitrary
            extra keys (e.g. ``"filepath"`` for file-based exporters).

        Raises:
            NotImplementedError: If the subclass has not implemented this.
            Exception: Any exception propagates to the route handler, which
                returns it as a 500 JSON error.
        """
        raise NotImplementedError(f"{type(self).__name__}.export() is not implemented")

    def to_dict(self) -> dict[str, Any]:
        """Serialise exporter metadata for the ``/api/exporters`` endpoint."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "icon": self.icon,
            "fields": [f.to_dict() for f in self.fields],
        }
