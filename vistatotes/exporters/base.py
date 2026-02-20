"""Base classes for Results Exporters.

To add a new exporter, subclass :class:`ResultsExporter`, define its class
attributes and :meth:`~ResultsExporter.export`, then expose a module-level
``EXPORTER`` instance from a package under this directory.  The registry will
discover it automatically.

Each exporter also supports CLI usage via :meth:`~ResultsExporter.add_cli_arguments`
and :meth:`~ResultsExporter.export_cli`.  The base class provides default
implementations that derive CLI arguments from the :attr:`fields` list, so most
exporters work on the command line without any extra code.  Exporters whose
:meth:`export` expects non-string values should override :meth:`export_cli` to
handle the CLI-appropriate types.

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

import argparse
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

    # ------------------------------------------------------------------
    # CLI support
    # ------------------------------------------------------------------

    def add_cli_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Register this exporter's fields as ``argparse`` arguments.

        The default implementation converts each :class:`ExporterField` into a
        CLI flag (e.g. a field with ``key="smtp_host"`` becomes
        ``--smtp-host``).  ``"select"`` fields gain a ``choices`` constraint.

        Override this method if you need custom argument handling.
        """
        for f in self.fields:
            arg_name = f"--{f.key.replace('_', '-')}"
            kwargs: dict[str, Any] = {
                "dest": f.key,
                "help": f.description or f.label,
            }
            if f.default:
                kwargs["default"] = f.default
            if f.field_type == "select" and f.options:
                kwargs["choices"] = f.options
            parser.add_argument(arg_name, **kwargs)

    def export_cli(self, results: dict[str, Any], field_values: dict[str, Any]) -> dict[str, Any]:
        """Export results from CLI-provided *field_values*.

        The default implementation simply delegates to :meth:`export`, which
        works for exporters whose ``export()`` only expects plain string values.
        Exporters that need different behaviour on the command line (e.g. the
        GUI exporter, which has no browser) should override this method.
        """
        return self.export(results, field_values)

    def validate_cli_field_values(self, field_values: dict[str, Any]) -> None:
        """Raise ``ValueError`` if any required field is missing or empty."""
        for f in self.fields:
            if f.required and not field_values.get(f.key):
                cli_flag = f"--{f.key.replace('_', '-')}"
                raise ValueError(f"Missing required argument: {cli_flag}")

    def to_dict(self) -> dict[str, Any]:
        """Serialise exporter metadata for the ``/api/exporters`` endpoint."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "icon": self.icon,
            "fields": [f.to_dict() for f in self.fields],
        }
