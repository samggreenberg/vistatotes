"""Base classes for Processor Importers.

To add a new processor importer, subclass :class:`ProcessorImporter`, define its
class attributes and :meth:`~ProcessorImporter.run`, then expose a module-level
``PROCESSOR_IMPORTER`` instance from a package under this directory.  The registry
will discover it automatically.

Each importer also supports CLI usage via :meth:`~ProcessorImporter.add_cli_arguments`
and :meth:`~ProcessorImporter.run_cli`.  The base class provides default
implementations that derive CLI arguments from the :attr:`fields` list, so most
importers work on the command line without any extra code.  Importers whose
:meth:`run` expects non-string values (e.g. Werkzeug ``FileStorage`` objects)
should override :meth:`run_cli` to handle the CLI-appropriate types (file paths
as strings).

The processor data returned by :meth:`run` is a dict with at minimum::

    {
        "media_type": "audio",
        "weights": {"0.weight": [...], "0.bias": [...], ...},
        "threshold": 0.5,
    }

The route handler adds the user-supplied ``name`` and saves the result as a
favorite detector via :func:`~vtsearch.utils.add_favorite_detector`.

Example -- a minimal S3 processor importer skeleton::

    # vtsearch/processors/importers/s3/__init__.py
    from vtsearch.processors.importers.base import ProcessorImporter, ProcessorImporterField

    class S3ProcessorImporter(ProcessorImporter):
        name         = "s3"
        display_name = "S3 Detector File"
        description  = "Download a detector JSON file from an S3 bucket."
        icon         = "☁️"
        fields = [
            ProcessorImporterField("bucket",  "S3 Bucket", "text"),
            ProcessorImporterField("key",     "Object Key", "text"),
        ]

        def run(self, field_values: dict) -> dict:
            import boto3
            s3 = boto3.client("s3")
            ...  # download & parse JSON
            return {"media_type": "audio", "weights": weights, "threshold": 0.5}

    PROCESSOR_IMPORTER = S3ProcessorImporter()

Then create ``vtsearch/processors/importers/s3/requirements.txt`` containing
``boto3`` and add::

    -r vtsearch/processors/importers/s3/requirements.txt

to ``requirements-processor-importers.txt``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any, Literal

FieldType = Literal["file", "text", "password", "select"]


@dataclass
class ProcessorImporterField:
    """Describes a single configurable input for a processor importer.

    The ``field_type`` value drives how the frontend renders it:

    - ``"file"``     -- OS file-picker; value arrives as a Werkzeug
      :class:`~werkzeug.datastructures.FileStorage` object.
    - ``"text"``     -- Generic single-line text input.
    - ``"password"`` -- Text input whose characters are masked.
    - ``"select"``   -- Drop-down; ``options`` must be populated.
    """

    key: str
    label: str
    field_type: FieldType
    description: str = ""
    #: For ``"file"`` fields: comma-separated extensions, e.g. ``".json"``.
    accept: str = ""
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
            "accept": self.accept,
            "options": self.options,
            "default": self.default,
            "required": self.required,
            "placeholder": self.placeholder,
        }


class ProcessorImporter:
    """Abstract base class for processor importers.

    Subclass this, set the class-level attributes, implement :meth:`run`,
    and expose a module-level ``PROCESSOR_IMPORTER = YourImporter()`` -- the
    registry picks it up automatically.

    The :meth:`run` method must return a dict with at minimum ``media_type``,
    ``weights``, and ``threshold`` keys.  The route handler will combine this
    with the user-supplied name and save it as a favorite detector.

    CLI support
    -----------
    Processor importers are used indirectly from the CLI via the autodetect
    workflow: add a processor recipe to the settings JSON file and run::

        python app.py --autodetect --dataset <file.pkl> --settings <settings.json>

    The default :meth:`add_cli_arguments` derives ``argparse`` arguments from
    :attr:`fields` and :meth:`run_cli` delegates to :meth:`run`.  Override
    either when the defaults are insufficient (e.g. when :meth:`run` expects a
    Werkzeug ``FileStorage`` rather than a plain file path).
    """

    #: Internal snake_case identifier used in API routes, e.g. ``"detector_file"``.
    name: str
    #: Human-readable label shown in the UI, e.g. ``"Detector File (.json)"``.
    display_name: str
    #: One-sentence description shown as a subtitle in the UI.
    description: str
    #: Emoji or icon string shown next to the display name in the UI.
    icon: str = "\U0001f9e9"  # puzzle piece
    #: Ordered list of fields the user must fill before importing.
    fields: list[ProcessorImporterField]

    def run(self, field_values: dict[str, Any]) -> dict[str, Any]:
        """Perform the import and return processor data.

        Args:
            field_values: Mapping of :attr:`ProcessorImporterField.key` to value.
                Fields with ``field_type="file"`` receive a Werkzeug
                :class:`~werkzeug.datastructures.FileStorage` object; all
                other fields receive plain strings.

        Returns:
            A dict with at minimum ``"media_type"`` (str), ``"weights"`` (dict),
            and ``"threshold"`` (float).  May also include ``"name"`` (suggested
            default name), ``"loaded"`` (int), and ``"skipped"`` (int) for
            status reporting.

        Raises:
            NotImplementedError: If the subclass has not implemented this.
            Exception: Any exception propagates to the route handler, which
                returns it as a 500 JSON error.
        """
        raise NotImplementedError(f"{type(self).__name__}.run() is not implemented")

    # ------------------------------------------------------------------
    # CLI support
    # ------------------------------------------------------------------

    def add_cli_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Register this importer's fields as ``argparse`` arguments.

        The default implementation converts each :class:`ProcessorImporterField`
        into a CLI flag (e.g. a field with ``key="filepath"`` becomes
        ``--filepath``).  ``"select"`` fields gain a ``choices`` constraint.

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

    def run_cli(self, field_values: dict[str, Any]) -> dict[str, Any]:
        """Import a processor from CLI-provided *field_values*.

        The default implementation simply delegates to :meth:`run`, which
        works for importers whose ``run()`` only expects plain string values.
        Importers that expect non-string objects (e.g. ``FileStorage``) must
        override this method to handle file-path strings appropriately.
        """
        return self.run(field_values)

    def validate_cli_field_values(self, field_values: dict[str, Any]) -> None:
        """Raise ``ValueError`` if any required field is missing or empty."""
        for f in self.fields:
            if f.required and not field_values.get(f.key):
                cli_flag = f"--{f.key.replace('_', '-')}"
                raise ValueError(f"Missing required argument: {cli_flag}")

    def to_dict(self) -> dict[str, Any]:
        """Serialise importer metadata for the ``/api/processor-importers`` endpoint."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "icon": self.icon,
            "fields": [f.to_dict() for f in self.fields],
        }
