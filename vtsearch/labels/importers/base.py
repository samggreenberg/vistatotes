"""Base classes for Label Importers.

To add a new label importer, subclass :class:`LabelImporter`, define its class
attributes and :meth:`~LabelImporter.run`, then expose a module-level
``LABEL_IMPORTER`` instance from a package under this directory.  The registry
will discover it automatically.

Each importer also supports CLI usage via :meth:`~LabelImporter.add_cli_arguments`
and :meth:`~LabelImporter.run_cli`.  The base class provides default
implementations that derive CLI arguments from the :attr:`fields` list, so most
importers work on the command line without any extra code.  Importers whose
:meth:`run` expects non-string values (e.g. Werkzeug ``FileStorage`` objects)
should override :meth:`run_cli` to handle the CLI-appropriate types (file paths
as strings).

The label format used throughout is a list of dicts::

    [{"md5": "<clip-md5>", "label": "good"}, ...]

where ``label`` is ``"good"`` or ``"bad"``.

Example – a minimal database label importer skeleton::

    # vtsearch/labels/importers/postgres/__init__.py
    from vtsearch.labels.importers.base import LabelImporter, LabelImporterField

    class PostgresLabelImporter(LabelImporter):
        name         = "postgres"
        display_name = "PostgreSQL Query"
        description  = "Import labels from a PostgreSQL database query."
        icon         = "🐘"
        fields = [
            LabelImporterField("host",     "Hostname", "text"),
            LabelImporterField("database", "Database", "text"),
            LabelImporterField("query",    "SQL Query", "text",
                               description="Must return md5 and label columns."),
        ]

        def run(self, field_values: dict) -> list[dict]:
            import psycopg2
            conn = psycopg2.connect(host=field_values["host"],
                                    database=field_values["database"])
            cur = conn.cursor()
            cur.execute(field_values["query"])
            return [{"md5": row[0], "label": row[1]} for row in cur.fetchall()]

    LABEL_IMPORTER = PostgresLabelImporter()

Then create ``vtsearch/labels/importers/postgres/requirements.txt`` containing
``psycopg2-binary`` and add::

    -r vtsearch/labels/importers/postgres/requirements.txt

to ``requirements-label-importers.txt``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any, Literal

FieldType = Literal["file", "text", "password", "select"]


@dataclass
class LabelImporterField:
    """Describes a single configurable input for a label importer.

    The ``field_type`` value drives how the frontend renders it:

    - ``"file"``     – OS file-picker; value arrives as a Werkzeug
      :class:`~werkzeug.datastructures.FileStorage` object.
    - ``"text"``     – Generic single-line text input.
    - ``"password"`` – Text input whose characters are masked.
    - ``"select"``   – Drop-down; ``options`` must be populated.
    """

    key: str
    label: str
    field_type: FieldType
    description: str = ""
    #: For ``"file"`` fields: comma-separated extensions, e.g. ``".csv"``.
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


class LabelImporter:
    """Abstract base class for label importers.

    Subclass this, set the class-level attributes, implement :meth:`run`,
    and expose a module-level ``LABEL_IMPORTER = YourImporter()`` – the
    registry picks it up automatically.

    The :meth:`run` method must return a list of label dicts in the form::

        [{"md5": "<clip-md5>", "label": "good"}, ...]

    where ``label`` is ``"good"`` or ``"bad"``.  The route handler applies
    these to the global vote state (``good_votes`` / ``bad_votes``) by
    matching clip MD5 hashes.

    CLI support
    -----------
    Every importer is automatically usable from the command line via::

        python app.py --label-importer <name> [importer args]

    The default :meth:`add_cli_arguments` derives ``argparse`` arguments from
    :attr:`fields` and :meth:`run_cli` delegates to :meth:`run`.  Override
    either when the defaults are insufficient (e.g. when :meth:`run` expects a
    Werkzeug ``FileStorage`` rather than a plain file path).
    """

    #: Internal snake_case identifier used in API routes, e.g. ``"csv"``.
    name: str
    #: Human-readable label shown in the UI, e.g. ``"CSV File"``.
    display_name: str
    #: One-sentence description shown as a subtitle in the UI.
    description: str
    #: Emoji or icon string shown next to the display name in the UI.
    icon: str = "🏷️"
    #: Ordered list of fields the user must fill before importing.
    fields: list[LabelImporterField]

    def run(self, field_values: dict[str, Any]) -> list[dict[str, str]]:
        """Perform the import and return a list of label dicts.

        Args:
            field_values: Mapping of :attr:`LabelImporterField.key` → value.
                Fields with ``field_type="file"`` receive a Werkzeug
                :class:`~werkzeug.datastructures.FileStorage` object; all
                other fields receive plain strings.

        Returns:
            A list of dicts, each with ``"md5"`` and ``"label"`` keys.
            ``label`` must be ``"good"`` or ``"bad"``; any other value will
            be skipped by the route handler.

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

        The default implementation converts each :class:`LabelImporterField`
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

    def run_cli(self, field_values: dict[str, Any]) -> list[dict[str, str]]:
        """Import labels from CLI-provided *field_values*.

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
        """Serialise importer metadata for the ``/api/label-importers`` endpoint."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "icon": self.icon,
            "fields": [f.to_dict() for f in self.fields],
        }
