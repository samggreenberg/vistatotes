"""Base classes for dataset importers.

To add a new importer, subclass :class:`DatasetImporter`, define its class
attributes and :meth:`~DatasetImporter.run`, then expose a module-level
``IMPORTER`` instance from a package under this directory.  The registry will
discover it automatically.

Each importer also supports CLI usage via :meth:`~DatasetImporter.add_cli_arguments`
and :meth:`~DatasetImporter.run_cli`.  The base class provides default
implementations that derive CLI arguments from the :attr:`fields` list, so most
importers work on the command line without any extra code.  Importers whose
:meth:`run` expects non-string values (e.g. Werkzeug ``FileStorage`` objects)
should override :meth:`run_cli` to handle the CLI-appropriate types (file paths
as strings).

Example \u2013 a minimal SFTP importer skeleton::

    # vtsearch/datasets/importers/sftp/__init__.py
    from vtsearch.datasets.importers.base import DatasetImporter, ImporterField

    class SftpImporter(DatasetImporter):
        name         = "sftp"
        display_name = "SFTP Server"
        description  = "Download media files from an SFTP server."
        fields = [
            ImporterField("host",       "Hostname",    "text"),
            ImporterField("user",       "Username",    "text"),
            ImporterField("password",   "Password",    "text"),
            ImporterField("path",       "Remote Path", "text"),
            ImporterField(
                "media_type", "Media Type", "select",
                options=["sounds", "videos", "images", "paragraphs"],
                default="sounds",
            ),
        ]

        def run(self, field_values: dict, clips: dict) -> None:
            import paramiko
            ...  # download files, then call load_dataset_from_folder(...)

    IMPORTER = SftpImporter()

Then add ``-r vtsearch/datasets/importers/sftp/requirements.txt`` to
``requirements-importers.txt`` (creating the file first with ``paramiko``).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any, Literal

FieldType = Literal["file", "folder", "url", "text", "select"]

__all__ = ["DatasetImporter", "FieldType", "ImporterField"]


@dataclass
class ImporterField:
    """Describes a single configurable input for an importer.

    The ``field_type`` value drives how the frontend renders it:

    - ``"file"``   \u2013 OS file-picker; value arrives as a Werkzeug
      :class:`~werkzeug.datastructures.FileStorage` object.
    - ``"folder"`` \u2013 Path text-input or OS folder-picker.
    - ``"url"``    \u2013 Text input pre-validated as a URL.
    - ``"text"``   \u2013 Generic single-line text input.
    - ``"select"`` \u2013 Drop-down; ``options`` must be populated.
    """

    key: str
    label: str
    field_type: FieldType
    description: str = ""
    #: For ``"file"`` fields: comma-separated extensions, e.g. ``".pkl"``.
    accept: str = ""
    #: For ``"select"`` fields: the list of allowed values.
    options: list[str] = field(default_factory=list)
    #: Pre-filled default value shown in the UI.
    default: str = ""
    required: bool = True

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
        }


class DatasetImporter:
    """Abstract base class for dataset importers.

    Subclass this, set the class-level attributes, implement :meth:`run`,
    and expose a module-level ``IMPORTER = YourImporter()`` \u2013 the registry
    picks it up automatically.

    Content vectors
    ---------------
    Some importers provide pre-computed content vectors (embeddings) alongside
    the media files.  To take advantage of this, populate
    :attr:`content_vectors` with a mapping of ``filename`` to
    ``numpy.ndarray`` during :meth:`run`.  When the dataset is later embedded
    (e.g. via :func:`~vtsearch.datasets.loader.load_dataset_from_folder`),
    files whose names appear in this mapping will reuse the supplied vector
    instead of running the embedding model.

    CLI support
    -----------
    Every importer is automatically usable from the command line via
    ``python app.py --autodetect --importer <name> [importer args] --settings <file>``.

    The default :meth:`add_cli_arguments` derives ``argparse`` arguments from
    :attr:`fields` and :meth:`run_cli` delegates to :meth:`run`.  Override
    either method when the defaults are not sufficient (e.g. when :meth:`run`
    expects a Werkzeug ``FileStorage`` rather than a plain file path).
    """

    #: Internal snake_case identifier used in API routes, e.g. ``"sftp"``.
    name: str
    #: Human-readable label shown in the UI, e.g. ``"SFTP Server"``.
    display_name: str
    #: One-sentence description shown as a subtitle in the UI.
    description: str
    #: Emoji or icon string shown next to the display name in the UI.
    icon: str = "\U0001f50c"
    #: Ordered list of fields the user must fill before importing.
    fields: list[ImporterField]

    def __init__(self) -> None:
        #: Mapping of filename to pre-computed embedding vector.  Importers
        #: that supply content vectors alongside media should populate this
        #: dict during :meth:`run` (keyed by the basename of each file).
        #: :func:`~vtsearch.datasets.loader.load_dataset_from_folder` will
        #: skip the embedding model for any file whose name appears here.
        self.content_vectors: dict[str, Any] = {}

    def run(self, field_values: dict[str, Any], clips: dict) -> None:
        """Perform the import, populating *clips* in-place.

        Args:
            field_values: Mapping of :attr:`ImporterField.key` \u2192 value.
                Fields with ``field_type="file"`` receive a Werkzeug
                :class:`~werkzeug.datastructures.FileStorage` object; all
                other fields receive plain strings.
            clips: The global clips dict to populate.  Modify it in-place;
                do not replace the reference.

        Raises:
            NotImplementedError: If the subclass has not implemented this.
            Exception: Any exception propagates to the route handler, which
                stores it in the progress tracker as an error message.
        """
        raise NotImplementedError(f"{type(self).__name__}.run() is not implemented")

    # ------------------------------------------------------------------
    # CLI support
    # ------------------------------------------------------------------

    def add_cli_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Register this importer's fields as ``argparse`` arguments.

        The default implementation converts each :class:`ImporterField` into a
        CLI flag (e.g. a field with ``key="media_type"`` becomes
        ``--media-type``).  ``"select"`` fields gain a ``choices`` constraint.

        Override this method if you need custom argument handling.
        """
        for f in self.fields:
            # file fields are not usable via browser upload on CLI;
            # they become simple path arguments instead
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

    def run_cli(self, field_values: dict[str, Any], clips: dict) -> None:
        """Load a dataset from CLI-provided *field_values* into *clips*.

        The default implementation simply delegates to :meth:`run`, which
        works for importers whose ``run()`` only expects plain string values.
        Importers that expect non-string objects (e.g. ``FileStorage``) must
        override this method to handle file-path strings appropriately.
        """
        self.run(field_values, clips)

    def validate_cli_field_values(self, field_values: dict[str, Any]) -> None:
        """Raise ``ValueError`` if any required field is missing or empty."""
        for f in self.fields:
            if f.required and not field_values.get(f.key):
                cli_flag = f"--{f.key.replace('_', '-')}"
                raise ValueError(f"Missing required argument: {cli_flag}")

    def build_cli_args(self, field_values: dict[str, Any]) -> str:
        """Build a CLI argument string that would recreate this import.

        The returned string contains only the importer-specific portion, e.g.
        ``"--importer folder --media-type sounds --path /data/audio"``.  The
        caller can prepend ``python app.py --autodetect`` and append
        ``--settings <file>`` as needed.

        Fields with ``field_type="file"`` are skipped because they correspond
        to browser uploads that don't translate directly to a CLI flag.

        Args:
            field_values: The same mapping passed to :meth:`run` /
                :meth:`run_cli`.

        Returns:
            A space-separated CLI argument string.
        """
        parts = [f"--importer {self.name}"]
        for f in self.fields:
            if f.field_type == "file":
                continue
            value = field_values.get(f.key, "")
            if value:
                arg_name = f"--{f.key.replace('_', '-')}"
                parts.append(f"{arg_name} {value}")
        return " ".join(parts)

    def build_origin(self, field_values: dict[str, Any]) -> dict[str, Any]:
        """Build an origin dict for elements imported by this importer.

        The returned dict is the serialised form of an
        :class:`~vtsearch.datasets.origin.Origin` object and is stored on
        each clip as ``clip["origin"]``.  It captures enough information to
        identify the data source (importer name + string-serialisable
        field values).

        Args:
            field_values: The field values used for the import.

        Returns:
            A dict with ``"importer"`` (str) and ``"params"`` (dict of str)
            keys.
        """
        params: dict[str, str] = {}
        for f in self.fields:
            if f.field_type == "file":
                continue
            val = field_values.get(f.key, "")
            if val:
                params[f.key] = str(val)
        return {"importer": self.name, "params": params}

    def to_dict(self) -> dict[str, Any]:
        """Serialise importer metadata for the ``/api/dataset/importers`` endpoint."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "icon": self.icon,
            "fields": [f.to_dict() for f in self.fields],
        }
