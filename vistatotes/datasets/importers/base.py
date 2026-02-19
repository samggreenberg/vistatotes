"""Base classes for dataset importers.

To add a new importer, subclass :class:`DatasetImporter`, define its class
attributes and :meth:`~DatasetImporter.run`, then expose a module-level
``IMPORTER`` instance from a package under this directory.  The registry will
discover it automatically.

Example – a minimal SFTP importer skeleton::

    # vistatotes/datasets/importers/sftp/__init__.py
    from vistatotes.datasets.importers.base import DatasetImporter, ImporterField

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

Then add ``-r vistatotes/datasets/importers/sftp/requirements.txt`` to
``requirements-importers.txt`` (creating the file first with ``paramiko``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

FieldType = Literal["file", "folder", "url", "text", "select"]


@dataclass
class ImporterField:
    """Describes a single configurable input for an importer.

    The ``field_type`` value drives how the frontend renders it:

    - ``"file"``   – OS file-picker; value arrives as a Werkzeug
      :class:`~werkzeug.datastructures.FileStorage` object.
    - ``"folder"`` – Path text-input or OS folder-picker.
    - ``"url"``    – Text input pre-validated as a URL.
    - ``"text"``   – Generic single-line text input.
    - ``"select"`` – Drop-down; ``options`` must be populated.
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
    and expose a module-level ``IMPORTER = YourImporter()`` – the registry
    picks it up automatically.
    """

    #: Internal snake_case identifier used in API routes, e.g. ``"sftp"``.
    name: str
    #: Human-readable label shown in the UI, e.g. ``"SFTP Server"``.
    display_name: str
    #: One-sentence description shown as a subtitle in the UI.
    description: str
    #: Emoji or icon string shown next to the display name in the UI.
    icon: str = "🔌"
    #: Ordered list of fields the user must fill before importing.
    fields: list[ImporterField]

    def run(self, field_values: dict[str, Any], clips: dict) -> None:
        """Perform the import, populating *clips* in-place.

        Args:
            field_values: Mapping of :attr:`ImporterField.key` → value.
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

    def to_dict(self) -> dict[str, Any]:
        """Serialise importer metadata for the ``/api/dataset/importers`` endpoint."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "icon": self.icon,
            "fields": [f.to_dict() for f in self.fields],
        }
