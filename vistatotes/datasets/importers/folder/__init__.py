"""Local-folder importer – scans a directory of media files and embeds them.

No additional pip packages are required; librosa, opencv, and Pillow are
already in the core requirements.
"""

from __future__ import annotations

from pathlib import Path

from vistatotes.datasets.importers.base import DatasetImporter, ImporterField
from vistatotes.datasets.loader import load_dataset_from_folder


class FolderImporter(DatasetImporter):
    """Embed all media files found in a local directory into a dataset.

    The user supplies an absolute filesystem path and selects the media type
    so that the correct file extensions are matched during the scan.
    """

    name = "folder"
    display_name = "Generate from Folder"
    description = "Import media files from a folder."
    icon = "📂"
    fields = [
        ImporterField(
            key="media_type",
            label="Media Type",
            field_type="select",
            description="Type of media files to scan for in the folder.",
            options=["sounds", "videos", "images", "paragraphs"],
            default="sounds",
        ),
        ImporterField(
            key="path",
            label="Folder",
            field_type="folder",
            description="Absolute path to the directory containing media files.",
        ),
    ]

    def run(self, field_values: dict, clips: dict) -> None:
        folder = Path(field_values["path"])
        media_type = field_values.get("media_type", "sounds")
        load_dataset_from_folder(folder, media_type, clips)


IMPORTER = FolderImporter()
