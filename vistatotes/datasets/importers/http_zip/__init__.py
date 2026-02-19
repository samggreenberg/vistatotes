"""HTTP-Archive importer – downloads a public archive of media files and loads them.

Supports .zip, .tar, .tar.gz, .tar.bz2, .tar.xz archives.
RAR support requires the optional ``rarfile`` package.

Requires only ``requests``, which is already a core dependency.
"""

from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path

from config import DATA_DIR
from vistatotes.datasets.downloader import download_file_with_progress
from vistatotes.datasets.importers.base import DatasetImporter, ImporterField
from vistatotes.datasets.loader import load_dataset_from_folder
from vistatotes.utils import update_progress


def _extract_archive(archive_path: Path, extract_dir: Path) -> None:
    """Extract *archive_path* into *extract_dir*, supporting zip/tar/rar."""
    name = archive_path.name.lower()

    if name.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            members = zf.namelist()
            total = len(members)
            for i, member in enumerate(members, 1):
                update_progress(
                    "loading",
                    f"Extracting {member.split('/')[-1]}...",
                    i,
                    total,
                )
                zf.extract(member, extract_dir)

    elif tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, "r:*") as tf:
            members = tf.getmembers()
            total = len(members)
            for i, member in enumerate(members, 1):
                update_progress(
                    "loading",
                    f"Extracting {member.name.split('/')[-1]}...",
                    i,
                    total,
                )
                tf.extract(member, extract_dir, filter="data")

    elif name.endswith(".rar"):
        try:
            import rarfile  # optional dependency
        except ImportError as exc:
            raise RuntimeError(
                "RAR extraction requires the 'rarfile' package. "
                "Install it with: pip install rarfile"
            ) from exc
        with rarfile.RarFile(archive_path, "r") as rf:
            members = rf.namelist()
            total = len(members)
            for i, member in enumerate(members, 1):
                update_progress(
                    "loading",
                    f"Extracting {member.split('/')[-1]}...",
                    i,
                    total,
                )
                rf.extract(member, extract_dir)

    else:
        raise ValueError(
            f"Unsupported archive format: {archive_path.name}. "
            "Supported formats: .zip, .tar, .tar.gz, .tar.bz2, .tar.xz, .rar"
        )


class HttpArchiveImporter(DatasetImporter):
    """Download a publicly-accessible archive and load its media files.

    The archive is streamed to a temporary file in ``DATA_DIR``, extracted
    to ``DATA_DIR/http_archive_extract/``, then scanned with the standard
    :func:`~vistatotes.datasets.loader.load_dataset_from_folder` pipeline.
    Both temporary paths are cleaned up after a successful run.

    Supported archive formats: ``.zip``, ``.tar``, ``.tar.gz``,
    ``.tar.bz2``, ``.tar.xz``, ``.rar`` (requires ``rarfile`` package).
    """

    name = "http_archive"
    display_name = "Generate from HTTP Archive"
    description = "Download a .zip, .tar, or .rar archive from a URL and load the media files inside."
    icon = "🌐"
    fields = [
        ImporterField(
            key="url",
            label="Archive URL",
            field_type="url",
            description="URL to a publicly accessible archive (.zip, .tar.gz, .rar, …) of media files.",
        ),
        ImporterField(
            key="media_type",
            label="Media Type",
            field_type="select",
            description="Type of media files contained in the archive.",
            options=["sounds", "videos", "images", "paragraphs"],
            default="sounds",
        ),
    ]

    def run(self, field_values: dict, clips: dict) -> None:
        url = field_values["url"]
        media_type = field_values.get("media_type", "sounds")

        DATA_DIR.mkdir(exist_ok=True)

        # Derive a local filename from the URL so we preserve the extension
        url_path = url.split("?")[0].rstrip("/")
        url_filename = url_path.split("/")[-1] or "archive"
        archive_path = DATA_DIR / f"http_archive_download_{url_filename}"
        extract_dir = DATA_DIR / "http_archive_extract"

        update_progress("downloading", "Downloading archive...", 0, 0)
        download_file_with_progress(url, archive_path)

        update_progress("loading", "Extracting archive...", 0, 0)
        extract_dir.mkdir(exist_ok=True)
        _extract_archive(archive_path, extract_dir)
        archive_path.unlink(missing_ok=True)

        load_dataset_from_folder(extract_dir, media_type, clips)


IMPORTER = HttpArchiveImporter()
