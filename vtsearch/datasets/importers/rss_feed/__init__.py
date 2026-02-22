"""RSS/Podcast feed importer – downloads audio enclosures from an RSS feed.

Requires the ``feedparser`` package.  Add ``feedparser`` to
``requirements-cpu.txt`` / ``requirements-gpu.txt`` if not already present.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable

import requests

from config import DATA_DIR
from vtsearch.datasets.importers.base import DatasetImporter, ImporterField

ProgressCallback = Callable[[str, str, int, int], None]


def _default_progress() -> ProgressCallback:
    from vtsearch.utils import update_progress

    return update_progress


def _download_enclosure(url: str, dest: Path, timeout: int = 120) -> None:
    """Stream an enclosure URL to *dest*."""
    resp = requests.get(url, stream=True, timeout=timeout)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)


class RSSDatasetImporter(DatasetImporter):
    """Download audio episodes from an RSS or podcast feed and embed them.

    The importer parses the feed, downloads each audio enclosure, then
    embeds the files using the selected media-type model.  Non-audio feeds
    work too — just pick the right media type.
    """

    name = "rss_feed"
    display_name = "Generate from RSS Podcast Feed"
    description = "Import media files from an RSS or podcast feed URL."
    icon = "\U0001f3b5"
    fields = [
        ImporterField(
            key="url",
            label="Feed URL",
            field_type="url",
            description="URL of an RSS or Atom feed with media enclosures.",
        ),
        ImporterField(
            key="media_type",
            label="Media Type",
            field_type="select",
            description="Type of media to extract from the feed.",
            options=["sounds", "videos", "images", "paragraphs"],
            default="sounds",
        ),
        ImporterField(
            key="max_episodes",
            label="Max Episodes",
            field_type="text",
            description="Maximum number of episodes to download (0 = all).",
            default="50",
            required=False,
        ),
    ]

    def run(self, field_values: dict[str, Any], clips: dict, thin: bool = False) -> None:
        try:
            import feedparser
        except ImportError as exc:
            raise RuntimeError(
                "RSS feed import requires the 'feedparser' package. Install it with: pip install feedparser"
            ) from exc

        from vtsearch.datasets.loader import load_dataset_from_folder
        from vtsearch.media import get_by_folder_name

        progress = _default_progress()

        url = field_values["url"]
        media_type = field_values.get("media_type", "sounds")
        max_episodes = int(field_values.get("max_episodes", "0") or "0")

        progress("downloading", "Parsing feed...", 0, 0)
        feed = feedparser.parse(url)

        if feed.bozo and not feed.entries:
            raise ValueError(f"Failed to parse feed: {feed.bozo_exception}")

        # Resolve valid file extensions for the chosen media type
        mt = get_by_folder_name(media_type)
        valid_exts = {ext.lstrip("*.").lower() for ext in mt.file_extensions}

        # Collect enclosure URLs
        enclosures: list[tuple[str, str]] = []
        for entry in feed.entries:
            for link in getattr(entry, "enclosures", []):
                href = link.get("href", "")
                if not href:
                    continue
                enclosures.append((entry.get("title", ""), href))
            if max_episodes and len(enclosures) >= max_episodes:
                enclosures = enclosures[:max_episodes]
                break

        if not enclosures:
            raise ValueError("No media enclosures found in the feed.")

        # Download enclosures to a temp folder
        download_dir = DATA_DIR / "rss_feed_download"
        download_dir.mkdir(parents=True, exist_ok=True)

        total = len(enclosures)
        for i, (title, href) in enumerate(enclosures, 1):
            # Derive a filename from the URL
            url_path = href.split("?")[0].rstrip("/")
            url_filename = url_path.split("/")[-1] or f"episode_{i}"

            # If the filename has no recognised extension, try to guess from
            # the first valid extension for this media type.
            suffix = Path(url_filename).suffix.lstrip(".").lower()
            if suffix not in valid_exts:
                default_ext = sorted(valid_exts)[0]
                url_filename = f"{Path(url_filename).stem}.{default_ext}"

            # Avoid filename collisions by prefixing with a short hash
            short_hash = hashlib.md5(href.encode()).hexdigest()[:8]
            dest_name = f"{short_hash}_{url_filename}"
            dest = download_dir / dest_name

            progress(
                "downloading",
                f"Downloading {url_filename} ({i}/{total})...",
                i,
                total,
            )
            try:
                _download_enclosure(href, dest)
            except Exception as exc:
                # Skip episodes that fail to download
                progress(
                    "downloading",
                    f"Skipped {url_filename}: {exc}",
                    i,
                    total,
                )
                continue

        load_dataset_from_folder(download_dir, media_type, clips, on_progress=progress, thin=thin)

    def run_cli(self, field_values: dict[str, Any], clips: dict, thin: bool = False) -> None:
        url = field_values.get("url", "")
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"Invalid URL (must start with http:// or https://): {url}")
        self.run(field_values, clips, thin=thin)


IMPORTER = RSSDatasetImporter()
