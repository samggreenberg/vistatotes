"""YouTube playlist importer – downloads videos via yt-dlp and embeds them.

Requires the ``yt-dlp`` package.  Install with: ``pip install yt-dlp``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from config import DATA_DIR
from vtsearch.datasets.importers.base import DatasetImporter, ImporterField

ProgressCallback = Callable[[str, str, int, int], None]


def _default_progress() -> ProgressCallback:
    from vtsearch.utils import update_progress

    return update_progress


class YouTubePlaylistDatasetImporter(DatasetImporter):
    """Download videos from a YouTube playlist or channel and embed them.

    Uses ``yt-dlp`` to download videos in mp4 format, then embeds them using
    the selected media-type model.  Works with playlists, channels, and
    individual video URLs.
    """

    name = "youtube_playlist"
    display_name = "Generate from YouTube Playlist"
    description = "Download videos from a YouTube playlist or channel URL via yt-dlp."
    icon = "\U0001f3ac"
    fields = [
        ImporterField(
            key="url",
            label="Playlist / Channel URL",
            field_type="url",
            description="URL of a YouTube playlist, channel, or single video.",
        ),
        ImporterField(
            key="media_type",
            label="Media Type",
            field_type="select",
            description="How to treat the downloaded files.",
            options=["videos", "sounds"],
            default="videos",
        ),
        ImporterField(
            key="max_videos",
            label="Max Videos",
            field_type="text",
            description="Maximum number of videos to download (0 = all).",
            default="20",
            required=False,
        ),
    ]

    def run(self, field_values: dict[str, Any], clips: dict, thin: bool = False) -> None:
        try:
            import yt_dlp  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "YouTube import requires the 'yt-dlp' package. Install it with: pip install yt-dlp"
            ) from exc

        from vtsearch.datasets.loader import load_dataset_from_folder

        progress = _default_progress()

        url = field_values["url"]
        media_type = field_values.get("media_type", "videos")
        max_videos = int(field_values.get("max_videos", "0") or "0")

        download_dir = DATA_DIR / "youtube_playlist_download"
        download_dir.mkdir(parents=True, exist_ok=True)

        progress("downloading", "Fetching playlist info...", 0, 0)

        # Build yt-dlp options
        ydl_opts: dict[str, Any] = {
            "format": "mp4[height<=720]/best[height<=720]/mp4/best",
            "outtmpl": str(download_dir / "%(id)s.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
            "ignoreerrors": True,
        }

        if max_videos:
            ydl_opts["playlistend"] = max_videos

        if media_type == "sounds":
            # Extract audio only
            ydl_opts["format"] = "bestaudio/best"
            ydl_opts["postprocessors"] = [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                    "preferredquality": "192",
                }
            ]
            ydl_opts["outtmpl"] = str(download_dir / "%(id)s.%(ext)s")

        # Progress hook
        downloaded_count = 0

        def _progress_hook(d: dict) -> None:
            nonlocal downloaded_count
            if d["status"] == "finished":
                downloaded_count += 1
                filename = Path(d.get("filename", "")).name
                progress(
                    "downloading",
                    f"Downloaded {filename} ({downloaded_count})...",
                    downloaded_count,
                    max_videos or 0,
                )

        ydl_opts["progress_hooks"] = [_progress_hook]

        import yt_dlp

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # Check that something was downloaded
        any_files = list(download_dir.iterdir())
        if not any_files:
            raise ValueError("No videos were downloaded. Check the URL and try again.")

        load_dataset_from_folder(download_dir, media_type, clips, on_progress=progress, thin=thin)

    def run_cli(self, field_values: dict[str, Any], clips: dict, thin: bool = False) -> None:
        url = field_values.get("url", "")
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"Invalid URL (must start with http:// or https://): {url}")
        self.run(field_values, clips, thin=thin)


IMPORTER = YouTubePlaylistDatasetImporter()
