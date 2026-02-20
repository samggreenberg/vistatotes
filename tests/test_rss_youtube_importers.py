"""Tests for RSS feed and YouTube playlist importers."""

import argparse
from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# RSS Feed Importer — metadata & CLI
# ---------------------------------------------------------------------------


class TestRssFeedImporterMetadata:
    """RSS feed importer metadata appears in /api/dataset/importers."""

    def test_rss_feed_in_importer_list(self, client):
        resp = client.get("/api/dataset/importers")
        data = resp.get_json()
        names = [imp["name"] for imp in data["importers"]]
        assert "rss_feed" in names

    def test_rss_feed_display_name(self, client):
        resp = client.get("/api/dataset/importers")
        data = resp.get_json()
        imp = next((i for i in data["importers"] if i["name"] == "rss_feed"), None)
        assert imp is not None
        assert imp["display_name"] == "Generate from RSS Podcast Feed"

    def test_rss_feed_icon(self, client):
        resp = client.get("/api/dataset/importers")
        data = resp.get_json()
        imp = next((i for i in data["importers"] if i["name"] == "rss_feed"), None)
        assert imp is not None
        assert imp["icon"] == "\U0001f3b5"

    def test_rss_feed_has_url_field(self, client):
        resp = client.get("/api/dataset/importers")
        data = resp.get_json()
        imp = next((i for i in data["importers"] if i["name"] == "rss_feed"), None)
        keys = [f["key"] for f in imp["fields"]]
        assert "url" in keys

    def test_rss_feed_has_media_type_field(self, client):
        resp = client.get("/api/dataset/importers")
        data = resp.get_json()
        imp = next((i for i in data["importers"] if i["name"] == "rss_feed"), None)
        keys = [f["key"] for f in imp["fields"]]
        assert "media_type" in keys

    def test_rss_feed_has_max_episodes_field(self, client):
        resp = client.get("/api/dataset/importers")
        data = resp.get_json()
        imp = next((i for i in data["importers"] if i["name"] == "rss_feed"), None)
        keys = [f["key"] for f in imp["fields"]]
        assert "max_episodes" in keys


class TestRssFeedImporterCLI:
    """CLI argument parsing for the RSS feed importer."""

    def test_adds_expected_args(self):
        from vtsearch.datasets.importers.rss_feed import RssFeedImporter

        imp = RssFeedImporter()
        parser = argparse.ArgumentParser()
        imp.add_cli_arguments(parser)

        args = parser.parse_args(["--url", "https://example.com/feed.xml"])
        assert args.url == "https://example.com/feed.xml"
        assert args.media_type == "sounds"

    def test_media_type_choices(self):
        from vtsearch.datasets.importers.rss_feed import RssFeedImporter

        imp = RssFeedImporter()
        parser = argparse.ArgumentParser()
        imp.add_cli_arguments(parser)

        args = parser.parse_args(["--url", "https://example.com/feed.xml", "--media-type", "images"])
        assert args.media_type == "images"

    def test_rejects_invalid_media_type(self):
        from vtsearch.datasets.importers.rss_feed import RssFeedImporter

        imp = RssFeedImporter()
        parser = argparse.ArgumentParser()
        imp.add_cli_arguments(parser)

        with pytest.raises(SystemExit):
            parser.parse_args(["--url", "https://example.com/feed.xml", "--media-type", "invalid"])

    def test_validate_missing_url(self):
        from vtsearch.datasets.importers.rss_feed import RssFeedImporter

        imp = RssFeedImporter()
        with pytest.raises(ValueError, match="Missing required argument: --url"):
            imp.validate_cli_field_values({"media_type": "sounds"})

    def test_validate_passes_with_url(self):
        from vtsearch.datasets.importers.rss_feed import RssFeedImporter

        imp = RssFeedImporter()
        # Should not raise — max_episodes is not required
        imp.validate_cli_field_values({"url": "https://example.com/feed.xml", "media_type": "sounds"})

    def test_run_cli_rejects_invalid_url(self):
        from vtsearch.datasets.importers.rss_feed import RssFeedImporter

        imp = RssFeedImporter()
        with pytest.raises(ValueError, match="Invalid URL"):
            imp.run_cli({"url": "not-a-url", "media_type": "sounds"}, {})


class TestRssFeedImporterRun:
    """Unit tests for the RSS feed importer's run() method using mocks."""

    def test_missing_feedparser_raises_runtime_error(self):
        """Attempting import without feedparser installed raises RuntimeError."""
        import sys

        from vtsearch.datasets.importers.rss_feed import RssFeedImporter

        imp = RssFeedImporter()
        with mock.patch.dict(sys.modules, {"feedparser": None}):
            with pytest.raises((RuntimeError, ImportError)):
                imp.run({"url": "https://example.com/feed.xml", "media_type": "sounds"}, {})

    def test_empty_feed_raises_value_error(self):
        """A feed with no enclosures raises ValueError."""
        from vtsearch.datasets.importers.rss_feed import RssFeedImporter

        imp = RssFeedImporter()

        fake_feed = mock.MagicMock()
        fake_feed.bozo = False
        fake_feed.entries = [mock.MagicMock(enclosures=[], title="Ep 1")]

        with mock.patch("vtsearch.datasets.importers.rss_feed.feedparser", create=True) as mock_fp:
            mock_fp.parse.return_value = fake_feed
            # Patch sys.modules so the import inside run() succeeds
            import sys

            with mock.patch.dict(sys.modules, {"feedparser": mock_fp}):
                with pytest.raises(ValueError, match="No media enclosures"):
                    imp.run({"url": "https://example.com/feed.xml", "media_type": "sounds"}, {})

    def test_bozo_feed_with_no_entries_raises(self):
        """A bozo (malformed) feed with no entries raises ValueError."""
        from vtsearch.datasets.importers.rss_feed import RssFeedImporter

        imp = RssFeedImporter()

        fake_feed = mock.MagicMock()
        fake_feed.bozo = True
        fake_feed.entries = []
        fake_feed.bozo_exception = Exception("malformed XML")

        import sys

        with mock.patch("vtsearch.datasets.importers.rss_feed.feedparser", create=True) as mock_fp:
            mock_fp.parse.return_value = fake_feed
            with mock.patch.dict(sys.modules, {"feedparser": mock_fp}):
                with pytest.raises(ValueError, match="Failed to parse feed"):
                    imp.run({"url": "https://example.com/feed.xml", "media_type": "sounds"}, {})


# ---------------------------------------------------------------------------
# YouTube Playlist Importer — metadata & CLI
# ---------------------------------------------------------------------------


class TestYouTubePlaylistImporterMetadata:
    """YouTube playlist importer metadata appears in /api/dataset/importers."""

    def test_youtube_in_importer_list(self, client):
        resp = client.get("/api/dataset/importers")
        data = resp.get_json()
        names = [imp["name"] for imp in data["importers"]]
        assert "youtube_playlist" in names

    def test_youtube_display_name(self, client):
        resp = client.get("/api/dataset/importers")
        data = resp.get_json()
        imp = next((i for i in data["importers"] if i["name"] == "youtube_playlist"), None)
        assert imp is not None
        assert imp["display_name"] == "Generate from YouTube Playlist"

    def test_youtube_icon(self, client):
        resp = client.get("/api/dataset/importers")
        data = resp.get_json()
        imp = next((i for i in data["importers"] if i["name"] == "youtube_playlist"), None)
        assert imp is not None
        assert imp["icon"] == "\U0001f3ac"

    def test_youtube_has_url_field(self, client):
        resp = client.get("/api/dataset/importers")
        data = resp.get_json()
        imp = next((i for i in data["importers"] if i["name"] == "youtube_playlist"), None)
        keys = [f["key"] for f in imp["fields"]]
        assert "url" in keys

    def test_youtube_has_media_type_field(self, client):
        resp = client.get("/api/dataset/importers")
        data = resp.get_json()
        imp = next((i for i in data["importers"] if i["name"] == "youtube_playlist"), None)
        keys = [f["key"] for f in imp["fields"]]
        assert "media_type" in keys

    def test_youtube_media_type_options(self, client):
        resp = client.get("/api/dataset/importers")
        data = resp.get_json()
        imp = next((i for i in data["importers"] if i["name"] == "youtube_playlist"), None)
        mt_field = next(f for f in imp["fields"] if f["key"] == "media_type")
        assert "videos" in mt_field["options"]
        assert "sounds" in mt_field["options"]

    def test_youtube_has_max_videos_field(self, client):
        resp = client.get("/api/dataset/importers")
        data = resp.get_json()
        imp = next((i for i in data["importers"] if i["name"] == "youtube_playlist"), None)
        keys = [f["key"] for f in imp["fields"]]
        assert "max_videos" in keys


class TestYouTubePlaylistImporterCLI:
    """CLI argument parsing for the YouTube playlist importer."""

    def test_adds_expected_args(self):
        from vtsearch.datasets.importers.youtube_playlist import YouTubePlaylistImporter

        imp = YouTubePlaylistImporter()
        parser = argparse.ArgumentParser()
        imp.add_cli_arguments(parser)

        args = parser.parse_args(["--url", "https://www.youtube.com/playlist?list=PLtest"])
        assert args.url == "https://www.youtube.com/playlist?list=PLtest"
        assert args.media_type == "videos"

    def test_media_type_choices(self):
        from vtsearch.datasets.importers.youtube_playlist import YouTubePlaylistImporter

        imp = YouTubePlaylistImporter()
        parser = argparse.ArgumentParser()
        imp.add_cli_arguments(parser)

        args = parser.parse_args(["--url", "https://www.youtube.com/watch?v=test", "--media-type", "sounds"])
        assert args.media_type == "sounds"

    def test_rejects_invalid_media_type(self):
        from vtsearch.datasets.importers.youtube_playlist import YouTubePlaylistImporter

        imp = YouTubePlaylistImporter()
        parser = argparse.ArgumentParser()
        imp.add_cli_arguments(parser)

        with pytest.raises(SystemExit):
            parser.parse_args(["--url", "https://youtube.com", "--media-type", "invalid"])

    def test_validate_missing_url(self):
        from vtsearch.datasets.importers.youtube_playlist import YouTubePlaylistImporter

        imp = YouTubePlaylistImporter()
        with pytest.raises(ValueError, match="Missing required argument: --url"):
            imp.validate_cli_field_values({"media_type": "videos"})

    def test_validate_passes_with_url(self):
        from vtsearch.datasets.importers.youtube_playlist import YouTubePlaylistImporter

        imp = YouTubePlaylistImporter()
        # Should not raise — max_videos is not required
        imp.validate_cli_field_values({"url": "https://www.youtube.com/playlist?list=PLtest", "media_type": "videos"})

    def test_run_cli_rejects_invalid_url(self):
        from vtsearch.datasets.importers.youtube_playlist import YouTubePlaylistImporter

        imp = YouTubePlaylistImporter()
        with pytest.raises(ValueError, match="Invalid URL"):
            imp.run_cli({"url": "not-a-url", "media_type": "videos"}, {})


class TestYouTubePlaylistImporterRun:
    """Unit tests for the YouTube playlist importer's run() method using mocks."""

    def test_missing_yt_dlp_raises_runtime_error(self):
        """Attempting import without yt-dlp installed raises RuntimeError."""
        import sys

        from vtsearch.datasets.importers.youtube_playlist import YouTubePlaylistImporter

        imp = YouTubePlaylistImporter()
        with mock.patch.dict(sys.modules, {"yt_dlp": None}):
            with pytest.raises((RuntimeError, ImportError)):
                imp.run({"url": "https://www.youtube.com/watch?v=test", "media_type": "videos"}, {})
