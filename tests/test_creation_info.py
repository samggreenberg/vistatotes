"""Tests for dataset importer CLI args and pickle round-trip."""

import pickle

import numpy as np

from vtsearch.datasets.importers import get_importer
from vtsearch.datasets.importers.base import DatasetImporter, ImporterField
from vtsearch.datasets.loader import export_dataset_to_file, load_dataset_from_pickle


# ---------------------------------------------------------------------------
# build_cli_args on the base class
# ---------------------------------------------------------------------------


class _DummyImporter(DatasetImporter):
    name = "test_dummy"
    display_name = "Test Dummy"
    description = "A dummy importer for testing."
    fields = [
        ImporterField("media_type", "Media Type", "select", options=["sounds", "images"], default="sounds"),
        ImporterField("path", "Folder", "folder"),
    ]

    def run(self, field_values, clips):
        pass


class TestBuildCliArgs:
    def test_basic_cli_args(self):
        imp = _DummyImporter()
        args = imp.build_cli_args({"media_type": "sounds", "path": "/data/audio"})
        assert "--importer test_dummy" in args
        assert "--media-type sounds" in args
        assert "--path /data/audio" in args

    def test_empty_values_skipped(self):
        imp = _DummyImporter()
        args = imp.build_cli_args({"media_type": "sounds", "path": ""})
        assert "--path" not in args

    def test_file_fields_skipped(self):
        """File fields don't translate to CLI flags."""

        class _FileImporter(DatasetImporter):
            name = "file_test"
            display_name = "File Test"
            description = "test"
            fields = [ImporterField("upload", "Upload", "file", accept=".pkl")]

            def run(self, fv, c):
                pass

        imp = _FileImporter()
        args = imp.build_cli_args({"upload": "<FileStorage object>"})
        assert "--upload" not in args


# ---------------------------------------------------------------------------
# Real importers produce correct CLI args
# ---------------------------------------------------------------------------


class TestRealImporterCliArgs:
    def test_folder_importer_cli_args(self):
        imp = get_importer("folder")
        args = imp.build_cli_args({"media_type": "sounds", "path": "/my/folder"})
        assert args == "--importer folder --media-type sounds --path /my/folder"

    def test_http_archive_importer_cli_args(self):
        imp = get_importer("http_archive")
        args = imp.build_cli_args({"url": "https://example.com/data.zip", "media_type": "images"})
        assert "--importer http_archive" in args
        assert "--url https://example.com/data.zip" in args
        assert "--media-type images" in args


# ---------------------------------------------------------------------------
# Pickle round-trip: clips survive export -> import
# ---------------------------------------------------------------------------


class TestPickleRoundTrip:
    def _make_clips(self):
        return {
            1: {
                "id": 1,
                "type": "audio",
                "duration": 1.0,
                "file_size": 100,
                "md5": "abc123",
                "embedding": np.zeros(10),
                "filename": "clip_1.wav",
                "category": "test",
                "clip_bytes": b"\x00" * 100,
                "clip_string": None,
            }
        }

    def test_export_does_not_include_creation_info(self):
        data_bytes = export_dataset_to_file(self._make_clips())
        data = pickle.loads(data_bytes)
        assert "creation_info" not in data

    def test_load_returns_none(self, tmp_path):
        data_bytes = export_dataset_to_file(self._make_clips())
        pkl_path = tmp_path / "test.pkl"
        pkl_path.write_bytes(data_bytes)

        loaded_clips: dict = {}
        result = load_dataset_from_pickle(pkl_path, loaded_clips)
        assert result is None
        assert len(loaded_clips) == 1

    def test_old_format_pickle_loads(self, tmp_path):
        """Old-style pickles (no wrapping 'clips' key) still load."""
        old_data = {
            1: {
                "id": 1,
                "type": "audio",
                "duration": 1.0,
                "file_size": 100,
                "md5": "abc123",
                "embedding": [0.0] * 10,
                "filename": "clip_1.wav",
                "category": "test",
                "wav_bytes": b"\x00" * 100,
            }
        }
        pkl_path = tmp_path / "old.pkl"
        pkl_path.write_bytes(pickle.dumps(old_data))
        loaded_clips: dict = {}
        result = load_dataset_from_pickle(pkl_path, loaded_clips)
        assert result is None
        assert len(loaded_clips) == 1
        # Old wav_bytes key should be migrated to clip_bytes
        assert loaded_clips[1]["clip_bytes"] == b"\x00" * 100

    def test_old_format_with_creation_info_uses_fallback_origin(self, tmp_path):
        """Old pickles with creation_info use it as fallback origin for clips without one."""
        old_data = {
            "clips": {
                1: {
                    "id": 1,
                    "type": "audio",
                    "duration": 1.0,
                    "file_size": 100,
                    "md5": "abc123",
                    "embedding": [0.0] * 10,
                    "filename": "clip_1.wav",
                    "category": "test",
                    "wav_bytes": b"\x00" * 100,
                }
            },
            "creation_info": {
                "importer": "folder",
                "display_name": "Generate from Folder",
                "field_values": {"path": "/data"},
                "cli_args": "--importer folder --path /data",
            },
        }
        pkl_path = tmp_path / "old_with_ci.pkl"
        pkl_path.write_bytes(pickle.dumps(old_data))
        loaded_clips: dict = {}
        load_dataset_from_pickle(pkl_path, loaded_clips)
        assert len(loaded_clips) == 1
        # Fallback origin should be derived from creation_info
        assert loaded_clips[1]["origin"]["importer"] == "folder"
        assert loaded_clips[1]["origin"]["params"]["path"] == "/data"


# ---------------------------------------------------------------------------
# Status endpoint no longer includes creation_info
# ---------------------------------------------------------------------------


class TestStatusEndpoint:
    def test_status_does_not_include_creation_info(self, client):
        resp = client.get("/api/dataset/status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "creation_info" not in data
