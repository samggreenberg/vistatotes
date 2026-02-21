"""Tests for dataset creation-info tracking.

Datasets should remember how they were created (which importer, what
arguments) so that exported references can describe how to reproduce the
dataset.
"""

import io
import pickle

import numpy as np
import pytest

import app as app_module
from vtsearch.datasets.importers import get_importer
from vtsearch.datasets.importers.base import DatasetImporter, ImporterField
from vtsearch.datasets.loader import export_dataset_to_file, load_dataset_from_pickle
from vtsearch.utils import clips, get_dataset_creation_info, set_dataset_creation_info


# ---------------------------------------------------------------------------
# build_cli_args / build_creation_info on the base class
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


class TestBuildCreationInfo:
    def test_creation_info_structure(self):
        imp = _DummyImporter()
        info = imp.build_creation_info({"media_type": "images", "path": "/pics"})
        assert info["importer"] == "test_dummy"
        assert info["display_name"] == "Test Dummy"
        assert info["field_values"] == {"media_type": "images", "path": "/pics"}
        assert "--importer test_dummy" in info["cli_args"]
        assert "--media-type images" in info["cli_args"]
        assert "--path /pics" in info["cli_args"]

    def test_file_fields_excluded_from_field_values(self):
        class _FileImporter(DatasetImporter):
            name = "pkl"
            display_name = "Pickle"
            description = "test"
            fields = [ImporterField("file", "File", "file", accept=".pkl")]

            def run(self, fv, c):
                pass

        imp = _FileImporter()
        info = imp.build_creation_info({"file": "<blob>"})
        assert "file" not in info["field_values"]


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
# Pickle round-trip: creation_info survives export → import
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
                "wav_bytes": b"\x00" * 100,
                "video_bytes": None,
                "image_bytes": None,
                "text_content": None,
            }
        }

    def test_export_includes_creation_info(self):
        info = {"importer": "folder", "display_name": "Generate from Folder", "field_values": {"path": "/x"}, "cli_args": "--importer folder --path /x"}
        data_bytes = export_dataset_to_file(self._make_clips(), creation_info=info)
        data = pickle.loads(data_bytes)
        assert "creation_info" in data
        assert data["creation_info"]["importer"] == "folder"

    def test_export_without_creation_info(self):
        data_bytes = export_dataset_to_file(self._make_clips())
        data = pickle.loads(data_bytes)
        assert "creation_info" not in data

    def test_load_restores_creation_info(self, tmp_path):
        info = {"importer": "folder", "display_name": "Generate from Folder", "field_values": {"path": "/x"}, "cli_args": "--importer folder --path /x"}
        data_bytes = export_dataset_to_file(self._make_clips(), creation_info=info)
        pkl_path = tmp_path / "test.pkl"
        pkl_path.write_bytes(data_bytes)

        loaded_clips: dict = {}
        restored_info = load_dataset_from_pickle(pkl_path, loaded_clips)
        assert restored_info is not None
        assert restored_info["importer"] == "folder"
        assert restored_info["field_values"]["path"] == "/x"
        assert len(loaded_clips) == 1

    def test_load_without_creation_info_returns_none(self, tmp_path):
        data_bytes = export_dataset_to_file(self._make_clips())
        pkl_path = tmp_path / "test.pkl"
        pkl_path.write_bytes(data_bytes)

        loaded_clips: dict = {}
        restored_info = load_dataset_from_pickle(pkl_path, loaded_clips)
        assert restored_info is None

    def test_old_format_pickle_returns_none(self, tmp_path):
        """Old-style pickles (no wrapping 'clips' key) return None."""
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
        restored_info = load_dataset_from_pickle(pkl_path, loaded_clips)
        assert restored_info is None


# ---------------------------------------------------------------------------
# API: /api/dataset/status includes creation_info
# ---------------------------------------------------------------------------


class TestStatusEndpoint:
    def test_status_includes_creation_info_field(self, client):
        resp = client.get("/api/dataset/status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "creation_info" in data

    def test_status_creation_info_is_null_initially(self, client):
        """Before any import, creation_info should be null."""
        set_dataset_creation_info(None)
        resp = client.get("/api/dataset/status")
        data = resp.get_json()
        assert data["creation_info"] is None

    def test_status_creation_info_after_set(self, client):
        info = {"importer": "folder", "display_name": "Generate from Folder", "field_values": {"path": "/x"}, "cli_args": "--importer folder --path /x"}
        set_dataset_creation_info(info)
        try:
            resp = client.get("/api/dataset/status")
            data = resp.get_json()
            assert data["creation_info"] is not None
            assert data["creation_info"]["importer"] == "folder"
            assert data["creation_info"]["cli_args"] == "--importer folder --path /x"
        finally:
            set_dataset_creation_info(None)


# ---------------------------------------------------------------------------
# Clearing the dataset clears creation_info
# ---------------------------------------------------------------------------


class TestClearDataset:
    def test_clear_dataset_clears_creation_info(self, client):
        set_dataset_creation_info({"importer": "folder", "display_name": "x", "field_values": {}, "cli_args": ""})
        assert get_dataset_creation_info() is not None

        # Clear via API
        resp = client.post("/api/dataset/clear")
        assert resp.status_code == 200
        assert get_dataset_creation_info() is None

        # Re-initialize for other tests
        app_module.init_clips()

    def test_clear_clips_clears_creation_info(self):
        from vtsearch.utils.state import clear_clips

        set_dataset_creation_info({"importer": "test", "display_name": "x", "field_values": {}, "cli_args": ""})
        clear_clips()
        assert get_dataset_creation_info() is None

        # Re-initialize for other tests
        app_module.init_clips()
