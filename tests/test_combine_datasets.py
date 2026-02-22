"""Tests for the combine-datasets importer.

Covers:
- Importer metadata (name, icon, description, fields)
- Importer is excluded from the generic /api/dataset/importers list
- Combining two pickle datasets with the same media type
- Duplicate detection by MD5 hash
- Media type mismatch rejection
- Fewer than two datasets rejection
- Missing file error handling
- Available-files endpoint
- Combine API endpoint
- CLI support (run_cli)
- build_origin method
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest


def _make_audio_clip(clip_id: int, md5: str = "", filename: str = "") -> dict:
    """Return a minimal clip dict for testing."""
    if not filename:
        filename = f"clip_{clip_id}.wav"
    if not md5:
        md5 = f"md5_{clip_id:04d}"
    return {
        "id": clip_id,
        "type": "audio",
        "duration": 1.0,
        "file_size": 1000,
        "md5": md5,
        "embedding": np.array([float(clip_id), float(clip_id) + 0.5]),
        "clip_bytes": b"\x00" * 100,
        "clip_string": None,
        "media_path": None,
        "filename": filename,
        "category": "test",
        "origin": None,
        "origin_name": filename,
    }


def _make_image_clip(clip_id: int, md5: str = "") -> dict:
    """Return a minimal image clip dict for testing."""
    if not md5:
        md5 = f"md5_{clip_id:04d}"
    return {
        "id": clip_id,
        "type": "image",
        "duration": 0,
        "file_size": 2000,
        "md5": md5,
        "embedding": np.array([float(clip_id)]),
        "clip_bytes": b"\x89PNG" + b"\x00" * 100,
        "clip_string": None,
        "media_path": None,
        "filename": f"img_{clip_id}.png",
        "category": "test",
        "origin": None,
        "origin_name": f"img_{clip_id}.png",
        "width": 32,
        "height": 32,
    }


def _write_pickle_dataset(path, clips_dict):
    """Write a pickle dataset file in the standard format."""
    data = {
        "clips": {
            cid: {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in clip.items()}
            for cid, clip in clips_dict.items()
        }
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)


# ---------------------------------------------------------------------------
# Importer metadata
# ---------------------------------------------------------------------------


class TestCombineDatasetsMetadata:
    def _get_importer(self):
        from vtsearch.datasets.importers.combine_datasets import IMPORTER

        return IMPORTER

    def test_name(self):
        assert self._get_importer().name == "combine_datasets"

    def test_display_name(self):
        assert "Combine" in self._get_importer().display_name

    def test_icon(self):
        assert self._get_importer().icon == "\U0001f500"

    def test_description_mentions_merge_or_combine(self):
        desc = self._get_importer().description.lower()
        assert "merge" in desc or "combine" in desc

    def test_description_mentions_duplicates(self):
        desc = self._get_importer().description.lower()
        assert "duplicate" in desc

    def test_to_dict_includes_icon(self):
        d = self._get_importer().to_dict()
        assert d["icon"] == "\U0001f500"
        assert d["name"] == "combine_datasets"

    def test_fields_include_datasets(self):
        fields = {f.key: f for f in self._get_importer().fields}
        assert "datasets" in fields


# ---------------------------------------------------------------------------
# Excluded from generic importer list (has dedicated UI)
# ---------------------------------------------------------------------------


class TestCombineDatasetsBuiltinExclusion:
    def test_combine_datasets_in_builtin_names(self):
        from vtsearch.routes.datasets import _BUILTIN_IMPORTER_NAMES

        assert "combine_datasets" in _BUILTIN_IMPORTER_NAMES

    def test_combine_datasets_not_in_extended_list(self, client):
        resp = client.get("/api/dataset/importers")
        data = resp.get_json()
        names = [imp["name"] for imp in data["importers"]]
        assert "combine_datasets" not in names


# ---------------------------------------------------------------------------
# Core combining logic
# ---------------------------------------------------------------------------


class TestCombineDatasetsRun:
    def test_combine_two_datasets(self, tmp_path):
        """Two datasets with the same media type are merged."""
        from vtsearch.datasets.importers.combine_datasets import IMPORTER

        ds1 = {1: _make_audio_clip(1), 2: _make_audio_clip(2)}
        ds2 = {1: _make_audio_clip(3), 2: _make_audio_clip(4)}
        p1, p2 = tmp_path / "ds1.pkl", tmp_path / "ds2.pkl"
        _write_pickle_dataset(p1, ds1)
        _write_pickle_dataset(p2, ds2)

        clips: dict = {}
        IMPORTER.run({"datasets": [str(p1), str(p2)]}, clips)

        assert len(clips) == 4
        # IDs should be re-assigned sequentially
        assert set(clips.keys()) == {1, 2, 3, 4}

    def test_deduplication_by_md5(self, tmp_path):
        """Clips with the same MD5 across datasets are included only once."""
        from vtsearch.datasets.importers.combine_datasets import IMPORTER

        shared_md5 = "deadbeef1234567890abcdef12345678"
        ds1 = {1: _make_audio_clip(1, md5=shared_md5), 2: _make_audio_clip(2)}
        ds2 = {1: _make_audio_clip(3, md5=shared_md5), 2: _make_audio_clip(4)}
        p1, p2 = tmp_path / "ds1.pkl", tmp_path / "ds2.pkl"
        _write_pickle_dataset(p1, ds1)
        _write_pickle_dataset(p2, ds2)

        clips: dict = {}
        IMPORTER.run({"datasets": [str(p1), str(p2)]}, clips)

        assert len(clips) == 3  # 4 total minus 1 duplicate
        md5s = [c["md5"] for c in clips.values()]
        assert md5s.count(shared_md5) == 1

    def test_media_type_mismatch_raises(self, tmp_path):
        """Combining audio and image datasets raises ValueError."""
        from vtsearch.datasets.importers.combine_datasets import IMPORTER

        ds1 = {1: _make_audio_clip(1)}
        ds2 = {1: _make_image_clip(2)}
        p1, p2 = tmp_path / "audio.pkl", tmp_path / "image.pkl"
        _write_pickle_dataset(p1, ds1)
        _write_pickle_dataset(p2, ds2)

        clips: dict = {}
        with pytest.raises(ValueError, match="Media type mismatch"):
            IMPORTER.run({"datasets": [str(p1), str(p2)]}, clips)

    def test_fewer_than_two_datasets_raises(self, tmp_path):
        """Providing only one dataset raises ValueError."""
        from vtsearch.datasets.importers.combine_datasets import IMPORTER

        ds1 = {1: _make_audio_clip(1)}
        p1 = tmp_path / "ds1.pkl"
        _write_pickle_dataset(p1, ds1)

        clips: dict = {}
        with pytest.raises(ValueError, match="At least two"):
            IMPORTER.run({"datasets": [str(p1)]}, clips)

    def test_missing_file_raises(self, tmp_path):
        """A non-existent path raises FileNotFoundError."""
        from vtsearch.datasets.importers.combine_datasets import IMPORTER

        ds1 = {1: _make_audio_clip(1)}
        p1 = tmp_path / "ds1.pkl"
        _write_pickle_dataset(p1, ds1)

        clips: dict = {}
        with pytest.raises(FileNotFoundError):
            IMPORTER.run({"datasets": [str(p1), str(tmp_path / "missing.pkl")]}, clips)

    def test_comma_separated_string_input(self, tmp_path):
        """The datasets field also accepts a comma-separated string."""
        from vtsearch.datasets.importers.combine_datasets import IMPORTER

        ds1 = {1: _make_audio_clip(1)}
        ds2 = {1: _make_audio_clip(2)}
        p1, p2 = tmp_path / "ds1.pkl", tmp_path / "ds2.pkl"
        _write_pickle_dataset(p1, ds1)
        _write_pickle_dataset(p2, ds2)

        clips: dict = {}
        IMPORTER.run({"datasets": f"{p1},{p2}"}, clips)

        assert len(clips) == 2

    def test_empty_dataset_skipped(self, tmp_path):
        """An empty pickle file is skipped without error."""
        from vtsearch.datasets.importers.combine_datasets import IMPORTER

        ds1 = {1: _make_audio_clip(1)}
        empty = {}
        p1, p2 = tmp_path / "ds1.pkl", tmp_path / "empty.pkl"
        _write_pickle_dataset(p1, ds1)
        _write_pickle_dataset(p2, empty)

        # Need a third non-empty dataset to meet minimum of 2 datasets
        ds3 = {1: _make_audio_clip(3)}
        p3 = tmp_path / "ds3.pkl"
        _write_pickle_dataset(p3, ds3)

        clips: dict = {}
        IMPORTER.run({"datasets": [str(p1), str(p2), str(p3)]}, clips)

        assert len(clips) == 2

    def test_preserves_clip_data_fields(self, tmp_path):
        """Merged clips retain their original data fields."""
        from vtsearch.datasets.importers.combine_datasets import IMPORTER

        ds1 = {1: _make_audio_clip(1, filename="song_a.wav")}
        ds2 = {1: _make_audio_clip(2, filename="song_b.wav")}
        p1, p2 = tmp_path / "ds1.pkl", tmp_path / "ds2.pkl"
        _write_pickle_dataset(p1, ds1)
        _write_pickle_dataset(p2, ds2)

        clips: dict = {}
        IMPORTER.run({"datasets": [str(p1), str(p2)]}, clips)

        filenames = {c["filename"] for c in clips.values()}
        assert "song_a.wav" in filenames
        assert "song_b.wav" in filenames

    def test_three_datasets_combined(self, tmp_path):
        """Three datasets combine correctly."""
        from vtsearch.datasets.importers.combine_datasets import IMPORTER

        ds1 = {1: _make_audio_clip(1)}
        ds2 = {1: _make_audio_clip(2)}
        ds3 = {1: _make_audio_clip(3)}
        p1, p2, p3 = tmp_path / "a.pkl", tmp_path / "b.pkl", tmp_path / "c.pkl"
        _write_pickle_dataset(p1, ds1)
        _write_pickle_dataset(p2, ds2)
        _write_pickle_dataset(p3, ds3)

        clips: dict = {}
        IMPORTER.run({"datasets": [str(p1), str(p2), str(p3)]}, clips)

        assert len(clips) == 3


# ---------------------------------------------------------------------------
# CLI support
# ---------------------------------------------------------------------------


class TestCombineDatasetsCli:
    def test_run_cli_delegates_to_run(self, tmp_path):
        from vtsearch.datasets.importers.combine_datasets import IMPORTER

        ds1 = {1: _make_audio_clip(1)}
        ds2 = {1: _make_audio_clip(2)}
        p1, p2 = tmp_path / "ds1.pkl", tmp_path / "ds2.pkl"
        _write_pickle_dataset(p1, ds1)
        _write_pickle_dataset(p2, ds2)

        clips: dict = {}
        IMPORTER.run_cli({"datasets": f"{p1},{p2}"}, clips)

        assert len(clips) == 2


# ---------------------------------------------------------------------------
# build_origin
# ---------------------------------------------------------------------------


class TestCombineDatasetsOrigin:
    def test_build_origin_with_list(self):
        from vtsearch.datasets.importers.combine_datasets import IMPORTER

        origin = IMPORTER.build_origin({"datasets": ["/a.pkl", "/b.pkl"]})
        assert origin["importer"] == "combine_datasets"
        assert "/a.pkl" in origin["params"]["datasets"]
        assert "/b.pkl" in origin["params"]["datasets"]

    def test_build_origin_with_string(self):
        from vtsearch.datasets.importers.combine_datasets import IMPORTER

        origin = IMPORTER.build_origin({"datasets": "/a.pkl,/b.pkl"})
        assert origin["importer"] == "combine_datasets"
        assert origin["params"]["datasets"] == "/a.pkl,/b.pkl"


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------


class TestAvailableFilesEndpoint:
    def test_returns_list(self, client):
        resp = client.get("/api/dataset/available-files")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "files" in data
        assert isinstance(data["files"], list)

    def test_lists_pkl_files(self, client, tmp_path):
        """When EMBEDDINGS_DIR contains .pkl files, they appear in the list."""
        from vtsearch.config import EMBEDDINGS_DIR

        EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
        test_pkl = EMBEDDINGS_DIR / "_test_combine.pkl"
        test_pkl.write_bytes(pickle.dumps({"clips": {}}))
        try:
            resp = client.get("/api/dataset/available-files")
            data = resp.get_json()
            names = [f["name"] for f in data["files"]]
            assert "_test_combine" in names
            # Check fields
            entry = next(f for f in data["files"] if f["name"] == "_test_combine")
            assert "path" in entry
            assert "size_mb" in entry
        finally:
            test_pkl.unlink(missing_ok=True)


class TestCombineEndpoint:
    def test_rejects_fewer_than_two(self, client):
        resp = client.post(
            "/api/dataset/combine",
            json={"datasets": ["/one.pkl"]},
        )
        assert resp.status_code == 400

    def test_rejects_missing_file(self, client, tmp_path):
        resp = client.post(
            "/api/dataset/combine",
            json={"datasets": ["/nonexistent_a.pkl", "/nonexistent_b.pkl"]},
        )
        assert resp.status_code == 404

    def test_accepts_valid_request(self, client, tmp_path):
        """A valid request returns 200 with ok=True."""
        ds1 = {1: _make_audio_clip(1)}
        ds2 = {1: _make_audio_clip(2)}
        p1, p2 = tmp_path / "ds1.pkl", tmp_path / "ds2.pkl"
        _write_pickle_dataset(p1, ds1)
        _write_pickle_dataset(p2, ds2)

        resp = client.post(
            "/api/dataset/combine",
            json={"datasets": [str(p1), str(p2)]},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ok"] is True
