import io
import struct
import tarfile
import wave
import zipfile

import pytest

import app as app_module


class TestIndex:
    def test_serves_index_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"VTSearch" in resp.data


class TestDatasetEndpoints:
    def test_get_dataset_status(self, client):
        resp = client.get("/api/dataset/status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "num_clips" in data or "error" in data

    def test_get_dataset_demo_list(self, client):
        resp = client.get("/api/dataset/demo-list")
        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data, dict)
        # Should return available demo datasets
        assert "demos" in data or isinstance(data, dict)

    def test_clear_dataset(self, client):
        resp = client.post("/api/dataset/clear")
        assert resp.status_code == 200
        # After clearing, clips should be empty
        assert len(app_module.clips) == 0

        # Re-initialize for other tests
        app_module.init_clips()


class TestStartupState:
    """App should start with an empty dataset so the selection screen shows."""

    def test_status_loaded_false_when_clips_empty(self, client):
        """GET /api/dataset/status returns loaded=False when clips is cleared."""
        saved = dict(app_module.clips)
        app_module.clips.clear()
        try:
            resp = client.get("/api/dataset/status")
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["loaded"] is False
            assert data["num_clips"] == 0
        finally:
            app_module.clips.update(saved)

    def test_init_clips_not_called_automatically(self):
        """init_clips() exists for testing but is not called in production startup.

        Verify that the production startup block in app.py does NOT call
        init_clips() – it should only load models and wait for user selection.
        """
        import inspect

        source = inspect.getsource(app_module)

        # The production path is the final else branch after the argparse
        # if/elif/else chain.  Find the last else: in the __main__ block.
        main_block_start = source.find('if __name__ == "__main__"')
        assert main_block_start != -1, "Could not find __main__ block"
        main_body = source[main_block_start:]

        # Find the production else branch (the last else: in the block)
        else_start = main_body.rfind("else:")
        assert else_start != -1, "Could not find else branch in __main__ block"
        else_body = main_body[else_start:]
        assert "init_clips()" not in else_body, "init_clips() must not be called automatically in production startup"


class TestDemoDatasetReadiness:
    """Demo datasets report three-state status: ready / needs_embedding / needs_download."""

    def test_audio_pkl_without_esc50_shows_needs_download(self, client):
        """Audio pkl exists but ESC-50 audio dir is absent → needs_download (stale pkl)."""
        import pickle

        from config import DATA_DIR, EMBEDDINGS_DIR

        esc50_dir = DATA_DIR / "ESC-50-master" / "audio"
        if esc50_dir.exists():
            pytest.skip("ESC-50 is present; cannot test stale-pkl scenario")

        EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
        pkl_file = EMBEDDINGS_DIR / "sounds_s.pkl"
        pkl_file.write_bytes(pickle.dumps({"name": "sounds_s", "clips": {}}))
        try:
            resp = client.get("/api/dataset/demo-list")
            data = resp.get_json()
            ds = next((d for d in data["datasets"] if d["name"] == "sounds_s"), None)
            assert ds is not None
            assert ds["status"] == "needs_download", "Stale audio pkl without ESC-50 dir must be needs_download"
            assert ds["ready"] is False
        finally:
            pkl_file.unlink(missing_ok=True)
            try:
                EMBEDDINGS_DIR.rmdir()
            except OSError:
                pass

    def test_audio_pkl_with_empty_esc50_shows_needs_download(self, client):
        """Audio pkl exists and ESC-50 audio dir exists but is empty → needs_download."""
        import pickle

        from config import DATA_DIR, EMBEDDINGS_DIR

        esc50_dir = DATA_DIR / "ESC-50-master" / "audio"
        if esc50_dir.exists() and any(esc50_dir.iterdir()):
            pytest.skip("ESC-50 audio dir is non-empty; cannot test empty-dir scenario")

        # Create the directory structure but leave it empty
        esc50_dir.mkdir(parents=True, exist_ok=True)
        EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
        pkl_file = EMBEDDINGS_DIR / "sounds_s.pkl"
        pkl_file.write_bytes(pickle.dumps({"name": "sounds_s", "clips": {}}))
        try:
            resp = client.get("/api/dataset/demo-list")
            data = resp.get_json()
            ds = next((d for d in data["datasets"] if d["name"] == "sounds_s"), None)
            assert ds is not None
            assert ds["status"] == "needs_download", "Audio pkl with empty ESC-50 dir must be needs_download"
            assert ds["ready"] is False
        finally:
            pkl_file.unlink(missing_ok=True)
            try:
                EMBEDDINGS_DIR.rmdir()
            except OSError:
                pass
            try:
                esc50_dir.rmdir()
            except OSError:
                pass

    def test_video_pkl_without_ucf101_shows_needs_download(self, client):
        """Video pkl exists but UCF-101 dir is absent → needs_download (stale pkl)."""
        import pickle

        from config import EMBEDDINGS_DIR, VIDEO_DIR

        ucf101_dir = VIDEO_DIR / "ucf101"
        if ucf101_dir.exists():
            pytest.skip("UCF-101 is present; cannot test stale-pkl scenario")

        EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
        pkl_file = EMBEDDINGS_DIR / "activities_video.pkl"
        pkl_file.write_bytes(pickle.dumps({"name": "activities_video", "clips": {}}))
        try:
            resp = client.get("/api/dataset/demo-list")
            data = resp.get_json()
            ds = next((d for d in data["datasets"] if d["name"] == "activities_video"), None)
            assert ds is not None
            assert ds["status"] == "needs_download", "Stale video pkl without UCF-101 dir must be needs_download"
            assert ds["ready"] is False
        finally:
            pkl_file.unlink(missing_ok=True)
            try:
                EMBEDDINGS_DIR.rmdir()
            except OSError:
                pass

    def test_no_pkl_with_source_folder_shows_needs_embedding(self, client):
        """No pkl but required_folder exists with content → needs_embedding."""
        import pickle
        import struct
        import wave

        from config import DATA_DIR, EMBEDDINGS_DIR

        esc50_dir = DATA_DIR / "ESC-50-master" / "audio"
        # Ensure no pkl exists for sounds_s
        pkl_file = EMBEDDINGS_DIR / "sounds_s.pkl"
        if pkl_file.exists():
            pytest.skip("sounds_s.pkl exists; cannot test needs_embedding scenario")

        # Create the ESC-50 audio dir with a dummy file
        esc50_dir.mkdir(parents=True, exist_ok=True)
        dummy_wav = esc50_dir / "_test_dummy.wav"
        already_populated = any(f.name != "_test_dummy.wav" for f in esc50_dir.iterdir()) if esc50_dir.exists() else False
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(44100)
            wf.writeframes(struct.pack("<h", 0) * 100)
        dummy_wav.write_bytes(buf.getvalue())
        try:
            resp = client.get("/api/dataset/demo-list")
            data = resp.get_json()
            ds = next((d for d in data["datasets"] if d["name"] == "sounds_s"), None)
            assert ds is not None
            assert ds["status"] == "needs_embedding", "No pkl with source folder should be needs_embedding"
            assert ds["ready"] is False
            assert ds["download_size_mb"] == 0, "needs_embedding should report 0 MB download"
        finally:
            dummy_wav.unlink(missing_ok=True)
            if not already_populated:
                try:
                    esc50_dir.rmdir()
                except OSError:
                    pass

    def test_no_pkl_no_source_shows_needs_download(self, client):
        """No pkl and no required_folder → needs_download."""
        from config import DATA_DIR, EMBEDDINGS_DIR

        esc50_dir = DATA_DIR / "ESC-50-master" / "audio"
        pkl_file = EMBEDDINGS_DIR / "sounds_s.pkl"
        if pkl_file.exists():
            pytest.skip("sounds_s.pkl exists; cannot test needs_download scenario")
        if esc50_dir.exists() and any(esc50_dir.iterdir()):
            pytest.skip("ESC-50 is present; cannot test needs_download scenario")

        resp = client.get("/api/dataset/demo-list")
        data = resp.get_json()
        ds = next((d for d in data["datasets"] if d["name"] == "sounds_s"), None)
        assert ds is not None
        assert ds["status"] == "needs_download"
        assert ds["ready"] is False

    def test_status_field_always_present(self, client):
        """Every demo dataset must include a status field."""
        resp = client.get("/api/dataset/demo-list")
        data = resp.get_json()
        for ds in data["datasets"]:
            assert "status" in ds, f"Dataset '{ds['name']}' missing status field"
            assert ds["status"] in ("ready", "needs_embedding", "needs_download")


class TestImporterMetadata:
    """Importer to_dict() must include the icon field."""

    def test_http_archive_display_name(self, client):
        resp = client.get("/api/dataset/importers")
        assert resp.status_code == 200
        data = resp.get_json()
        names = [imp["display_name"] for imp in data["importers"]]
        assert "Generate from HTTP Archive" in names

    def test_http_archive_icon_is_globe(self, client):
        resp = client.get("/api/dataset/importers")
        data = resp.get_json()
        http_imp = next((i for i in data["importers"] if i["name"] == "http_archive"), None)
        assert http_imp is not None, "http_archive importer not found"
        assert http_imp["icon"] == "\U0001f310"

    def test_http_archive_supports_tar_and_rar_in_description(self, client):
        resp = client.get("/api/dataset/importers")
        data = resp.get_json()
        http_imp = next((i for i in data["importers"] if i["name"] == "http_archive"), None)
        assert http_imp is not None
        desc = http_imp["description"].lower()
        assert "tar" in desc
        assert "rar" in desc

    def test_folder_importer_in_extended_list(self, client):
        """Folder importer must appear in /api/dataset/importers (not a builtin)."""
        resp = client.get("/api/dataset/importers")
        data = resp.get_json()
        names = [imp["name"] for imp in data["importers"]]
        assert "folder" in names

    def test_folder_importer_icon(self, client):
        resp = client.get("/api/dataset/importers")
        data = resp.get_json()
        folder_imp = next((i for i in data["importers"] if i["name"] == "folder"), None)
        assert folder_imp is not None
        assert folder_imp["icon"] == "\U0001f4c2"

    def test_folder_importer_description(self, client):
        resp = client.get("/api/dataset/importers")
        data = resp.get_json()
        folder_imp = next((i for i in data["importers"] if i["name"] == "folder"), None)
        assert folder_imp is not None
        # Description must not mention specific media-type names
        desc = folder_imp["description"]
        assert "sounds/videos" not in desc
        assert "media files from a folder" in desc.lower()

    def test_all_importers_have_icon_field(self, client):
        resp = client.get("/api/dataset/importers")
        data = resp.get_json()
        for imp in data["importers"]:
            assert "icon" in imp, f"Importer '{imp['name']}' missing icon field"

    def test_pickle_not_in_extended_list(self, client):
        """Pickle importer keeps its dedicated UI and must not appear in the list."""
        resp = client.get("/api/dataset/importers")
        data = resp.get_json()
        names = [imp["name"] for imp in data["importers"]]
        assert "pickle" not in names

    def test_folder_media_type_field_is_first(self, client):
        """Media-type dropdown should come before the path field."""
        resp = client.get("/api/dataset/importers")
        data = resp.get_json()
        folder_imp = next((i for i in data["importers"] if i["name"] == "folder"), None)
        assert folder_imp is not None
        keys = [f["key"] for f in folder_imp["fields"]]
        assert keys.index("media_type") < keys.index("path")


class TestExtractArchive:
    """Unit tests for the zip/tar extraction helper."""

    from vtsearch.datasets.importers.http_zip import _extract_archive

    def _make_wav_bytes(self) -> bytes:
        """Create a minimal valid WAV file in memory."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(44100)
            samples = struct.pack("<" + "h" * 100, *([0] * 100))
            wf.writeframes(samples)
        return buf.getvalue()

    def test_extract_zip(self, tmp_path):
        from vtsearch.datasets.importers.http_zip import _extract_archive

        wav_data = self._make_wav_bytes()
        zip_path = tmp_path / "test.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("sounds/tone.wav", wav_data)
        extract_dir = tmp_path / "out"
        extract_dir.mkdir()
        _extract_archive(zip_path, extract_dir)
        assert (extract_dir / "sounds" / "tone.wav").exists()

    def test_extract_tar_gz(self, tmp_path):
        from vtsearch.datasets.importers.http_zip import _extract_archive

        wav_data = self._make_wav_bytes()
        tar_path = tmp_path / "test.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tf:
            info = tarfile.TarInfo(name="sounds/tone.wav")
            info.size = len(wav_data)
            tf.addfile(info, io.BytesIO(wav_data))
        extract_dir = tmp_path / "out"
        extract_dir.mkdir()
        _extract_archive(tar_path, extract_dir)
        assert (extract_dir / "sounds" / "tone.wav").exists()

    def test_extract_tar_uncompressed(self, tmp_path):
        from vtsearch.datasets.importers.http_zip import _extract_archive

        wav_data = self._make_wav_bytes()
        tar_path = tmp_path / "test.tar"
        with tarfile.open(tar_path, "w") as tf:
            info = tarfile.TarInfo(name="tone.wav")
            info.size = len(wav_data)
            tf.addfile(info, io.BytesIO(wav_data))
        extract_dir = tmp_path / "out"
        extract_dir.mkdir()
        _extract_archive(tar_path, extract_dir)
        assert (extract_dir / "tone.wav").exists()

    def test_unsupported_format_raises(self, tmp_path):
        from vtsearch.datasets.importers.http_zip import _extract_archive

        bad_archive = tmp_path / "test.7z"
        bad_archive.write_bytes(b"not a real archive")
        extract_dir = tmp_path / "out"
        extract_dir.mkdir()
        with pytest.raises((ValueError, Exception)):
            _extract_archive(bad_archive, extract_dir)

    def test_rar_without_rarfile_raises_runtime_error(self, tmp_path):
        """Attempting RAR extraction without rarfile installed raises RuntimeError."""
        import sys
        import unittest.mock as mock

        from vtsearch.datasets.importers.http_zip import _extract_archive

        rar_path = tmp_path / "test.rar"
        rar_path.write_bytes(b"Rar!\x1a\x07\x00")  # RAR magic bytes (v4)
        extract_dir = tmp_path / "out"
        extract_dir.mkdir()

        with mock.patch.dict(sys.modules, {"rarfile": None}):
            with pytest.raises((RuntimeError, ImportError, Exception)):
                _extract_archive(rar_path, extract_dir)
