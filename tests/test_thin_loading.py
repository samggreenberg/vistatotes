"""Tests for thin (lazy-loading) media reference support.

Verifies that datasets can be loaded in thin mode (storing media_path
instead of clip_bytes) and that lazy loading correctly resolves media
content when needed.
"""

import hashlib
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from vtsearch.datasets.loader import (
    _streaming_md5,
    load_dataset_from_folder,
    load_dataset_from_pickle,
)


def _make_wav_bytes(frequency: float = 440.0, duration: float = 0.1) -> bytes:
    """Generate a minimal WAV file for testing."""
    from vtsearch.audio import generate_wav

    return generate_wav(frequency, duration)


def _make_text_file(tmp_dir: Path, name: str, content: str) -> Path:
    """Write a text file and return its path."""
    p = tmp_dir / name
    p.write_text(content, encoding="utf-8")
    return p


def _make_wav_file(tmp_dir: Path, name: str) -> Path:
    """Write a WAV file and return its path."""
    p = tmp_dir / name
    p.write_bytes(_make_wav_bytes())
    return p


class TestStreamingMD5:
    def test_matches_regular_md5(self, tmp_path):
        content = b"hello world test data"
        p = tmp_path / "test.bin"
        p.write_bytes(content)
        assert _streaming_md5(p) == hashlib.md5(content).hexdigest()

    def test_large_file(self, tmp_path):
        # File larger than the 8192 chunk size
        content = b"x" * 20000
        p = tmp_path / "large.bin"
        p.write_bytes(content)
        assert _streaming_md5(p) == hashlib.md5(content).hexdigest()


class TestThinLoadFromFolder:
    """Test load_dataset_from_folder with thin=True."""

    def test_thin_clips_have_media_path(self, tmp_path):
        _make_wav_file(tmp_path, "test1.wav")
        _make_wav_file(tmp_path, "test2.wav")
        clips: dict[int, dict[str, Any]] = {}
        load_dataset_from_folder(tmp_path, "sounds", clips, thin=True)
        assert len(clips) == 2
        for clip in clips.values():
            assert clip["media_path"] is not None
            assert Path(clip["media_path"]).exists()

    def test_thin_clips_have_no_bytes(self, tmp_path):
        _make_wav_file(tmp_path, "test.wav")
        clips: dict[int, dict[str, Any]] = {}
        load_dataset_from_folder(tmp_path, "sounds", clips, thin=True)
        clip = clips[1]
        assert clip["clip_bytes"] is None
        assert clip["clip_string"] is None

    def test_thin_clips_have_embedding(self, tmp_path):
        _make_wav_file(tmp_path, "test.wav")
        clips: dict[int, dict[str, Any]] = {}
        load_dataset_from_folder(tmp_path, "sounds", clips, thin=True)
        clip = clips[1]
        assert isinstance(clip["embedding"], np.ndarray)
        assert len(clip["embedding"]) > 0

    def test_thin_clips_have_correct_file_size(self, tmp_path):
        wav_path = _make_wav_file(tmp_path, "test.wav")
        expected_size = wav_path.stat().st_size
        clips: dict[int, dict[str, Any]] = {}
        load_dataset_from_folder(tmp_path, "sounds", clips, thin=True)
        assert clips[1]["file_size"] == expected_size

    def test_thin_clips_have_correct_md5(self, tmp_path):
        wav_path = _make_wav_file(tmp_path, "test.wav")
        expected_md5 = hashlib.md5(wav_path.read_bytes()).hexdigest()
        clips: dict[int, dict[str, Any]] = {}
        load_dataset_from_folder(tmp_path, "sounds", clips, thin=True)
        assert clips[1]["md5"] == expected_md5

    def test_thin_no_duration(self, tmp_path):
        """Thin mode skips load_clip_data, so duration stays at default 0."""
        _make_wav_file(tmp_path, "test.wav")
        clips: dict[int, dict[str, Any]] = {}
        load_dataset_from_folder(tmp_path, "sounds", clips, thin=True)
        assert clips[1]["duration"] == 0

    def test_full_mode_has_bytes(self, tmp_path):
        """Full mode (thin=False) should still load bytes as before."""
        _make_wav_file(tmp_path, "test.wav")
        clips: dict[int, dict[str, Any]] = {}
        load_dataset_from_folder(tmp_path, "sounds", clips, thin=False)
        assert clips[1]["clip_bytes"] is not None
        assert isinstance(clips[1]["clip_bytes"], bytes)

    def test_full_mode_also_has_media_path(self, tmp_path):
        """Full mode should also store media_path for potential future use."""
        _make_wav_file(tmp_path, "test.wav")
        clips: dict[int, dict[str, Any]] = {}
        load_dataset_from_folder(tmp_path, "sounds", clips, thin=False)
        assert clips[1]["media_path"] is not None


class TestThinLoadFromPickle:
    """Test load_dataset_from_pickle with thin=True."""

    def _make_pickle(self, tmp_path, inline_bytes=True, audio_dir=None):
        """Create a test pickle with one audio clip."""
        wav_bytes = _make_wav_bytes()
        clip_data: dict[str, Any] = {
            "id": 1,
            "type": "audio",
            "duration": 0.1,
            "file_size": len(wav_bytes),
            "md5": hashlib.md5(wav_bytes).hexdigest(),
            "embedding": np.zeros(512).tolist(),
            "filename": "test.wav",
            "category": "test",
        }
        if inline_bytes:
            clip_data["clip_bytes"] = wav_bytes

        pkl_data: dict[str, Any] = {"clips": {1: clip_data}}
        if audio_dir:
            pkl_data["audio_dir"] = str(audio_dir)
            # Write the actual file
            audio_dir.mkdir(exist_ok=True)
            (audio_dir / "test.wav").write_bytes(wav_bytes)

        pkl_path = tmp_path / "test.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(pkl_data, f)
        return pkl_path

    def test_thin_pickle_skips_inline_bytes(self, tmp_path):
        pkl_path = self._make_pickle(tmp_path, inline_bytes=True)
        clips: dict[int, dict[str, Any]] = {}
        load_dataset_from_pickle(pkl_path, clips, thin=True)
        assert len(clips) == 1
        assert clips[1]["clip_bytes"] is None

    def test_thin_pickle_has_embedding(self, tmp_path):
        pkl_path = self._make_pickle(tmp_path, inline_bytes=True)
        clips: dict[int, dict[str, Any]] = {}
        load_dataset_from_pickle(pkl_path, clips, thin=True)
        assert isinstance(clips[1]["embedding"], np.ndarray)

    def test_thin_pickle_resolves_media_path_from_audio_dir(self, tmp_path):
        audio_dir = tmp_path / "audio"
        pkl_path = self._make_pickle(tmp_path, inline_bytes=False, audio_dir=audio_dir)
        clips: dict[int, dict[str, Any]] = {}
        load_dataset_from_pickle(pkl_path, clips, thin=True)
        assert clips[1]["media_path"] is not None
        assert Path(clips[1]["media_path"]).exists()

    def test_thin_pickle_preserves_metadata(self, tmp_path):
        pkl_path = self._make_pickle(tmp_path, inline_bytes=True)
        clips: dict[int, dict[str, Any]] = {}
        load_dataset_from_pickle(pkl_path, clips, thin=True)
        assert clips[1]["type"] == "audio"
        assert clips[1]["filename"] == "test.wav"
        assert clips[1]["category"] == "test"

    def test_full_pickle_still_works(self, tmp_path):
        pkl_path = self._make_pickle(tmp_path, inline_bytes=True)
        clips: dict[int, dict[str, Any]] = {}
        load_dataset_from_pickle(pkl_path, clips, thin=False)
        assert clips[1]["clip_bytes"] is not None


class TestPickleMD5Preservation:
    """Test that load_dataset_from_pickle uses pre-existing MD5 from pickle data."""

    def test_full_mode_uses_md5_from_pickle_when_present(self, tmp_path):
        """Full mode should use the MD5 stored in the pickle instead of recalculating."""
        wav_bytes = _make_wav_bytes()
        pre_md5 = "a" * 32  # A fake MD5 that differs from the real hash
        pkl_data = {
            "clips": {
                1: {
                    "id": 1,
                    "type": "audio",
                    "duration": 0.1,
                    "file_size": len(wav_bytes),
                    "md5": pre_md5,
                    "embedding": np.zeros(512).tolist(),
                    "filename": "test.wav",
                    "category": "test",
                    "clip_bytes": wav_bytes,
                }
            }
        }
        pkl_path = tmp_path / "test.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(pkl_data, f)

        clips: dict[int, dict[str, Any]] = {}
        load_dataset_from_pickle(pkl_path, clips, thin=False)
        assert clips[1]["md5"] == pre_md5

    def test_full_mode_computes_md5_when_missing_from_pickle(self, tmp_path):
        """Full mode should compute the MD5 if the pickle doesn't have one."""
        wav_bytes = _make_wav_bytes()
        pkl_data = {
            "clips": {
                1: {
                    "id": 1,
                    "type": "audio",
                    "duration": 0.1,
                    "file_size": len(wav_bytes),
                    # no "md5" key
                    "embedding": np.zeros(512).tolist(),
                    "filename": "test.wav",
                    "category": "test",
                    "clip_bytes": wav_bytes,
                }
            }
        }
        pkl_path = tmp_path / "test.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(pkl_data, f)

        clips: dict[int, dict[str, Any]] = {}
        load_dataset_from_pickle(pkl_path, clips, thin=False)
        assert clips[1]["md5"] == hashlib.md5(wav_bytes).hexdigest()

    def test_thin_mode_uses_md5_from_pickle(self, tmp_path):
        """Thin mode should also preserve the MD5 from the pickle."""
        wav_bytes = _make_wav_bytes()
        pre_md5 = "b" * 32
        pkl_data = {
            "clips": {
                1: {
                    "id": 1,
                    "type": "audio",
                    "duration": 0.1,
                    "file_size": len(wav_bytes),
                    "md5": pre_md5,
                    "embedding": np.zeros(512).tolist(),
                    "filename": "test.wav",
                    "category": "test",
                    "clip_bytes": wav_bytes,
                }
            }
        }
        pkl_path = tmp_path / "test.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(pkl_data, f)

        clips: dict[int, dict[str, Any]] = {}
        load_dataset_from_pickle(pkl_path, clips, thin=True)
        assert clips[1]["md5"] == pre_md5


class TestThinImporters:
    """Test that importers pass thin parameter through correctly."""

    def test_folder_importer_thin(self, tmp_path):
        _make_wav_file(tmp_path, "test.wav")
        from vtsearch.datasets.importers.folder import FolderDatasetImporter

        importer = FolderDatasetImporter()
        clips: dict[int, dict[str, Any]] = {}
        importer.run({"path": str(tmp_path), "media_type": "sounds"}, clips, thin=True)
        assert len(clips) > 0
        assert clips[1]["clip_bytes"] is None
        assert clips[1]["media_path"] is not None

    def test_folder_importer_run_cli_thin(self, tmp_path):
        _make_wav_file(tmp_path, "test.wav")
        from vtsearch.datasets.importers.folder import FolderDatasetImporter

        importer = FolderDatasetImporter()
        clips: dict[int, dict[str, Any]] = {}
        importer.run_cli({"path": str(tmp_path), "media_type": "sounds"}, clips, thin=True)
        assert len(clips) > 0
        assert clips[1]["clip_bytes"] is None

    def test_pickle_importer_thin(self, tmp_path):
        # Create a pickle first
        wav_bytes = _make_wav_bytes()
        pkl_data = {
            "clips": {
                1: {
                    "id": 1,
                    "type": "audio",
                    "duration": 0.1,
                    "file_size": len(wav_bytes),
                    "md5": hashlib.md5(wav_bytes).hexdigest(),
                    "embedding": np.zeros(512).tolist(),
                    "filename": "test.wav",
                    "category": "test",
                    "clip_bytes": wav_bytes,
                }
            }
        }
        pkl_path = tmp_path / "test.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(pkl_data, f)

        from vtsearch.datasets.importers.pickle import PickleDatasetImporter

        importer = PickleDatasetImporter()
        clips: dict[int, dict[str, Any]] = {}
        importer.run_cli({"file": str(pkl_path)}, clips, thin=True)
        assert len(clips) == 1
        assert clips[1]["clip_bytes"] is None


class TestLazyLoadingMediaType:
    """Test that MediaType._resolve_clip_bytes/string lazy-loads from media_path."""

    def test_resolve_bytes_from_preloaded(self):
        from vtsearch.media.audio.media_type import AudioMediaType

        mt = AudioMediaType()
        clip = {"clip_bytes": b"hello", "media_path": None}
        assert mt._resolve_clip_bytes(clip) == b"hello"

    def test_resolve_bytes_from_media_path(self, tmp_path):
        from vtsearch.media.audio.media_type import AudioMediaType

        content = b"lazy loaded content"
        p = tmp_path / "test.wav"
        p.write_bytes(content)

        mt = AudioMediaType()
        clip = {"clip_bytes": None, "media_path": str(p)}
        assert mt._resolve_clip_bytes(clip) == content

    def test_resolve_bytes_missing_file(self):
        from vtsearch.media.audio.media_type import AudioMediaType

        mt = AudioMediaType()
        clip = {"clip_bytes": None, "media_path": "/nonexistent/file.wav"}
        assert mt._resolve_clip_bytes(clip) is None

    def test_resolve_bytes_no_path(self):
        from vtsearch.media.audio.media_type import AudioMediaType

        mt = AudioMediaType()
        clip = {"clip_bytes": None, "media_path": None}
        assert mt._resolve_clip_bytes(clip) is None

    def test_resolve_string_from_preloaded(self):
        from vtsearch.media.text.media_type import TextMediaType

        mt = TextMediaType()
        clip = {"clip_string": "hello world", "media_path": None}
        assert mt._resolve_clip_string(clip) == "hello world"

    def test_resolve_string_from_media_path(self, tmp_path):
        from vtsearch.media.text.media_type import TextMediaType

        content = "lazy loaded text content"
        p = tmp_path / "test.txt"
        p.write_text(content, encoding="utf-8")

        mt = TextMediaType()
        clip = {"clip_string": None, "media_path": str(p)}
        assert mt._resolve_clip_string(clip) == content


class TestClipResponseLazyLoading:
    """Test that clip_response works with lazy-loaded media."""

    def test_audio_clip_response_lazy(self, tmp_path):
        from vtsearch.media.audio.media_type import AudioMediaType

        wav_bytes = _make_wav_bytes()
        p = tmp_path / "test.wav"
        p.write_bytes(wav_bytes)

        mt = AudioMediaType()
        clip = {"id": 1, "clip_bytes": None, "media_path": str(p), "filename": "test.wav"}
        resp = mt.clip_response(clip)
        assert resp.data == wav_bytes
        assert resp.mimetype == "audio/wav"

    def test_image_clip_response_lazy(self, tmp_path):
        from vtsearch.media.image.media_type import ImageMediaType

        # Create a minimal PNG
        from PIL import Image as PILImage
        import io

        img = PILImage.new("RGB", (2, 2), color="red")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        p = tmp_path / "test.png"
        p.write_bytes(png_bytes)

        mt = ImageMediaType()
        clip = {"id": 1, "clip_bytes": None, "media_path": str(p), "filename": "test.png"}
        resp = mt.clip_response(clip)
        assert resp.data == png_bytes
        assert resp.mimetype == "image/png"

    def test_text_clip_response_lazy(self, tmp_path):
        from vtsearch.media.text.media_type import TextMediaType

        content = "lazy loaded paragraph"
        p = tmp_path / "test.txt"
        p.write_text(content, encoding="utf-8")

        mt = TextMediaType()
        clip = {
            "id": 1,
            "clip_string": None,
            "media_path": str(p),
            "word_count": 0,
            "character_count": 0,
        }
        resp = mt.clip_response(clip)
        assert resp.data["content"] == content
        assert resp.data["word_count"] == 3  # "lazy loaded paragraph"
        assert resp.data["character_count"] == len(content)

    def test_audio_clip_response_no_data(self):
        """clip_response returns empty bytes when no data is available."""
        from vtsearch.media.audio.media_type import AudioMediaType

        mt = AudioMediaType()
        clip = {"id": 1, "clip_bytes": None, "media_path": None, "filename": "test.wav"}
        resp = mt.clip_response(clip)
        assert resp.data == b""
