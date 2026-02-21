"""Tests for the importer subsystem that don't require ML/torch dependencies.

These tests verify:
- HTTP Archive importer metadata (name, icon, description)
- Folder importer metadata (icon, description, field ordering)
- _extract_archive helper (zip and tar)
- DatasetImporter base class icon field and content_vectors attribute
- Folder importer is not in _BUILTIN_IMPORTER_NAMES
- load_dataset_from_folder content_vectors support
"""

from __future__ import annotations

import io
import struct
import tarfile
import wave
import zipfile

import pytest


def _make_wav_bytes() -> bytes:
    """Create a minimal valid WAV file in memory."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        samples = struct.pack("<" + "h" * 100, *([0] * 100))
        wf.writeframes(samples)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# DatasetImporter base class – icon field and content_vectors
# ---------------------------------------------------------------------------


class TestImporterBaseContentVectors:
    def test_base_class_instance_has_content_vectors(self):
        from vtsearch.datasets.importers.base import DatasetImporter

        class MinimalImporter(DatasetImporter):
            name = "minimal"
            display_name = "Minimal"
            description = "Minimal importer."
            fields = []

            def run(self, field_values, clips):
                pass

        imp = MinimalImporter()
        assert hasattr(imp, "content_vectors")
        assert imp.content_vectors == {}

    def test_content_vectors_defaults_to_empty_dict(self):
        from vtsearch.datasets.importers.base import DatasetImporter

        class Imp(DatasetImporter):
            name = "t"
            display_name = "T"
            description = "T"
            fields = []

            def run(self, field_values, clips):
                pass

        assert Imp().content_vectors == {}

    def test_content_vectors_are_independent_across_instances(self):
        from vtsearch.datasets.importers.base import DatasetImporter

        class Imp(DatasetImporter):
            name = "t"
            display_name = "T"
            description = "T"
            fields = []

            def run(self, field_values, clips):
                pass

        a = Imp()
        b = Imp()
        a.content_vectors["file.wav"] = [1, 2, 3]
        assert b.content_vectors == {}

    def test_subclass_can_populate_content_vectors_during_run(self):
        import numpy as np

        from vtsearch.datasets.importers.base import DatasetImporter

        class VectorImporter(DatasetImporter):
            name = "vec"
            display_name = "Vector"
            description = "Provides vectors."
            fields = []

            def run(self, field_values, clips):
                self.content_vectors["test.wav"] = np.array([1.0, 2.0, 3.0])

        imp = VectorImporter()
        imp.run({}, {})
        assert "test.wav" in imp.content_vectors
        assert list(imp.content_vectors["test.wav"]) == [1.0, 2.0, 3.0]


class TestImporterBaseIcon:
    def test_base_class_has_icon_attribute(self):
        from vtsearch.datasets.importers.base import DatasetImporter

        assert hasattr(DatasetImporter, "icon")
        assert DatasetImporter.icon == "🔌"

    def test_to_dict_includes_icon(self):
        from vtsearch.datasets.importers.base import DatasetImporter

        class DummyImporter(DatasetImporter):
            name = "dummy"
            display_name = "Dummy"
            description = "A dummy importer."
            icon = "🧪"
            fields = []

            def run(self, field_values, clips):
                pass

        d = DummyImporter().to_dict()
        assert "icon" in d
        assert d["icon"] == "🧪"

    def test_to_dict_uses_default_icon_when_not_overridden(self):
        from vtsearch.datasets.importers.base import DatasetImporter

        class MinimalImporter(DatasetImporter):
            name = "minimal"
            display_name = "Minimal"
            description = "No icon override."
            fields = []

            def run(self, field_values, clips):
                pass

        d = MinimalImporter().to_dict()
        assert d["icon"] == "🔌"


# ---------------------------------------------------------------------------
# HTTP Archive importer metadata
# ---------------------------------------------------------------------------


class TestHttpArchiveImporterMetadata:
    def _get_importer(self):
        from vtsearch.datasets.importers.http_zip import IMPORTER

        return IMPORTER

    def test_name_is_http_archive(self):
        assert self._get_importer().name == "http_archive"

    def test_display_name(self):
        assert self._get_importer().display_name == "Generate from HTTP Archive"

    def test_icon_is_globe(self):
        assert self._get_importer().icon == "🌐"

    def test_description_mentions_zip_tar_rar(self):
        desc = self._get_importer().description.lower()
        assert "zip" in desc
        assert "tar" in desc
        assert "rar" in desc

    def test_to_dict_includes_icon(self):
        d = self._get_importer().to_dict()
        assert d["icon"] == "🌐"
        assert d["name"] == "http_archive"
        assert d["display_name"] == "Generate from HTTP Archive"

    def test_fields_include_url_and_media_type(self):
        fields = {f.key: f for f in self._get_importer().fields}
        assert "url" in fields
        assert "media_type" in fields

    def test_url_field_type(self):
        fields = {f.key: f for f in self._get_importer().fields}
        assert fields["url"].field_type == "url"

    def test_media_type_options(self):
        fields = {f.key: f for f in self._get_importer().fields}
        opts = fields["media_type"].options
        assert "sounds" in opts
        assert "videos" in opts
        assert "images" in opts
        assert "paragraphs" in opts


# ---------------------------------------------------------------------------
# Folder importer metadata
# ---------------------------------------------------------------------------


class TestFolderImporterMetadata:
    def _get_importer(self):
        from vtsearch.datasets.importers.folder import IMPORTER

        return IMPORTER

    def test_name_is_folder(self):
        assert self._get_importer().name == "folder"

    def test_icon_is_folder_emoji(self):
        assert self._get_importer().icon == "📂"

    def test_description_says_media_files_from_a_folder(self):
        desc = self._get_importer().description.lower()
        assert "media files from a folder" in desc

    def test_description_does_not_list_specific_media_types(self):
        desc = self._get_importer().description
        assert "sounds/videos" not in desc
        assert "(sounds" not in desc

    def test_media_type_field_before_path_field(self):
        keys = [f.key for f in self._get_importer().fields]
        assert keys.index("media_type") < keys.index("path")

    def test_path_field_type_is_folder(self):
        fields = {f.key: f for f in self._get_importer().fields}
        assert fields["path"].field_type == "folder"

    def test_to_dict_includes_icon(self):
        d = self._get_importer().to_dict()
        assert d["icon"] == "📂"


# ---------------------------------------------------------------------------
# Folder importer not in builtin names
# ---------------------------------------------------------------------------


class TestBuiltinImporterNames:
    def test_folder_not_in_builtin_names(self):
        from vtsearch.routes.datasets import _BUILTIN_IMPORTER_NAMES

        assert "folder" not in _BUILTIN_IMPORTER_NAMES

    def test_pickle_still_in_builtin_names(self):
        from vtsearch.routes.datasets import _BUILTIN_IMPORTER_NAMES

        assert "pickle" in _BUILTIN_IMPORTER_NAMES


# ---------------------------------------------------------------------------
# _extract_archive helper
# ---------------------------------------------------------------------------


class TestExtractArchive:
    def test_extract_zip(self, tmp_path):
        from vtsearch.datasets.importers.http_zip import _extract_archive

        wav_data = _make_wav_bytes()
        zip_path = tmp_path / "test.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("sounds/tone.wav", wav_data)
        extract_dir = tmp_path / "out"
        extract_dir.mkdir()
        _extract_archive(zip_path, extract_dir)
        assert (extract_dir / "sounds" / "tone.wav").exists()

    def test_extract_tar_uncompressed(self, tmp_path):
        from vtsearch.datasets.importers.http_zip import _extract_archive

        wav_data = _make_wav_bytes()
        tar_path = tmp_path / "test.tar"
        with tarfile.open(tar_path, "w") as tf:
            info = tarfile.TarInfo(name="tone.wav")
            info.size = len(wav_data)
            tf.addfile(info, io.BytesIO(wav_data))
        extract_dir = tmp_path / "out"
        extract_dir.mkdir()
        _extract_archive(tar_path, extract_dir)
        assert (extract_dir / "tone.wav").exists()

    def test_extract_tar_gz(self, tmp_path):
        from vtsearch.datasets.importers.http_zip import _extract_archive

        wav_data = _make_wav_bytes()
        tar_path = tmp_path / "test.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tf:
            info = tarfile.TarInfo(name="sounds/tone.wav")
            info.size = len(wav_data)
            tf.addfile(info, io.BytesIO(wav_data))
        extract_dir = tmp_path / "out"
        extract_dir.mkdir()
        _extract_archive(tar_path, extract_dir)
        assert (extract_dir / "sounds" / "tone.wav").exists()

    def test_extract_tar_bz2(self, tmp_path):
        from vtsearch.datasets.importers.http_zip import _extract_archive

        wav_data = _make_wav_bytes()
        tar_path = tmp_path / "test.tar.bz2"
        with tarfile.open(tar_path, "w:bz2") as tf:
            info = tarfile.TarInfo(name="tone.wav")
            info.size = len(wav_data)
            tf.addfile(info, io.BytesIO(wav_data))
        extract_dir = tmp_path / "out"
        extract_dir.mkdir()
        _extract_archive(tar_path, extract_dir)
        assert (extract_dir / "tone.wav").exists()

    def test_unsupported_extension_raises_value_error(self, tmp_path):
        from vtsearch.datasets.importers.http_zip import _extract_archive

        # A file that is not a zip or tar and doesn't end in .rar
        bad_archive = tmp_path / "test.7z"
        bad_archive.write_bytes(b"not a valid archive format at all")
        extract_dir = tmp_path / "out"
        extract_dir.mkdir()
        with pytest.raises((ValueError, Exception)):
            _extract_archive(bad_archive, extract_dir)

    def test_rar_without_rarfile_raises_runtime_error(self, tmp_path):
        """Attempting RAR extraction without rarfile installed should fail gracefully."""
        import sys
        import unittest.mock as mock

        from vtsearch.datasets.importers.http_zip import _extract_archive

        rar_path = tmp_path / "test.rar"
        # Write RAR v4 magic bytes so it's identified as .rar by extension
        rar_path.write_bytes(b"Rar!\x1a\x07\x00" + b"\x00" * 20)
        extract_dir = tmp_path / "out"
        extract_dir.mkdir()

        with mock.patch.dict(sys.modules, {"rarfile": None}):
            with pytest.raises((RuntimeError, ImportError, Exception)):
                _extract_archive(rar_path, extract_dir)

    def test_zip_preserves_multiple_files(self, tmp_path):
        from vtsearch.datasets.importers.http_zip import _extract_archive

        zip_path = tmp_path / "multi.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            for i in range(3):
                zf.writestr(f"file{i}.wav", _make_wav_bytes())
        extract_dir = tmp_path / "out"
        extract_dir.mkdir()
        _extract_archive(zip_path, extract_dir)
        assert len(list(extract_dir.glob("*.wav"))) == 3


# ---------------------------------------------------------------------------
# load_dataset_from_folder – content_vectors support
# ---------------------------------------------------------------------------


class TestLoadDatasetContentVectors:
    """Verify that load_dataset_from_folder uses pre-computed content vectors."""

    def _write_wav(self, path):
        """Write a minimal WAV file to *path*."""
        path.write_bytes(_make_wav_bytes())

    def _make_fake_media_type(self, embed_return):
        """Return a mock media-type object for testing.

        ``embed_return`` is the value returned by ``embed_media()``.
        """
        import unittest.mock as mock

        mt = mock.MagicMock()
        mt.type_id = "audio"
        mt.file_extensions = ["*.wav"]
        mt.embed_media.return_value = embed_return
        mt.load_clip_data.return_value = {"duration": 1.0}
        return mt

    def test_uses_content_vector_when_provided(self, tmp_path):
        """A file whose name is in content_vectors should use that vector."""
        import unittest.mock as mock

        import numpy as np

        from vtsearch.datasets.loader import load_dataset_from_folder

        wav = tmp_path / "a.wav"
        self._write_wav(wav)

        pre_vector = np.array([10.0, 20.0, 30.0])
        mt = self._make_fake_media_type(embed_return=np.zeros(3))

        clips: dict = {}
        _noop = lambda *a: None
        with mock.patch("vtsearch.media.get_by_folder_name", return_value=mt):
            load_dataset_from_folder(tmp_path, "sounds", clips, content_vectors={"a.wav": pre_vector}, on_progress=_noop)

        assert len(clips) == 1
        np.testing.assert_array_equal(clips[1]["embedding"], pre_vector)
        mt.embed_media.assert_not_called()

    def test_embeds_normally_when_not_in_content_vectors(self, tmp_path):
        """A file NOT in content_vectors falls back to embed_media()."""
        import unittest.mock as mock

        import numpy as np

        from vtsearch.datasets.loader import load_dataset_from_folder

        wav = tmp_path / "b.wav"
        self._write_wav(wav)

        model_vector = np.array([1.0, 2.0, 3.0])
        mt = self._make_fake_media_type(embed_return=model_vector)

        clips: dict = {}
        _noop = lambda *a: None
        with mock.patch("vtsearch.media.get_by_folder_name", return_value=mt):
            load_dataset_from_folder(tmp_path, "sounds", clips, content_vectors={}, on_progress=_noop)

        assert len(clips) == 1
        np.testing.assert_array_equal(clips[1]["embedding"], model_vector)
        mt.embed_media.assert_called_once()

    def test_mixed_content_vectors_and_embedding(self, tmp_path):
        """Only files in content_vectors skip embed_media; others are embedded."""
        import unittest.mock as mock

        import numpy as np

        from vtsearch.datasets.loader import load_dataset_from_folder

        wav_a = tmp_path / "a.wav"
        wav_b = tmp_path / "b.wav"
        self._write_wav(wav_a)
        self._write_wav(wav_b)

        pre_vector = np.array([10.0, 20.0])
        model_vector = np.array([1.0, 2.0])
        mt = self._make_fake_media_type(embed_return=model_vector)

        clips: dict = {}
        _noop = lambda *a: None
        with mock.patch("vtsearch.media.get_by_folder_name", return_value=mt):
            load_dataset_from_folder(tmp_path, "sounds", clips, content_vectors={"a.wav": pre_vector}, on_progress=_noop)

        assert len(clips) == 2
        # One clip should have the pre-computed vector, the other the model vector
        embeddings = {c["filename"]: c["embedding"] for c in clips.values()}
        np.testing.assert_array_equal(embeddings["a.wav"], pre_vector)
        np.testing.assert_array_equal(embeddings["b.wav"], model_vector)

    def test_no_content_vectors_param_embeds_all(self, tmp_path):
        """When content_vectors is None (default), all files are embedded."""
        import unittest.mock as mock

        import numpy as np

        from vtsearch.datasets.loader import load_dataset_from_folder

        wav = tmp_path / "c.wav"
        self._write_wav(wav)

        model_vector = np.array([5.0, 6.0])
        mt = self._make_fake_media_type(embed_return=model_vector)

        clips: dict = {}
        _noop = lambda *a: None
        with mock.patch("vtsearch.media.get_by_folder_name", return_value=mt):
            load_dataset_from_folder(tmp_path, "sounds", clips, on_progress=_noop)

        assert len(clips) == 1
        np.testing.assert_array_equal(clips[1]["embedding"], model_vector)
        mt.embed_media.assert_called_once()

    def test_content_vector_file_skips_none_embed_check(self, tmp_path):
        """A file with a content vector is included even if embed_media would return None."""
        import unittest.mock as mock

        import numpy as np

        from vtsearch.datasets.loader import load_dataset_from_folder

        wav = tmp_path / "d.wav"
        self._write_wav(wav)

        pre_vector = np.array([7.0, 8.0])
        # embed_media returns None, which would normally skip the file
        mt = self._make_fake_media_type(embed_return=None)

        clips: dict = {}
        _noop = lambda *a: None
        with mock.patch("vtsearch.media.get_by_folder_name", return_value=mt):
            load_dataset_from_folder(tmp_path, "sounds", clips, content_vectors={"d.wav": pre_vector}, on_progress=_noop)

        assert len(clips) == 1
        np.testing.assert_array_equal(clips[1]["embedding"], pre_vector)
