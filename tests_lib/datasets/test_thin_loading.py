"""Tests for thin (lazy-loading) media reference support.

Verifies that datasets can be loaded in thin mode (storing media_path
instead of media_bytes) and that lazy loading correctly resolves media
content when needed.
"""

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from vtscore.datasets.loader import (
    load_dataset_from_folder,
    load_dataset_from_pickle,
)
from vtscore.embedding.media_vectors import media_embedding
from vtscore.utils.hashing import content_md5, file_md5


from tests_lib.helpers import make_wav_bytes as _make_wav_bytes, make_wav_file as _make_wav_file  # noqa: F401


class TestFileMD5:
    def test_matches_regular_md5(self, tmp_path):
        content = b"hello world test data"
        p = tmp_path / "test.bin"
        p.write_bytes(content)
        assert file_md5(p) == content_md5(content)

    def test_large_file(self, tmp_path):
        # File larger than the 8192 chunk size
        content = b"x" * 20000
        p = tmp_path / "large.bin"
        p.write_bytes(content)
        assert file_md5(p) == content_md5(content)


class TestThinLoadFromFolder:
    """Test load_dataset_from_folder with thin=True."""

    def test_thin_clips_have_media_path(self, tmp_path):
        _make_wav_file(tmp_path, "test1.wav")
        _make_wav_file(tmp_path, "test2.wav")
        medias: dict[int, dict[str, Any]] = {}
        load_dataset_from_folder(tmp_path, "audio", medias, thin=True)
        assert len(medias) == 2
        for media in medias.values():
            assert media["media_path"] is not None
            assert Path(media["media_path"]).exists()

    def test_thin_clips_have_no_bytes(self, tmp_path):
        _make_wav_file(tmp_path, "test.wav")
        medias: dict[int, dict[str, Any]] = {}
        load_dataset_from_folder(tmp_path, "audio", medias, thin=True)
        media = medias[1]
        assert media["media_bytes"] is None
        assert media["media_string"] is None

    def test_thin_clips_leave_embedding_none(self, tmp_path):
        """The loader doesn't embed; framework ``embed_missing`` fills these in."""
        _make_wav_file(tmp_path, "test.wav")
        medias: dict[int, dict[str, Any]] = {}
        load_dataset_from_folder(tmp_path, "audio", medias, thin=True)
        media = medias[1]
        assert media_embedding(media) is None

    def test_thin_clips_have_correct_file_size(self, tmp_path):
        wav_path = _make_wav_file(tmp_path, "test.wav")
        expected_size = wav_path.stat().st_size
        medias: dict[int, dict[str, Any]] = {}
        load_dataset_from_folder(tmp_path, "audio", medias, thin=True)
        assert medias[1]["file_size"] == expected_size

    def test_thin_clips_have_correct_md5(self, tmp_path):
        wav_path = _make_wav_file(tmp_path, "test.wav")
        expected_md5 = content_md5(wav_path.read_bytes())
        medias: dict[int, dict[str, Any]] = {}
        load_dataset_from_folder(tmp_path, "audio", medias, thin=True)
        assert medias[1]["md5"] == expected_md5

    def test_thin_has_duration_and_waveform(self, tmp_path):
        """Thin mode still builds the ingest-time display fields.

        Audio has a browsable thumbnail (its waveform PNG), so a thin load runs
        ``load_thin_media_data`` and picks up the decoded duration alongside it.
        Only the *payload* is withheld — see ``test_thin_clips_have_no_bytes``.
        """
        _make_wav_file(tmp_path, "test.wav")
        medias: dict[int, dict[str, Any]] = {}
        load_dataset_from_folder(tmp_path, "audio", medias, thin=True)
        assert medias[1]["duration"] > 0
        assert medias[1]["thumbnail_bytes"]

    def test_full_mode_has_bytes(self, tmp_path):
        """Full mode (thin=False) should still load bytes as before."""
        _make_wav_file(tmp_path, "test.wav")
        medias: dict[int, dict[str, Any]] = {}
        load_dataset_from_folder(tmp_path, "audio", medias, thin=False)
        assert medias[1]["media_bytes"] is not None
        assert isinstance(medias[1]["media_bytes"], bytes)

    def test_full_mode_also_has_media_path(self, tmp_path):
        """Full mode should also store media_path for potential future use."""
        _make_wav_file(tmp_path, "test.wav")
        medias: dict[int, dict[str, Any]] = {}
        load_dataset_from_folder(tmp_path, "audio", medias, thin=False)
        assert medias[1]["media_path"] is not None


class TestThinLoadFromPickle:
    """Test load_dataset_from_pickle with thin=True."""

    def _make_pickle(self, tmp_path, inline_bytes=True, audio_dir=None):
        """Create a test pickle with one audio media."""
        wav_bytes = _make_wav_bytes()
        media_data: dict[str, Any] = {
            "id": 1,
            "media_type": "audio",
            "duration": 0.1,
            "file_size": len(wav_bytes),
            "md5": content_md5(wav_bytes),
            "embedding": np.zeros(512).tolist(),
            "embedder": "clap",
            "filename": "test.wav",
            "category": "test",
        }
        if inline_bytes:
            media_data["media_bytes"] = wav_bytes

        pkl_data: dict[str, Any] = {"medias": {1: media_data}}
        if audio_dir:
            pkl_data["audio_dir"] = str(audio_dir)
            # Write the actual file
            audio_dir.mkdir(exist_ok=True)
            (audio_dir / "test.wav").write_bytes(wav_bytes)

        pkl_path = tmp_path / "test.pkl"
        from vtscore.datasets.container import write_container

        write_container(pkl_path, pickle.dumps(pkl_data), {"format_version": 1})
        return pkl_path

    def test_thin_pickle_drops_bytes_it_can_re_read(self, tmp_path):
        """Bytes also on disk are dropped - that is thin's whole point."""
        audio_dir = tmp_path / "audio"
        pkl_path = self._make_pickle(tmp_path, inline_bytes=True, audio_dir=audio_dir)
        medias: dict[int, dict[str, Any]] = {}
        load_dataset_from_pickle(pkl_path, medias, thin=True)
        assert len(medias) == 1
        assert medias[1]["media_bytes"] is None
        assert Path(medias[1]["media_path"]).exists()

    def test_thin_pickle_keeps_bytes_nothing_can_re_read(self, tmp_path):
        """A self-contained pickle keeps its payload even under thin (issue #3556).

        Thin means "hold a reference instead of the payload".  When the entry
        has no reference to hold - no file on disk, no archive member, no URL -
        dropping the bytes does not defer the read, it destroys the only copy,
        and the media becomes permanently unembeddable.  It is then silently
        skipped at scoring, which is how the CLI returned different hits *and*
        a different threshold than the GUI on the same dataset.
        """
        pkl_path = self._make_pickle(tmp_path, inline_bytes=True)
        medias: dict[int, dict[str, Any]] = {}
        load_dataset_from_pickle(pkl_path, medias, thin=True)
        assert len(medias) == 1
        assert medias[1]["media_bytes"] is not None

    def test_thin_pickle_kept_bytes_are_resolvable(self, tmp_path):
        """The kept payload is reachable through the normal byte accessor."""
        from vtscore.media import get as get_media_type

        pkl_path = self._make_pickle(tmp_path, inline_bytes=True)
        medias: dict[int, dict[str, Any]] = {}
        load_dataset_from_pickle(pkl_path, medias, thin=True)
        mt = get_media_type(medias[1]["media_type"])
        assert mt._resolve_media_bytes(medias[1]) is not None

    def test_thin_pickle_has_embedding(self, tmp_path):
        pkl_path = self._make_pickle(tmp_path, inline_bytes=True)
        medias: dict[int, dict[str, Any]] = {}
        load_dataset_from_pickle(pkl_path, medias, thin=True)
        assert isinstance(media_embedding(medias[1]), np.ndarray)

    def test_thin_pickle_resolves_media_path_from_audio_dir(self, tmp_path):
        audio_dir = tmp_path / "audio"
        pkl_path = self._make_pickle(tmp_path, inline_bytes=False, audio_dir=audio_dir)
        medias: dict[int, dict[str, Any]] = {}
        load_dataset_from_pickle(pkl_path, medias, thin=True)
        assert medias[1]["media_path"] is not None
        assert Path(medias[1]["media_path"]).exists()

    def test_thin_pickle_preserves_metadata(self, tmp_path):
        pkl_path = self._make_pickle(tmp_path, inline_bytes=True)
        medias: dict[int, dict[str, Any]] = {}
        load_dataset_from_pickle(pkl_path, medias, thin=True)
        assert medias[1]["media_type"] == "audio"
        assert medias[1]["filename"] == "test.wav"
        assert medias[1]["category"] == "test"

    def test_full_pickle_still_works(self, tmp_path):
        pkl_path = self._make_pickle(tmp_path, inline_bytes=True)
        medias: dict[int, dict[str, Any]] = {}
        load_dataset_from_pickle(pkl_path, medias, thin=False)
        assert medias[1]["media_bytes"] is not None


class TestFullModeMediaPathReference:
    """Full-mode pickle load must honor a stored ``media_path`` reference.

    A reference dataset (imported with ``reference_files`` / thin) is saved to
    the registry pickle with ``media_bytes=None`` and a ``media_path`` pointing
    at the original file.  Reopening from the dashboard loads that pickle in
    *full* mode, which historically ignored ``media_path`` and dropped every
    such media.  It must instead fall back to the reference and load it lazily,
    so the reference dataset survives the save → reopen round-trip.
    """

    def _make_reference_pickle(self, tmp_path: Path, media_path: str | None) -> Path:
        """Pickle one audio media with no inline bytes and a ``media_path``."""
        wav_bytes = _make_wav_bytes()
        media: dict[str, Any] = {
            "id": 1,
            "media_type": "audio",
            "duration": 0.1,
            "file_size": len(wav_bytes),
            "md5": content_md5(wav_bytes),
            "embedding": np.zeros(512).tolist(),
            "embedder": "clap",
            "filename": "test.wav",
            "category": "test",
            "media_bytes": None,
            "media_path": media_path,
        }
        pkl_path = tmp_path / "ref.pkl"
        from vtscore.datasets.container import write_container

        write_container(pkl_path, pickle.dumps({"medias": {1: media}}), {"format_version": 1})
        return pkl_path

    def test_full_mode_keeps_media_when_path_exists(self, tmp_path):
        src = _make_wav_file(tmp_path, "test.wav")
        pkl_path = self._make_reference_pickle(tmp_path, str(src))
        medias: dict[int, dict[str, Any]] = {}
        load_dataset_from_pickle(pkl_path, medias, thin=False)
        assert len(medias) == 1
        # Loaded lazily: no bytes in RAM, but the reference resolves on disk.
        assert medias[1]["media_bytes"] is None
        assert medias[1]["media_path"] == str(src)
        assert Path(medias[1]["media_path"]).exists()
        assert isinstance(media_embedding(medias[1]), np.ndarray)

    def test_full_mode_drops_media_when_path_missing(self, tmp_path, capsys):
        pkl_path = self._make_reference_pickle(tmp_path, str(tmp_path / "gone.wav"))
        medias: dict[int, dict[str, Any]] = {}
        load_dataset_from_pickle(pkl_path, medias, thin=False)
        assert medias == {}
        assert "1 media files missing" in capsys.readouterr().out

    def test_full_mode_drops_media_when_no_path(self, tmp_path):
        pkl_path = self._make_reference_pickle(tmp_path, None)
        medias: dict[int, dict[str, Any]] = {}
        load_dataset_from_pickle(pkl_path, medias, thin=False)
        assert medias == {}


class TestFullModeExternalByteSource:
    """Full-mode pickle load must keep media whose bytes live outside the pickle.

    An archive-member media (``local_archive_member``: audio/video tiles cut
    from tar shards we never extract) and a URL-backed media (an importer's
    thin mode) both carry no inline bytes and no local file - their bytes
    re-resolve on demand from the shard / the URL.  Full mode used to drop
    every such entry, so reopening the dataset from the dashboard registry
    (which loads the pickle in full mode) produced an *empty* dataset and a
    "Dataset is empty - nothing to project" failure on Browse.
    """

    def _write_pickle(self, tmp_path: Path, media: dict[str, Any]) -> Path:
        from vtscore.datasets.container import write_container

        pkl_path = tmp_path / "external.pkl"
        write_container(pkl_path, pickle.dumps({"medias": {1: media}}), {"format_version": 1})
        return pkl_path

    def _base_media(self) -> dict[str, Any]:
        return {
            "id": 1,
            "media_type": "audio",
            "duration": 0,
            "file_size": 4096,
            "md5": "deadbeef",
            "embeddings": {"clap": np.zeros(512, dtype=np.float32)},
            "embedder": "clap",
            "filename": "shard0/clip.m4a",
            "category": "custom",
            "media_bytes": None,
            "media_string": None,
            "media_path": None,
        }

    def test_full_mode_keeps_archive_member_media(self, tmp_path):
        media = self._base_media()
        media["origin"] = {
            "importer": "local_archive_member",
            "params": {
                "archive_path": str(tmp_path / "shard0.tar"),
                "member": "clip.m4a",
                "media_type": "audio",
                "clip_start": 0.0,
                "clip_end": 10.0,
            },
        }
        medias: dict[int, dict[str, Any]] = {}
        load_dataset_from_pickle(self._write_pickle(tmp_path, media), medias, thin=False)
        assert len(medias) == 1
        # Kept lazily: the member is streamed from its shard on demand.
        assert medias[1]["media_bytes"] is None
        assert medias[1]["origin"]["importer"] == "local_archive_member"
        assert isinstance(media_embedding(medias[1]), np.ndarray)

    def test_full_mode_keeps_url_backed_media(self, tmp_path):
        media = self._base_media()
        media["origin"] = {"importer": "url_backed", "params": {}}
        media["media_url"] = "https://example.invalid/clip.wav"
        medias: dict[int, dict[str, Any]] = {}
        load_dataset_from_pickle(self._write_pickle(tmp_path, media), medias, thin=False)
        assert len(medias) == 1
        assert medias[1]["media_bytes"] is None
        # The URL is the media's byte source, so it has to survive the reload.
        assert medias[1]["media_url"] == "https://example.invalid/clip.wav"

    def test_full_mode_still_drops_media_with_no_source(self, tmp_path):
        """A media with no bytes, no file, no shard and no URL stays dropped."""
        media = self._base_media()
        media["origin"] = {"importer": "server_folder", "params": {}}
        medias: dict[int, dict[str, Any]] = {}
        load_dataset_from_pickle(self._write_pickle(tmp_path, media), medias, thin=False)
        assert medias == {}

    def test_media_url_survives_export_round_trip(self, tmp_path):
        """``export_dataset_to_file`` must persist ``media_url``."""
        from vtscore.datasets.loader import export_dataset_to_file

        media = self._base_media()
        media["origin"] = {"importer": "url_backed", "params": {}}
        media["media_url"] = "https://example.invalid/clip.wav"
        pkl_path = tmp_path / "roundtrip.pkl"
        pkl_path.write_bytes(export_dataset_to_file({1: media}, embedder="clap", media_type="audio"))

        medias: dict[int, dict[str, Any]] = {}
        load_dataset_from_pickle(pkl_path, medias, thin=False)
        assert medias[1]["media_url"] == "https://example.invalid/clip.wav"


class TestPickleNullEmbedding:
    """Skip-on-None pickle entries (audit M8 / M12).

    ``np.array(None)`` returns a 0-d ``dtype=object`` array that survives
    every ``is None`` guard in the codebase, so the pickle loader must
    drop entries whose ``embedding`` field is missing or ``None`` before
    they enter the medias dict.  Mirrors the folder loader, which
    already returns ``None`` for failed embeds.
    """

    def _wav_pickle(
        self,
        tmp_path: Path,
        *,
        embedding: Any,
        embedding_key_present: bool = True,
        include_bytes: bool = True,
    ) -> Path:
        wav_bytes = _make_wav_bytes()
        media: dict[str, Any] = {
            "id": 1,
            "media_type": "audio",
            "duration": 0.1,
            "file_size": len(wav_bytes),
            "md5": content_md5(wav_bytes),
            "filename": "test.wav",
            "category": "test",
        }
        if embedding_key_present:
            media["embedding"] = embedding
        if include_bytes:
            media["media_bytes"] = wav_bytes
        pkl_path = tmp_path / "test.pkl"
        from vtscore.datasets.container import write_container

        write_container(pkl_path, pickle.dumps({"medias": {1: media}}), {"format_version": 1})
        return pkl_path

    def test_full_mode_skips_explicit_none_embedding(self, tmp_path, capsys):
        pkl_path = self._wav_pickle(tmp_path, embedding=None)
        medias: dict[int, dict[str, Any]] = {}
        load_dataset_from_pickle(pkl_path, medias, thin=False)
        assert medias == {}
        out = capsys.readouterr().out
        assert "1 media files missing" in out

    def test_full_mode_skips_missing_embedding_key(self, tmp_path, capsys):
        pkl_path = self._wav_pickle(tmp_path, embedding=None, embedding_key_present=False)
        medias: dict[int, dict[str, Any]] = {}
        load_dataset_from_pickle(pkl_path, medias, thin=False)
        assert medias == {}
        out = capsys.readouterr().out
        assert "1 media files missing" in out

    def test_thin_mode_skips_explicit_none_embedding(self, tmp_path, capsys):
        """Regression for M8: prior code only checked key absence, not None."""
        pkl_path = self._wav_pickle(tmp_path, embedding=None, include_bytes=False)
        medias: dict[int, dict[str, Any]] = {}
        load_dataset_from_pickle(pkl_path, medias, thin=True)
        assert medias == {}
        out = capsys.readouterr().out
        assert "1 media files missing" in out

    def test_thin_mode_skips_missing_embedding_key(self, tmp_path, capsys):
        pkl_path = self._wav_pickle(tmp_path, embedding=None, embedding_key_present=False, include_bytes=False)
        medias: dict[int, dict[str, Any]] = {}
        load_dataset_from_pickle(pkl_path, medias, thin=True)
        assert medias == {}
        out = capsys.readouterr().out
        assert "1 media files missing" in out

    def test_full_mode_mixed_keeps_good_drops_null(self, tmp_path):
        wav_bytes = _make_wav_bytes()
        good = {
            "id": 1,
            "media_type": "audio",
            "duration": 0.1,
            "file_size": len(wav_bytes),
            "md5": content_md5(wav_bytes),
            "embedding": np.zeros(512).tolist(),
            "embedder": "clap",
            "filename": "good.wav",
            "category": "test",
            "media_bytes": wav_bytes,
        }
        bad = {
            "id": 2,
            "media_type": "audio",
            "duration": 0.1,
            "file_size": len(wav_bytes),
            "md5": content_md5(wav_bytes + b"x"),
            "embedding": None,
            "filename": "bad.wav",
            "category": "test",
            "media_bytes": wav_bytes,
        }
        pkl_path = tmp_path / "mixed.pkl"
        from vtscore.datasets.container import write_container

        write_container(pkl_path, pickle.dumps({"medias": {1: good, 2: bad}}), {"format_version": 1})

        medias: dict[int, dict[str, Any]] = {}
        load_dataset_from_pickle(pkl_path, medias, thin=False)
        assert list(medias.keys()) == [1]
        # No poisoned 0-d object array snuck through.
        assert media_embedding(medias[1]).ndim >= 1
        assert media_embedding(medias[1]).dtype != object

    def test_chunked_loader_skips_null_embedding(self, tmp_path):
        from vtscore.datasets.loader import load_dataset_from_pickle_chunked

        wav_bytes = _make_wav_bytes()
        pkl_data = {
            "medias": {
                1: {
                    "id": 1,
                    "media_type": "audio",
                    "duration": 0.1,
                    "file_size": len(wav_bytes),
                    "md5": content_md5(wav_bytes),
                    "embedding": np.zeros(512).tolist(),
                    "embedder": "clap",
                    "filename": "a.wav",
                    "category": "test",
                    "media_bytes": wav_bytes,
                },
                2: {
                    "id": 2,
                    "media_type": "audio",
                    "duration": 0.1,
                    "file_size": len(wav_bytes),
                    "md5": content_md5(wav_bytes + b"x"),
                    "embedding": None,
                    "filename": "b.wav",
                    "category": "test",
                    "media_bytes": wav_bytes,
                },
            }
        }
        pkl_path = tmp_path / "chunked.pkl"
        from vtscore.datasets.container import write_container

        write_container(pkl_path, pickle.dumps(pkl_data), {"format_version": 1})

        chunks = list(load_dataset_from_pickle_chunked(pkl_path, chunk_size=10))
        loaded = {cid: m for chunk in chunks for cid, m in chunk.items()}
        assert len(loaded) == 1
        only = next(iter(loaded.values()))
        assert only["filename"] == "a.wav"


class TestPickleMD5Preservation:
    """Test that load_dataset_from_pickle uses pre-existing MD5 from pickle data."""

    def test_full_mode_uses_md5_from_pickle_when_present(self, tmp_path):
        """Full mode should use the MD5 stored in the pickle instead of recalculating."""
        wav_bytes = _make_wav_bytes()
        pre_md5 = "a" * 32  # A fake MD5 that differs from the real hash
        pkl_data = {
            "medias": {
                1: {
                    "id": 1,
                    "media_type": "audio",
                    "duration": 0.1,
                    "file_size": len(wav_bytes),
                    "md5": pre_md5,
                    "embedding": np.zeros(512).tolist(),
                    "embedder": "clap",
                    "filename": "test.wav",
                    "category": "test",
                    "media_bytes": wav_bytes,
                }
            }
        }
        pkl_path = tmp_path / "test.pkl"
        from vtscore.datasets.container import write_container

        write_container(pkl_path, pickle.dumps(pkl_data), {"format_version": 1})

        medias: dict[int, dict[str, Any]] = {}
        load_dataset_from_pickle(pkl_path, medias, thin=False)
        assert medias[1]["md5"] == pre_md5

    def test_full_mode_computes_md5_when_missing_from_pickle(self, tmp_path):
        """Full mode should compute the MD5 if the pickle doesn't have one."""
        wav_bytes = _make_wav_bytes()
        pkl_data = {
            "medias": {
                1: {
                    "id": 1,
                    "media_type": "audio",
                    "duration": 0.1,
                    "file_size": len(wav_bytes),
                    # no "md5" key
                    "embedding": np.zeros(512).tolist(),
                    "embedder": "clap",
                    "filename": "test.wav",
                    "category": "test",
                    "media_bytes": wav_bytes,
                }
            }
        }
        pkl_path = tmp_path / "test.pkl"
        from vtscore.datasets.container import write_container

        write_container(pkl_path, pickle.dumps(pkl_data), {"format_version": 1})

        medias: dict[int, dict[str, Any]] = {}
        load_dataset_from_pickle(pkl_path, medias, thin=False)
        assert medias[1]["md5"] == content_md5(wav_bytes)

    def test_thin_mode_uses_md5_from_pickle(self, tmp_path):
        """Thin mode should also preserve the MD5 from the pickle."""
        wav_bytes = _make_wav_bytes()
        pre_md5 = "b" * 32
        pkl_data = {
            "medias": {
                1: {
                    "id": 1,
                    "media_type": "audio",
                    "duration": 0.1,
                    "file_size": len(wav_bytes),
                    "md5": pre_md5,
                    "embedding": np.zeros(512).tolist(),
                    "embedder": "clap",
                    "filename": "test.wav",
                    "category": "test",
                    "media_bytes": wav_bytes,
                }
            }
        }
        pkl_path = tmp_path / "test.pkl"
        from vtscore.datasets.container import write_container

        write_container(pkl_path, pickle.dumps(pkl_data), {"format_version": 1})

        medias: dict[int, dict[str, Any]] = {}
        load_dataset_from_pickle(pkl_path, medias, thin=True)
        assert medias[1]["md5"] == pre_md5


class TestThinImporters:
    """Test that importers pass thin parameter through correctly."""

    def test_folder_importer_thin(self, tmp_path):
        _make_wav_file(tmp_path, "test.wav")
        from vtscore.datasets.importers.server_folder import ServerFolderDatasetImporter

        importer = ServerFolderDatasetImporter()
        medias: dict[int, dict[str, Any]] = {}
        importer.run({"path": str(tmp_path), "media_type": "audio"}, medias, thin=True)
        assert len(medias) > 0
        assert medias[1]["media_bytes"] is None
        assert medias[1]["media_path"] is not None

    def test_folder_importer_run_cli_thin(self, tmp_path):
        _make_wav_file(tmp_path, "test.wav")
        from vtscore.datasets.importers.server_folder import ServerFolderDatasetImporter

        importer = ServerFolderDatasetImporter()
        medias: dict[int, dict[str, Any]] = {}
        importer.run_cli({"path": str(tmp_path), "media_type": "audio"}, medias, thin=True)
        assert len(medias) > 0
        assert medias[1]["media_bytes"] is None

    def test_pickle_importer_thin(self, tmp_path):
        # Create a pickle first
        wav_bytes = _make_wav_bytes()
        pkl_data = {
            "medias": {
                1: {
                    "id": 1,
                    "media_type": "audio",
                    "duration": 0.1,
                    "file_size": len(wav_bytes),
                    "md5": content_md5(wav_bytes),
                    "embedding": np.zeros(512).tolist(),
                    "embedder": "clap",
                    "filename": "test.wav",
                    "category": "test",
                    "media_bytes": wav_bytes,
                }
            }
        }
        pkl_path = tmp_path / "test.pkl"
        from vtscore.datasets.container import write_container

        write_container(pkl_path, pickle.dumps(pkl_data), {"format_version": 1})

        from vtscore.datasets.importers.pickle import PickleDatasetImporter

        importer = PickleDatasetImporter()
        medias: dict[int, dict[str, Any]] = {}
        importer.run_cli({"file": str(pkl_path)}, medias, thin=True)
        assert len(medias) == 1
        # Inline-only source: nothing outside the pickle can reproduce these
        # bytes, so thin keeps them rather than stranding the media without a
        # payload (issue #3556).
        assert medias[1]["media_bytes"] is not None


class TestLazyLoadingMediaType:
    """Test that MediaType._resolve_media_bytes/string lazy-loads from media_path."""

    def test_resolve_bytes_from_preloaded(self):
        from vtscore.media.audio.media_type import AudioMediaType

        mt = AudioMediaType()
        media = {"media_bytes": b"hello", "media_path": None}
        assert mt._resolve_media_bytes(media) == b"hello"

    def test_resolve_bytes_from_media_path(self, tmp_path):
        from vtscore.media.audio.media_type import AudioMediaType

        content = b"lazy loaded content"
        p = tmp_path / "test.wav"
        p.write_bytes(content)

        mt = AudioMediaType()
        media = {"media_bytes": None, "media_path": str(p)}
        assert mt._resolve_media_bytes(media) == content

    def test_resolve_bytes_missing_file(self):
        from vtscore.media.audio.media_type import AudioMediaType

        mt = AudioMediaType()
        media = {"media_bytes": None, "media_path": "/nonexistent/file.wav"}
        assert mt._resolve_media_bytes(media) is None

    def test_resolve_bytes_no_path(self):
        from vtscore.media.audio.media_type import AudioMediaType

        mt = AudioMediaType()
        media = {"media_bytes": None, "media_path": None}
        assert mt._resolve_media_bytes(media) is None

    def test_resolve_string_from_preloaded(self):
        from vtscore.media.text.media_type import TextMediaType

        mt = TextMediaType()
        media = {"media_string": "hello world", "media_path": None}
        assert mt._resolve_media_string(media) == "hello world"

    def test_resolve_string_from_media_path(self, tmp_path):
        from vtscore.media.text.media_type import TextMediaType

        content = "lazy loaded text content"
        p = tmp_path / "test.txt"
        p.write_text(content, encoding="utf-8")

        mt = TextMediaType()
        media = {"media_string": None, "media_path": str(p)}
        assert mt._resolve_media_string(media) == content


class TestClipResponseLazyLoading:
    """Test that media_response works with lazy-loaded media."""

    def test_audio_media_response_lazy(self, tmp_path):
        from vtscore.media.audio.media_type import AudioMediaType

        wav_bytes = _make_wav_bytes()
        p = tmp_path / "test.wav"
        p.write_bytes(wav_bytes)

        mt = AudioMediaType()
        media = {"id": 1, "media_bytes": None, "media_path": str(p), "filename": "test.wav"}
        resp = mt.media_response(media)
        assert resp.data == wav_bytes
        assert resp.mimetype == "audio/wav"

    def test_image_media_response_lazy(self, tmp_path):
        from vtscore.media.image.media_type import ImageMediaType

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
        media = {"id": 1, "media_bytes": None, "media_path": str(p), "filename": "test.png"}
        resp = mt.media_response(media)
        assert resp.data == png_bytes
        assert resp.mimetype == "image/png"

    def test_text_media_response_lazy(self, tmp_path):
        from vtscore.media.text.media_type import TextMediaType

        content = "lazy loaded paragraph"
        p = tmp_path / "test.txt"
        p.write_text(content, encoding="utf-8")

        mt = TextMediaType()
        media = {
            "id": 1,
            "media_string": None,
            "media_path": str(p),
            "word_count": 0,
            "character_count": 0,
        }
        resp = mt.media_response(media)
        assert isinstance(resp.data, dict)
        assert resp.data["content"] == content
        assert resp.data["word_count"] == 3  # "lazy loaded paragraph"
        assert resp.data["character_count"] == len(content)

    def test_audio_media_response_no_data(self):
        """media_response returns empty bytes when no data is available."""
        from vtscore.media.audio.media_type import AudioMediaType

        mt = AudioMediaType()
        media = {"id": 1, "media_bytes": None, "media_path": None, "filename": "test.wav"}
        resp = mt.media_response(media)
        assert resp.data == b""


class TestFullModeTextCompanionEncoding:
    """Regression for #2961: a non-UTF-8 ``text_dir`` companion must not abort the load.

    ``_load_pickle_media_payload`` used to open ``.txt``/``.md`` companion
    files with strict ``encoding="utf-8"``, so one latin-1/cp1252 file in a
    scraped text corpus raised ``UnicodeDecodeError`` and failed the entire
    load. It now falls back to latin-1, mirroring the CSV label importer.
    """

    def _write_pickle(self, tmp_path: Path, text_dir: Path, filename: str) -> Path:
        media: dict[str, Any] = {
            "id": 1,
            "media_type": "text",
            "embedding": np.zeros(512).tolist(),
            "embedder": "e5",
            "filename": filename,
            "category": "test",
            "media_bytes": None,
            "media_string": None,
            "media_path": None,
        }
        pkl_path = tmp_path / "text.pkl"
        from vtscore.datasets.container import write_container

        write_container(
            pkl_path,
            pickle.dumps({"medias": {1: media}, "text_dir": str(text_dir)}),
            {"format_version": 1},
        )
        return pkl_path

    def test_full_mode_falls_back_to_latin1_on_bad_utf8(self, tmp_path, caplog):
        text_dir = tmp_path / "texts"
        text_dir.mkdir()
        # "café" encoded as latin-1/cp1252, not valid UTF-8.
        raw = "café scraped corpus entry".encode("latin-1")
        (text_dir / "bad.txt").write_bytes(raw)

        pkl_path = self._write_pickle(tmp_path, text_dir, "bad.txt")
        medias: dict[int, dict[str, Any]] = {}
        with caplog.at_level("WARNING"):
            load_dataset_from_pickle(pkl_path, medias, thin=False)

        assert len(medias) == 1
        assert medias[1]["media_string"] == raw.decode("latin-1")
        assert "falling back to latin-1" in caplog.text

    def test_full_mode_still_reads_valid_utf8(self, tmp_path):
        text_dir = tmp_path / "texts"
        text_dir.mkdir()
        content = "plain ascii/utf-8 entry"
        (text_dir / "good.txt").write_text(content, encoding="utf-8")

        pkl_path = self._write_pickle(tmp_path, text_dir, "good.txt")
        medias: dict[int, dict[str, Any]] = {}
        load_dataset_from_pickle(pkl_path, medias, thin=False)

        assert len(medias) == 1
        assert medias[1]["media_string"] == content
