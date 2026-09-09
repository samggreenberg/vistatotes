"""Tests for chunked (piecewise) dataset loading.

Verifies that datasets can be loaded in chunks via the new
``load_dataset_from_folder_chunked``, ``load_dataset_from_pickle_chunked``
functions, the ``DatasetImporter.run_chunked`` / ``run_chunked_cli``
interface, and the CLI ``_merge_detector_results`` helper.
"""

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from vtscore.datasets.loader import (
    load_dataset_from_folder,
    load_dataset_from_folder_chunked,
    load_dataset_from_pickle_chunked,
)
from vtscore.embedding.media_vectors import media_embedding


from tests_lib.helpers import make_wav_bytes as _make_wav_bytes, make_wav_file as _make_wav_file
from vtscore.utils.hashing import content_md5


def _make_pickle_with_base_freq(tmp_path: Path, num_clips: int, base_freq: float = 440.0) -> Path:
    """Create a test pickle with distinct WAV bytes per media (using base_freq)."""
    medias_data: dict[int, dict[str, Any]] = {}
    for i in range(1, num_clips + 1):
        wav_bytes = _make_wav_bytes(frequency=base_freq + i * 10)
        media: dict[str, Any] = {
            "id": i,
            "media_type": "audio",
            "duration": 0.1,
            "file_size": len(wav_bytes),
            "md5": content_md5(wav_bytes),
            "embedder": "clap",
            "embedding": np.random.RandomState(42).randn(512).tolist(),
            "filename": f"clip_{i}.wav",
            "category": f"cat_{i % 3}",
            "media_bytes": wav_bytes,
        }
        medias_data[i] = media

    pkl_path = tmp_path / "test_chunked.pkl"
    from vtscore.datasets.container import write_container

    write_container(pkl_path, pickle.dumps({"medias": medias_data}), {"format_version": 1})
    return pkl_path


def _make_pickle(tmp_path: Path, num_clips: int, inline_bytes: bool = True) -> Path:
    """Create a test pickle with *num_clips* audio medias."""
    medias_data: dict[int, dict[str, Any]] = {}
    for i in range(1, num_clips + 1):
        wav_bytes = _make_wav_bytes(frequency=440.0 + i)
        media: dict[str, Any] = {
            "id": i,
            "media_type": "audio",
            "duration": 0.1,
            "file_size": len(wav_bytes),
            "md5": content_md5(wav_bytes),
            "embedder": "clap",
            "embedding": np.random.RandomState(42).randn(512).tolist(),
            "filename": f"clip_{i}.wav",
            "category": f"cat_{i % 3}",
        }
        if inline_bytes:
            media["media_bytes"] = wav_bytes
        medias_data[i] = media

    pkl_path = tmp_path / "test_chunked.pkl"
    from vtscore.datasets.container import write_container

    write_container(pkl_path, pickle.dumps({"medias": medias_data}), {"format_version": 1})
    return pkl_path


# ======================================================================
# load_dataset_from_folder_chunked
# ======================================================================


class TestFolderChunked:
    """Test load_dataset_from_folder_chunked."""

    def test_single_chunk_when_fewer_than_chunk_size(self, tmp_path):
        """When total files < chunk_size, yields exactly one chunk."""
        _make_wav_file(tmp_path, "a.wav")
        _make_wav_file(tmp_path, "b.wav")
        chunks = list(load_dataset_from_folder_chunked(tmp_path, "audio", chunk_size=10, thin=True))
        assert len(chunks) == 1
        assert len(chunks[0]) == 2

    def test_multiple_chunks(self, tmp_path):
        """Files are split across multiple chunks of the correct size."""
        for i in range(5):
            _make_wav_file(tmp_path, f"file_{i}.wav", frequency=440.0 + i * 10)
        chunks = list(load_dataset_from_folder_chunked(tmp_path, "audio", chunk_size=2, thin=True))
        assert len(chunks) == 3  # 2, 2, 1
        assert len(chunks[0]) == 2
        assert len(chunks[1]) == 2
        assert len(chunks[2]) == 1

    def test_chunk_ids_start_at_one(self, tmp_path):
        """Each chunk's media IDs start at 1 (not continuing from prior chunk)."""
        for i in range(4):
            _make_wav_file(tmp_path, f"file_{i}.wav", frequency=440.0 + i * 10)
        chunks = list(load_dataset_from_folder_chunked(tmp_path, "audio", chunk_size=2, thin=True))
        for chunk in chunks:
            assert 1 in chunk

    def test_thin_mode_no_bytes(self, tmp_path):
        """Thin mode: media_bytes is None, media_path is set."""
        _make_wav_file(tmp_path, "test.wav")
        chunks = list(load_dataset_from_folder_chunked(tmp_path, "audio", chunk_size=10, thin=True))
        media = chunks[0][1]
        assert media["media_bytes"] is None
        assert media["media_path"] is not None
        assert Path(media["media_path"]).exists()

    def test_full_mode_has_bytes(self, tmp_path):
        """Full mode: media_bytes is populated."""
        _make_wav_file(tmp_path, "test.wav")
        chunks = list(load_dataset_from_folder_chunked(tmp_path, "audio", chunk_size=10, thin=False))
        media = chunks[0][1]
        assert media["media_bytes"] is not None

    def test_embedding_is_none_until_framework_stage_runs(self, tmp_path):
        """The folder loader does not embed; items leave with embedding=None
        for the framework ``embed_missing`` stage to fill in.
        """
        _make_wav_file(tmp_path, "test.wav")
        chunks = list(load_dataset_from_folder_chunked(tmp_path, "audio", chunk_size=10, thin=True))
        media = chunks[0][1]
        assert media_embedding(media) is None

    def test_all_files_covered(self, tmp_path):
        """The total number of medias across all chunks equals total files."""
        for i in range(7):
            _make_wav_file(tmp_path, f"f_{i}.wav", frequency=440.0 + i * 10)
        chunks = list(load_dataset_from_folder_chunked(tmp_path, "audio", chunk_size=3, thin=True))
        total_clips = sum(len(c) for c in chunks)
        assert total_clips == 7

    def test_matches_monolithic_load(self, tmp_path):
        """Chunked loading produces the same filenames as monolithic loading."""
        for i in range(5):
            _make_wav_file(tmp_path, f"f_{i}.wav", frequency=440.0 + i * 10)

        # Monolithic
        mono_clips: dict[int, dict[str, Any]] = {}
        load_dataset_from_folder(tmp_path, "audio", mono_clips, thin=True)
        mono_filenames = {c["filename"] for c in mono_clips.values()}

        # Chunked
        chunked_filenames: set[str] = set()
        for chunk in load_dataset_from_folder_chunked(tmp_path, "audio", chunk_size=2, thin=True):
            for media in chunk.values():
                chunked_filenames.add(media["filename"])

        assert mono_filenames == chunked_filenames

    def test_invalid_media_type_raises(self, tmp_path):
        _make_wav_file(tmp_path, "test.wav")
        try:
            list(load_dataset_from_folder_chunked(tmp_path, "bogus", chunk_size=10, thin=True))
            assert False, "Expected ValueError"
        except ValueError as e:
            assert "Invalid media type" in str(e)

    def test_empty_folder_raises(self, tmp_path):
        try:
            list(load_dataset_from_folder_chunked(tmp_path, "audio", chunk_size=10, thin=True))
            assert False, "Expected ValueError"
        except ValueError as e:
            assert "No audio files found" in str(e)


# ======================================================================
# load_dataset_from_pickle_chunked
# ======================================================================


class TestPickleChunked:
    """Test load_dataset_from_pickle_chunked."""

    def test_single_chunk(self, tmp_path):
        pkl_path = _make_pickle(tmp_path, 3)
        chunks = list(load_dataset_from_pickle_chunked(pkl_path, chunk_size=10, thin=True))
        assert len(chunks) == 1
        assert len(chunks[0]) == 3

    def test_multiple_chunks(self, tmp_path):
        pkl_path = _make_pickle(tmp_path, 5)
        chunks = list(load_dataset_from_pickle_chunked(pkl_path, chunk_size=2, thin=True))
        assert len(chunks) == 3  # 2, 2, 1
        assert len(chunks[0]) == 2
        assert len(chunks[1]) == 2
        assert len(chunks[2]) == 1

    def test_chunk_ids_start_at_one(self, tmp_path):
        pkl_path = _make_pickle(tmp_path, 4)
        chunks = list(load_dataset_from_pickle_chunked(pkl_path, chunk_size=2, thin=True))
        for chunk in chunks:
            assert 1 in chunk

    def test_thin_mode_keeps_bytes_nothing_can_re_read(self, tmp_path):
        """These entries are inline-only, so thin has no reference to hold.

        Dropping the payload would not defer a read, it would destroy the only
        copy and leave the media unembeddable (issue #3556); thin only sheds
        bytes that are also reachable on disk, in an archive member, or via a
        URL.
        """
        pkl_path = _make_pickle(tmp_path, 2, inline_bytes=True)
        chunks = list(load_dataset_from_pickle_chunked(pkl_path, chunk_size=10, thin=True))
        for media in chunks[0].values():
            assert media["media_bytes"] is not None

    def test_full_mode_keeps_bytes(self, tmp_path):
        pkl_path = _make_pickle(tmp_path, 2, inline_bytes=True)
        chunks = list(load_dataset_from_pickle_chunked(pkl_path, chunk_size=10, thin=False))
        for media in chunks[0].values():
            assert media["media_bytes"] is not None

    def test_embeddings_are_numpy(self, tmp_path):
        pkl_path = _make_pickle(tmp_path, 2)
        chunks = list(load_dataset_from_pickle_chunked(pkl_path, chunk_size=10, thin=True))
        for media in chunks[0].values():
            assert isinstance(media_embedding(media), np.ndarray)

    def test_all_clips_covered(self, tmp_path):
        pkl_path = _make_pickle(tmp_path, 7)
        chunks = list(load_dataset_from_pickle_chunked(pkl_path, chunk_size=3, thin=True))
        total = sum(len(c) for c in chunks)
        assert total == 7

    def test_metadata_preserved(self, tmp_path):
        pkl_path = _make_pickle(tmp_path, 1)
        chunks = list(load_dataset_from_pickle_chunked(pkl_path, chunk_size=10, thin=True))
        media = chunks[0][1]
        assert media["media_type"] == "audio"
        assert media["filename"] == "clip_1.wav"
        assert media["category"] == "cat_1"

    def test_image_extra_fields_preserved(self, tmp_path):
        """Image-specific fields (width, height) are preserved via the registry."""
        medias_data = {
            1: {
                "media_type": "image",
                "embedder": "siglip",
                "embedding": np.random.RandomState(42).randn(512).tolist(),
                "media_bytes": b"\x89PNG fake",
                "filename": "photo.png",
                "width": 640,
                "height": 480,
                "category": "test",
            }
        }
        pkl_path = tmp_path / "img.pkl"
        from vtscore.datasets.container import write_container

        write_container(pkl_path, pickle.dumps({"medias": medias_data}), {"format_version": 1})

        # Full mode
        chunks = list(load_dataset_from_pickle_chunked(pkl_path, chunk_size=10, thin=False))
        media = chunks[0][1]
        assert media["media_type"] == "image"
        assert media["width"] == 640
        assert media["height"] == 480

        # Thin mode
        chunks = list(load_dataset_from_pickle_chunked(pkl_path, chunk_size=10, thin=True))
        media = chunks[0][1]
        assert media["width"] == 640
        assert media["height"] == 480

    def test_text_extra_fields_preserved(self, tmp_path):
        """Text-specific fields (word_count, character_count) are preserved via the registry."""
        medias_data = {
            1: {
                "media_type": "text",
                "embedder": "e5",
                "embedding": np.random.RandomState(42).randn(512).tolist(),
                "media_string": "Hello world",
                "media_bytes": b"Hello world",
                "filename": "doc.txt",
                "word_count": 2,
                "character_count": 11,
                "category": "test",
            }
        }
        pkl_path = tmp_path / "txt.pkl"
        from vtscore.datasets.container import write_container

        write_container(pkl_path, pickle.dumps({"medias": medias_data}), {"format_version": 1})

        # Full mode
        chunks = list(load_dataset_from_pickle_chunked(pkl_path, chunk_size=10, thin=False))
        media = chunks[0][1]
        assert media["media_type"] == "text"
        assert media["word_count"] == 2
        assert media["character_count"] == 11

        # Thin mode
        chunks = list(load_dataset_from_pickle_chunked(pkl_path, chunk_size=10, thin=True))
        media = chunks[0][1]
        assert media["word_count"] == 2
        assert media["character_count"] == 11

    def test_document_type_loaded_via_registry(self, tmp_path):
        """Document media type loads correctly in chunked mode (previously silently failed)."""
        medias_data = {
            1: {
                "media_type": "document",
                "embedder": "e5",
                "embedding": np.random.RandomState(42).randn(512).tolist(),
                "media_bytes": b"%PDF-1.4 fake",
                "filename": "report.pdf",
                "category": "test",
            }
        }
        pkl_path = tmp_path / "doc.pkl"
        from vtscore.datasets.container import write_container

        write_container(pkl_path, pickle.dumps({"medias": medias_data}), {"format_version": 1})

        # Full mode; was silently skipped before registry fix
        chunks = list(load_dataset_from_pickle_chunked(pkl_path, chunk_size=10, thin=False))
        assert len(chunks) == 1
        media = chunks[0][1]
        assert media["media_type"] == "document"
        assert media["media_bytes"] == b"%PDF-1.4 fake"

        # Thin mode
        chunks = list(load_dataset_from_pickle_chunked(pkl_path, chunk_size=10, thin=True))
        assert len(chunks) == 1
        assert chunks[0][1]["media_type"] == "document"

    def test_standard_bytes_keys_via_registry(self, tmp_path):
        """Standard media_bytes key is used for all media types."""
        medias_data = {
            1: {
                "media_type": "audio",
                "embedder": "clap",
                "embedding": np.random.RandomState(42).randn(512).tolist(),
                "media_bytes": _make_wav_bytes(),
                "filename": "clip.wav",
                "category": "test",
            },
            2: {
                "media_type": "image",
                "embedder": "siglip",
                "embedding": np.random.RandomState(42).randn(512).tolist(),
                "media_bytes": b"\x89PNG fake",
                "filename": "pic.png",
                "category": "test",
            },
        }
        pkl_path = tmp_path / "standard.pkl"
        from vtscore.datasets.container import write_container

        write_container(pkl_path, pickle.dumps({"medias": medias_data}), {"format_version": 1})

        chunks = list(load_dataset_from_pickle_chunked(pkl_path, chunk_size=10, thin=False))
        loaded = chunks[0]
        assert len(loaded) == 2
        types = {m["media_type"] for m in loaded.values()}
        assert types == {"audio", "image"}

    def test_external_dir_via_registry(self, tmp_path):
        """External directory loading uses registry dir_key instead of hardcoded keys."""
        doc_dir = tmp_path / "docs"
        doc_dir.mkdir()
        (doc_dir / "report.pdf").write_bytes(b"%PDF fake content")

        medias_data = {
            1: {
                "media_type": "document",
                "embedder": "e5",
                "embedding": np.random.RandomState(42).randn(512).tolist(),
                "filename": "report.pdf",
                "category": "test",
            }
        }
        pkl_path = tmp_path / "ext_doc.pkl"
        from vtscore.datasets.container import write_container

        write_container(
            pkl_path,
            pickle.dumps({"medias": medias_data, "document_dir": str(doc_dir)}),
            {"format_version": 1},
        )

        # Full mode; loads bytes from the external document_dir
        chunks = list(load_dataset_from_pickle_chunked(pkl_path, chunk_size=10, thin=False))
        assert len(chunks) == 1
        media = chunks[0][1]
        assert media["media_type"] == "document"
        assert media["media_bytes"] == b"%PDF fake content"

        # Thin mode; resolves media_path from document_dir
        chunks = list(load_dataset_from_pickle_chunked(pkl_path, chunk_size=10, thin=True))
        assert len(chunks) == 1
        media = chunks[0][1]
        assert media["media_path"] is not None
        assert "report.pdf" in media["media_path"]

    def test_text_media_string_key(self, tmp_path):
        """Text media using media_string key loads correctly."""
        medias_data = {
            1: {
                "media_type": "text",
                "embedder": "e5",
                "embedding": np.random.RandomState(42).randn(512).tolist(),
                "media_string": "Some text paragraph",
                "filename": "para.txt",
                "category": "test",
            }
        }
        pkl_path = tmp_path / "txt.pkl"
        from vtscore.datasets.container import write_container

        write_container(pkl_path, pickle.dumps({"medias": medias_data}), {"format_version": 1})

        chunks = list(load_dataset_from_pickle_chunked(pkl_path, chunk_size=10, thin=False))
        assert len(chunks) == 1
        media = chunks[0][1]
        assert media["media_type"] == "text"
        assert media["media_string"] == "Some text paragraph"
        assert media["media_bytes"] == b"Some text paragraph"


# ======================================================================
# DatasetImporter.run_chunked / run_chunked_cli (base class default)
# ======================================================================


class TestBaseImporterChunkedDefault:
    """Test that the default run_chunked/run_chunked_cli on the base class
    delegates to run/run_cli and yields one chunk."""

    def test_default_run_chunked_yields_one_chunk(self, tmp_path):
        from vtscore.datasets.importers.base import DatasetImporter, PluginField

        class DummyImporter(DatasetImporter):
            name = "dummy"
            display_name = "Dummy"
            description = "Test"
            icon = ""
            fields: list[PluginField] = []

            def run(self, field_values, medias, thin=False):
                medias[1] = {"id": 1, "media_type": "audio", "embedder": "clap", "embeddings": {"clap": np.zeros(4)}}
                medias[2] = {"id": 2, "media_type": "audio", "embedder": "clap", "embeddings": {"clap": np.ones(4)}}

        imp = DummyImporter()
        assert imp.supports_chunked is False

        chunks = list(imp.run_chunked({}, chunk_size=1, thin=True))
        assert len(chunks) == 1
        assert len(chunks[0]) == 2


# ======================================================================
# Folder importer run_chunked
# ======================================================================


class TestFolderImporterChunked:
    def test_supports_chunked(self):
        from vtscore.datasets.importers.server_folder import ServerFolderDatasetImporter

        assert ServerFolderDatasetImporter().supports_chunked is True

    def test_run_chunked(self, tmp_path):
        for i in range(4):
            _make_wav_file(tmp_path, f"s_{i}.wav", frequency=440.0 + i * 10)
        from vtscore.datasets.importers.server_folder import ServerFolderDatasetImporter

        imp = ServerFolderDatasetImporter()
        chunks = list(imp.run_chunked({"path": str(tmp_path), "media_type": "audio"}, chunk_size=2, thin=True))
        assert len(chunks) == 2
        for chunk in chunks:
            assert len(chunk) == 2

    def test_run_chunked_cli(self, tmp_path):
        _make_wav_file(tmp_path, "test.wav")
        from vtscore.datasets.importers.server_folder import ServerFolderDatasetImporter

        imp = ServerFolderDatasetImporter()
        chunks = list(imp.run_chunked_cli({"path": str(tmp_path), "media_type": "audio"}, chunk_size=10, thin=True))
        assert len(chunks) == 1
        assert len(chunks[0]) == 1

    def test_run_chunked_cli_missing_folder(self, tmp_path):
        from vtscore.datasets.importers.server_folder import ServerFolderDatasetImporter

        imp = ServerFolderDatasetImporter()
        try:
            list(imp.run_chunked_cli({"path": "/nonexistent/path", "media_type": "audio"}, chunk_size=10))
            assert False, "Expected FileNotFoundError"
        except FileNotFoundError:
            pass


# ======================================================================
# Pickle importer run_chunked_cli
# ======================================================================


class TestPickleImporterChunked:
    def test_supports_chunked(self):
        from vtscore.datasets.importers.pickle import PickleDatasetImporter

        assert PickleDatasetImporter().supports_chunked is True

    def test_run_chunked_cli(self, tmp_path):
        pkl_path = _make_pickle(tmp_path, 4)
        from vtscore.datasets.importers.pickle import PickleDatasetImporter

        imp = PickleDatasetImporter()
        chunks = list(imp.run_chunked_cli({"file": str(pkl_path)}, chunk_size=2, thin=True))
        assert len(chunks) == 2
        total = sum(len(c) for c in chunks)
        assert total == 4

    def test_run_chunked_cli_missing_file(self, tmp_path):
        from vtscore.datasets.importers.pickle import PickleDatasetImporter

        imp = PickleDatasetImporter()
        try:
            list(imp.run_chunked_cli({"file": "/nonexistent.pkl"}, chunk_size=10))
            assert False, "Expected FileNotFoundError"
        except FileNotFoundError:
            pass


# ======================================================================
# CombineDatasets importer run_chunked
# ======================================================================


class TestCombineDatasetsImporterChunked:
    def test_supports_chunked(self):
        from vtscore.datasets.importers.combine_datasets import CombineDatasetsImporter

        assert CombineDatasetsImporter().supports_chunked is True

    def test_yields_one_chunk_per_source(self, tmp_path):
        (tmp_path / "d1").mkdir(exist_ok=True)
        (tmp_path / "d2").mkdir(exist_ok=True)
        # Use different base frequencies so the WAV bytes (and MD5s) differ
        # between pickles, avoiding cross-source dedup.
        pkl1 = _make_pickle_with_base_freq(tmp_path / "d1", 3, base_freq=440.0)
        pkl2 = _make_pickle_with_base_freq(tmp_path / "d2", 2, base_freq=880.0)

        from vtscore.datasets.importers.combine_datasets import CombineDatasetsImporter

        imp = CombineDatasetsImporter()
        chunks = list(imp.run_chunked({"datasets": f"{pkl1},{pkl2}"}, chunk_size=100))
        assert len(chunks) == 2
        assert len(chunks[0]) == 3
        assert len(chunks[1]) == 2


# ======================================================================
# _merge_detector_results
# ======================================================================


class TestMergeDetectorResults:
    def test_merge_new_detector(self):
        from vtscore.cli import _merge_detector_results

        acc: dict[str, dict[str, Any]] = {}
        new = {
            "det_a": {
                "detector_name": "det_a",
                "threshold": 0.5,
                "total_hits": 2,
                "hits": [
                    {"filename": "f1.wav", "score": 0.9},
                    {"filename": "f2.wav", "score": 0.6},
                ],
            }
        }
        _merge_detector_results(acc, new)
        assert acc["det_a"]["total_hits"] == 2
        assert len(acc["det_a"]["hits"]) == 2

    def test_merge_extends_existing(self):
        from vtscore.cli import _merge_detector_results

        acc = {
            "det_a": {
                "detector_name": "det_a",
                "threshold": 0.5,
                "total_hits": 1,
                "hits": [{"filename": "f1.wav", "score": 0.9}],
            }
        }
        new = {
            "det_a": {
                "detector_name": "det_a",
                "threshold": 0.5,
                "total_hits": 1,
                "hits": [{"filename": "f2.wav", "score": 0.7}],
            }
        }
        _merge_detector_results(acc, new)
        assert acc["det_a"]["total_hits"] == 2
        assert len(acc["det_a"]["hits"]) == 2

    def test_merge_appends_without_sorting(self):
        """Merge only concatenates; ordering is deferred to _sort_detector_results."""
        from vtscore.cli import _merge_detector_results

        acc = {
            "det_a": {
                "detector_name": "det_a",
                "threshold": 0.5,
                "total_hits": 1,
                "hits": [{"filename": "low.wav", "score": 0.5}],
            }
        }
        new = {
            "det_a": {
                "detector_name": "det_a",
                "threshold": 0.5,
                "total_hits": 1,
                "hits": [{"filename": "high.wav", "score": 0.99}],
            }
        }
        _merge_detector_results(acc, new)
        # Appended in chunk order, not re-sorted per chunk.
        assert [h["filename"] for h in acc["det_a"]["hits"]] == ["low.wav", "high.wav"]

    def test_sort_orders_descending(self):
        from vtscore.cli import _merge_detector_results, _sort_detector_results

        acc = {
            "det_a": {
                "detector_name": "det_a",
                "threshold": 0.5,
                "total_hits": 1,
                "hits": [{"filename": "low.wav", "score": 0.5}],
                "negative_hits": [{"filename": "neg_low.wav", "score": 0.1}],
            }
        }
        new = {
            "det_a": {
                "detector_name": "det_a",
                "threshold": 0.5,
                "total_hits": 1,
                "hits": [{"filename": "high.wav", "score": 0.99}],
                "negative_hits": [{"filename": "neg_high.wav", "score": 0.3}],
            }
        }
        _merge_detector_results(acc, new)
        _sort_detector_results(acc)
        assert acc["det_a"]["hits"][0]["filename"] == "high.wav"
        assert acc["det_a"]["hits"][1]["filename"] == "low.wav"
        assert acc["det_a"]["negative_hits"][0]["filename"] == "neg_high.wav"
        assert acc["det_a"]["negative_hits"][1]["filename"] == "neg_low.wav"

    def test_merge_multiple_detectors(self):
        from vtscore.cli import _merge_detector_results

        acc: dict[str, dict[str, Any]] = {}
        new = {
            "det_a": {"detector_name": "det_a", "threshold": 0.5, "total_hits": 1, "hits": [{"score": 0.9}]},
            "det_b": {"detector_name": "det_b", "threshold": 0.3, "total_hits": 1, "hits": [{"score": 0.8}]},
        }
        _merge_detector_results(acc, new)
        assert "det_a" in acc
        assert "det_b" in acc
