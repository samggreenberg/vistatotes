"""The CLI runs the framework's embed stage over a fresh import.

Importers never call an embedder: they emit media dicts with ``embedding=None``
and the framework's ``embed_missing`` stage fills the vectors in once the
importer returns.  The GUI's ``load_pipeline`` runs that stage; the CLI ran none
of the post-import stages and left the vectors to appear incidentally at scoring
time, inside ``route_and_embed``'s per-detector-group embed pass.

That is the second half of issue #3556.  The first thing to read the haystack is
not scoring but *threshold calibration* - ``train_from_labelset`` fits the cut on
``scoring_rows_for_snap(snap)``, which drops every media with no vector - so an
import that arrived unembedded calibrated the detector against a strict subset of
the dataset: a lower cut over fewer items, and only afterwards were the vectors
filled in.  Same dataset, same detector, a different threshold in the CLI than in
the GUI.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any

import numpy as np
import pytest

from vtscore import cli_progress
from vtscore.cli import _load_importer_chunked, _load_importer_whole
from vtscore.embedding.media_vectors import media_embedding, set_media_embedding


DIM = 8


class _FakeEmbedder:
    """A registered-looking embedder that stamps a recognisable vector."""

    supports_patch_regions = False
    supports_geometric_verification = False

    def __init__(self, name: str, fill: float = 1.0, *, refuses: set[str] | None = None) -> None:
        self.name = name
        self.fill = fill
        self.refuses = refuses or set()
        self.seen: list[str] = []
        # Non-None so ``_ensure_model_loaded`` treats the model as warm.
        self._model = object()

    @contextmanager
    def progress_scope(self, _on_progress):
        yield

    def embed_media_bulk(self, medias: list[dict[str, Any]]):
        out = []
        for m in medias:
            self.seen.append(m.get("origin_name", ""))
            if m.get("origin_name") in self.refuses:
                out.append(None)
            else:
                out.append(np.full(DIM, self.fill, dtype=np.float32))
        return out


class _StubImporter:
    """Emits the media dicts it was constructed with, unembedded."""

    name = "stub"

    def __init__(self, medias: list[dict[str, Any]]) -> None:
        self._medias = medias

    def validate_cli_field_values(self, field_values: dict[str, Any]) -> None:
        return None

    def run_cli(self, field_values: dict[str, Any], medias: dict, thin: bool = False) -> None:
        for i, media in enumerate(self._medias, 1):
            medias[i] = dict(media, id=i)

    def run_chunked_cli(self, field_values: dict[str, Any], chunk_size: int, thin: bool = False):
        batch: dict[int, dict[str, Any]] = {}
        for media in self._medias:
            batch[len(batch) + 1] = dict(media)
            if len(batch) == chunk_size:
                yield batch
                batch = {}
        if batch:
            yield batch


def _media(name: str, media_type: str = "audio") -> dict[str, Any]:
    return {"media_type": media_type, "origin_name": name, "embedding": None}


@pytest.fixture
def registry(monkeypatch):
    """Register one fake embedder per media type and hand them back by type."""
    by_type = {"audio": _FakeEmbedder("fake_audio", 1.0), "image": _FakeEmbedder("fake_image", 2.0)}
    by_name = {e.name: e for e in by_type.values()}
    monkeypatch.setattr("vtscore.media.embedders_for_type", lambda mt: [by_type[mt]] if mt in by_type else [])
    monkeypatch.setattr("vtscore.media.get_embedder", lambda name: by_name[name])
    return by_type


def _install(monkeypatch, importer) -> None:
    monkeypatch.setattr("vtscore.datasets.importers.get_importer", lambda _n: importer)


class TestImportArrivesEmbedded:
    def test_every_media_carries_a_vector_before_the_chunk_is_yielded(self, monkeypatch, registry):
        """The #3556 regression: these used to reach calibration unembedded."""
        _install(monkeypatch, _StubImporter([_media("a"), _media("b"), _media("c")]))

        (chunk,) = list(_load_importer_whole("stub", {}))

        assert len(chunk) == 3
        assert all(media_embedding(m) is not None for m in chunk.values())

    def test_the_whole_import_is_embedded_not_just_the_first_item(self, monkeypatch, registry):
        _install(monkeypatch, _StubImporter([_media(n) for n in "abcdefgh"]))

        (chunk,) = list(_load_importer_whole("stub", {}))

        assert sorted(registry["audio"].seen) == list("abcdefgh")

    def test_each_chunk_is_embedded_as_it_is_yielded(self, monkeypatch, registry):
        """Chunking exists so a big dataset is scored a chunk at a time; each
        chunk is calibrated and scored before the next is loaded, so each must
        be embedded on the way out rather than once at the end."""
        _install(monkeypatch, _StubImporter([_media(n) for n in "abcd"]))

        chunks = list(_load_importer_chunked("stub", {}, 2))

        assert [len(c) for c in chunks] == [2, 2]
        for chunk in chunks:
            assert all(media_embedding(m) is not None for m in chunk.values())

    def test_already_embedded_media_are_left_alone(self, monkeypatch, registry):
        """A content-vector importer ships its own vectors; the stage is a no-op."""
        pre = _media("a")
        set_media_embedding(pre, "fake_audio", np.full(DIM, 9.0, dtype=np.float32))
        _install(monkeypatch, _StubImporter([pre]))

        (chunk,) = list(_load_importer_whole("stub", {}))

        assert registry["audio"].seen == []
        assert media_embedding(chunk[1]) == pytest.approx(np.full(DIM, 9.0))


class TestMixedTypeImports:
    def test_each_media_type_resolves_its_own_embedder(self, monkeypatch, registry):
        """A single whole-dict ``embed_missing`` call resolves one embedder from
        the first media type it finds, which would push every video in a mixed
        import through the image model.  Group by type instead."""
        _install(monkeypatch, _StubImporter([_media("a", "audio"), _media("i", "image"), _media("b", "audio")]))

        (chunk,) = list(_load_importer_whole("stub", {}))

        assert registry["audio"].seen == ["a", "b"]
        assert registry["image"].seen == ["i"]
        assert media_embedding(chunk[2]) == pytest.approx(np.full(DIM, 2.0))


class TestNothingIsDropped:
    def test_a_media_the_embedder_refused_is_kept(self, monkeypatch, registry):
        """The GUI drops these; the CLI must not.  Its dataset is not scored in
        the space it was loaded in - an item with no native vector may still be
        scoreable through the one-hop converter route ``route_and_embed``
        applies later, so dropping it here would lose hits the CLI finds today.
        """
        registry["audio"].refuses = {"b"}
        _install(monkeypatch, _StubImporter([_media("a"), _media("b"), _media("c")]))

        (chunk,) = list(_load_importer_whole("stub", {}))

        assert sorted(chunk) == [1, 2, 3]
        assert media_embedding(chunk[2]) is None

    def test_a_type_with_no_registered_embedder_is_kept(self, monkeypatch, registry):
        _install(monkeypatch, _StubImporter([_media("a"), _media("d", "document")]))

        (chunk,) = list(_load_importer_whole("stub", {}))

        assert sorted(chunk) == [1, 2]
        assert media_embedding(chunk[2]) is None

    def test_what_stayed_unembedded_is_announced(self, monkeypatch, registry, capsys):
        """Silence here is what made #3556 hard to see from the outside: the
        export was simply short.  Report it on the same event stream as the
        scoring-time skips, so ``--progress-format json`` shows the count."""
        registry["audio"].refuses = {"b"}
        _install(monkeypatch, _StubImporter([_media("a"), _media("b")]))
        monkeypatch.setattr(cli_progress, "_format", "json")

        list(_load_importer_whole("stub", {}))

        events = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]
        (announced,) = [e for e in events if e["event"] == "medias_unembedded"]
        assert announced["unembedded"] == 1
        assert announced["unembedded_ids"] == [2]

    def test_a_fully_embedded_import_announces_nothing(self, monkeypatch, registry, capsys):
        _install(monkeypatch, _StubImporter([_media("a"), _media("b")]))
        monkeypatch.setattr(cli_progress, "_format", "json")

        list(_load_importer_whole("stub", {}))

        events = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]
        assert [e for e in events if e["event"] == "medias_unembedded"] == []
