"""End-to-end tests for chunk-id renumbering in the CLI autodetect flow.

Every chunked importer/loader emits chunks with ids ``1..N``; the
in-process consumer ``consume_chunks_into`` renumbers them, but the CLI
pipeline used to score and merge them as-is, which collided ``id``
values across chunks in the merged hit lists and produced ambiguous
``id`` fields in the exported JSON.  These tests pin the fix in place
by running the actual CLI through to a JSON export and asserting the
``id`` fields are globally unique across hits.
"""

from __future__ import annotations

import json
import pickle
import shutil
from pathlib import Path

import pytest

from vtsearch.settings import get_detectors_dir
from vtscore.utils.hashing import content_md5


def _unique_bytes(media_id: int) -> bytes:
    """Return per-id unique bytes so MD5 hashes don't collide across pickles."""
    return media_id.to_bytes(4, "little") + b"\x00" * 96


def _make_audio_media(media_id: int) -> dict:
    """Build a minimal audio media dict for a self-contained pickle.

    The bytes are stored inline.  They used to be omitted, which only worked
    because the CLI forced thin loading on every import - thin needs no bytes,
    so a payload-less entry sailed through.  Now that the CLI resolves thin
    from the importer's ``reference_files`` field (off by default, as in the
    GUI), a full-mode load of a byte-less, path-less entry drops it the same
    way the GUI has always dropped it, and the fixture would import as an
    empty dataset (issue #3556).
    """
    raw = _unique_bytes(media_id)
    md5 = content_md5(raw)
    return {
        "id": media_id,
        "media_type": "audio",
        "duration": 1.0,
        "file_size": len(raw),
        "md5": md5,
        # Two-dim embedding keeps the trained MLP trivially small.
        "embedding": [float(media_id), float(media_id) + 0.5],
        "embedder": "clap",
        "media_bytes": raw,
        "media_string": None,
        "media_path": None,
        "filename": f"clip_{media_id:03d}.wav",
        "category": "test",
        "origin": None,
        "origin_name": f"clip_{media_id:03d}.wav",
    }


def _write_pickle_dataset(path: Path, medias: dict) -> None:
    from vtscore.datasets.container import write_container

    write_container(path, pickle.dumps({"medias": medias}), {"format_version": 1})


def _settings_file_with_detector(tmp_path: Path, detector_name: str) -> Path:
    settings = {
        "autofind_detectors": [detector_name],
        "detectors_dir": str(get_detectors_dir()),
    }
    p = tmp_path / "settings.json"
    p.write_text(json.dumps(settings))
    return p


def _write_pretrained_detector(name: str, dim: int = 2) -> Path:
    """Write a detector with a labelset that yields a trainable MLP.

    The CLI re-resolves the labelset's origins and re-embeds; we stub
    that resolution out at runtime in the test bodies, so the labelset
    contents only need to satisfy "has >=1 good and >=1 bad label".
    """
    from vtscore.detectors.store import _detector_path, _write_detector

    path = _detector_path(name)
    _write_detector(
        path,
        {
            "name": name,
            "text_query": "",
            "media_type": "audio",
            "examples": [],
            "labelset": {
                "labels": [
                    {
                        "md5": "a" * 32,
                        "label": "good",
                        "origin": {"importer": "stub_ds", "params": {}},
                        "origin_name": "label_a.wav",
                    },
                    {
                        "md5": "b" * 32,
                        "label": "bad",
                        "origin": {"importer": "stub_ds", "params": {}},
                        "origin_name": "label_b.wav",
                    },
                ]
            },
        },
    )
    return path


@pytest.fixture(autouse=True)
def _clean_detectors_dir():
    d = get_detectors_dir()
    if d.is_dir():
        shutil.rmtree(d)
    yield
    d = get_detectors_dir()
    if d.is_dir():
        shutil.rmtree(d)


@pytest.fixture
def _stub_detector_training(monkeypatch):
    """Bypass labelset-origin resolution and embed a 2-dim toy MLP.

    The CLI's ``_load_and_train_detectors`` walks the labelset, resolves
    origins, and re-embeds them.  For these tests we only care that
    *some* trained MLP comes back; we stub the loader to return a
    fixed (mlp, threshold) pair so the test focuses on chunk-id
    renumbering, not training mechanics.
    """
    import torch
    from torch import nn

    import vtscore.cli as cli_mod

    def _fake_load_and_train(detector_names, media_type, first_chunk_medias, routed=None):
        # Tiny linear "MLP" that always returns 0.5 logit → 0.62 sigmoid.
        linear = nn.Linear(2, 1)
        with torch.no_grad():
            linear.weight.data.zero_()
            linear.bias.data.fill_(0.5)
        mlp = nn.Sequential(linear)
        mlp.eval()
        return {name: {"mlp": mlp, "threshold": 0.5} for name in detector_names}

    monkeypatch.setattr(cli_mod, "_load_and_train_detectors", _fake_load_and_train)


class TestChunkedPickleIdRenumber:
    def test_pickle_chunked_export_has_globally_unique_hit_ids(self, client, tmp_path, _stub_detector_training):
        """A pickle of 5 medias loaded with chunk_size=2 yields 3 chunks.

        Without renumbering, the chunks would carry ids ``[1,2]``,
        ``[1,2]``, ``[1]``; and the merged JSON would have ``id``
        collisions.  After the fix every hit's ``id`` is unique.
        """
        _write_pretrained_detector("renum-tm")
        settings_path = _settings_file_with_detector(tmp_path, "renum-tm")

        # 5 medias, each with distinct MD5/filename.
        medias = {i: _make_audio_media(i) for i in range(1, 6)}
        ds_path = tmp_path / "ds.pkl"
        _write_pickle_dataset(ds_path, medias)

        export_path = tmp_path / "results.json"

        from vtscore.cli import autodetect_main_chunked

        autodetect_main_chunked(
            dataset_path=str(ds_path),
            chunk_size=2,
            settings_path=str(settings_path),
            exporter_name="server_json_file",
            exporter_field_values={"filepath": str(export_path)},
        )

        results = json.loads(export_path.read_text())
        det_result = results["results"]["renum-tm"]
        # 5 medias → 5 hits (all land in either hits or negative_hits;
        # combine both lists to check global uniqueness).
        all_ids = [h["id"] for h in det_result.get("hits", [])]
        all_ids += [h["id"] for h in det_result.get("negative_hits", [])]
        assert sorted(all_ids) == [1, 2, 3, 4, 5]
        # And every filename appears exactly once (sanity; medias
        # weren't dropped or duplicated).
        all_names = [h["filename"] for h in det_result.get("hits", [])]
        all_names += [h["filename"] for h in det_result.get("negative_hits", [])]
        assert sorted(all_names) == [f"clip_{i:03d}.wav" for i in range(1, 6)]


class TestChunkedCombineDatasetsIdRenumber:
    def test_combine_datasets_chunked_export_has_globally_unique_hit_ids(
        self, client, tmp_path, _stub_detector_training
    ):
        """combine_datasets emits one chunk per source pickle, each with ids 1..N.

        Two source pickles → two chunks → ids ``[1,2]`` and ``[1,2]``
        from the importer.  After CLI-boundary renumbering the merged
        JSON's hit ids must be globally unique.
        """
        _write_pretrained_detector("combine-tm")
        settings_path = _settings_file_with_detector(tmp_path, "combine-tm")

        # Two pickles, two distinct medias each (different MD5s so no
        # cross-pickle dedup).
        p1_medias = {1: _make_audio_media(11), 2: _make_audio_media(12)}
        p2_medias = {1: _make_audio_media(21), 2: _make_audio_media(22)}
        p1 = tmp_path / "ds1.pkl"
        p2 = tmp_path / "ds2.pkl"
        _write_pickle_dataset(p1, p1_medias)
        _write_pickle_dataset(p2, p2_medias)

        export_path = tmp_path / "combined.json"

        from vtscore.cli import autodetect_importer_main_chunked

        # one chunk per source pickle regardless; but the CLI requires
        # one chunk per source pickle regardless - but the CLI requires
        # a positive int.
        autodetect_importer_main_chunked(
            "combine_datasets",
            {"datasets": f"{p1},{p2}", "name": "test_combined"},
            chunk_size=100,
            settings_path=str(settings_path),
            exporter_name="server_json_file",
            exporter_field_values={"filepath": str(export_path)},
        )

        results = json.loads(export_path.read_text())
        det_result = results["results"]["combine-tm"]
        all_ids = [h["id"] for h in det_result.get("hits", [])]
        all_ids += [h["id"] for h in det_result.get("negative_hits", [])]
        # 4 hits across both pickles, all with unique ids 1..4.
        assert sorted(all_ids) == [1, 2, 3, 4]
        all_names = [h["filename"] for h in det_result.get("hits", [])]
        all_names += [h["filename"] for h in det_result.get("negative_hits", [])]
        assert sorted(all_names) == [
            "clip_011.wav",
            "clip_012.wav",
            "clip_021.wav",
            "clip_022.wav",
        ]
