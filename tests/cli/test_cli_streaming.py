"""End-to-end tests for the ``--stream-results`` CLI autodetect path.

Streaming scores each chunk and writes its hits straight to the exporter
with no global accumulation, so a media source larger than RAM can be
scanned.  These tests pin: (1) NDJSON streaming export shape, (2) negatives
dropped by default but kept with ``keep_negatives``, and (3) a clear error
when the chosen exporter can't stream.
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
    return media_id.to_bytes(4, "little") + b"\x00" * 96


def _make_audio_media(media_id: int) -> dict:
    raw = _unique_bytes(media_id)
    # 2-dim *unit-norm* embedding whose dim 0 strictly decreases with id
    # (0.9, 0.8, 0.7, 0.6, 0.5 for ids 1..5).  Embeddings are L2-normalized at
    # ingest, so we store a unit vector to make that normalization a no-op and
    # keep dim 0 the clean, id-ordered signal the stubbed MLP thresholds on.
    e0 = 1.0 - 0.1 * media_id
    e1 = (1.0 - e0 * e0) ** 0.5
    return {
        "id": media_id,
        "media_type": "audio",
        "duration": 1.0,
        "file_size": len(raw),
        "md5": content_md5(raw),
        "embedding": [e0, e1],
        "embedder": "clap",
        "media_bytes": None,
        "media_string": None,
        "media_path": None,
        "filename": f"clip_{media_id:03d}.wav",
        "category": "test",
        "origin": {"importer": "stub_ds", "params": {}},
        "origin_name": f"clip_{media_id:03d}.wav",
    }


def _write_pickle_dataset(path: Path, medias: dict) -> None:
    from vtscore.datasets.container import write_container

    write_container(path, pickle.dumps({"medias": medias}), {"format_version": 1})


def _settings_file_with_detector(tmp_path: Path, detector_name: str) -> Path:
    settings = {"autofind_detectors": [detector_name], "detectors_dir": str(get_detectors_dir())}
    p = tmp_path / "settings.json"
    p.write_text(json.dumps(settings))
    return p


def _settings_file_with_detector_and_exporter(
    tmp_path: Path,
    detector_name: str,
    exporter_name: str,
    exporter_field_values: dict,
) -> Path:
    """Like ``_settings_file_with_detector`` but also configures the Auto-Find exporter.

    Used to exercise the settings-based exporter fallback: a streaming run that
    passes no ``--exporter`` must pick up this exporter (and its field values)
    from settings, exactly as the buffered path does.
    """
    settings = {
        "autofind_detectors": [detector_name],
        "detectors_dir": str(get_detectors_dir()),
        "autofind_exporter": exporter_name,
        "autofind_exporter_field_values": {exporter_name: exporter_field_values},
    }
    p = tmp_path / "settings.json"
    p.write_text(json.dumps(settings))
    return p


def _write_pretrained_detector(name: str) -> None:
    from vtscore.detectors.store import _detector_path, _write_detector

    _write_detector(
        _detector_path(name),
        {
            "name": name,
            "media_type": "audio",
            "labelset": {
                "labels": [
                    {"md5": "a" * 32, "label": "good", "origin": {"importer": "s", "params": {}}, "origin_name": "a"},
                    {"md5": "b" * 32, "label": "bad", "origin": {"importer": "s", "params": {}}, "origin_name": "b"},
                ]
            },
        },
    )


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
def _stub_split_training(monkeypatch):
    """Train a deterministic MLP whose logit is ``100 * (embedding[0] - 0.65)``.

    With the unit-norm embeddings from ``_make_audio_media`` (dim 0 = 0.9, 0.8,
    0.7, 0.6, 0.5 for ids 1..5) and threshold 0.5 (sigmoid), the 0.65 boundary
    puts ids 1/2/3 above threshold (good) and 4/5 below (bad), giving a fixed
    positive/negative split to assert against.  The wide ``100`` scale keeps the
    sigmoid margins clear of float-precision wobble.
    """
    import torch
    from torch import nn

    import vtscore.cli as cli_mod

    def _fake_load_and_train(detector_names, media_type, first_chunk_medias, routed=None):
        linear = nn.Linear(2, 1)
        with torch.no_grad():
            linear.weight.data = torch.tensor([[100.0, 0.0]])
            linear.bias.data = torch.tensor([-65.0])
        mlp = nn.Sequential(linear)
        mlp.eval()
        return {name: {"mlp": mlp, "threshold": 0.5} for name in detector_names}

    monkeypatch.setattr(cli_mod, "_load_and_train_detectors", _fake_load_and_train)


def _read_ndjson(path: Path) -> tuple[dict, list[dict]]:
    """Return ``(meta, hit_records)`` from an NDJSON export file."""
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    objs = [json.loads(ln) for ln in lines]
    meta = objs[0]["_meta"]
    return meta, objs[1:]


class TestStreamingNdjsonExport:
    def test_default_streams_only_positive_hits(self, client, tmp_path, _stub_split_training):
        _write_pretrained_detector("stream-tm")
        settings_path = _settings_file_with_detector(tmp_path, "stream-tm")
        ds_path = tmp_path / "ds.pkl"
        _write_pickle_dataset(ds_path, {i: _make_audio_media(i) for i in range(1, 6)})
        out = tmp_path / "hits.ndjson"

        from vtscore.cli import autodetect_main_chunked

        autodetect_main_chunked(
            dataset_path=str(ds_path),
            chunk_size=2,
            settings_path=str(settings_path),
            exporter_name="server_json_file",
            exporter_field_values={"filepath": str(out)},
            stream_results=True,
        )

        meta, hits = _read_ndjson(out)
        assert meta["format"] == "vtsearch-hits-ndjson/v1"
        assert meta["keep_negatives"] is False
        assert {d["detector_name"] for d in meta["detectors"]} == {"stream-tm"}
        # ids 1,2,3 are above threshold; negatives dropped by default.
        assert all(h["label"] == "good" for h in hits)
        assert sorted(h["id"] for h in hits) == [1, 2, 3]
        assert all(h["detector"] == "stream-tm" for h in hits)
        # The atomic temp file must not be left behind.
        assert not out.with_name(out.name + ".tmp").exists()

    def test_keep_negatives_streams_both(self, client, tmp_path, _stub_split_training):
        _write_pretrained_detector("stream-tm2")
        settings_path = _settings_file_with_detector(tmp_path, "stream-tm2")
        ds_path = tmp_path / "ds.pkl"
        _write_pickle_dataset(ds_path, {i: _make_audio_media(i) for i in range(1, 6)})
        out = tmp_path / "hits.ndjson"

        from vtscore.cli import autodetect_main_chunked

        autodetect_main_chunked(
            dataset_path=str(ds_path),
            chunk_size=2,
            settings_path=str(settings_path),
            exporter_name="server_json_file",
            exporter_field_values={"filepath": str(out)},
            stream_results=True,
            keep_negatives=True,
        )

        meta, hits = _read_ndjson(out)
        assert meta["keep_negatives"] is True
        good = sorted(h["id"] for h in hits if h["label"] == "good")
        bad = sorted(h["id"] for h in hits if h["label"] == "bad")
        assert good == [1, 2, 3]
        assert bad == [4, 5]


class TestStreamingSettingsExporterFallback:
    """A streaming run with no ``--exporter`` picks up the exporter from settings.

    The settings-based fallback lives at the top of ``_run_pipeline``, before the
    buffered/streaming split, so streaming inherits it just like the buffered
    path: both the exporter name *and* its per-exporter field values come from
    the settings file when the CLI omits ``--exporter``.
    """

    def test_streaming_falls_back_to_settings_exporter(self, client, tmp_path, _stub_split_training):
        out = tmp_path / "hits.ndjson"
        settings_path = _settings_file_with_detector_and_exporter(
            tmp_path, "stream-fb", "server_json_file", {"filepath": str(out)}
        )
        _write_pretrained_detector("stream-fb")
        ds_path = tmp_path / "ds.pkl"
        _write_pickle_dataset(ds_path, {i: _make_audio_media(i) for i in range(1, 6)})

        from vtscore.cli import autodetect_main_chunked

        # No exporter_name / exporter_field_values: both must come from settings.
        autodetect_main_chunked(
            dataset_path=str(ds_path),
            chunk_size=2,
            settings_path=str(settings_path),
            stream_results=True,
        )

        # The file at the settings-configured path exists (name fallback) and
        # carries the streamed hits (field-value fallback).
        assert out.exists()
        meta, hits = _read_ndjson(out)
        assert meta["format"] == "vtsearch-hits-ndjson/v1"
        assert {d["detector_name"] for d in meta["detectors"]} == {"stream-fb"}
        assert sorted(h["id"] for h in hits) == [1, 2, 3]


class TestStreamingExporterGuard:
    def test_non_streaming_exporter_raises(self):
        from vtscore.cli import _run_streaming_pipeline

        # open_url is a registered exporter with no incremental (streaming) mode.
        with pytest.raises(ValueError, match="does not support --stream-results"):
            _run_streaming_pipeline(
                iter([{1: _make_audio_media(1)}]),
                exporter_name="open_url",
                exporter_field_values={},
                override_detectors=None,
                autofind_detectors=[],
                keep_negatives=False,
                empty_error="none",
            )


class TestStreamingWholeDatasetVariants:
    """``stream_results`` is accepted by the *whole* entry points too.

    It used to be chunked-only, so a caller who did not want chunking had no
    way to hand a streaming-capable exporter a lazy record iterator even
    though ``_run_pipeline`` supported it.  A whole-dataset run holds all the
    medias in RAM by construction, but streaming still keeps the *hits* from
    accumulating - and, more to the point, the four entry points now differ
    only in their source.
    """

    def test_whole_pickle_run_streams_positive_hits(self, client, tmp_path, _stub_split_training):
        _write_pretrained_detector("stream-whole")
        settings_path = _settings_file_with_detector(tmp_path, "stream-whole")
        ds_path = tmp_path / "ds.pkl"
        _write_pickle_dataset(ds_path, {i: _make_audio_media(i) for i in range(1, 6)})
        out = tmp_path / "hits.ndjson"

        from vtscore.cli import autodetect_main

        autodetect_main(
            str(ds_path),
            settings_path=str(settings_path),
            exporter_name="server_json_file",
            exporter_field_values={"filepath": str(out)},
            stream_results=True,
        )

        meta, hits = _read_ndjson(out)
        assert meta["format"] == "vtsearch-hits-ndjson/v1"
        assert meta["keep_negatives"] is False
        assert all(h["label"] == "good" for h in hits)
        assert sorted(h["id"] for h in hits) == [1, 2, 3]

    def test_whole_pickle_run_keeps_negatives(self, client, tmp_path, _stub_split_training):
        _write_pretrained_detector("stream-whole-neg")
        settings_path = _settings_file_with_detector(tmp_path, "stream-whole-neg")
        ds_path = tmp_path / "ds.pkl"
        _write_pickle_dataset(ds_path, {i: _make_audio_media(i) for i in range(1, 6)})
        out = tmp_path / "hits.ndjson"

        from vtscore.cli import autodetect_main

        autodetect_main(
            str(ds_path),
            settings_path=str(settings_path),
            exporter_name="server_json_file",
            exporter_field_values={"filepath": str(out)},
            stream_results=True,
            keep_negatives=True,
        )

        meta, hits = _read_ndjson(out)
        assert meta["keep_negatives"] is True
        assert sorted(h["id"] for h in hits if h["label"] == "good") == [1, 2, 3]
        assert sorted(h["id"] for h in hits if h["label"] == "bad") == [4, 5]

    def test_whole_run_dry_run_reports_streaming(self, client, tmp_path, capsys):
        _write_pretrained_detector("stream-whole-dry")
        settings_path = _settings_file_with_detector(tmp_path, "stream-whole-dry")
        ds_path = tmp_path / "ds.pkl"
        _write_pickle_dataset(ds_path, {i: _make_audio_media(i) for i in range(1, 6)})

        from vtscore.cli import autodetect_main

        autodetect_main(
            str(ds_path),
            settings_path=str(settings_path),
            exporter_name="server_json_file",
            exporter_field_values={"filepath": str(tmp_path / "out.ndjson")},
            dry_run=True,
            stream_results=True,
        )

        out = capsys.readouterr().out
        assert "Chunk size: whole dataset" in out
        assert "Streaming: yes" in out
