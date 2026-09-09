"""A media whose embedding failed is skipped, not fatal (issue #3179).

Before this, one unembeddable image aborted an entire CLI run: the matrix
builders raise on a missing vector (``ValueError``, matrix.py) or a
wrong-width one (``MismatchedVectorError``, precomputed.py), and every CLI
scoring path fed them the raw chunk.  The importer path was worse than
"one bad image" - it crashed unconditionally, because the safe-threshold
population pass runs over the chunk *before* ``route_and_embed`` has
embedded anything.

The policy is now the one the load pipeline's ``_drop_none_embeddings_stage``
and ``route_and_embed`` already used: drop the item, say so, score the rest.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from vtsearch.settings import get_detectors_dir
from vtscore.cli import _score_direct_all, _score_medias_with_detectors
from vtscore.media.audio.audio_generator import generate_wav

DIM = 4


def _mlp() -> nn.Sequential:
    linear = nn.Linear(DIM, 1)
    with torch.no_grad():
        linear.weight.fill_(1.0)
        linear.bias.fill_(0.0)
    return nn.Sequential(linear).eval()


def _media(cid: int, vec, name: str = "siglip") -> dict:
    return {
        "id": cid,
        "media_type": "image",
        "embedder": name,
        "embeddings": {} if vec is None else {name: vec},
        "filename": f"m{cid}.png",
        "md5": f"{cid:032d}",
    }


def _scored_ids(result: dict) -> set[int]:
    return {h["id"] for h in result["hits"]} | {h["id"] for h in result["negative_hits"]}


class TestScoreDirectSkipsFailedEmbeds:
    def test_missing_vector_is_skipped_not_fatal(self, client):
        medias = {
            1: _media(1, np.ones(DIM, dtype=np.float32)),
            2: _media(2, None),
            3: _media(3, np.zeros(DIM, dtype=np.float32)),
        }
        out = _score_direct_all(["d"], {"d": {"mlp": _mlp(), "threshold": 0.5}}, medias, "siglip")
        assert _scored_ids(out["d"]) == {1, 3}

    def test_wrong_width_vector_is_skipped_not_fatal(self, client):
        medias = {
            1: _media(1, np.ones(DIM, dtype=np.float32)),
            2: _media(2, np.ones(DIM * 2, dtype=np.float32)),
            3: _media(3, np.zeros(DIM, dtype=np.float32)),
        }
        out = _score_direct_all(["d"], {"d": {"mlp": _mlp(), "threshold": 0.5}}, medias, "siglip")
        assert _scored_ids(out["d"]) == {1, 3}

    def test_nothing_scoreable_yields_no_results(self, client):
        medias = {1: _media(1, None), 2: _media(2, None)}
        out = _score_direct_all(["d"], {"d": {"mlp": _mlp(), "threshold": 0.5}}, medias, "siglip")
        assert out == {}

    def test_skip_is_announced_on_the_event_stream(self, client, capsys):
        medias = {1: _media(1, np.ones(DIM, dtype=np.float32)), 2: _media(2, None)}
        _score_direct_all(["d"], {"d": {"mlp": _mlp(), "threshold": 0.5}}, medias, "siglip")
        assert "Skipped 1 media" in capsys.readouterr().out


class TestTypedDetectorSkipsFailedEmbeds:
    def test_routed_scoring_drops_the_failed_media(self, client, monkeypatch):
        """The typed path already dropped vector-less clips in
        ``route_and_embed``; this pins that a *wrong-width* one is dropped too
        rather than reaching the matrix builder."""

        def _fake_embed(medias, name="", on_progress=None):
            return None  # every media already carries (or lacks) its vector

        monkeypatch.setattr("vtscore.datasets.stages.embedding.embed_missing", _fake_embed)

        medias = {
            1: _media(1, np.ones(DIM, dtype=np.float32)),
            2: _media(2, np.ones(DIM * 2, dtype=np.float32)),
        }
        results = _score_medias_with_detectors(
            medias,
            {"d": {"mlp": _mlp(), "threshold": 0.5, "media_type": "image", "embedder": "siglip"}},
        )
        assert _scored_ids(results["d"]) == {1}


class TestTrainAndThresholdOverAPartlyEmbeddedHaystack:
    def test_population_pass_skips_unscoreable_media(self, client):
        from vtscore.detectors.training import train_and_threshold

        snap = {
            1: _media(1, np.ones(DIM, dtype=np.float32)),
            2: _media(2, None),
            3: _media(3, np.zeros(DIM, dtype=np.float32)),
        }
        X = [np.ones(DIM, dtype=np.float32), np.zeros(DIM, dtype=np.float32)]
        model, threshold = train_and_threshold(X, [1.0, 0.0], snap=snap, embedder_name="siglip")
        assert model is not None
        assert np.isfinite(threshold)

    def test_a_clean_haystack_never_pays_for_the_filter(self, client, monkeypatch):
        """The filter repairs a failed build; it is not a per-vote pre-pass.

        ``_score_all_media`` runs on every vote-driven retrain (and once more
        per calibration fold), so an unconditional O(N) scan would tax every
        vote on a large dataset to catch a case the load pipeline has already
        made impossible there.
        """
        import vtscore.embedding.matrix as matrix_mod
        from vtscore.detectors.training import _score_all_media

        calls: list[int] = []
        real = matrix_mod.scoreable_snapshot
        monkeypatch.setattr(
            matrix_mod,
            "scoreable_snapshot",
            lambda *a, **kw: (calls.append(1), real(*a, **kw))[1],
        )

        snap = {
            1: _media(1, np.ones(DIM, dtype=np.float32)),
            2: _media(2, np.zeros(DIM, dtype=np.float32)),
        }
        ids, _scores, _best = _score_all_media(_mlp(), snap, "siglip")
        assert ids == [1, 2]
        assert calls == []

    def test_wholly_unembedded_haystack_falls_back_to_the_no_snap_threshold(self, client):
        """A haystack in which nothing is scoreable - every media reachable
        only through a converter route, embedded later by ``route_and_embed``
        in the detector's target space - fits on no distribution at all."""
        from vtscore.detectors.training import train_and_threshold

        snap = {1: _media(1, None), 2: _media(2, None)}
        X = [np.ones(DIM, dtype=np.float32), np.zeros(DIM, dtype=np.float32)]
        _model, with_empty_snap = train_and_threshold(X, [1.0, 0.0], snap=snap, embedder_name="siglip")
        _model2, with_no_snap = train_and_threshold(X, [1.0, 0.0], snap=None, embedder_name="siglip")
        assert with_empty_snap == pytest.approx(with_no_snap)


def _stage_folder_run(tmp_path, monkeypatch) -> tuple[str, str, str]:
    """Stage a four-file audio folder, a detector over two of them, and settings.

    Returns ``(folder, settings_path, out_path)`` ready for
    ``autodetect_importer_main("server_folder", ...)``.
    """
    from contextlib import contextmanager

    import vtscore.detectors.resolver as resolver_mod
    from vtscore.detectors.store import _detector_path, _write_detector

    folder = tmp_path / "sounds"
    folder.mkdir()
    files = {}
    for i, name in enumerate(["alpha.wav", "beta.wav", "gamma.wav", "delta.wav"]):
        path = folder / name
        path.write_bytes(generate_wav(220 + 110 * i, 0.1))
        files[name] = path

    @contextmanager
    def _fake_ctx(origin, origin_name="", filename=""):
        yield files.get(origin_name) or files.get(filename)

    monkeypatch.setattr(resolver_mod, "resolve_file_context", _fake_ctx)

    labelset = {
        "labels": [
            {
                "md5": "a" * 32,
                "label": "good",
                "origin": {"importer": "ds_a", "params": {}},
                "origin_name": "alpha.wav",
            },
            {
                "md5": "c" * 32,
                "label": "bad",
                "origin": {"importer": "ds_a", "params": {}},
                "origin_name": "gamma.wav",
            },
        ]
    }
    _write_detector(
        _detector_path("folder-det"),
        {"name": "folder-det", "media_type": "audio", "labelset": labelset},
    )

    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        json.dumps({"autofind_detectors": ["folder-det"], "detectors_dir": str(get_detectors_dir())})
    )
    return str(folder), str(settings_path), str(tmp_path / "hits.json")


class TestCliImporterRunCompletes:
    """End-to-end regression: ``--autodetect --importer server_folder`` used to
    exit(1) on every run, since the importer leaves each media unembedded and
    the training pass scored that raw chunk."""

    def test_folder_importer_autodetect_exports_hits(self, client, tmp_path, monkeypatch):
        from vtscore.cli import autodetect_importer_main

        folder, settings_path, out_path = _stage_folder_run(tmp_path, monkeypatch)

        autodetect_importer_main(
            "server_folder",
            {"path": folder, "media_type": "audio"},
            settings_path=settings_path,
            exporter_name="server_json_file",
            exporter_field_values={"filepath": out_path},
        )

        det = json.loads(Path(out_path).read_text())["results"]["folder-det"]
        # All four files scored: none was dropped, and the run did not exit(1).
        assert len(det["hits"]) + len(det["negative_hits"]) == 4

    def test_the_haystack_reaches_calibration_fully_embedded(self, client, tmp_path, monkeypatch):
        """Issue #3556's second half.  ``train_from_labelset`` fits the detector's
        threshold on the snap it is handed, and ``scoring_rows_for_snap`` drops
        every media in it with no vector - so a snap that arrives part-embedded
        calibrates the cut on a strict subset of the dataset, silently and
        plausibly.  The CLI used to hand it the raw importer output and let
        ``route_and_embed`` fill the vectors in afterwards.
        """
        import vtscore.detectors.labelset_training as labelset_training
        from vtscore.cli import autodetect_importer_main
        from vtscore.embedding.media_vectors import media_embedding

        folder, settings_path, out_path = _stage_folder_run(tmp_path, monkeypatch)

        snaps: list[list[bool]] = []
        real = labelset_training.train_from_labelset

        def _spy(ctx, labelset, media_type="", snap=None, **kw):
            snaps.append([media_embedding(m) is not None for m in (snap or {}).values()])
            return real(ctx, labelset, media_type=media_type, snap=snap, **kw)

        monkeypatch.setattr(labelset_training, "train_from_labelset", _spy)

        autodetect_importer_main(
            "server_folder",
            {"path": folder, "media_type": "audio"},
            settings_path=settings_path,
            exporter_name="server_json_file",
            exporter_field_values={"filepath": out_path},
        )

        assert snaps == [[True] * 4]
