"""The CLI calibrates each detector on the snapshot it is about to score.

``_load_and_train_detectors`` used to hand ``train_from_labelset`` the loaded
medias while ``_score_medias_with_detectors`` scored the converted, re-clipped
and re-embedded ones.  On a natively-typed dataset the detector needs no
re-clip for, those are the same set - which is why it went unnoticed.
Everywhere else the cut was fitted on one population and applied to another
(issue #3647).  These tests pin the two halves that keep them the same
snapshot: the routed memo the training pass calibrates against, and the
scoring pass reusing that same object rather than preparing a second one.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

import vtscore.cli as cli
import vtscore.detectors.converter_routing as cr
from vtscore.cli import _RoutedSnapshots, _detector_group_key, _score_medias_with_detectors

DIM = 4


def _constant_mlp() -> torch.nn.Module:
    mlp = torch.nn.Linear(DIM, 1)
    with torch.no_grad():
        mlp.weight.fill_(1.0)
        mlp.bias.fill_(0.0)
    return mlp


def _stub_embed(monkeypatch) -> None:
    def _fake_embed(medias, name="", on_progress=None):
        vec = np.ones(DIM, dtype=np.float32)
        for m in medias.values():
            m["embeddings"] = {name: vec}
            m["embedder"] = name

    monkeypatch.setattr("vtscore.datasets.stages.embedding.embed_missing", _fake_embed)


class TestRoutedSnapshotsMemo:
    def test_one_route_pass_per_group(self, client, monkeypatch):
        calls: list[tuple] = []

        def _fake_route(medias, target_type, embedder_name, clipper="", clipper_params=None):
            calls.append((target_type, embedder_name, clipper))
            out = {i + 1: dict(m) for i, m in enumerate(medias.values())}
            return out, {i + 1: cid for i, cid in enumerate(medias)}

        monkeypatch.setattr(cr, "route_and_embed", _fake_route)

        medias = {1: {"id": 1, "media_type": "image"}, 2: {"id": 2, "media_type": "image"}}
        routed = _RoutedSnapshots(medias)
        key = _detector_group_key("image", "fake", "", {})

        first = routed.get(key)
        second = routed.get(key)

        assert len(calls) == 1, "a repeated group must not re-route"
        assert second is first, "both passes must see the same objects, not equal copies"

    def test_distinct_groups_route_separately(self, client, monkeypatch):
        calls: list[tuple] = []

        def _fake_route(medias, target_type, embedder_name, clipper="", clipper_params=None):
            calls.append((target_type, embedder_name, clipper))
            return {1: {"id": 1, "media_type": target_type}}, {1: 1}

        monkeypatch.setattr(cr, "route_and_embed", _fake_route)

        routed = _RoutedSnapshots({1: {"id": 1, "media_type": "image"}})
        routed.get(_detector_group_key("image", "fake", "", {}))
        routed.get(_detector_group_key("image", "fake", "tiling", {"duration": "2.0"}))

        assert [c[2] for c in calls] == ["", "tiling"]

    def test_clipper_params_key_is_order_insensitive(self):
        """Two detectors that declare the same params in a different dict order
        are one group, not two - otherwise they'd re-clip the dataset twice."""
        a = _detector_group_key("image", "e", "tiling", {"duration": "2.0", "overlap": "0.5"})
        b = _detector_group_key("image", "e", "tiling", {"overlap": "0.5", "duration": "2.0"})
        assert a == b


class TestScoringReusesTheCalibrationSnapshot:
    def test_scoring_pass_takes_the_memo(self, client, monkeypatch):
        """The first chunk is both trained on and scored; preparing it twice
        would re-decode every video and re-slice every clip."""
        calls: list[str] = []

        def _fake_route(medias, target_type, embedder_name, clipper="", clipper_params=None):
            calls.append(target_type)
            out = {
                i + 1: {**m, "embeddings": {embedder_name: np.ones(DIM, dtype=np.float32)}}
                for i, m in enumerate(medias.values())
            }
            return out, {i + 1: cid for i, cid in enumerate(medias)}

        monkeypatch.setattr(cr, "route_and_embed", _fake_route)
        _stub_embed(monkeypatch)

        medias = {1: {"id": 1, "media_type": "image", "filename": "a.png"}}
        detector_mlps = {
            "det": {"mlp": _constant_mlp(), "threshold": 0.5, "media_type": "image", "embedder": "fake"},
        }

        routed = _RoutedSnapshots(medias)
        routed.get(_detector_group_key("image", "fake", "", {}))
        assert len(calls) == 1

        results = _score_medias_with_detectors(medias, detector_mlps, routed)

        assert len(calls) == 1, "scoring must reuse the prepared snapshot, not re-route"
        assert "det" in results

    def test_without_a_memo_scoring_routes_for_itself(self, client, monkeypatch):
        """Every chunk after the first has no prepared snapshot to inherit."""
        calls: list[str] = []

        def _fake_route(medias, target_type, embedder_name, clipper="", clipper_params=None):
            calls.append(target_type)
            out = {
                i + 1: {**m, "embeddings": {embedder_name: np.ones(DIM, dtype=np.float32)}}
                for i, m in enumerate(medias.values())
            }
            return out, {i + 1: cid for i, cid in enumerate(medias)}

        monkeypatch.setattr(cr, "route_and_embed", _fake_route)
        _stub_embed(monkeypatch)

        medias = {1: {"id": 1, "media_type": "image", "filename": "a.png"}}
        detector_mlps = {
            "det": {"mlp": _constant_mlp(), "threshold": 0.5, "media_type": "image", "embedder": "fake"},
        }

        _score_medias_with_detectors(medias, detector_mlps, None)
        assert calls == ["image"]


class TestGroupKeysCannotDrift:
    def test_scoring_groups_on_the_same_key_the_haystack_was_built_from(self, client, monkeypatch):
        """The whole fix rests on the two passes agreeing on what a group is,
        so both read it off ``_detector_group_key`` rather than spelling it
        out twice."""
        info: dict[str, Any] = {
            "media_type": "image",
            "embedder": "fake",
            "clipper": "tiling",
            "clipper_params": {"duration": "2.0"},
        }
        seen: list[tuple] = []

        def _fake_route(medias, target_type, embedder_name, clipper="", clipper_params=None):
            seen.append(_detector_group_key(target_type, embedder_name, clipper, clipper_params or {}))
            return {}, {}

        monkeypatch.setattr(cr, "route_and_embed", _fake_route)
        _stub_embed(monkeypatch)

        _score_medias_with_detectors(
            {1: {"id": 1, "media_type": "image"}},
            {"det": {**info, "mlp": _constant_mlp(), "threshold": 0.5}},
            None,
        )

        assert seen == [
            _detector_group_key(info["media_type"], info["embedder"], info["clipper"], info["clipper_params"])
        ]


class TestTrainingIsHandedTheRoutedSnapshot:
    def _detector(self, input_spec: dict | None = None) -> dict:
        det = {
            "name": "det",
            "media_type": "image",
            "labelset": {"labels": [{"md5": "a", "label": "good"}, {"md5": "b", "label": "bad"}]},
        }
        if input_spec is not None:
            det["input_spec"] = input_spec
        return det

    def _run(self, monkeypatch, det: dict, medias: dict, routed_out: dict, to_source: dict):
        """Train through ``_load_and_train_detectors``, capturing the haystack."""
        import vtscore.detectors.labelset_training as lt
        import vtscore.detectors.store as store_mod

        monkeypatch.setattr(store_mod, "_read_detector", lambda _p: det)
        monkeypatch.setattr(cr, "route_and_embed", lambda *a, **k: (routed_out, to_source))
        monkeypatch.setattr("vtscore.detectors.input_spec.extract_input_spec_from_medias", lambda _m: None)

        captured: dict[str, Any] = {}

        def _fake_train(det_ctx, labelset, *, media_type, snap, haystack_for=None, on_progress=None):
            det_ctx.embedder = "fake"
            det_ctx.model = _constant_mlp()
            det_ctx.threshold = 0.5
            captured["snap"] = snap
            captured["haystack"] = haystack_for("fake") if haystack_for is not None else None
            return True

        monkeypatch.setattr(lt, "train_from_labelset", _fake_train)
        routed = _RoutedSnapshots(medias)
        cli._load_and_train_detectors(["det"], "image", medias, routed)
        return captured

    def test_haystack_is_the_routed_snapshot_not_the_loaded_medias(self, client, monkeypatch):
        medias = {1: {"id": 1, "media_type": "image"}}
        clips = {1: {"id": 1, "media_type": "image"}, 2: {"id": 2, "media_type": "image"}}
        captured = self._run(
            monkeypatch,
            self._detector({"clipper": "tiling", "clipper_params": {"duration": "2.0"}}),
            medias,
            clips,
            {1: 1, 2: 1},
        )

        # ``snap`` keeps its other job - resolving the labelset's origins - and
        # is still the loaded medias.  The haystack is what scoring will read.
        assert captured["snap"] is medias
        assert captured["haystack"] is not None
        assert captured["haystack"].medias is clips
        assert captured["haystack"].to_source == {1: 1, 2: 1}

    def test_empty_routed_snapshot_falls_back_to_the_snap(self, client, monkeypatch):
        """Nothing converted or embedded: keep the loaded medias rather than
        fitting the population estimator on an empty distribution."""
        medias = {1: {"id": 1, "media_type": "image"}}
        captured = self._run(monkeypatch, self._detector(), medias, {}, {})
        assert captured["haystack"] is None

    def test_no_memo_leaves_the_snap_as_the_haystack(self, client, monkeypatch):
        """A caller with no scoring pass to agree with keeps the old behaviour."""
        import vtscore.detectors.labelset_training as lt
        import vtscore.detectors.store as store_mod

        monkeypatch.setattr(store_mod, "_read_detector", lambda _p: self._detector())
        monkeypatch.setattr("vtscore.detectors.input_spec.extract_input_spec_from_medias", lambda _m: None)
        captured: dict[str, Any] = {}

        def _fake_train(det_ctx, labelset, *, media_type, snap, haystack_for=None, on_progress=None):
            det_ctx.embedder = "fake"
            det_ctx.model = _constant_mlp()
            det_ctx.threshold = 0.5
            captured["haystack"] = haystack_for("fake") if haystack_for is not None else None
            return True

        monkeypatch.setattr(lt, "train_from_labelset", _fake_train)
        cli._load_and_train_detectors(["det"], "image", {1: {"id": 1, "media_type": "image"}})
        assert captured["haystack"] is None
