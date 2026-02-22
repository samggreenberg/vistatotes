"""Tests for processor versioning.

Covers:
- Version auto-increment when re-adding a detector with the same name
- embedding_dim computed from weights
- parent_version tracking
- training_samples passthrough
- Compatibility checking in detector-sort and auto-detect
- Version metadata in detector export
- Round-trip through detector_file importer
"""

from __future__ import annotations

import io
import json

import pytest

import app as app_module  # noqa: F401 — triggers conftest clip init


# ---------------------------------------------------------------------------
# Version tracking in add_favorite_detector
# ---------------------------------------------------------------------------


class TestVersionTracking:
    def setup_method(self):
        from vtsearch.utils import favorite_detectors

        self._saved = dict(favorite_detectors)

    def teardown_method(self):
        from vtsearch.utils import favorite_detectors

        favorite_detectors.clear()
        favorite_detectors.update(self._saved)

    def test_first_detector_gets_version_1(self):
        from vtsearch.utils import add_favorite_detector, favorite_detectors

        add_favorite_detector("v_test", "audio", {"0.weight": [[1.0, 2.0]], "0.bias": [0.5]}, 0.5)
        det = favorite_detectors["v_test"]
        assert det["version"] == 1
        assert det["parent_version"] is None

    def test_re_add_increments_version(self):
        from vtsearch.utils import add_favorite_detector, favorite_detectors

        add_favorite_detector("v_test", "audio", {"0.weight": [[1.0, 2.0]], "0.bias": [0.5]}, 0.5)
        assert favorite_detectors["v_test"]["version"] == 1

        add_favorite_detector("v_test", "audio", {"0.weight": [[3.0, 4.0]], "0.bias": [0.6]}, 0.6)
        assert favorite_detectors["v_test"]["version"] == 2
        assert favorite_detectors["v_test"]["parent_version"] == 1

    def test_multiple_re_adds_increment_sequentially(self):
        from vtsearch.utils import add_favorite_detector, favorite_detectors

        for i in range(5):
            add_favorite_detector("v_test", "audio", {"0.weight": [[float(i)]], "0.bias": [0.0]}, 0.5)

        assert favorite_detectors["v_test"]["version"] == 5
        assert favorite_detectors["v_test"]["parent_version"] == 4

    def test_different_names_get_independent_versions(self):
        from vtsearch.utils import add_favorite_detector, favorite_detectors

        add_favorite_detector("det_a", "audio", {"0.weight": [[1.0]], "0.bias": [0.5]}, 0.5)
        add_favorite_detector("det_b", "audio", {"0.weight": [[2.0]], "0.bias": [0.5]}, 0.5)

        assert favorite_detectors["det_a"]["version"] == 1
        assert favorite_detectors["det_b"]["version"] == 1

        add_favorite_detector("det_a", "audio", {"0.weight": [[3.0]], "0.bias": [0.6]}, 0.6)
        assert favorite_detectors["det_a"]["version"] == 2
        assert favorite_detectors["det_b"]["version"] == 1


# ---------------------------------------------------------------------------
# embedding_dim computed from weights
# ---------------------------------------------------------------------------


class TestEmbeddingDim:
    def setup_method(self):
        from vtsearch.utils import favorite_detectors

        self._saved = dict(favorite_detectors)

    def teardown_method(self):
        from vtsearch.utils import favorite_detectors

        favorite_detectors.clear()
        favorite_detectors.update(self._saved)

    def test_embedding_dim_computed_from_weights(self):
        from vtsearch.utils import add_favorite_detector, favorite_detectors

        weights = {"0.weight": [[1.0, 2.0, 3.0]], "0.bias": [0.5]}
        add_favorite_detector("dim_test", "audio", weights, 0.5)
        assert favorite_detectors["dim_test"]["embedding_dim"] == 3

    def test_embedding_dim_none_for_empty_weights(self):
        from vtsearch.utils import add_favorite_detector, favorite_detectors

        add_favorite_detector("dim_test", "audio", {}, 0.5)
        assert favorite_detectors["dim_test"]["embedding_dim"] is None

    def test_embedding_dim_single_feature(self):
        from vtsearch.utils import add_favorite_detector, favorite_detectors

        weights = {"0.weight": [[42.0]], "0.bias": [0.1]}
        add_favorite_detector("dim_test", "audio", weights, 0.5)
        assert favorite_detectors["dim_test"]["embedding_dim"] == 1


# ---------------------------------------------------------------------------
# training_samples passthrough
# ---------------------------------------------------------------------------


class TestTrainingSamples:
    def setup_method(self):
        from vtsearch.utils import favorite_detectors

        self._saved = dict(favorite_detectors)

    def teardown_method(self):
        from vtsearch.utils import favorite_detectors

        favorite_detectors.clear()
        favorite_detectors.update(self._saved)

    def test_training_samples_stored(self):
        from vtsearch.utils import add_favorite_detector, favorite_detectors

        add_favorite_detector("ts_test", "audio", {"0.weight": [[1.0]], "0.bias": [0.5]}, 0.5, training_samples=42)
        assert favorite_detectors["ts_test"]["training_samples"] == 42

    def test_training_samples_default_none(self):
        from vtsearch.utils import add_favorite_detector, favorite_detectors

        add_favorite_detector("ts_test", "audio", {"0.weight": [[1.0]], "0.bias": [0.5]}, 0.5)
        assert favorite_detectors["ts_test"]["training_samples"] is None


# ---------------------------------------------------------------------------
# _embedding_dim_from_weights helper
# ---------------------------------------------------------------------------


class TestEmbeddingDimFromWeights:
    def test_normal_weights(self):
        from vtsearch.utils.state import _embedding_dim_from_weights

        assert _embedding_dim_from_weights({"0.weight": [[1, 2, 3], [4, 5, 6]]}) == 3

    def test_empty_dict(self):
        from vtsearch.utils.state import _embedding_dim_from_weights

        assert _embedding_dim_from_weights({}) is None

    def test_missing_first_layer(self):
        from vtsearch.utils.state import _embedding_dim_from_weights

        assert _embedding_dim_from_weights({"2.weight": [[1, 2]]}) is None

    def test_non_list_first_layer(self):
        from vtsearch.utils.state import _embedding_dim_from_weights

        assert _embedding_dim_from_weights({"0.weight": "not a list"}) is None


# ---------------------------------------------------------------------------
# Rename preserves version
# ---------------------------------------------------------------------------


class TestRenamePreservesVersion:
    def setup_method(self):
        from vtsearch.utils import favorite_detectors

        self._saved = dict(favorite_detectors)

    def teardown_method(self):
        from vtsearch.utils import favorite_detectors

        favorite_detectors.clear()
        favorite_detectors.update(self._saved)

    def test_rename_preserves_version_fields(self):
        from vtsearch.utils import add_favorite_detector, favorite_detectors, rename_favorite_detector

        add_favorite_detector("old", "audio", {"0.weight": [[1.0, 2.0]], "0.bias": [0.5]}, 0.5, training_samples=10)
        add_favorite_detector("old", "audio", {"0.weight": [[3.0, 4.0]], "0.bias": [0.6]}, 0.6, training_samples=20)
        assert favorite_detectors["old"]["version"] == 2

        rename_favorite_detector("old", "new")
        assert "old" not in favorite_detectors
        det = favorite_detectors["new"]
        assert det["version"] == 2
        assert det["parent_version"] == 1
        assert det["embedding_dim"] == 2
        assert det["training_samples"] == 20


# ---------------------------------------------------------------------------
# Detector export includes version metadata
# ---------------------------------------------------------------------------


class TestDetectorExportVersionInfo:
    def test_export_includes_training_samples_and_embedding_dim(self, client):
        from vtsearch.utils import bad_votes, clips, good_votes

        # Vote on clips
        clip_ids = sorted(clips.keys())
        if len(clip_ids) < 2:
            pytest.skip("Need at least 2 clips")
        good_votes[clip_ids[0]] = None
        bad_votes[clip_ids[1]] = None

        res = client.post("/api/detector/export")
        assert res.status_code == 200
        data = res.get_json()
        assert "training_samples" in data
        assert data["training_samples"] == 2
        assert "embedding_dim" in data
        assert isinstance(data["embedding_dim"], int)
        assert data["embedding_dim"] > 0


# ---------------------------------------------------------------------------
# Detector-sort embedding dimension compatibility check
# ---------------------------------------------------------------------------


class TestDetectorSortCompatibility:
    def test_mismatched_embedding_dim_returns_400(self, client):
        # Create weights with dimension 2 (clips have higher dimension)
        detector = {
            "weights": {"0.weight": [[1.0, 2.0]], "0.bias": [0.5], "2.weight": [[0.5, 0.5]], "2.bias": [0.1]},
            "threshold": 0.5,
        }
        res = client.post("/api/detector-sort", json={"detector": detector})
        assert res.status_code == 400
        assert "dimension" in res.get_json()["error"].lower()


# ---------------------------------------------------------------------------
# Detector file importer preserves version metadata
# ---------------------------------------------------------------------------


class TestDetectorFileVersionRoundTrip:
    def _get_importer(self):
        from vtsearch.processors.importers.detector_file import PROCESSOR_IMPORTER

        return PROCESSOR_IMPORTER

    def test_preserves_training_samples(self):
        from werkzeug.datastructures import FileStorage

        payload = {
            "weights": {"0.weight": [[1.0, 2.0]], "0.bias": [0.5]},
            "threshold": 0.75,
            "media_type": "image",
            "training_samples": 100,
        }
        raw = json.dumps(payload).encode()
        fs = FileStorage(stream=io.BytesIO(raw), filename="detector.json")
        result = self._get_importer().run({"file": fs})
        assert result["training_samples"] == 100

    def test_preserves_embedding_dim(self):
        from werkzeug.datastructures import FileStorage

        payload = {
            "weights": {"0.weight": [[1.0, 2.0, 3.0]], "0.bias": [0.5]},
            "threshold": 0.75,
            "embedding_dim": 3,
        }
        raw = json.dumps(payload).encode()
        fs = FileStorage(stream=io.BytesIO(raw), filename="detector.json")
        result = self._get_importer().run({"file": fs})
        assert result["embedding_dim"] == 3

    def test_missing_version_fields_not_in_result(self):
        from werkzeug.datastructures import FileStorage

        payload = {
            "weights": {"0.weight": [[1.0]], "0.bias": [0.5]},
            "threshold": 0.5,
        }
        raw = json.dumps(payload).encode()
        fs = FileStorage(stream=io.BytesIO(raw), filename="detector.json")
        result = self._get_importer().run({"file": fs})
        # When not in file, these keys should not be in the result
        assert "training_samples" not in result
        assert "embedding_dim" not in result

    def test_cli_preserves_version_fields(self, tmp_path):
        payload = {
            "weights": {"0.weight": [[1.0, 2.0]], "0.bias": [0.1]},
            "threshold": 0.6,
            "training_samples": 50,
            "embedding_dim": 2,
        }
        p = tmp_path / "detector.json"
        p.write_text(json.dumps(payload))
        result = self._get_importer().run_cli({"file": str(p)})
        assert result["training_samples"] == 50
        assert result["embedding_dim"] == 2


# ---------------------------------------------------------------------------
# Processor importer route passes training_samples
# ---------------------------------------------------------------------------


class TestProcessorImporterVersioning:
    def setup_method(self):
        from vtsearch.utils import favorite_detectors

        self._saved = dict(favorite_detectors)

    def teardown_method(self):
        from vtsearch.utils import favorite_detectors

        favorite_detectors.clear()
        favorite_detectors.update(self._saved)

    def test_import_sets_version_and_training_samples(self, client):
        from vtsearch.utils import favorite_detectors

        payload = {
            "weights": {"0.weight": [[1.0, 2.0]], "0.bias": [0.5]},
            "threshold": 0.75,
            "media_type": "image",
            "training_samples": 42,
        }
        raw = json.dumps(payload).encode()
        data = {
            "file": (io.BytesIO(raw), "detector.json"),
            "name": "ver_import_test",
        }
        res = client.post(
            "/api/processor-importers/import/detector_file",
            data=data,
            content_type="multipart/form-data",
        )
        assert res.status_code == 200

        det = favorite_detectors["ver_import_test"]
        assert det["version"] == 1
        assert det["parent_version"] is None
        assert det["embedding_dim"] == 2
        assert det["training_samples"] == 42

    def test_re_import_increments_version(self, client):
        from vtsearch.utils import favorite_detectors

        for i in range(3):
            payload = {
                "weights": {"0.weight": [[float(i), float(i + 1)]], "0.bias": [0.5]},
                "threshold": 0.5,
                "media_type": "audio",
            }
            raw = json.dumps(payload).encode()
            data = {
                "file": (io.BytesIO(raw), "detector.json"),
                "name": "ver_reimport_test",
            }
            res = client.post(
                "/api/processor-importers/import/detector_file",
                data=data,
                content_type="multipart/form-data",
            )
            assert res.status_code == 200

        det = favorite_detectors["ver_reimport_test"]
        assert det["version"] == 3
        assert det["parent_version"] == 2


# ---------------------------------------------------------------------------
# GET /api/favorite-detectors includes version fields
# ---------------------------------------------------------------------------


class TestFavoriteDetectorsVersionInResponse:
    def setup_method(self):
        from vtsearch.utils import favorite_detectors

        self._saved = dict(favorite_detectors)

    def teardown_method(self):
        from vtsearch.utils import favorite_detectors

        favorite_detectors.clear()
        favorite_detectors.update(self._saved)

    def test_version_fields_in_response(self, client):
        from vtsearch.utils import add_favorite_detector

        add_favorite_detector(
            "api_ver_test", "audio", {"0.weight": [[1.0, 2.0]], "0.bias": [0.5]}, 0.5, training_samples=10
        )

        res = client.get("/api/favorite-detectors")
        assert res.status_code == 200
        detectors = res.get_json()["detectors"]
        det = next(d for d in detectors if d["name"] == "api_ver_test")
        assert det["version"] == 1
        assert det["parent_version"] is None
        assert det["embedding_dim"] == 2
        assert det["training_samples"] == 10
