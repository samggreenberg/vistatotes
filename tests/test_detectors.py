import io
import json

import numpy as np
import pytest

import app as app_module


class TestDetectorExport:
    def test_export_with_sufficient_votes(self, client):
        app_module.good_votes.update({k: None for k in [1, 2, 3]})
        app_module.bad_votes.update({k: None for k in [18, 19, 20]})
        resp = client.post("/api/detector/export")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "weights" in data
        assert "threshold" in data
        assert isinstance(data["weights"], dict)
        assert isinstance(data["threshold"], (int, float))

    def test_export_requires_good_votes(self, client):
        app_module.bad_votes.update({k: None for k in [1, 2]})
        resp = client.post("/api/detector/export")
        assert resp.status_code == 400
        data = resp.get_json()
        assert "need at least one good and one bad vote" in data["error"]

    def test_export_requires_bad_votes(self, client):
        app_module.good_votes.update({k: None for k in [1, 2]})
        resp = client.post("/api/detector/export")
        assert resp.status_code == 400

    def test_export_weights_structure(self, client):
        app_module.good_votes.update({k: None for k in [1, 2]})
        app_module.bad_votes.update({k: None for k in [3, 4]})
        resp = client.post("/api/detector/export")
        data = resp.get_json()
        weights = data["weights"]
        # MLP has 3 layers: Linear(input_dim, 64), ReLU, Linear(64, 1)
        # So we expect 4 keys: 0.weight, 0.bias, 2.weight, 2.bias
        assert "0.weight" in weights
        assert "0.bias" in weights
        assert "2.weight" in weights
        assert "2.bias" in weights


class TestDetectorSort:
    def test_sort_with_valid_detector(self, client):
        # First export a detector
        app_module.good_votes.update({k: None for k in [1, 2]})
        app_module.bad_votes.update({k: None for k in [3, 4]})
        export_resp = client.post("/api/detector/export")
        detector = export_resp.get_json()

        # Now use it to sort
        resp = client.post("/api/detector-sort", json={"detector": detector})
        assert resp.status_code == 200
        data = resp.get_json()
        assert "results" in data
        assert "threshold" in data
        assert len(data["results"]) == app_module.NUM_CLIPS

    def test_sort_results_sorted_descending(self, client):
        app_module.good_votes.update({k: None for k in [1, 2]})
        app_module.bad_votes.update({k: None for k in [3, 4]})
        export_resp = client.post("/api/detector/export")
        detector = export_resp.get_json()

        resp = client.post("/api/detector-sort", json={"detector": detector})
        data = resp.get_json()
        scores = [e["score"] for e in data["results"]]
        assert scores == sorted(scores, reverse=True)

    def test_sort_scores_in_valid_range(self, client):
        app_module.good_votes.update({k: None for k in [1, 2]})
        app_module.bad_votes.update({k: None for k in [3, 4]})
        export_resp = client.post("/api/detector/export")
        detector = export_resp.get_json()

        resp = client.post("/api/detector-sort", json={"detector": detector})
        data = resp.get_json()
        for entry in data["results"]:
            assert 0.0 <= entry["score"] <= 1.0

    def test_sort_missing_detector(self, client):
        resp = client.post("/api/detector-sort", json={})
        assert resp.status_code == 400

    def test_sort_missing_weights(self, client):
        resp = client.post("/api/detector-sort", json={"detector": {"threshold": 0.5}})
        assert resp.status_code == 400

    def test_detector_roundtrip(self, client):
        """Export a detector and verify it produces reasonable scores."""
        app_module.good_votes.update({k: None for k in [1, 2, 3]})
        app_module.bad_votes.update({k: None for k in [18, 19, 20]})

        # Export detector
        export_resp = client.post("/api/detector/export")
        detector = export_resp.get_json()

        # Use detector to sort
        resp = client.post("/api/detector-sort", json={"detector": detector})
        data = resp.get_json()
        score_map = {e["id"]: e["score"] for e in data["results"]}

        # Good clips should score higher than bad clips on average
        avg_good = np.mean([score_map[i] for i in app_module.good_votes])
        avg_bad = np.mean([score_map[i] for i in app_module.bad_votes])
        assert avg_good > avg_bad


class TestFavoriteDetectors:
    """Tests for the favorite-detectors management endpoints."""

    @pytest.fixture(autouse=True)
    def clear_favorites(self):
        from vtsearch.utils.state import favorite_detectors

        favorite_detectors.clear()
        yield
        favorite_detectors.clear()

    def _export_detector(self, client):
        """Helper: vote on some clips and export a valid detector payload."""
        app_module.good_votes.update({k: None for k in [1, 2, 3]})
        app_module.bad_votes.update({k: None for k in [18, 19, 20]})
        resp = client.post("/api/detector/export")
        assert resp.status_code == 200
        return resp.get_json()

    def _post_favorite(self, client, name, detector):
        return client.post(
            "/api/favorite-detectors",
            json={
                "name": name,
                "media_type": "audio",
                "weights": detector["weights"],
                "threshold": detector["threshold"],
            },
        )

    # -- GET list --

    def test_get_empty_list(self, client):
        resp = client.get("/api/favorite-detectors")
        assert resp.status_code == 200
        assert resp.get_json()["detectors"] == []

    def test_get_list_after_add(self, client):
        det = self._export_detector(client)
        self._post_favorite(client, "my-detector", det)

        resp = client.get("/api/favorite-detectors")
        data = resp.get_json()
        assert len(data["detectors"]) == 1
        d = data["detectors"][0]
        assert d["name"] == "my-detector"
        assert d["media_type"] == "audio"
        assert "threshold" in d

    # -- POST add --

    def test_add_detector_returns_success(self, client):
        det = self._export_detector(client)
        resp = self._post_favorite(client, "test-det", det)
        assert resp.status_code == 200
        assert resp.get_json()["success"] is True

    def test_add_missing_name_returns_400(self, client):
        det = self._export_detector(client)
        resp = client.post(
            "/api/favorite-detectors",
            json={"media_type": "audio", "weights": det["weights"]},
        )
        assert resp.status_code == 400

    def test_add_missing_media_type_returns_400(self, client):
        det = self._export_detector(client)
        resp = client.post(
            "/api/favorite-detectors",
            json={"name": "test", "weights": det["weights"]},
        )
        assert resp.status_code == 400

    def test_add_missing_weights_returns_400(self, client):
        resp = client.post(
            "/api/favorite-detectors",
            json={"name": "test", "media_type": "audio"},
        )
        assert resp.status_code == 400

    def test_add_multiple_detectors(self, client):
        det = self._export_detector(client)
        self._post_favorite(client, "det-a", det)
        app_module.good_votes.clear()
        app_module.bad_votes.clear()
        self._post_favorite(client, "det-b", det)

        resp = client.get("/api/favorite-detectors")
        names = {d["name"] for d in resp.get_json()["detectors"]}
        assert names == {"det-a", "det-b"}

    def test_add_overwrites_existing_name(self, client):
        det = self._export_detector(client)
        self._post_favorite(client, "dup", det)
        self._post_favorite(client, "dup", det)

        resp = client.get("/api/favorite-detectors")
        assert len(resp.get_json()["detectors"]) == 1

    # -- DELETE --

    def test_delete_detector(self, client):
        det = self._export_detector(client)
        self._post_favorite(client, "to-delete", det)

        resp = client.delete("/api/favorite-detectors/to-delete")
        assert resp.status_code == 200
        assert resp.get_json()["success"] is True

        resp = client.get("/api/favorite-detectors")
        assert resp.get_json()["detectors"] == []

    def test_delete_nonexistent_returns_404(self, client):
        resp = client.delete("/api/favorite-detectors/does-not-exist")
        assert resp.status_code == 404

    # -- RENAME --

    def test_rename_detector(self, client):
        det = self._export_detector(client)
        self._post_favorite(client, "old-name", det)

        resp = client.put(
            "/api/favorite-detectors/old-name/rename",
            json={"new_name": "new-name"},
        )
        assert resp.status_code == 200
        assert resp.get_json()["new_name"] == "new-name"

        names = [d["name"] for d in client.get("/api/favorite-detectors").get_json()["detectors"]]
        assert "new-name" in names
        assert "old-name" not in names

    def test_rename_nonexistent_returns_400(self, client):
        resp = client.put(
            "/api/favorite-detectors/ghost/rename",
            json={"new_name": "anything"},
        )
        assert resp.status_code == 400

    def test_rename_to_existing_name_returns_400(self, client):
        det = self._export_detector(client)
        self._post_favorite(client, "det-a", det)
        app_module.good_votes.clear()
        app_module.bad_votes.clear()
        self._post_favorite(client, "det-b", det)

        resp = client.put(
            "/api/favorite-detectors/det-a/rename",
            json={"new_name": "det-b"},
        )
        assert resp.status_code == 400

    def test_rename_missing_new_name_returns_400(self, client):
        det = self._export_detector(client)
        self._post_favorite(client, "some-det", det)

        resp = client.put(
            "/api/favorite-detectors/some-det/rename",
            json={},
        )
        assert resp.status_code == 400

    # -- import-pkl (detector JSON file) --

    def test_import_pkl_from_detector_json(self, client):
        det = self._export_detector(client)
        json_bytes = json.dumps(det).encode("utf-8")
        data = {
            "file": (io.BytesIO(json_bytes), "detector.json"),
            "name": "imported",
        }
        resp = client.post(
            "/api/favorite-detectors/import-pkl",
            data=data,
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200
        result = resp.get_json()
        assert result["success"] is True
        assert result["name"] == "imported"

    def test_import_pkl_uses_filename_stem_as_default_name(self, client):
        det = self._export_detector(client)
        json_bytes = json.dumps(det).encode("utf-8")
        data = {"file": (io.BytesIO(json_bytes), "my_detector.json")}
        resp = client.post(
            "/api/favorite-detectors/import-pkl",
            data=data,
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200
        assert resp.get_json()["name"] == "my_detector"

    def test_import_pkl_preserves_media_type_from_file(self, client):
        det = self._export_detector(client)
        # Embed explicit media_type in the "file" payload
        det["media_type"] = "image"
        json_bytes = json.dumps(det).encode("utf-8")
        data = {
            "file": (io.BytesIO(json_bytes), "image_detector.json"),
            "name": "img-det",
        }
        resp = client.post(
            "/api/favorite-detectors/import-pkl",
            data=data,
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200
        assert resp.get_json()["media_type"] == "image"

    def test_import_pkl_no_file_returns_400(self, client):
        resp = client.post("/api/favorite-detectors/import-pkl", data={})
        assert resp.status_code == 400

    def test_import_pkl_invalid_format_returns_400(self, client):
        data = {"file": (io.BytesIO(b'{"not_a_detector": true}'), "bad.json")}
        resp = client.post(
            "/api/favorite-detectors/import-pkl",
            data=data,
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400

    # -- Detector data is stored correctly --

    def test_stored_detector_has_correct_fields(self, client):
        det = self._export_detector(client)
        self._post_favorite(client, "field-check", det)

        from vtsearch.utils.state import favorite_detectors

        assert "field-check" in favorite_detectors
        stored = favorite_detectors["field-check"]
        assert stored["name"] == "field-check"
        assert stored["media_type"] == "audio"
        assert "weights" in stored
        assert "threshold" in stored
        assert "created_at" in stored


class TestAutoDetect:
    """Tests for POST /api/auto-detect."""

    @pytest.fixture(autouse=True)
    def clear_favorites(self):
        from vtsearch.utils.state import favorite_detectors

        favorite_detectors.clear()
        yield
        favorite_detectors.clear()

    def _add_audio_detector(self, client, name="test-detector"):
        """Helper: create and save an audio detector."""
        app_module.good_votes.update({k: None for k in [1, 2, 3]})
        app_module.bad_votes.update({k: None for k in [18, 19, 20]})
        export_resp = client.post("/api/detector/export")
        assert export_resp.status_code == 200
        detector = export_resp.get_json()

        save_resp = client.post(
            "/api/favorite-detectors",
            json={
                "name": name,
                "media_type": "audio",
                "weights": detector["weights"],
                "threshold": detector["threshold"],
            },
        )
        assert save_resp.status_code == 200
        app_module.good_votes.clear()
        app_module.bad_votes.clear()

    # -- no matching detectors --

    def test_no_favorites_returns_400(self, client):
        resp = client.post("/api/auto-detect")
        assert resp.status_code == 400
        assert "No favorite detectors" in resp.get_json()["error"]

    def test_no_matching_media_type_returns_400(self, client):
        """A detector for a different media type should not match audio clips."""
        app_module.good_votes.update({k: None for k in [1, 2, 3]})
        app_module.bad_votes.update({k: None for k in [18, 19, 20]})
        export_resp = client.post("/api/detector/export")
        detector = export_resp.get_json()
        app_module.good_votes.clear()
        app_module.bad_votes.clear()

        # Save as "image" type — clips are audio, so it won't match
        client.post(
            "/api/favorite-detectors",
            json={
                "name": "image-detector",
                "media_type": "image",
                "weights": detector["weights"],
                "threshold": detector["threshold"],
            },
        )
        resp = client.post("/api/auto-detect")
        assert resp.status_code == 400

    # -- basic success --

    def test_returns_200_with_matching_detector(self, client):
        self._add_audio_detector(client)
        resp = client.post("/api/auto-detect")
        assert resp.status_code == 200

    def test_response_has_required_top_level_fields(self, client):
        self._add_audio_detector(client)
        data = client.post("/api/auto-detect").get_json()
        assert "media_type" in data
        assert "detectors_run" in data
        assert "results" in data

    def test_media_type_matches_clips(self, client):
        self._add_audio_detector(client)
        data = client.post("/api/auto-detect").get_json()
        assert data["media_type"] == "audio"

    def test_detectors_run_count(self, client):
        self._add_audio_detector(client, name="det-1")
        self._add_audio_detector(client, name="det-2")
        data = client.post("/api/auto-detect").get_json()
        assert data["detectors_run"] == 2

    # -- per-detector result structure --

    def test_each_result_has_required_fields(self, client):
        self._add_audio_detector(client, name="struct-check")
        data = client.post("/api/auto-detect").get_json()
        result = data["results"]["struct-check"]
        assert "detector_name" in result
        assert "threshold" in result
        assert "total_hits" in result
        assert "hits" in result

    def test_detector_name_matches_key(self, client):
        self._add_audio_detector(client, name="named-detector")
        data = client.post("/api/auto-detect").get_json()
        result = data["results"]["named-detector"]
        assert result["detector_name"] == "named-detector"

    def test_total_hits_matches_hits_length(self, client):
        self._add_audio_detector(client)
        data = client.post("/api/auto-detect").get_json()
        for result in data["results"].values():
            assert result["total_hits"] == len(result["hits"])

    # -- hit data safety --

    def test_hits_do_not_contain_embeddings(self, client):
        self._add_audio_detector(client)
        data = client.post("/api/auto-detect").get_json()
        for result in data["results"].values():
            for hit in result["hits"]:
                assert "embedding" not in hit

    def test_hits_do_not_contain_clip_bytes(self, client):
        self._add_audio_detector(client)
        data = client.post("/api/auto-detect").get_json()
        for result in data["results"].values():
            for hit in result["hits"]:
                assert "clip_bytes" not in hit

    def test_hits_contain_score(self, client):
        self._add_audio_detector(client)
        data = client.post("/api/auto-detect").get_json()
        for result in data["results"].values():
            for hit in result["hits"]:
                assert "score" in hit
                assert 0.0 <= hit["score"] <= 1.0

    def test_hits_sorted_descending_by_score(self, client):
        self._add_audio_detector(client)
        data = client.post("/api/auto-detect").get_json()
        for result in data["results"].values():
            scores = [h["score"] for h in result["hits"]]
            assert scores == sorted(scores, reverse=True)

    # -- threshold correctness --

    def test_all_hits_score_at_or_above_threshold(self, client):
        self._add_audio_detector(client)
        data = client.post("/api/auto-detect").get_json()
        for result in data["results"].values():
            threshold = result["threshold"]
            for hit in result["hits"]:
                assert hit["score"] >= threshold - 1e-6  # float tolerance
