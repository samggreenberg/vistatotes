import io

import numpy as np

import app as app_module


class TestSortClips:
    def test_returns_all_clips(self, client):
        resp = client.post("/api/sort", json={"text": "high pitched beep"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data["results"]) == app_module.NUM_CLIPS
        assert "threshold" in data

    def test_result_contains_id_and_similarity(self, client):
        resp = client.post("/api/sort", json={"text": "low tone"})
        data = resp.get_json()
        for entry in data["results"]:
            assert "id" in entry
            assert "similarity" in entry

    def test_sorted_by_descending_similarity(self, client):
        resp = client.post("/api/sort", json={"text": "a beeping sound"})
        data = resp.get_json()
        similarities = [e["similarity"] for e in data["results"]]
        assert similarities == sorted(similarities, reverse=True)

    def test_all_clip_ids_present(self, client):
        resp = client.post("/api/sort", json={"text": "sine wave"})
        data = resp.get_json()
        ids = {e["id"] for e in data["results"]}
        assert ids == set(range(1, app_module.NUM_CLIPS + 1))

    def test_similarity_values_in_range(self, client):
        resp = client.post("/api/sort", json={"text": "high pitch"})
        data = resp.get_json()
        for entry in data["results"]:
            assert -1.0 <= entry["similarity"] <= 1.0

    def test_empty_text_returns_400(self, client):
        resp = client.post("/api/sort", json={"text": ""})
        assert resp.status_code == 400

    def test_missing_text_returns_400(self, client):
        resp = client.post("/api/sort", json={"other": "field"})
        assert resp.status_code == 400

    def test_whitespace_only_returns_400(self, client):
        resp = client.post("/api/sort", json={"text": "   "})
        assert resp.status_code == 400


class TestTrainAndScore:
    def test_returns_list_of_scored_clips(self):
        app_module.good_votes.update({k: None for k in [1, 2]})
        app_module.bad_votes.update({k: None for k in [3, 4]})
        results, threshold = app_module.train_and_score(app_module.clips, app_module.good_votes, app_module.bad_votes)
        assert len(results) == app_module.NUM_CLIPS
        assert isinstance(threshold, float)
        for entry in results:
            assert "id" in entry
            assert "score" in entry

    def test_scores_between_zero_and_one(self):
        app_module.good_votes.update({k: None for k in [1, 2]})
        app_module.bad_votes.update({k: None for k in [3, 4]})
        results, threshold = app_module.train_and_score(app_module.clips, app_module.good_votes, app_module.bad_votes)
        for entry in results:
            assert 0.0 <= entry["score"] <= 1.0

    def test_results_sorted_descending(self):
        app_module.good_votes.update({k: None for k in [1, 2]})
        app_module.bad_votes.update({k: None for k in [3, 4]})
        results, threshold = app_module.train_and_score(app_module.clips, app_module.good_votes, app_module.bad_votes)
        scores = [e["score"] for e in results]
        assert scores == sorted(scores, reverse=True)

    def test_good_clips_scored_higher_than_bad(self):
        app_module.good_votes.update({k: None for k in [1, 2, 3]})
        app_module.bad_votes.update({k: None for k in [18, 19, 20]})
        results, threshold = app_module.train_and_score(app_module.clips, app_module.good_votes, app_module.bad_votes)
        score_map = {e["id"]: e["score"] for e in results}
        avg_good = np.mean([score_map[i] for i in app_module.good_votes])
        avg_bad = np.mean([score_map[i] for i in app_module.bad_votes])
        assert avg_good > avg_bad

    def test_order_changes_after_new_vote(self):
        """After adding a vote and retraining, the sort order should change."""
        app_module.good_votes.update({k: None for k in [1, 2]})
        app_module.bad_votes.update({k: None for k in [19, 20]})
        results_before, _ = app_module.train_and_score(
            app_module.clips, app_module.good_votes, app_module.bad_votes
        )
        order_before = [e["id"] for e in results_before]

        # Add a new good vote on a clip that was in the middle
        app_module.good_votes[10] = None
        results_after, _ = app_module.train_and_score(
            app_module.clips, app_module.good_votes, app_module.bad_votes
        )
        order_after = [e["id"] for e in results_after]

        assert order_before != order_after, (
            "Sort order did not change after adding a new vote"
        )


class TestLearnedSort:
    def test_returns_all_clips(self, client):
        app_module.good_votes.update({k: None for k in [1, 2]})
        app_module.bad_votes.update({k: None for k in [3, 4]})
        resp = client.post("/api/learned-sort")
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data["results"]) == app_module.NUM_CLIPS
        assert "threshold" in data

    def test_result_fields(self, client):
        app_module.good_votes.update({k: None for k in [1, 2]})
        app_module.bad_votes.update({k: None for k in [3, 4]})
        resp = client.post("/api/learned-sort")
        data = resp.get_json()
        for entry in data["results"]:
            assert "id" in entry
            assert "score" in entry

    def test_sorted_descending(self, client):
        app_module.good_votes.update({k: None for k in [1, 2]})
        app_module.bad_votes.update({k: None for k in [3, 4]})
        resp = client.post("/api/learned-sort")
        data = resp.get_json()
        scores = [e["score"] for e in data["results"]]
        assert scores == sorted(scores, reverse=True)

    def test_all_clip_ids_present(self, client):
        app_module.good_votes.update({k: None for k in [1, 2]})
        app_module.bad_votes.update({k: None for k in [3, 4]})
        resp = client.post("/api/learned-sort")
        data = resp.get_json()
        ids = {e["id"] for e in data["results"]}
        assert ids == set(range(1, app_module.NUM_CLIPS + 1))

    def test_only_good_votes_returns_400(self, client):
        app_module.good_votes.update({k: None for k in [1, 2]})
        resp = client.post("/api/learned-sort")
        assert resp.status_code == 400

    def test_only_bad_votes_returns_400(self, client):
        app_module.bad_votes.update({k: None for k in [3, 4]})
        resp = client.post("/api/learned-sort")
        assert resp.status_code == 400

    def test_scores_in_valid_range(self, client):
        app_module.good_votes.update({k: None for k in [1, 2]})
        app_module.bad_votes.update({k: None for k in [3, 4]})
        resp = client.post("/api/learned-sort")
        data = resp.get_json()
        for entry in data["results"]:
            assert 0.0 <= entry["score"] <= 1.0


class TestExampleSort:
    def test_sort_with_audio_file(self, client):
        # Create a test WAV file in memory
        wav_bytes = app_module.generate_wav(440.0, 1.0)
        data = {"file": (io.BytesIO(wav_bytes), "test.wav")}

        resp = client.post("/api/example-sort", data=data, content_type="multipart/form-data")
        assert resp.status_code == 200
        result_data = resp.get_json()
        assert "results" in result_data
        assert "threshold" in result_data
        assert len(result_data["results"]) == app_module.NUM_CLIPS

    def test_sort_results_sorted_descending(self, client):
        wav_bytes = app_module.generate_wav(440.0, 1.0)
        data = {"file": (io.BytesIO(wav_bytes), "test.wav")}

        resp = client.post("/api/example-sort", data=data, content_type="multipart/form-data")
        result_data = resp.get_json()
        similarities = [e["similarity"] for e in result_data["results"]]
        assert similarities == sorted(similarities, reverse=True)

    def test_sort_similarity_in_valid_range(self, client):
        wav_bytes = app_module.generate_wav(440.0, 1.0)
        data = {"file": (io.BytesIO(wav_bytes), "test.wav")}

        resp = client.post("/api/example-sort", data=data, content_type="multipart/form-data")
        result_data = resp.get_json()
        for entry in result_data["results"]:
            assert -1.0 <= entry["similarity"] <= 1.0

    def test_sort_no_file(self, client):
        resp = client.post("/api/example-sort", data={})
        assert resp.status_code == 400

    def test_sort_empty_filename(self, client):
        data = {"file": (io.BytesIO(b""), "")}
        resp = client.post("/api/example-sort", data=data, content_type="multipart/form-data")
        assert resp.status_code == 400
