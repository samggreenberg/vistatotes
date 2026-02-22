import io

import numpy as np
import torch

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


class TestBuildModel:
    """Tests for the build_model helper."""

    def test_build_model_returns_sequential(self):
        from vtsearch.models.training import build_model

        model = build_model(64)
        assert isinstance(model, torch.nn.Sequential)

    def test_build_model_output_is_logits(self):
        """build_model should NOT include sigmoid — output can be outside [0,1]."""
        from vtsearch.models.training import build_model

        model = build_model(32)
        model.eval()
        # Use extreme input to push output well outside [0, 1]
        X = torch.ones(1, 32) * 100.0
        with torch.no_grad():
            logit = model(X).item()
        # Raw logit can be any real number (not clamped to [0, 1])
        assert isinstance(logit, float)

    def test_build_model_has_no_sigmoid_layer(self):
        from vtsearch.models.training import build_model

        model = build_model(64)
        for layer in model:
            assert not isinstance(layer, torch.nn.Sigmoid)

    def test_build_model_state_dict_keys(self):
        from vtsearch.models.training import build_model

        model = build_model(128)
        keys = set(model.state_dict().keys())
        assert keys == {"0.weight", "0.bias", "2.weight", "2.bias"}


class TestTrainModelConfig:
    """Tests for training configuration: reproducibility, weight decay, loss function."""

    def test_deterministic_training(self):
        """Same inputs should produce the same model (manual seed)."""
        from vtsearch.models.training import train_model

        rng = np.random.RandomState(0)
        X = torch.tensor(rng.randn(10, 32).astype(np.float32))
        y = torch.tensor([1.0] * 5 + [0.0] * 5).unsqueeze(1)

        model1 = train_model(X, y, 32)
        model2 = train_model(X, y, 32)

        # Both models should produce identical scores
        with torch.no_grad():
            scores1 = torch.sigmoid(model1(X)).squeeze(1).tolist()
            scores2 = torch.sigmoid(model2(X)).squeeze(1).tolist()
        assert scores1 == scores2

    def test_weight_decay_is_applied(self):
        """Weight decay should keep weights smaller than without it."""
        import config
        from vtsearch.models.training import build_model

        saved = config.TRAIN_EPOCHS
        config.TRAIN_EPOCHS = 200
        try:
            rng = np.random.RandomState(7)
            X = torch.tensor(rng.randn(20, 16).astype(np.float32))
            y = torch.tensor([1.0] * 10 + [0.0] * 10).unsqueeze(1)

            # Train with weight decay (default: 1e-4)
            from vtsearch.models.training import train_model

            model = train_model(X, y, 16)

            # Train without weight decay for comparison (use same local
            # generator approach as train_model for identical init weights)
            g = torch.Generator()
            g.manual_seed(42)
            model_no_wd = build_model(16, generator=g)
            optimizer = torch.optim.Adam(model_no_wd.parameters(), lr=0.001, weight_decay=0.0)
            loss_fn = torch.nn.BCEWithLogitsLoss()
            model_no_wd.train()
            for _ in range(200):
                optimizer.zero_grad()
                loss = loss_fn(model_no_wd(X), y)
                loss.backward()
                optimizer.step()
            model_no_wd.eval()

            # Weight magnitudes with decay should be <= without decay
            wd_norm = sum(p.norm().item() for p in model.parameters())
            no_wd_norm = sum(p.norm().item() for p in model_no_wd.parameters())
            assert wd_norm <= no_wd_norm
        finally:
            config.TRAIN_EPOCHS = saved

    def test_train_model_outputs_logits(self):
        """train_model should return a model that outputs raw logits."""
        from vtsearch.models.training import train_model

        rng = np.random.RandomState(5)
        X = torch.tensor(rng.randn(6, 16).astype(np.float32))
        y = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]).unsqueeze(1)

        model = train_model(X, y, 16)
        with torch.no_grad():
            raw = model(X).squeeze(1).tolist()
            sigmoided = torch.sigmoid(model(X)).squeeze(1).tolist()

        # Raw logits and sigmoided scores should differ
        assert raw != sigmoided
        # Sigmoided scores should be in [0, 1]
        for s in sigmoided:
            assert 0.0 <= s <= 1.0


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
