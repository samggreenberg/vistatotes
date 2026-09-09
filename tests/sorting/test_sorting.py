import io
import unittest.mock

import numpy as np
import torch


class TestSortClips:
    def test_returns_all_clips(self, client):
        resp = client.post("/api/sort", json={"text": "high pitched beep"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data["results"]) == NUM_MEDIAS
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

    def test_all_media_ids_present(self, client):
        resp = client.post("/api/sort", json={"text": "sine wave"})
        data = resp.get_json()
        ids = {e["id"] for e in data["results"]}
        assert ids == set(range(1, NUM_MEDIAS + 1))

    def test_similarity_values_in_range(self, client):
        resp = client.post("/api/sort", json={"text": "high pitch"})
        data = resp.get_json()
        for entry in data["results"]:
            assert -1.0 <= entry["similarity"] <= 1.0

    def test_empty_text_returns_400(self, client):
        resp = client.post("/api/sort", json={"text": ""})
        assert resp.status_code == 400

    def test_missing_text_returns_422(self, client):
        # Schema-level validation: the marshmallow ``SortRequestSchema``
        # rejects requests without a ``text`` key as 422 with the
        # standard ``errors`` envelope.
        resp = client.post("/api/sort", json={"other": "field"})
        assert resp.status_code == 422
        assert "text" in resp.get_json()["errors"]["json"]

    def test_whitespace_only_returns_400(self, client):
        resp = client.post("/api/sort", json={"text": "   "})
        assert resp.status_code == 400


class TestTrainAndScore:
    def test_returns_list_of_scored_clips(self):
        good_votes.update({k: None for k in [1, 2]})
        bad_votes.update({k: None for k in [3, 4]})
        results, threshold, _model = train_and_score(medias, good_votes, bad_votes)
        assert len(results) == NUM_MEDIAS
        assert isinstance(threshold, float)
        for entry in results:
            assert "id" in entry
            assert "score" in entry

    def test_scores_between_zero_and_one(self):
        good_votes.update({k: None for k in [1, 2]})
        bad_votes.update({k: None for k in [3, 4]})
        results, threshold, _model = train_and_score(medias, good_votes, bad_votes)
        for entry in results:
            assert 0.0 <= entry["score"] <= 1.0

    def test_results_sorted_descending(self):
        good_votes.update({k: None for k in [1, 2]})
        bad_votes.update({k: None for k in [3, 4]})
        results, threshold, _model = train_and_score(medias, good_votes, bad_votes)
        scores = [e["score"] for e in results]
        assert scores == sorted(scores, reverse=True)

    def test_good_clips_scored_higher_than_bad(self):
        good_votes.update({k: None for k in [1, 2, 3]})
        bad_votes.update({k: None for k in [18, 19, 20]})
        results, threshold, _model = train_and_score(medias, good_votes, bad_votes)
        score_map = {e["id"]: e["score"] for e in results}
        avg_good = np.mean([score_map[i] for i in good_votes])
        avg_bad = np.mean([score_map[i] for i in bad_votes])
        assert avg_good > avg_bad

    def test_trains_the_linear_head(self):
        """The vote path's detector is the linear (logistic) head, not the MLP.

        Pins the #2790 swap on the pipeline users actually drive: a revert to
        ``_auto_hidden_dim`` would put back a ReLU and a second Linear here.
        """
        good_votes.update({k: None for k in [1, 2, 3]})
        bad_votes.update({k: None for k in [18, 19, 20]})
        _results, _threshold, model = train_and_score(medias, good_votes, bad_votes)
        assert model is not None
        assert [type(layer) for layer in model] == [torch.nn.Linear]
        assert set(model.state_dict()) == {"0.weight", "0.bias"}

    def test_order_changes_after_new_vote(self):
        """After adding a vote and retraining, the sort order should change."""
        good_votes.update({k: None for k in [1, 2, 3, 4, 5]})
        bad_votes.update({k: None for k in [16, 17, 18, 19, 20]})
        results_before, _, _m = train_and_score(medias, good_votes, bad_votes)
        order_before = [e["id"] for e in results_before]

        # Add a new good vote on a media that was in the middle
        good_votes[10] = None
        results_after, _, _m = train_and_score(medias, good_votes, bad_votes)
        order_after = [e["id"] for e in results_after]

        assert order_before != order_after, "Sort order did not change after adding a new vote"


class TestBuildModel:
    """Tests for the build_model helper."""

    def test_build_model_returns_sequential(self):
        from vtscore.training.mlp import build_model

        model = build_model(64)
        assert isinstance(model, torch.nn.Sequential)

    def test_build_model_output_is_logits(self):
        """build_model should NOT include sigmoid; output can be outside [0,1]."""
        from vtscore.training.mlp import build_model

        # Use a seeded generator so the random weights are deterministic;
        # without this, the test is flaky because random initialisation can
        # occasionally produce weights that map extreme input into [0, 1].
        gen = torch.Generator().manual_seed(42)
        model = build_model(32, generator=gen)
        model.eval()
        # Use extreme input to push output well outside [0, 1]
        X = torch.ones(1, 32) * 100.0
        with torch.no_grad():
            logit = model(X).item()
        # Raw logit is unbounded; with extreme input it should land outside [0, 1]
        assert isinstance(logit, float)
        assert logit < 0.0 or logit > 1.0, f"Expected unbounded logit outside [0,1] with extreme input, got {logit}"

    def test_build_model_has_no_sigmoid_layer(self):
        from vtscore.training.mlp import build_model

        model = build_model(64)
        for layer in model:
            assert not isinstance(layer, torch.nn.Sigmoid)

    def test_build_model_state_dict_keys(self):
        from vtscore.training.mlp import build_model

        model = build_model(128)
        keys = set(model.state_dict().keys())
        # 4 layers: Linear(0), ReLU(1), Dropout(2), Linear(3)
        assert keys == {"0.weight", "0.bias", "3.weight", "3.bias"}


class TestTrainModelConfig:
    """Tests for training configuration: reproducibility, weight decay, loss function."""

    def test_deterministic_training(self):
        """Same inputs should produce the same model (manual seed)."""
        from vtscore.training.mlp import train_model

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
        import vtscore.config as config
        from vtscore.training.mlp import _auto_hidden_dim, build_model

        saved = config.TRAIN_EPOCHS
        config.TRAIN_EPOCHS = 200
        try:
            rng = np.random.RandomState(7)
            X = torch.tensor(rng.randn(20, 16).astype(np.float32))
            y = torch.tensor([1.0] * 10 + [0.0] * 10).unsqueeze(1)

            # Train with weight decay (default: 1e-4)
            from vtscore.training.mlp import train_model

            model = train_model(X, y, 16)

            # Train without weight decay for comparison (use same local
            # generator approach as train_model for identical init weights)
            hidden_dim = _auto_hidden_dim(len(X))
            g = torch.Generator()
            g.manual_seed(42)
            model_no_wd = build_model(16, hidden_dim=hidden_dim, dropout=config.MLP_DROPOUT, generator=g)
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
        from vtscore.training.mlp import train_model

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


class TestTrainModelEpochs:
    """Tests for env-tunable epochs and early-stopping on loss plateau."""

    @staticmethod
    def _count_optimizer_steps(fn):
        """Run ``fn`` and return the number of ``Adam.step`` invocations.

        The training loop calls ``optimizer.step()`` exactly once per epoch,
        so this counter equals the number of epochs actually executed.
        """
        calls = {"n": 0}
        real_step = torch.optim.Adam.step

        def counting_step(self, *args, **kwargs):
            calls["n"] += 1
            return real_step(self, *args, **kwargs)

        with unittest.mock.patch.object(torch.optim.Adam, "step", counting_step):
            fn()
        return calls["n"]

    def test_early_stop_fires_on_loss_plateau(self):
        """With a small patience, training stops well before TRAIN_EPOCHS."""
        import vtscore.config as config
        from vtscore.training import mlp

        saved_epochs = config.TRAIN_EPOCHS
        saved_patience = config.TRAIN_PATIENCE
        config.TRAIN_EPOCHS = 500
        try:
            rng = np.random.RandomState(11)
            # Trivially separable so the loss plateaus quickly.
            good = rng.randn(8, 16).astype(np.float32) + 5.0
            bad = rng.randn(8, 16).astype(np.float32) - 5.0
            X = torch.tensor(np.vstack([good, bad]))
            y = torch.tensor([1.0] * 8 + [0.0] * 8).unsqueeze(1)

            config.TRAIN_PATIENCE = 0
            full_epochs = self._count_optimizer_steps(lambda: mlp.train_model(X, y, 16))

            config.TRAIN_PATIENCE = 5
            stopped_epochs = self._count_optimizer_steps(lambda: mlp.train_model(X, y, 16))

            assert full_epochs == 500
            assert stopped_epochs < full_epochs
        finally:
            config.TRAIN_EPOCHS = saved_epochs
            config.TRAIN_PATIENCE = saved_patience

    def test_patience_zero_disables_early_stop(self):
        """``TRAIN_PATIENCE=0`` should always run the full ``TRAIN_EPOCHS``."""
        import vtscore.config as config
        from vtscore.training import mlp

        saved_epochs = config.TRAIN_EPOCHS
        saved_patience = config.TRAIN_PATIENCE
        config.TRAIN_EPOCHS = 42
        config.TRAIN_PATIENCE = 0
        try:
            rng = np.random.RandomState(2)
            X = torch.tensor(rng.randn(8, 16).astype(np.float32))
            y = torch.tensor([1.0] * 4 + [0.0] * 4).unsqueeze(1)
            n_epochs = self._count_optimizer_steps(lambda: mlp.train_model(X, y, 16))
            assert n_epochs == 42
        finally:
            config.TRAIN_EPOCHS = saved_epochs
            config.TRAIN_PATIENCE = saved_patience


class TestCalibrationAtTinyLabelCounts:
    """Calibration runs at every label count, safe thresholds on or off.

    The shipped fold-anchored estimator anchors on the fold models' held-out
    scores, so the folds are an *input* to it - the pre-fusion skip below the
    blend schedule's floor (where the schedule multiplied the cross-cal cut by
    zero) has nothing left to save."""

    def test_calibrates_when_safe_and_under_six_labels(self):
        """Below 6 labels the fold trainings run: the fold-anchored estimator
        needs their models."""
        from vtscore.detectors import training as detector_training
        from vtscore.training.thresholds import conformal

        good_votes.update({k: None for k in [1, 2]})
        bad_votes.update({k: None for k in [3, 4, 5]})  # 5 labels < 6

        with unittest.mock.patch.object(
            conformal,
            "compute_fold_orderings",
            wraps=conformal.compute_fold_orderings,
        ) as patched:
            _, threshold, _model = detector_training.train_and_score(
                medias,
                good_votes,
                bad_votes,
            )
        assert patched.call_count == 1
        assert 0.0 <= threshold <= 1.0

    def test_calibrates_when_safe_off_and_under_six_labels(self):
        """Below 6 labels cross-calibration runs, so both training entry
        points agree instead of one hard-coding 0.5."""
        from vtscore.detectors import training as detector_training
        from vtscore.training.thresholds import conformal

        good_votes.update({k: None for k in [1, 2]})
        bad_votes.update({k: None for k in [3, 4, 5]})  # 5 labels < 6

        with unittest.mock.patch.object(
            conformal,
            "compute_fold_orderings",
            wraps=conformal.compute_fold_orderings,
        ) as patched:
            detector_training.train_and_score(
                medias,
                good_votes,
                bad_votes,
            )
        assert patched.call_count == 1

    def test_calibrates_at_exactly_the_old_ramp_floor(self):
        """At exactly 6 labels - the old ramp's last zero-weight step - the fold
        trainings run too.  The old schedule-derived skip is gone entirely.
        """
        from vtscore.detectors import training as detector_training
        from vtscore.training.thresholds import conformal

        good_votes.update({k: None for k in [1, 2, 3]})
        bad_votes.update({k: None for k in [18, 19, 20]})  # 6 labels

        with unittest.mock.patch.object(
            conformal,
            "compute_fold_orderings",
            wraps=conformal.compute_fold_orderings,
        ) as patched:
            detector_training.train_and_score(
                medias,
                good_votes,
                bad_votes,
            )
        assert patched.call_count == 1

    def test_still_calibrates_once_the_ramp_leaves_zero(self):
        """At 7 labels - the app's first trained-detector step - calibration
        must run, as it does at every other count."""
        from vtscore.detectors import training as detector_training
        from vtscore.training.thresholds import conformal

        good_votes.update({k: None for k in [1, 2, 3]})
        bad_votes.update({k: None for k in [17, 18, 19, 20]})  # 7 labels

        with unittest.mock.patch.object(
            conformal,
            "compute_fold_orderings",
            wraps=conformal.compute_fold_orderings,
        ) as patched:
            detector_training.train_and_score(
                medias,
                good_votes,
                bad_votes,
            )
        assert patched.call_count == 1


class TestCalibrateCountEnvDefault:
    """``VTSEARCH_CALIBRATE_COUNT`` should drive the settings default."""

    def test_default_calibrate_count_constant_exists(self):
        from vtscore import config

        assert isinstance(config.DEFAULT_CALIBRATE_COUNT, int)
        assert config.DEFAULT_CALIBRATE_COUNT >= 1


class TestCalibrationCache:
    """When the same labels are passed twice in a row with the same settings,
    the cross-calibration trainings should be skipped on the second call."""

    def _det_ctx(self):
        from vtscore.state.core import DetectorContext

        return DetectorContext("test-det")

    def _seed_six_labels(self):
        # Six labels puts us above the ``< 6`` skip floor so calibration
        # actually runs (and can therefore be cached).
        good_votes.update({k: None for k in [1, 2, 3]})
        bad_votes.update({k: None for k in [18, 19, 20]})

    def test_second_call_with_same_inputs_skips_calibration(self):
        from vtscore.detectors import training as detector_training
        from vtscore.training.thresholds import conformal

        self._seed_six_labels()
        det_ctx = self._det_ctx()

        detector_training.train_and_score(
            medias,
            good_votes,
            bad_votes,
            det_ctx=det_ctx,
        )
        assert det_ctx.calibration_cache is not None

        with unittest.mock.patch.object(
            conformal,
            "compute_fold_orderings",
            side_effect=AssertionError("fold orderings should be cached on repeat call"),
        ) as patched:
            detector_training.train_and_score(
                medias,
                good_votes,
                bad_votes,
                det_ctx=det_ctx,
            )
        patched.assert_not_called()

    def test_second_call_returns_same_threshold(self):
        from vtscore.detectors import training as detector_training

        self._seed_six_labels()
        det_ctx = self._det_ctx()

        _, t1, _ = detector_training.train_and_score(
            medias,
            good_votes,
            bad_votes,
            det_ctx=det_ctx,
        )
        _, t2, _ = detector_training.train_and_score(
            medias,
            good_votes,
            bad_votes,
            det_ctx=det_ctx,
        )
        assert t1 == t2

    def test_label_change_invalidates_cache(self):
        from vtscore.detectors import training as detector_training
        from vtscore.training.thresholds import conformal

        self._seed_six_labels()
        det_ctx = self._det_ctx()

        detector_training.train_and_score(
            medias,
            good_votes,
            bad_votes,
            det_ctx=det_ctx,
        )
        assert det_ctx.calibration_cache is not None
        first_key = det_ctx.calibration_cache[0]

        # Flip one media's label; calibration must recompute.
        good_votes.pop(3)
        bad_votes[3] = None

        with unittest.mock.patch.object(
            conformal,
            "compute_fold_orderings",
            wraps=conformal.compute_fold_orderings,
        ) as patched:
            detector_training.train_and_score(
                medias,
                good_votes,
                bad_votes,
                det_ctx=det_ctx,
            )
        assert patched.call_count == 1
        assert det_ctx.calibration_cache is not None
        assert det_ctx.calibration_cache[0] != first_key

    def test_inclusion_change_reuses_cached_orderings(self):
        """Inclusion is a pure threshold knob now: changing it must reuse the
        cached fold orderings (no fold refit) and only re-run the cheap
        min-cost search."""
        from vtscore.detectors import training as detector_training
        from vtscore.training.thresholds import conformal

        self._seed_six_labels()
        det_ctx = self._det_ctx()

        detector_training.train_and_score(
            medias,
            good_votes,
            bad_votes,
            inclusion_value=0,
            det_ctx=det_ctx,
        )
        assert det_ctx.calibration_cache is not None
        key_before = det_ctx.calibration_cache[0]

        with unittest.mock.patch.object(
            conformal,
            "compute_fold_orderings",
            wraps=conformal.compute_fold_orderings,
        ) as patched:
            detector_training.train_and_score(
                medias,
                good_votes,
                bad_votes,
                inclusion_value=2,
                det_ctx=det_ctx,
            )
        # No fold refit, and the cache key is unchanged (inclusion is not in it).
        assert patched.call_count == 0
        assert det_ctx.calibration_cache is not None
        assert det_ctx.calibration_cache[0] == key_before

    def test_no_cache_when_det_ctx_missing(self):
        """Without a det_ctx, every call must recompute calibration."""
        from vtscore.detectors import training as detector_training
        from vtscore.training.thresholds import conformal

        self._seed_six_labels()

        with unittest.mock.patch.object(
            conformal,
            "compute_fold_orderings",
            wraps=conformal.compute_fold_orderings,
        ) as patched:
            detector_training.train_and_score(
                medias,
                good_votes,
                bad_votes,
            )
            detector_training.train_and_score(
                medias,
                good_votes,
                bad_votes,
            )
        assert patched.call_count == 2

    def test_settings_default_matches_config(self):
        from vtsearch import settings

        from vtscore import config

        assert settings._DEFAULTS["calibrate_count"] == config.DEFAULT_CALIBRATE_COUNT


class TestLearnedSort:
    def test_returns_all_clips(self, client):
        good_votes.update({k: None for k in [1, 2]})
        bad_votes.update({k: None for k in [3, 4]})
        resp = client.post("/api/learned-sort", json={"wait": True})
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data["results"]) == NUM_MEDIAS
        assert "threshold" in data

    def test_result_fields(self, client):
        good_votes.update({k: None for k in [1, 2]})
        bad_votes.update({k: None for k in [3, 4]})
        resp = client.post("/api/learned-sort", json={"wait": True})
        data = resp.get_json()
        for entry in data["results"]:
            assert "id" in entry
            assert "score" in entry

    def test_sorted_descending(self, client):
        good_votes.update({k: None for k in [1, 2]})
        bad_votes.update({k: None for k in [3, 4]})
        resp = client.post("/api/learned-sort", json={"wait": True})
        data = resp.get_json()
        scores = [e["score"] for e in data["results"]]
        assert scores == sorted(scores, reverse=True)

    def test_all_media_ids_present(self, client):
        good_votes.update({k: None for k in [1, 2]})
        bad_votes.update({k: None for k in [3, 4]})
        resp = client.post("/api/learned-sort", json={"wait": True})
        data = resp.get_json()
        ids = {e["id"] for e in data["results"]}
        assert ids == set(range(1, NUM_MEDIAS + 1))

    def test_publishes_the_acquisition_cut_alongside_the_reporting_one(self, client):
        """Autopilot's Hard / New picks sample around a *different* cut.

        The picks live in the frontend and read whatever the sort response
        carries, so if the field goes missing the app silently reverts to the
        coupled behaviour PR #2876 measured as costing 4.5x the positives - with
        nothing failing.
        """
        good_votes.update({k: None for k in [1, 2]})
        bad_votes.update({k: None for k in [3, 4]})
        resp = client.post("/api/learned-sort", json={"wait": True})
        assert resp.status_code == 200
        data = resp.get_json()
        assert "acq_threshold" in data
        # A fold-anchored fit is not guaranteed on this fixture; where there is
        # one the acquisition cut sits above the decision line, and where there
        # is not the two coincide.  Never below - that is the falsified
        # direction.
        assert data["acq_threshold"] >= data["threshold"]

    def test_text_sort_carries_no_acquisition_cut(self, client):
        """No detector behind it, so there is nothing to re-cut."""
        resp = client.post("/api/sort", json={"text": "a sound"})
        assert resp.status_code == 200
        assert resp.get_json()["acq_threshold"] is None

    def test_only_good_votes_returns_400(self, client):
        good_votes.update({k: None for k in [1, 2]})
        resp = client.post("/api/learned-sort", json={"wait": True})
        assert resp.status_code == 400

    def test_only_bad_votes_returns_400(self, client):
        bad_votes.update({k: None for k in [3, 4]})
        resp = client.post("/api/learned-sort", json={"wait": True})
        assert resp.status_code == 400

    def test_scores_in_valid_range(self, client):
        good_votes.update({k: None for k in [1, 2]})
        bad_votes.update({k: None for k in [3, 4]})
        resp = client.post("/api/learned-sort", json={"wait": True})
        data = resp.get_json()
        for entry in data["results"]:
            assert 0.0 <= entry["score"] <= 1.0


class TestLearnedSortAsync:
    """The endpoint now hands the work off to a background thread and the
    client polls ``/api/learned-sort/result?job_id=...`` until done."""

    def test_async_returns_job_id_then_polling_yields_result(self, client):
        from tests.conftest import _wait_for_job
        from vtscore.concurrency.async_jobs import learned_sort_jobs

        good_votes.update({k: None for k in [1, 2]})
        bad_votes.update({k: None for k in [3, 4]})

        resp = client.post("/api/learned-sort", json={})
        assert resp.status_code == 200
        envelope = resp.get_json()
        assert envelope["status"] in ("running", "done")
        job_id = envelope["job_id"]

        # Wait for the background thread to finish before polling, since the
        # test client otherwise sees the still-running snapshot.
        _wait_for_job(learned_sort_jobs)

        result = client.get(f"/api/learned-sort/result?job_id={job_id}").get_json()
        assert result["status"] == "done"
        assert result["job_id"] == job_id
        assert "results" in result and len(result["results"]) > 0
        assert "threshold" in result

    def test_unchanged_votes_short_circuit_to_cached(self, client):
        """The signature cache lets re-sorts skip training entirely."""
        from vtscore.concurrency.async_jobs import learned_sort_jobs

        good_votes.update({k: None for k in [1, 2]})
        bad_votes.update({k: None for k in [3, 4]})

        first = client.post("/api/learned-sort", json={"wait": True}).get_json()
        assert first["status"] == "done"
        first_job_id = first["job_id"]

        # Second call with the same signature should reuse the cached result;
        # job_id is the original job's id and we get back done immediately
        # without ``wait=true``.
        second = client.post("/api/learned-sort", json={}).get_json()
        assert second["status"] == "done"
        assert second["job_id"] == first_job_id

        # Cache invalidates when votes change.
        bad_votes.update({5: None})
        third = client.post("/api/learned-sort", json={"wait": True}).get_json()
        assert third["status"] == "done"
        assert third["job_id"] != first_job_id

        learned_sort_jobs.reset_for_tests()

    def test_polling_unknown_job_returns_404(self, client):
        # 404s are intercepted by the app-level ``NotFound`` errorhandler
        # in ``app.py``, which renders the legacy
        # ``{"error": "Not Found", "request_id": ...}`` envelope.
        # Frontends rely on the HTTP status code for the missing-job
        # branch rather than a body field.
        resp = client.get("/api/learned-sort/result?job_id=does-not-exist")
        assert resp.status_code == 404

    def test_polling_without_job_id_returns_422(self, client):
        # Schema-level validation: the marshmallow
        # ``LearnedSortResultQuerySchema`` rejects requests without a
        # ``job_id`` query parameter as 422 with the standard ``errors``
        # envelope.
        resp = client.get("/api/learned-sort/result")
        assert resp.status_code == 422
        assert "job_id" in resp.get_json()["errors"]["query"]


class TestEvalTrainAndScoreAsync:
    """The eval train-and-score endpoint mirrors the learned-sort pattern:
    return a job envelope, poll a result endpoint, short-circuit unchanged
    runs via the signature cache."""

    def _seed_history(self):
        from vtsearch.state import label_history

        # A handful of "good" votes are enough to exercise the smart metric.
        for cid, lbl in [(1, "good"), (2, "good"), (3, "bad"), (4, "bad")]:
            if lbl == "good":
                good_votes[cid] = None
            else:
                bad_votes[cid] = None
            label_history.append((cid, lbl, 0.0))

    def test_wait_returns_metric_inline(self, client):
        self._seed_history()
        resp = client.post("/api/eval/train-and-score", json={"metric": "smart", "wait": True})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "done"
        assert data["metric"] == "smart"
        assert "error_cost" in data

    def test_async_polls_to_done(self, client):
        from tests.conftest import _wait_for_job
        from vtscore.concurrency.async_jobs import eval_jobs

        self._seed_history()
        envelope = client.post("/api/eval/train-and-score", json={"metric": "stable"}).get_json()
        assert envelope["status"] in ("running", "done")
        job_id = envelope["job_id"]

        _wait_for_job(eval_jobs)

        result = client.get(f"/api/eval/train-and-score/result?job_id={job_id}").get_json()
        assert result["status"] == "done"
        assert result["metric"] == "stable"
        assert "stability" in result

    def test_invalid_metric_rejected(self, client):
        resp = client.post("/api/eval/train-and-score", json={"metric": "bogus", "wait": True})
        assert resp.status_code == 422

    def test_result_reports_job_progress_not_singleton(self, client):
        """The poll endpoint must report the polled job's own progress, not
        the global ``eval_progress`` singleton (which an overlapping eval can
        pollute, decorrelating the bar from job identity)."""
        import threading

        from tests.conftest import _wait_for_job
        from vtscore.concurrency.async_jobs import eval_jobs
        from vtscore.concurrency.progress import update_eval_progress

        self._seed_history()  # 4 labels -> n_total == 3
        entered = threading.Event()
        release = threading.Event()

        def blocking_stable(*args, **kwargs):
            entered.set()
            release.wait(timeout=10)
            return []

        with unittest.mock.patch(
            "vtsearch.routes.eval.calculate_prediction_stability_over_time",
            side_effect=blocking_stable,
        ):
            envelope = client.post("/api/eval/train-and-score", json={"metric": "stable"}).get_json()
            assert envelope["status"] == "running"
            assert envelope["total"] == 3
            job_id = envelope["job_id"]

            # The job thread is now inside the (blocked) computation with its
            # own current=0 / total=3 recorded.
            assert entered.wait(timeout=10)

            # Simulate a concurrent/overlapping eval clobbering the singleton.
            update_eval_progress("running", "other eval", 99, 12345)

            result = client.get(f"/api/eval/train-and-score/result?job_id={job_id}").get_json()
            assert result["status"] == "running"
            assert result["total"] == 3, "must report the job's own total, not the singleton's 12345"
            assert result["current"] == 0

            release.set()
            _wait_for_job(eval_jobs)


class TestExampleSort:
    def test_sort_with_audio_file(self, client):
        # Create a test WAV file in memory
        wav_bytes = generate_wav(440.0, 1.0)
        data = {"file": (io.BytesIO(wav_bytes), "test.wav")}

        resp = client.post("/api/example-sort", data=data, content_type="multipart/form-data")
        assert resp.status_code == 200
        result_data = resp.get_json()
        assert "results" in result_data
        assert "threshold" in result_data
        assert len(result_data["results"]) == NUM_MEDIAS

    def test_sort_results_sorted_descending(self, client):
        wav_bytes = generate_wav(440.0, 1.0)
        data = {"file": (io.BytesIO(wav_bytes), "test.wav")}

        resp = client.post("/api/example-sort", data=data, content_type="multipart/form-data")
        result_data = resp.get_json()
        similarities = [e["similarity"] for e in result_data["results"]]
        assert similarities == sorted(similarities, reverse=True)

    def test_sort_similarity_in_valid_range(self, client):
        wav_bytes = generate_wav(440.0, 1.0)
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


class TestTextsortSuggestions:
    def test_get_empty(self, client):
        resp = client.get("/api/textsort-suggestions")
        assert resp.status_code == 200
        assert resp.get_json() == {"suggestions": []}

    def test_add_and_get(self, client):
        resp = client.post("/api/textsort-suggestions", json={"text": "dog barking"})
        assert resp.status_code == 200
        assert resp.get_json() == {"ok": True}

        resp = client.get("/api/textsort-suggestions")
        assert resp.get_json()["suggestions"] == ["dog barking"]

    def test_multiple_suggestions_ordered(self, client):
        client.post("/api/textsort-suggestions", json={"text": "birds"})
        client.post("/api/textsort-suggestions", json={"text": "rain"})
        client.post("/api/textsort-suggestions", json={"text": "thunder"})

        resp = client.get("/api/textsort-suggestions")
        assert resp.get_json()["suggestions"] == ["birds", "rain", "thunder"]

    def test_duplicate_moves_to_end(self, client):
        client.post("/api/textsort-suggestions", json={"text": "birds"})
        client.post("/api/textsort-suggestions", json={"text": "rain"})
        client.post("/api/textsort-suggestions", json={"text": "birds"})

        resp = client.get("/api/textsort-suggestions")
        assert resp.get_json()["suggestions"] == ["rain", "birds"]

    def test_empty_text_returns_400(self, client):
        resp = client.post("/api/textsort-suggestions", json={"text": ""})
        assert resp.status_code == 400

    def test_missing_text_returns_422(self, client):
        # Schema-level validation: the marshmallow
        # ``TextsortSuggestionRequestSchema`` rejects requests without a
        # ``text`` key as 422 with the standard ``errors`` envelope.
        resp = client.post("/api/textsort-suggestions", json={"other": "x"})
        assert resp.status_code == 422
        assert "text" in resp.get_json()["errors"]["json"]

    def test_whitespace_only_returns_400(self, client):
        resp = client.post("/api/textsort-suggestions", json={"text": "   "})
        assert resp.status_code == 400

    def test_cleared_with_votes(self, client):
        """Suggestions are cleared when votes are cleared."""
        client.post("/api/textsort-suggestions", json={"text": "cat meowing"})
        resp = client.get("/api/textsort-suggestions")
        assert len(resp.get_json()["suggestions"]) == 1

        from vtsearch.state import clear_votes

        clear_votes()

        resp = client.get("/api/textsort-suggestions")
        assert resp.get_json()["suggestions"] == []


class TestLoadEmbedderConcurrentCallback:
    """Verify _load_embedder_with_progress does not trample _on_progress."""

    def test_lock_exists(self):
        """The module-level lock must exist to serialise concurrent callers."""
        import threading

        from vtsearch.routes.sorting import _embedder_load_lock

        assert isinstance(_embedder_load_lock, type(threading.Lock()))

    def test_concurrent_calls_restore_original_callback(self):
        """Two threads calling _load_embedder_with_progress must leave
        _on_progress set to the *original* callback, not a stale lambda."""
        import threading
        import time
        from unittest.mock import MagicMock

        from vtsearch.routes.sorting import _load_embedder_with_progress

        original_cb = MagicMock(name="original_cb")
        mock_mt = MagicMock()
        mock_mt._model = None  # force "needs loading"
        mock_mt._on_progress = original_cb
        # Residency is read through the public accessor (#3596), so a mock has
        # to answer it rather than only carry the private attribute.
        mock_mt.models_loaded.side_effect = lambda: mock_mt._model is not None

        def slow_load_models():
            """Simulate a slow load; first call loads, second sees it loaded."""
            time.sleep(0.05)
            mock_mt._model = True  # mark as loaded

        mock_mt.load_models = slow_load_models

        with unittest.mock.patch("vtsearch.routes.sorting._get_embedder_for_loaded_data", return_value=mock_mt):
            t1 = threading.Thread(target=_load_embedder_with_progress)
            t2 = threading.Thread(target=_load_embedder_with_progress)
            t1.start()
            t2.start()
            t1.join(timeout=10)
            t2.join(timeout=10)

        # The lock ensures thread 1 finishes (restores original_cb) before
        # thread 2 enters; thread 2 sees _model is loaded and returns early.
        assert mock_mt._on_progress is original_cb, (
            "_on_progress was not restored to the original callback after concurrent calls"
        )

    def test_callback_restored_after_load_error(self):
        """_on_progress must be restored even when load_models raises."""
        import pytest
        from unittest.mock import MagicMock

        from vtsearch.routes.sorting import _load_embedder_with_progress

        original_cb = MagicMock(name="original_cb")
        mock_emb = MagicMock()
        mock_emb._model = None
        mock_emb.models_loaded.return_value = False  # cold: the load is attempted
        mock_emb._on_progress = original_cb
        mock_emb.load_models.side_effect = RuntimeError("boom")

        with unittest.mock.patch("vtsearch.routes.sorting._get_embedder_for_loaded_data", return_value=mock_emb):
            with pytest.raises(RuntimeError):
                _load_embedder_with_progress()

        assert mock_emb._on_progress is original_cb


class TestLearnedSortVoteSnapshot:
    """Regression (audit #6): the learned-sort job must train on the votes as
    they were at *request* time, not whatever the live vote proxy holds when
    the background job happens to run.

    The signature is computed at request time; if the job re-read the live
    proxy at run time, a vote change (or an ``ensure_votes_match_active_dataset``
    rehydrate) in between would cache a result trained on V2 under the key for
    V1 — poisoning ``_last_done`` for later identical-signature requests.
    """

    def test_job_uses_request_time_snapshot_not_live_votes(self, client, monkeypatch):
        from vtscore.concurrency.async_jobs import learned_sort_jobs
        from vtscore.detectors import learned_sort as ls_mod

        learned_sort_jobs.reset_for_tests()
        good_votes.clear()
        bad_votes.clear()
        good_votes.update({1: None, 2: None})
        bad_votes.update({3: None})

        captured: dict[str, dict] = {}

        def fake_run(*, good, bad, **_kwargs):
            # Simulate a vote landing after the request computed its signature
            # but before/while the job runs: mutate the live proxy.
            good_votes[99] = None
            captured["good"] = dict(good)
            captured["bad"] = dict(bad)
            return [], 0.5

        monkeypatch.setattr(ls_mod, "run_learned_sort", fake_run)

        resp = client.post("/api/learned-sort", json={"wait": True})
        assert resp.status_code == 200

        # The job trained on the frozen request-time votes, not the live proxy
        # that gained id 99 mid-run.
        assert captured["good"] == {1: None, 2: None}
        assert 99 not in captured["good"]
        assert captured["bad"] == {3: None}


from tests.fixtures.medias import NUM_MEDIAS
from vtscore.detectors.training import train_and_score
from vtscore.media.audio.audio_generator import generate_wav
from vtsearch.state import bad_votes, good_votes, medias
