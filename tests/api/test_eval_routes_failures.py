"""Failure-path tests for the eval / labeling-progress routes.

The happy paths for these routes live in ``tests/sorting/test_sorting.py``
(``TestEvalTrainAndScoreAsync``), ``tests/api/test_api_contracts.py``, and
``tests/core/test_votes.py``.  This module covers the error branches those
suites skip: precondition rejects (missing votes / history), internal
computation failures surfacing as 500, schema rejects (422), context-
resolution rejects (409), and the job-lifecycle branches of the poll/cancel
endpoints (missing job → 404, errored job → 500, cancelled job →
``"cancelled"``).
"""

from __future__ import annotations

import unittest.mock

from vtscore.concurrency.progress import CancelledError
from vtsearch.state import bad_votes, good_votes, label_history, medias


def _inject_model_for_seeded_votes():
    """Register a trained head for the label set :func:`_seed_votes_and_history` leaves.

    Stands in for the learned sort that would have injected one, which is the
    only thing that puts a model on a step.
    """
    import numpy as np
    import torch

    from vtscore.detectors.labeling_progress import inject_live_model
    from vtscore.embedding.media_vectors import media_embedding
    from vtscore.training.mlp import train_model

    ids = [1, 2, 3, 4, 5, 6]
    X = torch.tensor(np.array([media_embedding(medias[cid]) for cid in ids]), dtype=torch.float32)
    y = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]).unsqueeze(1)
    model = train_model(X, y, X.shape[1])
    inject_live_model({1: None, 2: None, 3: None}, {4: None, 5: None, 6: None}, model, 0.5)


def _seed_votes_and_history():
    """A handful of good/bad votes plus matching label history — enough to
    satisfy the labeling-progress preconditions and drive a real eval job."""
    for cid in (1, 2, 3):
        good_votes[cid] = None
        label_history.append((cid, "good", 0.0))
    for cid in (4, 5, 6):
        bad_votes[cid] = None
        label_history.append((cid, "bad", 0.0))


class TestLabelingProgressFailures:
    """POST /api/labeling-progress precondition and computation failures."""

    def test_no_votes_returns_400(self, client):
        resp = client.post("/api/labeling-progress")
        assert resp.status_code == 400

    def test_votes_but_no_history_returns_400(self, client):
        # Votes present but no label history recorded (the second precondition
        # branch, distinct from the no-votes reject).  Set votes directly so
        # no history is appended by the vote endpoint.
        good_votes[1] = None
        bad_votes[2] = None
        assert not label_history
        resp = client.post("/api/labeling-progress")
        assert resp.status_code == 400
        assert "no label history" in resp.get_json()["message"].lower()

    def test_computation_error_returns_500(self, client):
        _seed_votes_and_history()
        with unittest.mock.patch(
            "vtsearch.routes.eval.analyze_labeling_progress",
            side_effect=RuntimeError("boom"),
        ):
            resp = client.post("/api/labeling-progress")
        assert resp.status_code == 500
        assert "computation failed" in resp.get_json()["message"].lower()


class TestLabelingStatusFailures:
    """GET /api/labeling-status computation failure surfaces as 500."""

    def test_computation_error_returns_500(self, client):
        with unittest.mock.patch(
            "vtsearch.routes.eval.compute_labeling_status",
            side_effect=RuntimeError("boom"),
        ):
            resp = client.get("/api/labeling-status")
        assert resp.status_code == 500
        assert "computation failed" in resp.get_json()["message"].lower()


class TestIndicatorScoreHistoryFailures:
    """GET /api/indicator-score-history schema and computation failures."""

    def test_missing_metric_returns_422(self, client):
        resp = client.get("/api/indicator-score-history")
        assert resp.status_code == 422
        assert "metric" in resp.get_json()["errors"]["query"]

    def test_invalid_metric_returns_422(self, client):
        resp = client.get("/api/indicator-score-history?metric=bogus")
        assert resp.status_code == 422

    def test_computation_error_returns_500(self, client):
        with unittest.mock.patch(
            "vtsearch.routes.eval.cached_indicator_history",
            side_effect=RuntimeError("boom"),
        ):
            resp = client.get("/api/indicator-score-history?metric=smart")
        assert resp.status_code == 500
        assert "score history" in resp.get_json()["message"].lower()


class TestContextErrorsAreNot500:
    """A not-yet-loaded pair must keep its 409, not be masked as a 500.

    Regression for issue #3644.  Every route in this blueprint reads the
    request-scoped proxies, which raise ``DetectorNotLoadedError`` /
    ``DatasetNotLoadedError`` when the client names a pair the backend has not
    finished loading - the app-wide 409 contract that ``vtsearch/hooks.py``
    hands off to the global error handlers.  These handlers each wrap their
    body in ``except Exception`` and abort 500, which used to swallow that
    contract and report a poll landing inside a detector's load window as an
    opaque "computation failed" 500.  The reviewer in #3644 saw it as a red
    toast over an empty panel that cleared on its own once the load landed;
    because the 500 carried no detail, it read as a bug in the empty-labelset
    branch (the failing detectors were the ones just opened, so also the ones
    with no labels yet) rather than as a detector still loading.
    """

    UNLOADED = {"X-Detector-Id": "no-such-detector"}

    def _assert_detector_409(self, resp):
        assert resp.status_code == 409, resp.get_json()
        body = resp.get_json()
        assert body["error_code"] == "detector_not_loaded"

    def test_labeling_status_returns_409(self, client):
        self._assert_detector_409(client.get("/api/labeling-status", headers=self.UNLOADED))

    def test_labeling_progress_returns_409(self, client):
        self._assert_detector_409(client.post("/api/labeling-progress", headers=self.UNLOADED))

    def test_indicator_score_history_returns_409(self, client):
        self._assert_detector_409(client.get("/api/indicator-score-history?metric=smart", headers=self.UNLOADED))

    def test_unloaded_dataset_returns_409(self, client):
        resp = client.get("/api/labeling-status", headers={"X-Dataset-Id": "no-such-dataset"})
        assert resp.status_code == 409, resp.get_json()
        assert resp.get_json()["error_code"] == "dataset_not_loaded"

    def test_loaded_detector_with_no_labels_still_returns_200(self, client):
        """The other half of #3644: an empty labelset is *not* the fault.

        Once the pair is loaded, zero votes and zero label history take the
        inline branch and answer 200 with an honest red/red - which is why the
        issue's proposed "treat no labels as the placeholder case" fix would
        have masked the load-window 409 while replacing a true status with a
        transient "Computing indicators..." placeholder.
        """
        resp = client.get("/api/labeling-status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["total_count"] == 0
        assert data["stale"] is False
        assert data["smart"]["status"] == "red"
        assert data["stable"]["status"] == "red"


class TestIndicatorScoreHistoryIsReadOnly:
    """GET /api/indicator-score-history must never advance the per-step cache.

    Advancing it inline is what made the progress-plot modal hang: it retrains
    an MLP per uncached label step on the request thread, which is precisely the
    work ``/api/labeling-status`` defers to a background worker (issue #2397).
    """

    def test_cold_cache_returns_incomplete_and_advances_nothing(self, client):
        import vtscore.detectors.labeling_progress as lp

        _seed_votes_and_history()
        lp.clear_progress_cache()

        # ``_advance_cache`` is the only way a step is ever built, and building
        # one scores the pool.  Making it fatal proves the route took neither
        # path, which is what "read-only" has to mean now that the module
        # trains nothing of its own to intercept instead (#3757).
        with unittest.mock.patch.object(lp, "_advance_cache", side_effect=AssertionError("advanced the cache")):
            resp = client.get("/api/indicator-score-history?metric=smart")

        assert resp.status_code == 200
        body = resp.get_json()
        assert body["complete"] is False
        assert body["history"] == []
        with lp._progress_lock:
            assert lp._active_cache().steps == []

    def test_warm_cache_returns_the_series(self, client):
        import vtscore.detectors.labeling_progress as lp

        _seed_votes_and_history()
        lp.clear_progress_cache()
        # Only the steps a learned sort ran against carry a detector, and only
        # those become points (#3757), so give the last label set one - as the
        # sort would - or the warm series is legitimately empty.
        _inject_model_for_seeded_votes()
        # The background worker's job, done inline here.
        client.post("/api/eval/train-and-score", json={"metric": "smart", "wait": True})

        resp = client.get("/api/indicator-score-history?metric=smart")

        assert resp.status_code == 200
        body = resp.get_json()
        assert body["complete"] is True
        assert len(body["history"]) > 0

    def test_warm_cache_with_no_trained_detector_is_complete_and_empty(self, client):
        """A caught-up cache that never saw a sort has a real, empty answer.

        Not ``complete=False``: nothing is left to compute, so sending the modal
        to the async job would only recompute the same emptiness.
        """
        import vtscore.detectors.labeling_progress as lp

        _seed_votes_and_history()
        lp.clear_progress_cache()
        client.post("/api/eval/train-and-score", json={"metric": "smart", "wait": True})

        body = client.get("/api/indicator-score-history?metric=smart").get_json()

        assert body["complete"] is True
        assert body["history"] == []


class TestEvalTrainAndScoreStartFailures:
    """POST /api/eval/train-and-score schema and (wait=true) error branches."""

    def test_invalid_metric_returns_422(self, client):
        resp = client.post("/api/eval/train-and-score", json={"metric": "bogus", "wait": True})
        assert resp.status_code == 422

    def test_wait_true_job_error_returns_500(self, client):
        _seed_votes_and_history()
        with unittest.mock.patch(
            "vtsearch.routes.eval.calculate_prediction_stability_over_time",
            side_effect=RuntimeError("boom"),
        ):
            resp = client.post("/api/eval/train-and-score", json={"metric": "stable", "wait": True})
        assert resp.status_code == 500
        assert resp.get_json()["message"]


class TestEvalTrainAndScoreResultFailures:
    """GET /api/eval/train-and-score/result job-lifecycle branches."""

    def test_missing_job_id_returns_422(self, client):
        resp = client.get("/api/eval/train-and-score/result")
        assert resp.status_code == 422
        assert "job_id" in resp.get_json()["errors"]["query"]

    def test_unknown_job_returns_404(self, client):
        # The missing-job branch is signalled by the HTTP status code; the
        # body is the standard ``{"error", "request_id"}`` Not-Found envelope
        # (the abort's extra kwargs don't surface in it).
        resp = client.get("/api/eval/train-and-score/result?job_id=does-not-exist")
        assert resp.status_code == 404

    def test_errored_job_polls_to_500(self, client):
        from tests.conftest import _wait_for_job
        from vtscore.concurrency.async_jobs import eval_jobs

        _seed_votes_and_history()
        with unittest.mock.patch(
            "vtsearch.routes.eval.calculate_prediction_stability_over_time",
            side_effect=RuntimeError("boom"),
        ):
            envelope = client.post("/api/eval/train-and-score", json={"metric": "stable"}).get_json()
            job_id = envelope["job_id"]
            _wait_for_job(eval_jobs)
            resp = client.get(f"/api/eval/train-and-score/result?job_id={job_id}")
        assert resp.status_code == 500

    def test_cancelled_job_polls_to_cancelled(self, client):
        """A running job that unwinds via ``CancelledError`` (cooperative user
        cancel) is a terminal *non-error* state: the poll reports
        ``"cancelled"`` with a 200, not a 500."""
        from tests.conftest import _wait_for_job
        from vtscore.concurrency.async_jobs import eval_jobs

        _seed_votes_and_history()
        with unittest.mock.patch(
            "vtsearch.routes.eval.calculate_prediction_stability_over_time",
            side_effect=CancelledError("cancelled by user"),
        ):
            envelope = client.post("/api/eval/train-and-score", json={"metric": "stable"}).get_json()
            job_id = envelope["job_id"]
            _wait_for_job(eval_jobs)
            resp = client.get(f"/api/eval/train-and-score/result?job_id={job_id}")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "cancelled"
        assert data["job_id"] == job_id


class TestEvalTrainAndScoreCancelFailures:
    """POST /api/eval/train-and-score/cancel/<job_id> branches."""

    def test_unknown_job_returns_404(self, client):
        resp = client.post("/api/eval/train-and-score/cancel/does-not-exist")
        assert resp.status_code == 404

    def test_cancel_existing_job_returns_ok(self, client):
        """Cancel returns 200/ok for a real job id, even when the job has
        already finished (the flag-set is idempotent and never 404s a job it
        can see)."""
        from tests.conftest import _wait_for_job
        from vtscore.concurrency.async_jobs import eval_jobs

        _seed_votes_and_history()
        envelope = client.post("/api/eval/train-and-score", json={"metric": "stable"}).get_json()
        job_id = envelope["job_id"]

        resp = client.post(f"/api/eval/train-and-score/cancel/{job_id}")
        assert resp.status_code == 200
        assert resp.get_json()["ok"] is True

        # Drain so the background thread doesn't leak into the next test.
        _wait_for_job(eval_jobs)
