"""Tests for parallel dataset loading: per-task progress, concurrent loads."""

import threading
import time
from unittest import mock

import numpy as np
import pytest

from tests.helpers import current_loading_progress
from vtscore.concurrency.progress import (
    CancelledError,
    LoadingTasksTracker,
    loading_tasks,
    set_thread_progress,
    get_thread_progress,
    clear_thread_progress,
)


# ---------------------------------------------------------------------------
# LoadingTasksTracker unit tests
# ---------------------------------------------------------------------------


class TestLoadingTasksTracker:
    """Unit tests for the LoadingTasksTracker."""

    def test_create_and_list(self):
        tracker = LoadingTasksTracker()
        pt = tracker.create_task("t1", "Dataset A")
        pt.update("loading", "Working...", 50, 100)

        tasks = tracker.list_tasks()
        assert len(tasks) == 1
        assert tasks[0]["task_id"] == "t1"
        assert tasks[0]["name"] == "Dataset A"
        assert tasks[0]["status"] == "loading"
        assert tasks[0]["current"] == 50
        assert tasks[0]["total"] == 100

    def test_multiple_tasks(self):
        tracker = LoadingTasksTracker()
        tracker.create_task("t1", "A").update("loading", "A loading", 10, 100)
        tracker.create_task("t2", "B").update("embedding", "B embedding", 50, 200)

        tasks = tracker.list_tasks()
        assert len(tasks) == 2
        names = {t["task_id"]: t["name"] for t in tasks}
        assert names == {"t1": "A", "t2": "B"}

    def test_cancel_task(self):
        tracker = LoadingTasksTracker()
        pt = tracker.create_task("t1", "A")
        assert tracker.cancel_task("t1") is True
        with pytest.raises(CancelledError):
            pt.check_cancelled()

    def test_cancel_nonexistent(self):
        tracker = LoadingTasksTracker()
        assert tracker.cancel_task("nope") is False

    def test_cancel_all(self):
        tracker = LoadingTasksTracker()
        pt1 = tracker.create_task("t1", "A")
        pt2 = tracker.create_task("t2", "B")
        tracker.cancel_all()
        with pytest.raises(CancelledError):
            pt1.check_cancelled()
        with pytest.raises(CancelledError):
            pt2.check_cancelled()

    def test_remove_task(self):
        tracker = LoadingTasksTracker()
        tracker.create_task("t1", "A")
        tracker.remove_task("t1")
        assert tracker.list_tasks() == []

    def test_mark_finished_and_auto_cleanup(self):
        """Finished tasks older than 5s are removed from list_tasks."""
        tracker = LoadingTasksTracker()
        pt = tracker.create_task("t1", "A")
        pt.update("idle", "Done")
        # Mark as finished 10 seconds ago
        with tracker._lock:
            tracker._tasks["t1"]["finished_at"] = time.time() - 10
        tasks = tracker.list_tasks()
        assert len(tasks) == 0

    def test_recently_finished_still_visible(self):
        """Tasks finished less than 5s ago should still appear."""
        tracker = LoadingTasksTracker()
        pt = tracker.create_task("t1", "A")
        pt.update("idle", "Done")
        tracker.mark_finished("t1")
        tasks = tracker.list_tasks()
        assert len(tasks) == 1

    def test_has_active_tasks(self):
        tracker = LoadingTasksTracker()
        assert tracker.has_active_tasks() is False
        pt = tracker.create_task("t1", "A")
        pt.update("loading", "Working")
        assert tracker.has_active_tasks() is True
        pt.update("idle", "Done")
        assert tracker.has_active_tasks() is False

    def test_reset_for_tests(self):
        tracker = LoadingTasksTracker()
        tracker.create_task("t1", "A")
        tracker.reset_for_tests()
        assert tracker.list_tasks() == []

    def test_media_type_in_task(self):
        """Tasks created with media_type expose it in list_tasks output."""
        tracker = LoadingTasksTracker()
        tracker.create_task("t1", "Image DS", media_type="image")
        tracker.create_task("t2", "Audio DS", media_type="audio")
        tracker.create_task("t3", "No Type")

        tasks = {t["task_id"]: t for t in tracker.list_tasks()}
        assert tasks["t1"]["media_type"] == "image"
        assert tasks["t2"]["media_type"] == "audio"
        assert "media_type" not in tasks["t3"]

    def test_media_type_in_finished_task(self):
        """media_type is still visible on recently-finished tasks."""
        tracker = LoadingTasksTracker()
        pt = tracker.create_task("t1", "DS", media_type="image")
        pt.update("idle", "Done")
        tracker.mark_finished("t1")

        tasks = tracker.list_tasks()
        assert len(tasks) == 1
        assert tasks[0]["media_type"] == "image"

    def test_set_dataset_id(self):
        """set_dataset_id updates the dataset_id on an existing task."""
        tracker = LoadingTasksTracker()
        tracker.create_task("t1", "DS")
        tasks = tracker.list_tasks()
        assert tasks[0].get("dataset_id") is None or tasks[0].get("dataset_id") == ""

        tracker.set_dataset_id("t1", "real-registry-id")
        tasks = tracker.list_tasks()
        assert tasks[0]["dataset_id"] == "real-registry-id"

    def test_set_dataset_id_nonexistent(self):
        """set_dataset_id on a missing task is a no-op."""
        tracker = LoadingTasksTracker()
        tracker.set_dataset_id("nope", "some-id")  # should not raise

    def test_set_dataset_id_visible_after_finish(self):
        """dataset_id is still visible on recently-finished tasks."""
        tracker = LoadingTasksTracker()
        pt = tracker.create_task("t1", "DS")
        tracker.set_dataset_id("t1", "ds-123")
        pt.update("idle", "Done")
        tracker.mark_finished("t1")

        tasks = tracker.list_tasks()
        assert len(tasks) == 1
        assert tasks[0]["dataset_id"] == "ds-123"


# ---------------------------------------------------------------------------
# Thread-local progress tests
# ---------------------------------------------------------------------------


class TestThreadLocalProgress:
    """Verify that per-thread progress callbacks work correctly."""

    def test_default_is_none(self):
        clear_thread_progress()
        assert get_thread_progress() is None

    def test_set_and_get(self):
        cb = lambda s, m="", c=0, t=0: None  # noqa: E731
        set_thread_progress(cb)
        assert get_thread_progress() is cb
        clear_thread_progress()
        assert get_thread_progress() is None

    def test_thread_isolation(self):
        """Each thread has its own callback."""
        results = {}
        barrier = threading.Barrier(2, timeout=5)

        def worker(name, cb):
            set_thread_progress(cb)
            barrier.wait()  # sync so both threads are alive
            results[name] = get_thread_progress()
            clear_thread_progress()

        cb_a = lambda s, m="", c=0, t=0: "a"  # noqa: E731
        cb_b = lambda s, m="", c=0, t=0: "b"  # noqa: E731

        t1 = threading.Thread(target=worker, args=("a", cb_a))
        t2 = threading.Thread(target=worker, args=("b", cb_b))
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        assert results["a"] is cb_a
        assert results["b"] is cb_b


# ---------------------------------------------------------------------------
# The loading-task registry is the only dataset progress there is
# ---------------------------------------------------------------------------


class TestLoadingTaskProgressIsAuthoritative:
    """Every dataset-progress read resolves to a per-task tracker.

    The global ``dataset_progress`` singleton these used to fall back to is
    gone, along with the ``get_progress()`` free function that merged the two:
    a load that has no task has no progress, rather than progress that belongs
    to nothing and that nothing terminates (#3167).
    """

    def test_returns_active_loading_task(self):
        pt = loading_tasks.create_task("test_gp", "TestDS")
        pt.update("loading", "Embedding files", 42, 100, step=3, total_steps=4)
        try:
            progress = current_loading_progress()
            assert progress["status"] == "loading"
            assert progress["message"] == "Embedding files"
            assert progress["current"] == 42
            assert progress["task_id"] == "test_gp"
        finally:
            loading_tasks.remove_task("test_gp")

    def test_returns_errored_task(self):
        pt = loading_tasks.create_task("test_err", "FailDS")
        pt.update("idle", "", error="Something went wrong")
        try:
            progress = current_loading_progress()
            assert progress["error"] == "Something went wrong"
            assert progress["task_id"] == "test_err"
        finally:
            loading_tasks.remove_task("test_err")

    def test_no_global_fallback_survives(self):
        """The removed singleton and its accessors must stay removed.

        A re-introduced process-wide sink would look harmless — it renders
        nowhere — right up until it parked on a message no worker could clear.
        """
        import vtscore.concurrency.progress as progress_mod

        for gone in ("dataset_progress", "get_progress", "check_dataset_cancelled", "LEGACY_PROGRESS_TARGET"):
            assert not hasattr(progress_mod, gone), f"{gone} is back; the legacy global progress system is not"

    def test_unbound_thread_reports_into_a_noop(self):
        """With nothing bound, the resolved sink discards rather than parks."""
        from vtscore.concurrency.progress import noop_progress, resolve_progress_callback

        assert resolve_progress_callback() is noop_progress
        # And it accepts the four-argument contract without raising.
        resolve_progress_callback()("loading", "nobody is watching", 1, 2)


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------


class TestLoadingTasksTrackerEndpoint:
    """Test the loading_tasks tracker (streamed via the SSE `loading-tasks` channel)."""

    def test_returns_empty_when_no_tasks(self, client):
        assert loading_tasks.list_tasks() == []

    def test_returns_active_tasks(self, client):
        pt = loading_tasks.create_task("api_test", "API Test DS")
        pt.update("loading", "Processing", 25, 50)
        try:
            tasks = loading_tasks.list_tasks()
            assert len(tasks) == 1
            task = tasks[0]
            assert task["task_id"] == "api_test"
            assert task["name"] == "API Test DS"
            assert task["status"] == "loading"
            assert task["current"] == 25
        finally:
            loading_tasks.remove_task("api_test")


class TestLoadingTasksMediaType:
    """Test that the loading_tasks tracker exposes media_type."""

    def test_tasks_include_media_type(self, client):
        pt = loading_tasks.create_task("mt_test", "Image DS", media_type="image")
        pt.update("loading", "Working", 10, 100)
        try:
            tasks = loading_tasks.list_tasks()
            assert len(tasks) == 1
            assert tasks[0]["media_type"] == "image"
        finally:
            loading_tasks.remove_task("mt_test")

    def test_tasks_omit_empty_media_type(self, client):
        pt = loading_tasks.create_task("mt_test2", "Unknown DS")
        pt.update("loading", "Working", 10, 100)
        try:
            tasks = loading_tasks.list_tasks()
            assert len(tasks) == 1
            assert "media_type" not in tasks[0]
        finally:
            loading_tasks.remove_task("mt_test2")


class TestCancelTaskEndpoint:
    """Test the /api/dataset/cancel/<task_id> endpoint.

    Cancellation is cooperative: the endpoint sets a flag, and something has
    to be *running* to observe it.  So the response distinguishes a cancel
    that reached a live worker from one that reached nothing at all — the
    ambiguity that made a finished import look wedged (#3167).
    """

    def test_cancel_running_task_reports_it_was_delivered(self, client):
        pt = loading_tasks.create_task("cancel_test", "CancelDS")
        pt.update("loading", "Working…", 10, 100)
        stop = threading.Event()
        worker = threading.Thread(target=stop.wait, daemon=True)
        loading_tasks.set_worker("cancel_test", worker)
        worker.start()
        try:
            resp = client.post("/api/dataset/cancel/cancel_test")
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["ok"] is True
            assert data["pending"] == ["cancel_test"], f"the worker is alive and will observe the flag, got {data!r}"
            assert pt.is_cancelled
        finally:
            stop.set()
            worker.join(timeout=5)
            loading_tasks.remove_task("cancel_test")

    def test_cancel_task_whose_worker_is_gone_refuses(self, client):
        """A tracker claiming work with no live worker is stale, not running."""
        pt = loading_tasks.create_task("stale_test", "StaleDS")
        pt.update("loading", "Loading SigLIP processor…", 0, 0)
        dead = threading.Thread(target=lambda: None, daemon=True)
        loading_tasks.set_worker("stale_test", dead)
        dead.start()
        dead.join(timeout=5)
        try:
            resp = client.post("/api/dataset/cancel/stale_test")
            assert resp.status_code == 409, f"no thread remained to act on the flag, got {resp.status_code}"
            data = resp.get_json()
            assert data["ok"] is False
            assert data["unresponsive"] == ["stale_test"]
            assert pt.get()["status"] == "idle", (
                f"the phantom that forced the refusal should be cleared, not left on screen, got {pt.get()!r}"
            )
        finally:
            loading_tasks.remove_task("stale_test")

    def test_cancel_idle_task_refuses(self, client):
        """An already-finished task has nothing left to cancel."""
        loading_tasks.create_task("idle_test", "IdleDS")
        try:
            resp = client.post("/api/dataset/cancel/idle_test")
            assert resp.status_code == 409
            assert resp.get_json()["ok"] is False
        finally:
            loading_tasks.remove_task("idle_test")

    def test_cancel_nonexistent_returns_404(self, client):
        resp = client.post("/api/dataset/cancel/nonexistent_task")
        assert resp.status_code == 404


class TestImportEndpointsReturnTaskId:
    """Verify that import endpoints include task_id in the response."""

    def test_load_demo_returns_task_id(self, client):
        from vtscore.datasets import DEMO_DATASETS

        demo_name = list(DEMO_DATASETS.keys())[0]

        with mock.patch("vtsearch.routes.datasets.load._run_importer_in_background", return_value="test_task_123"):
            resp = client.post(
                "/api/dataset/load-demo",
                json={"name": demo_name},
            )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ok"] is True
        assert data["task_id"] == "test_task_123"


# ---------------------------------------------------------------------------
# Parallel load integration test
# ---------------------------------------------------------------------------


class TestParallelLoadConcurrency:
    """Test that two loads can run concurrently without interfering."""

    def test_two_concurrent_loads_have_separate_progress(self):
        """Two concurrent loading tasks must each have their own progress."""
        pt_a = loading_tasks.create_task("load_a", "Dataset A")
        pt_b = loading_tasks.create_task("load_b", "Dataset B")

        pt_a.update("loading", "Loading A", 10, 100)
        pt_b.update("embedding", "Embedding B", 50, 200)

        try:
            tasks = loading_tasks.list_tasks()
            by_id = {t["task_id"]: t for t in tasks}

            assert by_id["load_a"]["status"] == "loading"
            assert by_id["load_a"]["current"] == 10
            assert by_id["load_b"]["status"] == "embedding"
            assert by_id["load_b"]["current"] == 50
        finally:
            loading_tasks.remove_task("load_a")
            loading_tasks.remove_task("load_b")

    def test_cancel_one_does_not_affect_other(self):
        """Cancelling one task should not cancel the other."""
        pt_a = loading_tasks.create_task("cancel_a", "A")
        pt_b = loading_tasks.create_task("cancel_b", "B")

        try:
            loading_tasks.cancel_task("cancel_a")
            assert pt_a.is_cancelled
            assert not pt_b.is_cancelled
        finally:
            loading_tasks.remove_task("cancel_a")
            loading_tasks.remove_task("cancel_b")


class TestErrorVisibility:
    """Verify that loading errors are visible to the polling frontend."""

    def test_error_message_always_non_empty(self):
        """Exception handlers must produce non-empty error strings."""
        pt = loading_tasks.create_task("err_test", "Fail")
        # Simulate the backend error handler with an exception that has
        # an empty str() representation:
        e = Exception()
        error_msg = str(e) or repr(e) or "Unknown error during dataset loading"
        pt.update("idle", "", 0, 0, error=error_msg)
        try:
            tasks = loading_tasks.list_tasks()
            assert len(tasks) == 1
            assert tasks[0]["error"]  # must be truthy
            assert tasks[0]["error"] != ""
        finally:
            loading_tasks.remove_task("err_test")

    def test_errored_task_visible_after_finish(self):
        """Errored tasks stay visible in list_tasks longer than success tasks."""
        pt = loading_tasks.create_task("err_vis", "ErrorDS")
        pt.update("idle", "", 0, 0, error="Something broke")
        # Mark finished 10 seconds ago; non-error tasks would be cleaned up
        with loading_tasks._lock:
            loading_tasks._tasks["err_vis"]["finished_at"] = time.time() - 10
        try:
            tasks = loading_tasks.list_tasks()
            assert len(tasks) == 1
            assert tasks[0]["error"] == "Something broke"
        finally:
            loading_tasks.remove_task("err_vis")

    def test_errored_task_cleaned_after_30s(self):
        """Errored tasks are eventually cleaned up after 30 seconds."""
        pt = loading_tasks.create_task("err_old", "OldError")
        pt.update("idle", "", 0, 0, error="Old error")
        with loading_tasks._lock:
            loading_tasks._tasks["err_old"]["finished_at"] = time.time() - 35
        tasks = loading_tasks.list_tasks()
        assert len(tasks) == 0

    def test_errored_task_listed_after_finish(self, client):
        """list_tasks() returns errored tasks until the stale window elapses."""
        pt = loading_tasks.create_task("api_err", "API Err DS")
        pt.update("idle", "", 0, 0, error="Load failed")
        loading_tasks.mark_finished("api_err")
        try:
            tasks = loading_tasks.list_tasks()
            errored = [t for t in tasks if t.get("error")]
            assert len(errored) == 1
            assert errored[0]["error"] == "Load failed"
        finally:
            loading_tasks.remove_task("api_err")

    def test_concurrent_load_one_fails_other_succeeds(self):
        """When two tasks run and one errors, the error is visible while the other continues."""
        pt_ok = loading_tasks.create_task("ok_task", "Good DS")
        pt_fail = loading_tasks.create_task("fail_task", "Bad DS")

        pt_ok.update("loading", "Still working", 50, 100)
        pt_fail.update("idle", "", 0, 0, error="Download failed")
        loading_tasks.mark_finished("fail_task")

        try:
            tasks = loading_tasks.list_tasks()
            by_id = {t["task_id"]: t for t in tasks}

            assert "ok_task" in by_id
            assert by_id["ok_task"]["status"] == "loading"

            assert "fail_task" in by_id
            assert by_id["fail_task"]["error"] == "Download failed"
        finally:
            loading_tasks.remove_task("ok_task")
            loading_tasks.remove_task("fail_task")


class TestCancelIsolation:
    """Starting a new load must not disturb any other load's cancel flag.

    This used to be enforced by a guard around a global ``reset_cancel()``
    ("only if no other loads are running"), which had to be right for both
    halves: reset too eagerly and an in-flight cancel was silently revoked;
    never reset and a new load aborted the instant it started.  Per-task
    trackers remove the trade-off, so the property is asserted directly.
    """

    def test_new_load_leaves_a_running_task_cancelled(self):
        pt = loading_tasks.create_task("active_task", "Running")
        pt.update("loading", "Downloading", 10, 100)
        pt.cancel()
        assert pt.is_cancelled

        try:
            from unittest.mock import patch

            from vtscore.datasets.load_pipeline import _run_origin_load_in_background

            with patch("vtscore.datasets.load_pipeline.threading.Thread"):
                _run_origin_load_in_background(
                    lambda: None,
                    {"importer": "test", "params": {}},
                )

            # The cancel is still owed to the task it was aimed at.
            assert pt.is_cancelled
        finally:
            loading_tasks.remove_task("active_task")

    def test_new_load_starts_uncancelled(self):
        from unittest.mock import patch

        from vtscore.datasets.load_pipeline import _run_origin_load_in_background

        stale = loading_tasks.create_task("stale_task", "Cancelled earlier")
        stale.update("loading", "", 0, 0)
        stale.cancel()

        with patch("vtscore.datasets.load_pipeline.threading.Thread"):
            task_id = _run_origin_load_in_background(
                lambda: None,
                {"importer": "test", "params": {}},
            )

        try:
            fresh = loading_tasks.get_tracker(task_id)
            assert fresh is not None
            assert not fresh.is_cancelled
        finally:
            loading_tasks.remove_task(task_id)
            loading_tasks.remove_task("stale_task")


class TestConcurrentModelLoading:
    """Verify that concurrent load_models() calls are serialised."""

    def test_concurrent_load_models_only_loads_once(self):
        """Two threads calling load_models() on the same embedder must not
        both execute _load_models_impl() concurrently; the lock should
        serialise them so the second caller sees the model already loaded."""
        from vtscore.media.embedder import MediaEmbedder

        call_count = 0
        started = threading.Event()
        proceed = threading.Event()

        class FakeEmbedder(MediaEmbedder):
            name = "fake"  # pyright: ignore[reportAssignmentType]
            media_type_id = "test"  # pyright: ignore[reportAssignmentType]

            def __init__(self):
                super().__init__()
                self._model = None

            def _load_models_impl(self):
                nonlocal call_count
                if self._model is not None:
                    return
                started.set()
                proceed.wait(timeout=5)
                call_count += 1
                self._model = "loaded"

            def _embed_media_impl(self, media):
                return None

            def embed_text(self, text):
                return None

        emb = FakeEmbedder()

        t1 = threading.Thread(target=emb.load_models)
        t2 = threading.Thread(target=emb.load_models)

        t1.start()
        started.wait(timeout=5)
        # t1 is inside _load_models_impl holding the lock.
        # Start t2; it must block on the lock.
        t2.start()
        # Let t1 finish.
        proceed.set()

        t1.join(timeout=5)
        t2.join(timeout=5)

        assert call_count == 1, f"_load_models_impl ran {call_count} times, expected 1"
        assert emb._model == "loaded"


class TestLoadingGates:
    """Verify the download/embed gates serialise (or pipeline) concurrent loads.

    These tests read ``_download_gate.active`` / ``_embed_gate.active`` as
    absolute counts, which is only meaningful because the per-test reset in
    ``tests_shared.state_reset`` hands every test freshly-constructed gates.
    This class used to carry its own drain-and-assert fixture instead, and that
    could only *detect* a gate another test had left dirty, never prevent it —
    so it turned a leak elsewhere in the worker process into an error here
    (issue #3613).
    """

    def test_second_load_waits_for_first(self):
        """With the download limit at 1, a second load should show 'Waiting…'
        for the download gate and only proceed after the first releases it."""
        from vtsearch import settings as settings_mod
        from vtscore.datasets.load_pipeline import (
            _download_gate,
            _run_origin_load_in_background,
        )

        original = settings_mod.get_max_concurrent_dataset_downloads()
        settings_mod.set_max_concurrent_dataset_downloads(1)
        try:
            first_started = threading.Event()
            first_proceed = threading.Event()
            second_started = threading.Event()
            load_order = []

            def first_load(medias):
                load_order.append("first_start")
                first_started.set()
                first_proceed.wait(timeout=10)
                load_order.append("first_end")

            def second_load(medias):
                load_order.append("second_start")
                second_started.set()

            task1 = _run_origin_load_in_background(
                first_load,
                {"importer": "test1", "params": {}},
                name="First",
            )

            # Wait for first load to actually start running.
            assert first_started.wait(timeout=10)

            _download_gate.waiter_parked.clear()
            task2 = _run_origin_load_in_background(
                second_load,
                {"importer": "test2", "params": {}},
                name="Second",
            )

            # Wait until the second load is actually parked on the gate
            # (deterministic; no fixed race-window sleep).
            assert _download_gate.waiter_parked.wait(timeout=10), "Second load never blocked on the download gate"
            assert not second_started.is_set(), "Second load should be queued, not running"

            # Check that the second task shows a "Waiting" message.
            task2_info = loading_tasks.get_tracker(task2)
            assert task2_info is not None
            status = task2_info.get()
            assert "Waiting" in status.get("message", "")

            # Let the first load finish.
            first_proceed.set()

            # Now the second should proceed.
            assert second_started.wait(timeout=10), "Second load never started after first finished"
            assert load_order[:2] == ["first_start", "first_end"]
            assert "second_start" in load_order

            # Clean up: wait for tasks to finish.
            deadline = time.time() + 10
            while loading_tasks.has_active_tasks() and time.time() < deadline:
                time.sleep(0.1)
            loading_tasks.remove_task(task1)
            loading_tasks.remove_task(task2)
            # Sanity: gates fully released after both tasks finish.
            assert _download_gate.active == 0
        finally:
            settings_mod.set_max_concurrent_dataset_downloads(original)

    def test_cancel_while_waiting_does_not_corrupt_gate(self):
        """Cancelling a queued task must not release the gate it never
        acquired, which would let extra loads through."""
        from vtsearch import settings as settings_mod
        from vtscore.datasets.load_pipeline import (
            _download_gate,
            _run_origin_load_in_background,
        )

        original = settings_mod.get_max_concurrent_dataset_downloads()
        settings_mod.set_max_concurrent_dataset_downloads(1)
        try:
            first_started = threading.Event()
            first_proceed = threading.Event()

            def first_load(medias):
                first_started.set()
                first_proceed.wait(timeout=10)

            task1 = _run_origin_load_in_background(
                first_load,
                {"importer": "test1", "params": {}},
                name="First",
            )
            assert first_started.wait(timeout=10)

            # Start a second load; it will be queued on the download gate.
            _download_gate.waiter_parked.clear()
            task2 = _run_origin_load_in_background(
                lambda medias: None,
                {"importer": "test2", "params": {}},
                name="Second",
            )
            assert _download_gate.waiter_parked.wait(timeout=10), "Second load never blocked on the download gate"

            # Cancel the queued task before it acquires the gate, then wait
            # for its worker thread to fully unwind (mark_finished runs in
            # the task's outermost finally, after any gate release).
            loading_tasks.cancel_task(task2)
            deadline = time.time() + 10
            while not loading_tasks.is_finished(task2) and time.time() < deadline:
                time.sleep(0.05)
            assert loading_tasks.is_finished(task2), "Cancelled task never finished"

            # The gate should still show exactly one holder (the first load).
            # If the cancel wrongly released, active would drop to 0.
            assert _download_gate.active == 1, "Cancelled task that never held the gate must not release it"

            # Clean up.
            first_proceed.set()
            deadline = time.time() + 10
            while loading_tasks.has_active_tasks() and time.time() < deadline:
                time.sleep(0.1)
            loading_tasks.remove_task(task1)
            loading_tasks.remove_task(task2)
            assert _download_gate.active == 0
        finally:
            settings_mod.set_max_concurrent_dataset_downloads(original)

    def test_minimalist_importer_releases_both_gates(self):
        """C1 regression: an importer that never fires ``status="embedding"``
        must still release both gates after the task finishes.

        The callback-driven download→embed swap in
        ``_make_stepped_progress`` only triggers when an importer
        signals per-file embedding.  Minimalist importers that don't
        emit progress at all rely on the unconditional
        ``controller.swap_to_embed()`` after ``_run_importer`` returns
        plus the ``finally: controller.release()`` block to keep the
        gates from leaking.  If either safety net is removed, the
        download gate would stay held forever and every subsequent
        dataset load would block.
        """
        from vtscore.datasets.load_pipeline import (
            _download_gate,
            _embed_gate,
            _run_origin_load_in_background,
        )

        started = threading.Event()

        def minimalist_load(medias):
            # Return immediately without firing any progress callback;
            # the callback-driven swap in stepped() never triggers.
            started.set()

        task_id = _run_origin_load_in_background(
            minimalist_load,
            {"importer": "minimalist", "params": {}},
            name="Minimalist",
        )

        assert started.wait(timeout=10), "Minimalist importer never started"

        # The task's ``finally`` block calls ``controller.release()`` after
        # the post-load stages finish.  Poll the gates (rather than
        # ``has_active_tasks``) because a successful task leaves its
        # progress status at "loading"; gate release is the true signal
        # that the task has exited.
        deadline = time.time() + 10
        while time.time() < deadline:
            if _download_gate.active == 0 and _embed_gate.active == 0:
                break
            time.sleep(0.05)

        assert _download_gate.active == 0, (
            "Download gate leaked after a minimalist importer: the "
            "unconditional swap_to_embed() after the importer, or the "
            "finally-release in the task body, has regressed (audit C1)."
        )
        assert _embed_gate.active == 0, "Embed gate leaked after minimalist importer"

        loading_tasks.remove_task(task_id)

    def test_download_and_embed_can_overlap(self):
        """When the importer signals the embedding phase, the download gate
        is released so a second dataset can start downloading in parallel
        even though the first hasn't finished embedding."""
        from vtscore.datasets.load_pipeline import (
            _download_gate,
            _embed_gate,
            _run_origin_load_in_background,
        )

        first_in_embed = threading.Event()
        first_proceed = threading.Event()
        second_started = threading.Event()
        second_proceed = threading.Event()

        def first_load(medias):
            cb = get_thread_progress()
            assert cb is not None
            # Signal the importer's per-file embedding phase.  This must
            # cause the orchestrator to swap from the download gate to the
            # embed gate, freeing the download slot for task 2.
            cb("embedding", "Embedding…", 0, 1)
            first_in_embed.set()
            first_proceed.wait(timeout=10)

        def second_load(medias):
            second_started.set()
            second_proceed.wait(timeout=10)

        task1 = _run_origin_load_in_background(
            first_load,
            {"importer": "first", "params": {}},
            name="First",
        )
        assert first_in_embed.wait(timeout=10)

        # Task 1 should now be holding the embed gate, not the download gate.
        assert _embed_gate.active == 1
        assert _download_gate.active == 0

        # Task 2 should be able to acquire the download gate immediately and
        # start running its load_fn in parallel with task 1's embedding.
        task2 = _run_origin_load_in_background(
            second_load,
            {"importer": "second", "params": {}},
            name="Second",
        )
        assert second_started.wait(timeout=10), (
            "Second load never started: download gate was not released after the swap"
        )
        assert _download_gate.active == 1

        # Let both finish.
        first_proceed.set()
        second_proceed.set()
        deadline = time.time() + 10
        while loading_tasks.has_active_tasks() and time.time() < deadline:
            time.sleep(0.1)
        loading_tasks.remove_task(task1)
        loading_tasks.remove_task(task2)
        assert _download_gate.active == 0
        assert _embed_gate.active == 0

    def test_download_limit_is_user_configurable(self):
        """Bumping ``max_concurrent_dataset_downloads`` should let the second
        load start its download phase in parallel with the first."""
        from vtsearch import settings as settings_mod
        from vtscore.datasets.load_pipeline import (
            _download_gate,
            _run_origin_load_in_background,
        )

        original = settings_mod.get_max_concurrent_dataset_downloads()
        settings_mod.set_max_concurrent_dataset_downloads(2)
        try:
            first_started = threading.Event()
            first_proceed = threading.Event()
            second_started = threading.Event()
            second_proceed = threading.Event()

            def first_load(medias):
                first_started.set()
                first_proceed.wait(timeout=10)

            def second_load(medias):
                second_started.set()
                second_proceed.wait(timeout=10)

            task1 = _run_origin_load_in_background(
                first_load,
                {"importer": "first", "params": {}},
                name="First",
            )
            assert first_started.wait(timeout=10)

            task2 = _run_origin_load_in_background(
                second_load,
                {"importer": "second", "params": {}},
                name="Second",
            )
            assert second_started.wait(timeout=10), (
                "Second load did not start in parallel: limit change did not take effect"
            )
            assert _download_gate.active == 2

            first_proceed.set()
            second_proceed.set()
            deadline = time.time() + 10
            while loading_tasks.has_active_tasks() and time.time() < deadline:
                time.sleep(0.1)
            loading_tasks.remove_task(task1)
            loading_tasks.remove_task(task2)
        finally:
            settings_mod.set_max_concurrent_dataset_downloads(original)

    # -- Staging imports run under the same gates -------------------------
    #
    # A staging import (the combine flow) downloads and embeds exactly like a
    # regular load, so it has to queue behind the same limits.  It used to run
    # ungated: N stagings downloaded and embedded fully in parallel with each
    # other *and* with gated loads, defeating the configured caps (#3394).

    def test_staging_import_swaps_download_gate_for_embed_gate(self):
        """A staging import holds the download gate through the importer and
        the embed gate through embedding, then releases both."""
        from vtscore.datasets.load_pipeline import (
            _download_gate,
            _embed_gate,
            _stage_importer_in_background,
        )

        in_run = threading.Event()
        run_proceed = threading.Event()
        during_run: list[tuple[int, int]] = []
        during_embed: list[tuple[int, int]] = []

        class _BlockingImporter:
            name = "gated_stage"
            fields: list = []

            def resolve_display_name(self, field_values):
                return "Gated"

            def run(self, field_values, temp_medias, thin=False):
                during_run.append((_download_gate.active, _embed_gate.active))
                in_run.set()
                run_proceed.wait(timeout=10)
                # Produce nothing: the staging body then parks at "Import
                # produced no medias." without writing a pkl to the shared
                # staging dir.

        def _fake_embed(*args, **kwargs):
            during_embed.append((_download_gate.active, _embed_gate.active))

        with mock.patch("vtscore.datasets.load_pipeline.embed_missing", _fake_embed):
            task_id = _stage_importer_in_background(_BlockingImporter(), {})
            assert in_run.wait(timeout=10), "staging importer never started"
            run_proceed.set()

            deadline = time.time() + 10
            while not loading_tasks.is_finished(task_id) and time.time() < deadline:
                time.sleep(0.05)
            assert loading_tasks.is_finished(task_id), "staging task never finished"

        assert during_run == [(1, 0)], "staging must run its importer under the download gate"
        assert during_embed == [(0, 1)], "staging must swap to the embed gate before embedding"
        assert _download_gate.active == 0
        assert _embed_gate.active == 0
        loading_tasks.remove_task(task_id)

    def test_staging_queues_behind_a_running_load(self):
        """With the download limit at 1, a staging import parks on the gate
        while a regular load holds it, and reports the same waiting message."""
        from vtsearch import settings as settings_mod
        from vtscore.datasets.load_pipeline import (
            _download_gate,
            _run_origin_load_in_background,
            _stage_importer_in_background,
        )

        original = settings_mod.get_max_concurrent_dataset_downloads()
        settings_mod.set_max_concurrent_dataset_downloads(1)
        try:
            load_started = threading.Event()
            load_proceed = threading.Event()
            stage_started = threading.Event()

            def blocking_load(medias):
                load_started.set()
                load_proceed.wait(timeout=10)

            class _StubImporter:
                name = "queued_stage"
                fields: list = []

                def resolve_display_name(self, field_values):
                    return "Queued"

                def run(self, field_values, temp_medias, thin=False):
                    stage_started.set()

            load_id = _run_origin_load_in_background(blocking_load, {"importer": "first", "params": {}}, name="First")
            assert load_started.wait(timeout=10)

            _download_gate.waiter_parked.clear()
            with mock.patch("vtscore.datasets.load_pipeline.embed_missing", lambda *a, **k: None):
                stage_id = _stage_importer_in_background(_StubImporter(), {})
                assert _download_gate.waiter_parked.wait(timeout=10), (
                    "staging import never blocked on the download gate"
                )
                assert not stage_started.is_set(), "staging importer ran while the gate was full"

                tracker = loading_tasks.get_tracker(stage_id)
                assert tracker is not None
                status = tracker.get()
                assert "Waiting" in status.get("message", "")
                # The wait message must carry the staging step structure, not
                # the load pipeline's, or the whole-job bar rescales on queueing.
                assert status.get("total_steps") == 3

                load_proceed.set()
                assert stage_started.wait(timeout=10), "staging never ran after the load released the gate"

                deadline = time.time() + 10
                while loading_tasks.has_active_tasks() and time.time() < deadline:
                    time.sleep(0.05)

            loading_tasks.remove_task(load_id)
            loading_tasks.remove_task(stage_id)
            assert _download_gate.active == 0
        finally:
            settings_mod.set_max_concurrent_dataset_downloads(original)

    def test_cancelled_staging_reports_cancelled_not_a_raw_error(self):
        """Cancelling a queued staging import must surface ``error="Cancelled"``.

        The dashboard and the toast service both test for that exact string to
        tell a user-requested stop from a genuine failure, so staging's own
        except-taxonomy (which had no ``CancelledError`` branch) popped a red
        "failed" toast for a cancel the user asked for.
        """
        from vtsearch import settings as settings_mod
        from vtscore.datasets.load_pipeline import (
            _download_gate,
            _run_origin_load_in_background,
            _stage_importer_in_background,
        )

        original = settings_mod.get_max_concurrent_dataset_downloads()
        settings_mod.set_max_concurrent_dataset_downloads(1)
        try:
            load_started = threading.Event()
            load_proceed = threading.Event()

            def blocking_load(medias):
                load_started.set()
                load_proceed.wait(timeout=10)

            class _NeverRunsImporter:
                name = "cancelled_stage"
                fields: list = []

                def resolve_display_name(self, field_values):
                    return "Cancelled"

                def run(self, field_values, temp_medias, thin=False):
                    raise AssertionError("importer must not run: the staging was cancelled while queued")

            load_id = _run_origin_load_in_background(blocking_load, {"importer": "first", "params": {}}, name="First")
            assert load_started.wait(timeout=10)

            _download_gate.waiter_parked.clear()
            stage_id = _stage_importer_in_background(_NeverRunsImporter(), {})
            assert _download_gate.waiter_parked.wait(timeout=10)

            loading_tasks.cancel_task(stage_id)
            deadline = time.time() + 10
            while not loading_tasks.is_finished(stage_id) and time.time() < deadline:
                time.sleep(0.05)
            assert loading_tasks.is_finished(stage_id), "cancelled staging never finished"

            tracker = loading_tasks.get_tracker(stage_id)
            assert tracker is not None
            assert tracker.get().get("error") == "Cancelled"
            # The cancel must not release a gate it never acquired.
            assert _download_gate.active == 1

            load_proceed.set()
            deadline = time.time() + 10
            while loading_tasks.has_active_tasks() and time.time() < deadline:
                time.sleep(0.05)
            loading_tasks.remove_task(load_id)
            loading_tasks.remove_task(stage_id)
            assert _download_gate.active == 0
        finally:
            settings_mod.set_max_concurrent_dataset_downloads(original)


class TestImportFailureMessages:
    """The two import pipelines share one exception taxonomy (#3394)."""

    def test_cancel_import_and_oom_read_the_same_either_side(self):
        from vtscore.concurrency.progress import CancelledError as _Cancelled
        from vtscore.datasets.load_pipeline import _failure_message

        assert _failure_message(_Cancelled("Operation cancelled by user"), "fallback") == "Cancelled"
        assert _failure_message(ImportError("no torch"), "fallback").startswith("Missing dependency: no torch.")
        assert "Out of memory" in _failure_message(MemoryError(), "fallback")
        assert _failure_message(ValueError("boom"), "fallback") == "boom"
        # An exception with no message still says *something*: repr first, and
        # only a blank repr falls through to the caller's family-specific text.
        assert _failure_message(ValueError(), "fallback") == "ValueError()"


class TestConcurrencyGate:
    """Unit tests for the dynamic-limit ConcurrencyGate."""

    def test_blocking_acquire_when_limit_changes(self):
        """A waiter blocked at limit=1 must wake up when limit grows to 2."""
        from vtscore.concurrency.gate import ConcurrencyGate

        limit = [1]
        gate = ConcurrencyGate(lambda: limit[0])
        assert gate.acquire(blocking=False)

        # Second acquisition should block.
        acquired = threading.Event()

        def second():
            gate.acquire()
            acquired.set()

        t = threading.Thread(target=second, daemon=True)
        t.start()
        # Confirm it's actually blocked.
        assert not acquired.wait(timeout=0.3)

        # Raise the limit and notify; the waiter should wake up.
        with gate._cv:  # type: ignore[attr-defined]
            limit[0] = 2
            gate._cv.notify_all()  # type: ignore[attr-defined]
        assert acquired.wait(timeout=2)

        gate.release()
        gate.release()
        assert gate.active == 0

    def test_non_blocking_acquire_respects_limit(self):
        from vtscore.concurrency.gate import ConcurrencyGate

        gate = ConcurrencyGate(lambda: 2)
        assert gate.acquire(blocking=False)
        assert gate.acquire(blocking=False)
        assert not gate.acquire(blocking=False)
        gate.release()
        assert gate.acquire(blocking=False)
        gate.release()
        gate.release()

    def test_zero_limit_is_clamped_to_one(self):
        """A configured limit of 0 should still allow one acquisition."""
        from vtscore.concurrency.gate import ConcurrencyGate

        gate = ConcurrencyGate(lambda: 0)
        assert gate.acquire(blocking=False)
        assert not gate.acquire(blocking=False)
        gate.release()


class TestBuildCoverageAtlasForContext:
    """Test the context-specific coverage atlas builder."""

    def test_builds_tree_on_context(self):
        from vtscore.state.core import DatasetContext
        from vtscore.state.coverage import build_coverage_atlas_for_context

        rng = np.random.default_rng(42)
        ctx = DatasetContext("test_coverage")
        for i in range(10):
            ctx.medias[i] = {
                "id": i,
                "embeddings": {"e5": rng.standard_normal(128).astype(np.float32)},
            }

        build_coverage_atlas_for_context(ctx)
        assert ctx.coverage_atlas is not None

    def test_empty_context_sets_none(self):
        from vtscore.state.core import DatasetContext
        from vtscore.state.coverage import build_coverage_atlas_for_context

        ctx = DatasetContext("test_empty")
        build_coverage_atlas_for_context(ctx)
        assert ctx.coverage_atlas is None

    def test_survives_concurrent_media_inserts(self):
        """Regression for #2958: this used to iterate ``ctx.medias.items()``

        directly, with no lock, from the atlas route's spawned thread --
        unlike every other cross-thread reader.  A writer inserting under
        ``_state_lock`` (e.g. add-to-pile) while this iterates raised
        ``RuntimeError: dictionary changed size during iteration``.  It now
        snapshots under ``_state_lock`` first, so hammering inserts alongside
        repeated builds must never raise.

        The writer is bounded and paced: a tight unbounded loop would balloon
        the vector set (and the k-means cost of every rebuild) without adding
        anything to the race coverage.
        """
        import threading
        import time

        from vtscore.state.core import DatasetContext, _state_lock
        from vtscore.state.coverage import build_coverage_atlas_for_context

        rng = np.random.default_rng(42)
        ctx = DatasetContext("test_coverage_concurrent")
        for i in range(10):
            ctx.medias[i] = {
                "id": i,
                "embeddings": {"e5": rng.standard_normal(128).astype(np.float32)},
            }

        stop = threading.Event()

        def writer() -> None:
            next_id = 10_000
            for _ in range(60):
                if stop.is_set():
                    return
                with _state_lock:
                    ctx.medias[next_id] = {
                        "id": next_id,
                        "embeddings": {"e5": rng.standard_normal(128).astype(np.float32)},
                    }
                next_id += 1
                time.sleep(0.001)

        thread = threading.Thread(target=writer, daemon=True)
        thread.start()
        try:
            for _ in range(15):
                build_coverage_atlas_for_context(ctx)
        finally:
            stop.set()
            thread.join(timeout=5)

        assert ctx.coverage_atlas is not None


class TestBackgroundLoadThreadContext:
    """Regression for C3: background load tasks must pin the in-flight
    DatasetContext to the worker thread so importer / clipper / dedup /
    coverage-atlas helpers that resolve via ``get_active_context()`` see
    the dataset being built, not the empty fallback context.
    """

    def test_load_fn_sees_in_flight_dataset_context(self):
        from vtscore.datasets.load_pipeline import _run_origin_load_in_background
        from vtscore.state.core import (
            _empty_dataset_context,
            get_active_context,
            get_thread_dataset_context,
        )

        from vtscore.state.core import DatasetContext

        observed: dict[str, DatasetContext | None] = {}
        ran = threading.Event()

        def capture_load(target_medias):
            import numpy as np  # noqa: PLC0415

            observed["thread_ctx"] = get_thread_dataset_context()
            observed["active_ctx"] = get_active_context()
            # Mutating the active context from the load_fn must land on
            # the in-flight context, not on the empty fallback.  Give the
            # stub media a real embedding so the load pipeline's
            # ``_drop_none_embeddings_stage`` (M11 finalize) doesn't
            # remove it before we can assert on it.
            get_active_context().medias[1] = {"id": 1, "embeddings": {"e5": np.ones(4, dtype=np.float32)}}
            ran.set()

        task_id = _run_origin_load_in_background(
            capture_load,
            {"importer": "ctx_probe", "params": {}},
            name="ctx-probe",
        )

        try:
            assert ran.wait(timeout=10), "load_fn never ran"
            deadline = time.time() + 10
            while loading_tasks.has_active_tasks() and time.time() < deadline:
                time.sleep(0.05)

            thread_ctx = observed["thread_ctx"]
            active_ctx = observed["active_ctx"]

            assert thread_ctx is not None, (
                "Background task did not set a thread-local dataset context; "
                "importer-level get_active_context() would land on the empty fallback."
            )
            assert active_ctx is not None
            assert active_ctx is not _empty_dataset_context, (
                "get_active_context() resolved to _empty_dataset_context inside the background load (C3 regression)."
            )
            assert thread_ctx is active_ctx, "Thread-local context and active context disagreed inside the load."
            # Mutation through the active context proxy must have hit the
            # in-flight context, not the empty fallback.
            assert 1 in active_ctx.medias
            assert 1 not in _empty_dataset_context.medias
        finally:
            loading_tasks.remove_task(task_id)

    def test_thread_context_cleared_after_task(self):
        """The worker thread's dataset-context thread-local must be cleared
        when the task finishes so a reused thread does not leak the prior
        context to unrelated work.  Wrap ``thread_dataset_context`` so we
        can observe both the in-task pin and the post-task restore without
        racing the daemon worker's thread-local from another thread.
        """
        import contextlib

        import vtscore.state.core as state_core
        from vtscore.datasets.load_pipeline import _run_origin_load_in_background

        original_cm = state_core.thread_dataset_context
        observations: list[tuple[str, object]] = []

        @contextlib.contextmanager
        def recording_cm(ctx):
            observations.append(("enter", ctx))
            with original_cm(ctx):
                observations.append(("inside", state_core.get_thread_dataset_context()))
                yield
            observations.append(("after_exit", state_core.get_thread_dataset_context()))

        ran = threading.Event()

        def load_fn(target_medias):
            ran.set()

        # The production task imports ``thread_dataset_context`` from
        # ``vtscore.state.core`` lazily inside the worker function, so
        # patching the attribute on that module is what the worker
        # actually resolves.
        with mock.patch.object(state_core, "thread_dataset_context", recording_cm):
            task_id = _run_origin_load_in_background(
                load_fn,
                {"importer": "ctx_cleanup", "params": {}},
                name="ctx-cleanup",
            )
            try:
                assert ran.wait(timeout=10)
                deadline = time.time() + 10
                while loading_tasks.has_active_tasks() and time.time() < deadline:
                    time.sleep(0.05)
            finally:
                loading_tasks.remove_task(task_id)

        kinds = [k for k, _ in observations]
        assert "enter" in kinds and "inside" in kinds and "after_exit" in kinds, (
            f"Background task did not enter+exit the thread_dataset_context scope; observed kinds: {kinds}"
        )
        inside = next(v for k, v in observations if k == "inside")
        after = next(v for k, v in observations if k == "after_exit")
        assert inside is not None, "Background task never pinned a dataset context"
        assert after is None, (
            "Background task did not restore its thread-local dataset context on exit: "
            f"a reused worker thread would leak the prior dataset to unrelated work (saw {after!r})."
        )
