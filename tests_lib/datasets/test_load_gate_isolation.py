"""Per-test isolation of the process-global dataset-load concurrency gates.

``_download_gate`` / ``_embed_gate`` live at module scope in
``vtscore.datasets.load_pipeline``, so every test landing in the same xdist
worker shares them.  The tests here cover the two halves of what keeps that
sharing invisible: the load-task registry stops the threads it forgets, and the
gates themselves are rebound between tests so a thread it could not stop cannot
be observed by — or corrupt the counts of — the next one (issue #3613).
"""

from __future__ import annotations

import threading

from vtscore.concurrency.progress import CancelledError, LoadingTasksTracker
from vtscore.datasets import load_pipeline
from vtscore.datasets.load_pipeline import _LoadGateController, reset_load_gates_for_tests


class _NullTracker:
    """The slice of ``ProgressTracker`` that ``_LoadGateController`` uses."""

    def update(self, *args, **kwargs) -> None:
        pass

    def check_cancelled(self) -> None:
        pass


class TestGateRebinding:
    def test_reset_hands_out_fresh_gates(self):
        before_download = load_pipeline._download_gate
        before_embed = load_pipeline._embed_gate
        assert before_download.acquire(blocking=False)

        reset_load_gates_for_tests()

        assert load_pipeline._download_gate is not before_download
        assert load_pipeline._embed_gate is not before_embed
        assert load_pipeline._download_gate.active == 0
        assert load_pipeline._embed_gate.active == 0

        # The permit taken above is still outstanding — on the old gate.
        assert before_download.active == 1
        before_download.release()

    def test_leaked_holder_releases_the_gate_it_acquired(self):
        """A controller from before the reset must not decrement the new gate.

        This is what makes rebinding safe in the presence of a thread the task
        registry could not join: were the release to resolve the module global
        at release time, it would drive the *current* test's gate negative and
        silently let extra loads through.
        """
        controller = _LoadGateController(_NullTracker())
        controller.acquire_download()
        stale_gate = load_pipeline._download_gate
        assert stale_gate.active == 1

        reset_load_gates_for_tests()
        fresh_gate = load_pipeline._download_gate

        controller.release()

        assert stale_gate.active == 0
        assert fresh_gate.active == 0

    def test_swap_to_embed_releases_the_download_gate_it_holds(self):
        controller = _LoadGateController(_NullTracker())
        controller.acquire_download()
        download_gate = load_pipeline._download_gate

        controller.swap_to_embed()

        assert controller.held == "embed"
        assert download_gate.active == 0
        assert load_pipeline._embed_gate.active == 1

        controller.release()
        assert load_pipeline._embed_gate.active == 0


class TestLoadingTasksReset:
    def test_reset_cancels_and_joins_a_live_worker(self):
        tracker_registry = LoadingTasksTracker()
        tracker = tracker_registry.create_task("leaky", "Leaky DS")

        running = threading.Event()
        finished = threading.Event()

        def body():
            running.set()
            try:
                while True:
                    tracker.check_cancelled()
                    tracker.cancel_event.wait(timeout=0.05)
            except CancelledError:
                finished.set()

        worker = threading.Thread(target=body, daemon=True)
        tracker_registry.set_worker("leaky", worker)
        worker.start()
        assert running.wait(timeout=5)

        tracker_registry.reset_for_tests()

        assert finished.is_set(), "reset_for_tests returned before the worker unwound"
        assert not worker.is_alive()
        assert tracker_registry.list_tasks() == []

    def test_reset_is_bounded_when_a_worker_will_not_stop(self):
        """An unstoppable worker costs the budget once, not the whole run."""
        tracker_registry = LoadingTasksTracker()
        tracker_registry.create_task("wedged", "Wedged DS")

        release = threading.Event()
        running = threading.Event()

        def body():
            running.set()
            release.wait(timeout=30)

        worker = threading.Thread(target=body, daemon=True)
        tracker_registry.set_worker("wedged", worker)
        worker.start()
        assert running.wait(timeout=5)

        try:
            tracker_registry.reset_for_tests(join_timeout=0.1)
            assert worker.is_alive(), "worker was expected to outlast the join budget"
            assert tracker_registry.list_tasks() == []
        finally:
            release.set()
            worker.join(timeout=5)
