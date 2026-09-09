"""A staging import must report its importer's progress against the right step.

``dataset_stage`` declares three steps — ``acquire``, ``embed``, ``serialize`` —
and the timing recorder labels a measured duration with whichever step the
tracker was on when it ran. The staging flow used to bind the importer's
progress sink straight to ``tracker.update``, so the importer's calls arrived
stepless and the tracker kept step 1 for the whole of ``run()``. Every demo
source embeds *inside* ``run()``, so the entire embed landed under ``acquire``:
#3521 §5 fitted that step at ``b = 0.0136 s/item, r² 0.9995`` — an embed curve
wearing the wrong step's name — beside an ``embed`` at ``b = 7.2e-07``, on a
sweep that had cleared the embeddings cache before every rep (#3593).

``_make_staging_progress`` is the fix, mirroring the load pipeline's
``_make_stepped_progress``. These tests pin the mapping, the two places it
deliberately differs from the load pipeline's version (an unknown status leaves
the step alone; the step never walks backwards), and the gate handoff it now
makes mid-``run()``.
"""

from __future__ import annotations

from typing import Any, cast

from vtscore.concurrency.progress import ProgressTracker
from vtscore.datasets.load_pipeline import _TOTAL_STAGE_STEPS, _make_staging_progress

_STAGE_EXTRAS = {"step": None, "total_steps": None, "overall": None, "eta_seconds": None, "error": None}


class _StubController:
    """The slice of ``_LoadGateController`` the progress callback touches."""

    def __init__(self, held: str | None = "download") -> None:
        self.held = held
        self.swaps = 0

    def swap_to_embed(self) -> None:
        self.swaps += 1
        self.held = "embed"


def _make(held: str | None = "download"):
    tracker = ProgressTracker(extra_fields=dict(_STAGE_EXTRAS))
    tracker.update("loading", "Preparing dataset…", 0, 0, step=1, total_steps=_TOTAL_STAGE_STEPS)
    controller = _StubController(held)
    # ``_make_staging_progress`` is annotated for the real controller; the stub
    # is the two members it touches.
    return tracker, controller, _make_staging_progress(cast(Any, controller), tracker)


def _step(tracker) -> int:
    return tracker.get()["step"]


class TestTheImportersEmbeddingLandsOnTheEmbedStep:
    def test_an_embedding_status_advances_to_step_two(self):
        """The regression: an importer that embeds inside ``run()`` is on step 2."""
        tracker, _, stepped = _make()

        stepped("embedding", "Embedding…", 3, 10)

        assert _step(tracker) == 2, "the importer's embedding must be recorded as the embed step"
        assert tracker.get()["message"] == "Embedding…"
        assert tracker.get()["total_steps"] == _TOTAL_STAGE_STEPS

    def test_pre_embed_statuses_share_the_acquire_step(self):
        """Staging folds the load's four steps into three, so download, unpack,
        read and convert all report against acquire."""
        for status in ("downloading", "extracting", "loading", "converting"):
            tracker, _, stepped = _make()
            stepped(status, "…", 0, 1)
            assert _step(tracker) == 1, f"{status} belongs to acquire"

    def test_an_unknown_status_leaves_the_step_alone(self):
        """The load pipeline passes ``step=None`` here, which nulls the whole-job
        fraction for that update. Staging has no pacer to absorb that, so an
        unrecognised status keeps the step the tracker was last told."""
        tracker, _, stepped = _make()
        stepped("embedding", "Embedding…", 0, 1)

        stepped("thinking", "Something new", 1, 2)

        assert _step(tracker) == 2
        assert tracker.get()["message"] == "Something new"

    def test_the_step_never_walks_backwards(self):
        """A demo's clipper reports its clip embedding under a plain ``loading``
        status, which would otherwise send the bar back to acquire after the
        embed slice had already started."""
        tracker, _, stepped = _make()
        stepped("embedding", "Embedding…", 0, 1)

        stepped("loading", "Embedding clips…", 1, 4)

        assert _step(tracker) == 2

    def test_an_explicit_step_from_the_importer_wins(self):
        """``update_progress(step=…)`` is part of the plugin contract; the map is
        a default for the callers that do not use it."""
        tracker, _, stepped = _make()

        stepped("loading", "Writing…", 0, 0, step=3)

        assert _step(tracker) == 3


class TestTheImportersTerminalIdle:
    def test_an_idle_does_not_park_the_staging_task(self):
        """``load_demo_dataset`` ends with ``on_progress("idle", …)``. That is the
        *importer* finishing, not the staging job: serialization still has to
        run, and the terminal update is the one the staging body writes."""
        tracker, _, stepped = _make()
        stepped("embedding", "Embedding…", 0, 1)

        stepped("idle", "Loaded demo dataset", 0, 0)

        assert tracker.get()["status"] == "embedding"
        assert _step(tracker) == 2

    def test_an_idle_carrying_a_failure_is_forwarded(self):
        """Dropping the status must not drop an error riding along with it."""
        tracker, _, stepped = _make()

        stepped("idle", "", 0, 0, error="importer gave up")

        assert tracker.get()["error"] == "importer gave up"


class TestTheGateHandoff:
    def test_the_first_embedding_status_swaps_to_the_embed_gate(self):
        """Staging used to swap only after ``run()`` returned, so a demo staging
        did its embedding while holding the *download* gate — the one whose
        limit exists to bound how many datasets fetch at once."""
        tracker, controller, stepped = _make()

        stepped("embedding", "Embedding…", 0, 1)

        assert controller.swaps == 1
        assert controller.held == "embed"

    def test_later_embedding_statuses_do_not_swap_again(self):
        tracker, controller, stepped = _make()

        stepped("embedding", "Embedding…", 0, 1)
        stepped("embedding", "Embedding…", 1, 1)

        assert controller.swaps == 1

    def test_a_pre_embed_status_holds_the_download_gate(self):
        tracker, controller, stepped = _make()

        stepped("downloading", "Fetching…", 0, 1)

        assert controller.swaps == 0
        assert controller.held == "download"
