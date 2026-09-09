"""The text-sort bar is paced for the branch this sort will actually take (#3596).

`text_sort`'s three steps are `load_model`, `embed_query`, `score`, and the
first one is bimodal in a way no coefficient can express: seconds on a process's
first sort, and *exactly zero* on every later one, because the encoder is
already resident and `_load_embedder_with_progress` returns before it even
reports step 1. #3521 measured the consequence — two fitted profiles and the
shipped defaults alike put 0.80-0.85 of this task's bar in the wrong step.

The route does not have to guess: residency is a lookup. These tests hold the
weight vector to that lookup in both directions.
"""

from unittest.mock import patch

import pytest

from vtsearch.routes import sorting
from vtscore.concurrency.progress import sort_progress


class _StubEmbedder:
    """Just enough embedder for `_load_embedder_with_progress` to run."""

    def __init__(self, loaded: bool):
        self._loaded = loaded
        self._on_progress = None

    def models_loaded(self) -> bool:
        return self._loaded

    def load_models(self) -> None:
        self._loaded = True


@pytest.fixture
def captured_weights(monkeypatch):
    """Record what the route hands `set_step_weights`, then apply it as usual."""
    seen: list[list[float] | None] = []
    original = sort_progress.set_step_weights

    def _record(weights):
        seen.append(weights)
        original(weights)

    monkeypatch.setattr(sort_progress, "set_step_weights", _record)
    yield seen
    original(None)


def _sort(client, loaded: bool):
    with patch.object(sorting, "_get_embedder_for_loaded_data", return_value=_StubEmbedder(loaded)):
        return client.post("/api/sort", json={"text": "a high pitched beep"})


class TestTextSortPacing:
    def test_a_warm_sort_budgets_nothing_for_the_model_load(self, client, captured_weights):
        assert _sort(client, loaded=True).status_code == 200
        weights = captured_weights[-1]
        assert weights is not None
        # Step 1 is never reported on this branch, so its slice must be zero
        # rather than the 0.75 the shipped defaults carry for a cold load.
        assert weights[0] == 0.0
        assert sum(weights) == pytest.approx(1.0)
        assert weights[2] > weights[1]  # scoring is the step the user waits on

    def test_a_cold_sort_still_budgets_for_it(self, client, captured_weights):
        assert _sort(client, loaded=False).status_code == 200
        weights = captured_weights[-1]
        assert weights is not None
        # The branch somebody waits seconds for keeps the shipped pacing.
        assert weights == pytest.approx([0.75, 0.05, 0.20])

    def test_the_two_branches_disagree(self, client, captured_weights):
        # The point of the change: one static vector cannot serve both, so the
        # route must produce different ones. A regression that dropped the
        # residency lookup would make these identical and pass every other test.
        _sort(client, loaded=True)
        _sort(client, loaded=False)
        assert captured_weights[-2] != captured_weights[-1]

    def test_no_embedder_at_all_is_the_warm_shape(self, client, captured_weights):
        # `_load_embedder_with_progress` returns on `emb is None` just as it does
        # on a resident model, without reporting step 1 either way.
        with patch.object(sorting, "_get_embedder_for_loaded_data", return_value=None):
            assert client.post("/api/sort", json={"text": "beep"}).status_code == 200
        assert captured_weights[-1] is not None
        assert captured_weights[-1][0] == 0.0
