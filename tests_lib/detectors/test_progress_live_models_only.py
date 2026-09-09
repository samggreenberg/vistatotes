"""The Smart / Stable series measure the shipped detector, and nothing else (#3757).

Two rules, both of which the per-step cache used to break on a patch dataset:

* **Only models the app trained are plotted.**  A step whose label set never had
  a learned sort run against it carries no model, contributes no point, and is
  simply absent from the series.  The cache used to fill those steps with a
  stand-in it trained itself - a whole-image linear SVM with an in-sample
  threshold - and because the sort job coalesces votes that arrive while it is
  running, roughly every *other* step got one.  The plotted curve then alternated
  between two unrelated model families, which is what "the detector is bouncing
  back and forth" looked like.
* **A model is scored where it is served.**  A production head on a patch dataset
  is fitted on a Good vote's boxed patch and against every row of a Bad vote, and
  served by a max over that row stack.  Scoring it on image-level vectors alone
  measures a geometry it was never fitted for, and does so *worse* the more
  labels there are - the rising half of the same chart.

The fixtures below plant a target vector in one patch cell and leave the
image-level vector near-orthogonal to it, so the two geometries disagree by
construction and a test cannot pass by scoring the wrong rows.
"""

from __future__ import annotations

import numpy as np
import pytest

import vtscore.detectors.labeling_progress as lp

DIM = 16
GRID = 4

#: A registered **patch-slot** embedder name.  The name matters: the patch gate
#: (``_scores_in_patch_space``) only pools patch rows when the detector's
#: embedder *is* the snapshot's patch-slot embedder, so a made-up name would
#: send production - and therefore this module - down the whole-image path and
#: quietly make the geometry tests below vacuous.
PATCH_EMBEDDER = "dinov3_patch"


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / max(float(np.linalg.norm(v)), 1e-12)


#: The direction planted in one patch of every media.  A head aligned with it
#: fires hard on that patch and barely at all on the image-level vector.
TARGET = _unit(np.eye(DIM, dtype=np.float32)[0])

#: The direction the image-level (CLS) vector carries instead, orthogonal to
#: TARGET so whole-image scoring cannot see the planted patch at all.
IMAGE_DIR = _unit(np.eye(DIM, dtype=np.float32)[1])


def _patch_media(mid: int, planted: bool, rng) -> dict:
    """A synthetic patch media: an image-level vector plus a raw patch grid.

    When *planted*, one grid cell holds :data:`TARGET`.  The image-level vector
    is :data:`IMAGE_DIR` either way, so it carries no evidence of the plant.

    Every *other* patch has its :data:`TARGET` component projected out, so a
    head aligned with TARGET scores it at exactly ``sigmoid(0)``.  Without that
    the max over a grid of random patches clears any cut the planted patch
    clears - 16 draws at ``1/sqrt(DIM)`` spread is more than enough - and the
    fixture would separate nothing.
    """
    grid = rng.normal(0, 1.0, (GRID, GRID, DIM)).astype(np.float32)
    grid -= (grid @ TARGET)[..., None] * TARGET
    grid /= np.linalg.norm(grid, axis=-1, keepdims=True)
    if planted:
        grid[int(rng.integers(0, GRID)), int(rng.integers(0, GRID))] = TARGET
    return {
        "id": mid,
        "embeddings": {PATCH_EMBEDDER: IMAGE_DIR.copy()},
        "embedder": PATCH_EMBEDDER,
        "media_type": "image",
        "patch_grid": grid.astype(np.float16),
        "md5": f"m{mid:031d}",
    }


def _clips(n_each: int = 6, seed: int = 0) -> dict[int, dict]:
    """``2 * n_each`` media; the even ids carry the planted patch."""
    rng = np.random.default_rng(seed)
    return {mid: _patch_media(mid, planted=(mid % 2 == 0), rng=rng) for mid in range(2 * n_each)}


def _history(n_votes: int) -> list[tuple[int, str, float]]:
    """Alternating good/bad votes, one per media id in order."""
    return [(k, "good" if k % 2 == 0 else "bad", float(k)) for k in range(n_votes)]


def _votes(n_votes: int) -> tuple[dict[int, None], dict[int, None]]:
    return (
        {k: None for k in range(n_votes) if k % 2 == 0},
        {k: None for k in range(n_votes) if k % 2 == 1},
    )


def _patch_head(direction=TARGET, scale: float = 12.0):
    """A linear head that fires on *direction* - the shape every production fit returns."""
    import torch
    import torch.nn as nn

    linear = nn.Linear(DIM, 1)
    with torch.no_grad():
        linear.weight.copy_(torch.tensor(_unit(direction)[None, :] * scale))
        linear.bias.zero_()
    model = nn.Sequential(linear)
    model.eval()
    return model


def _inject_at(clips, history, steps: set[int], threshold: float = 0.5):
    """Inject a live model for the label set as it stands after each step in *steps*.

    Mimics what ``run_learned_sort`` does for the sorts that actually ran, and
    only for those - the coalesced ones leave no model behind.
    """
    good: dict[int, None] = {}
    bad: dict[int, None] = {}
    for t, (mid, label, _) in enumerate(history):
        (good if label == "good" else bad)[mid] = None
        if t in steps:
            lp.inject_live_model(dict(good), dict(bad), _patch_head(), threshold)


class TestOnlyTrainedModelsArePlotted:
    def test_no_sorts_means_no_points_at_all(self):
        """Voting without ever sorting leaves nothing to plot - not a stand-in curve."""
        clips, history = _clips(), _history(10)
        good, bad = _votes(10)
        lp.clear_progress_cache()

        series = lp.calculate_error_cost_over_time(clips, history, good, bad, 0)

        assert series == []
        with lp._progress_lock:
            steps = lp._active_cache().steps
        assert len(steps) == 10, "every label event still gets a step"
        assert all(s["model"] is None for s in steps), "no step may carry a model the app never trained"
        assert all(s["stability"] is None for s in steps)

    def test_points_appear_only_where_a_sort_ran(self):
        """The series is the injected models, at their own steps - no filler between."""
        clips, history = _clips(), _history(10)
        good, bad = _votes(10)
        sorted_at = {3, 5, 9}
        lp.clear_progress_cache()
        _inject_at(clips, history, sorted_at)

        series = lp.calculate_error_cost_over_time(clips, history, good, bad, 0)

        assert {e["time_index"] for e in series} == sorted_at

    def test_gaps_are_gaps_not_interpolated_labels(self):
        """Each point carries the label count of its own step, so the x-axis shows the gap."""
        clips, history = _clips(), _history(10)
        good, bad = _votes(10)
        lp.clear_progress_cache()
        _inject_at(clips, history, {2, 7})

        series = lp.calculate_error_cost_over_time(clips, history, good, bad, 0)

        assert [(e["time_index"], e["num_labels"]) for e in series] == [(2, 3), (7, 8)]


class TestScoredWhereItIsServed:
    def test_error_cost_reads_the_patch_rows_not_the_image_vector(self):
        """A head that fires only on the planted patch must score as the sort ranks it.

        Every Good media carries ``TARGET`` in one patch; no image-level vector
        does.  Scored at production geometry the head clears the cut on exactly
        the Good media, so both error rates are zero.  Scored on image-level
        vectors it sees ``IMAGE_DIR`` everywhere, misses every positive, and the
        FNR would be 1.
        """
        pytest.importorskip("torch")
        clips, history = _clips(), _history(8)
        good, bad = _votes(8)
        lp.clear_progress_cache()
        # A cut only the planted patch can clear: sigmoid(12) for the patch,
        # sigmoid(~0) for anything else.
        _inject_at(clips, history, {7}, threshold=0.9)

        series = lp.calculate_error_cost_over_time(clips, history, good, bad, 0)

        assert len(series) == 1
        assert (series[0]["fpr"], series[0]["fnr"]) == (0.0, 0.0)
        assert series[0]["error_cost"] == 0.0

    def test_stability_predictions_use_the_same_geometry(self):
        """The Stable pool is scored through the serving path too.

        With a cut only a planted patch clears, exactly the planted (even-id)
        unlabeled media predict positive.  Whole-image scoring would predict
        every one of them negative instead, so a flip count taken there is a
        count over predictions the user never saw.
        """
        pytest.importorskip("torch")
        clips, history = _clips(n_each=8), _history(4)
        lp.clear_progress_cache()
        _inject_at(clips, history, {3}, threshold=0.9)

        lp.calculate_prediction_stability_over_time(clips, history, 0)

        with lp._progress_lock:
            predictions = lp._active_cache().prev_predictions
        assert predictions is not None
        planted = {cid for cid in predictions if cid % 2 == 0}
        assert planted, "the fixture must leave some planted media unlabeled"
        assert all(predictions[cid] == 1 for cid in planted)
        assert all(predictions[cid] == 0 for cid in predictions if cid % 2 == 1)


class TestStabilityPool:
    def test_pool_is_every_media_in_the_snapshot(self):
        """No subsampling: the pool is the row set a learned sort already scores.

        It used to be a seeded sample capped at ``_STABILITY_MAX_SAMPLES``,
        because the pass ran on the request thread once per label step.  Now a
        pass happens only where a sort ran - i.e. at most once per sort - so the
        bound is what the sort itself already pays, and the flip rate is exact
        rather than estimated.
        """
        clips, history = _clips(n_each=10), _history(6)
        lp.clear_progress_cache()
        _inject_at(clips, history, {3, 4, 5})

        stability = lp.calculate_prediction_stability_over_time(clips, history, 0)

        assert stability
        for entry in stability:
            assert entry["num_unlabeled"] == len(clips) - entry["num_labels"]

    def test_flip_chain_spans_a_gap_between_two_detectors(self):
        """Successive *detectors* are compared, even when steps separate them.

        Steps 1 and 5 both have models and nothing between them does.  Nothing
        moved across that gap, because no detector existed in it, so step 5's
        entry measures exactly one retraining.

        Requiring adjacency instead would be silently fatal in the very session
        #3757 came from: sorts coalesced onto every other vote never produce two
        adjacent model-bearing steps, so Stable would collect no entries at all,
        sit on "not enough history" forever, and stop Autopilot finishing.
        """
        clips, history = _clips(n_each=8), _history(6)
        lp.clear_progress_cache()
        _inject_at(clips, history, {1, 5})

        stability = lp.calculate_prediction_stability_over_time(clips, history, 0)

        assert [e["time_index"] for e in stability] == [5]
        assert stability[0]["num_unlabeled"] == len(clips) - stability[0]["num_labels"]
