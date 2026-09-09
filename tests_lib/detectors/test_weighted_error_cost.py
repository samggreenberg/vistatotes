"""The one weighted-FPR/FNR definition, and the two tiers that score through it.

Issue #3414: the shipped Smart indicator
(:func:`vtscore.detectors.labeling_progress._score_step`) and the eval
harness's window scorer
(:func:`vtscore.eval.step_trainers._labelset_error_costs`) used to carry
independent hand copies of the same FP/FN counting loop, with nothing in
``scripts/check-eval-app-sync.py`` pinning them together.  Both now delegate to
:func:`vtscore.training.thresholds.weighted_error_cost`, so the parity test
below is a check on the *plumbing* rather than on two copies of the arithmetic staying in
step - which is the point: a delegation cannot drift.
"""

from __future__ import annotations

import numpy as np
import pytest

from vtscore.training.thresholds import inclusion_cost_weights, weighted_error_cost


class TestWeightedErrorCost:
    def test_rates_and_weighting(self):
        # 2 positives (1.0, 0.9), 2 negatives (0.6, 0.1); cut at 0.5 flags three
        # of them, so one negative is a false positive and nothing is missed.
        scores = np.array([1.0, 0.9, 0.6, 0.1])
        labels = np.array([1.0, 1.0, 0.0, 0.0])
        cost, fpr, fnr = weighted_error_cost(scores, labels, 0.5, 1.0, 1.0)
        assert (fpr, fnr) == (0.5, 0.0)
        assert cost == pytest.approx(0.5)

        # Inclusion +2 quadruples the price of a miss and leaves a false alarm
        # at 1.0, so the same errors cost the same here but a miss would not.
        wf, wn = inclusion_cost_weights(2)
        assert weighted_error_cost(scores, labels, 0.5, wf, wn)[0] == pytest.approx(0.5)
        assert weighted_error_cost(scores, labels, 0.95, wf, wn) == (
            pytest.approx(4 * 0.5),
            0.0,
            0.5,
        )

    def test_threshold_is_inclusive(self):
        """``>=``, the codebase-wide convention: a score *at* the cut is Good."""
        scores = np.array([0.5, 0.5])
        labels = np.array([1.0, 0.0])
        assert weighted_error_cost(scores, labels, 0.5, 1.0, 1.0) == (1.0, 1.0, 0.0)

    def test_empty_denominators_are_zero_rates(self):
        """No negatives -> FPR 0; no positives -> FNR 0.  Never a NaN or a raise."""
        assert weighted_error_cost([0.9, 0.9], [1.0, 1.0], 0.5, 3.0, 3.0) == (0.0, 0.0, 0.0)
        assert weighted_error_cost([0.1, 0.1], [0.0, 0.0], 0.5, 3.0, 3.0) == (0.0, 0.0, 0.0)
        assert weighted_error_cost([], [], 0.5, 1.0, 1.0) == (0.0, 0.0, 0.0)

    def test_non_positive_labels_count_as_negatives(self):
        """Only ``1.0`` is a positive; both callers pass exactly ``1.0``/``0.0``."""
        assert weighted_error_cost([0.9], [0.0], 0.5, 1.0, 1.0)[1] == 1.0

    def test_operating_cost_is_this_function(self):
        """The harness's public name delegates rather than re-deriving."""
        from vtscore.eval.calibration_metrics import operating_cost

        rng = np.random.default_rng(3414)
        scores = rng.random(64)
        labels = (rng.random(64) < 0.4).astype(np.float64)
        for k in (-3, 0, 1, 5):
            wf, wn = inclusion_cost_weights(k)
            for thr in (0.0, 0.25, 0.5, 0.9, 1.5):
                assert operating_cost(scores, labels, thr, wf, wn) == weighted_error_cost(scores, labels, thr, wf, wn)


class TestAppAndHarnessScoreAlike:
    """One model, one labelset, one cost - through both tiers' plumbing."""

    @staticmethod
    def _clips(values):
        return {cid: {"id": cid, "embeddings": {"emb": np.array([v], np.float32)}} for cid, v in values.items()}

    @pytest.mark.parametrize("inclusion", [-2, 0, 3])
    def test_score_step_matches_labelset_error_costs(self, inclusion):
        torch = pytest.importorskip("torch")

        from vtscore.detectors.labeling_progress import _build_eval_rows, _score_step
        from vtscore.eval.step_model import StepModel
        from vtscore.eval.step_trainers import _labelset_error_costs

        # A 1-D sigmoid ranker both tiers can run: the app side calls the torch
        # module directly, the harness side goes through StepModel.predict.
        torch.manual_seed(0)
        net = torch.nn.Linear(1, 1)
        with torch.no_grad():
            net.weight.fill_(4.0)
            net.bias.fill_(-2.0)

        def predict(embs):
            with torch.no_grad():
                x = torch.tensor(np.asarray(embs), dtype=torch.float32)
                return torch.sigmoid(net(x)).squeeze(1).numpy()

        # Two of each class, one of them on the wrong side of the cut, so the
        # cost is neither 0 nor saturated and both error kinds are exercised.
        values = {1: 0.9, 2: 0.2, 3: 0.1, 4: 0.8}
        good, bad = {1: None, 2: None}, {3: None, 4: None}
        clips = self._clips(values)
        threshold = 0.5

        harness = _labelset_error_costs(
            [(StepModel(predict, net, "test", "cpu"), threshold)], good, bad, clips, inclusion
        )

        eval_set = _build_eval_rows(clips, good, bad)
        assert eval_set is not None
        eval_rows, eval_labels = eval_set
        wf, wn = inclusion_cost_weights(inclusion)
        step = {"model": net, "threshold": threshold, "good_ids": list(good), "bad_ids": list(bad)}
        app = _score_step(step, eval_rows, eval_labels, wf, wn, 0)

        assert app["error_cost"] == pytest.approx(harness[0], abs=1e-4)
        assert (app["fpr"], app["fnr"]) == (0.5, 0.5)
