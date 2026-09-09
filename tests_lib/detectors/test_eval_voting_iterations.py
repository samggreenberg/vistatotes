"""Tests for the voting-iterations evaluation framework.

All tests use small synthetic datasets with known, well-separated
embeddings so no real model downloads are needed.
"""

import math

import numpy as np
import pandas as pd
import pytest

from vtscore.eval.calibration_metrics import inclusion_weights
from vtscore.eval.step_model import StepModel, good_training_vec
from vtscore.eval.step_trainers import _labelset_error_costs
from vtscore.eval.voting_columns import TIMING_COLUMNS
from vtscore.eval.voting_iterations import (
    _split_media_ids,
    run_voting_iterations_eval,
    simulate_voting_iterations,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


def _make_separable_clips(dim=16, n_per_cat=20, seed=0):
    """Two categories with well-separated embeddings.

    Category "alpha" clusters around [+1, 0, 0, ...],
    category "beta"  clusters around [-1, 0, 0, ...].
    """
    rng = np.random.RandomState(seed)
    medias = {}
    media_id = 1
    for _ in range(n_per_cat):
        emb = rng.normal(1.0, 0.2, dim).astype(np.float32)
        medias[media_id] = {"id": media_id, "embeddings": {"emb": emb}, "category": "alpha"}
        media_id += 1
    for _ in range(n_per_cat):
        emb = rng.normal(-1.0, 0.2, dim).astype(np.float32)
        medias[media_id] = {"id": media_id, "embeddings": {"emb": emb}, "category": "beta"}
        media_id += 1
    return medias


def _make_overlapping_clips(dim=16, n_per_cat=20, seed=0):
    """Two categories with overlapping embeddings (harder to classify).

    Category "alpha" centred at [+0.3, 0, 0, ...],
    category "beta"  centred at [-0.3, 0, 0, ...], with large noise.
    """
    rng = np.random.RandomState(seed)
    medias = {}
    media_id = 1
    for _ in range(n_per_cat):
        emb = rng.normal(0.3, 1.0, dim).astype(np.float32)
        medias[media_id] = {"id": media_id, "embeddings": {"emb": emb}, "category": "alpha"}
        media_id += 1
    for _ in range(n_per_cat):
        emb = rng.normal(-0.3, 1.0, dim).astype(np.float32)
        medias[media_id] = {"id": media_id, "embeddings": {"emb": emb}, "category": "beta"}
        media_id += 1
    return medias


def _make_three_category_clips(dim=16, n_per_cat=15, seed=0):
    """Three categories: alpha, beta, gamma."""
    rng = np.random.RandomState(seed)
    medias = {}
    media_id = 1
    centres = {"alpha": 1.0, "beta": -1.0, "gamma": 0.0}
    for cat, centre in centres.items():
        for _ in range(n_per_cat):
            emb = rng.normal(centre, 0.2, dim).astype(np.float32)
            medias[media_id] = {"id": media_id, "embeddings": {"emb": emb}, "category": cat}
            media_id += 1
    return medias


# ------------------------------------------------------------------
# Unit tests: helpers
# ------------------------------------------------------------------


class TestInclusionWeights:
    def test_zero_inclusion(self):
        fpr_w, fnr_w = inclusion_weights(0)
        assert fpr_w == 1.0
        assert fnr_w == 1.0

    def test_positive_inclusion(self):
        fpr_w, fnr_w = inclusion_weights(3)
        assert fpr_w == 1.0
        assert fnr_w == 8.0

    def test_negative_inclusion(self):
        fpr_w, fnr_w = inclusion_weights(-2)
        assert fpr_w == 4.0
        assert fnr_w == 1.0


class TestSplitClipIds:
    def test_split_sizes(self):
        medias = _make_separable_clips(n_per_cat=10)
        rng = np.random.RandomState(42)
        sim, test = _split_media_ids(medias, 0.5, rng)
        assert len(sim) + len(test) == len(medias)
        assert len(sim) == 10
        assert len(test) == 10

    def test_no_overlap(self):
        medias = _make_separable_clips(n_per_cat=10)
        rng = np.random.RandomState(42)
        sim, test = _split_media_ids(medias, 0.5, rng)
        assert set(sim).isdisjoint(set(test))

    def test_deterministic(self):
        medias = _make_separable_clips(n_per_cat=10)
        rng1 = np.random.RandomState(42)
        sim1, test1 = _split_media_ids(medias, 0.5, rng1)
        rng2 = np.random.RandomState(42)
        sim2, test2 = _split_media_ids(medias, 0.5, rng2)
        assert sim1 == sim2
        assert test1 == test2


class TestLabelsetErrorCosts:
    """The Smart indicator's input: every recent model, one current labelset.

    Mirrors ``labeling_progress._eval_cached_models``.  A frozen per-step cost
    (the pre-#2923 behavior) confounds model improvement with labelset growth,
    which biases the simulated user out of the Hard phase early.
    """

    @staticmethod
    def _clips(scores):
        """One 1-D media per ``{media_id: value}`` entry; value IS the embedding."""
        return {cid: {"id": cid, "embeddings": {"emb": np.array([v], np.float32)}} for cid, v in scores.items()}

    @staticmethod
    def _model(bias):
        """A ranker whose score is the embedding value plus *bias*."""
        return StepModel(
            predict=lambda embs, b=bias: np.asarray(embs, dtype=np.float64).ravel() + b,
            torch_model=None,
            backend="test",
            device="cpu",
        )

    def test_returns_one_cost_per_model_in_window_order(self):
        clips = self._clips({1: 1.0, 2: 1.0, 3: 0.0, 4: 0.0})
        good, bad = {1: None, 2: None}, {3: None, 4: None}
        # bias 0: perfect at threshold 0.5.  bias -1: everything below the cut,
        # so both positives are missed (fnr 1.0).  bias +1: everything above,
        # so both negatives are false positives (fpr 1.0).
        window = [(self._model(-1.0), 0.5), (self._model(0.0), 0.5), (self._model(1.0), 0.5)]
        costs = _labelset_error_costs(window, good, bad, clips, 0)
        assert costs == [1.0, 0.0, 1.0]

    def test_inclusion_weights_the_two_error_kinds(self):
        clips = self._clips({1: 1.0, 2: 1.0, 3: 0.0, 4: 0.0})
        good, bad = {1: None, 2: None}, {3: None, 4: None}
        # inclusion 2 => fnr weight 4, fpr weight 1.
        window = [(self._model(-1.0), 0.5), (self._model(1.0), 0.5)]
        assert _labelset_error_costs(window, good, bad, clips, 2) == [4.0, 1.0]

    def test_old_models_are_re_scored_against_the_grown_labelset(self):
        """The regression's whole point: a fixed model's cost must move."""
        clips = self._clips({1: 1.0, 2: 1.0, 3: 0.0, 4: 0.0, 5: 0.6})
        model = self._model(0.0)
        window = [(model, 0.5)]
        before = _labelset_error_costs(window, {1: None, 2: None}, {3: None, 4: None}, clips, 0)
        # Media 5 is a freshly voted boundary negative this model gets wrong -
        # exactly the item autopilot's Hard phase goes looking for.
        after = _labelset_error_costs(window, {1: None, 2: None}, {3: None, 4: None, 5: None}, clips, 0)
        assert before == [0.0]
        assert after == [1 / 3]

    def test_empty_without_models_or_without_both_classes(self):
        clips = self._clips({1: 1.0, 3: 0.0})
        window = [(self._model(0.0), 0.5)]
        assert _labelset_error_costs([], {1: None}, {3: None}, clips, 0) == []
        assert _labelset_error_costs(window, {}, {3: None}, clips, 0) == []
        assert _labelset_error_costs(window, {1: None}, {}, clips, 0) == []


# ------------------------------------------------------------------
# Unit tests: simulate_voting_iterations
# ------------------------------------------------------------------


class TestSimulateVotingIterations:
    # Shape/plumbing tests use small media pools and ``calibrate_count=1``:
    # each simulated voting step trains an MLP and calibrates per split, so a
    # full-size sweep costs seconds for assertions that only look at row
    # structure.  Behavioral tests (cost decrease, inclusion sensitivity)
    # keep the full sweep they actually measure.
    def test_returns_rows(self):
        medias = _make_separable_clips(n_per_cat=6)
        rows = simulate_voting_iterations(
            medias,
            target_category="alpha",
            seed=42,
            dataset_name="test_ds",
            inclusion=0,
            sim_fraction=0.5,
            calibrate_count=1,
        )
        assert len(rows) > 0

    def test_row_schema(self):
        medias = _make_separable_clips(n_per_cat=6)
        rows = simulate_voting_iterations(
            medias,
            target_category="alpha",
            seed=42,
            dataset_name="test_ds",
            calibrate_count=1,
        )
        from vtscore.eval.voting_columns import VOTING_COLUMNS

        for row in rows:
            assert set(row.keys()) == set(VOTING_COLUMNS)
            assert row["strategy"] == "autopilot"
            assert row["trainer"] == "app"
            assert row["prevalence_arm"] == "natural"

    def test_vote_counts_reported(self):
        """Each row carries the good/bad vote counts the model was trained on.

        The first scored row is the earliest trainable step (≥1 good and ≥1
        bad), and the counts never exceed the votes seen so far (t).  Autopilot
        seeds goods before bads, so that first step carries its initial bad
        (``n_bad == 1``) alongside however many goods have been seeded.
        """
        medias = _make_separable_clips(n_per_cat=6)
        rows = simulate_voting_iterations(medias, "alpha", seed=42, calibrate_count=1)
        assert rows  # at least one scored step
        first = rows[0]
        assert first["n_good"] >= 1
        assert first["n_bad"] == 1  # goods are seeded first, so the first bad triggers training
        for row in rows:
            assert row["n_good"] + row["n_bad"] == row["t"]
            assert row["n_good"] >= 1
            assert row["n_bad"] >= 1

    def test_seed_determinism(self):
        medias = _make_separable_clips(n_per_cat=6)
        rows1 = simulate_voting_iterations(medias, "alpha", seed=42, calibrate_count=1)
        rows2 = simulate_voting_iterations(medias, "alpha", seed=42, calibrate_count=1)
        assert len(rows1) == len(rows2)
        # Wall-clock timing columns vary between runs; compare everything else.
        _timing = TIMING_COLUMNS

        def _same(a, b) -> bool:
            """Value equality that reads NaN as reproducing itself.

            Several columns are deliberately NaN rather than 0.0 where the
            quantity does not exist (``_pool_percentile`` on an exhausted pool,
            for one - 0.0 would read as "the cut is at the very top", a real
            value).  Plain ``==`` calls two identical NaNs different, so a dict
            comparison would report the *last* step of every run as
            non-deterministic.
            """
            if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
                return True
            return bool(a == b)

        for r1, r2 in zip(rows1, rows2):
            keys1 = sorted(k for k in r1 if k not in _timing)
            keys2 = sorted(k for k in r2 if k not in _timing)
            assert keys1 == keys2
            for k in keys1:
                assert _same(r1[k], r2[k]), f"column {k!r} differs: {r1[k]!r} != {r2[k]!r}"

    def test_different_seeds_differ(self):
        # Overlapping (not separable) categories on purpose: the cost sequence
        # only reflects the seed while the task is hard enough for the vote
        # ordering to matter.  On separable clips every seed converges to a
        # perfect cut (cost 0 at every t), so the sequences coincide and the
        # assertion below would be measuring nothing.
        medias = _make_overlapping_clips(n_per_cat=10)
        rows1 = simulate_voting_iterations(medias, "alpha", seed=42, calibrate_count=1)
        rows2 = simulate_voting_iterations(medias, "alpha", seed=99, calibrate_count=1)
        # Different seeds should produce different vote orderings / splits,
        # so the t-indexed costs should differ (not guaranteed for every row,
        # but at least the full sequence should differ).
        costs1 = [r["cost"] for r in rows1]
        costs2 = [r["cost"] for r in rows2]
        assert costs1 != costs2

    def test_separable_clips_converge_to_a_usable_cut(self):
        """Issue #2781: on separable categories the threshold must not land
        above every score once enough votes exist.

        The old rule pinned the cut to the lowest held-out calibration
        positive - a fold-model-scale value applied to final-model scores -
        which on saturating folds rejected the whole collection (FNR 1.0 /
        FPR 0.0) at scattered vote counts, recovering on the next vote.
        """
        medias = _make_separable_clips(n_per_cat=10)
        for seed in (42, 99):
            rows = simulate_voting_iterations(medias, "alpha", seed=seed, calibrate_count=1)
            for row in rows:
                # Only rows where a stratified fold split can actually form are
                # governed by the conformal rule; below that
                # ``compute_fold_orderings`` returns its flat 0.5
                # ``too_few_default`` (issue #2788), which this regression is
                # not about.  Keyed on the calibrator's own precondition rather
                # than on row position, so a change in the simulated vote order
                # cannot quietly widen or narrow what is asserted.
                if row["n_good"] < 2 or row["n_bad"] < 2:
                    continue
                assert row["fnr"] < 1.0, f"seed {seed}, t={row['t']}: cut rejected every positive"

    def test_t_values_monotonically_increase(self):
        medias = _make_separable_clips(n_per_cat=6)
        rows = simulate_voting_iterations(medias, "alpha", seed=42, calibrate_count=1)
        t_vals = [r["t"] for r in rows]
        assert t_vals == sorted(t_vals)
        # t starts >=2 because we need at least 1 good + 1 bad
        assert all(t >= 2 for t in t_vals)

    def test_cost_decreases_over_time_for_overlapping_data(self):
        """With overlapping data, cost should generally decrease as more votes come in."""
        medias = _make_overlapping_clips(n_per_cat=60, dim=16)
        rows = simulate_voting_iterations(
            medias,
            "alpha",
            seed=42,
            sim_fraction=0.5,
        )
        costs = [r["cost"] for r in rows]
        # Compare average of first half vs last half.
        # With overlapping data and a regularised model the decrease is
        # gradual, so we allow a 15% tolerance.
        n = len(costs)
        mid = max(1, n // 2)
        early_avg = sum(costs[:mid]) / mid
        late_avg = sum(costs[mid:]) / max(1, n - mid)
        assert late_avg <= early_avg * 1.15

    def test_empty_when_no_test_positives(self):
        """If all medias of target category land in sim, test set has no positives -> empty."""
        # Only 1 media of target category; likely all end up in sim with 50% split
        medias = {
            1: {"id": 1, "embeddings": {"emb": np.ones(8, dtype=np.float32)}, "category": "rare"},
            2: {"id": 2, "embeddings": {"emb": -np.ones(8, dtype=np.float32)}, "category": "common"},
            3: {"id": 3, "embeddings": {"emb": -np.ones(8, dtype=np.float32) * 0.9}, "category": "common"},
            4: {"id": 4, "embeddings": {"emb": -np.ones(8, dtype=np.float32) * 0.8}, "category": "common"},
        }
        rows = simulate_voting_iterations(medias, "rare", seed=42, sim_fraction=0.5)
        # Might be empty or not depending on split; just shouldn't crash
        assert isinstance(rows, list)

    def test_inclusion_affects_cost(self):
        """With overlapping data, different inclusion values produce different costs."""
        medias = _make_overlapping_clips(n_per_cat=20)
        rows_inc0 = simulate_voting_iterations(medias, "alpha", seed=42, inclusion=0)
        rows_inc5 = simulate_voting_iterations(medias, "alpha", seed=42, inclusion=5)
        # Same splits but different inclusion -> costs should differ
        costs0 = [r["cost"] for r in rows_inc0]
        costs5 = [r["cost"] for r in rows_inc5]
        assert costs0 != costs5

    def test_elapsed_seconds_non_negative_and_increasing(self):
        """elapsed_seconds should be non-negative and non-decreasing over rows."""
        medias = _make_separable_clips(n_per_cat=6)
        rows = simulate_voting_iterations(medias, "alpha", seed=42, calibrate_count=1)
        times = [r["elapsed_seconds"] for r in rows]
        assert all(t >= 0.0 for t in times)
        for i in range(1, len(times)):
            assert times[i] >= times[i - 1]


# ------------------------------------------------------------------
# Production fidelity: the per-step calibration must match the live
# _train_and_score_xy / train_and_threshold pipeline, or the reported
# cost measures a pipeline the detector never runs.
# ------------------------------------------------------------------


def _mt_key(rng: np.random.RandomState):
    """Return the MT19937 key array of *rng* as a tuple (pyright-narrowed).

    ``get_state(legacy=True)`` returns a tuple, but the numpy stub types it as a
    ``dict | tuple`` union; the ``isinstance`` narrows it so ``state[1]`` (the
    624-word key array) type-checks.
    """
    state = rng.get_state(legacy=True)
    assert isinstance(state, tuple)
    return state


class TestProductionCalibrationFidelity:
    """The eval's per-step threshold calibration mirrors production's protocol.

    Production (`_train_and_score_xy` / `train_and_threshold`) threads a single
    head architecture through both the final model and the calibration folds,
    and always calibrates with a fresh ``RandomState(42)`` (the fixed seed baked
    into ``calibration_folds_cached``).  These tests spy on the calibration call
    to prove the eval does the same, so overlapping the fold split RNG with the
    per-seed simulation RNG or letting folds auto-size can't silently
    reintroduce a production mismatch.

    The *head* is threaded the same way whichever arm runs: the default arm
    trains production's linear head (#2790/#2916) and ``head="mlp"`` the legacy
    auto-sized one, and in both cases the folds must take the final model's
    width rather than sizing themselves from their own smaller split.
    """

    def _spy_calibration(self, monkeypatch):
        from vtscore.eval import step_trainers

        real = step_trainers.calibration_folds
        captured: list[dict] = []

        def spy(X_list, y_list, input_dim, *, rng=None, hidden_dim: int = 0, **kw):
            # get_state() copies without advancing, so recording it here does
            # not perturb the real calibration that runs on the next line.
            captured.append(
                {
                    "n": len(X_list),
                    "hidden_dim": hidden_dim,
                    "mt_key": _mt_key(rng)[1] if rng is not None else None,
                }
            )
            return real(X_list, y_list, input_dim, rng=rng, hidden_dim=hidden_dim, **kw)

        monkeypatch.setattr(step_trainers, "calibration_folds", spy)
        return captured

    @pytest.mark.parametrize("head", [None, "mlp"])
    def test_folds_forced_to_full_data_hidden_dim(self, head, monkeypatch):
        from vtscore.eval.step_model import PRODUCTION_HEAD, resolve_hidden_dim

        captured = self._spy_calibration(monkeypatch)
        medias = _make_separable_clips(n_per_cat=10)
        simulate_voting_iterations(medias, "alpha", seed=42, head=head)

        assert captured  # at least one calibrated step
        for c in captured:
            # The fold models must be sized from the full label count for the
            # step, not auto-sized per fold (hidden_dim=None) - on the default
            # (production) arm and on the legacy MLP arm alike.
            assert c["hidden_dim"] == resolve_hidden_dim(head or PRODUCTION_HEAD, c["n"])

    def test_folds_calibrate_with_fixed_random_state_42(self, monkeypatch):
        captured = self._spy_calibration(monkeypatch)
        medias = _make_separable_clips(n_per_cat=10)
        simulate_voting_iterations(medias, "alpha", seed=7)

        assert captured
        ref_key = _mt_key(np.random.RandomState(42))[1]
        for c in captured:
            mt_key = c["mt_key"]
            assert mt_key is not None
            # A fresh RandomState(42), not the shared per-seed simulation RNG
            # (which the media split + vote sequence would have advanced).
            assert np.array_equal(mt_key, ref_key)

    def test_calibration_rng_independent_of_eval_seed(self, monkeypatch):
        """The fold split RNG is pinned, so it does not vary with the eval seed."""
        medias = _make_separable_clips(n_per_cat=10)

        captured_a = self._spy_calibration(monkeypatch)
        simulate_voting_iterations(medias, "alpha", seed=1)
        captured_b = self._spy_calibration(monkeypatch)
        simulate_voting_iterations(medias, "alpha", seed=2)

        # Same first-step vote count is not guaranteed across seeds, but every
        # calibrated step in both runs must start from the identical RNG state.
        assert captured_a and captured_b
        ref_key = tuple(_mt_key(np.random.RandomState(42))[1].tolist())
        states_a = {tuple(c["mt_key"].tolist()) for c in captured_a}
        states_b = {tuple(c["mt_key"].tolist()) for c in captured_b}
        assert states_a == states_b == {ref_key}


# ------------------------------------------------------------------
# Integration test: run_voting_iterations_eval
# ------------------------------------------------------------------


class TestRunVotingIterationsEval:
    # Plumbing tests (columns, cross-product coverage): small pools and
    # ``calibrate_count=1`` keep each seed×category sweep cheap.
    def test_returns_dataframe(self):
        medias = _make_separable_clips(n_per_cat=6)
        df = run_voting_iterations_eval(
            dataset_clips={"ds1": medias},
            seeds=[42],
            categories={"ds1": ["alpha"]},
            calibrate_count=1,
        )
        from vtscore.eval.voting_columns import VOTING_COLUMNS

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == list(VOTING_COLUMNS)

    def test_multiple_seeds(self):
        medias = _make_separable_clips(n_per_cat=6)
        df = run_voting_iterations_eval(
            dataset_clips={"ds1": medias},
            seeds=[1, 2, 3],
            categories={"ds1": ["alpha"]},
            calibrate_count=1,
        )
        assert set(df["seed"].unique()) == {1, 2, 3}

    def test_multiple_categories(self):
        medias = _make_separable_clips(n_per_cat=6)
        df = run_voting_iterations_eval(
            dataset_clips={"ds1": medias},
            seeds=[42],
            categories={"ds1": ["alpha", "beta"]},
            calibrate_count=1,
        )
        assert set(df["category"].unique()) == {"alpha", "beta"}

    def test_auto_categories(self):
        """When categories=None, all unique categories are used."""
        medias = _make_three_category_clips(n_per_cat=6)
        df = run_voting_iterations_eval(
            dataset_clips={"ds1": medias},
            seeds=[42],
            calibrate_count=1,
        )
        assert set(df["category"].unique()) == {"alpha", "beta", "gamma"}

    def test_multiple_datasets(self):
        clips1 = _make_separable_clips(n_per_cat=6, seed=0)
        clips2 = _make_separable_clips(n_per_cat=6, seed=1)
        df = run_voting_iterations_eval(
            dataset_clips={"ds1": clips1, "ds2": clips2},
            seeds=[42],
            categories={"ds1": ["alpha"], "ds2": ["beta"]},
            calibrate_count=1,
        )
        assert set(df["dataset"].unique()) == {"ds1", "ds2"}

    def test_cost_column_numeric(self):
        medias = _make_separable_clips(n_per_cat=6)
        df = run_voting_iterations_eval(
            dataset_clips={"ds1": medias},
            seeds=[42],
            categories={"ds1": ["alpha"]},
            calibrate_count=1,
        )
        assert df["cost"].dtype == np.float64
        assert df["fpr"].dtype == np.float64
        assert df["fnr"].dtype == np.float64

    def test_full_cross_product_shape(self):
        """2 seeds x 1 dataset x 2 categories -> each combo produces rows."""
        medias = _make_separable_clips(n_per_cat=6)
        df = run_voting_iterations_eval(
            dataset_clips={"ds1": medias},
            seeds=[1, 2],
            categories={"ds1": ["alpha", "beta"]},
            calibrate_count=1,
        )
        combos = df.groupby(["seed", "dataset", "category"]).ngroups
        assert combos == 4  # 2 seeds x 1 dataset x 2 categories


# ------------------------------------------------------------------
# Region voting (patch datasets)
# ------------------------------------------------------------------

_PATCH_DIM = 8
_GRID = 3  # 3x3 patch grid


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n else v


def _patch_media(media_id, positive, *, category, with_box=True):
    """A synthetic patch-embedder media (a raw ``patch_grid``).

    Positive media have grid cells pointing along ``+e0`` and a ground-truth
    box; negatives point along ``-e0`` and carry no box.  Separable so the MLP
    trains cleanly without flakiness.
    """
    rng = np.random.default_rng(media_id)
    sign = 1.0 if positive else -1.0
    grid = np.zeros((_GRID, _GRID, _PATCH_DIM), dtype=np.float32)
    for r in range(_GRID):
        for c in range(_GRID):
            base = np.zeros(_PATCH_DIM, dtype=np.float32)
            base[0] = sign
            grid[r, c] = _unit(base + rng.standard_normal(_PATCH_DIM).astype(np.float32) * 0.05)
    img_vec = _unit(grid.reshape(-1, _PATCH_DIM).mean(axis=0))

    media = {
        "id": media_id,
        "media_type": "image",
        "embedder": "dinov3_patch",
        "embeddings": {"dinov3_patch": img_vec},
        "patch_grid": grid,
        "category": category,
    }
    if positive and with_box:
        media["regions"] = [{"box": [0.0, 0.0, 2 / 3, 1.0], "label": category}]
    return media


def _make_patch_clips(n_per_cat=10):
    medias = {}
    media_id = 1
    for _ in range(n_per_cat):
        medias[media_id] = _patch_media(media_id, positive=True, category="apple")
        media_id += 1
    for _ in range(n_per_cat):
        medias[media_id] = _patch_media(media_id, positive=False, category="other")
        media_id += 1
    return medias


class TestGoodTrainingVec:
    """The per-Good-vote training vector, region-pooled or whole-image."""

    def test_image_level_when_region_voting_off(self):
        from vtscore.embedding.media_vectors import media_embedding

        media = _patch_media(1, positive=True, category="apple")
        vec = good_training_vec(media, "apple", region_voting=False)
        np.testing.assert_allclose(vec, media_embedding(media))

    def test_takes_the_nearest_patch_when_region_voting_on(self):
        """With a ``patch_grid`` present, the simulated region vote trains on the
        raw patch nearest the ground-truth box - the same path the live vote
        flow takes, not a fresh uniform grid pool."""
        from vtscore.media.patch_embed import nearest_patch_to_box

        media = _patch_media(1, positive=True, category="apple")
        vec = good_training_vec(media, "apple", region_voting=True)
        expected = nearest_patch_to_box(np.asarray(media["patch_grid"]), (0.0, 0.0, 2 / 3, 1.0))
        np.testing.assert_allclose(vec, expected)
        # The chosen vector is one of the grid's actual patch vectors, i.e. a
        # row the scorer max-pools.
        flat = np.asarray(media["patch_grid"], dtype=np.float32).reshape(-1, _PATCH_DIM)
        assert any(np.allclose(vec, row) for row in flat)

    def test_falls_back_without_patch_grid(self):
        from vtscore.embedding.media_vectors import media_embedding

        media = _patch_media(1, positive=True, category="apple")
        del media["patch_grid"]
        vec = good_training_vec(media, "apple", region_voting=True)
        np.testing.assert_allclose(vec, media_embedding(media))

    def test_falls_back_without_matching_box(self):
        from vtscore.embedding.media_vectors import media_embedding

        # Positive image but no annotated box for this category.
        media = _patch_media(1, positive=True, category="apple", with_box=False)
        vec = good_training_vec(media, "apple", region_voting=True)
        np.testing.assert_allclose(vec, media_embedding(media))


class TestRegionVotingSimulate:
    """End-to-end region voting on a synthetic patch dataset."""

    def test_region_voting_produces_finite_rows(self):
        medias = _make_patch_clips(n_per_cat=10)
        rows = simulate_voting_iterations(medias, target_category="apple", seed=0, region_voting=True)
        assert rows  # region-aware scoring path runs end-to-end
        for row in rows:
            assert np.isfinite(row["cost"])
            assert np.isfinite(row["fpr"])
            assert np.isfinite(row["fnr"])

    def test_baseline_on_patch_data_also_scores_region_aware(self):
        # region_voting=False still works on a patch dataset: Good votes train
        # whole-image, but scoring max-pools over the patch rows (live inference).
        medias = _make_patch_clips(n_per_cat=10)
        rows = simulate_voting_iterations(medias, target_category="apple", seed=0, region_voting=False)
        assert rows
        assert all(np.isfinite(r["cost"]) for r in rows)

    def test_run_eval_threads_region_voting_flag(self):
        medias = _make_patch_clips(n_per_cat=8)
        df = run_voting_iterations_eval(
            dataset_clips={"vg": medias},
            seeds=[0],
            categories={"vg": ["apple"]},
            region_voting=True,
        )
        assert not df.empty
        assert (df["category"] == "apple").all()


# ------------------------------------------------------------------
# Autopilot vote-order strategy
# ------------------------------------------------------------------


class TestAutopilotStrategy:
    """``strategy=`` / ``strategies=`` / ``max_steps=`` axes and the result column."""

    def test_default_strategy_is_autopilot(self):
        medias = _make_separable_clips(n_per_cat=6)
        rows = simulate_voting_iterations(medias, "alpha", seed=42, calibrate_count=1)
        assert rows
        assert all(r["strategy"] == "autopilot" for r in rows)

    def test_produces_finite_rows(self):
        medias = _make_separable_clips(n_per_cat=8, dim=12)
        rows = simulate_voting_iterations(
            medias,
            "alpha",
            seed=3,
            calibrate_count=1,
            max_steps=10,
            atlas_min_node_size=3,
        )
        assert rows  # the autopilot flow drives the loop end-to-end
        for row in rows:
            assert row["strategy"] == "autopilot"
            assert np.isfinite(row["cost"])
            assert np.isfinite(row["fpr"])
            assert np.isfinite(row["fnr"])
            # One vote per step, so the counts still sum to t.
            assert row["n_good"] + row["n_bad"] == row["t"]

    def test_seeds_the_initial_goods_before_bads(self):
        # No text sort: autopilot hands the tool known-good examples first, so
        # the first trainable step already carries the full 3-good seed and its
        # first bad (a 3-vs-1 model), never a 1-vs-1 warm-up.
        medias = _make_separable_clips(n_per_cat=12)
        rows = simulate_voting_iterations(medias, "alpha", seed=0, calibrate_count=1)
        assert rows
        assert rows[0]["n_good"] == 3
        assert rows[0]["n_bad"] == 1

    def test_max_steps_caps_votes(self):
        medias = _make_separable_clips(n_per_cat=20)
        rows = simulate_voting_iterations(medias, "alpha", seed=1, calibrate_count=1, max_steps=6)
        assert rows
        # No row can reflect more than max_steps votes cast.
        assert max(r["t"] for r in rows) <= 6

    def test_determinism(self):
        medias = _make_separable_clips(n_per_cat=10)
        a = simulate_voting_iterations(medias, "alpha", seed=7, calibrate_count=1, max_steps=10)
        b = simulate_voting_iterations(medias, "alpha", seed=7, calibrate_count=1, max_steps=10)
        assert [r["t"] for r in a] == [r["t"] for r in b]
        assert [r["cost"] for r in a] == [r["cost"] for r in b]

    def test_unknown_strategy_raises(self):
        medias = _make_separable_clips(n_per_cat=6)
        with pytest.raises(KeyError):
            simulate_voting_iterations(medias, "alpha", seed=1, strategy="does_not_exist", calibrate_count=1)

    def test_text_seed_scores_change_the_ordering(self):
        # Supplying a text-sort ranking routes the seed through the text path;
        # a ranking that inverts the ground-truth order produces a different
        # vote sequence (and cost curve) than the no-text random-good seed.
        medias = _make_separable_clips(n_per_cat=12)
        no_text = simulate_voting_iterations(medias, "alpha", seed=5, calibrate_count=1, max_steps=10)
        # Rank every media by a synthetic "similarity" = its id, so the text
        # seed walks the pool in a fixed, non-random order.
        seed_scores = {cid: float(cid) for cid in medias}
        with_text = simulate_voting_iterations(
            medias, "alpha", seed=5, calibrate_count=1, max_steps=10, seed_scores=seed_scores
        )
        assert with_text
        assert [r["cost"] for r in with_text] != [r["cost"] for r in no_text]

    def test_run_eval_defaults_to_autopilot(self):
        medias = _make_separable_clips(n_per_cat=6)
        df = run_voting_iterations_eval(
            dataset_clips={"ds1": medias},
            seeds=[1],
            categories={"ds1": ["alpha"]},
            calibrate_count=1,
        )
        assert set(df["strategy"].unique()) == {"autopilot"}

    def test_run_eval_threads_seed_scores(self):
        medias = _make_separable_clips(n_per_cat=8)
        seed_scores = {"ds1": {"alpha": {cid: float(cid) for cid in medias}}}
        df = run_voting_iterations_eval(
            dataset_clips={"ds1": medias},
            seeds=[1],
            categories={"ds1": ["alpha"]},
            calibrate_count=1,
            max_steps=10,
            seed_scores=seed_scores,
        )
        assert not df.empty
        assert set(df["strategy"].unique()) == {"autopilot"}
