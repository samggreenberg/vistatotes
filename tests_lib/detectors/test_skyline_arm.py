"""The supervised-skyline arms and the training-regret decomposition (issue #3322).

``oracle_cost`` conflates two very different causes: no linear head in this
embedding can separate the class (the **floor**), and a head could but 10-200
clicks did not find it (the **shortfall**).  The skyline arm splits them, so
``cost = skyline_oracle_cost + training_regret + regret``.

These tests pin the three properties that make that split mean anything: the
telescope is *exact*, the skyline is *vote-independent* (one row per run, and
turning it on cannot move a single vote), and the bracket arm is genuinely
**cross-fitted** rather than a train-on-test fit wearing the word "skyline".

Everything runs on small synthetic single-vector datasets - no model downloads.
"""

from __future__ import annotations

import numpy as np
import pytest

from vtscore.eval.patch_styles import WholeImageStyle
from vtscore.eval.voting_columns import CALIBRATION_COLUMNS, SKYLINE_COLUMNS, TIMING_COLUMNS
from vtscore.eval.voting_iterations import (
    SKYLINE_ARMS,
    SKYLINE_PROVENANCE,
    SKYLINE_TEST_XFIT,
    SKYLINE_TRAIN_FULL,
    _WHOLE_IMAGE_STYLE,
    _skyline_arm_rows,
    _skyline_fit_and_score,
    simulate_voting_iterations,
)

from .test_max_patch_style import _planted_dataset

DIM = 64


def _blob_dataset(n_per_cat=30, seed=0, separation=3.0, dim=DIM):
    """Two Gaussian blobs, linearly separable at *separation* - a learnable class."""
    rng = np.random.default_rng(seed)
    direction = np.zeros(dim, dtype=np.float32)
    direction[0] = 1.0
    medias: dict[int, dict] = {}
    mid = 1
    for cat, sign in (("cat0", 1.0), ("cat1", -1.0)):
        for _ in range(n_per_cat):
            vec = rng.normal(0, 1.0, dim).astype(np.float32) + sign * separation * direction
            medias[mid] = {"id": mid, "category": cat, "embeddings": {"emb": vec}}
            mid += 1
    return medias


def _noise_dataset(n_per_cat=25, seed=0, dim=DIM):
    """Labels independent of features, with ``dim`` comparable to the sample size.

    This is the fixture the cross-fitting check needs: a linear head with ``dim``
    free parameters shatters near-arbitrary labelings of a sample this size, so a
    naive train-on-test skyline reports near-zero cost on a class nothing can
    learn.  A cross-fitted one cannot.
    """
    rng = np.random.default_rng(seed)
    medias: dict[int, dict] = {}
    mid = 1
    for cat in ("cat0", "cat1"):
        for _ in range(n_per_cat):
            vec = rng.normal(0, 1.0, dim).astype(np.float32)
            medias[mid] = {"id": mid, "category": cat, "embeddings": {"emb": vec}}
            mid += 1
    return medias


def _run(medias, *, arms=(SKYLINE_TRAIN_FULL,), seed=0, max_steps=14, **kw):
    return simulate_voting_iterations(
        medias,
        target_category="cat0",
        seed=seed,
        dataset_name="synthetic",
        inclusion=0,
        safe_thresholds=False,
        max_steps=max_steps,
        style="whole_image",
        emit_calibration_metrics=True,
        skyline_arms=list(arms),
        **kw,
    )


def _split(rows):
    """``(mortal rows, skyline rows)``."""
    return (
        [r for r in rows if not r["gmm_variant"]],
        [r for r in rows if r["gmm_variant"] in SKYLINE_ARMS],
    )


# ---------------------------------------------------------------------------
# Emission shape
# ---------------------------------------------------------------------------


def test_style_constant_matches_the_style_registry():
    """The spelled-out name is pinned against the class it stands for."""
    assert _WHOLE_IMAGE_STYLE == WholeImageStyle.name


def test_one_row_per_arm_per_run_at_step_zero():
    rows = _run(_blob_dataset(), arms=SKYLINE_ARMS)
    mortal, skyline = _split(rows)
    assert mortal, "no mortal rows produced"
    assert [r["gmm_variant"] for r in skyline] == [SKYLINE_TRAIN_FULL, SKYLINE_TEST_XFIT]
    for r in skyline:
        # A skyline belongs to no step, so it is given a step index no
        # trajectory can occupy rather than being duplicated onto all of them.
        assert r["t"] == 0
        assert r["app_trained"] == 0
        assert r["threshold_provenance"] == SKYLINE_PROVENANCE
        assert set(CALIBRATION_COLUMNS).issubset(r.keys())
    assert all(r["t"] >= 1 for r in mortal)


def test_full_supervision_is_the_whole_simulation_split():
    medias = _blob_dataset(n_per_cat=30)
    rows = _run(medias)
    _mortal, skyline = _split(rows)
    row = skyline[0]
    assert row["n_good"] + row["n_bad"] == row["n_haystack"]
    # Full supervision leaves nothing in the haystack unlabelled.
    assert row["n_remainder"] == 0


def test_skyline_is_vote_independent():
    """The decomposition columns are one constant of the run, not a per-step number."""
    rows = _run(_blob_dataset())
    mortal, skyline = _split(rows)
    floor = skyline[0]["oracle_cost"]
    assert {r["skyline_oracle_cost"] for r in mortal} == {floor}
    assert len({r["skyline_oracle_cost_honest"] for r in mortal}) == 1


def test_learnable_class_has_a_low_floor():
    """A separable blob is learnable, so the floor is near zero and the loop's shortfall is the rest."""
    rows = _run(_blob_dataset(separation=4.0))
    _mortal, skyline = _split(rows)
    assert skyline[0]["oracle_cost"] < 0.05
    assert skyline[0]["average_precision"] > 0.95


def test_columns_are_nan_without_the_arm():
    rows = _run(_blob_dataset(), arms=())
    assert rows
    assert all(not r["gmm_variant"].startswith("skyline") for r in rows)
    for r in rows:
        for col in SKYLINE_COLUMNS:
            assert np.isnan(r[col])


# ---------------------------------------------------------------------------
# The telescope
# ---------------------------------------------------------------------------


def test_three_term_decomposition_telescopes_exactly():
    rows = _run(_blob_dataset(), arms=SKYLINE_ARMS)
    assert rows
    for r in rows:
        assert r["skyline_oracle_cost"] + r["training_regret"] + r["regret"] == pytest.approx(r["cost"], abs=1e-5)
        # #3116's honest reference re-bases the same split, and telescopes too.
        assert r["skyline_oracle_cost_honest"] + r["training_regret_honest"] + r["regret_honest"] == pytest.approx(
            r["cost"], abs=1e-5
        )
        # `training_regret` is defined on RANKINGS - a difference of oracle
        # costs - which is exactly what makes the identity above exact.  Routing
        # the skyline through a calibrated cut would re-mix in `regret`.
        assert r["training_regret"] == pytest.approx(r["oracle_cost"] - r["skyline_oracle_cost"], abs=1e-5)


def test_primary_arm_is_its_own_reference():
    rows = _run(_blob_dataset(), arms=SKYLINE_ARMS)
    _mortal, skyline = _split(rows)
    ref = next(r for r in skyline if r["gmm_variant"] == SKYLINE_TRAIN_FULL)
    assert ref["training_regret"] == pytest.approx(0.0, abs=1e-9)
    # A skyline row is a statement about a ranking: its cut is the test oracle's,
    # so `cost == oracle_cost` and `regret == 0` BY CONSTRUCTION on that row.
    assert ref["cost"] == pytest.approx(ref["oracle_cost"], abs=1e-6)
    assert ref["regret"] == pytest.approx(0.0, abs=1e-6)


def test_negative_training_regret_is_not_clamped():
    """A mortal step that out-ranks the skyline is information, not a bug."""
    # The whole-image column of a planted-PATCH dataset: the CLS vector averages
    # the signal away, so the image-labelled skyline is genuinely weak and some
    # step beats it.  Nothing in the harness may clamp that to zero.
    medias, _ = _planted_dataset(n_per_cat=30, seed=0)
    rows = _run(medias, region_voting=True)
    mortal, skyline = _split(rows)
    assert skyline, "no skyline row"
    assert any(r["training_regret"] < 0 for r in mortal), "fixture no longer produces a negative regret"


# ---------------------------------------------------------------------------
# The arm cannot move the run it describes
# ---------------------------------------------------------------------------


def test_skyline_does_not_perturb_the_trajectory():
    """Turning the arm on must not move a single vote, threshold or metric."""
    medias = _blob_dataset()
    without, _ = _split(_run(medias, arms=()))
    with_sky, _ = _split(_run(medias, arms=SKYLINE_ARMS))
    assert len(without) == len(with_sky)
    ignore = TIMING_COLUMNS | set(SKYLINE_COLUMNS)
    for a, b in zip(without, with_sky, strict=True):
        for key in set(a) | set(b):
            if key in ignore:
                continue
            va, vb = a[key], b[key]
            if isinstance(va, float) and np.isnan(va):
                assert isinstance(vb, float) and np.isnan(vb), key
            else:
                assert va == vb, key


# ---------------------------------------------------------------------------
# Cross-fitting
# ---------------------------------------------------------------------------


def _xfit_row(medias, seed=0):
    """Run the bracket arm alone, straight through the helper, on *medias*' test half."""
    # Interleaved, not sliced: the fixtures lay one category out after the
    # other, so a contiguous halving would hand the arm a single-class test set.
    ids = sorted(medias)
    sim_ids, test_ids = ids[0::2], ids[1::2]
    rows = _skyline_arm_rows(
        [SKYLINE_TEST_XFIT],
        medias,
        "cat0",
        sim_ids,
        test_ids,
        0,
        trainer="app",
        head="linear_svm",
        style_obj=WholeImageStyle(),
        region_voting=False,
        input_dim=DIM,
        calibrate_count=2,
        calibration_fraction=0.5,
        seed=seed,
    )
    return rows, test_ids


def test_bracket_arm_is_cross_fitted_not_trained_on_test():
    """A naive test-side fit shatters random labels; the shipped arm must not.

    This is the failure the arm exists to avoid: a ~d-parameter linear head on a
    test set of comparable size reports near-zero cost on a class nothing can
    learn, so its "regret" would measure ``d / n_test`` rather than learnability.
    """
    medias = _noise_dataset(n_per_cat=25)
    rows, test_ids = _xfit_row(medias)
    assert len(rows) == 1
    xfit_cost = rows[0]["oracle_cost"]

    # The naive cheat, built here so the comparison is against a measured number
    # rather than an asserted constant: fit on the whole test set, score it.
    pos = [c for c in test_ids if medias[c]["category"] == "cat0"]
    neg = [c for c in test_ids if c not in set(pos)]
    score_map, _step, _timings, _secs = _skyline_fit_and_score(
        pos,
        neg,
        test_ids,
        medias,
        "cat0",
        trainer="app",
        head="linear_svm",
        style_obj=WholeImageStyle(),
        region_voting=False,
        input_dim=DIM,
        inclusion=0,
        calibrate_count=2,
        calibration_fraction=0.5,
    )
    from vtscore.eval.calibration_metrics import inclusion_weights, oracle_cut

    wf, wn = inclusion_weights(0)
    scores = np.array([score_map[c] for c in test_ids], dtype=np.float64)
    labels = np.array([1.0 if medias[c]["category"] == "cat0" else 0.0 for c in test_ids])
    _thr, naive_cost, _f, _fn = oracle_cut(scores, labels, wf, wn)

    assert naive_cost < 0.1, "fixture no longer shatters in-sample; the leak check would be vacuous"
    assert xfit_cost > naive_cost + 0.25, (
        f"cross-fitted cost {xfit_cost} is too close to the in-sample cheat {naive_cost}: "
        "the bracket arm looks like it is scoring items its head was trained on"
    )


def test_bracket_arm_ranks_a_learnable_class():
    """Cross-fitting is honest, not blind: real signal still comes through."""
    rows, _ = _xfit_row(_blob_dataset(n_per_cat=30, separation=4.0))
    assert len(rows) == 1
    assert rows[0]["average_precision"] > 0.9


def test_bracket_arm_is_dropped_when_it_cannot_be_cross_fitted():
    """Too few test items to fold honestly leaves the row out, not a naive fit in."""
    medias = _blob_dataset(n_per_cat=2)
    rows, _ = _xfit_row(medias)
    assert rows == []


# ---------------------------------------------------------------------------
# Scope and validation
# ---------------------------------------------------------------------------


def test_patch_column_is_skipped_loudly():
    """v1 is the whole-image column; a patch column's skyline is #3321's open item."""
    medias, _ = _planted_dataset(n_per_cat=25, seed=0)
    with pytest.warns(RuntimeWarning, match="skyline"):
        rows = simulate_voting_iterations(
            medias,
            target_category="cat0",
            seed=0,
            dataset_name="planted",
            inclusion=0,
            region_voting=True,
            safe_thresholds=False,
            max_steps=10,
            style="max_patch",
            emit_calibration_metrics=True,
            skyline_arms=[SKYLINE_TRAIN_FULL],
        )
    assert rows
    assert all(r["gmm_variant"] not in SKYLINE_ARMS for r in rows)
    assert all(np.isnan(r["skyline_oracle_cost"]) for r in rows)


def test_unknown_arm_name_is_rejected():
    with pytest.raises(ValueError, match="unknown skyline arm"):
        _run(_blob_dataset(), arms=("skyline_train_ful",))


def test_arm_requires_the_calibration_frame():
    with pytest.raises(ValueError, match="emit_calibration_metrics"):
        simulate_voting_iterations(
            _blob_dataset(),
            target_category="cat0",
            seed=0,
            style="whole_image",
            emit_calibration_metrics=False,
            skyline_arms=[SKYLINE_TRAIN_FULL],
        )
