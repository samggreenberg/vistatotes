"""The harness's **default** head is the production head (#2799, #2916).

The calibration/voting harness historically trained a small auto-sized MLP,
then the linear (logistic) head; the live detector now trains a **linear SVM**.
Measuring a shipped default like ``safe_thresholds`` on the wrong head measures
the wrong product, so an unspecified ``head`` resolves to
:data:`~vtscore.eval.voting_iterations.PRODUCTION_HEAD` and ``head="linear"`` /
``head="mlp"`` are the explicitly-named legacy arms.  (``head`` picks what the
app-pipeline trainer fits; the ``trainer`` knob beside it picks the pipeline -
see issue #3764.)

These tests pin the three things that make a default run faithful:

* the resolved default really is the head the *app* trains — pinned against
  ``train_and_threshold`` itself, so flipping the shipped head fails the suite
  instead of silently leaving the harness behind;
* the head reaches **both** the final model and the calibration folds (the
  folds must share the final model's fit — production threads one sentinel
  through ``_train_and_score_xy`` for exactly this reason);
* the final per-step model really is a single ``Linear(d, 1)`` on the
  whole-image path *and* the region/style path.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch.nn as nn

import vtscore.eval.step_model as step_model
import vtscore.eval.step_trainers as step_trainers
import vtscore.eval.voting_iterations as vi
from vtscore.eval.patch_styles import resolve_style
from vtscore.eval.voting_columns import IDENT_COLUMNS
from vtscore.training.mlp import LINEAR_HEAD, LINEAR_SVM_HEAD, _auto_hidden_dim

from .test_max_patch_style import DIM, _planted_dataset


def _votes(medias, n=6):
    goods = {m["id"]: None for m in list(medias.values()) if m["category"] == "cat0"}
    bads = {m["id"]: None for m in list(medias.values()) if m["category"] == "cat1"}
    return dict(list(goods.items())[:n]), dict(list(bads.items())[:n])


def test_resolve_hidden_dim_maps_heads_to_sentinels():
    assert step_model.resolve_hidden_dim("linear_svm", 40) == LINEAR_SVM_HEAD
    assert step_model.resolve_hidden_dim("linear", 40) == LINEAR_HEAD == 0
    assert step_model.resolve_hidden_dim("mlp", 40) == _auto_hidden_dim(40)
    with pytest.raises(ValueError, match="unknown head"):
        step_model.resolve_hidden_dim("logreg", 40)


def test_resolve_trainer_name_normalises_the_retired_mlp_spelling():
    """#3764: ``"mlp"`` named the app pipeline, whose default head is the SVM.

    The alias is accepted on input so archived launch scripts keep running;
    everything downstream sees exactly one spelling.
    """
    assert step_model.resolve_trainer_name("mlp") == step_model.APP_TRAINER == "app"
    assert step_model.resolve_trainer_name("app") == "app"
    # Standalone estimators pass through untouched - they are not the app arm.
    assert step_model.resolve_trainer_name("svm_linear") == "svm_linear"
    assert step_model.resolve_trainer_name("svm_rbf@C=3") == "svm_rbf@C=3"


def test_the_default_head_is_the_head_the_app_trains(monkeypatch):
    """#2916: ``PRODUCTION_HEAD`` must track ``train_and_threshold``'s own width.

    Pinned by *running* the app's pipeline rather than by restating its
    constant: if the shipped detector ever changes head, this fails and the
    harness default has to move with it, instead of the default arm quietly
    measuring a detector nobody ships.
    """
    import vtscore.training as app_training_pkg
    from vtscore.detectors.training import train_and_threshold

    seen: list = []
    real_train = app_training_pkg.train_model

    def spy_train(X, y, input_dim, **kw):
        seen.append(kw.get("hidden_dim"))
        return real_train(X, y, input_dim, **kw)

    # ``train_and_threshold`` imports ``train_model`` from the package at call
    # time, so the package attribute is the binding it will actually use.
    monkeypatch.setattr(app_training_pkg, "train_model", spy_train)

    rng = np.random.default_rng(0)
    X_list = [rng.standard_normal(8).astype(np.float32) for _ in range(8)]
    y_list = [1.0] * 4 + [0.0] * 4
    train_and_threshold(X_list, y_list)

    assert seen, "the app never trained a model"
    assert set(seen) == {step_model.resolve_hidden_dim(step_model.PRODUCTION_HEAD, len(y_list))}


def test_unknown_head_is_rejected_early():
    medias, _ = _planted_dataset(n_per_cat=6, seed=0)
    with pytest.raises(ValueError, match="unknown head"):
        vi.simulate_voting_iterations(medias, target_category="cat0", seed=0, head="logreg", max_steps=1)


def test_head_does_not_apply_to_the_svm_trainer():
    medias, _ = _planted_dataset(n_per_cat=6, seed=0)
    with pytest.raises(ValueError, match="only applies to the production trainer"):
        vi.simulate_voting_iterations(
            medias, target_category="cat0", seed=0, trainer="svm_linear", head="linear", max_steps=1
        )


@pytest.mark.parametrize("head,sentinel", [("linear_svm", LINEAR_SVM_HEAD), ("linear", LINEAR_HEAD)])
@pytest.mark.parametrize("style", [None, "max_patch"])
def test_linear_head_reaches_the_final_model_and_the_calibration_folds(style, head, sentinel, monkeypatch):
    """A linear head must reach the fit *and* the folds under the same sentinel."""
    medias, _ = _planted_dataset(n_per_cat=10, seed=0)
    good_votes, bad_votes = _votes(medias)

    seen: dict[str, list] = {"train": [], "calib": []}
    real_train = step_trainers.train_model

    def spy_train(X, y, input_dim, **kw):
        seen["train"].append(kw.get("hidden_dim"))
        return real_train(X, y, input_dim, **kw)

    monkeypatch.setattr(step_trainers, "train_model", spy_train)
    for name in ("calibration_folds", "compute_grouped_fold_node_scores"):
        real = getattr(step_trainers, name)

        def spy_calib(*args, _real=real, **kw):
            seen["calib"].append(kw.get("hidden_dim"))
            return _real(*args, **kw)

        monkeypatch.setattr(step_trainers, name, spy_calib)

    style_obj = None if style is None else resolve_style(style)

    step, threshold, _n, _timings, _details = step_trainers._train_and_calibrate(
        "mlp",
        good_votes,
        bad_votes,
        medias,
        "cat0",
        region_voting=True,
        input_dim=DIM,
        inclusion=0,
        calibrate_count=2,
        calibration_fraction=0.5,
        head=head,
        style_obj=style_obj,
    )

    assert seen["train"], "the final model was never trained"
    assert set(seen["train"]) == {sentinel}
    assert seen["calib"], "the calibration folds were never fitted"
    assert set(seen["calib"]) == {sentinel}
    # A single Linear(d, 1) with no hidden layer.
    assert step.torch_model is not None
    layers = [m for m in step.torch_model if isinstance(m, nn.Linear)]
    assert len(layers) == 1
    assert layers[0].out_features == 1
    assert not any(isinstance(m, nn.ReLU) for m in step.torch_model)
    assert np.isfinite(threshold)


def _default_arm_step(medias):
    good_votes, bad_votes = _votes(medias)
    return step_trainers._train_and_calibrate(
        "mlp",
        good_votes,
        bad_votes,
        medias,
        "cat0",
        region_voting=True,
        input_dim=DIM,
        inclusion=0,
        calibrate_count=2,
        calibration_fraction=0.5,
    )


def test_the_default_arm_trains_the_production_head():
    """#2916: no explicit head == the shipped head, not the retired MLP."""
    medias, _ = _planted_dataset(n_per_cat=10, seed=0)

    step, _threshold, _n_labels, _t, _d = _default_arm_step(medias)
    assert step.torch_model is not None
    layers = [m for m in step.torch_model if isinstance(m, nn.Linear)]
    assert len(layers) == 1
    assert not any(isinstance(m, nn.ReLU) for m in step.torch_model)


def test_the_mlp_arm_is_still_reachable_by_name():
    """The legacy #2781 head stays available — just not as the default."""
    medias, _ = _planted_dataset(n_per_cat=10, seed=0)
    good_votes, bad_votes = _votes(medias)

    step, _threshold, n_labels, _t, _d = step_trainers._train_and_calibrate(
        "mlp",
        good_votes,
        bad_votes,
        medias,
        "cat0",
        region_voting=True,
        input_dim=DIM,
        inclusion=0,
        calibrate_count=2,
        calibration_fraction=0.5,
        head="mlp",
    )
    assert step.torch_model is not None
    layers = [m for m in step.torch_model if isinstance(m, nn.Linear)]
    assert len(layers) == 2
    assert layers[0].out_features == _auto_hidden_dim(n_labels)


def test_default_runs_record_the_production_head():
    """A run that names no head must report — and train — the shipped one."""
    medias, _ = _planted_dataset(n_per_cat=20, seed=0)
    rows = vi.simulate_voting_iterations(medias, target_category="cat0", seed=0, max_steps=8)
    assert rows, "no rows produced"
    assert {r["head"] for r in rows} == {step_model.PRODUCTION_HEAD}


def test_rows_record_the_head_and_linear_runs_end_to_end():
    medias, _ = _planted_dataset(n_per_cat=30, seed=0)
    rows = vi.simulate_voting_iterations(
        medias,
        target_category="cat0",
        seed=0,
        dataset_name="planted",
        inclusion=0,
        region_voting=True,
        safe_thresholds=True,
        max_steps=12,
        style="max_patch",
        head="linear_svm",
        emit_calibration_metrics=True,
    )
    assert rows, "no rows produced"
    assert {r["head"] for r in rows} == {"linear_svm"}
    assert "head" in IDENT_COLUMNS
    assert step_model.PRODUCTION_HEAD == "linear_svm"
    # The #2799 variant rows still ride along under safe_thresholds.
    assert {r["gmm_variant"] for r in rows} >= {"", "xcal_only", "pooled_cross"}
    for r in rows:
        assert np.isfinite(r["threshold"])
