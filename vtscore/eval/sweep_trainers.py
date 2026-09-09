"""Standalone estimator registry for the label-curve and timing sweeps.

A **sweep trainer** is a self-contained estimator: a callable
``(X_train, y_train, seed) -> predict_fn`` where the returned ``predict_fn``
maps an ``(N, D)`` embedding matrix to per-row ``P(positive)`` scores in
``[0, 1]`` (ensembles additionally return a per-item std; see
:data:`PredictFn`).  It owns its own fit and its own scoring, and knows nothing
about VTSearch's detector pipeline.  :data:`SWEEP_TRAINERS` holds the
fixed-name entries; :func:`resolve_trainer` additionally parses
**parameterised** SVM names such as ``"svm_rbf@C=3,gamma=scale"`` so the
kernel/hyperparameter screen can sweep the SVM configuration space without a
registry entry per point.

**This is not the voting simulation's ``trainer`` knob.**  The two registries
are deliberately named apart (issue #3764), because they answer different
questions and their names used to collide:

* Here — :mod:`vtscore.eval.label_curve` and :mod:`vtscore.eval.timing_benchmark`
  ask *how does this estimator rank, given N labels?*, so every arm is a bare
  estimator and ``"mlp"`` really is an MLP.
* There — :mod:`vtscore.eval.step_trainers`, driven by
  :mod:`vtscore.eval.voting_iterations`, asks *what does VTSearch do with these
  votes?*, so its ``trainer="app"`` arm is the shipped pipeline (whose head is
  picked separately by ``head=``) and its ``svm_*`` arms are the estimators
  here, wrapped in a threshold rule.

Both sweeps share the parameterised SVM parser and the trainer-agnostic
cross-calibration threshold below, which is why this module is imported from
both.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


PredictFn = Callable[[np.ndarray], "np.ndarray | tuple[np.ndarray, np.ndarray]"]
"""Callable returning per-row P(positive) - optionally with per-item std.

A plain trainer returns just ``scores`` (P(positive) in ``[0, 1]``).  An
*ensemble* trainer returns ``(scores, per_item_std)`` where ``scores`` is
the mean sigmoid across ensemble members and ``per_item_std`` is the
member-to-member standard deviation - a cheap epistemic-uncertainty
signal.  Callers that only need the ranking pull ``scores`` out via
:func:`_as_scores`; callers that want the uncertainty (the diagnostic
``std_mean`` column) unpack the tuple explicitly.
"""

TrainerFn = Callable[[np.ndarray, np.ndarray, int], PredictFn]
"""Trainer signature: ``(X_train, y_train, seed) -> predict_fn``."""


def _as_scores(result: "np.ndarray | tuple[np.ndarray, np.ndarray]") -> np.ndarray:
    """Return the score array from a ``predict()`` result.

    Ensemble trainers return ``(scores, per_item_std)``; every other
    trainer returns a bare score array.  This collapses both to the score
    array so ranking metrics and threshold calibration don't have to care
    which trainer produced the prediction.
    """
    if isinstance(result, tuple):
        return np.asarray(result[0], dtype=np.float64)
    return np.asarray(result, dtype=np.float64)


def _train_mlp(X: np.ndarray, y: np.ndarray, seed: int) -> PredictFn:
    """Adapt :func:`vtscore.training.mlp.train_model` to the sweep API."""
    import torch  # noqa: PLC0415

    from vtscore.training.mlp import train_model

    X_t = torch.from_numpy(np.asarray(X, dtype=np.float32))
    y_t = torch.from_numpy(np.asarray(y, dtype=np.float32)).unsqueeze(1)
    model = train_model(X_t, y_t, input_dim=X.shape[1], seed=seed)

    def predict(X_test: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X_test, dtype=np.float32)
        with torch.no_grad():
            t = torch.from_numpy(X_arr).to(next(model.parameters()).device)
            return torch.sigmoid(model(t)).squeeze(1).cpu().numpy()

    return predict


def _train_mlp_ensemble_factory(n_seeds: int) -> TrainerFn:
    """Build a trainer that averages *n_seeds* seed-varied MLPs.

    Each member is a full :func:`vtscore.training.mlp.train_model` run on
    the same labels but a different weight-init/dropout seed, so the
    ensemble captures the MLP's epistemic uncertainty (how much the
    decision surface wobbles under reseeding) rather than aleatoric label
    noise.  The returned ``predict`` reports the mean sigmoid as the score
    and the member-to-member standard deviation as ``per_item_std`` - high
    where the members disagree, low where they agree.
    """

    def trainer(X: np.ndarray, y: np.ndarray, seed: int) -> PredictFn:
        import torch  # noqa: PLC0415

        from vtscore.training.mlp import train_model

        X_t = torch.from_numpy(np.asarray(X, dtype=np.float32))
        y_t = torch.from_numpy(np.asarray(y, dtype=np.float32)).unsqueeze(1)
        input_dim = X.shape[1]
        # Distinct seeds per member: ``seed + k`` keeps the whole ensemble a
        # deterministic function of the cell's ``seed`` while decorrelating
        # the members' weight inits.
        models = [train_model(X_t, y_t, input_dim=input_dim, seed=seed + k) for k in range(n_seeds)]

        def predict(X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            X_arr = np.asarray(X_test, dtype=np.float32)
            member_scores: list[np.ndarray] = []
            for model in models:
                with torch.no_grad():
                    t = torch.from_numpy(X_arr).to(next(model.parameters()).device)
                    member_scores.append(torch.sigmoid(model(t)).squeeze(1).cpu().numpy())
            stacked = np.stack(member_scores, axis=0)  # (n_seeds, n_items)
            return stacked.mean(axis=0), stacked.std(axis=0)

        return predict

    return trainer


def _train_svm_factory(kernel: str, **svm_kwargs: Any) -> TrainerFn:
    """Build a sweep-shaped trainer that fits an SVM with the given kernel.

    ``svm_kwargs`` (``C``, ``gamma``, ``gamma_mult``, ``degree``) are forwarded
    to :func:`vtscore.training.svm.train_svm`, so :func:`resolve_trainer` can
    turn a parameterised name into a concrete trainer.
    """

    def trainer(X: np.ndarray, y: np.ndarray, seed: int) -> PredictFn:
        from vtscore.training.svm import train_svm

        clf = train_svm(X, y, kernel=kernel, seed=seed, **svm_kwargs)  # type: ignore[arg-type]
        return clf.predict_proba

    return trainer


# Registry of fixed-name sweep trainers.  Adding a new *named* candidate model
# here is the only place it needs to be plugged in for both sweeps that use this
# registry.  Parameterised SVM configurations do not need an entry - see
# :func:`resolve_trainer`.
#
# ``"mlp"`` here is a genuine MLP: ``vtscore.training.mlp.train_model`` with an
# auto-sized hidden layer, which is the head VTSearch shipped *before* #2790.
# It is emphatically **not** the head the app ships today (the linear SVM,
# ``vtscore.training.mlp.LINEAR_SVM_HEAD``); nothing in this registry is, because
# these arms are bare estimators rather than the app's pipeline.  Read a
# label-curve row's ``trainer`` column with that in mind, and use the voting
# simulation's ``trainer="app"`` arm when the question is about the shipped
# detector.
SWEEP_TRAINERS: dict[str, TrainerFn] = {
    "mlp": _train_mlp,
    "svm_linear": _train_svm_factory("linear"),
    "svm_rbf": _train_svm_factory("rbf"),
    # MLP ensembles: N seed-varied members, mean sigmoid as the score and
    # member disagreement as per-item uncertainty.  Registered at 3/5/7/10
    # members so the sweep can trace how ranking quality and the reported
    # ``std_mean`` move with ensemble size.
    "mlp_ens3": _train_mlp_ensemble_factory(3),
    "mlp_ens5": _train_mlp_ensemble_factory(5),
    "mlp_ens7": _train_mlp_ensemble_factory(7),
    "mlp_ens10": _train_mlp_ensemble_factory(10),
}


# Kernels that :func:`resolve_trainer` accepts as ``svm_<kernel>[@params]``.
_SVM_KERNELS = {"linear": "linear", "rbf": "rbf", "poly": "poly", "sigmoid": "sigmoid"}


def _coerce_param(key: str, value: str) -> Any:
    """Coerce one ``key=value`` token from a parameterised trainer name.

    ``C`` is a float; ``degree`` an int; ``gamma`` is either the sklearn string
    ``"scale"``/``"auto"``, a bare float, or a ``"<mult>x"`` multiplier of the
    ``scale`` heuristic (e.g. ``"4x"`` / ``"0.25x"``) — the last is returned as
    ``("gamma_mult", <float>)`` so the SVM keeps sklearn's data-driven ``scale``
    and merely rescales it.
    """
    if key == "C":
        return ("C", float(value))
    if key == "degree":
        return ("degree", int(value))
    if key == "gamma":
        if value in ("scale", "auto"):
            return ("gamma", value)
        if value.endswith("x"):
            return ("gamma_mult", float(value[:-1]))
        return ("gamma", float(value))
    raise ValueError(f"Unknown SVM trainer parameter {key!r} (expected C, gamma, or degree)")


def _parse_trainer_spec(name: str) -> tuple[str, dict[str, Any]]:
    """Split ``"svm_rbf@C=3,gamma=4x"`` into ``("rbf", {"C": 3.0, "gamma_mult": 4.0})``.

    Raises ``KeyError`` for a name that is neither a registry entry nor a
    recognised ``svm_<kernel>`` spec, so a typo fails loudly.
    """
    base, _, param_str = name.partition("@")
    if not base.startswith("svm_"):
        raise KeyError(
            f"Unknown trainer {name!r}; choices: {sorted(SWEEP_TRAINERS)} or svm_<kernel>[@C=..,gamma=..,degree=..]"
        )
    kernel_key = base[len("svm_") :]
    if kernel_key not in _SVM_KERNELS:
        raise KeyError(f"Unknown SVM kernel {kernel_key!r}; choices: {sorted(_SVM_KERNELS)}")
    kwargs: dict[str, Any] = {}
    if param_str:
        for token in param_str.split(","):
            if not token:
                continue
            key, sep, value = token.partition("=")
            if not sep:
                raise ValueError(f"Malformed trainer parameter {token!r} in {name!r} (expected key=value)")
            k, v = _coerce_param(key.strip(), value.strip())
            kwargs[k] = v
    return _SVM_KERNELS[kernel_key], kwargs


def resolve_trainer(name: str) -> TrainerFn:
    """Return the :class:`TrainerFn` for *name*.

    Accepts both a fixed registry key (``"mlp"``, ``"svm_linear"``, an ensemble)
    and a parameterised SVM spec (``"svm_rbf@C=3,gamma=scale"``,
    ``"svm_poly@degree=2,C=0.3"``, ``"svm_linear@C=0.03"``).  The bare
    ``"svm_linear"`` / ``"svm_rbf"`` names resolve to the registry entries
    unchanged, so existing callers are byte-identical.
    """
    if name in SWEEP_TRAINERS:
        return SWEEP_TRAINERS[name]
    kernel, kwargs = _parse_trainer_spec(name)
    return _train_svm_factory(kernel, **kwargs)


def _cross_calibrated_threshold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    trainer_fn: TrainerFn,
    seed: int,
    *,
    inclusion_value: int = 0,
    calibrate_count: int = 2,
    cal_fraction: float = 0.5,
) -> float:
    """Trainer-agnostic port of ``calculate_cross_calibration_threshold``.

    Mirrors what production does at vote time: split the labels k ways
    into train/cal halves, retrain on each half, score the held-out cal
    half, then pool every fold's (score, label) pairs and apply the
    conformal inclusion rule once.  This is the threshold ``f1_at_xcal``
    is measured at, so a trainer that ranks well but doesn't admit a
    stable cross-validated threshold pays the price here.

    Returns ``0.5`` when the label budget is too small to form valid
    splits (mirrors the production fallback).
    """
    from vtscore.training.thresholds import conformal_threshold

    n = int(y_train.size)
    if n < 4:
        return 0.5
    n_cal = max(1, round(n * cal_fraction))
    n_tr = n - n_cal
    if n_tr < 2 or n_cal < 1:
        return 0.5

    rng = np.random.default_rng(seed)
    pooled_scores: list[float] = []
    pooled_labels: list[float] = []
    for k in range(max(1, calibrate_count)):
        order = rng.permutation(n)
        tr_idx = order[:n_tr]
        cal_idx = order[n_tr:]
        # Single-class splits would crash the trainer or short-circuit
        # the threshold rule; just skip this fold.
        if len({int(v) for v in y_train[tr_idx]}) < 2:
            continue
        if len({int(v) for v in y_train[cal_idx]}) < 2:
            continue
        try:
            predict = trainer_fn(X_train[tr_idx], y_train[tr_idx], seed + k)
        except ValueError:
            continue
        pooled_scores.extend(_as_scores(predict(X_train[cal_idx])).tolist())
        pooled_labels.extend(float(v) for v in y_train[cal_idx])

    if not pooled_scores:
        return 0.5
    return conformal_threshold(pooled_scores, pooled_labels, inclusion_value)
