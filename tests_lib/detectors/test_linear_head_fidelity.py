"""The two linear heads are what they claim to be, structurally and numerically.

Both are ``build_model(input_dim, hidden_dim=<sentinel>)`` - a single
``Linear(d, 1)`` emitting a logit - and they differ only in the objective that
fits them:

* :data:`~vtscore.training.mlp.LINEAR_SVM_HEAD` is the **production** head.
  Fitted through :func:`vtscore.training.svm.fit_linear_svm_head` it *is*
  ``LinearSVC(class_weight="balanced")``, and these tests pin it to the very
  ``svm_linear`` arm the eval harness scores - score for score, not merely in
  rank - so the shipped detector and the measured arm can't drift apart.
* :data:`~vtscore.training.mlp.LINEAR_HEAD` is the logistic head the SVM
  replaced, still reachable as a named eval arm.  Pushed through
  :func:`vtscore.training.mlp.train_model`'s balanced BCE-with-logits loop it
  *is* ``LogisticRegression(class_weight="balanced")``; on seeded synthetic
  2-class data it must rank a held-out set in near-lockstep with scikit-learn's.
  If a future change to the training loop (an added nonlinearity, a different
  loss, a dropped class weighting) quietly stops it from being logistic
  regression, the rank agreement drops and that test fails.

**Structure / round-trip** is shared by both: one Linear layer, ``0.weight`` /
``0.bias`` as the only state-dict keys, and a clean trip through
``build_model_from_weights`` and the portable ONNX exporter (whose 1-layer
branch only a linear head exercises).

The *logistic* fidelity tests raise ``TRAIN_EPOCHS`` and disable early-stop: the
claim under test is about the *objective* the loop optimises, so the loop has to
be run to convergence.  Production's 200-epoch budget (and this suite's 30) stop
well short of the optimum - an early-stopped linear model, still linear, but not
the ``LogisticRegression`` fixed point.  The synthetic features are deliberately
raw Gaussians rather than the unit-norm vectors a real embedder emits, because
un-normalised features make the problem well-conditioned enough for Adam to
actually reach that fixed point in a unit test's time budget.  The SVM head
needs none of that: liblinear solves its objective outright, so its fidelity
test is an exact comparison at production settings.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from vtscore import config
from vtscore.detectors.portable_bundle import (
    ONNX_INPUT_NAME,
    ONNX_OUTPUT_NAME,
    embedding_dim_from_weights,
    mlp_weights_to_onnx,
)
from vtscore.detectors.training import serialize_weights
from vtscore.training.mlp import (
    LINEAR_HEAD,
    LINEAR_SVM_HEAD,
    build_model,
    build_model_from_weights,
    train_model,
)

DIM = 16


@pytest.fixture
def converged_training(monkeypatch):
    """Run ``train_model`` to convergence instead of the suite's 30-epoch budget."""
    monkeypatch.setattr(config, "TRAIN_EPOCHS", 2000, raising=False)
    monkeypatch.setattr(config, "TRAIN_PATIENCE", 0, raising=False)


def _two_class_data(n_per_class: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Two overlapping Gaussian blobs: learnable, but not trivially separable.

    The overlap matters - on perfectly separable data the logistic MLE runs off
    to infinity, the two fits are then decided by their (different) penalties
    rather than by the shared objective, and the comparison proves nothing.
    """
    rng = np.random.default_rng(seed)
    direction = rng.standard_normal(DIM).astype(np.float32)
    direction /= np.linalg.norm(direction)
    pos = rng.standard_normal((n_per_class, DIM)).astype(np.float32) + 1.2 * direction
    neg = rng.standard_normal((n_per_class, DIM)).astype(np.float32) - 1.2 * direction
    X = np.concatenate([pos, neg]).astype(np.float32)
    y = np.concatenate([np.ones(n_per_class), np.zeros(n_per_class)]).astype(np.float32)
    return X, y


def _head_scores(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_score: np.ndarray,
    hidden_dim: int = LINEAR_HEAD,
) -> np.ndarray:
    """Sigmoid scores of the linear head *hidden_dim* names, fitted the production way."""
    model = train_model(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train).unsqueeze(1),
        DIM,
        seed=0,
        hidden_dim=hidden_dim,
    )
    with torch.no_grad():
        device = next(model.parameters()).device
        return torch.sigmoid(model(torch.from_numpy(X_score).to(device))).squeeze(1).cpu().numpy()


def _sklearn_scores(X_train: np.ndarray, y_train: np.ndarray, X_score: np.ndarray) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415

    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=10000).fit(X_train, y_train)
    return clf.predict_proba(X_score)[:, 1]


def _rank_agreement(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation - Pearson over the two score vectors' ranks.

    Computed here rather than via ``scipy.stats.spearmanr`` to keep the return
    type concrete.  Both inputs are continuous sigmoid scores, so ties don't
    arise and plain ordinal ranks are exact.
    """
    ranks = [np.argsort(np.argsort(v)).astype(np.float64) for v in (a, b)]
    return float(np.corrcoef(ranks[0], ranks[1])[0, 1])


class TestLogisticFidelity:
    def test_ranks_agree_with_sklearn_logistic_regression(self, converged_training):
        """Spearman >= 0.95 against ``LogisticRegression(class_weight='balanced')``."""
        X, y = _two_class_data(n_per_class=60, seed=12345)
        X_score, _ = _two_class_data(n_per_class=100, seed=999)

        rho = _rank_agreement(_head_scores(X, y, X_score), _sklearn_scores(X, y, X_score))
        assert rho >= 0.95, f"linear head ranks disagree with logistic regression (Spearman {rho:.4f})"

    def test_ranks_agree_when_positives_are_sparse(self, converged_training):
        """The loop's inverse-frequency weights match ``class_weight='balanced'``.

        12 positives against 60 negatives - the sparse-positive regime the linear
        head was adopted for (#2790).  An unweighted loss would tilt away from
        sklearn's balanced fit here even though it matches on balanced data.
        """
        X, y = _two_class_data(n_per_class=60, seed=12345)
        keep = np.concatenate([np.arange(12), np.arange(60, 120)])
        X, y = X[keep], y[keep]
        X_score, _ = _two_class_data(n_per_class=100, seed=999)

        rho = _rank_agreement(_head_scores(X, y, X_score), _sklearn_scores(X, y, X_score))
        assert rho >= 0.95, f"linear head ranks disagree under class imbalance (Spearman {rho:.4f})"


def _weight_of(model) -> torch.Tensor:
    """The single ``Linear`` layer's weight tensor of a linear head."""
    layer = model[0]
    assert isinstance(layer, torch.nn.Linear)
    return layer.weight


class TestSVMFidelity:
    """The production head *is* the eval harness's ``svm_linear`` arm.

    Not "ranks like": the same numbers.  ``fit_linear_svm_head`` delegates to
    the very ``train_svm(kernel="linear")`` call the harness scores, and
    production's ``sigmoid(model(x))`` reproduces that fit's
    ``decision_sigmoid`` ``predict_proba``.  If the two ever diverge, an
    experiment that says "svm_linear wins" stops being a statement about the
    shipped detector - which is the whole reason the head was switched.
    """

    def _svm_linear_arm_scores(self, X_train, y_train, X_score) -> np.ndarray:
        """Exactly what ``TRAINERS["svm_linear"]`` computes for these labels."""
        from vtscore.eval.sweep_trainers import resolve_trainer  # noqa: PLC0415

        predict = resolve_trainer("svm_linear")(X_train, y_train, 0)
        return np.asarray(predict(X_score), dtype=np.float64)

    def test_scores_match_the_svm_linear_eval_arm(self):
        X, y = _two_class_data(n_per_class=60, seed=12345)
        X_score, _ = _two_class_data(n_per_class=100, seed=999)

        head = _head_scores(X, y, X_score, hidden_dim=LINEAR_SVM_HEAD)
        arm = self._svm_linear_arm_scores(X, y, X_score)
        assert np.allclose(head, arm, atol=1e-5), f"max deviation {np.abs(head - arm).max():.3e}"

    def test_scores_match_when_positives_are_sparse(self):
        """12 positives against 60 negatives - the sparse-positive regime."""
        X, y = _two_class_data(n_per_class=60, seed=12345)
        keep = np.concatenate([np.arange(12), np.arange(60, 120)])
        X, y = X[keep], y[keep]
        X_score, _ = _two_class_data(n_per_class=100, seed=999)

        head = _head_scores(X, y, X_score, hidden_dim=LINEAR_SVM_HEAD)
        arm = self._svm_linear_arm_scores(X, y, X_score)
        assert np.allclose(head, arm, atol=1e-5), f"max deviation {np.abs(head - arm).max():.3e}"

    def test_the_head_is_a_maximum_margin_fit_not_a_logistic_one(self):
        """The two heads must actually be different fits, or the switch is a no-op."""
        X, y = _two_class_data(n_per_class=60, seed=12345)
        X_score, _ = _two_class_data(n_per_class=100, seed=999)

        svm = _head_scores(X, y, X_score, hidden_dim=LINEAR_SVM_HEAD)
        logistic = _head_scores(X, y, X_score, hidden_dim=LINEAR_HEAD)
        assert not np.allclose(svm, logistic, atol=1e-3)

    def test_per_row_weights_replace_the_class_balance(self):
        """Region flooding's per-bag weights must reach liblinear.

        Weighting every negative down must move the boundary; if the weights
        were dropped the two fits would be identical and a Bad image's ~197
        region rows would each count as an independent negative.
        """
        X, y = _two_class_data(n_per_class=40, seed=7)
        X_t = torch.from_numpy(X)
        y_t = torch.from_numpy(y).unsqueeze(1)
        weights = torch.where(y_t.reshape(-1) == 1.0, 1.0, 0.05)

        plain = train_model(X_t, y_t, DIM, seed=0, hidden_dim=LINEAR_SVM_HEAD)
        weighted = train_model(X_t, y_t, DIM, seed=0, hidden_dim=LINEAR_SVM_HEAD, sample_weights=weights)
        assert not torch.allclose(_weight_of(plain), _weight_of(weighted), atol=1e-4)

    def test_mismatched_weight_length_is_rejected(self):
        X, y = _two_class_data(n_per_class=8, seed=1)
        with pytest.raises(ValueError, match="does not match training-set size"):
            train_model(
                torch.from_numpy(X),
                torch.from_numpy(y).unsqueeze(1),
                DIM,
                hidden_dim=LINEAR_SVM_HEAD,
                sample_weights=torch.ones(3),
            )


class TestProductionPathTrainsTheSVMHead:
    """``train_and_threshold`` (the Find path) hands back a one-layer SVM head.

    Guards the shipped head end-to-end: a revert to ``_auto_hidden_dim`` here
    reinstates the MLP, and a revert to ``LINEAR_HEAD`` reinstates the logistic
    fit - both fail here rather than sliding back in silently.
    The vote/labelset path (``_train_and_score_xy``) and the load-time
    re-derivation (``train_detector_from_origins``) are pinned the same way from
    the app tier, where their snapshot fixtures live.
    """

    N_MEDIA = 12
    # Ids well clear of the active context's own: the embedding-matrix cache
    # keys on the sorted id list, so a colliding snap gets the wrong matrix.
    ID_BASE = 7000

    def _snap_and_labels(self, seed: int = 5):
        rng = np.random.default_rng(seed)

        def _unit(vec: np.ndarray) -> np.ndarray:
            return (vec / (np.linalg.norm(vec) + 1e-8)).astype(np.float32)

        good_proto = _unit(rng.standard_normal(DIM))
        bad_proto = _unit(rng.standard_normal(DIM))
        snap = {
            cid: {
                "id": cid,
                "media_type": "audio",
                "embedder": "test",
                "embeddings": {"test": _unit(rng.standard_normal(DIM))},
            }
            for cid in range(self.ID_BASE + 1, self.ID_BASE + self.N_MEDIA + 1)
        }
        X_list, y_list = [], []
        for _ in range(4):
            X_list.append(_unit(good_proto + 0.1 * rng.standard_normal(DIM)))
            y_list.append(1.0)
            X_list.append(_unit(bad_proto + 0.1 * rng.standard_normal(DIM)))
            y_list.append(0.0)
        return snap, X_list, y_list

    def test_train_and_threshold_returns_a_linear_head(self):
        from vtscore.detectors.training import train_and_threshold  # noqa: PLC0415

        snap, X_list, y_list = self._snap_and_labels()
        model, _threshold = train_and_threshold(X_list, y_list, snap=snap)

        assert [type(layer) for layer in model] == [torch.nn.Linear]
        assert set(serialize_weights(model)) == {"0.weight", "0.bias"}

    def _region_labels(self, seed: int = 5, rows_per_bad_bag: int = 3):
        """Flooded region label set: 4 Good rows (one bag each), 4 Bad bags of N rows.

        Mirrors what region flooding hands ``train_and_threshold``: a Good vote
        contributes its one snapped-box row, a Bad vote floods its leaf set.
        """
        rng = np.random.default_rng(seed)

        def _unit(vec: np.ndarray) -> np.ndarray:
            return (vec / (np.linalg.norm(vec) + 1e-8)).astype(np.float32)

        good_proto = _unit(rng.standard_normal(DIM))
        bad_proto = _unit(rng.standard_normal(DIM))
        X_list: list[np.ndarray] = []
        y_list: list[float] = []
        groups: list = []
        for i in range(4):
            X_list.append(_unit(good_proto + 0.1 * rng.standard_normal(DIM)))
            y_list.append(1.0)
            groups.append(("g", i))
        for i in range(4):
            for _ in range(rows_per_bad_bag):
                X_list.append(_unit(bad_proto + 0.1 * rng.standard_normal(DIM)))
                y_list.append(0.0)
                groups.append(("b", i))
        return X_list, y_list, groups

    def _spy_hidden_dims(self, monkeypatch) -> list:
        """Record the ``hidden_dim`` of every ``train_model`` call (folds + final).

        Patches both the defining module and the package re-export: the
        calibration folds resolve ``train_model`` from ``vtscore.training.mlp``
        at call time, while ``train_and_threshold``'s final fit resolves it from
        the ``vtscore.training`` package namespace.
        """
        import vtscore.training as training_pkg  # noqa: PLC0415
        import vtscore.training.mlp as mlp_module  # noqa: PLC0415

        seen: list = []
        real = mlp_module.train_model

        def spy(*args, **kwargs):
            seen.append(kwargs.get("hidden_dim", "MISSING"))
            return real(*args, **kwargs)

        monkeypatch.setattr(mlp_module, "train_model", spy)
        monkeypatch.setattr(training_pkg, "train_model", spy)
        return seen

    def test_uncached_calibration_folds_use_the_svm_head(self, monkeypatch):
        """The ``det_ctx is None`` branch calibrates on the production head too (#2824).

        Before the fix this branch omitted ``hidden_dim`` from
        ``calculate_cross_calibration_threshold``, so the fold models auto-sized
        to the MLP while the final model was linear - the threshold was measured
        on a head the detector never ships.
        """
        from vtscore.detectors.training import train_and_threshold  # noqa: PLC0415

        seen = self._spy_hidden_dims(monkeypatch)
        _snap, X_list, y_list = self._snap_and_labels()
        train_and_threshold(X_list, y_list)

        # At least the calibration folds plus the final fit.
        assert len(seen) >= 2
        assert all(h == LINEAR_SVM_HEAD for h in seen), f"non-SVM train_model calls: {seen}"

    def test_region_bag_uncached_path_uses_the_svm_head(self, monkeypatch):
        """A flooded region label set calibrates + trains on the SVM head end-to-end (#2824)."""
        from vtscore.detectors.training import train_and_threshold  # noqa: PLC0415

        seen = self._spy_hidden_dims(monkeypatch)
        X_list, y_list, groups = self._region_labels()
        model, _threshold = train_and_threshold(X_list, y_list, groups=groups)

        assert len(seen) >= 2
        assert all(h == LINEAR_SVM_HEAD for h in seen), f"non-SVM train_model calls: {seen}"
        assert [type(layer) for layer in model] == [torch.nn.Linear]

    def test_region_bag_cached_path_uses_the_svm_head(self, monkeypatch):
        """The det_ctx (cached grouped-calibration) path uses the SVM head too."""
        from types import SimpleNamespace  # noqa: PLC0415

        from vtscore.detectors.training import train_and_threshold  # noqa: PLC0415

        seen = self._spy_hidden_dims(monkeypatch)
        X_list, y_list, groups = self._region_labels()
        det_ctx = SimpleNamespace(calibration_cache=None)
        model, _threshold = train_and_threshold(X_list, y_list, det_ctx=det_ctx, groups=groups)

        assert len(seen) >= 2
        assert all(h == LINEAR_SVM_HEAD for h in seen), f"non-SVM train_model calls: {seen}"
        assert [type(layer) for layer in model] == [torch.nn.Linear]
        # The fold orderings were cached for a later no-retrain Inclusion slide.
        assert det_ctx.calibration_cache is not None


class TestLinearHeadStructure:
    def test_single_linear_layer(self):
        model = build_model(DIM, hidden_dim=LINEAR_HEAD)
        layers = list(model)
        assert len(layers) == 1
        assert isinstance(layers[0], torch.nn.Linear)
        assert layers[0].in_features == DIM
        assert layers[0].out_features == 1

    def test_dropout_argument_is_ignored(self):
        """A bare linear map has nothing to regularise - no Dropout is inserted."""
        model = build_model(DIM, hidden_dim=LINEAR_HEAD, dropout=0.5)
        assert not any(isinstance(layer, torch.nn.Dropout) for layer in model)

    def test_state_dict_keys(self):
        weights = serialize_weights(build_model(DIM, hidden_dim=LINEAR_HEAD))
        assert set(weights) == {"0.weight", "0.bias"}
        assert np.asarray(weights["0.weight"]).shape == (1, DIM)
        assert embedding_dim_from_weights(weights) == DIM

    def test_round_trips_through_build_model_from_weights(self):
        gen = torch.Generator().manual_seed(3)
        model = build_model(DIM, hidden_dim=LINEAR_HEAD, generator=gen).eval()
        rebuilt = build_model_from_weights(serialize_weights(model))

        assert len(list(rebuilt)) == 1
        rng = np.random.default_rng(4)
        x = torch.from_numpy(rng.standard_normal((5, DIM)).astype(np.float32))
        with torch.no_grad():
            np.testing.assert_allclose(rebuilt(x).numpy(), model(x).numpy(), atol=1e-6)

    def test_onnx_export_is_sigmoid_of_a_single_gemm(self):
        import onnx  # noqa: PLC0415

        gen = torch.Generator().manual_seed(5)
        weights = serialize_weights(build_model(DIM, hidden_dim=LINEAR_HEAD, generator=gen))
        exported = onnx.load_from_string(mlp_weights_to_onnx(weights))

        onnx.checker.check_model(exported)
        # No Relu: the linear head has no hidden activation to model.
        assert [node.op_type for node in exported.graph.node] == ["Gemm", "Sigmoid"]

    def test_onnx_scores_match_torch(self):
        ort = pytest.importorskip("onnxruntime")

        gen = torch.Generator().manual_seed(6)
        model = build_model(DIM, hidden_dim=LINEAR_HEAD, generator=gen).eval()
        weights = serialize_weights(model)

        rng = np.random.default_rng(7)
        x = rng.standard_normal((6, DIM)).astype(np.float32)
        with torch.no_grad():
            expected = torch.sigmoid(model(torch.from_numpy(x))).numpy().ravel()

        session = ort.InferenceSession(mlp_weights_to_onnx(weights))
        got = session.run([ONNX_OUTPUT_NAME], {ONNX_INPUT_NAME: x})[0].ravel()
        np.testing.assert_allclose(got, expected, atol=1e-5)
