"""GPU runtime microbenchmark: MLP (torch) vs SVM (cuML) train/inference scaling.

Stage C of the MLP-vs-SVM study.
The per-step timings inside the voting simulation only exercise the tiny
vote-regime fits; this benchmark instead measures how each trainer *scales*:

* **Training time vs training-set size** — how fit time grows as the label
  budget grows (``n_train`` from 8 up to 16 384).  The MLP's fixed-epoch
  full-batch training is near-flat; a kernel SVM's fit is super-linear.
* **Inference time vs inference-set size** — how scoring time grows with the
  number of items scored (``n_infer`` from 1e3 to 1e6), for a model trained at
  the realistic vote budget (``n_train=100``).  The MLP is a fixed two-layer
  matmul; a kernel SVM's inference grows with its support-vector count.

Methodology (matching the plan): ``torch.cuda.synchronize()`` (plus cuML's
device→host copy) bracket every timed region; 2 warmup runs are discarded; the
median of 7 repeats is reported with its inter-quartile range.  Backends and
device/library versions are recorded on every row.  The MLP runs on torch CUDA
(the existing AMP path); the SVM runs on cuML when available, else sklearn-CPU
(rows are labelled so the report never silently compares CPU to GPU).

Runs on CPU too (for smoke tests) — the rows are simply labelled ``torch-cpu`` /
``sklearn-cpu``.  A GPU-marked test exercises the CUDA path and the sklearn↔cuML
score-parity cross-check.
"""

from __future__ import annotations

import argparse
import time
from typing import TYPE_CHECKING, Any, Optional, Sequence

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

_DEFAULT_TRAIN_SIZES = (8, 16, 32, 64, 128, 256, 512, 1024, 4096, 16384)
_DEFAULT_INFER_SIZES = (1000, 10000, 100000, 1000000)
_DEFAULT_TRAINERS = ("mlp", "svm_linear", "svm_rbf")
_DEFAULT_DIM = 768  # SigLIP base


# ---------------------------------------------------------------------------
# Timing primitives
# ---------------------------------------------------------------------------


def _cuda_sync() -> None:
    """Block until all queued CUDA work finishes, if a GPU is in play."""
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:  # noqa: BLE001 - timing barrier is best-effort
        pass


def _median_iqr(samples: list[float]) -> tuple[float, float]:
    """Return ``(median, IQR)`` of *samples* (IQR = 75th - 25th percentile)."""
    import numpy as np  # noqa: PLC0415

    arr = np.asarray(samples, dtype=np.float64)
    return float(np.median(arr)), float(np.percentile(arr, 75) - np.percentile(arr, 25))


def _timed(fn, *, repeats: int, warmup: int) -> tuple[float, float]:
    """Run *fn* ``warmup + repeats`` times; return ``(median, IQR)`` of the timed runs.

    ``fn`` must itself trigger any device→host sync it needs (the timing wraps a
    trailing :func:`_cuda_sync` so queued CUDA work is included).
    """
    for _ in range(warmup):
        fn()
        _cuda_sync()
    times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        _cuda_sync()
        times.append(time.perf_counter() - t0)
    return _median_iqr(times)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def _make_data(
    n: int,
    dim: int,
    rng: np.random.Generator,
    x_source: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(X, y)``: *n* unit-norm rows of dim *dim* with balanced 0/1 labels.

    When *x_source* is given (a pool of real embeddings), rows are sampled from
    it (with replacement if *n* exceeds the pool); otherwise Gaussian vectors are
    generated.  Labels come from a random hyperplane so the two classes are
    linearly separable enough for every trainer to fit without a degenerate
    single-class batch, which is all the timing needs.
    """
    import numpy as np  # noqa: PLC0415

    if x_source is not None and len(x_source) > 0:
        idx = rng.integers(0, len(x_source), size=n)
        X = np.asarray(x_source, dtype=np.float32)[idx]
    else:
        X = rng.standard_normal((n, dim)).astype(np.float32)
        X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    w = rng.standard_normal(X.shape[1]).astype(np.float32)
    proj = X @ w
    y = (proj >= np.median(proj)).astype(np.int32)  # balanced by construction
    return X, y


# ---------------------------------------------------------------------------
# Trainer train/predict closures
# ---------------------------------------------------------------------------


def _mlp_fit(X: np.ndarray, y: np.ndarray):
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    from vtscore.training.mlp import train_model

    X_t = torch.from_numpy(np.asarray(X, dtype=np.float32))
    y_t = torch.from_numpy(np.asarray(y, dtype=np.float32)).unsqueeze(1)
    return train_model(X_t, y_t, input_dim=X.shape[1])


def _mlp_predict_closure(model, X: np.ndarray):
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    device = next(model.parameters()).device
    X_t = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(device)

    def predict() -> None:
        with torch.no_grad():
            _ = torch.sigmoid(model(X_t)).squeeze(1).cpu().numpy()

    return predict


def _svm_fit(trainer: str, X: np.ndarray, y: np.ndarray, backend: str):
    from vtscore.eval.sweep_trainers import _parse_trainer_spec  # noqa: PLC0415
    from vtscore.training.svm import train_svm  # noqa: PLC0415

    kernel, kwargs = _parse_trainer_spec(trainer)
    return train_svm(X, y, kernel=kernel, seed=42, backend=backend, **kwargs)  # type: ignore[arg-type]


def _mlp_fit_closure(X: np.ndarray, y: np.ndarray, sink: dict):
    def fit() -> None:
        sink["model"] = _mlp_fit(X, y)

    return fit


def _svm_fit_closure(trainer: str, X: np.ndarray, y: np.ndarray, backend: str, sink: dict):
    def fit() -> None:
        sink["clf"] = _svm_fit(trainer, X, y, backend)

    return fit


def _svm_predict_closure(clf, X: np.ndarray):
    def predict() -> None:
        _ = clf.predict_proba(X)

    return predict


def _backend_label(trainer: str, clf_or_device: Any) -> str:
    """Human-readable backend for a fitted model."""
    if trainer == "mlp":
        return "torch-cuda" if str(clf_or_device).startswith("cuda") else "torch-cpu"
    return getattr(clf_or_device, "backend", "sklearn-cpu")


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def _provenance() -> dict[str, str]:
    prov: dict[str, str] = {}
    try:
        import torch  # noqa: PLC0415

        prov["torch"] = torch.__version__
        prov["torch_cuda"] = str(torch.version.cuda)  # type: ignore[attr-defined]
        if torch.cuda.is_available():
            prov["gpu"] = torch.cuda.get_device_name(0)
            prov["driver"] = str(getattr(torch.version, "cuda", "?"))  # type: ignore[attr-defined]
        else:
            prov["gpu"] = "cpu"
    except Exception:  # noqa: BLE001
        prov["torch"] = "?"
    try:
        import cuml  # noqa: PLC0415  # pyright: ignore[reportMissingImports]

        prov["cuml"] = cuml.__version__
    except Exception:  # noqa: BLE001
        prov["cuml"] = "absent"
    try:
        import sklearn  # noqa: PLC0415

        prov["sklearn"] = sklearn.__version__
    except Exception:  # noqa: BLE001
        prov["sklearn"] = "?"
    return prov


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


def run_timing_benchmark(
    trainers: Sequence[str] = _DEFAULT_TRAINERS,
    train_sizes: Sequence[int] = _DEFAULT_TRAIN_SIZES,
    infer_sizes: Sequence[int] = _DEFAULT_INFER_SIZES,
    *,
    dim: int = _DEFAULT_DIM,
    infer_n_train: int = 100,
    repeats: int = 7,
    warmup: int = 2,
    svm_backend: str = "auto",
    x_source: Optional[np.ndarray] = None,
    seed: int = 0,
    progress: bool = False,
) -> pd.DataFrame:
    """Run the training- and inference-scaling benchmarks; return a tidy frame.

    Columns: ``phase`` (``"train"`` | ``"infer"``), ``trainer``, ``backend``,
    ``n`` (n_train for train phase, n_infer for infer phase), ``dim``,
    ``median_seconds``, ``iqr_seconds``, ``repeats``, plus one column per
    provenance key (``torch``, ``cuml``, ``gpu``, …) so every row is
    self-describing.
    """
    import numpy as np  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415

    prov = _provenance()
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []

    def emit(phase: str, trainer: str, backend: str, n: int, med: float, iqr: float) -> None:
        rows.append(
            {
                "phase": phase,
                "trainer": trainer,
                "backend": backend,
                "n": int(n),
                "dim": int(dim),
                "median_seconds": round(med, 8),
                "iqr_seconds": round(iqr, 8),
                "repeats": int(repeats),
                **prov,
            }
        )

    # --- Training time vs n_train ---
    for trainer in trainers:
        for n_train in train_sizes:
            X, y = _make_data(int(n_train), dim, rng, x_source)
            if trainer == "mlp":
                sink: dict[str, Any] = {}
                med, iqr = _timed(_mlp_fit_closure(X, y, sink), repeats=repeats, warmup=warmup)
                backend = _backend_label("mlp", next(sink["model"].parameters()).device)
            else:
                sink = {}
                med, iqr = _timed(_svm_fit_closure(trainer, X, y, svm_backend, sink), repeats=repeats, warmup=warmup)
                backend = _backend_label(trainer, sink["clf"])
            emit("train", trainer, backend, n_train, med, iqr)
            if progress:
                print(f"[train] {trainer:12} n={n_train:>6} {med * 1e3:8.2f} ms  ({backend})", flush=True)

    # --- Inference time vs n_infer (model trained at infer_n_train) ---
    for trainer in trainers:
        Xtr, ytr = _make_data(int(infer_n_train), dim, rng, x_source)
        if trainer == "mlp":
            model = _mlp_fit(Xtr, ytr)
            backend = _backend_label("mlp", next(model.parameters()).device)
        else:
            clf = _svm_fit(trainer, Xtr, ytr, svm_backend)
            backend = _backend_label(trainer, clf)
        for n_infer in infer_sizes:
            Xinf, _ = _make_data(int(n_infer), dim, rng, x_source)
            predict = _mlp_predict_closure(model, Xinf) if trainer == "mlp" else _svm_predict_closure(clf, Xinf)
            med, iqr = _timed(predict, repeats=repeats, warmup=warmup)
            emit("infer", trainer, backend, n_infer, med, iqr)
            if progress:
                print(f"[infer] {trainer:12} n={n_infer:>8} {med * 1e3:8.2f} ms  ({backend})", flush=True)

    return pd.DataFrame(rows)


def svm_backend_parity(
    trainer: str = "svm_rbf",
    n_train: int = 256,
    dim: int = _DEFAULT_DIM,
    *,
    seed: int = 0,
) -> float:
    """Return the Spearman rank correlation between cuML and sklearn SVM scores.

    Fits the same SVM on the GPU (cuML) and CPU (sklearn) on identical data and
    correlates their scores over a fresh sample.  The timing/quality runs trust
    the two backends to describe the same model only if this is ~1.0; the plan
    asserts > 0.99 at ``n_train <= 256``.  Returns ``nan`` if cuML is
    unavailable (no GPU to compare against).
    """
    import numpy as np  # noqa: PLC0415

    from vtscore.gpu_backends import cuml_enabled  # noqa: PLC0415

    if not cuml_enabled():
        return float("nan")

    rng = np.random.default_rng(seed)
    X, y = _make_data(n_train, dim, rng)
    Xtest, _ = _make_data(512, dim, rng)

    # cuML SVM can be present-but-broken on a mismatched CUDA toolchain (its
    # kernels compile lazily via nvrtc); treat any failure — or a silent
    # fallback to sklearn — as "no GPU SVM to compare against".
    try:
        cuml_clf = _svm_fit(trainer, X, y, "cuml")
    except Exception:  # noqa: BLE001
        return float("nan")
    if getattr(cuml_clf, "backend", "") != "cuml":
        return float("nan")
    sklearn_clf = _svm_fit(trainer, X, y, "sklearn")
    s_gpu = np.asarray(cuml_clf.predict_proba(Xtest), dtype=np.float64)
    s_cpu = np.asarray(sklearn_clf.predict_proba(Xtest), dtype=np.float64)

    # Spearman = Pearson on ranks.
    r_gpu = np.argsort(np.argsort(s_gpu))
    r_cpu = np.argsort(np.argsort(s_cpu))
    return float(np.corrcoef(r_gpu, r_cpu)[0, 1])


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m vtscore.eval.timing_benchmark",
        description="GPU train/inference scaling benchmark: MLP (torch) vs SVM (cuML).",
    )
    parser.add_argument("--trainers", nargs="+", default=list(_DEFAULT_TRAINERS))
    parser.add_argument("--train-sizes", nargs="+", type=int, default=list(_DEFAULT_TRAIN_SIZES))
    parser.add_argument("--infer-sizes", nargs="+", type=int, default=list(_DEFAULT_INFER_SIZES))
    parser.add_argument("--dim", type=int, default=_DEFAULT_DIM)
    parser.add_argument("--infer-n-train", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--svm-backend", default="auto", choices=["auto", "sklearn", "cuml"])
    parser.add_argument(
        "--x-source",
        default=None,
        help="Optional .npy of real embeddings (N, dim) to sample from instead of Gaussian vectors.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None, help="Write the tidy CSV here.")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args(argv)

    import numpy as np  # noqa: PLC0415

    from vtscore.embedding import initialize_models  # noqa: PLC0415

    initialize_models()

    x_source = None
    if args.x_source:
        x_source = np.load(args.x_source).astype(np.float32)

    df = run_timing_benchmark(
        trainers=args.trainers,
        train_sizes=args.train_sizes,
        infer_sizes=args.infer_sizes,
        dim=args.dim,
        infer_n_train=args.infer_n_train,
        repeats=args.repeats,
        warmup=args.warmup,
        svm_backend=args.svm_backend,
        x_source=x_source,
        seed=args.seed,
        progress=not args.no_progress,
    )
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nWrote {len(df)} rows to {args.output}")
    else:
        print(df.to_string(index=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
