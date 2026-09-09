"""Deprecated alias for :mod:`vtscore.eval.sweep_trainers`.

This module was renamed in issue #3764 so that the eval framework's two
trainer registries stop colliding by name:

* :mod:`vtscore.eval.sweep_trainers` (this module's new home) holds the
  **standalone estimators** the label-curve and timing sweeps compare, where
  ``"mlp"`` really is an MLP.
* :mod:`vtscore.eval.step_trainers` holds the **per-step pipelines** the
  voting simulation runs, whose ``"app"`` arm is the shipped detector.

Importing from here still works and is byte-identical, so out-of-tree code that
pinned the old path keeps running.  New code should import
:mod:`vtscore.eval.sweep_trainers` directly.
"""

from __future__ import annotations

from vtscore.eval.sweep_trainers import (  # noqa: F401  - re-exported for backward compatibility
    SWEEP_TRAINERS,
    SWEEP_TRAINERS as TRAINERS,
    PredictFn,
    TrainerFn,
    _as_scores,
    _cross_calibrated_threshold,
    _parse_trainer_spec,
    _train_svm_factory,
    resolve_trainer,
)

__all__ = [
    "SWEEP_TRAINERS",
    "TRAINERS",
    "PredictFn",
    "TrainerFn",
    "resolve_trainer",
]
