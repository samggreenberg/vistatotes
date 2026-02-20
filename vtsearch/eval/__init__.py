"""Evaluation framework for VTSearch sorting quality."""

from vtsearch.eval.config import EVAL_DATASETS, EvalQuery
from vtsearch.eval.metrics import compute_metrics
from vtsearch.eval.runner import run_eval

__all__ = [
    "EVAL_DATASETS",
    "EvalQuery",
    "compute_metrics",
    "run_eval",
]
