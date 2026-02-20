"""Evaluation framework for VTSearch sorting quality."""

from vtsearch.eval.config import EVAL_DATASETS, EvalQuery
from vtsearch.eval.metrics import compute_metrics
from vtsearch.eval.runner import run_eval
from vtsearch.eval.voting_iterations import (
    run_voting_iterations_eval,
    run_voting_iterations_eval_from_pickles,
    simulate_voting_iterations,
)

__all__ = [
    "EVAL_DATASETS",
    "EvalQuery",
    "compute_metrics",
    "run_eval",
    "run_voting_iterations_eval",
    "run_voting_iterations_eval_from_pickles",
    "simulate_voting_iterations",
]
