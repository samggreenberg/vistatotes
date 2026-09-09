"""CLI entry point for the MLP-vs-SVM label-curve sweep.

Usage::

    # Quick smoke test on the smallest demo dataset
    python -m vtscore.eval.label_curve_main --datasets esc50_s \
        --label-counts 5 10 20 --seeds 0 1 2

    # Full sweep, writing a tidy CSV
    python -m vtscore.eval.label_curve_main \
        --datasets esc50_s flowers102_s \
        --trainers mlp svm_linear svm_rbf \
        --label-counts 5 10 20 50 100 200 \
        --seeds 0 1 2 3 4 \
        --output label_curve.csv

The CLI loads one demo dataset at a time (via ``load_demo_dataset``) so
embedding downloads only happen for the datasets you actually ask for.
"""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING, Any

from vtscore.eval.label_curve import SWEEP_TRAINERS, run_label_curve_eval, summarise

if TYPE_CHECKING:
    import pandas as pd


def _load_dataset(demo_id: str) -> dict[int, dict[str, Any]]:
    """Load one demo dataset into a fresh medias dict."""
    from vtscore.datasets.loader import load_demo_dataset

    medias: dict[int, dict[str, Any]] = {}
    load_demo_dataset(demo_id, medias)
    return medias


def _print_summary(summary: pd.DataFrame) -> None:
    """Print one row per (dataset, category, trainer, n_labels) cell.

    Leads with rank-based metrics (AUROC, AP) and the production-path F1
    (``f1_at_xcal``), since VTSearch never trusts the raw score as a
    probability - it always picks a threshold via cross-calibration.
    """
    if summary.empty:
        print("(no rows - every cell was skipped)")
        return
    for _, row in summary.iterrows():
        # row['n_labels'] is a scalar at runtime but pandas stubs widen it.
        n_labels = int(row["n_labels"])  # pyright: ignore[reportArgumentType]
        print(
            f"  {row['dataset']:>14s} | {row['category']:>20s} | "
            f"{row['trainer']:>10s} | N={n_labels:>3d} | "
            f"AUROC={row['auroc_mean']:.3f}±{row['auroc_std']:.3f}  "
            f"AP={row['average_precision_mean']:.3f}±{row['average_precision_std']:.3f}  "
            f"bestF1={row['best_f1_mean']:.3f}  "
            f"F1@xcal={row['f1_at_xcal_mean']:.3f}±{row['f1_at_xcal_std']:.3f}  "
            f"train={row['train_seconds_mean']:.3f}s"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m vtscore.eval.label_curve_main",
        description="MLP vs SVM label-curve sweep on demo datasets.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        metavar="DEMO_ID",
        help="Demo dataset IDs to load (e.g. esc50_s, flowers102_s).",
    )
    parser.add_argument(
        "--trainers",
        nargs="+",
        default=["mlp", "svm_linear"],
        choices=sorted(SWEEP_TRAINERS),
        help="Which trainers to compare (default: mlp svm_linear).",
    )
    parser.add_argument(
        "--label-counts",
        nargs="+",
        type=int,
        default=[5, 10, 20, 50, 100],
        metavar="N",
        help="Training-set sizes to sweep (default: 5 10 20 50 100).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
        metavar="SEED",
        help="Random seeds (default: 0 1 2 3 4).",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        metavar="CAT",
        help="Restrict evaluation to these target categories (applies to "
        "every dataset).  Default: every unique category per dataset.",
    )
    parser.add_argument(
        "--sim-fraction",
        type=float,
        default=0.5,
        help="Fraction of medias in the sim pool (default: 0.5).",
    )
    parser.add_argument(
        "--inclusion-value",
        type=int,
        default=0,
        metavar="V",
        help="Inclusion bias passed to conformal_threshold (default: 0).",
    )
    parser.add_argument(
        "--calibrate-count",
        type=int,
        default=2,
        metavar="K",
        help="Number of train/cal folds for cross-calibration (default: 2).",
    )
    parser.add_argument(
        "--cal-fraction",
        type=float,
        default=0.5,
        help="Fraction of training labels held out as the cal portion (default: 0.5).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        metavar="FILE",
        help="Write the tidy result table to FILE (.csv or .json by extension).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Suppress per-cell progress lines.",
    )
    args = parser.parse_args(argv)

    # Lazy-init models so the eval script doesn't pay the startup cost
    # when the user just runs --help.
    from vtscore.embedding import initialize_models

    initialize_models()

    dataset_clips: dict[str, dict[int, dict[str, Any]]] = {}
    for demo_id in args.datasets:
        print(f"Loading {demo_id} ...", flush=True)
        try:
            dataset_clips[demo_id] = _load_dataset(demo_id)
        except Exception as e:  # pragma: no cover - exercised via CLI
            print(f"  ERROR loading {demo_id}: {e}", file=sys.stderr)
            continue

    if not dataset_clips:
        print("No datasets loaded - aborting.", file=sys.stderr)
        return 2

    categories = None
    if args.categories:
        categories = {ds: args.categories for ds in dataset_clips}

    df = run_label_curve_eval(
        dataset_clips=dataset_clips,
        trainers=args.trainers,
        label_counts=args.label_counts,
        seeds=args.seeds,
        categories=categories,
        sim_fraction=args.sim_fraction,
        inclusion_value=args.inclusion_value,
        calibrate_count=args.calibrate_count,
        cal_fraction=args.cal_fraction,
        progress=not args.no_progress,
    )

    print(f"\nCollected {len(df)} rows.\n")
    print("=" * 78)
    print("SUMMARY (mean ± stddev across seeds)")
    print("=" * 78)
    _print_summary(summarise(df))

    if args.output:
        out = args.output
        if out.endswith(".json"):
            df.to_json(out, orient="records", indent=2)
        else:
            df.to_csv(out, index=False)
        print(f"\nResults written to {out}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
