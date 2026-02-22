"""CLI entry point for the VTSearch evaluation framework.

Usage::

    # Run all evaluations (text-sort + learned-sort) on all datasets
    python -m vtsearch.eval

    # Only text-sort evaluation
    python -m vtsearch.eval --mode text

    # Only learned-sort evaluation
    python -m vtsearch.eval --mode learned

    # Evaluate specific datasets
    python -m vtsearch.eval --datasets images_s images_m

    # Custom train/test split for learned-sort
    python -m vtsearch.eval --mode learned --train-fraction 0.6

    # Save results as JSON
    python -m vtsearch.eval --output results.json

    # List available eval datasets
    python -m vtsearch.eval --list
"""

import argparse
import sys

# Ensure models are initialised (creates cache dirs, sets thread counts)
from vtsearch.models import initialize_models

initialize_models()

from vtsearch.eval.config import EVAL_DATASETS
from vtsearch.eval.runner import format_results_json, run_eval


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m vtsearch.eval",
        description="Evaluate VTSearch text-sort and learned-sort quality on demo datasets.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        metavar="ID",
        help="Eval dataset IDs to run (default: all).  Use --list to see available IDs.",
    )
    parser.add_argument(
        "--mode",
        choices=["text", "learned", "both"],
        default="both",
        help="Which evaluation to run (default: both).",
    )
    parser.add_argument(
        "--k",
        nargs="+",
        type=int,
        default=None,
        metavar="K",
        help="k values for P@k / R@k (default: 5 10 20).",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.5,
        help="Fraction of clips used for training in learned-sort eval (default: 0.5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits (default: 42).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        metavar="FILE",
        help="Write JSON results to FILE.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available eval datasets and exit.",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Generate visualisation PNGs in DIR (default: no plots).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable plot generation even when --plot-dir is set.",
    )
    parser.add_argument(
        "--enrich-descriptions",
        action="store_true",
        help="Use enriched (wrapper-averaged) text embeddings for text-sort evaluation.",
    )

    args = parser.parse_args()

    if args.list:
        print("Available eval datasets:\n")
        for ds_id, ds_cfg in sorted(EVAL_DATASETS.items()):
            queries = ds_cfg["queries"]
            cats = ", ".join(q.target_category for q in queries)
            print(f"  {ds_id}")
            print(f"    demo dataset: {ds_cfg['demo_dataset']}")
            print(f"    categories:   {cats}")
            print(f"    queries:      {len(queries)}")
            print()
        sys.exit(0)

    results = run_eval(
        dataset_ids=args.datasets,
        mode=args.mode,
        k_values=args.k,
        train_fraction=args.train_fraction,
        seed=args.seed,
        enrich=args.enrich_descriptions,
    )

    # Print summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for r in results:
        line = f"  {r.dataset_id:25s}"
        if r.text_sort:
            line += f"  mAP={r.mean_average_precision:.4f}"
        if r.learned_sort:
            line += f"  mean_F1={r.mean_learned_f1:.4f}"
        print(line)

    # Optionally write JSON
    if args.output:
        json_str = format_results_json(results)
        with open(args.output, "w") as f:
            f.write(json_str)
        print(f"\nResults written to {args.output}")

    # Generate visualisations
    if args.plot_dir and not args.no_plot:
        from vtsearch.eval.visualize import plot_eval_results

        generated = plot_eval_results(results, output_dir=args.plot_dir)
        if generated:
            print(f"\nPlots written to {args.plot_dir}/:")
            for p in generated:
                print(f"  {p.name}")
        else:
            print("\nNo plots generated (no results to visualise).")


if __name__ == "__main__":
    main()
