"""Command-line interface utilities for VistaTotes."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from vistatotes.datasets.loader import load_dataset_from_pickle


def _score_clips_with_detector(
    clips: dict[int, dict[str, Any]],
    detector_path: str,
) -> list[dict[str, Any]]:
    """Score all *clips* using the detector at *detector_path*.

    Returns a list of dicts for clips predicted as "Good", sorted descending
    by score.  Each dict contains ``id``, ``filename``, ``category``, and
    ``score``.

    Raises:
        FileNotFoundError: If the detector file does not exist.
        ValueError: If the clips dict is empty or the detector is invalid.
    """
    detector_file = Path(detector_path)

    if not detector_file.exists():
        raise FileNotFoundError(f"Detector file not found: {detector_path}")

    if not clips:
        raise ValueError("No clips loaded from dataset")

    # Load detector
    with open(detector_file, "r") as f:
        detector_data = json.load(f)

    if "weights" not in detector_data:
        raise ValueError("Detector file missing 'weights' field")
    if "threshold" not in detector_data:
        raise ValueError("Detector file missing 'threshold' field")

    weights = detector_data["weights"]
    threshold = detector_data["threshold"]

    # Reconstruct the MLP model from weights
    input_dim = len(weights["0.weight"][0])

    model = nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
        nn.Sigmoid(),
    )

    state_dict = {}
    for key, value in weights.items():
        state_dict[key] = torch.tensor(value, dtype=torch.float32)
    model.load_state_dict(state_dict)
    model.eval()

    # Score all clips
    all_ids = sorted(clips.keys())
    all_embs = np.array([clips[cid]["embedding"] for cid in all_ids])
    X_all = torch.tensor(all_embs, dtype=torch.float32)

    with torch.no_grad():
        scores = model(X_all).squeeze(1).tolist()

    # Collect positive hits (score >= threshold)
    positive_hits = []
    for cid, score in zip(all_ids, scores):
        if score >= threshold:
            clip = clips[cid]
            positive_hits.append(
                {
                    "id": cid,
                    "filename": clip.get("filename", f"clip_{cid}"),
                    "category": clip.get("category", "unknown"),
                    "score": round(score, 4),
                }
            )

    # Sort by score descending
    positive_hits.sort(key=lambda x: x["score"], reverse=True)

    return positive_hits


def run_autodetect(dataset_path: str, detector_path: str) -> list[dict[str, Any]]:
    """Load a dataset and detector, run the detector, and return positive hits.

    Args:
        dataset_path: Path to a pickle file containing the dataset.
        detector_path: Path to a JSON file containing detector weights and threshold.

    Returns:
        A list of dicts for clips predicted as "Good", each containing
        the clip's ``id``, ``filename``, ``category``, and ``score``.

    Raises:
        FileNotFoundError: If the dataset or detector file does not exist.
        ValueError: If the dataset is empty or the detector file is invalid.
    """
    dataset_file = Path(dataset_path)

    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    # Load dataset
    clips: dict[int, dict[str, Any]] = {}
    load_dataset_from_pickle(dataset_file, clips)

    if not clips:
        raise ValueError(f"No clips loaded from dataset: {dataset_path}")

    return _score_clips_with_detector(clips, detector_path)


def run_autodetect_with_importer(
    importer_name: str,
    field_values: dict[str, Any],
    detector_path: str,
) -> list[dict[str, Any]]:
    """Load a dataset via a named importer, then run a detector on it.

    This is the importer-aware counterpart of :func:`run_autodetect`.

    Args:
        importer_name: Registered name of the importer (e.g. ``"folder"``).
        field_values: Mapping of importer field keys to their CLI values.
        detector_path: Path to a JSON file containing detector weights and threshold.

    Returns:
        A list of dicts for clips predicted as "Good", each containing
        the clip's ``id``, ``filename``, ``category``, and ``score``.

    Raises:
        ValueError: If the importer is unknown, required fields are missing,
            the loaded dataset is empty, or the detector file is invalid.
        FileNotFoundError: If the detector file does not exist.
    """
    from vistatotes.datasets.importers import get_importer

    importer = get_importer(importer_name)
    if importer is None:
        available = _list_importer_names()
        raise ValueError(f"Unknown importer: {importer_name}. Available: {', '.join(available)}")

    importer.validate_cli_field_values(field_values)

    clips: dict[int, dict[str, Any]] = {}
    importer.run_cli(field_values, clips)

    if not clips:
        raise ValueError(f"No clips loaded by importer '{importer_name}'")

    return _score_clips_with_detector(clips, detector_path)


def _list_importer_names() -> list[str]:
    """Return the names of all registered importers."""
    from vistatotes.datasets.importers import list_importers

    return [imp.name for imp in list_importers()]


def _print_hits(hits: list[dict[str, Any]]) -> None:
    """Print autodetect results to stdout."""
    if not hits:
        print("No items predicted as Good.")
        return

    print(f"Predicted Good ({len(hits)} items):\n")
    for hit in hits:
        print(f"  {hit['filename']}  (score: {hit['score']}, category: {hit['category']})")


def autodetect_main(dataset_path: str, detector_path: str) -> None:
    """CLI entry point: run autodetect and print results to stdout.

    Prints each predicted-Good clip as a line with its filename and score.
    Exits with code 0 on success, 1 on error.

    Args:
        dataset_path: Path to the dataset pickle file.
        detector_path: Path to the detector JSON file.
    """
    try:
        hits = run_autodetect(dataset_path, detector_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    _print_hits(hits)


def autodetect_importer_main(
    importer_name: str,
    field_values: dict[str, Any],
    detector_path: str,
) -> None:
    """CLI entry point: run autodetect with a named importer and print results.

    Exits with code 0 on success, 1 on error.

    Args:
        importer_name: Registered name of the importer.
        field_values: Mapping of importer field keys to their CLI values.
        detector_path: Path to the detector JSON file.
    """
    try:
        hits = run_autodetect_with_importer(importer_name, field_values, detector_path)
    except (FileNotFoundError, ValueError, NotADirectoryError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    _print_hits(hits)
