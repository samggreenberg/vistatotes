"""Command-line interface utilities for VTSearch.

The only CLI workflow is autodetect: load a dataset (from pickle or via an
importer), score it against favourite processors from a settings file, and
export the results.  Datasets, detectors, and labelsets are loaded
indirectly as part of this workflow — there are no standalone CLI commands
for importing them individually.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vtsearch.datasets.loader import load_dataset_from_pickle


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
    from vtsearch.models.training import build_model

    input_dim = len(weights["0.weight"][0])

    model = build_model(input_dim)

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
        scores = torch.sigmoid(model(X_all)).squeeze(1).tolist()

    # Collect positive hits (score >= threshold)
    positive_hits = []
    for cid, score in zip(all_ids, scores):
        if score >= threshold:
            clip = clips[cid]
            hit: dict[str, Any] = {
                "id": cid,
                "filename": clip.get("filename", f"clip_{cid}"),
                "category": clip.get("category", "unknown"),
                "score": round(score, 4),
            }
            if clip.get("origin") is not None:
                hit["origin"] = clip["origin"]
            if clip.get("origin_name"):
                hit["origin_name"] = clip["origin_name"]
            if clip.get("md5"):
                hit["md5"] = clip["md5"]
            positive_hits.append(hit)

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
    from vtsearch.datasets.importers import get_importer

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
    from vtsearch.datasets.importers import list_importers

    return [imp.name for imp in list_importers()]


def _list_exporter_names() -> list[str]:
    """Return the names of all registered exporters."""
    from vtsearch.exporters import list_exporters

    return [exp.name for exp in list_exporters()]


def _print_hits(hits: list[dict[str, Any]]) -> None:
    """Print autodetect results to stdout."""
    if not hits:
        print("No items predicted as Good.")
        return

    print(f"Predicted Good ({len(hits)} items):\n")
    for hit in hits:
        print(f"  {hit['filename']}  (score: {hit['score']}, category: {hit['category']})")


def _build_results_dict(
    hits: list[dict[str, Any]],
    detector_path: str,
    media_type: str = "unknown",
) -> dict[str, Any]:
    """Build the full results dict expected by exporters.

    Args:
        hits: List of hit dicts from :func:`_score_clips_with_detector`.
        detector_path: Path to the detector JSON (re-read for metadata).
        media_type: The media type string for the dataset.

    Returns:
        A dict matching the shape expected by
        :meth:`~vtsearch.exporters.base.LabelsetExporter.export`.
    """
    detector_data = json.loads(Path(detector_path).read_text())
    detector_name = detector_data.get("name", Path(detector_path).stem)
    threshold = detector_data.get("threshold", 0.5)

    return {
        "media_type": media_type,
        "detectors_run": 1,
        "results": {
            detector_name: {
                "detector_name": detector_name,
                "threshold": threshold,
                "total_hits": len(hits),
                "hits": hits,
            }
        },
    }


def _score_clips_with_detectors(
    clips: dict[int, dict[str, Any]],
    detectors: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Score all *clips* against every detector in *detectors*.

    Embeddings are extracted once and reused across all detectors.

    Args:
        clips: The clips dict (must contain ``"embedding"`` per clip).
        detectors: Mapping of detector name to detector data dict (with
            ``"weights"`` and ``"threshold"`` keys).

    Returns:
        A dict mapping detector name to its results sub-dict (with
        ``"detector_name"``, ``"threshold"``, ``"total_hits"``, ``"hits"``).

    Raises:
        ValueError: If *clips* or *detectors* is empty.
    """
    if not clips:
        raise ValueError("No clips loaded from dataset")
    if not detectors:
        raise ValueError("No favorite processors found for the dataset's media type")

    all_ids = sorted(clips.keys())
    all_embs = np.array([clips[cid]["embedding"] for cid in all_ids])
    X_all = torch.tensor(all_embs, dtype=torch.float32)

    results: dict[str, dict[str, Any]] = {}
    for detector_name, detector_data in detectors.items():
        weights = detector_data["weights"]
        threshold = detector_data["threshold"]

        input_dim = len(weights["0.weight"][0])

        from vtsearch.models.training import build_model

        model = build_model(input_dim)

        state_dict = {}
        for key, value in weights.items():
            state_dict[key] = torch.tensor(value, dtype=torch.float32)
        model.load_state_dict(state_dict)
        model.eval()

        with torch.no_grad():
            scores = torch.sigmoid(model(X_all)).squeeze(1).tolist()

        positive_hits: list[dict[str, Any]] = []
        for cid, score in zip(all_ids, scores):
            if score >= threshold:
                clip = clips[cid]
                hit: dict[str, Any] = {
                    "id": cid,
                    "filename": clip.get("filename", f"clip_{cid}"),
                    "category": clip.get("category", "unknown"),
                    "score": round(score, 4),
                }
                if clip.get("origin") is not None:
                    hit["origin"] = clip["origin"]
                if clip.get("origin_name"):
                    hit["origin_name"] = clip["origin_name"]
                if clip.get("md5"):
                    hit["md5"] = clip["md5"]
                positive_hits.append(hit)

        positive_hits.sort(key=lambda x: x["score"], reverse=True)

        results[detector_name] = {
            "detector_name": detector_name,
            "threshold": round(threshold, 4),
            "total_hits": len(positive_hits),
            "hits": positive_hits,
        }

    return results


def _build_multi_results_dict(
    detector_results: dict[str, dict[str, Any]],
    media_type: str = "unknown",
) -> dict[str, Any]:
    """Build the full results dict from multi-detector scoring.

    Args:
        detector_results: Per-detector results from
            :func:`_score_clips_with_detectors`.
        media_type: The media type string for the dataset.

    Returns:
        A dict matching the shape expected by exporters.
    """
    return {
        "media_type": media_type,
        "detectors_run": len(detector_results),
        "results": detector_results,
    }


def _detect_media_type(clips: dict[int, dict[str, Any]]) -> str:
    """Return the media type from the first clip, or ``"unknown"``."""
    for clip in clips.values():
        return clip.get("type", "unknown")
    return "unknown"


def _run_exporter(
    exporter_name: str,
    field_values: dict[str, Any],
    results: dict[str, Any],
) -> None:
    """Validate and run a named exporter, printing its confirmation message.

    Raises:
        ValueError: If the exporter is unknown or required fields are missing.
    """
    from vtsearch.exporters import get_exporter

    exporter = get_exporter(exporter_name)
    if exporter is None:
        available = _list_exporter_names()
        raise ValueError(f"Unknown exporter: {exporter_name}. Available: {', '.join(available)}")

    exporter.validate_cli_field_values(field_values)
    result = exporter.export_cli(results, field_values)
    print(result.get("message", "Export complete."))


def _import_favorite_processors(settings_path: str | None = None) -> None:
    """Import favorite processors from settings (if any).

    When *settings_path* is provided the settings module is pointed at that
    file before importing; otherwise the default ``data/settings.json`` is
    used.

    Errors are logged as warnings but do not halt the autodetect run.
    """
    try:
        if settings_path:
            from vtsearch.settings import set_settings_path

            set_settings_path(settings_path)

        from vtsearch.settings import ensure_favorite_processors_imported

        imported = ensure_favorite_processors_imported()
        if imported:
            print(f"Imported {len(imported)} favorite processor(s) from settings: {', '.join(imported)}")
    except Exception as exc:
        print(f"Warning: could not load favorite processors from settings: {exc}", file=sys.stderr)


def autodetect_main(
    dataset_path: str,
    settings_path: str | None = None,
    exporter_name: str | None = None,
    exporter_field_values: dict[str, Any] | None = None,
) -> None:
    """CLI entry point: run autodetect with all favorite processors and output results.

    Loads favorite processors from the settings file (defaulting to the
    normal ``data/settings.json``), scores the dataset against every
    processor matching the dataset's media type, and exports a combined
    result set with one column per processor.

    When *exporter_name* is ``None`` the hits are printed to stdout.
    Otherwise the named exporter is used to deliver the results.

    Exits with code 0 on success, 1 on error.

    Args:
        dataset_path: Path to the dataset pickle file.
        settings_path: Optional path to a settings JSON file.  When ``None``
            the default ``data/settings.json`` is used.
        exporter_name: Optional registered exporter name.
        exporter_field_values: Optional exporter field values.
    """
    try:
        _import_favorite_processors(settings_path)

        dataset_file = Path(dataset_path)
        if not dataset_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

        clips: dict[int, dict[str, Any]] = {}
        load_dataset_from_pickle(dataset_file, clips)
        if not clips:
            raise ValueError(f"No clips loaded from dataset: {dataset_path}")

        media_type = _detect_media_type(clips)

        from vtsearch.utils import get_favorite_detectors_by_media

        detectors = get_favorite_detectors_by_media(media_type)
        if not detectors:
            raise ValueError(
                f"No favorite processors found for media type: {media_type}. "
                "Add processors to the settings file or use --settings to specify one."
            )

        detector_results = _score_clips_with_detectors(clips, detectors)
        results = _build_multi_results_dict(detector_results, media_type)
        _run_exporter(exporter_name or "gui", exporter_field_values or {}, results)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def autodetect_importer_main(
    importer_name: str,
    field_values: dict[str, Any],
    settings_path: str | None = None,
    exporter_name: str | None = None,
    exporter_field_values: dict[str, Any] | None = None,
) -> None:
    """CLI entry point: run autodetect with a named importer and output results.

    Loads favorite processors from the settings file (defaulting to the
    normal ``data/settings.json``), scores the imported dataset against
    every processor matching the dataset's media type, and exports a
    combined result set with one column per processor.

    When *exporter_name* is ``None`` the hits are printed to stdout.
    Otherwise the named exporter is used to deliver the results.

    Exits with code 0 on success, 1 on error.

    Args:
        importer_name: Registered name of the importer.
        field_values: Mapping of importer field keys to their CLI values.
        settings_path: Optional path to a settings JSON file.  When ``None``
            the default ``data/settings.json`` is used.
        exporter_name: Optional registered exporter name.
        exporter_field_values: Optional exporter field values.
    """
    try:
        _import_favorite_processors(settings_path)

        from vtsearch.datasets.importers import get_importer

        importer = get_importer(importer_name)
        if importer is None:
            available = _list_importer_names()
            raise ValueError(f"Unknown importer: {importer_name}. Available: {', '.join(available)}")

        importer.validate_cli_field_values(field_values)

        clips: dict[int, dict[str, Any]] = {}
        importer.run_cli(field_values, clips)
        if not clips:
            raise ValueError(f"No clips loaded by importer '{importer_name}'")

        media_type = _detect_media_type(clips)

        from vtsearch.utils import get_favorite_detectors_by_media

        detectors = get_favorite_detectors_by_media(media_type)
        if not detectors:
            raise ValueError(
                f"No favorite processors found for media type: {media_type}. "
                "Add processors to the settings file or use --settings to specify one."
            )

        detector_results = _score_clips_with_detectors(clips, detectors)
        results = _build_multi_results_dict(detector_results, media_type)
        _run_exporter(exporter_name or "gui", exporter_field_values or {}, results)
    except (FileNotFoundError, ValueError, NotADirectoryError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


