"""Command-line interface utilities for VTSearch."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

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


def _import_favorite_processors() -> None:
    """Import favorite processors from settings (if any).

    Errors are logged as warnings but do not halt the autodetect run.
    """
    try:
        from vtsearch.settings import ensure_favorite_processors_imported

        imported = ensure_favorite_processors_imported()
        if imported:
            print(f"Imported {len(imported)} favorite processor(s) from settings: {', '.join(imported)}")
    except Exception as exc:
        print(f"Warning: could not load favorite processors from settings: {exc}", file=sys.stderr)


def autodetect_main(
    dataset_path: str,
    detector_path: str,
    exporter_name: str | None = None,
    exporter_field_values: dict[str, Any] | None = None,
) -> None:
    """CLI entry point: run autodetect and output results.

    When *exporter_name* is ``None`` the hits are printed to stdout.
    Otherwise the named exporter is used to deliver the results.

    Exits with code 0 on success, 1 on error.

    Args:
        dataset_path: Path to the dataset pickle file.
        detector_path: Path to the detector JSON file.
        exporter_name: Optional registered exporter name.
        exporter_field_values: Optional exporter field values.
    """
    try:
        _import_favorite_processors()

        dataset_file = Path(dataset_path)
        if not dataset_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

        clips: dict[int, dict[str, Any]] = {}
        load_dataset_from_pickle(dataset_file, clips)
        if not clips:
            raise ValueError(f"No clips loaded from dataset: {dataset_path}")

        hits = _score_clips_with_detector(clips, detector_path)

        media_type = _detect_media_type(clips)
        results = _build_results_dict(hits, detector_path, media_type)
        _run_exporter(exporter_name or "gui", exporter_field_values or {}, results)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def import_labels_main(
    dataset_path: str,
    label_importer_name: str,
    field_values: dict[str, Any],
    *,
    auto_import_missing: bool | None = None,
) -> None:
    """CLI entry point: load a dataset and import labels via a named label importer.

    Prints a summary of applied and skipped labels to stdout.
    Exits with code 0 on success, 1 on error.

    When label entries reference elements not present in the dataset (neither
    by origin+name nor md5), the user is prompted whether to import the
    missing clips from their origins.  Pass *auto_import_missing* to bypass
    the prompt (``True`` to always import, ``False`` to always skip).

    Args:
        dataset_path: Path to the dataset pickle file.
        label_importer_name: Registered name of the label importer.
        field_values: Mapping of label importer field keys to their CLI values.
        auto_import_missing: If ``True``/``False``, skip the interactive
            prompt and automatically import/skip missing entries.
    """
    try:
        from vtsearch.labels.importers import get_label_importer

        label_importer = get_label_importer(label_importer_name)
        if label_importer is None:
            from vtsearch.labels.importers import list_label_importers

            available = ", ".join(imp.name for imp in list_label_importers())
            raise ValueError(f"Unknown label importer: {label_importer_name}. Available: {available}")

        label_importer.validate_cli_field_values(field_values)

        dataset_file = Path(dataset_path)
        if not dataset_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

        clips: dict[int, dict[str, Any]] = {}
        load_dataset_from_pickle(dataset_file, clips)
        if not clips:
            raise ValueError(f"No clips loaded from dataset: {dataset_path}")

        label_entries = label_importer.run_cli(field_values)
        if not isinstance(label_entries, list):
            raise ValueError("Label importer did not return a list of label dicts.")

        # Apply labels by origin+origin_name and md5 matching (union)
        from vtsearch.utils import build_clip_lookup, find_missing_entries, resolve_clip_ids

        origin_lookup, md5_lookup = build_clip_lookup(clips)
        applied = 0
        skipped = 0
        for entry in label_entries:
            label = entry.get("label", "")
            if label not in ("good", "bad"):
                skipped += 1
                continue
            cids = resolve_clip_ids(entry, origin_lookup, md5_lookup)
            if not cids:
                skipped += 1
                continue
            applied += 1

        # Detect entries not matched by origin+name or md5
        missing = find_missing_entries(label_entries, origin_lookup, md5_lookup)
        skipped -= len(missing)

        print(f"Applied {applied} label(s), skipped {skipped}.")

        if missing:
            should_import = auto_import_missing
            if should_import is None:
                # Interactive prompt
                answer = input(
                    f"{len(missing)} element(s) not found in dataset. "
                    "Import them from their origins? [y/N] "
                ).strip().lower()
                should_import = answer in ("y", "yes")

            if should_import:
                from vtsearch.datasets.ingest import ingest_missing_clips

                def _cli_progress(status: str, message: str, current: int, total: int) -> None:
                    if message:
                        print(message)

                ingested = ingest_missing_clips(missing, clips, on_progress=_cli_progress)

                # Apply labels to the newly ingested clips
                origin_lookup2, md5_lookup2 = build_clip_lookup(clips)
                extra_applied = 0
                for entry in missing:
                    cids = resolve_clip_ids(entry, origin_lookup2, md5_lookup2)
                    if cids:
                        extra_applied += 1
                print(f"Ingested {ingested} clip(s), applied {extra_applied} additional label(s).")
            else:
                print(f"Skipped {len(missing)} missing element(s).")

    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def autodetect_importer_main(
    importer_name: str,
    field_values: dict[str, Any],
    detector_path: str,
    exporter_name: str | None = None,
    exporter_field_values: dict[str, Any] | None = None,
) -> None:
    """CLI entry point: run autodetect with a named importer and output results.

    When *exporter_name* is ``None`` the hits are printed to stdout.
    Otherwise the named exporter is used to deliver the results.

    Exits with code 0 on success, 1 on error.

    Args:
        importer_name: Registered name of the importer.
        field_values: Mapping of importer field keys to their CLI values.
        detector_path: Path to the detector JSON file.
        exporter_name: Optional registered exporter name.
        exporter_field_values: Optional exporter field values.
    """
    try:
        _import_favorite_processors()

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

        hits = _score_clips_with_detector(clips, detector_path)

        media_type = _detect_media_type(clips)
        results = _build_results_dict(hits, detector_path, media_type)
        _run_exporter(exporter_name or "gui", exporter_field_values or {}, results)
    except (FileNotFoundError, ValueError, NotADirectoryError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def import_processor_main(
    processor_importer_name: str,
    field_values: dict[str, Any],
    processor_name: str,
) -> None:
    """CLI entry point: import a processor (detector) via a named processor importer.

    Runs the processor importer and saves the result as a favorite detector.
    Prints a summary to stdout.  Exits with code 0 on success, 1 on error.

    Args:
        processor_importer_name: Registered name of the processor importer.
        field_values: Mapping of importer field keys to their CLI values.
        processor_name: Name to assign to the imported detector.
    """
    try:
        from vtsearch.processors.importers import get_processor_importer

        proc_importer = get_processor_importer(processor_importer_name)
        if proc_importer is None:
            from vtsearch.processors.importers import list_processor_importers

            available = ", ".join(imp.name for imp in list_processor_importers())
            raise ValueError(f"Unknown processor importer: {processor_importer_name}. Available: {available}")

        proc_importer.validate_cli_field_values(field_values)
        result = proc_importer.run_cli(field_values)

        if not isinstance(result, dict):
            raise ValueError("Processor importer did not return a dict.")

        media_type = result.get("media_type", "audio")
        weights = result.get("weights")
        threshold = result.get("threshold", 0.5)

        if not weights:
            raise ValueError("Processor importer result missing 'weights'.")

        from vtsearch.utils import add_favorite_detector

        add_favorite_detector(processor_name, media_type, weights, threshold)

        loaded = result.get("loaded", "")
        skipped = result.get("skipped", "")
        extra = ""
        if loaded:
            extra += f", loaded {loaded} files"
        if skipped:
            extra += f", skipped {skipped}"
        print(f"Imported detector '{processor_name}' ({media_type}{extra}).")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
