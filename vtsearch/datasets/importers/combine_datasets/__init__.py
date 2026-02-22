"""Combine-datasets importer -- merge multiple pickle datasets into one.

All source datasets must share the same media type.  Duplicate entries
(identified by MD5 hash) are kept only once.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vtsearch.datasets.importers.base import DatasetImporter, ImporterField


def _get_progress():
    from vtsearch.utils import update_progress

    return update_progress


def _load_clips_from_pickle(file_path: Path) -> dict[int, dict[str, Any]]:
    """Load clips from a pickle file without clearing a target dict.

    Returns a fresh dict mapping clip-id to clip-data.  This is a
    simplified version of :func:`~vtsearch.datasets.loader.load_dataset_from_pickle`
    that avoids side-effects on any global state.
    """
    from vtsearch.datasets.loader import load_dataset_from_pickle

    temp_clips: dict[int, dict[str, Any]] = {}
    load_dataset_from_pickle(file_path, temp_clips)
    return temp_clips


class CombineDatasetsImporter(DatasetImporter):
    """Merge two or more existing ``.pkl`` datasets into a single dataset.

    All datasets must be of the same media type.  Entries with duplicate
    MD5 hashes are included only once (the first occurrence wins).
    """

    name = "combine_datasets"
    display_name = "Combine Existing Datasets"
    description = "Merge multiple .pkl datasets into one, skipping duplicates."
    icon = "\U0001f500"  # twisted rightwards arrows
    fields = [
        ImporterField(
            key="datasets",
            label="Dataset Files",
            field_type="text",
            description="Comma-separated paths to .pkl dataset files.",
        ),
    ]

    def run(self, field_values: dict[str, Any], clips: dict, thin: bool = False) -> None:
        """Combine datasets specified by *field_values['datasets']*.

        ``field_values["datasets"]`` may be either:
        - a comma-separated string of file paths, or
        - a Python list of path strings (when called from the API route).
        """
        raw = field_values.get("datasets", "")
        if isinstance(raw, list):
            paths = [Path(p) for p in raw if p]
        else:
            paths = [Path(p.strip()) for p in raw.split(",") if p.strip()]

        if len(paths) < 2:
            raise ValueError("At least two datasets are required to combine.")

        for p in paths:
            if not p.exists():
                raise FileNotFoundError(f"Dataset file not found: {p}")

        progress = _get_progress()

        # Load each dataset, validate media types, collect clips
        all_clips: list[dict[str, Any]] = []
        media_type: str | None = None
        seen_md5s: set[str] = set()
        total_dupes = 0

        for i, pkl_path in enumerate(paths):
            progress(
                "loading",
                f"Loading dataset {i + 1}/{len(paths)}: {pkl_path.name}...",
                i + 1,
                len(paths),
            )
            source_clips = _load_clips_from_pickle(pkl_path)

            if not source_clips:
                progress("loading", f"Skipping empty dataset: {pkl_path.name}", i + 1, len(paths))
                continue

            # Check media type consistency
            first_clip = next(iter(source_clips.values()))
            source_media_type = first_clip.get("type", "audio")

            if media_type is None:
                media_type = source_media_type
            elif source_media_type != media_type:
                raise ValueError(
                    f"Media type mismatch: expected '{media_type}' but "
                    f"'{pkl_path.name}' contains '{source_media_type}' clips."
                )

            # Collect clips, deduplicating by MD5
            for clip in source_clips.values():
                md5 = clip.get("md5", "")
                if md5 and md5 in seen_md5s:
                    total_dupes += 1
                    continue
                if md5:
                    seen_md5s.add(md5)
                all_clips.append(clip)

        if not all_clips:
            raise ValueError("No clips found in any of the selected datasets.")

        # Assign fresh sequential IDs and populate the target clips dict
        clips.clear()
        for new_id, clip in enumerate(all_clips, start=1):
            clip["id"] = new_id
            clips[new_id] = clip

        msg = f"Combined {len(clips)} clips from {len(paths)} datasets"
        if total_dupes:
            msg += f" ({total_dupes} duplicate(s) skipped)"
        progress("idle", msg)

    def run_cli(self, field_values: dict[str, Any], clips: dict, thin: bool = False) -> None:
        """CLI entry point -- *datasets* is a comma-separated path string."""
        self.run(field_values, clips, thin=thin)

    def build_origin(self, field_values: dict[str, Any]) -> dict[str, Any]:
        """Build an origin dict listing the source dataset paths."""
        raw = field_values.get("datasets", "")
        if isinstance(raw, list):
            datasets_str = ",".join(raw)
        else:
            datasets_str = raw
        return {"importer": self.name, "params": {"datasets": datasets_str}}


IMPORTER = CombineDatasetsImporter()
