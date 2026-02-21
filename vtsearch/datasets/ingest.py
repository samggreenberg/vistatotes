"""Ingest missing clips into an existing dataset by re-running their origins.

When a label-set references elements that are not present in the current
dataset, the user may choose to pull them in from their original sources.
This module groups the missing entries by origin, runs the appropriate
dataset importer for each origin, and cherry-picks only the clips that
match the missing entries (by ``origin_name``).  The recovered clips are
assigned fresh IDs that do not collide with existing clips and are
appended to the in-memory dataset.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Optional

from vtsearch.utils.state import next_clip_id

ProgressCallback = Callable[[str, str, int, int], None]


def _default_progress() -> ProgressCallback:
    from vtsearch.utils import update_progress

    return update_progress


def _group_by_origin(
    entries: list[dict[str, Any]],
) -> dict[str, tuple[dict[str, Any], list[dict[str, Any]]]]:
    """Group label entries by their serialised origin.

    Returns a dict mapping ``json.dumps(origin, sort_keys=True)`` to a tuple
    of ``(origin_dict, [entries_with_that_origin])``.  Entries without an
    origin are silently skipped (they cannot be re-ingested).
    """
    groups: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    for entry in entries:
        origin = entry.get("origin")
        if origin is None:
            continue
        key = json.dumps(origin, sort_keys=True)
        if key not in groups:
            groups[key] = (origin, [])
        groups[key][1].append(entry)
    return groups


def ingest_missing_clips(
    missing_entries: list[dict[str, Any]],
    clips: dict[int, dict[str, Any]],
    on_progress: Optional[ProgressCallback] = None,
) -> int:
    """Re-ingest missing clips from their origins into *clips*.

    Groups *missing_entries* by origin, runs each origin's dataset importer
    to recover the full clip data (media bytes + embedding), then appends
    the matched clips to *clips* with fresh, non-colliding IDs.

    Args:
        missing_entries: Label entries (dicts with ``origin``,
            ``origin_name``, ``md5``, ``label``, etc.) that were not found
            in the current dataset.
        clips: The live dataset dict to extend in-place.
        on_progress: Optional progress callback.

    Returns:
        The number of clips successfully ingested.
    """
    from vtsearch.datasets.importers import get_importer

    if on_progress is None:
        on_progress = _default_progress()

    groups = _group_by_origin(missing_entries)
    if not groups:
        return 0

    total_ingested = 0

    for origin_key, (origin_dict, entries) in groups.items():
        importer_name = origin_dict.get("importer", "")
        importer = get_importer(importer_name)
        if importer is None:
            continue

        params = origin_dict.get("params", {})

        # Build a set of origin_names we're looking for
        wanted_names: set[str] = set()
        wanted_md5s: set[str] = set()
        for entry in entries:
            name = entry.get("origin_name", "")
            if name:
                wanted_names.add(name)
            md5 = entry.get("md5", "")
            if md5:
                wanted_md5s.add(md5)

        on_progress(
            "ingesting",
            f"Re-ingesting from {importer_name} ({len(wanted_names)} clips)...",
            0,
            0,
        )

        # Run the importer into a temporary clips dict
        temp_clips: dict[int, dict[str, Any]] = {}
        try:
            importer.run_cli(params, temp_clips)
        except Exception:
            # If the importer fails (e.g. folder not found), skip this origin
            continue

        # Set origin on temp clips that don't have one
        for clip in temp_clips.values():
            if clip.get("origin") is None:
                clip["origin"] = origin_dict
            if not clip.get("origin_name"):
                clip["origin_name"] = clip.get("filename", "")

        # Cherry-pick matching clips
        cid = next_clip_id(clips)
        for temp_clip in temp_clips.values():
            clip_origin_name = temp_clip.get("origin_name", "")
            clip_md5 = temp_clip.get("md5", "")
            if clip_origin_name in wanted_names or clip_md5 in wanted_md5s:
                temp_clip["id"] = cid
                clips[cid] = temp_clip
                cid += 1
                total_ingested += 1

    on_progress("idle", f"Ingested {total_ingested} clip(s) from origins.", 0, 0)
    return total_ingested
