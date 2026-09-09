"""Reading the Visual Genome source tree: image paths, records, dimensions."""

from __future__ import annotations

import json
from pathlib import Path

import pile_config as pc

from pilebuild.env import log


def vg_image_paths() -> dict[int, Path]:
    """``{image_id: path}`` over both VG image dirs."""
    vg_root = pc.DEMO_CACHE / "visual_genome"
    paths: dict[int, Path] = {}
    for d in (vg_root / "VG_100K", vg_root / "VG_100K_2"):
        for p in d.iterdir():
            if p.suffix.lower() == ".jpg":
                try:
                    paths[int(p.stem)] = p
                except ValueError:
                    continue
    return paths


def vg_objects_json() -> Path:
    """The VG annotation file every VG-derived build reads.

    Named once, and both the loaders and the rebuild canary go through it. A
    canary that spells a source path of its own is how ``coco_val`` reported
    REBUILD-BROKEN against a staging area that was entirely intact (#3299).
    """
    return pc.DEMO_CACHE / "visual_genome" / "objects.json"


def vg_boxes_by_name(rec: dict, wanted: set[str]) -> dict[str, list[list[float]]]:
    """This record's boxes for the categories in *wanted*, in VG pixel space.

    VG names an object with a list of synonyms and the first is its primary, so
    only that one is matched -- taking any of them would file one object under
    several categories. Degenerate boxes drop out.

    **In this release of VG there is nothing else to take.** All 2,516,939
    objects carry a ``names`` list of length exactly one (#3618), so the
    primary-name restriction costs nothing here and reading the rest of the list
    is not an available fix for a class built from one spelling. That fix is
    :data:`pile_config.SCALE_VG_NAMES`.
    """
    from collections import defaultdict  # noqa: PLC0415

    by_name: dict[str, list[list[float]]] = defaultdict(list)
    for obj in rec.get("objects") or []:
        names = obj.get("names") or []
        if not names:
            continue
        name = str(names[0]).strip().lower()
        if name not in wanted:
            continue
        x, y = float(obj.get("x", 0)), float(obj.get("y", 0))
        w, h = float(obj.get("w", 0)), float(obj.get("h", 0))
        if w > 0 and h > 0:
            by_name[name].append([x, y, x + w, y + h])
    return dict(by_name)


def vg_source() -> tuple[dict[int, Path], list, dict[int, tuple[int, int]]]:
    """``(image paths, objects.json records, image dims)`` for the whole VG source.

    Dims come from ``scan_vg_boxes.py``'s cache when it exists (it always does
    in practice -- the scan is what chooses the classes), and are read from the
    JPEG headers otherwise, which costs ~30 s.
    """
    from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

    from PIL import Image  # noqa: PLC0415

    objects_json = vg_objects_json()
    if not objects_json.exists():
        raise SystemExit(f"missing {objects_json}")

    paths = vg_image_paths()
    cache = pc.PILE / "vg_image_dims.json"
    dims: dict[int, tuple[int, int]] = {}
    if cache.exists():
        raw = json.loads(cache.read_text())
        # Unreadable images are cached as null, so a complete cache has one
        # entry per file (see scan_vg_boxes._read_dims).
        if len(raw) >= len(paths):
            dims = {int(k): tuple(v) for k, v in raw.items() if v}  # type: ignore[misc]
    if not dims:
        log("  no dims cache; reading JPEG headers")

        def one(item):
            iid, path = item
            try:
                with Image.open(path) as im:
                    return iid, im.size
            except Exception:  # noqa: BLE001 - a corrupt file just drops out
                return iid, None

        with ThreadPoolExecutor(max_workers=16) as ex:
            for iid, size in ex.map(one, paths.items(), chunksize=256):
                if size:
                    dims[iid] = size

    with objects_json.open() as fh:
        records = json.load(fh)
    return paths, records, dims
