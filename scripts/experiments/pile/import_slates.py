"""Import each slate as a VTSearch dataset and open an empty detector for it.

The slate builder writes folders of JPEGs; this is the step that puts them in
front of a reviewer. One dataset and one detector per class, both carrying the
**same** name -- the one from ``pile_config.SCALE_CLASS_RULES``, which states
the definition the class is being reviewed under. That name is the only string
the app shows while voting, so it is the only place a rule can live where a
reviewer will actually see it. `book` split over magazines because the rule
lived in a manifest instead (``make_definition_reslate.py``).

The detector is created **empty**: no labels, seeded only with the class name as
a text query so Autopilot has somewhere to start. It exists so the votes have
somewhere to land and so the dashboard shows the work outstanding.

Runs the importer in-process rather than against a live server. There is no
HTTP hop to authenticate, and the memory cost is one dataset at a time -- but
it does mean this must not run while the app is up on the same data dir, since
both would write the registry. Stop the app first.

**Every image is embedded again, and the pile already holds its vector** (#3669).
The importer takes a folder and computes; there is no reuse path, so a slate
drawn from the pile pays ~1.2 s/image on CPU for numbers that exist on disk.
This now prints the device it resolved and what that implies for the slate in
front of it, because the failure was silent: `auto` becomes CUDA or CPU
depending on which partition the job landed on, and a twelve-minute import of
200 images looks exactly like a slow one.

Usage::

    python import_slates.py --slates /expscratch/$USER/classes-3588/slates
    python import_slates.py --slates ... --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pile_config as pc


def log(msg: str) -> None:
    print(f"[import] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slates", default=str(pc.PILE.parent / "classes-3588" / "slates"))
    ap.add_argument(
        "--data-dir",
        default="/exp/sgreenberg/projects/VTSearch/data",
        help="the VTSearch data dir the reviewer's app reads (NOT the pile's)",
    )
    ap.add_argument("--embedder", default="siglip")
    ap.add_argument("--media-type", default="image")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--timeout", type=int, default=1800, help="seconds to wait for one import")
    args = ap.parse_args()

    index = json.loads((Path(args.slates) / "slates.json").read_text())
    log(f"{len(index)} slates under {args.slates}")
    total_images = sum(int(e.get("n") or 0) for e in index)

    # WHICH DEVICE, before anything is embedded, and before --dry-run returns --
    # planning the run is exactly what a dry run is for (#3669). The embedder
    # resolves `auto` against whatever host it lands on, so the same command is
    # minutes on a GPU node and hours on a CPU one, and nothing said so: the
    # #3588 pass paid ~12 minutes for well under a minute of GPU work, and a
    # slate import sent to the wrong partition looked exactly like a slow one.
    #
    # `app` lives at the repo root; this script runs from scripts/experiments/pile.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from vtscore.config import resolve_device  # noqa: PLC0415

    device = resolve_device()
    # 1.2 s/image measured on CPU (#3669: 200 images, 238 s). The GPU figure is
    # an order-of-magnitude placeholder and is labelled as one rather than
    # quoted: nobody has timed this import on a GPU.
    rate, sure = (1.2, "measured") if device == "cpu" else (0.05, "guessed")
    log(f"embedding device: {device} -- {total_images} images, roughly {total_images * rate / 60:.0f} min ({sure})")
    if device == "cpu":
        log("  every one of those vectors is already in the pile; a CPU import is hours (#3669)")

    if args.dry_run:
        for e in index:
            print(f"  {e['name']:<40} {e['n']:>4} images  {e['dir']}")
        return 0

    # Point at the reviewer's data dir BEFORE importing app, which resolves the
    # registries at import time.
    os.environ["VTSEARCH_DATA_DIR"] = args.data_dir
    os.environ.setdefault("VTSEARCH_MODELS_DIR", str(pc.PILE / "models"))
    os.environ.setdefault("HF_HOME", str(pc.PILE / "models"))

    import app  # noqa: F401, PLC0415  -- wires the registries

    from vtscore.datasets import registry as ds_registry  # noqa: PLC0415
    from vtscore.datasets.importers import get_importer  # noqa: PLC0415
    from vtscore.datasets.labelset import LabelSet  # noqa: PLC0415
    from vtscore.datasets.load_pipeline import _run_importer_in_background, loading_tasks  # noqa: PLC0415
    from vtscore.detectors import registry as det_registry  # noqa: PLC0415
    from vtscore.detectors.store import _detector_path, _write_detector  # noqa: PLC0415

    # The success path calls mark_finished and never returns the tracker to
    # 'idle', so has_active_tasks() stays True forever afterwards -- waiting on
    # it would hang. Watch mark_finished instead.
    finished: dict[str, float] = {}
    _orig = loading_tasks.mark_finished

    def _mark(tid, *a, **k):
        finished.setdefault(tid, time.monotonic())
        return _orig(tid, *a, **k)

    loading_tasks.mark_finished = _mark  # type: ignore[method-assign]

    def dataset_named(n: str) -> dict | None:
        return next((d for d in ds_registry.list_datasets() if d.get("name") == n), None)

    made_ds = made_det = skipped_ds = skipped_det = 0
    for e in index:
        name, folder = e["name"], e["dir"]

        if dataset_named(name):
            log(f"  dataset exists, skipping: {name!r}")
            skipped_ds += 1
        else:
            t0 = time.monotonic()
            tid = _run_importer_in_background(
                get_importer("server_folder"),
                {
                    "path": folder,
                    "media_type": args.media_type,
                    "embedder": args.embedder,
                    "recursive": "false",
                    "dig_archives": "false",
                    # The importer's own default name is the folder leaf, i.e.
                    # the rule with underscores. `dataset_name` is the field the
                    # UI's name box writes, so the reviewer sees the rule as
                    # prose rather than as a slug.
                    "dataset_name": name,
                },
            )
            while tid not in finished:
                if time.monotonic() - t0 > args.timeout:
                    log(f"  TIMEOUT importing {name!r} after {args.timeout}s")
                    break
                time.sleep(0.5)
            entry = dataset_named(name)
            if entry is None:
                log(f"  FAILED to import {name!r} -- no registry entry")
                continue
            made_ds += 1
            log(f"  dataset {name!r}: {entry.get('num_items')} items in {time.monotonic() - t0:.0f}s")

        if det_registry.find_by_name(name):
            log(f"  detector exists, skipping: {name!r}")
            skipped_det += 1
            continue
        # Seeded with the class name, not the rule: the rule is prose for a
        # human and would drag the text sort somewhere the class is not.
        examples = [{"type": "text", "value": e["class"]}]
        _write_detector(
            _detector_path(name),
            {
                "name": name,
                "text_query": e["class"],
                "media_example": "",
                "media_type": args.media_type,
                "examples": examples,
                "created_at": time.time(),
                "embedder_type": "semantic",
                "labelset": LabelSet([]).to_dict(),
            },
        )
        det_registry.register_detector(
            name=name,
            media_type=args.media_type,
            num_training=0,
            text_query=e["class"],
            examples=examples,
            embedder_type="semantic",
        )
        made_det += 1
        log(f"  detector {name!r}: empty")

    print(
        f"\n{made_ds} datasets imported ({skipped_ds} already there), "
        f"{made_det} empty detectors created ({skipped_det} already there)"
    )
    print(f"data dir: {args.data_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
