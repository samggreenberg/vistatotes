#!/usr/bin/env python3
"""Import the 25 per-class slates into the app and check they actually landed.

Two traps this deliberately avoids, both hit before on this project:

* **`/api/detectors` writes a detector file without registering it**, so the UI
  never lists it. Anything touching detectors goes through
  `/api/detectors/registry`, whose reply nests the object under `detector`.
* **an import call returns as soon as it is queued.** Reporting that as success
  is how a run gets called done while nothing has been ingested, so every import
  here is polled to completion and then re-counted from the datasets listing.

Existing detectors are reused, never recreated: they carry 1,223 human verdicts
and a detector is not the sort of thing to rebuild for tidiness.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def api(base: str, path: str, payload=None, method="GET", timeout=180):
    url = base.rstrip("/") + path
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(  # noqa: S310 - our own app on the cluster
        url, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as fh:  # noqa: S310
            body = fh.read().decode()
    except urllib.error.HTTPError as exc:
        return {"_error": f"{exc.code}: {exc.read().decode()[:300]}"}
    return json.loads(body) if body.strip() else {}


def count_of(d: dict) -> int:
    """Ingested item count. The registry reports it per file type, not as a total."""
    fc = d.get("file_type_counts") or {}
    if fc:
        return sum(int(v) for v in fc.values())
    return int(d.get("num_items") or d.get("count") or 0)


def datasets(base: str) -> dict[str, dict]:
    got = api(base, "/api/datasets/registry")
    rows = got.get("datasets", got if isinstance(got, list) else [])
    return {d.get("name", ""): d for d in rows if isinstance(d, dict)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://rack7n06:11850")
    ap.add_argument("--slates", default="/expscratch/sgreenberg/vlm-3720/slates")
    ap.add_argument("--prefix", default="vgscale")
    ap.add_argument("--wait", type=int, default=900)
    args = ap.parse_args()

    sys.path.insert(0, "scripts/experiments/pile")
    import pile_config as pc  # noqa: PLC0415

    manifest = json.loads((Path(args.slates) / "slates.json").read_text())
    have = datasets(args.api)
    print(f"{len(have)} datasets already registered")

    made, skipped, failed = [], [], []
    for cls in pc.SCALE_CLASSES:
        info = manifest.get(cls, {})
        folder = Path(args.slates) / cls.replace(" ", "_") / "images"
        n_files = len(list(folder.glob("*.jpg"))) if folder.exists() else 0
        name = f"{args.prefix} {cls} candidates"
        if n_files == 0:
            print(f"  {name:<42} SKIP (no images)")
            skipped.append(cls)
            continue
        if name in have:
            print(f"  {name:<42} exists")
            skipped.append(cls)
            continue

        resp = api(
            args.api,
            "/api/dataset/import/server_folder",
            {
                "path": str(folder),
                "media_type": "image",
                "recursive": "false",
                "dig_archives": "false",
                "dataset_name": name,
            },
            method="POST",
        )
        if "_error" in resp:
            print(f"  {name:<42} IMPORT FAILED {resp['_error'][:80]}")
            failed.append(cls)
            continue

        # queued, not done: poll until it appears in the registry with its files
        t0 = time.time()
        landed = None
        while time.time() - t0 < args.wait:
            time.sleep(3)
            d = datasets(args.api).get(name)
            if d and count_of(d) >= n_files * 0.95:
                landed = d
                break
        if landed:
            got = count_of(landed)
            print(f"  {name:<42} OK  {got}/{n_files} items, cut {info.get('cut')}")
            made.append(cls)
        else:
            print(f"  {name:<42} TIMED OUT after {args.wait}s (may still be ingesting)")
            failed.append(cls)

    print(f"\nimported {len(made)}, skipped {len(skipped)}, failed {len(failed)}")
    if failed:
        print("failed:", ", ".join(failed))
    dets = api(args.api, "/api/detectors/registry").get("detectors", [])
    print(f"\n{len(dets)} detectors in the registry (reused, none created)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
