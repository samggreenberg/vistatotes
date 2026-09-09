#!/usr/bin/env python3
"""Import the below-cut samples and give each its own detector (#3768)."""

from __future__ import annotations
import json
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

OUT = Path("/expscratch/sgreenberg/vlm-3720/belowcut")
# The app moves node whenever its allocation is renewed, so the node is
# discovered rather than hardcoded -- a stale host is the failure that reads as
# "the detector is broken".
_SQUEUE = shutil.which("squeue") or "/usr/bin/squeue"
_ARGS = [_SQUEUE, "-u", "sgreenberg", "-h", "-n", "vtsearch", "-o", "%N"]
node = (
    subprocess.run(  # noqa: S603 - fixed argv, no shell, no user input
        _ARGS, capture_output=True, text=True, check=False
    )
    .stdout.strip()
    .split()[0]
)
BASE = f"http://{node}:11850"
print(f"app node: {node}")


def api(p, payload=None, method="GET"):
    req = urllib.request.Request(  # noqa: S310 - our own app
        BASE + p,
        data=json.dumps(payload).encode() if payload is not None else None,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as fh:  # noqa: S310
            return fh.getcode(), json.loads(fh.read().decode() or "{}")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()[:200]


def count_of(d):
    fc = d.get("file_type_counts") or {}
    return sum(int(v) for v in fc.values()) if fc else int(d.get("num_items") or 0)


import sys

sys.path.insert(0, "scripts/experiments/pile")
import pile_config as pc  # noqa: E402

summary = json.loads((OUT / "belowcut.json").read_text())
have = {d["name"] for d in api("/api/datasets/registry")[1]["datasets"]}
dets = {d["name"] for d in api("/api/detectors/registry")[1]["detectors"]}

for cls in summary:
    folder = OUT / cls.replace(" ", "_") / "images"
    n = len(list(folder.glob("*.jpg")))
    dsname = f"vgscale {cls} below-cut"
    if dsname not in have:
        api(
            "/api/dataset/import/server_folder",
            {
                "path": str(folder),
                "media_type": "image",
                "recursive": "false",
                "dig_archives": "false",
                "dataset_name": dsname,
            },
            method="POST",
        )
        t0, landed = time.time(), False
        while time.time() - t0 < 600:
            time.sleep(3)
            d = {x["name"]: x for x in api("/api/datasets/registry")[1]["datasets"]}.get(dsname)
            if d and count_of(d) >= n * 0.95:
                landed = True
                break
        print(f"  {dsname:<38} {'OK' if landed else 'TIMED OUT'} {n} items")
    else:
        print(f"  {dsname:<38} exists")

    rule = pc.SCALE_CLASS_RULES.get(cls)
    base = getattr(rule, "name", "") or cls
    dname = f"{base} [below-cut: any in image]"
    if dname in dets:
        print("    detector exists")
        continue
    code, resp = api(
        "/api/detectors/registry",
        {
            "name": dname,
            "media_type": "image",
            "embedder_type": "semantic",
            "text_query": cls,
            "examples": [{"type": "text", "value": cls}],
        },
        method="POST",
    )
    print(f"    detector {'OK' if code == 201 else f'FAILED {code}'}  {dname[:56]}")

print(
    f"\n{len(api('/api/datasets/registry')[1]['datasets'])} datasets, "
    f"{len(api('/api/detectors/registry')[1]['detectors'])} detectors"
)
