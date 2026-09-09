#!/usr/bin/env python3
"""Fresh, empty detectors for the slate datasets (#3720).

The slates deliberately exclude every image the reviewer already answered, and
the existing detectors hold exactly those answers -- so pairing an old detector
with its new slate gives a panel in which *none* of its 936 labels resolve, and
every one renders as a dead thumbnail. The two decisions were individually right
and jointly wrong.

The old detectors are left alone: they are the reviewer's work, there is no
rename endpoint, and their labelsets are already committed under `human_record/`.
The name still carries the class's rule, because that is the only string the
reviewer reads while voting (#3612).
"""

from __future__ import annotations
import json
import sys
import urllib.error
import urllib.request

BASE = "http://rack5n03:11850"
SUFFIX = " [slate]"


def api(path, payload=None, method="GET"):
    req = urllib.request.Request(  # noqa: S310 - our own app
        BASE + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as fh:  # noqa: S310
            body = fh.read().decode()
    except urllib.error.HTTPError as exc:
        return {"_error": f"{exc.code}: {exc.read().decode()[:200]}"}
    return json.loads(body) if body.strip() else {}


sys.path.insert(0, "scripts/experiments/pile")
import pile_config as pc  # noqa: E402

existing = {d["name"]: d for d in api("/api/detectors/registry").get("detectors", [])}
made, skipped, failed = [], [], []
for cls in pc.SCALE_CLASSES:
    rule = pc.SCALE_CLASS_RULES.get(cls)
    name = (getattr(rule, "name", "") or cls) + SUFFIX
    if name in existing:
        print(f"  {name[:52]:<54} exists")
        skipped.append(cls)
        continue
    resp = api(
        "/api/detectors/registry",
        {
            "name": name,
            "media_type": "image",
            "embedder_type": "semantic",
            "text_query": cls,
            "examples": [{"type": "text", "value": cls}],
        },
        method="POST",
    )
    if "_error" in resp or not resp.get("ok"):
        print(f"  {name[:52]:<54} FAILED {str(resp)[:70]}")
        failed.append(cls)
        continue
    d = resp["detector"]
    print(f"  {name[:52]:<54} OK  {d['id'][:12]}  training={d.get('num_training')}")
    made.append(cls)

print(f"\ncreated {len(made)}, existed {len(skipped)}, failed {len(failed)}")
now = api("/api/detectors/registry").get("detectors", [])
print(f"{len(now)} detectors in the registry")
