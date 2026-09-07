"""``--provenance`` / ``--backfill-provenance``: which device built each cell."""

from __future__ import annotations

import json
import time
from collections import defaultdict

import pile_config as pc

from pilebuild.env import log
from pilebuild.provenance import cell_fingerprint


def _sacct_build_nodes() -> dict[str, str]:
    """dataset -> node, recovered from SLURM's accounting of the ``pile-*`` jobs.

    Cells built before the sidecar existed are not anonymous after all: the build
    ran as ``pile-<dataset>`` and ``sacct`` still knows which node took it. This
    is recorded as ``hostname_recovered``, never as ``hostname`` -- it is an
    inference from a job name, and one ambiguous dataset (two completed jobs) is
    left out rather than guessed. It matters because the node determines the CPU,
    and the CPU determines how the 384px resize rounds (#3160).
    """
    import subprocess  # noqa: PLC0415, S404 -- fixed argv, no shell

    try:
        out = subprocess.run(  # noqa: S603
            ["sacct", "-X", "-S", "2026-01-01", "-n", "-P", "--format=JobName,NodeList,State"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    seen: dict[str, set[str]] = defaultdict(set)
    for line in out.stdout.splitlines():
        parts = line.split("|")
        if len(parts) < 3 or not parts[0].startswith("pile-") or parts[2] != "COMPLETED":
            continue
        seen[parts[0][len("pile-") :]].add(parts[1])
    return {ds: next(iter(nodes)) for ds, nodes in seen.items() if len(nodes) == 1}


def provenance_report(backfill: bool = False) -> int:
    """Show which device built each cell -- and, with ``--backfill-provenance``,
    stamp what is still knowable for the cells built before this existed.

    A backfilled sidecar deliberately records ``gpu_name: null``: the node a 2026
    job ran on is not recoverable from the pickle, and writing a guess would be
    worse than writing nothing. What it *can* record is the fingerprint, and that
    is the half that matters for a rebuild -- it turns "did the rebuild reproduce
    the cell?" from an unanswerable question into a hash comparison.
    """
    rows, missing, devices = [], [], defaultdict(list)
    checkouts: defaultdict[str, list[str]] = defaultdict(list)
    recovered = _sacct_build_nodes() if backfill else {}
    for ds, emb in pc.cells():
        cell = pc.cell_path(ds, emb)
        if not cell.exists():
            continue
        path = pc.provenance_path(ds, emb)
        if not path.exists():
            if backfill:
                stat = cell.stat()
                record = {
                    "dataset": ds,
                    "embedder": emb,
                    "cell": cell.name,
                    "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(stat.st_mtime)),
                    "backfilled": True,
                    "device": {
                        "gpu_name": None,
                        "hostname_recovered": recovered.get(ds),
                        "recovered_from": "sacct pile-<dataset> job" if recovered.get(ds) else None,
                        "note": "unknown: cell predates per-cell provenance (#3160)",
                    },
                    "cell_summary": {"megabytes": round(stat.st_size / 1e6, 1)},
                    "fingerprint": cell_fingerprint(ds, emb),
                }
                path.write_text(json.dumps(record, indent=2) + "\n")
                log(f"backfilled {path.name} ({record['fingerprint']['vectors_sha256'][:12]})")
            else:
                missing.append(f"{ds} x {emb}")
                continue
        rec = json.loads(path.read_text())
        dev = rec.get("device", {})
        # `code` since #3693; older sidecars kept the commit under `device` and
        # recorded no checkout at all, so a null repo here means "unrecorded",
        # not "same tree as everything else".
        code = rec.get("code", {})
        if backfill and not dev.get("hostname") and not dev.get("hostname_recovered") and recovered.get(ds):
            dev["hostname_recovered"] = recovered[ds]
            dev["recovered_from"] = "sacct pile-<dataset> job"
            rec["device"] = dev
            path.write_text(json.dumps(rec, indent=2) + "\n")
            log(f"recovered build node for {ds} x {emb}: {recovered[ds]}")
        rows.append(
            (
                ds,
                emb,
                dev.get("gpu_name") or "unknown",
                dev.get("hostname") or (f"{dev['hostname_recovered']}?" if dev.get("hostname_recovered") else "-"),
                str(dev.get("cpu_capability") or "-"),
                (code.get("commit") or dev.get("commit") or "-")[:9],
                rec.get("fingerprint", {}).get("vectors_sha256", "")[:12],
            )
        )
        devices[(dev.get("gpu_name") or "unknown", dev.get("cpu_capability"))].append(f"{ds}x{emb}")
        checkouts[code.get("repo") or "unrecorded"].append(f"{ds}x{emb}")

    log(f"{'dataset':<18} {'embedder':<14} {'device':<26} {'node':<10} {'dispatch':<9} {'commit':<10} vectors")
    for row in sorted(rows):
        log("{:<18} {:<14} {:<26} {:<10} {:<9} {:<10} {}".format(*row))
    if missing:
        log(f"\n{len(missing)} cell(s) with NO provenance (run --backfill-provenance): {', '.join(missing)}")
    if len(devices) > 1:
        log(f"\nthis pile MIXES {len(devices)} build environments. The measured cost of mixing")
        log("hosts is 1.5e-04 median 1-cos on siglip2_l when CPU dispatch is unpinned (#3160):")
        for (name, cap), cells in sorted(devices.items(), key=lambda kv: str(kv[0])):
            log(f"  {str(name):<26} dispatch={cap or 'unrecorded':<10} {len(cells)} cell(s)")
    # The same warning one axis over. A pile built from two checkouts is what
    # #3693 was: a launcher whose fixed default pointed at a tree 1,420 commits
    # behind dev, while the tree you were reading built everything else. Nothing
    # said so, because nothing recorded the path -- and the cells look identical
    # from outside. Now they do not.
    if len(checkouts) > 1:
        log(f"\nthis pile was built from {len(checkouts)} DIFFERENT checkouts:")
        for repo, cells in sorted(checkouts.items()):
            log(f"  {repo:<52} {len(cells)} cell(s)")
        log("cells built from different trees ran different code; compare their commits")
        log("before reading them as one pile (#3693).")
    return 0
