#!/usr/bin/env python3
"""The durable copy of every human answer, and a check that it is current (#3729).

A pile cell is purgeable on purpose: `pile_config`'s module docstring requires
every cell to be rebuildable from sources that are **not** on scratch, so a
purge costs GPU hours and nothing else. That rule was written for cells and
never extended to the one input it cannot cover. A human verdict is not
rebuildable from anything -- it is a person having looked at a picture -- and
the verdicts, the labelsets behind them, the adjudications and the roster
recording which images a review was carried out against have been living on the
same purgeable mount as the cells.

What was actually on disk was not "no backup" but something harder to notice:
**ad-hoc snapshots in two unrelated directories, with dated filenames and no
manifest**, one of them load-bearing as an absolute path in a script default,
beside a September adjudication with no copy at all. Nothing distinguished a
file no rebuild can regenerate from a file every rebuild rewrites, so nobody
could tell which of them mattered. :data:`pile_config.HUMAN_RECORD` is that
distinction written down, and it is the actual fix here; the copying is
bookkeeping.

* ``export`` writes every declared artifact into ``data/vg_scale/``,
  canonicalised so unchanged content produces an unchanged file and a diff is a
  diff of *answers* rather than of key order.
* ``check`` compares the working copies against the committed ones and reports
  how many rows moved. It is what stops the two drifting quietly for a month.
* ``restore`` writes the committed copies back out, which is what recovering
  from a purge looks like.

**The repository, not another directory on the cluster.** `/exp` is not
documented as backed up anywhere in this repo, and treating it as durable is an
assumption nobody has checked. The whole inventory is a few MB, git compresses
it well, and a commit gives the answers what a copy cannot: review, attribution
and a date.

**A `human` divergence fails; the others are notes.** `corrections.json` and the
rosters are rewritten by every build, so failing on them would fire constantly
and train everyone to ignore the check. Losing them costs a rebuild. Losing a
verdict costs a person's afternoon and cannot be bought back at any price.

Usage::

    python verdict_store.py check              # do the working copies match the repo?
    python verdict_store.py export             # update the repo from the working copies
    python verdict_store.py restore --force    # write the repo's copies back out
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pile_config as pc

#: Where the canonical copies live: in the repo, so they are reviewed, diffable
#: and attributable to a commit -- and beside the code that reads them, because
#: the repo's top-level ``data/`` is the app's runtime directory and gitignored,
#: which is exactly the kind of quiet non-storage this exists to end.
STORE = Path(__file__).resolve().parent / "human_record"
MANIFEST = STORE / "MANIFEST.json"


def log(msg: str) -> None:
    print(f"[verdicts] {msg}", flush=True)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def stored_name(art: pc.HumanArtifact, rel: Path) -> str:
    """The flat filename one artifact is committed under: ``ROOT__path__to__file``.

    Both halves are load-bearing, and both were learned by the first export
    refusing to run. The *relative path* rather than the basename, because
    twelve slate directories each hold a ``manifest.csv``. The *root token*
    because the same basename exists under two roots -- the same snapshot sits
    on scratch and on `/exp`, and a store keyed on the path alone would keep one
    of the two and silently drop the other, which is this script's own failure
    mode turned inward.
    """
    return f"{art.root_token()}__{str(rel).replace('/', '__')}"


def _row_key(row: dict) -> tuple:
    """A verdict-shaped row's identity: the pair it is a judgement about.

    Falls back to the whole row so a file of some other shape still sorts
    deterministically instead of raising -- canonicalising has to work on every
    file in the inventory, including ones added later that this has never seen.
    """
    if "image_id" in row:
        return (0, int(row["image_id"]), str(row.get("class", "")))
    return (1, json.dumps(row, sort_keys=True))


def canonicalise(text: str, suffix: str = ".json") -> str:
    """Stable bytes for the same content: sorted keys, sorted rows, one format.

    Without it a re-export that reorders a dict reports a divergence that is not
    one, and a check that cries wolf is a check nobody runs. Non-JSON artifacts
    (the slate manifests are CSV) pass through untouched: they are already
    line-stable, and rewriting them would be a change this cannot verify.
    """
    if suffix != ".json":
        return text
    data = json.loads(text)
    if isinstance(data, list) and all(isinstance(r, dict) for r in data):
        data = sorted(data, key=_row_key)
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def row_delta(committed: str, working: str) -> tuple[int, int, int] | None:
    """``(added, removed, changed)`` rows, or ``None`` when not row-shaped.

    A hash answers "are these the same file". This answers "how many judgements
    moved", which is the question a person actually has when the check fires.
    """
    try:
        a, b = json.loads(committed), json.loads(working)
    except json.JSONDecodeError:
        return None
    if not (isinstance(a, list) and isinstance(b, list)):
        return None
    if not all(isinstance(r, dict) and "image_id" in r for r in [*a, *b]):
        return None
    ka = {_row_key(r): json.dumps(r, sort_keys=True) for r in a}
    kb = {_row_key(r): json.dumps(r, sort_keys=True) for r in b}
    added = len(kb.keys() - ka.keys())
    removed = len(ka.keys() - kb.keys())
    changed = sum(1 for k in ka.keys() & kb.keys() if ka[k] != kb[k])
    return added, removed, changed


def working_copies() -> list[tuple[pc.HumanArtifact, Path, Path]]:
    """``(artifact, path relative to its root, absolute path)`` for what exists here."""
    found = []
    for art in pc.HUMAN_RECORD:
        for path in sorted(art.resolve()):
            found.append((art, path.relative_to(art.root()), path))
    return found


def do_export(dry_run: bool) -> int:
    STORE.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict] = {}
    written = 0
    seen_empty = []
    for art in pc.HUMAN_RECORD:
        if not art.resolve():
            seen_empty.append(art.source)
    for art, rel, path in working_copies():
        name = stored_name(art, rel)
        if name in manifest:
            raise SystemExit(f"two artifacts want to be stored as {name!r}; give one a distinct source")
        text = canonicalise(path.read_text(), path.suffix)
        dest = STORE / name
        if not dry_run and (not dest.exists() or dest.read_text() != text):
            dest.write_text(text)
            written += 1
        manifest[name] = {
            "tier": art.tier,
            "source": art.source,
            "rel": str(rel),
            "why": art.why,
            "sha256": sha256(text),
            "bytes": len(text.encode()),
        }
    for source in seen_empty:
        log(f"no working copy here, nothing exported: {source}")
    if not dry_run:
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    by_tier = {
        t: sum(1 for r in manifest.values() if r["tier"] == t) for t in sorted({r["tier"] for r in manifest.values()})
    }
    log(f"{len(manifest)} artifacts in the store ({by_tier}), {written} written")
    return 0


def do_check(strict: bool) -> int:
    """Compare working copies against the committed ones. Exit 1 on divergence.

    Silent success when this machine has no working copies at all: it runs from
    the test suite on boxes that have never seen a pile, and a check that fails
    there is a check people learn to skip.
    """
    if not MANIFEST.exists():
        log(f"no store yet at {MANIFEST}; run `verdict_store.py export`")
        return 1
    manifest = json.loads(MANIFEST.read_text())
    here = working_copies()
    if not here:
        log("no working copies on this machine; nothing to compare")
        return 0

    problems: list[tuple[bool, str]] = []  # (fatal, message)
    for art, rel, path in here:
        name = stored_name(art, rel)
        working = canonicalise(path.read_text(), path.suffix)
        row = manifest.get(name)
        if row is None:
            problems.append((True, f"{name} [{art.tier}]: on disk but NOT in the store -- export it"))
            continue
        if row["sha256"] == sha256(working):
            continue
        delta = row_delta((STORE / name).read_text(), working)
        detail = (
            f"{delta[0]} added, {delta[1]} removed, {delta[2]} changed"
            if delta
            else f"{row['bytes']} bytes committed, {len(working.encode())} on disk"
        )
        problems.append((art.tier == "human" or strict, f"{name} [{art.tier}]: differs -- {detail}"))

    # In the store and gone from disk is the store doing its job, not a fault:
    # say so, because it is also what a purge looks like from here.
    on_disk = {stored_name(a, rel) for a, rel, _p in here}
    for name in sorted(set(manifest) - on_disk):
        log(f"note {name}: in the store, no working copy here -- `restore` brings it back")

    for fatal, msg in problems:
        log(("FAIL " if fatal else "note ") + msg)
    if any(f for f, _ in problems):
        log("Export before the working copy is lost: `python verdict_store.py export`")
        return 1
    log(f"store is current ({len(manifest)} artifacts)")
    return 0


def do_restore(force: bool) -> int:
    manifest = json.loads(MANIFEST.read_text())
    by_source = {art.source: art for art in pc.HUMAN_RECORD}
    restored = skipped = 0
    for name, row in sorted(manifest.items()):
        art = by_source.get(row["source"])
        if art is None:
            log(f"note {name}: its source {row['source']!r} is no longer declared; not restored")
            continue
        dest = art.root() / row["rel"]
        if dest.exists() and not force:
            skipped += 1
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text((STORE / name).read_text())
        restored += 1
        log(f"restored {dest}")
    log(f"{restored} restored, {skipped} already present (use --force to overwrite)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=("export", "check", "restore"))
    ap.add_argument("--dry-run", action="store_true", help="export: report, write nothing")
    ap.add_argument("--force", action="store_true", help="restore: overwrite working copies that exist")
    ap.add_argument("--strict", action="store_true", help="check: fail on a `derived` divergence too")
    args = ap.parse_args()
    if args.action == "export":
        return do_export(args.dry_run)
    if args.action == "check":
        return do_check(args.strict)
    return do_restore(args.force)


if __name__ == "__main__":
    sys.exit(main())
