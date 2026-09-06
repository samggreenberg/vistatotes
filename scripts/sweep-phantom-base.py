"""Re-measure both stale-tree signals against every merge on `dev`.

`scripts/check-phantom-base.py` makes empirical claims — 4/4 clobbers caught,
zero false positives for the reverted-window signal, two for the deletion one
— and those claims are only worth what they can be re-derived from. This
replays every two-parent merge on `origin/dev` through the gate's own
functions: each merge's second parent is the branch, its first parent is the
base, and `find_phantom_base` / `find_reverted_window` are asked what they
would have said the moment that branch was merged.

It drives the shipped code rather than a copy of it (that is what the `tip`
argument on both functions is for), so a change to the signal shows up here
instead of quietly diverging from a stale reimplementation.

**A squash-merged PR cannot be swept.** Both signals need the branch and the
base it landed on, and take them from a merge's two parents; a squash has one
parent and no surviving branch tip, so there is nothing to replay. That is a
limit of this sweep, not of the gate -- `check-phantom-base.py` compares a
working tree against `origin/dev` and never cared how anything merged. The
count of unsweepable commits is printed alongside the swept total so the
coverage this sweep claims stays honest as squashes accumulate.

Takes about a minute for ~300 merges. Run from the repo root:

    python scripts/sweep-phantom-base.py

The four known clobbers are listed in `CLOBBERS`; every other hit is a false
positive, and the output labels them that way.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
import time
from pathlib import Path

# The PRs that silently reverted the work merged before them (#3205, #3206).
CLOBBERS = {"2741", "2793", "2821", "3184"}

BASE_REF = "origin/dev"


def _load_gate():
    """Import the gate as a module, hyphenated filename and all."""
    path = Path(__file__).with_name("check-phantom-base.py")
    spec = importlib.util.spec_from_file_location("check_phantom_base", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise SystemExit(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git(*args: str) -> str:
    proc = subprocess.run(  # noqa: S603 - fixed argv, no shell, no user input
        ["git", *args],  # noqa: S607 - git resolved from PATH
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout


def _pr_number(subject: str) -> str:
    found = re.search(r"#(\d+)", subject)
    return found.group(1) if found else "?"


def main() -> int:
    gate = _load_gate()
    merges = _git("log", "--first-parent", "--merges", "--format=%H", BASE_REF).split()
    landed = _git("log", "--first-parent", "--format=%H", BASE_REF).split()
    unsweepable = len(landed) - len(merges)
    if not merges:
        print(f"No merges on {BASE_REF}; nothing to sweep.")
        return 1

    deletions: list[str] = []
    reverts: list[str] = []
    started = time.time()
    swept = 0

    for merge in merges:
        parents = _git("rev-parse", f"{merge}^@").split()
        if len(parents) != 2:
            continue
        swept += 1
        base, tip = parents
        pr = _pr_number(_git("log", "-1", "--format=%s", merge))
        verdict = "<< CLOBBER" if pr in CLOBBERS else "FALSE POSITIVE"

        hit = gate.find_phantom_base(base, tip)
        if hit is not None:
            deletions.append(f"  #{pr:<6} {len(hit[1]):3d} path(s) deleted            {verdict}")

        hit = gate.find_reverted_window(base, tip)
        if hit is not None:
            _, _, reverted, span = hit
            reverts.append(f"  #{pr:<6} {len(reverted):3d} path(s) over {span} commit(s)   {verdict}")

    elapsed = time.time() - started
    print(f"{swept} two-parent merges on {BASE_REF} in {elapsed:.0f}s")
    print(
        f"{unsweepable} first-parent commit(s) not sweepable "
        f"(squash merges and direct pushes have no branch tip to replay)\n"
    )
    for name, hits in (("deletion", deletions), ("reverted-window", reverts)):
        caught = sorted(c for c in CLOBBERS if any(f"#{c} " in h for h in hits))
        false_positives = sum(1 for h in hits if "FALSE POSITIVE" in h)
        print(f"{name} signal: {len(hits)} hit(s), {len(caught)}/4 clobbers, {false_positives} FP")
        for line in hits:
            print(line)
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
