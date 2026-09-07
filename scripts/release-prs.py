#!/usr/bin/env python3
"""List the PRs merged into ``dev`` across a release window -- both merge shapes.

``docs/RELEASE.md`` steps 4 and 6 both need the same thing: the set of pull
requests merged into ``dev`` since the last release.  Both used to say
"inspect the merge commits in that range", and for most of this repo's history
that was the whole truth, because every PR landed as a two-parent
``Merge pull request #N from ...`` commit.

**Squash-merged PRs are not merge commits.**  GitHub's "Squash and merge"
flattens the branch into a single one-parent commit whose subject ends in
``(#N)``, so ``git log --merges`` does not return it and a runbook enumerating
merges walks straight past it.  Four PRs landed that way on 2026-09-06
(#3671, #3672, #3682, #3685); at the time this script was written the pending
release window held 41 findable merges and those 4 invisible squashes.  A
release run that missed them would have left their issues open forever -- the
sweep's window is ``origin/main..origin/dev`` and nothing re-examines an
already-merged PR -- and dropped 4 rows from the punch-card data file.  That is
the same silent-orphan failure ``docs/RELEASE.md`` step 6 spends three
paragraphs guarding against, arriving through the enumeration instead of
through a keyword.

So the enumeration lives here, once, rather than as prose in two steps.  It
walks ``--first-parent`` -- which is what actually means "landed on ``dev``",
independent of how it landed -- and reads the PR number off each subject:

* ``Merge pull request #N from <branch>``  ->  N.  Checked first, so a branch
  name that happens to end in ``(#123)`` cannot outvote it.
* anything else ending in ``(#N)``  ->  N, the suffix GitHub appends when it
  squashes.  **The trailing one, not the first.**  Three of the four squashes
  above read like ``... measured (#3673) (#3685)``, where the first number is
  the *issue* the PR closes and only the last is the PR.  Taking the first
  match would have quietly enumerated issue numbers as PRs.
* neither  ->  no PR.  Release housekeeping is pushed straight to ``dev``
  (punch-card refreshes, plan-pointer prunes), so this bucket is normal and
  non-empty.  It prints to stderr rather than being dropped, because the other
  thing that lands in it is a PR whose squash title was hand-edited to drop
  GitHub's suffix -- indistinguishable from a direct push from here, and worth
  a human glance.

A number read off a subject is a claim, not a verified fact: both steps go on
to read each PR from GitHub, so a mis-parse surfaces there as a lookup that
does not match rather than as a silent omission.

``CLAUDE.md``'s "Merging a PR" rule now forbids squashing outright, so in
principle this only ever meets merge commits.  It reads both shapes anyway:
the four squashes above are already on ``dev`` and no convention reaches
backwards, and a rule that has been broken once is a thing to survive rather
than to trust.

Usage::

    python scripts/release-prs.py                  # origin/main..origin/dev
    python scripts/release-prs.py --range A..B
    python scripts/release-prs.py --json

stdout is ``<pr>|<sha>|<subject>`` lines (newest first; ``--reverse`` flips
it), so ``| cut -d'|' -f1`` gives the numbers.  Counts and the no-PR bucket go
to stderr, keeping stdout pipeable.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys

DEFAULT_RANGE = "origin/main..origin/dev"

#: A GitHub merge commit.  Anchored at the start and checked first.
MERGE_SUBJECT_RE = re.compile(r"^Merge pull request #(\d+) from \S")

#: GitHub's squash suffix.  Anchored at the *end*: the subject may well carry
#: an earlier ``(#N)`` naming the issue, and that one is not the PR.
SQUASH_SUBJECT_RE = re.compile(r"\(#(\d+)\)\s*$")


def pr_number(subject: str) -> int | None:
    """The PR number a first-parent commit subject names, or None.

    Pure function of the subject line -- the whole parsing rule lives here so
    the tests can pin it without a git repo.
    """
    merge = MERGE_SUBJECT_RE.match(subject)
    if merge is not None:
        return int(merge.group(1))
    squash = SQUASH_SUBJECT_RE.search(subject)
    if squash is not None:
        return int(squash.group(1))
    return None


def _git(*args: str) -> str:
    proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
        ["git", *args],  # noqa: S607 - git resolved from PATH
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout


def landed(rev_range: str) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    """Split a window's first-parent commits into (PRs, no-PR commits)."""
    out = _git("log", rev_range, "--first-parent", "--format=%H%x09%s")
    prs: list[dict[str, object]] = []
    other: list[dict[str, str]] = []
    for line in out.splitlines():
        if not line.strip():
            continue
        sha, _, subject = line.partition("\t")
        number = pr_number(subject)
        if number is None:
            other.append({"sha": sha, "subject": subject})
        else:
            prs.append({"pr": number, "sha": sha, "subject": subject})
    return prs, other


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--range", dest="rev_range", default=DEFAULT_RANGE)
    parser.add_argument("--json", action="store_true", help="emit a JSON object on stdout")
    parser.add_argument("--reverse", action="store_true", help="oldest first")
    args = parser.parse_args()

    try:
        prs, other = landed(args.rev_range)
    except subprocess.CalledProcessError as exc:
        print(f"git failed on {args.rev_range!r}: {exc.stderr.strip()}", file=sys.stderr)
        return 1

    if args.reverse:
        prs = list(reversed(prs))
        other = list(reversed(other))

    if args.json:
        json.dump({"range": args.rev_range, "prs": prs, "no_pr": other}, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        for row in prs:
            print(f"{row['pr']}|{row['sha']}|{row['subject']}")

    squashed = sum(1 for row in prs if not str(row["subject"]).startswith("Merge pull request #"))
    print(
        f"{len(prs)} PR(s) in {args.rev_range}: {len(prs) - squashed} merge commit(s), {squashed} squashed.",
        file=sys.stderr,
    )
    if other:
        print(
            f"{len(other)} first-parent commit(s) name no PR (direct pushes to dev, "
            f"or a squash title with GitHub's suffix edited out -- check these):",
            file=sys.stderr,
        )
        for row in other:
            print(f"  {row['sha'][:8]} {row['subject']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
