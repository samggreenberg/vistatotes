# 2026-09-06 — a force-push left the suite silently testing the old commit (#3666)

**Cost:** ~25m and two wasted suite jobs.

**What broke:** the branch was squashed and force-pushed before the suite ran.
`suite.sbatch` takes a *ref* and runs `git checkout --detach "$REF"` on a local
branch name, and its `git fetch origin` is not forced — so a non-fast-forward
update is refused, the grid's local branch stays where it was, and the job
checks out the **previous** commit. It then failed the docs gate for a missing
`docs/experiments/README.md` row that the new commit adds, which reads exactly
like a real finding about the branch under test. The `=== HEAD <sha> <subject>`
line the job prints is what gave it away, and it is the only thing that did.

Force-fetching the ref by hand was not enough either: the local branch is
whatever the *working* worktree's HEAD is, so the fix was
`git reset --hard origin/<branch>` in the worktree that owns the branch.

**Prevented?** **Yes**, by a guard in `suite.sbatch` (#3677). After its
`checkout --detach "$REF"` the script compares `HEAD` against `origin/$REF` and
refuses the run when they differ, naming which way:

```
FATAL: checked out 22fe41c12… for 'claude/tmp-3677-guard', but origin/… is f206d7048…
  The local branch is BEHIND origin -- a force-push, or a fetch that could
  not fast-forward it. …
    git -C <that worktree> fetch -f origin && git reset --hard origin/<branch>
```

It **refuses rather than auto-corrects**, and that is the whole design decision.
Checking out `origin/$REF` would fix this incident and cause the opposite one:
whenever the grid clone is legitimately *ahead* of origin, it would silently
test older code and discard the commit somebody made here — which is
[the #3292 failure](2026-09-05-a-commit-made-during-the-suite-job-was-orphaned.md),
already in this log. Both directions are mistakes and only a human knows which,
so the script names the direction and stops. Every run now prints a `=== ref`
line beside `=== HEAD` saying which state it was in.

**Testing the guard found a second defect in the same line.** `git checkout
--detach <name>` *refuses* a name that exists only on the remote: git's DWIM
wants to create a tracking branch, `--detach` forbids it, and the job dies on
`'--detach' cannot be used with '-b/-B/--orphan'` — a message about nothing to
do with the situation. That is every branch pushed from the laptop and not yet
checked out on the grid, and it had been hidden by the habit of running
`git worktree add <path> <branch>` first, which creates the local branch as a
side effect. The script now falls back to `origin/$REF` when **no** local ref
exists — safe there and nowhere else, because with no local ref there is no
local commit to discard.

All four states were exercised against the live script before this was written:
behind and ahead each exit 2 with SLURM reporting `FAILED`; in-sync and
remote-only run the suite and say which they were.

Two things still follow from it, because the guard covers one script and not the
habit:

- **Read the job's `=== HEAD` line before reading its verdict.** A suite result
  is about a commit, not about a branch name, and this is the same failure
  [`which-branch-did-you-measure`](2026-09-05-a-commit-made-during-the-suite-job-was-orphaned.md)
  records from the other direction — there a commit made during the job was
  orphaned, here a commit made before it was ignored.
- **Prefer adding a commit to rewriting one** once anything on the grid has
  fetched the branch. A squash before the first suite run costs nothing; a
  squash after it costs a job and can be read as a test failure.

The guard lives in `/exp/sgreenberg/suite.sbatch`, **outside this repository**, so
nothing here reviews it, tests it, or notices if it is edited away — the previous
version is kept beside it as `suite.sbatch.bak.20260906`. That gap is filed as
**#3694**.
