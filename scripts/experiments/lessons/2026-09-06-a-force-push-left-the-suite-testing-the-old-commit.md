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
so the script names the direction and stops. A ref with no `origin/` counterpart
(a bare SHA, a local-only branch) is not compared, and every run now prints a
`=== ref` line beside `=== HEAD` saying which of the three it was.

Both refusal paths were verified against the real script before this was
written — a branch deliberately left behind, then one deliberately left ahead —
each exiting 2 with SLURM reporting `FAILED`.

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
