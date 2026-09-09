# 2026-09-05 — a commit made while the suite job held the worktree detached was orphaned, and the push said nothing (#3635)

**What broke.** `/exp/sgreenberg/suite.sbatch <worktree> <ref>` runs the test
suite by **checking the worktree out to the ref it was told to test** and
restoring the previous branch on exit. That is deliberate and documented in the
script. What it also means is that for the ~10 minutes the suite runs, the
worktree is on a **detached HEAD**, and it is the same worktree the session is
still working in.

A commit made in that window lands on the detached HEAD, not on the branch. The
suite's cleanup then checks the branch back out and the commit becomes
unreferenced. The reflog is unambiguous once you look:

```
f7415103c checkout: moving from 7c66fa0ef... to claude/stop-sign-3635   <- cleanup
7c66fa0ef commit: Record the #3635 ruling...                            <- my commit, orphaned
f7415103c checkout: moving from claude/stop-sign-3635 to f7415103c      <- suite detaches
f7415103c commit: Index the stop-sign pool study...
```

**Why it was invisible.** Both halves reported success:

* `git commit` printed `[claude/stop-sign-3635 7c66fa0ef] Record the #3635
  ruling` — *with the branch name*, because the branch still pointed there at
  that instant — and exited `0`.
* `git push -q origin claude/stop-sign-3635` also exited `0`. It pushed the
  **branch ref**, which by then had been restored to `f7415103c` and was already
  up to date with the remote. Pushing nothing is not an error.

So the session had a green commit, a green push, and a remote that did not carry
the change. It surfaced only because a later `git log --oneline origin/dev..HEAD`
showed two commits where three were expected — and it would just as easily not
have been checked.

**Cost.** Small here: one prose commit, recovered with
`git cherry-pick 7c66fa0ef` since the object was still in the store and the
reflog named it. The cost is bounded by the reflog's expiry, not by anything
that would have raised an alarm. A commit lost this way and noticed a week later
is not recoverable by inspection.

**The check, and it is one command.** Before trusting a commit made any time a
suite job might be running against the same worktree:

```bash
git rev-parse --abbrev-ref HEAD     # prints `HEAD` when detached, not the branch
```

More directly, verify the remote actually moved rather than that the push
succeeded:

```bash
git ls-remote origin <branch>        # compare to git rev-parse HEAD
```

**Status: advice, not prevented.** The real fix is for the suite launcher to
stop borrowing the caller's worktree — `git worktree add` a throwaway checkout
for the ref under test, run there, and delete it — which removes the window
entirely rather than asking every session to remember it. `suite.sbatch` lives
outside the repo (`/exp/sgreenberg/suite.sbatch`), so that change is not in this
tree and is not something `preflight.sh` can gate: preflight guards experiment
*launches*, and this is a hazard of the interval *after* one.

Until then the rule is the simple one: **do not commit in a worktree while a
suite job is running against it.** Wait for the job, or commit from a different
worktree on the same branch.

Related: `2026-09-03-a-deep-grid-sized-from-its-shipped-arm.md` for the same
shape of failure — a step that reported success while doing something other than
what the caller believed.
