# 2026-09-07 — the launch output never named the checkout (#3693)

**Study:** the shared pile (`launch_pile.sh`), while building `vg_scale` for
#3670. **Cost:** none, by luck. The stale tree happened to be old enough to lack
`build_pile.py --rebuildable`, which the launcher's own canary calls, so it
failed loudly on a missing flag. Any *younger* stale tree would have rebuilt
every cell and reported success.

`launch_pile.sh` set the checkout its jobs import from a fixed path:

```sh
REPO="${VTS_REPO:-/exp/$USER/projects/vts-pile}"
```

So `bash launch_pile.sh vg_scale`, run from any other worktree, submitted jobs
that built the pile out of `/exp/$USER/projects/vts-pile` — 1,420 commits behind
`origin/dev`, predating the `vg_scale` dataset entirely. Deriving `REPO` from the
script's own location fixes the default in one line.

**The one-line fix is not the lesson.** The default had been wrong for months and
nobody saw it, because the launcher printed

```
submitted vg_scale -> job 4412291
```

and never said *which tree at which commit*. There was nothing on screen to be
wrong. A launcher that names its checkout is one a human can catch reading the
scrollback; a launcher that names nothing can only be caught by an unrelated
crash, which is exactly how this one surfaced.

The same blind spot ran one layer deeper. Each cell's provenance sidecar recorded
the *machine* in detail — GPU, CPU model, kernel dispatch, transformers version
— and a bare commit hash, but not the **path**. A cell built by the stale tree
and one built by the tree you were reading were therefore indistinguishable in
the record unless somebody resolved the hash by hand. It is the same shape as
[a fresh worktree ran the *other* worktree's
code](2026-08-07-a-fresh-worktree-ran-the-other-worktrees-code.md) and
[a launcher default outlives the directory it
names](2026-08-12-a-launcher-default-outlives-the-directory-it-names.md): the
error is invisible, not improbable.

**Prevented.** `scripts/experiments/repo_stamp.sh` does two things a default
cannot, and `launch_pile.sh` calls it before it submits anything:

```
=== code under launch ===
repo:    /exp/sgreenberg/projects/VTSearch
head:    a9dd62ff (dev)
base:    level with origin/dev
```

and it **refuses** a checkout `VTS_MAX_BEHIND` (default 100) or more commits
behind `origin/dev` — the bar `preflight.sh` check 4 already applied to study
worktrees, now applied to the pile, which is the artifact every study reads.
`VTS_MAX_BEHIND=0` waives it for a build that means to run old code. Each cell's
sidecar now carries a `code` block (repo, commit, branch, dirty) and
`--provenance` says so out loud when a pile's cells came from more than one tree.

**The audit the fix asks for.** Every other GRID launcher that pins a fixed
`VTS_REPO` names a *study-specific* worktree on purpose, and all but a handful
run `preflight.sh`, whose check 4 already applies this bar. The pile launcher was
the outlier precisely because it is shared: no study "owns" it, so no study's
preflight covered it.

**The general rule:** a launcher's output must name every resolved default that
changes what runs — #3269's lesson was "record a resolved default as data, not
just as a launcher argument", and a checkout is the largest such default there
is. If the banner cannot be wrong, nothing on screen can be checked.
