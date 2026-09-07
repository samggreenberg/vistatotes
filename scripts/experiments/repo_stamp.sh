#!/usr/bin/env bash
# Say which checkout a GRID launch will import, and refuse one that is stale.
#
# Source it, then call once per launch, before anything is submitted:
#
#   source "$(dirname "${BASH_SOURCE[0]}")/../repo_stamp.sh"
#   repo_stamp "$REPO" || exit 1
#
# Why this exists. `launch_pile.sh` set the checkout its jobs import from a
# fixed path (`/exp/$USER/projects/vts-pile`), so running it from any other
# worktree submitted jobs that built the pile from a DIFFERENT tree -- 1,420
# commits behind `dev` by 2026-09-06, predating the `vg_scale` dataset entirely
# (#3693). The path default is fixed now, but the reason nobody noticed for
# months is the part a default cannot fix: **the launch output never named the
# checkout**. It printed `submitted vg_scale -> job NNN` and nothing else, so
# there was nothing on screen to be wrong. Had the stale tree been any closer in
# age it would have rebuilt every cell from code nobody was looking at and
# reported success, the way #3269 measured a retired head from a worktree 321
# commits behind.
#
# Two things, therefore, and they are different:
#
#   1. **Print it.** A launch that names its checkout and commit is one a human
#      can catch. This costs nothing and catches the cases the gate cannot
#      (right distance from `dev`, wrong tree).
#   2. **Gate it.** `preflight.sh` check 4 fails a study worktree more than
#      `PREFLIGHT_MAX_BEHIND` commits behind `origin/dev`; nothing applied that
#      to the pile, which is the artifact every study reads. Same bar, same
#      default, so the two agree about what "stale" means.
#
# Knobs:
#   VTS_MAX_BEHIND    commits behind origin/<base> that REFUSES the launch
#                     (default 100, matching PREFLIGHT_MAX_BEHIND). `0` disables
#                     the gate -- for a run that deliberately builds from an old
#                     checkout. The banner still prints; only the refusal goes.
#   VTS_BASE_BRANCH   integration branch to measure against (default `dev`).
#
# On success it exports, for a caller that wants to stamp the launch into the
# job it submits: REPO_STAMP_COMMIT (full sha), REPO_STAMP_BRANCH,
# REPO_STAMP_BEHIND. Each is empty when it could not be measured.

repo_stamp() {
  local repo="${1:?repo_stamp: pass the checkout the jobs will import}"
  local max_behind="${VTS_MAX_BEHIND:-100}"
  local base="${VTS_BASE_BRANCH:-dev}"

  REPO_STAMP_COMMIT=""
  REPO_STAMP_BRANCH=""
  REPO_STAMP_BEHIND=""

  echo "=== code under launch ==="
  echo "repo:    $repo"

  if ! git -C "$repo" rev-parse --git-dir >/dev/null 2>&1; then
    echo "head:    NOT A GIT CHECKOUT -- the build can stamp no commit"
    if [[ "$max_behind" == "0" ]]; then
      echo "gate:    disabled (VTS_MAX_BEHIND=0)"
      return 0
    fi
    echo >&2
    echo "refusing to submit: a cell built from an unidentifiable tree cannot be" >&2
    echo "reproduced, and nothing later can say what code produced it." >&2
    echo "  -> launch from a git worktree, or set VTS_MAX_BEHIND=0 if you mean it." >&2
    return 1
  fi

  local sha short branch dirty=""
  sha="$(git -C "$repo" rev-parse HEAD 2>/dev/null || true)"
  short="$(git -C "$repo" rev-parse --short HEAD 2>/dev/null || true)"
  branch="$(git -C "$repo" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
  if [[ -n "$(git -C "$repo" status --porcelain --untracked-files=no 2>/dev/null)" ]]; then
    dirty="  +uncommitted tracked changes"
  fi
  REPO_STAMP_COMMIT="$sha"
  REPO_STAMP_BRANCH="$branch"
  echo "head:    ${short:-unknown} (${branch:-unknown})$dirty"

  # Best effort: a login node without the remote still gets the banner, and the
  # unmeasured distance is said out loud rather than passing as "level".
  git -C "$repo" fetch -q origin "$base" >/dev/null 2>&1 || true
  if ! git -C "$repo" rev-parse --verify -q "origin/$base" >/dev/null 2>&1; then
    echo "base:    origin/$base unavailable -- staleness UNMEASURED"
    return 0
  fi

  local behind
  behind="$(git -C "$repo" rev-list --count "HEAD..origin/$base" 2>/dev/null || echo "")"
  if [[ -z "$behind" ]]; then
    echo "base:    could not measure distance from origin/$base"
    return 0
  fi
  REPO_STAMP_BEHIND="$behind"

  if [[ "$max_behind" == "0" ]]; then
    echo "base:    $behind commits behind origin/$base (gate disabled, VTS_MAX_BEHIND=0)"
    return 0
  fi
  if [[ "$behind" -ge "$max_behind" ]]; then
    echo "base:    $behind commits behind origin/$base  (gate: $max_behind)"
    echo >&2
    echo "refusing to submit: this checkout is $behind commits behind origin/$base." >&2
    echo "  -> the jobs would build from code that old, and report success." >&2
    echo "     Rebase the worktree, point VTS_REPO at a current one, or set" >&2
    echo "     VTS_MAX_BEHIND=0 if building from this commit is the point." >&2
    return 1
  fi
  if [[ "$behind" -gt 0 ]]; then
    echo "base:    $behind commits behind origin/$base (under the $max_behind gate)"
  else
    echo "base:    level with origin/$base"
  fi
  return 0
}
