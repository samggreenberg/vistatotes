# Controlling who can change `main` (and `dev`)

Goal: **only @samggreenberg should land changes on `main`.** `dev` is the shared
integration branch and stays open to collaborators via PRs.

## Current state (verified 2026-09-08)

This repository is **public**, so branch protection and rulesets are available at
no cost, and both long-lived branches carry protection:

| | |
|---|---|
| visibility | **public** (`"private": false`) |
| default branch | `main` |
| `main` | **protected** |
| `dev` | **protected** |
| collaborators | 4 — @samggreenberg (admin), @xofm31, @MatthewELucio, @CarHorseBatteryStaple (write) |

That is the *existence* of protection, which is what the branch API reports. It
does not say what each rule enforces — see
[Reading the rules that are actually set](#reading-the-rules-that-are-actually-set)
below, and read them before relying on any specific guarantee.

### The one limitation that has not changed

**Every collaborator on a personal-account repo has write access.** The
Read / Triage / Write / Maintain roles exist only inside organizations, so the
three collaborators above cannot be made read-only while the repo lives under a
personal account. Restricting *who may push to `main`* is therefore done by the
branch rule's push restriction, not by giving anyone a weaker role.

## Why `dev` survives a Dev2Main release

Worth stating explicitly, because the repo now has **"Automatically delete head
branches"** enabled and the [Dev2Main](RELEASE.md) release PR is `dev` → `main` —
which makes `dev` the *head* branch of a PR that gets merged every release.

`dev` is not deleted, for two independent reasons:

1. **It is protected.** GitHub's automatic head-branch deletion skips protected
   branches. A protection carrying `allow_deletions: false` blocks the deletion
   outright rather than merely declining to initiate it.
2. **It is the base of other open PRs.** GitHub does not auto-delete a branch
   that another open PR is targeting, and on any ordinary day `dev` is the base
   of several.

If it were ever deleted anyway, nothing is lost: after the release merge `dev`'s
tip is an ancestor of `main`, and the merged PR page offers **Restore branch** to
recreate it at the same commit. The cost would be disruption — the
`.claude/hooks/session-start.sh` hook and every open PR's base break until it is
restored — not lost work.

## Soft controls, which still do useful work

Protection rules are the hard gate; these remain worth keeping because they shape
behaviour before anyone reaches the gate.

### CODEOWNERS auto-requests review

[`.github/CODEOWNERS`](../.github/CODEOWNERS) is `* @samggreenberg`, so GitHub
automatically requests that review on every PR. Combined with a `main` rule that
requires Code Owner review, it makes @samggreenberg the only valid approver for
anything landing on `main`.

### Team convention

`CLAUDE.md` encodes the working agreement, and it is what keeps `main` quiet in
practice — nobody working off `dev` has a routine reason to touch `main`:

- All work branches off `dev`; all PRs target `dev`, **never** `main`.
- `main` is updated **only by @samggreenberg**, by promoting `dev` → `main`.
- Collaborators do not push directly to `main`.

### Notifications

So an unwanted push to `main` is visible rather than silent: **Watch → All
Activity** (or **Custom → Pushes**), or subscribe to the `main` commit feed at
`https://github.com/samggreenberg/vtsearch/commits/main.atom`.

## Reading the rules that are actually set

The branch listing reports only whether a branch is protected. To see what the
protection contains:

```bash
gh api repos/samggreenberg/vtsearch/branches/main/protection
gh api repos/samggreenberg/vtsearch/branches/dev/protection
```

The UI equivalent is **Settings → Rules → Rulesets**, or the classic
**Settings → Branches** editor.

## The intended rules

Recorded here as the reference for what protection *should* say. Check the live
rules against this rather than assuming they match.

Lock `main` — PR plus Code Owner review required, only @samggreenberg may push:

```bash
gh api -X PUT repos/samggreenberg/vtsearch/branches/main/protection \
  --input - <<'JSON'
{
  "required_status_checks": null,
  "enforce_admins": false,
  "required_pull_request_reviews": {
    "required_approving_review_count": 1,
    "require_code_owner_reviews": true,
    "dismiss_stale_reviews": true
  },
  "restrictions": { "users": ["samggreenberg"], "teams": [], "apps": [] },
  "allow_force_pushes": false,
  "allow_deletions": false
}
JSON
```

With CODEOWNERS as above, `require_code_owner_reviews` means only
@samggreenberg's review satisfies the gate, and `restrictions.users` limits
pushes. Set `enforce_admins: true` to bind yourself to the same rules.

Keep `dev` lighter — a PR guardrail with no mandatory approver, so routine work
and Claude PRs keep flowing:

```bash
gh api -X PUT repos/samggreenberg/vtsearch/branches/dev/protection \
  --input - <<'JSON'
{
  "required_status_checks": null,
  "enforce_admins": false,
  "required_pull_request_reviews": {
    "required_approving_review_count": 0,
    "require_code_owner_reviews": false
  },
  "restrictions": null,
  "allow_force_pushes": false,
  "allow_deletions": false
}
JSON
```

`allow_deletions: false` on `dev` is the setting that makes the release flow
above safe. It is worth confirming it is actually set.

## History

This document previously described the repo as **private on the Free plan**,
where protected branches and rulesets are unavailable, and concluded that
"nothing GitHub-side can hard-*prevent* a collaborator from pushing to `main`."
It carried a cost table of ways to *obtain* enforcement — GitHub Pro, making the
repo public, or moving to an organization.

The repo has since taken the free option in that table and gone public, so
enforcement is available and in place, and the sections describing its absence
were removed rather than left to be reasoned from. The collaborator list in that
version (seven accounts) no longer matches either.
