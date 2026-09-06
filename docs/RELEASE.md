# Release runbook: `dev` → `main`

This is the procedure the **Dev2Main** Routine follows to promote `dev` to `main`. It lives in the repo (not in Claude settings) so it's versioned, PR-reviewable, and run by reference: the Routine prompt is a thin pointer at this file.

> **Override for this procedure only:** the final release PR's `base` is
> **`main`**, not `dev`. This is the one sanctioned exception to CLAUDE.md's
> "never open a PR to `main`" rule — it applies solely to the release PR
> opened in step 5, and only when running this runbook.

Work through the steps in order.

## 1. Vulture dead-code audit

Run:

```
python scripts/vulture-audit.py
```

The script owns the whole invocation — scan paths, excludes, ignore lists, confidence floor — so there is nothing to copy-paste and nothing to drift. It scans every tier that defines or consumes first-party Python (`vtsearch/`, `app.py`, `tests/`, `vtscore/`, `tests_lib/`, `scripts/`) and applies `.vulture-whitelist.py` itself.

The **findings** are intentionally **not** a CI gate (false positives against the plugin-discovery pattern), so a non-clean exit doesn't block the promotion — but every finding gets triaged:

- **Genuinely unused** → delete the symbol and any imports/references that fall out.
- **Used reflectively** → add it to `.vulture-whitelist.py` with a one-line comment explaining the indirect use.

Two constraints on that triage:

- **A hit on a public, documented, or entry-point-facing `vtscore` name is not evidence it can be deleted.** Out-of-tree extensions import `vtscore` symbols no in-repo grep can see, so the library tier surfaces public names with no *internal* caller by construction. Those get a whitelist entry with a true reason. Removing one is a deliberate library break: it needs an `[Unreleased]` entry in `vtscore/CHANGELOG.md` and the owner's explicit sign-off. See CLAUDE.md, "Dead code in `vtscore/` is a claim you cannot verify by grepping".
- **The whitelist entry's reason has to be true.** `python scripts/vulture-audit.py --check-whitelist` (a `run-tests.sh` lane) fails when an entry suppresses nothing, so a fabricated justification does not stay hidden — but it also means whitelisting is no longer free. Whitelist what is genuinely reflective; delete what is genuinely dead.

## 2. Land the cleanup

Commit the triage changes on the current branch **before** opening the PR — not as a follow-up. If vulture was clean, skip this step.

## 3. Write the release summary

Run `git fetch origin --prune`, then summarize
`git log origin/main..origin/dev --no-merges --reverse`.

**Format:** categorized bullets under these headings (collapse any with zero items): **Features**, **Bug fixes**, **Performance**, **Refactors / internals**, **Documentation**, **Dev tooling**.

**Constraints:**

- Hard cap **1000 characters**. If you'd exceed it, drop or combine the lowest-impact items.
- No PR numbers, no `#1234`, no commit SHAs, no author handles, no branch names, no issue references. The summary is for end-users.
- If vulture was clean, append a single line: `vulture: clean.` Otherwise:
  `vulture: <N> findings triaged.`

Use the summary verbatim as the PR body (step 5) and also output it in chat, written in plaintext, so the MD formatting can be copied.

## 4. Rebuild the punch-card graphic

`scripts/punchcard/punchcard.py` is a **pure renderer**: it reads the hand-maintained data file `scripts/punchcard/pr_merges.txt` (one `<pr_number>|<merged_at_utc_iso8601>` line per merged PR) and rewrites the PNG. It does **not** generate the data file — you refresh that yourself first.

- Append a line for each PR merged into `dev` since the last release. `python scripts/release-prs.py` enumerates them from step 3's `origin/main..origin/dev` window (step 6 walks the same window for the same list, and step 6 says why it is a script rather than `git log --merges`); take each PR's number and merge timestamp (`merged_at`, UTC ISO 8601) and add `<pr_number>|<merged_at>` to `scripts/punchcard/pr_merges.txt`. The file is sorted-unique by PR number, so keep it that way.
- Run `python scripts/punchcard/punchcard.py`, which rewrites `scripts/punchcard/vtsearch_pr_punchcard.png`.
- Commit the regenerated PNG and the updated `pr_merges.txt`.

## 5. Open the release PR

- **Title:** `Release: dev → main (YYYY-MM-DD)` using today's date.
- **Base:** `main`. **Head:** `dev`.
- **Body:** the step-3 summary, verbatim.

## 6. Close the issues shipped in this release

Now that the release PR is open, close the GitHub issues whose fixes are included in this `dev → main` batch. This is the counterpart to the per-fix rule in CLAUDE.md: individual fix PRs link their issue with a `Closes #N` keyword but leave it **open** (their merge to `dev` can't auto-close it), and this step is what finally closes it once the fix reaches `main`.

**Find the candidate issues** from this release's PRs:

- List the pull requests merged into `dev` within this release range with `python scripts/release-prs.py` — the same `origin/main..origin/dev` window used for the summary in step 3. It prints `<pr>|<sha>|<subject>` per PR, so `| cut -d'|' -f1` gives the numbers; anything it could not attribute to a PR goes to stderr for a glance rather than being dropped.
- For each such PR, read its body and collect **every** issue it references, keeping track of which keyword introduced each reference. Sort them into two buckets:
  - **Closing** — `Closes #N`, `Fixes #N`, `Resolves #N` (case-insensitive).
  - **Non-closing** — `Refs #N`, `Part of #N`, or a bare `#N` mention.
- Then add a third bucket, the **orphan backstop**: list the repo's still-open issues and check each one's comments for a pointer at a PR in this release range (`Addressed in #M`, `Fixed in #M`, and similar). Collect any whose pointer names a PR in the range that never referenced it back. (To keep this cheap, it's enough to check issues updated since the previous release.) A pointer naming a **commit** rather than a PR (`Fixed on dev by <sha>`) belongs in this bucket too — resolve the SHA to the PR that carried it (step 6b's `NEEDS REVIEW` notes give the recipe, including the case where the SHA is not on `dev` at all), then reconcile it like any other orphan.

**Both merge shapes count, which is why the list comes from a script.** A PR merged with GitHub's "Squash and merge" lands as a single **one-parent** commit — there is no merge commit, so `git log --merges` walks straight past it. Four PRs landed that way on 2026-09-06 (#3671, #3672, #3682, #3685), in a window that otherwise held 41 findable merges. Enumerating merges would have orphaned all four *by construction*: none of the three buckets below would ever have been offered their issues, and no later release re-examines an already-merged PR. `scripts/release-prs.py` walks `--first-parent` instead — which is what "landed on `dev`" actually means, independent of how it landed — and reads the PR number off the **trailing** `(#N)` of a squash subject, because the leading one is usually the issue the PR closes (`... measured (#3673) (#3685)`).

Non-closing references are **not** silently skipped. A PR that finishes an issue but writes `Refs #N` would otherwise orphan it permanently: this step skips it, and because no later release re-examines an already-merged PR, nothing ever revisits it — the issue stays open forever while its fix is live in `main`. Real incident: #2940, #2930 and #2951 each shipped in the 2026-08-12 release under `Refs`, with an "Addressed in #M" comment on the issue, and all three stayed open. So the non-closing and orphan buckets get **reconciled** rather than dropped.

**The hardest orphan is a duplicate.** #2911 shipped to `main` in the 2026-08-11 release and stayed open until 2026-08-17. It was a duplicate of #3025, the fix PR wrote `Closes #3025` only, and at release time #2911 had no comments and no PR references at all — so all three buckets were blind to it by construction, not by a keyword slip. No sweep rule recovers an issue that nothing on GitHub links to; that one is prevented upstream, by CLAUDE.md's duplicate rule. What this step *can* do is catch the late pointer: the resolving comment landed a day after the release, so an issue whose newest comment claims a fix by a PR or commit from **any earlier** release deserves a look, not just this one's.

**Reconcile each issue in the non-closing and orphan buckets.** Read the issue (body *and* comments) alongside the PR, then close it only when **both** hold:

- The PR (or a comment on the issue pointing at it) claims to address the issue **without qualification** — e.g. an `Addressed in #M` comment, or a PR body that plainly does everything the issue body asks.
- Neither the PR body nor any later comment names work still owed **by that issue**. Scope the PR explicitly deferred into a plan file or a separate issue is no longer owed here and does not make it partial; likewise, an issue that was rescoped narrower counts as finished if the PR does all of what remains.

Anything else stays open — a genuinely partial `Refs` is doing its job.

**Then, for each issue to be closed (closing bucket, plus the reconciled ones):**

- Skip it if it is already closed. Never reopen or re-close.
- Close it with `state_reason: completed`.
- **Strip the `solved` label in the same write.** `solved` means "the development is done; only merges remain" (see CLAUDE.md), and this close is the moment the last merge lands — so the label has nothing left to say. Pass `labels` explicitly with every label the issue keeps (`claude`, `experiment`, …) minus `solved`; `labels` *replaces* the whole set, so passing `[]` would wipe the rest. A `PreToolUse` hook blocks a `completed` close that keeps `solved` or omits the array. If you close through `gh` instead, the same hook looks the issue's labels up and blocks the close when `solved` is really on it — strip it in the same motion (`gh issue edit <n> --remove-label solved && gh issue close <n> --reason completed`), and clear the assignee too (`--remove-assignee samggreenberg`). The lookup allows whenever it cannot reach GitHub, so step 6b's audit is still the thing that guarantees the strip.
- **Clear the assignee in the same write.** Pass `assignees: []`. An assignee means "somebody is working this right now" (see CLAUDE.md), which a closed issue cannot be. Unlike `labels`, there is nothing to preserve here — the empty array is always the right value on a close.
- Add a one-line comment noting it shipped to `main` in today's release and linking the fix PR (e.g. `Shipped to main in the 2026-07-14 release — fixed
  in #M.`). When the PR used a non-closing keyword, say so in that comment, so the mislabel is visible on the issue rather than silently corrected.

**Report the reconciliation in chat**, briefly: which issues came from the closing bucket, which were closed after reconciliation (and under which PR keyword), and which non-closing references were deliberately left open. This is the only place a crossed wire between a PR keyword and an issue comment becomes visible, so do not collapse it to a bare count. If no qualifying issues are found, state that and do nothing.

## 6b. Audit the `solved` label

`solved` means "the development is done; only merges remain", and the fix session applies it when it opens the PR (CLAUDE.md). So by the time you get here the label should already be right, and this step is an **audit**: `scripts/reconcile-solved-labels.py` catches issues a session forgot to label, issues whose fix PR was later abandoned, and stale labels left behind by step 6's closes. It audits the **assignee** on the same pass, for the same reason — an assignee outlives its purpose exactly when `solved` does.

Run it right after step 6 to confirm nothing was left behind. It is also worth running **between** releases — the views it keeps honest (`is:issue is:open -label:solved`, what a human should pick up next) matter most while the release is still weeks away.

The script is a pure function from data to plan — it does no network I/O of its own, so gather the data first and pipe it in.

**Use the `gh` CLI to gather it.** This paragraph used to say the GitHub REST API was unreachable from a Claude session and that access was intermediated by the MCP server. That was only ever true of a raw `GITHUB_TOKEN` (which 403s, and is not even set in a session); `gh` carries its own authenticated token and `gh api` works normally. Writing it as "unreachable" made the MCP server look load-bearing, and this recipe became unrunnable the moment that server was removed from a machine — which has now happened. `gh api` and `gh pr list`/`gh issue list` are the path; the MCP tools are an equivalent alternative where they happen to be configured, not a prerequisite.

**Run it from the laptop, though.** The correction above is about the laptop, where `gh auth login` has run. A Claude Code on the web container ships no `gh` at all, and installing one there does not help: it picks up the ambient `GH_TOKEN`, which 403s with `GitHub access is not enabled for this session` on REST and GraphQL alike (measured while building the `gh issue close` guard in #3634). In a web session the `github` MCP tools are the only working path.

1. List the PRs merged into `dev` since the last release — the same `origin/main..origin/dev` window as step 3 — and read each one's **body**. These are `release_prs`.
2. List the PRs currently **open** against `dev` (`open_prs`) and those **closed without merging** since the last release (`abandoned_prs`), with their bodies. The open ones are why an issue can be labelled before any merge; the abandoned ones are the only way a label comes off outside a close.
3. List the repo's issues with their `labels`, `state`, and `assignees`, and fetch each one's **comments** in chronological order (the API default).
4. Assemble them into one JSON object and run the script:

```json
{
  "release_prs": [{"number": 3128, "body": "... Closes #3077 ..."}],
  "open_prs": [{"number": 3160, "body": "... Closes #3081 ..."}],
  "abandoned_prs": [{"number": 3155, "body": "... Closes #3090 ..."}],
  "issues": [
    {"number": 3077, "state": "open", "labels": ["claude"],
     "assignees": ["samggreenberg"],
     "comments": [{"body": "Addressed in #3128"}]}
  ]
}
```

```
python scripts/reconcile-solved-labels.py --input plan-input.json
```

It prints four action buckets (`ADD`, `REMOVE`, `NEEDS REVIEW`, `CLEAR ASSIGNEE`) plus a `no change: N issue(s)` summary line at the end. **Apply `ADD`, `REMOVE`, and `CLEAR ASSIGNEE` directly** — they are unambiguous. **Do not apply `NEEDS REVIEW`**; read those issues yourself. An issue lands there for one of three reasons, all genuinely undecidable from the outside:

- **A fix pointer is not the newest comment.** The later comment may be a maintainer saying "thanks" or the reporter saying the fix does not work. Tagging would bury a dispute — hiding an issue that still needs solving — while skipping would leave solved work in the human queue.
- **A comment claims a fix by commit SHA** instead of `Addressed in #M`. The script has only the JSON you piped in, so it cannot map a commit to its PR — but the claim is real, so it is surfaced. Resolve it with `git log --ancestry-path <sha>..origin/dev --first-parent --oneline | tail -1`, which names the commit that landed it — a merge commit or a squash, and either one carries its PR number. If `git merge-base --is-ancestor <sha> origin/dev` fails, the SHA is not on `dev` at all and the recipe returns nothing: its branch was **squashed** on merge, which discards the branch's own commits. Match the subject against the window instead (`python scripts/release-prs.py | grep -i '<words from the subject>'`), or open the SHA on GitHub, which still knows its PR. Then re-run with a corrected pointer.
- **The issue carries `solved` but nothing resolves it** — stale, or fixed in an earlier release.

That is the same "not silently skipped" principle step 6 applies to non-closing references.

`CLEAR ASSIGNEE` is orthogonal to the label buckets — an issue whose label is already correct can still owe an assignee removal, so it appears under `CLEAR ASSIGNEE` while also counting as "no change" for its label. The bucket only ever asks you to *remove* an assignee: the script cannot tell "nobody is working this" from "a session started five minutes ago", so it never proposes assigning anyone, and it leaves `NEEDS REVIEW` issues and fallen-through fixes alone.

Add `--check` to make it exit non-zero when anything needs attention, and `--json` for machine-readable output.

## 7. Prune plan pointers for the closed issues

Per CLAUDE.md's "Issues vs `docs/plans/`: one item, one home" invariant, plan files reference shipped issues by a one-line checkbox pointer (`- [ ] #N — title`) rather than duplicating their bodies. When an issue closes, its pointer is stale and should go.

For **every** issue closed in step 6, grep `docs/plans/` for its number:

```
grep -rn '#<number>' docs/plans/
```

For each hit, delete that pointer line (or check its box, `- [x]`, if the umbrella deliberately keeps a shipped-slice ledger — prefer deletion unless the surrounding plan clearly does the latter). Leave the `<!-- item-sep -->` sentinels around it in place, per the plan-file policy. If the deletion empties a plan entirely and no follow-ups remain, delete the plan file (after absorbing any lasting design notes into the permanent docs), as the plan-file policy directs. Commit these prunes.

This is what makes issue-dismissal trickle back automatically: because plans hold only pointers (never bodies), pruning is always a safe one-line deletion.

**Whenever this step deletes a plan file, also grep the source tree for it** — not just `docs/plans/`. Module docstrings and inline comments cite plan files by path far more often than other plan files do, and `docs/plans/` alone misses all of them:

```
grep -rl 'docs/plans/<deleted-name>\.md' --include="*.py" --include="*.ts" --include="*.sh" --include="*.md" --include="*.json" --include="*.html" .
```

Fix every hit in the same commit: repoint it at the permanent doc the rationale was folded into, or drop the pointer outright when the surrounding prose is already self-contained (the common case). See CLAUDE.md's plan-file policy for the full rule; issue #2982 is the incident that motivated it — 94 source files had gone dangling this way across 13 deleted plans before anyone grepped for them.
