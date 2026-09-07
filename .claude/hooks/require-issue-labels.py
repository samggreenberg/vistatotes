#!/usr/bin/env python3
"""PreToolUse gate: no Claude-filed GitHub issue lands without its labels.

CLAUDE.md ("Label every issue you file") requires `claude` on every issue
Claude creates, and `experiment` on any issue that cannot be closed without a
measurement (a GRID/SLURM sweep, an eval arm, a calibration run).

That rule used to be prose only, which made it unenforceable in two distinct
ways -- and issue #3127 was filed unlabeled by hitting the first one:

1. **Staleness.** The rule reached `dev` at 2026-08-12 19:29 UTC; #3127 was
   filed at 17:17 UTC by a session whose checkout predated it by two hours.
   Prose in CLAUDE.md can only bind a session that checked it out.
2. **Attention.** CLAUDE.md is long, and a rule near the bottom competes with
   everything else in the window on a session that has been running for hours.

A hook is immune to both: it runs at tool-call time, from the checkout as of
session start, and it does not need to be remembered.

**It is only immune to them on the call paths it actually watches.** For its
first weeks this hook policed `mcp__github__issue_write` alone, while sessions
here filed issues with the `gh` CLI through `Bash` -- 76 `gh issue create`
commands against no MCP calls at all, of which 29 carried no `--label` and only
9 carried `experiment`. The gate was not bypassed; it was never on the road.
That is the same shape as #3440 (a hook that grepped an empty string for its
whole life): a safety net is worth nothing until something has been observed to
hit it. So the hook now reads `Bash` commands too, and `TestWiring` asserts
both registrations rather than trusting the file's existence.

The hook has a second job at the other end of an issue's life: the `solved` label
(docs/RELEASE.md step 6) means "the development is done; only merges remain",
so it must come off in the write that closes the issue -- that close *is* the
last merge landing. See `_close_problems`.

That job had the same one-road problem (#3634): the MCP rule is "restate the
label set, minus `solved`", and `gh issue close` has no `--label` flag at all,
so it has no text-only translation. The three candidates each fail somewhere --
demanding a chained `--remove-label solved` fires on the majority of closes
whose issue never carried the label; leaving it to
`scripts/reconcile-solved-labels.py` repairs the state only when someone
remembers to run it, which is the "no session is around to observe it" problem
`solved` was invented to dodge. So this path asks GitHub instead: a
command-position `gh issue close` triggers one `gh issue view --json labels`
lookup, and the close is blocked only when `solved` is *provably* on the issue.
See `_gh_close_problems`.

That is the one place this hook does I/O, and it is deliberately bounded:

* it runs for `gh issue close` alone, so the cost lands on a handful of calls
  rather than on every `Bash` command in every session;
* it is skipped entirely when the command already chains the fix, so following
  the denial message costs no second lookup;
* **any** failure -- `gh` absent, unauthenticated, offline, slow, an
  unparseable answer -- returns "could not tell" and allows, per the contract
  below. A network blip must not wedge a close.

So this half of the guard is a catch, not a guarantee, and it is worth knowing
where it is awake: on the laptop, where `gh auth login` has run and where the
`gh issue close` commands in the transcripts were actually typed. A Claude Code
on the web container has no `gh` at all, and the ambient `GH_TOKEN` there 403s
on REST and GraphQL alike, so a close issued from one is allowed unexamined.
`scripts/reconcile-solved-labels.py` remains the thing that guarantees the
strip; this hook is what makes the repair unnecessary most of the time.

Contract: read the PreToolUse payload on stdin, exit 2 to block (stderr is fed
back to Claude as the reason), exit 0 to allow. Anything unexpected -- a
payload we cannot parse, a tool we do not police -- allows, because a hook that
fails closed would wedge every unrelated GitHub call.

Escape hatch for the `experiment` heuristic (see `_looks_like_an_experiment`):
put `<!-- not-an-experiment: <reason> -->` in the issue body. It renders as
nothing on GitHub and greps cleanly.

**The reason is read** (see `_reason_problem`). For its first weeks the marker
matched on its prefix alone, so the text after the colon was never looked at by
anything -- it forced a reason to be *typed*, not to be *true*. Issue #3708
counted the result: four markers in three days had to be corrected by hand, and
in every one of them the sentence that disproves the marker was inside the
marker itself. #3683 promised "no GPU, no sweep" while closing on
`build_pile.py --provenance` over a newly built cell; #3693 closed on
"confirming the submitted job imports THAT worktree"; #3694 closed on `cat`ing
a file that exists only on the GRID plus one submitted cpu job. So a reason
that describes a run is now refused, and the refusal quotes the offending
phrase back.

That costs a wrongly-blocked issue one sentence of rewording at filing time. A
wrongly-passed one costs a whole session: `label:experiment` is a *scheduling*
queue, so a false marker puts GRID-only work into the pick-up-now queue, and
the session that takes it gets as far as the acceptance check and stops -- or
invents the missing half. The gate is deliberately biased toward the cheap
failure.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hook_payload import read_payload, tool_arguments  # noqa: E402

TOOL_SUFFIX = "issue_write"
BASH_TOOL = "Bash"

SOLVED_LABEL = "solved"

# `gh issue create`, wherever it sits in a compound command -- these arrive as
# `... && gh issue create ...`, inside `ssh grid '...'`, and after a heredoc
# that builds the body. The lookbehind keeps `foo-gh issue create` out.
GH_ISSUE_CREATE = re.compile(r"(?<![\w./-])gh\s+issue\s+create\b")

# ...but only where it is being *run*. A command that merely mentions the
# string -- a commit message, a doc heredoc, the body of an issue about this
# very rule -- must pass straight through; blocking those is a false block on
# ordinary work, not a caught mistake. So the match must sit in command
# position: at the start, after a separator, or after a wrapper that takes a
# command as its argument. The known gap is a quoted one-liner with no
# separator inside it (`ssh grid 'gh issue create ...'`), which reads
# identically to a quoted mention; erring toward the miss there is deliberate,
# since every such form observed in practice chains through `&&` first.
#
# A bare backtick is deliberately NOT a separator here, though it opens command
# substitution in shell. It was, and it false-blocked an ordinary edit in the
# first session after this hook shipped: markdown-quoting the command name --
# `` `gh issue create` `` in a commit message, a PR body, a docstring, a line of
# CLAUDE.md -- is overwhelmingly the common meaning of a backtick in this repo's
# commands, and every one of those is work *about* the rule rather than a call
# that files anything. Nothing is lost by dropping it: the closing backtick
# lands inside the arguments, so a real `` `gh issue create ...` `` substitution
# was never parsed correctly anyway, and the modern spelling `$(...)` is still
# matched below.
GH_COMMAND_POSITION = re.compile(
    r"(?:^|[\n;|&(){}]|\$\(|\b(?:timeout\s+[\d.]+[smhd]?|nohup|sudo|env|command|exec|time))\s*$"
)

# `--label x`, `--label=x`, `-l x`; repeated flags and comma-separated values
# both appear in the transcripts. `--add-label`/`--remove-label` deliberately do
# NOT match: those belong to `gh issue edit`, and a create must carry its labels
# at creation time, not acquire them in a follow-up nobody is around to make.
GH_LABEL_FLAG = re.compile(r"""(?<![\w-])(?:--label[=\s]+|-l\s+)(['"]?)([A-Za-z0-9_,\- ]+?)\1(?=\s|$)""")

# A `--body-file` puts the issue body outside the command, out of the
# heuristic's reach, so it is read back when the path resolves.
GH_BODY_FILE_FLAG = re.compile(r"""(?<![\w-])(?:--body-file[=\s]+|-F\s+)(['"]?)([^\s'"]+)\1""")

# Invocations that file nothing: `--help` prints usage (and is exactly what
# someone types *after* being told to add `--label`, so blocking it would make
# the denial message a dead end), and `--web` hands the whole form to a browser
# where the CLI's flags do not apply.
GH_NON_FILING = re.compile(r"(?<![\w-])(?:--help|-h|--web|-w)(?=\s|$)")

# `--help` on any subcommand does nothing but print usage, and it is exactly
# what someone types *after* a denial tells them which flag to add.
GH_HELP = re.compile(r"(?<![\w-])(?:--help|-h)(?=\s|$)")

MAX_BODY_FILE_BYTES = 256 * 1024

# `gh issue close`, wherever it sits in a compound command -- the transcripts
# have it bare, chained after a `gh issue comment`, and inside `ssh grid '...'`.
GH_ISSUE_CLOSE = re.compile(r"(?<![\w./-])gh\s+issue\s+close\b")

# Where one command's arguments stop and the next command begins. Used to bound
# the flag parsing to the close's own arguments, so a chained
# `gh issue edit N --repo other/repo` cannot be read as the close's `--repo`.
GH_SEGMENT_END = re.compile(r"&&|\|\||[;\n|&]")

# The chained fix the denial message asks for. `gh issue edit` spells it
# `--remove-label` with no short form, so there is only the one shape to match.
GH_REMOVE_LABEL_FLAG = re.compile(r"""(?<![\w-])--remove-label[=\s]+(['"]?)([A-Za-z0-9_,\- ]+?)\1(?=\s|$)""")

# `--repo owner/name`, `-R owner/name`. Absent, `gh` resolves the repo from the
# working directory's git remote -- which is what the close itself would do, so
# the lookup and the close always agree on which issue is meant.
GH_REPO_FLAG = re.compile(r"""(?<![\w-])(?:--repo[=\s]+|-R\s+)(['"]?)([A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+)\1""")

# What `gh issue close` accepts as its target: a number, `#number`, or an
# issue URL. Anything else is something this hook does not understand, and "does not
# understand" always means allow.
GH_ISSUE_TARGET = re.compile(r"^#?(\d+)$|^https?://[^\s]*/issues/(\d+)/?$")

# Flags of `gh issue close` that swallow the next token, so a numeric value
# (`--comment 3`) is never mistaken for the issue number.
GH_CLOSE_VALUE_FLAGS = frozenset({"-c", "--comment", "-r", "--reason", "-R", "--repo"})

# Seconds to wait for the label lookup. A hook runs in front of the user's
# command, so a slow answer must be abandoned rather than waited out; the
# override exists for a slow link, and any value that does not parse falls back.
GH_LOOKUP_TIMEOUT_ENV = "VTSEARCH_GH_LOOKUP_TIMEOUT"
GH_LOOKUP_TIMEOUT_DEFAULT = 5.0

# The marker, with its reason captured. Non-greedy up to the terminator rather
# than `[^>]*`, so a reason containing `>` (a shell redirect, an arrow, a
# quoted diff line) still parses as a marker instead of silently ceasing to be
# one -- a marker the hook cannot see is reported as a missing marker, which is
# the one denial message a filer who wrote one cannot act on.
OPT_OUT = re.compile(r"<!--\s*not-an-experiment\s*:(?P<reason>.*?)-->", re.IGNORECASE | re.DOTALL)

# Phrases that describe machine time. These are read against the *marker's own
# reason*, not the issue body -- the body of an issue about a sweep will always
# talk about sweeps, so scanning it would make the marker unusable on exactly
# the issues that legitimately need it. The four corrections in #3708 are all
# catchable inside the marker, which is what makes this mechanical rather than
# a matter of taste.
RUN_SHAPED = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bsbatch\b",
        r"\bsqueue\b",
        r"\bsrun\b",
        r"\bscancel\b",
        # "the submitted job", "submit one cpu job to prove it". Bounded to one
        # clause so a marker ending "nothing is submitted." cannot reach a
        # "job" in the next sentence.
        r"\bsubmit\w*\b[^.;]{0,60}?\bjobs?\b",
        r"\b(?:cpu|gpu)\s+jobs?\b",
        r"\bnewly\s+built\s+\w+",
        r"\bre-?build\w*",
        r"\bbuild(?:ing|s)?\s+(?:a|the|every|one)\s+cells?\b",
        r"\bpile cells?\b",
        r"\bbuild_pile\.py\b",
        r"\blaunch_\w+\.sh\b",
        r"\bpreflight\.sh\b",
        r"\bslate import\b",
        r"\bimport\w*\s+(?:a\s+|the\s+)?(?:slate|pile|dataset)s?\b",
        r"\bsweeps?\b",
        r"\beval arms?\b",
        r"\bcalibration runs?\b",
        r"\bvtscore\.eval\b",
        r"scripts/experiments",
    )
]

# Case-sensitive, unlike everything above: a lowercase "grid" is a CSS grid far
# more often than it is this cluster, and a false block whose quoted phrase is
# `grid` from "the grid layout" reads as a bug rather than as a rule.
RUN_SHAPED += [re.compile(r"\bGRID\b"), re.compile(r"\bSLURM\b")]

# A run-shaped word inside a denial ("no sweep", "no GRID/eval run is
# involved") is the marker doing its job, not failing it. Bounded to the same
# clause -- no `.`, `;` or `:` between the negator and the phrase -- so
# "nothing measured. Submit one cpu job" is not read as a denial of the job.
NEGATED = re.compile(r"\b(?:no|not|never|nothing|none|neither|nor|without|n't)\b[^.;:]{0,24}$", re.IGNORECASE)

# A reason with fewer than this many non-space characters is not a reason.
# Deliberately a low floor: its whole job is the degenerate case (#3669's
# marker claimed only "not an experiment", which is the marker's own name), and
# a floor high enough to judge a *short but real* reason would be a floor high
# enough to block one. The run-shaped check above is what does the actual work.
MIN_REASON_CHARS = 12

# ...and the restatement itself, which clears any floor low enough to be safe.
BARE_RESTATEMENT = re.compile(r"^(?:it'?s |this is |it is )?not (?:an )?experiment[.!]?$", re.IGNORECASE)

# One of these alone means the issue cannot be closed without machine time.
# They are repo-specific enough that a false positive is a real surprise.
STRONG_SIGNALS = [
    r"vtscore\.eval",
    r"scripts/experiments",
    r"\bsbatch\b",
    r"\bSLURM\b",
    r"\bGRID\b",
    r"\bCALIB_EXP\b",
    r"\bmAP\b",
    r"\bnDCG\b",
    r"\beval arm\b",
    r"\bsweeps?\b",
    r"\bre-?measure",
    r"\bre-?run the\b",
]

# Individually weak -- "measure" shows up in plenty of pure code changes -- so
# two are required before the hook will block on them.
WEAK_SIGNALS = [
    r"\bcalibrat\w*",
    r"\bmeasur\w*",
    r"\bstud(?:y|ies)\b",
    r"\bbaselines?\b",
    r"\bbenchmarks?\b",
    r"\bablations?\b",
    r"\bexperiments?\b",
    r"\barms?\b",
    r"\brecall@",
    r"\bprecision@",
]


def _looks_like_an_experiment(text: str) -> bool:
    """Heuristic: does closing this issue require a run, not just an edit?"""
    if any(re.search(p, text, re.IGNORECASE) for p in STRONG_SIGNALS):
        return True
    hits = sum(1 for p in WEAK_SIGNALS if re.search(p, text, re.IGNORECASE))
    return hits >= 2


def _run_shaped_phrase(reason: str) -> str | None:
    """The first phrase in `reason` that describes machine time, if any."""
    for pattern in RUN_SHAPED:
        for match in pattern.finditer(reason):
            if NEGATED.search(reason[: match.start()]):
                continue
            return match.group(0).strip()
    return None


def _reason_problem(reason: str) -> str | None:
    """Why this marker does not count as a stated reason, or `None` if it does.

    Two refusals, in the order that produces the more useful message. A
    run-shaped reason gets the phrase quoted back, because that phrase *is* the
    argument against the marker and the filer wrote it themselves. A reason too
    thin to disagree with gets the criterion instead.
    """
    stripped = " ".join(reason.split())

    phrase = _run_shaped_phrase(stripped)
    if phrase is not None:
        return RUN_SHAPED_REASON.format(phrase=phrase)

    if len(stripped.replace(" ", "")) < MIN_REASON_CHARS or BARE_RESTATEMENT.match(stripped):
        return THIN_REASON.format(reason=stripped or "(empty)")

    return None


def _opt_out_refusal(text: str) -> str | None:
    """`None` if some marker in `text` releases the heuristic; else why none does.

    Every marker is judged and any one acceptable marker is enough, because a
    body can carry the real marker *and* a fenced example of the syntax (the
    body of an issue about this very rule does). Judging only the first would
    turn writing about the marker into a block.
    """
    refusals = []
    for match in OPT_OUT.finditer(text):
        problem = _reason_problem(match.group("reason"))
        if problem is None:
            return None
        refusals.append(problem)
    return refusals[0] if refusals else MISSING_EXPERIMENT


def _close_problems(args: dict) -> list[str]:
    """Guard the `solved` label on a completing close (the docs/RELEASE.md step-6 sweep).

    `solved` means "the development is done; only merges remain", so it must not
    survive the close that lands the last of those merges -- a closed issue
    still carrying it asserts something false, and the views it powers
    (`is:open -label:solved`, what a human should work on next) are only
    trustworthy if the strip is reliable.

    The hook sees the call's arguments, never the issue's current state, so it
    cannot check whether an issue *has* the label when `labels` is omitted.
    What it can require is that a completing close states the label set
    explicitly, which is the only form of the call that provably strips `dev`.
    """
    if str(args.get("state") or "").strip().lower() != "closed":
        return []

    raw = args.get("labels")
    labels = {str(item).strip().lower() for item in (raw or [])}

    if SOLVED_LABEL in labels:
        return [
            f'KEEPS `{SOLVED_LABEL}` ON A CLOSED ISSUE: `{SOLVED_LABEL}` means "solved, waiting only on merges". '
            "Closing this issue is the act of landing the last merge, so the label is now false. "
            "Drop it from the `labels` array."
        ]

    if raw is None and str(args.get("state_reason") or "").strip().lower() == "completed":
        return [
            f"CLOSE DOES NOT STRIP `{SOLVED_LABEL}`: a `completed` close ships the fix to `main`, so "
            f'`{SOLVED_LABEL}` ("solved, waiting only on merges") must come off in the same write.\n'
            "  Pass `labels` explicitly. NOTE: `labels` REPLACES the whole set, so list every label "
            f"the issue should keep (`claude`, `experiment`, ...) and simply omit `{SOLVED_LABEL}`. "
            "Read the issue first if you do not already know its labels -- passing `[]` would wipe them.\n"
            f"  If the issue never had `{SOLVED_LABEL}`, passing its existing labels unchanged satisfies this."
        ]

    return []


MISSING_CLAUDE = (
    "MISSING `claude`: you are filing this issue, so it is Claude-authored. "
    "Claude and humans file through the same GitHub account, so this label is "
    "the ONLY thing that keeps your issues out of the human-issue view "
    "(`is:issue is:open -label:claude`). It is never optional."
)

# The criterion, stated once and reused by all three refusals below.
#
# It deliberately does NOT say "closeable from a laptop with the test suite",
# which is what this message said until #3708. That named a configuration that
# does not exist: per #3694, `run-tests.sh` runs nowhere but the GRID, because
# the laptop has 3 GB of RAM -- so a filer applying the old wording literally
# got the wrong answer, and every issue would have read as an experiment. The
# criterion the four hand corrections actually used is narrower and true.
CRITERION = (
    "The test is whether the issue can be closed WITHOUT running the product -- "
    "an app, a dataset, an embedder, a pile cell, a submitted job."
)

MISSING_EXPERIMENT = (
    "MISSING `experiment`: this reads like an issue that cannot be closed without a run. "
    "Add the label so it lands in the queue of work that needs machine time booked.\n"
    f"  {CRITERION}\n"
    "  If closing it genuinely needs none of that, say so explicitly instead of "
    "rewording the body: add `<!-- not-an-experiment: <reason> -->`. The reason is READ, "
    "not merely counted."
)

RUN_SHAPED_REASON = (
    "MISSING `experiment`, AND THE `not-an-experiment` MARKER DESCRIBES A RUN: the reason "
    'you gave says "{phrase}", which is machine time.\n'
    f"  {CRITERION}\n"
    "  So the marker argues against itself, and the marker is not what decides this -- the "
    "closing condition is. Either add `experiment` (the usual answer when the reason reads "
    "like that), or, if the phrase is incidental and nothing has to be run, say what the "
    "closing condition IS without naming a run.\n"
    "  #3708 has the four markers that were wrong this way; each was disproved by its own text."
)

THIN_REASON = (
    '`not-an-experiment` MARKER STATES NO REASON: the marker reads "{reason}", which '
    "restates the marker's name instead of giving a reason.\n"
    f"  {CRITERION}\n"
    "  Answer that question in the marker -- name the closing condition, so the next reader "
    "can check the claim instead of taking it. If you cannot name one that avoids a run, the "
    "answer is the `experiment` label."
)


def _label_problems(labels: set[str], text: str) -> list[str]:
    """The rule itself, shared by both call paths.

    `_create_problems` and `_gh_create_problems` differ only in how they dig a
    label set and some text out of their payload. Keeping the *rule* in one
    place is what stops the two paths from drifting into two rules, which is
    how the `gh` path came to be unpoliced in the first place.
    """
    found = []

    if "claude" not in labels:
        found.append(MISSING_CLAUDE)

    if "experiment" not in labels and _looks_like_an_experiment(text):
        refusal = _opt_out_refusal(text)
        if refusal is not None:
            found.append(refusal)

    return found


def _create_problems(args: dict) -> list[str]:
    labels = {str(item).strip().lower() for item in (args.get("labels") or [])}
    body = str(args.get("body") or "")
    return _label_problems(labels, f"{args.get('title') or ''}\n{body}")


def _gh_labels(command: str) -> set[str]:
    """Every label named by a `--label`/`-l` flag anywhere in the command."""
    found = set()
    for _quote, raw in GH_LABEL_FLAG.findall(command):
        for piece in raw.split(","):
            piece = piece.strip().lower()
            if piece:
                found.add(piece)
    return found


def _gh_issue_text(command: str) -> str:
    """The text the `experiment` heuristic reads for a `gh issue create`.

    The whole command is used rather than a parsed `--title`/`--body`, because
    the bodies here are routinely heredocs (`--body "$(cat <<'EOF' ... EOF)"`)
    that no tokeniser survives -- and that heredoc text genuinely *is* part of
    the command. A `--body-file` is the one shape that puts the body out of
    reach, so it is read back when the path resolves; an unresolvable one
    (`$SP/issue.md`, a path on the GRID) simply leaves the heuristic reading
    the title, which still catches the common case.
    """
    parts = [command]
    for _quote, raw_path in GH_BODY_FILE_FLAG.findall(command):
        try:
            path = Path(raw_path)
            if path.is_file() and path.stat().st_size <= MAX_BODY_FILE_BYTES:
                parts.append(path.read_text(errors="replace"))
        except OSError:
            continue
    return "\n".join(parts)


def _runs_gh_issue_create(command: str) -> bool:
    """Is `gh issue create` actually being invoked here, or merely mentioned?"""
    if GH_NON_FILING.search(command):
        return False
    return any(GH_COMMAND_POSITION.search(command[: match.start()]) for match in GH_ISSUE_CREATE.finditer(command))


def _gh_create_problems(command: str) -> list[str]:
    """Police `gh issue create` -- the path this repo's sessions actually use."""
    if not _runs_gh_issue_create(command):
        return []
    return _label_problems(_gh_labels(command), _gh_issue_text(command))


def _gh_close_segments(command: str) -> list[str]:
    """The argument text of every `gh issue close` actually *invoked* here.

    Same command-position rule as `_runs_gh_issue_create`, for the same reason:
    a commit message, a doc line, or the body of an issue about this very rule
    all mention the string without running it, and blocking those would be a
    false block on ordinary work rather than a caught mistake.

    Each segment stops at the next command separator, so the flags parsed out of
    it belong to the close and not to whatever is chained after it.
    """
    segments = []
    for match in GH_ISSUE_CLOSE.finditer(command):
        if not GH_COMMAND_POSITION.search(command[: match.start()]):
            continue
        tail = command[match.end() :]
        cut = GH_SEGMENT_END.search(tail)
        segments.append(tail[: cut.start()] if cut else tail)
    return segments


def _removes_solved(command: str) -> bool:
    """Does the command already strip `solved` itself?

    This is the fix the denial message asks for, so recognising it is what makes
    that message actionable in one round-trip -- and it short-circuits before
    the lookup, so following the advice costs no second call to GitHub.
    """
    for _quote, raw in GH_REMOVE_LABEL_FLAG.findall(command):
        if any(piece.strip().lower() == SOLVED_LABEL for piece in raw.split(",")):
            return True
    return False


def _gh_close_target(segment: str) -> str | None:
    """The issue `gh issue close` was pointed at, or `None` if it cannot be read.

    `gh issue close` takes exactly one positional, so the answer is the one
    token that is shaped like an issue reference and is not some flag's value.
    Requiring it to be the *only* such token is what makes a misread impossible
    rather than merely unlikely: this reads a shell command without running a
    shell, so a quoted value splits into tokens the way no shell would --
    `--comment "fixed 3 bugs" 3319` offers up both `3` and `3319`, and an
    earliest-wins scan picks `3`. That is not a miss, it is a *wrong* answer:
    the hook would go on to judge this close against a different issue's labels
    and block it by name. Two candidates therefore mean "could not tell", which
    allows, in line with the contract.
    """
    candidates = []
    skip_next = False
    for token in segment.split():
        if skip_next:
            skip_next = False
            continue
        if token.startswith("-"):
            skip_next = "=" not in token and token in GH_CLOSE_VALUE_FLAGS
            continue
        target = token.strip("'\"")
        if GH_ISSUE_TARGET.match(target):
            candidates.append(target)
    return candidates[0] if len(candidates) == 1 else None


def _gh_repo(segment: str) -> str | None:
    match = GH_REPO_FLAG.search(segment)
    return match.group(2) if match else None


def _gh_lookup_timeout() -> float:
    raw = os.environ.get(GH_LOOKUP_TIMEOUT_ENV)
    try:
        timeout = float(raw) if raw else GH_LOOKUP_TIMEOUT_DEFAULT
    except ValueError:
        return GH_LOOKUP_TIMEOUT_DEFAULT
    return timeout if timeout > 0 else GH_LOOKUP_TIMEOUT_DEFAULT


def _issue_labels(target: str, repo: str | None) -> set[str] | None:
    """The issue's labels right now, or `None` for "could not tell".

    `None` is not an error path to be tightened later -- it is the contract.
    The hook's promise is that anything unexpected allows, and here "unexpected"
    covers a laptop without `gh`, an unauthenticated one, a dropped link, and a
    slow answer. Blocking on any of those would wedge a close over a network
    blip, which is a far worse failure than the stale label
    `scripts/reconcile-solved-labels.py` would go on to catch anyway.

    `gh issue view` (rather than `gh api`) is deliberate: with no `--repo` it
    resolves the repository from the working directory's git remote, exactly as
    the `gh issue close` being judged would, so the two cannot disagree about
    which issue is meant.
    """
    args = ["gh", "issue", "view", target, "--json", "labels", "--jq", ".labels[].name"]
    if repo:
        args += ["--repo", repo]
    try:
        proc = subprocess.run(  # noqa: S603,S607  # fixed argv, no shell; `gh` is resolved from PATH by design
            args, capture_output=True, text=True, timeout=_gh_lookup_timeout(), check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return {line.strip().lower() for line in proc.stdout.splitlines() if line.strip()}


def _gh_close_problems(command: str) -> list[str]:
    """Police `gh issue close` -- the path closes here actually take.

    Every close is checked, not only `--reason completed`. A bare
    `gh issue close 3319` closes as *completed* (that is GitHub's default
    `state_reason`), so a rule keyed on the literal flag would miss the shortest
    spelling of the very case it exists for. And the invariant is uniform
    anyway: CLAUDE.md says a closed issue must never carry `solved`, and the MCP
    path already blocks it on a `not_planned` close too. Scoping the lookup to
    `gh issue close` is what keeps it cheap; scoping it further would only make
    it wrong.
    """
    segments = _gh_close_segments(command)
    if not segments or _removes_solved(command):
        return []

    found: list[str] = []
    for segment in segments:
        if GH_HELP.search(segment):
            continue
        target = _gh_close_target(segment)
        if target is None:
            continue
        labels = _issue_labels(target, _gh_repo(segment))
        if labels is None or SOLVED_LABEL not in labels:
            continue
        number = GH_ISSUE_TARGET.match(target)
        problem = SOLVED_SURVIVES_CLOSE.format(number=number.group(1) or number.group(2))
        if problem not in found:
            found.append(problem)
    return found


CREATE_FOOTER = (
    "\nDecide BOTH labels now and re-issue the call, so this takes one round-trip:\n"
    "  `claude`     -> always, on every issue you file.\n"
    "  `experiment` -> only if it cannot be closed without a run (sweep, eval arm, calibration)."
)

GH_CREATE_FOOTER = (
    "\nDecide BOTH labels now and re-issue the command, so this takes one round-trip:\n"
    "  `--label claude`     -> always, on every issue you file.\n"
    "  `--label experiment` -> only if it cannot be closed without a run (sweep, eval arm, calibration).\n"
    "  `--label claude,experiment` sets both in one flag."
)

SOLVED_SURVIVES_CLOSE = (
    "CLOSE LEAVES `{solved}` ON A CLOSED ISSUE: #{{number}} carries `{solved}` right now, and this "
    'close does not take it off. `{solved}` means "solved, waiting only on merges"; closing the issue '
    "is the act of landing the last of those merges, so the label is about to become false.\n"
    "  `gh issue close` has no `--label` flag, so strip it in the same motion:\n"
    "    gh issue edit {{number}} --remove-label {solved} && <your close command>\n"
    "  Unlike the MCP path, this is not a guess: the hook asked GitHub, and the label is there."
).format(solved=SOLVED_LABEL)

CLOSE_FOOTER = (
    "\nSee docs/RELEASE.md step 6. `solved` is a transient status, not a historical fact:\n"
    "  it goes ON when the fix PR is opened, and comes OFF in the write that closes the issue."
)

GH_CLOSE_FOOTER = (
    "\nSee docs/RELEASE.md step 6. `solved` is a transient status, not a historical fact:\n"
    "  it goes ON when the fix PR is opened, and comes OFF when the issue closes.\n"
    "  While you are there, step 6 clears the assignee too: `gh issue edit <n> --remove-assignee samggreenberg`."
)


def _deny(headline: str, problems: list[str], footer: str) -> int:
    print(headline, file=sys.stderr)
    for problem in problems:
        print(f"\n  - {problem}", file=sys.stderr)
    print(footer, file=sys.stderr)
    return 2


def main() -> int:
    payload = read_payload()

    bash_args = tool_arguments(payload, BASH_TOOL, bare_keys=("command",))
    if bash_args is not None:
        command = str(bash_args.get("command") or "")
        found = _gh_create_problems(command)
        if found:
            return _deny(
                "BLOCKED: this `gh issue create` is missing a required label (CLAUDE.md, 'Label every issue you file').",
                found,
                GH_CREATE_FOOTER,
            )
        found = _gh_close_problems(command)
        if found:
            return _deny(
                f"BLOCKED: this `gh issue close` mishandles the `{SOLVED_LABEL}` label.",
                found,
                GH_CLOSE_FOOTER,
            )
        return 0

    args = tool_arguments(payload, TOOL_SUFFIX, bare_keys=("method", "repo"))
    if args is None:
        return 0

    method = str(args.get("method") or "").strip().lower()
    if method == "create":
        found, footer = _create_problems(args), CREATE_FOOTER
        headline = "BLOCKED: this issue is missing a required label (CLAUDE.md, 'Label every issue you file')."
    elif method == "update":
        found, footer = _close_problems(args), CLOSE_FOOTER
        headline = f"BLOCKED: this close mishandles the `{SOLVED_LABEL}` label."
    else:
        return 0

    if not found:
        return 0

    return _deny(headline, found, footer)


if __name__ == "__main__":
    sys.exit(main())
