"""Tests for the .claude/hooks/require-issue-labels.py PreToolUse gate.

The hook enforces CLAUDE.md's "Label every issue you file" rule at tool-call
time. It is the only mechanical check on that rule -- there is no CI, and
`run-tests.sh` never sees a GitHub issue -- so its two failure directions both
matter and are tested here:

* a miss (allowing an unlabeled issue) silently contaminates the human-issue
  view, which is the regression that produced issue #3127;
* a false block (rejecting an unrelated GitHub call) would wedge ordinary work,
  so every non-create, non-issue payload must pass straight through.

Both directions are tested twice over, because the rule has two call paths:
`mcp__github__issue_write` and `gh issue create` run through `Bash`. The `gh`
path went unwatched for weeks while being the only one this repo's sessions
actually used, so `TestGhCli` carries the shell shapes those sessions really
emit -- compound commands, heredoc bodies, `--body-file`, `ssh grid '...'` --
rather than a tidy flag list that would pass without proving anything.

The `experiment` half of the rule has an escape hatch, and `TestOptOutMarkerReason`
is the check that the hatch is a hatch rather than a hole: its fixtures are the
real markers off GitHub -- four that had to be corrected by hand and three that
are right -- rather than invented ones, because a pattern list tuned against
invented markers proves nothing about the mistake sessions actually make.

The `solved`-strip guard at the other end of an issue's life has the same two
paths, and the `gh` half of it (`TestGhCloseSolved`) is the one place this hook
does I/O: `gh issue close` has no `--label` flag to restate a label set with, so
the guard asks GitHub whether `solved` is really there. Testing that needs a
stubbed `gh` on `PATH` rather than a text fixture -- there is no `gh` in the
container these tests run in, so a test that only asserted "allowed" would pass
just as happily against a hook that had been deleted. Three classes split the
job: the guard itself, its failure modes (all of which must allow), and
`TestGhCloseCostsNothingOnOtherCommands`, which watches for a lookup firing on
commands that close nothing -- a cost no exit-code assertion can see.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parents[2] / ".claude" / "hooks" / "require-issue-labels.py"

ALLOW = 0
BLOCK = 2

# An experiment-shaped body: two weak signals ("calibration", "measure") and
# no strong one, so it also covers the >=2-weak-signals branch.
EXPERIMENT_BODY = "Re-run the calibration arm and measure mAP against the fold-anchored threshold."
PLAIN_BODY = "The Back button in the importer modal is left-aligned but should use .back-btn."


def run_hook(payload: dict, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(  # noqa: S603  # interpreter + repo-local hook path, no shell
        [sys.executable, str(HOOK)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )


def create(body: str = PLAIN_BODY, labels: list[str] | None = None, **overrides) -> dict:
    args: dict = {"method": "create", "owner": "samggreenberg", "repo": "vtsearch", "title": "A title", "body": body}
    if labels is not None:
        args["labels"] = labels
    args.update(overrides)
    return {"tool_name": "mcp__github__issue_write", "tool_input": args}


class TestClaudeLabel:
    """`claude` is mechanically decidable: Claude is making the call."""

    def test_create_without_any_labels_is_blocked(self):
        result = run_hook(create())
        assert result.returncode == BLOCK
        assert "MISSING `claude`" in result.stderr

    def test_create_with_unrelated_labels_is_blocked(self):
        result = run_hook(create(labels=["bug", "enhancement"]))
        assert result.returncode == BLOCK
        assert "MISSING `claude`" in result.stderr

    def test_create_with_claude_label_is_allowed(self):
        assert run_hook(create(labels=["claude"])).returncode == ALLOW

    def test_label_matching_is_case_and_whitespace_insensitive(self):
        assert run_hook(create(labels=[" Claude "])).returncode == ALLOW


class TestExperimentLabel:
    """`experiment` is a judgment call, so the hook blocks heuristically."""

    def test_experiment_shaped_body_without_the_label_is_blocked(self):
        result = run_hook(create(body=EXPERIMENT_BODY, labels=["claude"]))
        assert result.returncode == BLOCK
        assert "MISSING `experiment`" in result.stderr
        assert "MISSING `claude`" not in result.stderr

    def test_a_single_strong_signal_is_enough(self):
        body = "Add a new arm to `python -m vtscore.eval` for the region-voting path."
        result = run_hook(create(body=body, labels=["claude"]))
        assert result.returncode == BLOCK
        assert "MISSING `experiment`" in result.stderr

    def test_experiment_shaped_body_with_the_label_is_allowed(self):
        assert run_hook(create(body=EXPERIMENT_BODY, labels=["claude", "experiment"])).returncode == ALLOW

    def test_the_heuristic_reads_the_title_too(self):
        result = run_hook(create(title="Re-run the GRID sweep", body=PLAIN_BODY, labels=["claude"]))
        assert result.returncode == BLOCK
        assert "MISSING `experiment`" in result.stderr

    def test_a_single_weak_signal_does_not_block(self):
        body = "The progress bar should measure elapsed time, not step count."
        assert run_hook(create(body=body, labels=["claude"])).returncode == ALLOW

    def test_opt_out_marker_releases_the_heuristic(self):
        body = f"{EXPERIMENT_BODY}\n\n<!-- not-an-experiment: the numbers already exist in #3077 -->"
        assert run_hook(create(body=body, labels=["claude"])).returncode == ALLOW

    def test_opt_out_marker_does_not_release_the_claude_label(self):
        body = f"{EXPERIMENT_BODY}\n\n<!-- not-an-experiment: already measured -->"
        result = run_hook(create(body=body))
        assert result.returncode == BLOCK
        assert "MISSING `claude`" in result.stderr

    def test_both_problems_are_reported_together(self):
        """One round-trip must be enough to fix both labels."""
        result = run_hook(create(body=EXPERIMENT_BODY))
        assert result.returncode == BLOCK
        assert "MISSING `claude`" in result.stderr
        assert "MISSING `experiment`" in result.stderr


# The markers that were corrected by hand, as issue #3708 catalogued them. The
# first three are the marker text verbatim off GitHub; #3669's was rewritten in
# place when it was corrected, so it is reconstructed from what #3708 records
# of it ("the marker claimed `not an experiment`") -- which is why it is the
# one row here that tests the no-reason-at-all path rather than the
# run-shaped-reason path.
#
# The value of using the real ones is that they were written by sessions trying
# to be honest, not by someone inventing a bad marker for a test. A pattern
# list tuned against invented markers proves nothing; these four are the actual
# distribution of the mistake.
WRONG_MARKERS = {
    "3669": "not an experiment",
    "3683": (
        "the measurements are already done and are in the #3667 report; what is left is a "
        "provenance field and a sentence in the README. Closes with `python build_pile.py "
        "--provenance` showing `embed_batch_size` on a newly built cell, plus "
        "`./run-tests.sh meta` for the test that asserts the key is written. No GPU, no "
        "sweep. The one genuinely run-shaped question -- whether a 1e-4 perturbation changes "
        "any study's conclusion -- is deliberately NOT part of the closing condition; see "
        "the last section."
    ),
    "3693": (
        "a launcher default and a guard around it; closed by running `bash launch_pile.sh "
        "vg_scale` from a worktree and confirming the submitted job imports THAT worktree"
    ),
    "3694": (
        "closing this is `git mv`-shaped: a file move plus `bash -n` and a text test. No "
        "arms, no cells, no GPU, nothing measured; `cat` the live script and submit one cpu "
        "job to prove it."
    ),
}

# ...and the markers that are right, which is the half that stops the pattern
# list from being tuned until it matches nothing. All three are verbatim: two
# from #3708's own "must still pass" list, and one from #3708 itself -- the
# issue asking for this check was filed under a marker, so a gate that blocks
# its own filing is not shippable.
GOOD_MARKERS = {
    "3657": "one formula, four call sites, same constant; the fix and its test are a laptop change.",
    "3452": ("an investigation with external developers plus documentation; no GRID/eval run is involved."),
    "3708": (
        "a regex, a hook message and a `tests_lib/meta/` test over "
        "`.claude/hooks/require-issue-labels.py`. The hook is pure string logic with no I/O "
        "on the create path, and `tests_lib/meta/` already tests it, so the gate is its own "
        "test. Nothing is measured, nothing is submitted."
    ),
}


def marked(reason: str, body: str = EXPERIMENT_BODY) -> str:
    return f"{body}\n\n<!-- not-an-experiment: {reason} -->"


class TestOptOutMarkerReason:
    """The marker's reason is read, not merely counted (issue #3708).

    `OPT_OUT` used to match the marker's prefix only, so any text at all after
    the colon satisfied the gate. That made the escape hatch an unchecked free
    pass, and it was wrong four times in three days -- every one of them
    disproved by a sentence inside the marker itself, which is what makes it
    mechanically catchable rather than a matter of taste.

    Each fixture body pairs a real marker with `EXPERIMENT_BODY`, so the
    heuristic definitely fires and the marker is the only thing under test.
    Every issue's real body also trips the heuristic; pinning the body here
    keeps a later edit to `STRONG_SIGNALS` from quietly turning these into
    assertions about nothing.
    """

    @pytest.mark.parametrize("number", sorted(WRONG_MARKERS))
    def test_a_marker_that_describes_a_run_is_refused(self, number):
        result = run_hook(create(body=marked(WRONG_MARKERS[number]), labels=["claude"]))
        assert result.returncode == BLOCK, f"#{number}'s marker was accepted"
        assert "MISSING `experiment`" in result.stderr or "STATES NO REASON" in result.stderr

    @pytest.mark.parametrize("number", sorted(GOOD_MARKERS))
    def test_a_laptop_shaped_marker_still_releases_the_heuristic(self, number):
        result = run_hook(create(body=marked(GOOD_MARKERS[number]), labels=["claude"]))
        assert result.returncode == ALLOW, f"#{number}'s marker was refused:\n{result.stderr}"

    @pytest.mark.parametrize(
        ("number", "phrase"),
        [("3683", "newly built cell"), ("3693", "submitted job"), ("3694", "submit one cpu job")],
    )
    def test_the_denial_quotes_the_offending_phrase_back(self, number, phrase):
        """The filer wrote the argument against their own marker; show it to them."""
        result = run_hook(create(body=marked(WRONG_MARKERS[number]), labels=["claude"]))
        assert phrase in result.stderr

    def test_a_marker_restating_its_own_name_is_not_a_reason(self):
        result = run_hook(create(body=marked(WRONG_MARKERS["3669"]), labels=["claude"]))
        assert result.returncode == BLOCK
        assert "STATES NO REASON" in result.stderr

    @pytest.mark.parametrize("reason", ["", "  ", "n/a", "no", "obvious", "This is not an experiment."])
    def test_an_empty_or_token_reason_is_not_a_reason(self, reason):
        assert run_hook(create(body=marked(reason), labels=["claude"])).returncode == BLOCK

    def test_a_run_shaped_word_inside_a_denial_is_not_held_against_the_marker(self):
        """#3452 says "no GRID/eval run is involved" -- that is the marker working."""
        for reason in ("no sweep is involved", "closes without any sbatch job", "nothing is submitted"):
            body = marked(f"one constant, four call sites; {reason}.")
            assert run_hook(create(body=body, labels=["claude"])).returncode == ALLOW, reason

    def test_a_reason_containing_an_angle_bracket_is_still_read_as_a_marker(self):
        """`[^>]*` would stop at the `>` and report the marker as missing.

        That is the one denial a filer who wrote a marker cannot act on: the
        message asks for the thing already on the screen.
        """
        body = marked("one call site; the fix is `a > b` and its test, both a laptop change.")
        assert run_hook(create(body=body, labels=["claude"])).returncode == ALLOW

    def test_a_multi_line_marker_is_read_whole(self):
        body = marked("closes by\nsubmitting one cpu job\non the GRID")
        result = run_hook(create(body=body, labels=["claude"]))
        assert result.returncode == BLOCK

    def test_a_fenced_example_of_the_syntax_does_not_block_a_real_marker(self):
        """#3708's own body carries both, and it had to be fileable."""
        body = f"{marked(GOOD_MARKERS['3708'])}\n\n```\n<!-- not-an-experiment: <reason> -->\n```"
        assert run_hook(create(body=body, labels=["claude"])).returncode == ALLOW

    def test_the_label_still_beats_the_marker(self):
        """A wrong marker on an issue that carries `experiment` is nobody's problem."""
        body = marked(WRONG_MARKERS["3693"])
        assert run_hook(create(body=body, labels=["claude", "experiment"])).returncode == ALLOW

    def test_a_refused_marker_does_not_swallow_the_claude_problem(self):
        result = run_hook(create(body=marked(WRONG_MARKERS["3694"])))
        assert result.returncode == BLOCK
        assert "MISSING `claude`" in result.stderr

    def test_the_criterion_names_running_the_product_not_a_laptop_test_suite(self):
        """#3694: `run-tests.sh` runs nowhere but the GRID, so the old wording was unusable."""
        result = run_hook(create(body=EXPERIMENT_BODY, labels=["claude"]))
        assert "WITHOUT running the product" in result.stderr
        assert "laptop with the test suite" not in result.stderr

    def test_the_gh_path_reads_the_reason_too(self):
        """Both call paths share `_label_problems`; prove it rather than assume it."""
        command = f'gh issue create --title "T" --body "{marked(WRONG_MARKERS["3693"])}" --label claude'
        payload = {"tool_name": "Bash", "tool_input": {"command": command}}
        assert run_hook(payload).returncode == BLOCK


class TestSolvedLabelOnClose:
    """`solved` means "development done, only merges remain", so a close must strip it.

    The hook only ever sees the call's arguments, never the issue's current
    state, so the enforceable form is "a completing close must state its label
    set explicitly" -- that being the only shape of the call that provably
    strips the label.
    """

    @staticmethod
    def close(**overrides) -> dict:
        args = {"method": "update", "owner": "samggreenberg", "repo": "vtsearch", "issue_number": 3077}
        args.update(overrides)
        return {"tool_name": "mcp__github__issue_write", "tool_input": args}

    def test_completed_close_carrying_solved_is_blocked(self):
        result = run_hook(self.close(state="closed", state_reason="completed", labels=["claude", "solved"]))
        assert result.returncode == BLOCK
        assert "KEEPS `solved` ON A CLOSED ISSUE" in result.stderr

    def test_completed_close_without_explicit_labels_is_blocked(self):
        result = run_hook(self.close(state="closed", state_reason="completed"))
        assert result.returncode == BLOCK
        assert "CLOSE DOES NOT STRIP `solved`" in result.stderr

    def test_the_denial_warns_that_labels_replaces_the_whole_set(self):
        """Passing `[]` to satisfy the hook would silently wipe `claude`."""
        result = run_hook(self.close(state="closed", state_reason="completed"))
        assert "REPLACES the whole set" in result.stderr

    def test_completed_close_with_labels_minus_solved_is_allowed(self):
        assert run_hook(self.close(state="closed", state_reason="completed", labels=["claude"])).returncode == ALLOW

    def test_an_issue_with_no_labels_left_can_still_be_closed(self):
        assert run_hook(self.close(state="closed", state_reason="completed", labels=[])).returncode == ALLOW

    def test_solved_is_blocked_on_any_close_not_just_completed(self):
        """A not_planned close carrying `solved` is just as false a statement."""
        result = run_hook(self.close(state="closed", state_reason="not_planned", labels=["claude", "solved"]))
        assert result.returncode == BLOCK
        assert "KEEPS `solved` ON A CLOSED ISSUE" in result.stderr

    def test_non_completed_close_does_not_require_explicit_labels(self):
        """Only the release sweep must strip; a not_planned close need not restate labels."""
        assert run_hook(self.close(state="closed", state_reason="not_planned")).returncode == ALLOW

    def test_solved_label_matching_is_case_insensitive(self):
        result = run_hook(self.close(state="closed", state_reason="completed", labels=["claude", "SOLVED"]))
        assert result.returncode == BLOCK

    def test_create_path_rules_do_not_leak_onto_closes(self):
        """A close needs no `claude` label -- the issue may well be a human's."""
        assert (
            run_hook(self.close(state="closed", state_reason="completed", labels=["enhancement"])).returncode == ALLOW
        )


class TestGhCli:
    """`gh issue create` through Bash -- the path the sessions here actually take.

    The shapes below are taken from real transcripts, not invented: bodies
    arrive as `$(cat <<'EOF' ... EOF)` heredocs, commands are chained with
    `&&`, some run inside `ssh grid '...'`, and some pass `--body-file` for a
    body that never appears in the command at all.
    """

    @staticmethod
    def bash(command: str) -> dict:
        return {"tool_name": "Bash", "tool_input": {"command": command}}

    def run(self, command: str) -> subprocess.CompletedProcess:
        return run_hook(self.bash(command))

    def test_unlabeled_create_is_blocked(self):
        result = self.run(f'gh issue create --title "A title" --body "{PLAIN_BODY}"')
        assert result.returncode == BLOCK
        assert "MISSING `claude`" in result.stderr

    def test_the_denial_names_the_flag_not_the_api_field(self):
        """The caller is holding a shell command; `labels: [...]` is no help to them."""
        result = self.run('gh issue create --title "A title" --body "x"')
        assert "--label claude" in result.stderr

    def test_claude_label_is_enough_for_a_plain_issue(self):
        assert self.run(f'gh issue create --title "T" --body "{PLAIN_BODY}" --label claude').returncode == ALLOW

    @pytest.mark.parametrize(
        "flag",
        ["--label claude", "--label=claude", "-l claude", '--label "claude"', "--label 'claude'"],
        ids=["space", "equals", "short", "double-quoted", "single-quoted"],
    )
    def test_every_flag_spelling_is_recognised(self, flag):
        assert self.run(f'gh issue create --title "T" --body "{PLAIN_BODY}" {flag}').returncode == ALLOW

    @pytest.mark.parametrize(
        "flag",
        ["--label claude,experiment", '--label "claude, experiment"', "--label claude --label experiment"],
        ids=["comma", "comma-spaced-quoted", "repeated"],
    )
    def test_both_labels_in_every_combining_form(self, flag):
        assert self.run(f'gh issue create --title "T" --body "{EXPERIMENT_BODY}" {flag}').returncode == ALLOW

    def test_experiment_shaped_body_without_the_label_is_blocked(self):
        result = self.run(f'gh issue create --title "T" --body "{EXPERIMENT_BODY}" --label claude')
        assert result.returncode == BLOCK
        assert "MISSING `experiment`" in result.stderr
        assert "MISSING `claude`" not in result.stderr

    def test_a_heredoc_body_is_read_by_the_heuristic(self):
        """The real shape: the body is a heredoc inside the command string."""
        command = (
            'gh issue create --repo samggreenberg/VTSearch --title "Widen the sweep" '
            f"--body \"$(cat <<'EOF'\n{EXPERIMENT_BODY}\nEOF\n)\" --label claude"
        )
        result = self.run(command)
        assert result.returncode == BLOCK
        assert "MISSING `experiment`" in result.stderr

    def test_a_body_file_on_disk_is_read_back(self, tmp_path):
        body = tmp_path / "issue_body.md"
        body.write_text(EXPERIMENT_BODY)
        result = self.run(f'gh issue create --title "T" --body-file {body} --label claude')
        assert result.returncode == BLOCK
        assert "MISSING `experiment`" in result.stderr

    def test_a_body_file_that_does_not_resolve_still_enforces_claude(self):
        """`$SP/issue.md` and GRID-side paths cannot be read; the label rule still binds."""
        result = self.run('gh issue create --title "T" --body-file $SP/issue.md')
        assert result.returncode == BLOCK
        assert "MISSING `claude`" in result.stderr

    def test_a_body_file_that_does_not_resolve_does_not_crash_the_heuristic(self):
        assert self.run('gh issue create --title "T" --body-file $SP/issue.md --label claude').returncode == ALLOW

    def test_a_compound_command_is_still_policed(self):
        command = f'gh issue create --title "T" --body "{PLAIN_BODY}"'
        assert self.run(command).returncode == BLOCK

    @pytest.mark.parametrize(
        "prefix",
        ["", "timeout 120 ", "cd /tmp && ", "nohup ", "cat body.md | "],
        ids=["bare", "timeout", "chained", "nohup", "piped"],
    )
    def test_every_invocation_shape_is_policed(self, prefix):
        assert self.run(f'{prefix}gh issue create --title "T" --body "{PLAIN_BODY}"').returncode == BLOCK

    def test_a_create_after_a_heredoc_on_its_own_line_is_policed(self):
        command = f"cat > /tmp/body.md <<'EOF'\n{PLAIN_BODY}\nEOF\ngh issue create --title \"T\" -F /tmp/body.md"
        assert self.run(command).returncode == BLOCK

    def test_a_create_inside_ssh_quotes_is_still_policed(self):
        command = 'timeout 300 ssh grid \'cd /exp/sgreenberg/projects/vts && gh issue create --title "T" --body "x"\''
        assert self.run(command).returncode == BLOCK

    def test_add_label_does_not_satisfy_a_create(self):
        """A follow-up edit is not a label at creation time -- and often never happens."""
        command = f'gh issue create --title "T" --body "{PLAIN_BODY}" && gh issue edit 1 --add-label claude'
        result = self.run(command)
        assert result.returncode == BLOCK
        assert "MISSING `claude`" in result.stderr

    def test_opt_out_marker_works_from_the_command_line_too(self):
        body = f"{EXPERIMENT_BODY} <!-- not-an-experiment: measured already in #3077 -->"
        assert self.run(f'gh issue create --title "T" --body "{body}" --label claude').returncode == ALLOW

    @pytest.mark.parametrize(
        "command",
        [
            "gh issue create --help",
            "gh issue create -h",
            "gh issue create --web",
            'gh issue create --web --title "T"',
        ],
        ids=["help", "short-help", "web", "web-with-title"],
    )
    def test_non_filing_invocations_are_not_policed(self, command):
        """`--help` files nothing, and is what you type *after* the denial message.

        Blocking it turns the denial into a dead end -- the hook tells you to
        add `--label` and then refuses to let you look up the flag. `--web`
        hands the form to a browser, where the CLI's flags do not apply.
        """
        assert self.run(command).returncode == ALLOW

    def test_the_smoke_probe_shape_is_blocked(self):
        """`false && gh issue create ...` is the live-session probe for this hook.

        The lesson this change exists to fix is that a gate is worth nothing
        until it has been observed to fire. This shape is the honest way to
        watch it: the hook sees a command-position `gh issue create` and blocks,
        and if the hook were ever broken the shell would still run nothing,
        because `false &&` short-circuits. No issue can be filed either way.
        """
        assert self.run('false && gh issue create --title "probe" --body "probe"').returncode == BLOCK

    def test_bare_bash_payload_is_still_policed(self):
        result = run_hook({"command": f'gh issue create --title "T" --body "{PLAIN_BODY}"'})
        assert result.returncode == BLOCK
        assert "MISSING `claude`" in result.stderr


class TestGhPassthrough:
    """A Bash hook sees every command in the session, so it must be near-silent."""

    @staticmethod
    def run(command: str) -> subprocess.CompletedProcess:
        return run_hook({"tool_name": "Bash", "tool_input": {"command": command}})

    @pytest.mark.parametrize(
        "command",
        [
            "./run-tests.sh",
            "python -m pytest tests_lib/meta -q",
            "git commit -m 'gh issue create is mentioned only in this message'",
            "gh issue view 3127 --json labels",
            "gh issue list --label claude",
            "gh issue comment 3127 --body 'Addressed in #3130'",
            "gh issue edit 3127 --add-label solved",
            "gh pr create --base dev --title 'T' --body 'B'",
            "echo 'gh issue created earlier today'",
        ],
        ids=[
            "run-tests",
            "pytest",
            "commit-message",
            "issue-view",
            "issue-list",
            "issue-comment",
            "issue-edit",
            "pr-create",
            "prose-mention",
        ],
    )
    def test_unrelated_commands_pass_straight_through(self, command):
        assert self.run(command).returncode == ALLOW

    @pytest.mark.parametrize(
        "command",
        [
            "git commit -m 'Teach the hook about gh issue create'",
            'gh pr create --title "T" --body "This makes gh issue create enforce labels"',
            "grep -rn 'gh issue create' docs/",
            'echo "run gh issue create with --label claude" >> docs/RELEASE.md',
            "# gh issue create --title 'T' --body 'B'",
            "echo hi\n# gh issue create --title 'T' --body 'B'",
            "git commit -m 'Police `gh issue create`, the unwatched path'",
            "python3 -c \"print('`gh issue create` needs --label')\"",
        ],
        ids=[
            "commit-message",
            "pr-body",
            "grep",
            "doc-line",
            "commented-out",
            "commented-out-line",
            "backticked-name",
            "backticked-in-heredoc",
        ],
    )
    def test_merely_mentioning_the_command_is_not_running_it(self, command):
        """The hook must not block work *about* the rule -- including this PR.

        Every one of these is a real command from the change that added this
        test. A hook that blocked them would make its own repo unworkable.

        The two backticked cases are a regression: a bare backtick used to count
        as command position, so markdown-quoting the command name -- in a commit
        message, a PR body, a docstring, a line of CLAUDE.md -- was read as
        running it. That false-blocked an ordinary edit in the first session
        after this hook shipped, which is why the backtick is no longer a
        separator (see `GH_COMMAND_POSITION`).
        """
        assert self.run(command).returncode == ALLOW

    def test_a_close_of_an_unlabeled_issue_costs_nothing(self):
        """`gh issue close` is policed by `TestGhCloseSolved`, but only via a lookup.

        With no `gh` on PATH the lookup cannot answer, and "could not tell"
        always allows -- so a close is never blocked on a hunch.
        """
        assert self.run("gh issue close 3319 --reason completed").returncode == ALLOW


class TestGhCloseSolved:
    """`gh issue close` -- the path closes here actually take (issue #3634).

    The MCP rule is "restate the label set, minus `solved`", and
    `gh issue close` has no `--label` flag to restate anything with. So this
    path asks GitHub instead: one `gh issue view --json labels` lookup, and a
    block only when `solved` is *provably* on the issue.

    Every test below stubs `gh` on `PATH`. That is not a convenience -- it is
    the only way to exercise the blocking branch at all, since the check is a
    subprocess call rather than a text rule, and a test that merely asserted
    "allowed" against a machine with no `gh` would pass just as happily against
    a hook that had been deleted.
    """

    @staticmethod
    def stub_gh(tmp_path, script: str) -> dict:
        """A `gh` on `PATH` that answers the label lookup however we like."""
        bin_dir = tmp_path / "stub-bin"
        bin_dir.mkdir(exist_ok=True)
        exe = bin_dir / "gh"
        exe.write_text(script)
        exe.chmod(0o755)
        return {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}

    @classmethod
    def labelled(cls, tmp_path, *labels: str) -> dict:
        emitted = "".join(f"{label}\\n" for label in labels)
        return cls.stub_gh(tmp_path, f"#!/bin/sh\nprintf '{emitted}'\n")

    @staticmethod
    def run(command: str, env: dict) -> subprocess.CompletedProcess:
        return run_hook({"tool_name": "Bash", "tool_input": {"command": command}}, env=env)

    def solved(self, tmp_path) -> dict:
        return self.labelled(tmp_path, "claude", "solved")

    def test_a_close_that_would_leave_solved_behind_is_blocked(self, tmp_path):
        result = self.run("gh issue close 3319 --reason completed", self.solved(tmp_path))
        assert result.returncode == BLOCK
        assert "CLOSE LEAVES `solved` ON A CLOSED ISSUE" in result.stderr

    def test_the_denial_names_the_gh_fix_not_a_labels_array(self, tmp_path):
        """The caller is holding a shell command, and `gh issue close` has no `--label`."""
        result = self.run("gh issue close 3319 --reason completed", self.solved(tmp_path))
        assert "gh issue edit 3319 --remove-label solved" in result.stderr

    def test_the_denial_names_the_issue(self, tmp_path):
        result = self.run("gh issue close 3257 --reason completed", self.solved(tmp_path))
        assert "#3257" in result.stderr

    def test_a_bare_close_is_policed_too(self, tmp_path):
        """`gh issue close N` with no `--reason` closes as *completed*.

        A rule keyed on the literal `--reason completed` would miss the shortest
        spelling of exactly the case it exists for.
        """
        assert self.run("gh issue close 3319", self.solved(tmp_path)).returncode == BLOCK

    def test_a_not_planned_close_carrying_solved_is_blocked(self, tmp_path):
        """A closed issue must never carry `solved`, whatever the reason.

        The MCP path already blocks it on a `not_planned` close; the invariant
        is about the label being false, not about which sweep did the closing.
        """
        assert self.run("gh issue close 3257 --reason not_planned", self.solved(tmp_path)).returncode == BLOCK

    def test_an_issue_that_never_carried_solved_closes_freely(self, tmp_path):
        """The majority case: the guard must not turn an ordinary close into a dance."""
        env = self.labelled(tmp_path, "claude", "enhancement")
        assert self.run("gh issue close 3319 --reason completed", env).returncode == ALLOW

    def test_an_issue_with_no_labels_at_all_closes_freely(self, tmp_path):
        assert self.run("gh issue close 3319", self.labelled(tmp_path)).returncode == ALLOW

    def test_the_lookup_answer_is_matched_case_insensitively(self, tmp_path):
        env = self.labelled(tmp_path, "Claude", "SOLVED")
        assert self.run("gh issue close 3319", env).returncode == BLOCK

    @pytest.mark.parametrize(
        "command",
        [
            "gh issue edit 3319 --remove-label solved && gh issue close 3319 --reason completed",
            "gh issue close 3319 --reason completed && gh issue edit 3319 --remove-label solved",
            "gh issue edit 3319 --remove-label claude,solved && gh issue close 3319",
            "gh issue edit 3319 --remove-label='solved' && gh issue close 3319",
        ],
        ids=["fix-first", "fix-after", "comma-list", "quoted-equals"],
    )
    def test_chaining_the_fix_satisfies_the_guard(self, tmp_path, command):
        """Following the denial message must work in one round-trip."""
        assert self.run(command, self.solved(tmp_path)).returncode == ALLOW

    def test_removing_some_other_label_does_not_satisfy_the_guard(self, tmp_path):
        command = "gh issue edit 3319 --remove-label claude && gh issue close 3319"
        assert self.run(command, self.solved(tmp_path)).returncode == BLOCK

    def test_a_close_chained_after_the_addressed_in_comment_is_policed(self, tmp_path):
        """The real shape from docs/RELEASE.md step 6."""
        command = "gh issue comment 3319 --body 'Addressed in #3320' && gh issue close 3319 --reason completed"
        assert self.run(command, self.solved(tmp_path)).returncode == BLOCK

    def test_two_closes_in_one_command_are_both_named(self, tmp_path):
        command = "gh issue close 3319 --reason completed && gh issue close 3257 --reason completed"
        result = self.run(command, self.solved(tmp_path))
        assert result.returncode == BLOCK
        assert "#3319" in result.stderr
        assert "#3257" in result.stderr

    @pytest.mark.parametrize(
        "target",
        ["3319", "#3319", "'3319'", "https://github.com/samggreenberg/VTSearch/issues/3319"],
        ids=["number", "hash", "quoted", "url"],
    )
    def test_every_target_spelling_is_understood(self, tmp_path, target):
        result = self.run(f"gh issue close {target} --reason completed", self.solved(tmp_path))
        assert result.returncode == BLOCK
        assert "#3319" in result.stderr

    def test_a_numeric_flag_value_is_not_mistaken_for_the_issue(self, tmp_path):
        """`--comment 3` swallows its value; the issue is the bare token after it."""
        result = self.run("gh issue close --comment 3 3319 --reason completed", self.solved(tmp_path))
        assert result.returncode == BLOCK
        assert "#3319" in result.stderr

    def test_gh_s_own_spelling_of_not_planned_is_understood(self, tmp_path):
        """`gh` wants `--reason "not planned"` -- with a space, so it splits in two."""
        result = self.run('gh issue close --reason "not planned" 3257', self.solved(tmp_path))
        assert result.returncode == BLOCK
        assert "#3257" in result.stderr

    def test_a_number_inside_a_quoted_flag_value_never_wins(self, tmp_path):
        """The failure that matters here is a *wrong* answer, not a missed one.

        This hook reads a shell command without running a shell, so
        `--comment "fixed 3 bugs"` offers up a bare `3` that no shell would.
        An earliest-wins scan takes it, looks up issue #3, and blocks a close of
        #3319 while naming #3 -- a false block against an issue the command
        never mentioned. Two candidates mean "could not tell", which allows.
        """
        command = 'gh issue close --comment "fixed 3 bugs" 3319 --reason completed'
        assert self.run(command, self.solved(tmp_path)).returncode == ALLOW

    def test_an_unreadable_target_allows(self, tmp_path):
        """A shell variable is not something this hook can resolve, so it does not try."""
        assert self.run("gh issue close $ISSUE --reason completed", self.solved(tmp_path)).returncode == ALLOW

    @pytest.mark.parametrize(
        "flag", ["--repo samggreenberg/VTSearch", "--repo=samggreenberg/VTSearch", "-R samggreenberg/VTSearch"]
    )
    def test_the_repo_flag_is_forwarded_to_the_lookup(self, tmp_path, flag):
        """The lookup must ask about the same issue the close would act on."""
        env = self.stub_gh(
            tmp_path,
            '#!/bin/sh\ncase " $* " in *" --repo samggreenberg/VTSearch "*) printf "solved\\n" ;; *) exit 1 ;; esac\n',
        )
        assert self.run(f"gh issue close 3319 {flag}", env).returncode == BLOCK

    def test_a_repo_flag_on_a_chained_command_is_not_read_as_the_close_s(self, tmp_path):
        """Each close is parsed from its own argument segment, not the whole line."""
        env = self.stub_gh(
            tmp_path, '#!/bin/sh\ncase " $* " in *" --repo "*) exit 1 ;; *) printf "solved\\n" ;; esac\n'
        )
        command = "gh issue close 3319 && gh issue list --repo other/repo"
        assert self.run(command, env).returncode == BLOCK

    def test_help_is_never_a_close(self, tmp_path):
        assert self.run("gh issue close --help", self.solved(tmp_path)).returncode == ALLOW

    @pytest.mark.parametrize(
        "command",
        [
            "git commit -m 'Teach the hook about gh issue close'",
            "grep -rn 'gh issue close 3319' docs/",
            "echo 'see `gh issue close 3319` for the shape'",
            "# gh issue close 3319 --reason completed",
        ],
        ids=["commit-message", "grep", "backticked-mention", "commented-out"],
    )
    def test_merely_mentioning_a_close_is_not_running_one(self, tmp_path, command):
        assert self.run(command, self.solved(tmp_path)).returncode == ALLOW


class TestGhCloseLookupFailures:
    """ "Could not tell" always allows -- that is the contract, not a gap to tighten.

    A hook runs in front of the user's command. Blocking a close because GitHub
    was slow, or because this laptop has no `gh`, would wedge ordinary work over
    a network blip -- a far worse failure than the stale label that
    `scripts/reconcile-solved-labels.py` goes on to catch anyway.
    """

    CLOSE = "gh issue close 3319 --reason completed"

    @staticmethod
    def run(env: dict) -> subprocess.CompletedProcess:
        payload = {"tool_name": "Bash", "tool_input": {"command": TestGhCloseLookupFailures.CLOSE}}
        return run_hook(payload, env=env)

    def test_no_gh_on_path_allows(self, tmp_path):
        """The container this repo's web sessions run in has no `gh` at all."""
        assert self.run({**os.environ, "PATH": str(tmp_path)}).returncode == ALLOW

    def test_an_unauthenticated_or_offline_gh_allows(self, tmp_path):
        env = TestGhCloseSolved.stub_gh(tmp_path, "#!/bin/sh\necho 'gh: not authenticated' >&2\nexit 1\n")
        assert self.run(env).returncode == ALLOW

    def test_a_slow_gh_is_abandoned_rather_than_waited_out(self, tmp_path):
        env = TestGhCloseSolved.stub_gh(tmp_path, "#!/bin/sh\nsleep 30\nprintf 'solved\\n'\n")
        env["VTSEARCH_GH_LOOKUP_TIMEOUT"] = "0.5"
        assert self.run(env).returncode == ALLOW

    def test_an_unparseable_timeout_override_falls_back_to_the_default(self, tmp_path):
        env = TestGhCloseSolved.labelled(tmp_path, "solved")
        env["VTSEARCH_GH_LOOKUP_TIMEOUT"] = "not a number"
        assert self.run(env).returncode == BLOCK

    def test_a_silent_gh_allows(self, tmp_path):
        """No labels printed is a legitimate answer: the issue has none."""
        env = TestGhCloseSolved.stub_gh(tmp_path, "#!/bin/sh\nexit 0\n")
        assert self.run(env).returncode == ALLOW


class TestGhCloseCostsNothingOnOtherCommands:
    """The Bash matcher sees every command, so the lookup must fire on closes alone.

    This is the half of "do not make ordinary work harder" that an exit-code
    assertion cannot show: a hook that shelled out to GitHub on every `git
    status` would still return 0 every time, and every other test here would
    still pass.
    """

    @staticmethod
    def _tattling_gh(tmp_path):
        marker = tmp_path / "gh-was-called"
        bin_dir = tmp_path / "stub-bin"
        bin_dir.mkdir(exist_ok=True)
        exe = bin_dir / "gh"
        exe.write_text(f"#!/bin/sh\ntouch {marker}\nprintf 'solved\\n'\n")
        exe.chmod(0o755)
        return marker, {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}

    @pytest.mark.parametrize(
        "command",
        [
            "./run-tests.sh",
            "git status",
            "gh issue view 3319 --json labels",
            "gh issue list --label solved",
            "gh issue edit 3319 --add-label solved",
            "gh pr create --base dev --title 'T' --body 'B'",
            "gh issue create --title 'T' --body 'B' --label claude",
            "gh issue close --help",
            "git commit -m 'gh issue close is mentioned only here'",
        ],
        ids=[
            "run-tests",
            "git-status",
            "issue-view",
            "issue-list",
            "issue-edit",
            "pr-create",
            "issue-create",
            "close-help",
            "mention",
        ],
    )
    def test_no_lookup_happens(self, tmp_path, command):
        marker, env = self._tattling_gh(tmp_path)
        assert run_hook({"tool_name": "Bash", "tool_input": {"command": command}}, env=env).returncode == ALLOW
        assert not marker.exists(), "the hook called GitHub for a command that closes nothing"

    def test_but_a_real_close_does_look_up(self, tmp_path):
        """The control: without this, the assertions above would pass on a dead guard."""
        marker, env = self._tattling_gh(tmp_path)
        assert run_hook({"tool_name": "Bash", "tool_input": {"command": "gh issue close 3319"}}, env=env).returncode
        assert marker.exists()

    def test_and_chaining_the_fix_skips_the_lookup(self, tmp_path):
        """Following the denial costs no second call to GitHub."""
        marker, env = self._tattling_gh(tmp_path)
        command = "gh issue edit 3319 --remove-label solved && gh issue close 3319"
        assert run_hook({"tool_name": "Bash", "tool_input": {"command": command}}, env=env).returncode == ALLOW
        assert not marker.exists()


class TestPassthrough:
    """A hook that fails closed would wedge unrelated GitHub work."""

    def test_non_close_updates_are_never_blocked(self):
        """Relabeling an existing issue -- including a human's -- must pass.

        Only a *close* is policed on the update path; an edit that does not
        touch `state` is none of the hook's business, and the create-path
        label rules must not leak onto it.
        """
        payload = create(body=EXPERIMENT_BODY, method="update", issue_number=3127)
        payload["tool_input"].pop("labels", None)
        assert run_hook(payload).returncode == ALLOW

    def test_reopening_is_not_policed(self):
        payload = create(method="update", issue_number=3127, state="open")
        payload["tool_input"].pop("labels", None)
        assert run_hook(payload).returncode == ALLOW

    def test_other_tools_are_ignored(self):
        payload = {"tool_name": "mcp__github__create_pull_request", "tool_input": {"method": "create", "repo": "x"}}
        assert run_hook(payload).returncode == ALLOW

    @pytest.mark.parametrize(
        "raw",
        ["", "   ", "not json at all", "[]", "null", '"a string"'],
        ids=["empty", "blank", "garbage", "list", "null", "string"],
    )
    def test_unparseable_payloads_allow(self, raw):
        result = subprocess.run(  # noqa: S603  # interpreter + repo-local hook path, no shell
            [sys.executable, str(HOOK)], input=raw, capture_output=True, text=True, timeout=30
        )
        assert result.returncode == ALLOW

    def test_bare_argument_payload_is_still_policed(self):
        """Some harness versions pass the arguments dict without an envelope."""
        result = run_hook({"method": "create", "owner": "samggreenberg", "repo": "vtsearch", "body": PLAIN_BODY})
        assert result.returncode == BLOCK
        assert "MISSING `claude`" in result.stderr


class TestWiring:
    """The hook is inert unless settings.json actually points at it."""

    def test_hook_file_exists(self):
        assert HOOK.is_file()

    @staticmethod
    def _matcher(name: str) -> dict:
        settings = json.loads((HOOK.parents[1] / "settings.json").read_text())
        return next(h for h in settings["hooks"]["PreToolUse"] if h["matcher"] == name)

    def test_registered_as_a_pretooluse_hook_for_issue_write(self):
        entry = self._matcher("mcp__github__issue_write")
        assert any(HOOK.name in hook["command"] for hook in entry["hooks"])

    def test_registered_as_a_pretooluse_hook_for_bash(self):
        """The `gh` path is the one this repo's sessions actually use.

        Registering the hook only for the MCP tool is what let 29 unlabeled
        issues through: the file existed, the tests passed, and the matcher
        named a tool nobody was calling.
        """
        entry = self._matcher("Bash")
        assert any(HOOK.name in hook["command"] for hook in entry["hooks"])

    def test_bash_registration_keeps_the_dep_gate(self):
        """Two hooks share the Bash matcher; adding one must not evict the other."""
        entry = self._matcher("Bash")
        assert any("ensure-test-deps-gate" in hook["command"] for hook in entry["hooks"])
