"""The release-window PR enumeration: `scripts/release-prs.py`.

`docs/RELEASE.md` steps 4 and 6 both walk the release window to get its PR
numbers, and both used to do it by enumerating *merge commits*. That was true
of every PR here until 2026-09-06, when four landed as GitHub squashes -- one
parent, no merge commit, invisible to `git log --merges`. Missing a PR at
release time is the expensive kind of miss: the sweep's window is
`origin/main..origin/dev`, so nothing re-examines it in a later release, and
the issues it closed stay open while their fix is live in `main`.

So the tests below pin the two properties that make the enumeration able to
see both shapes, plus the one a plausible implementation gets wrong:

* a first-parent walk finds squashes and merge commits alike, and does *not*
  descend into a branch's own commits;
* the PR number on a squash is the **trailing** `(#N)`, because GitHub appends
  its suffix after a title that usually already names the issue -- reading the
  first match enumerates issue numbers as PRs, silently and plausibly;
* a commit naming no PR (release housekeeping pushed straight to `dev`) is
  reported rather than dropped, since the other thing that lands in that
  bucket is a squash whose suffix was edited away.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "release-prs.py"


def _load_script():
    """Import the script as a module, hyphenated filename and all."""
    spec = importlib.util.spec_from_file_location("release_prs", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


release_prs = _load_script()


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(  # noqa: S603 - fixed argv, no shell, no user input
        ["git", "-c", "user.email=t@example.com", "-c", "user.name=T", *args],  # noqa: S607 - git resolved from PATH
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()


def _commit(repo: Path, name: str, message: str) -> None:
    (repo / name).write_text(f"{name}\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", message)


def _run(repo: Path, *args: str) -> tuple[int, str, str]:
    proc = subprocess.run(  # noqa: S603 - interpreter + repo-local script path, no shell
        [sys.executable, str(SCRIPT), *args],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    return proc.returncode, proc.stdout, proc.stderr


class TestSubjectParsing:
    """`pr_number` is the whole rule, so it is pinned without a repo."""

    def test_merge_commit(self) -> None:
        subject = "Merge pull request #3687 from samggreenberg/claude/vg-scale-use-register"
        assert release_prs.pr_number(subject) == 3687

    def test_squash_takes_the_trailing_number_not_the_first(self) -> None:
        """The trap: the leading `(#3673)` is the issue, `(#3685)` is the PR."""
        subject = "Write the twelve shipped classes' review rules, measured (#3673) (#3685)"
        assert release_prs.pr_number(subject) == 3685

    def test_squash_ignores_a_bare_issue_reference_mid_subject(self) -> None:
        subject = "Complete the #3588 review: all thirteen clear, and agreement is not the bar (#3671)"
        assert release_prs.pr_number(subject) == 3671

    def test_merge_shape_outvotes_a_branch_name_ending_in_a_number(self) -> None:
        """Checked first, so a branch named `...(#123)` cannot hijack the read."""
        subject = "Merge pull request #3600 from samggreenberg/claude/fix-(#123)"
        assert release_prs.pr_number(subject) == 3600

    @pytest.mark.parametrize(
        "subject",
        [
            "Refresh punch-card for the dev → main release",
            "Merge origin/main into dev to reconcile PR #3229 (accidental direct-to-main merge)",
            "Prune the #3156 pointer from the vg-scale plan",
        ],
    )
    def test_no_pr(self, subject: str) -> None:
        assert release_prs.pr_number(subject) is None


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A `dev` carrying one merge-commit PR, one squash PR, and one direct push.

    The merge-commit PR's branch holds two commits of its own plus a merge of
    `dev` back into it, none of which is a PR that shipped -- exactly the noise
    a naive `git log` (no `--first-parent`) would report as four more PRs.
    """
    r = tmp_path / "r"
    r.mkdir()
    _git(r, "init", "-q", "-b", "dev", ".")
    _commit(r, "base.txt", "Base")
    _git(r, "branch", "release-base")

    # A branch merged the historical way, with its own commits kept.
    _git(r, "checkout", "-q", "-b", "feature-a")
    _commit(r, "a1.txt", "First half of the work")
    _commit(r, "a2.txt", "Second half, and a fix for #4001")
    _git(r, "checkout", "-q", "dev")
    _commit(r, "direct.txt", "Refresh punch-card for the dev → main release")
    _git(r, "checkout", "-q", "feature-a")
    _git(r, "merge", "-q", "--no-ff", "dev", "-m", "Merge origin/dev into feature-a")
    _git(r, "checkout", "-q", "dev")
    _git(
        r,
        "merge",
        "-q",
        "--no-ff",
        "feature-a",
        "-m",
        "Merge pull request #4100 from samggreenberg/feature-a",
    )

    # A branch squashed on merge: one commit, GitHub's suffix, issue first.
    _commit(r, "squashed.txt", "Flatten the study into one commit (#4002) (#4101)")
    return r


class TestWindowEnumeration:
    def test_finds_both_shapes_and_only_them(self, repo: Path) -> None:
        code, out, _ = _run(repo, "--range", "release-base..dev")
        assert code == 0
        numbers = [int(line.split("|", 1)[0]) for line in out.splitlines()]
        assert sorted(numbers) == [4100, 4101]

    def test_counts_the_shapes_separately(self, repo: Path) -> None:
        _, _, err = _run(repo, "--range", "release-base..dev")
        assert "2 PR(s)" in err
        assert "1 merge commit(s), 1 squashed" in err

    def test_direct_push_is_reported_not_dropped(self, repo: Path) -> None:
        _, out, err = _run(repo, "--range", "release-base..dev")
        assert "Refresh punch-card" not in out, "housekeeping is not a PR"
        assert "Refresh punch-card" in err
        assert "name no PR" in err

    def test_stdout_stays_pipeable(self, repo: Path) -> None:
        """Counts and warnings go to stderr so `cut -d'|' -f1` keeps working."""
        _, out, _ = _run(repo, "--range", "release-base..dev")
        for line in out.splitlines():
            pr, sha, _subject = line.split("|", 2)
            assert pr.isdigit()
            assert len(sha) == 40

    def test_json_carries_both_buckets(self, repo: Path) -> None:
        import json

        _, out, _ = _run(repo, "--range", "release-base..dev", "--json")
        payload = json.loads(out)
        assert sorted(row["pr"] for row in payload["prs"]) == [4100, 4101]
        assert len(payload["no_pr"]) == 1

    def test_reverse_is_oldest_first(self, repo: Path) -> None:
        _, forward, _ = _run(repo, "--range", "release-base..dev")
        _, backward, _ = _run(repo, "--range", "release-base..dev", "--reverse")
        assert backward.splitlines() == list(reversed(forward.splitlines()))

    def test_bad_range_fails_loudly(self, repo: Path) -> None:
        code, _, err = _run(repo, "--range", "no-such-ref..dev")
        assert code == 1
        assert "git failed" in err
