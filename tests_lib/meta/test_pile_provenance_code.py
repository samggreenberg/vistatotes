"""The ``code`` block of a pile cell's provenance sidecar (#3693).

Every cell already recorded the *machine* that built it. It did not record the
**checkout**, and that is the axis that actually went wrong: ``launch_pile.sh``
built the pile from a fixed path 1,420 commits behind ``dev``, and a cell built
by that tree was indistinguishable in the sidecar from one built by the tree you
were reading. These pin the fields that make the difference legible -- which
tree, at which commit, on which branch, dirty or not -- and the launch-time
comparison that catches a worktree moving while the job sat in the queue.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_PILE_DIR = REPO_ROOT / "scripts" / "experiments" / "pile"


@pytest.fixture(scope="module")
def provenance():
    if str(_PILE_DIR) not in sys.path:
        sys.path.insert(0, str(_PILE_DIR))
    import pilebuild.provenance as mod

    return mod


def _git(cwd: Path, *args: str) -> str:
    out = subprocess.run(  # noqa: S603 - fixed argv, no shell, no user input
        ["git", *args],  # noqa: S607 - git resolved from PATH
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


@pytest.fixture
def worktree(tmp_path: Path) -> Path:
    repo = tmp_path / "checkout"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "dev", ".")
    (repo / "file.txt").write_text("one")
    _git(repo, "add", "file.txt")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "one")
    return repo


class TestCodeRecord:
    def test_records_the_checkout_the_job_imports(self, provenance, worktree, monkeypatch):
        monkeypatch.setenv("VTS_REPO", str(worktree))
        monkeypatch.delenv("VTS_LAUNCH_COMMIT", raising=False)
        rec = provenance._code_record()
        assert rec["repo"] == str(worktree)
        assert rec["commit"] == _git(worktree, "rev-parse", "HEAD")
        assert rec["branch"] == "dev"
        assert rec["dirty"] is False

    def test_uncommitted_changes_are_recorded_not_hidden(self, provenance, worktree, monkeypatch):
        monkeypatch.setenv("VTS_REPO", str(worktree))
        (worktree / "file.txt").write_text("edited")
        assert provenance._code_record()["dirty"] is True

    def test_a_tree_without_git_records_nulls_rather_than_failing(self, provenance, tmp_path, monkeypatch):
        """Provenance must never sink a build that has already produced a cell."""
        plain = tmp_path / "no-git"
        plain.mkdir()
        monkeypatch.setenv("VTS_REPO", str(plain))
        rec = provenance._code_record()
        assert rec["repo"] == str(plain)
        assert rec["commit"] is None
        assert rec["branch"] is None
        # None, not False: "git could not say" is not "clean".
        assert rec["dirty"] is None


class TestLaunchComparison:
    def test_flags_a_checkout_that_moved_while_the_job_queued(self, provenance, worktree, monkeypatch):
        monkeypatch.setenv("VTS_REPO", str(worktree))
        monkeypatch.setenv("VTS_LAUNCH_COMMIT", "0" * 40)
        rec = provenance._code_record()
        assert rec["commit_at_launch"] == "0" * 40
        assert rec["matches_launch"] is False

    def test_agrees_when_the_tree_stood_still(self, provenance, worktree, monkeypatch):
        monkeypatch.setenv("VTS_REPO", str(worktree))
        monkeypatch.setenv("VTS_LAUNCH_COMMIT", _git(worktree, "rev-parse", "HEAD"))
        assert provenance._code_record()["matches_launch"] is True

    def test_unknown_rather_than_false_when_the_launcher_stamped_nothing(self, provenance, worktree, monkeypatch):
        """A hand-run build has no launch commit; that is not a mismatch."""
        monkeypatch.setenv("VTS_REPO", str(worktree))
        monkeypatch.delenv("VTS_LAUNCH_COMMIT", raising=False)
        assert provenance._code_record()["matches_launch"] is None
