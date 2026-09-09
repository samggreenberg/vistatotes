"""Tests for ``scripts/experiments/repo_stamp.sh``.

The helper is what stops a GRID launch importing a checkout nobody is looking
at. #3693: ``launch_pile.sh`` built the pile from a fixed path 1,420 commits
behind ``dev``, and the launch output never named the tree or the commit, so
there was nothing on screen to be wrong. The path default is fixed; this is the
half that keeps it fixed -- it prints the checkout and refuses one that is stale.

A shell gate is only worth having if it fails when it should, so these drive the
real script against real temp git repositories: a fresh clone, one left behind,
one that is not a checkout at all, and the deliberate-old-build escape hatch.
No SLURM and no network -- ``origin`` is a local repo on disk.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
STAMP = REPO_ROOT / "scripts" / "experiments" / "repo_stamp.sh"


def _git(cwd: Path, *args: str) -> str:
    out = subprocess.run(  # noqa: S603 - fixed argv, no shell, no user input
        ["git", *args],  # noqa: S607 - git resolved from PATH
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
        env={"PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": str(cwd), "GIT_CONFIG_GLOBAL": "/dev/null"},
    )
    return out.stdout.strip()


def _commit(repo: Path, message: str) -> None:
    (repo / "file.txt").write_text(message)
    _git(repo, "add", "file.txt")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", message)


def _run_stamp(repo: Path, **env: str) -> subprocess.CompletedProcess[str]:
    """Source the helper and call ``repo_stamp`` on ``repo``; return the result."""
    script = f'set -euo pipefail\nsource "{STAMP}"\nrepo_stamp "{repo}"\n'
    return subprocess.run(  # noqa: S603 - repo-local script path, no user input
        ["bash", "-c", script],  # noqa: S607 - bash resolved from PATH
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": str(repo), **env},
    )


@pytest.fixture
def origin_and_clone(tmp_path: Path) -> tuple[Path, Path]:
    """A bare ``origin`` with a ``dev`` branch, and a clone level with it."""
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    _git(upstream, "init", "-q", "-b", "dev", ".")
    _commit(upstream, "one")

    clone = tmp_path / "clone"
    _git(tmp_path, "clone", "-q", str(upstream), str(clone))
    return upstream, clone


class TestBanner:
    def test_names_the_checkout_and_its_commit(self, origin_and_clone):
        _, clone = origin_and_clone
        res = _run_stamp(clone)
        assert res.returncode == 0, res.stderr
        assert str(clone) in res.stdout
        assert _git(clone, "rev-parse", "--short", "HEAD") in res.stdout
        assert "level with origin/dev" in res.stdout

    def test_reports_uncommitted_changes_without_refusing(self, origin_and_clone):
        """A dirty tree is unreproducible, but it is also how the pile gets built
        while its build code is being changed. Say so; do not refuse."""
        _, clone = origin_and_clone
        (clone / "file.txt").write_text("edited")
        res = _run_stamp(clone)
        assert res.returncode == 0, res.stderr
        assert "uncommitted" in res.stdout

    def test_says_so_when_the_distance_cannot_be_measured(self, tmp_path: Path):
        """A repo with no ``origin/dev`` gets a banner and an explicit
        'UNMEASURED', never a silent pass that reads like 'level'."""
        repo = tmp_path / "solo"
        repo.mkdir()
        _git(repo, "init", "-q", "-b", "main", ".")
        _commit(repo, "one")
        res = _run_stamp(repo)
        assert res.returncode == 0, res.stderr
        assert "UNMEASURED" in res.stdout
        assert "level with" not in res.stdout


class TestStalenessGate:
    def _leave_behind(self, upstream: Path, clone: Path, n: int) -> None:
        for i in range(n):
            _commit(upstream, f"upstream {i}")
        _git(clone, "fetch", "-q", "origin")

    def test_refuses_a_checkout_past_the_gate(self, origin_and_clone):
        upstream, clone = origin_and_clone
        self._leave_behind(upstream, clone, 4)
        res = _run_stamp(clone, VTS_MAX_BEHIND="3")
        assert res.returncode == 1
        assert "4 commits behind origin/dev" in res.stdout
        assert "refusing to submit" in res.stderr

    def test_allows_a_checkout_under_the_gate_and_names_the_lag(self, origin_and_clone):
        upstream, clone = origin_and_clone
        self._leave_behind(upstream, clone, 2)
        res = _run_stamp(clone, VTS_MAX_BEHIND="3")
        assert res.returncode == 0, res.stderr
        assert "2 commits behind origin/dev" in res.stdout

    def test_zero_disables_the_refusal_but_keeps_the_banner(self, origin_and_clone):
        """Building from an old commit on purpose stays possible -- loudly."""
        upstream, clone = origin_and_clone
        self._leave_behind(upstream, clone, 4)
        res = _run_stamp(clone, VTS_MAX_BEHIND="0")
        assert res.returncode == 0, res.stderr
        assert "4 commits behind origin/dev" in res.stdout
        assert "gate disabled" in res.stdout

    def test_measures_against_the_branch_the_caller_names(self, origin_and_clone):
        upstream, clone = origin_and_clone
        _git(upstream, "branch", "release")
        self._leave_behind(upstream, clone, 4)
        # origin/release is still at the shared commit, so the same clone that
        # is 4 behind dev is level with it.
        res = _run_stamp(clone, VTS_BASE_BRANCH="release", VTS_MAX_BEHIND="3")
        assert res.returncode == 0, res.stderr
        assert "level with origin/release" in res.stdout

    def test_refuses_a_tree_that_is_not_a_checkout(self, tmp_path: Path):
        """No git means no commit stamp, which means a cell nothing can identify."""
        plain = tmp_path / "rsynced-copy"
        plain.mkdir()
        res = _run_stamp(plain)
        assert res.returncode == 1
        assert "NOT A GIT CHECKOUT" in res.stdout
        assert "refusing to submit" in res.stderr

    def test_zero_also_waives_the_not_a_checkout_refusal(self, tmp_path: Path):
        plain = tmp_path / "rsynced-copy"
        plain.mkdir()
        res = _run_stamp(plain, VTS_MAX_BEHIND="0")
        assert res.returncode == 0, res.stderr
