"""The deep-grid sizing gate: `harvest_headroom.py` and `preflight.sh` check 16c.

A grid is sized from its **deepest** arm, not from its shipped one (issue
#3611).  #3547 sized `vg_scale_deep` at 900 positives per class from a supply
bound and a horizon bound, neither of which is an *aggression* bound; its two
deepest arms then harvested 56% and 60% against a pre-registered 50%
compression bar, and two of its three deep contrasts were excluded.

What is pinned here is the part that has to hold when nobody is reading the
output: that the checker **fails a launch it can prove is compressed**, and —
just as important — that it says UNKNOWN rather than ok on a pilot that cannot
carry the claim.  A gate that clears a grid off a pilot it should not believe is
how #3319 read 38% off one cell and met 85% on the wave; the asymmetry (a pilot
can fail a grid without being able to clear one) is the whole design and is
asserted directly.

Meta-group: nothing here imports shipped ``vtsearch``/``vtscore`` code — the
subject is repo tooling under ``scripts/``, which has no other test or type
coverage (``pyrightconfig.json`` excludes it).
"""

from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "scripts" / "experiments" / "calibration" / "harvest_headroom.py"
_PREFLIGHT = _ROOT / "scripts" / "experiments" / "preflight.sh"

#: The identity + metric columns `run_cells.py` writes that the checker reads.
_COLUMNS = [
    "seed",
    "dataset",
    "category",
    "strategy",
    "trainer",
    "head",
    "style",
    "prevalence_arm",
    "realized_prevalence",
    "t",
    "n_good",
    "n_bad",
    "n_haystack",
]

_CATEGORIES = ("dog", "kite", "stop sign")


@pytest.fixture(scope="module")
def hh():
    spec = importlib.util.spec_from_file_location("harvest_headroom", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_cell(path: Path, seed: str, category: str, found: dict[int, int], *, haystack=3000, prevalence=0.05) -> None:
    """One cell's main frame: `found` maps step -> positives found by then."""
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, _COLUMNS)
        writer.writeheader()
        for t, n_good in sorted(found.items()):
            writer.writerow(
                {
                    "seed": seed,
                    "dataset": "vg_scale_deep",
                    "category": category,
                    "strategy": "autopilot",
                    "trainer": "app",
                    "head": "mlp",
                    "style": "whole_image",
                    "prevalence_arm": "",
                    "realized_prevalence": prevalence,
                    "t": t,
                    "n_good": n_good,
                    "n_bad": t - n_good,
                    "n_haystack": haystack,
                }
            )


def _pilot(root: Path, arms: dict[str, float], *, categories=_CATEGORIES, horizon: int = 400) -> Path:
    """A pilot wave: one cell per (arm, category), each finding `rate * t` positives."""
    base = root / "bin"
    for arm, rate in arms.items():
        cells = base / arm / "results" / "cells"
        cells.mkdir(parents=True)
        for i, category in enumerate(categories):
            found = {100: int(100 * rate * 0.7), horizon: int(horizon * rate)}
            _write_cell(cells / f"task_{i:04d}.csv", str(i), category, found)
            # A side frame beside every cell: `main_frame_files` must skip it,
            # or every count here doubles and every median moves (#3407).
            (cells / f"task_{i:04d}__picks.csv").write_text("seed,t,pick\n0,1,x\n")
    return base


def _prepare_info(root: Path, per_class: int, *, categories=_CATEGORIES) -> Path:
    path = root / "prepare_info.json"
    path.write_text(
        json.dumps(
            {
                "datasets": {
                    "vg_scale_deep": {
                        "siglip": {
                            "category_counts": {c: per_class for c in categories},
                            "selected_categories": list(categories),
                        }
                    }
                }
            }
        )
    )
    return path


def _run(hh, *args: str | Path) -> tuple[int, str]:
    """`main()` in-process, returning (exit status, the verdict line)."""
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        status = hh.main([str(a) for a in args])
    return status, buf.getvalue().strip().splitlines()[-1]


class TestVerdicts:
    def test_a_deep_arm_over_the_bar_fails_and_names_the_pile_it_needs(self, hh, tmp_path):
        """The #3547 shape: the shipped arm is fine and the deepest one is not."""
        base = _pilot(tmp_path, {"acq_m4": 0.35, "acq_m6": 0.62})
        info = _prepare_info(tmp_path, 900)
        status, line = _run(hh, "--pilot", base, "--bar", "0.5", "--horizon", "400", "--prepare-info", info)
        assert status == 1
        assert line.startswith("OVER\t")
        # The verdict is taken on the WORST arm, not on the first or the mean.
        assert "acq_m6" in line and "acq_m4" not in line
        # 248 positives at 400 clicks / 0.5 bar = 496 sim positives, and the sim
        # half is half the pile: 992 per class, against the 900 that was shipped.
        assert "992 positives per class" in line
        assert hh.size_for(248, 0.5, 0.5) == 992

    def test_the_same_pilot_clears_a_shallower_horizon(self, hh, tmp_path):
        """A long pilot sizes a short grid: it is read AT the horizon, not at its end."""
        base = _pilot(tmp_path, {"acq_m4": 0.35, "acq_m6": 0.62})
        info = _prepare_info(tmp_path, 900)
        status, line = _run(hh, "--pilot", base, "--bar", "0.5", "--horizon", "100", "--prepare-info", info)
        assert status == 0
        assert line.startswith("CLEAR\t")
        assert "t=100" in line

    def test_side_frames_are_not_counted_as_cells(self, hh, tmp_path):
        base = _pilot(tmp_path, {"acq_m6": 0.62})
        info = _prepare_info(tmp_path, 900)
        opts = hh.parse_args(["--bar", "0.5", "--horizon", "400", "--prepare-info", str(info)])
        per_category, thinnest = hh.planned_positives(info, 0.5)
        arms = hh.summarise(hh.arm_dirs(base, []), opts, thinnest, per_category)
        assert [a.n_cells for a in arms] == [len(_CATEGORIES)]

    def test_no_pilot_on_a_deep_enough_pile_needs_none(self, hh, tmp_path):
        """400 clicks cannot harvest 50% of 2000 positives however aggressive the arm."""
        status, line = _run(hh, "--bar", "0.5", "--horizon", "400", "--sim-positives", "2000")
        assert status == 0
        assert line.startswith("CLEAR\t")

    def test_no_pilot_on_a_pile_the_horizon_could_eat_is_unknown(self, hh, tmp_path):
        info = _prepare_info(tmp_path, 900)
        status, line = _run(hh, "--bar", "0.5", "--horizon", "400", "--prepare-info", info)
        assert status == 3
        assert line.startswith("UNKNOWN\t")
        assert "1600 positives per class" in line  # 400 clicks / 0.5 bar / 0.5 sim half


class TestAPilotCanFailAGridButNotClearOne:
    """The asymmetry that #3319 paid for: weak evidence still counts *against*."""

    def test_a_pilot_short_of_the_horizon_cannot_clear(self, hh, tmp_path):
        base = _pilot(tmp_path, {"acq_m6": 0.30}, horizon=100)
        info = _prepare_info(tmp_path, 900)
        status, line = _run(hh, "--pilot", base, "--bar", "0.5", "--horizon", "400", "--prepare-info", info)
        assert status == 3
        assert line.startswith("UNKNOWN\t")
        assert "FLOOR" in line

    def test_but_a_pilot_short_of_the_horizon_can_still_fail(self, hh, tmp_path):
        """Already over the bar at t=100 settles it: harvest only goes up."""
        base = _pilot(tmp_path, {"acq_m6": 0.90}, horizon=100)
        info = _prepare_info(tmp_path, 300)
        status, line = _run(hh, "--pilot", base, "--bar", "0.5", "--horizon", "400", "--prepare-info", info)
        assert status == 1
        assert line.startswith("OVER\t")

    def test_a_pilot_missing_planned_categories_cannot_clear(self, hh, tmp_path):
        """Harvest is per-category, so a pilot that skipped the thin ones proves nothing."""
        base = _pilot(tmp_path, {"acq_m6": 0.20}, categories=("dog",))
        info = _prepare_info(tmp_path, 900)
        status, line = _run(hh, "--pilot", base, "--bar", "0.5", "--horizon", "400", "--prepare-info", info)
        assert status == 3
        assert line.startswith("UNKNOWN\t")
        assert "kite" in line and "stop sign" in line

    def test_a_one_category_pilot_cannot_clear_even_without_a_prepare(self, hh, tmp_path):
        base = _pilot(tmp_path, {"acq_m6": 0.20}, categories=("dog",))
        status, line = _run(hh, "--pilot", base, "--bar", "0.5", "--horizon", "400", "--sim-positives", "900")
        assert status == 3
        assert "one category" in line

    def test_a_missing_pilot_directory_is_unknown_not_ok(self, hh, tmp_path):
        status, line = _run(hh, "--pilot", tmp_path / "nope", "--bar", "0.5", "--horizon", "400")
        assert status == 3
        assert line.startswith("UNKNOWN\t")


class TestArguments:
    @pytest.mark.parametrize(
        "bad",
        (
            ["--bar", "50", "--horizon", "400"],  # a percentage, not a fraction
            ["--bar", "0", "--horizon", "400"],
            ["--bar", "0.5", "--horizon", "0"],
            ["--bar", "0.5", "--horizon", "400", "--sim-fraction", "2"],
        ),
        ids=("percent", "zero-bar", "zero-horizon", "sim-fraction"),
    )
    def test_a_bar_or_horizon_that_cannot_mean_what_it_says_is_refused(self, hh, bad):
        with pytest.raises(SystemExit) as exc:
            hh.parse_args(bad)
        assert exc.value.code == 2

    def test_the_horizon_defaults_to_the_studys_own_declaration(self, hh, monkeypatch):
        monkeypatch.setenv("CALIB_MAX_STEPS", "400")
        monkeypatch.setenv("CALIB_SIM_FRACTION", "0.25")
        opts = hh.parse_args(["--bar", "0.5"])
        assert (opts.horizon, opts.sim_fraction) == (400, 0.25)


class TestPreflightWiring:
    """Check 16c: the verdict has to reach the launch gate, not just stdout."""

    def _preflight(self, tmp_path: Path, *args: str | Path, steps: str = "400", **env: str) -> str:
        exp = tmp_path / "exp" / "results"
        exp.mkdir(parents=True)
        (exp / "prepare_info.json").write_text((_prepare_info(tmp_path, 900)).read_text())
        proc = subprocess.run(  # noqa: S603  # fixed argv, repo-local script path, no shell
            ["bash", str(_PREFLIGHT), "--exp", str(tmp_path / "exp"), *[str(a) for a in args]],  # noqa: S607 - bash from PATH
            capture_output=True,
            text=True,
            env={
                "PATH": "/usr/bin:/bin:/usr/local/bin",
                "HOME": str(tmp_path),
                "CALIB_MAX_STEPS": steps,
                **env,
            },
        )
        return proc.stdout + proc.stderr

    def test_over_the_bar_is_a_FAIL(self, tmp_path):
        base = _pilot(tmp_path, {"acq_m4": 0.35, "acq_m6": 0.62})
        out = self._preflight(tmp_path, "--require-harvest-headroom", "0.5", "--pilot-cells", base)
        assert "FAIL  the deepest arm is over the compression bar" in out
        assert "PREFLIGHT FAILED" in out

    def test_under_the_bar_is_an_ok(self, tmp_path):
        base = _pilot(tmp_path, {"acq_m4": 0.15})
        out = self._preflight(tmp_path, "--require-harvest-headroom", "0.5", "--pilot-cells", base)
        assert "ok    the deepest arm keeps its headroom" in out

    def test_no_pilot_is_a_note_rather_than_a_refusal(self, tmp_path):
        out = self._preflight(tmp_path, "--require-harvest-headroom", "0.5")
        assert "note  harvest headroom NOT established" in out

    def test_the_check_is_opt_in(self, tmp_path):
        """A launcher that never pre-registered a bar must not meet this gate."""
        out = self._preflight(tmp_path)
        assert "harvest headroom" not in out

    def test_the_study_can_declare_the_bar_once_instead_of_per_call(self, tmp_path):
        """A launcher preflights once per arm; the declaration should not have to."""
        base = _pilot(tmp_path, {"acq_m4": 0.35, "acq_m6": 0.62})
        out = self._preflight(tmp_path, CALIB_HARVEST_BAR="0.5", CALIB_HARVEST_PILOT=str(base))
        assert "FAIL  the deepest arm is over the compression bar" in out

    def test_a_bar_without_a_horizon_fails_loudly(self, tmp_path):
        out = self._preflight(tmp_path, "--require-harvest-headroom", "0.5", steps="")
        assert "needs CALIB_MAX_STEPS" in out


def test_the_checker_is_stdlib_only(hh):
    """Check 16c runs before the venv is proven usable, so it may not need one.

    `preflight.sh` sets `PY_USABLE=0` when `import vtscore` cannot be resolved
    (a non-interactive ssh with no venv), and every python check downstream of
    that skips.  This one does not, and must not start needing to.
    """
    imported = {
        line.split()[1].split(".")[0]
        for line in _SCRIPT.read_text().splitlines()
        if line.startswith(("import ", "from ")) and "__future__" not in line
    }
    assert imported <= set(sys.stdlib_module_names) | {"_cells_paths"}, imported
