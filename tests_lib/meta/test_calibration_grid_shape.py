"""A truncated study must not keep logging the launched cell count (#3736).

``run_cells.py`` recomputes the grid from its own environment on every task, so
the denominator it logs is the one that was *launched*.  Cutting a study
mid-run -- ``scancel`` on the tail of the array, ``results/grid_shape.json``
rewritten to the reduced shape -- leaves every surviving task still naming the
old, larger number.  Job 622816 was cut from 3,750 cells to 1,875 and a peer
session read ``cell 1496/3750`` as ~40% done when the run was 76% done, and
nearly deferred a queued rebuild on that reading.

``cell_progress`` is the fix: it asks ``grid_shape.json`` -- the file
``launch_scale.sh`` writes and ``analyze_scale.py`` already reads for this exact
denominator -- and names *both* numbers when the two disagree.  The enumeration
itself stays full length, because ``idx`` indexes into it to resolve the cell.

Nothing here tests shipped ``vtsearch``/``vtscore`` behaviour, which is why it
lives in the ``meta`` group.  The calibration modules are loose scripts rather
than package members, so ``run_cells`` is loaded by path with an inert stub
``common``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

_CALIB = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "calibration"


def _load(name: str, path: Path):
    """Import one calibration script by path, without importing the package."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(path.parent))
    return module


@pytest.fixture
def run_cells(tmp_path: Path):
    """``run_cells`` with ``common.RESULTS`` pointed at an empty results dir."""
    stub: Any = types.ModuleType("common")
    stub.setup_env = lambda: None
    stub.log = lambda _msg: None
    stub.RESULTS = tmp_path
    # ``run_cells`` imports ``experiment_config`` by bare name while its own
    # directory is on ``sys.path``, so that lands in ``sys.modules`` too;
    # both are restored rather than left shadowing for the rest of the session.
    saved = {k: sys.modules.get(k) for k in ("common", "experiment_config")}
    sys.modules["common"] = stub
    try:
        yield _load("_calib_run_cells_3736", _CALIB / "run_cells.py")
    finally:
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value


def _shape(results: Path, payload: object) -> Path:
    (results / "grid_shape.json").write_text(json.dumps(payload))
    return results


# --- The truncated study, which is the whole point ---------------------------


def test_a_truncated_study_names_both_numbers(run_cells, tmp_path):
    """The #3736 case, in the numbers it actually happened with."""
    _shape(tmp_path, {"n_cells": 1875, "n_seeds": 5})
    line = run_cells.cell_progress(1496, 3750)
    assert "1496/3750" in line
    assert "1875" in line
    assert "truncated" in line


def test_the_launched_count_is_not_replaced(run_cells, tmp_path):
    """Both numbers, not the smaller one alone.

    The launched denominator is what an array element's own index is drawn
    against, so a log that dropped it would make ``cell 1496`` unreadable
    against a 1,875-cell study.  The enumeration is full length by design.
    """
    _shape(tmp_path, {"n_cells": 1875})
    assert run_cells.cell_progress(1496, 3750).startswith("cell 1496/3750 launched")


def test_a_shape_file_larger_than_the_grid_is_also_flagged(run_cells, tmp_path):
    """The mirror: a stale shape file left by a bigger, earlier grid.

    Whichever way they disagree, two files describing one study with two
    different sizes is the finding; only the wording differs.
    """
    _shape(tmp_path, {"n_cells": 6480})
    line = run_cells.cell_progress(10, 3750)
    assert "6480" in line
    assert "disagrees" in line


# --- Everything that must stay a bare ``cell i/N`` ---------------------------


def test_an_agreeing_shape_file_adds_nothing(run_cells, tmp_path):
    _shape(tmp_path, {"n_cells": 3750})
    assert run_cells.cell_progress(1496, 3750) == "cell 1496/3750"


def test_no_shape_file_adds_nothing(run_cells):
    """A cell run by hand from a bare results dir has no shape file at all."""
    assert run_cells.cell_progress(0, 12) == "cell 0/12"


@pytest.mark.parametrize(
    "payload",
    [
        {"n_seeds": 5},  # written by an older launcher, or half-written
        {"n_cells": None},
        {"n_cells": "not a number"},
        ["n_cells", 1875],  # a list indexes by int, not by str
    ],
    ids=["missing-key", "null", "not-a-number", "wrong-shape"],
)
def test_an_unusable_shape_file_never_fails_the_cell(run_cells, tmp_path, payload):
    """A log line is not worth failing a several-hour cell over."""
    _shape(tmp_path, payload)
    assert run_cells.cell_progress(7, 3750) == "cell 7/3750"


def test_malformed_json_never_fails_the_cell(run_cells, tmp_path):
    (tmp_path / "grid_shape.json").write_text("{not json")
    assert run_cells.cell_progress(7, 3750) == "cell 7/3750"


def test_an_explicit_results_dir_wins_over_the_module_default(run_cells, tmp_path):
    """``analyze``-side callers pass the run they mean, not the one in the env."""
    other = tmp_path / "elsewhere"
    other.mkdir()
    _shape(other, {"n_cells": 1875})
    assert "1875" in run_cells.cell_progress(1496, 3750, results=other)
    assert run_cells.cell_progress(1496, 3750) == "cell 1496/3750"
