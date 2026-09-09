"""Will the grid's DEEPEST arm still have harvest headroom at its horizon?

**Size a deep grid from its deepest arm, not from its shipped one** (issue
#3611).  Harvest is the share of the simulation half's positives an arm has
already found; an arm that has consumed most of them cannot show a late gain,
because its tail is capped by the pool rather than by the knob under study.  A
pre-registered *compression bar* (#3547 used a median harvest of 50%) is what
decides which of a grid's contrasts are readable at all, and it is set by
**aggression**: the more aggressive the acquisition arm, the more of the pile it
eats over the same horizon.

#3547 sized `vg_scale_deep` at 900 positives per class from two bounds that are
both real and neither of which is an aggression bound — what the twelve classes
could supply, and what a 400-click session can find (`preflight.sh` check 16b).
Its two deepest arms then harvested 56% and 60% against a 50% bar, so two of its
three deep contrasts were excluded as compressed and one survived.  Compression
is **one-sided** — a capped tail biases a difference-in-differences toward "no
move" or "shallower", never toward "deeper" — so those arms were not merely
noisy, they leaned in a known direction, and all three of the study's
"shallower" readings landed on them.

This is the sizing arithmetic run *before* the grid, off a short pilot wave of
the deepest arm:

    positives per class  >=  (positives that arm finds by the horizon) / bar / sim_fraction

Run standalone to size a pile, or through `preflight.sh
--require-harvest-headroom BAR --pilot-cells DIR`, which is where it actually
stops a launch.  `harvest_3547.py` reports the same quantity for a *finished*
study's arms; this one turns it into a bound on the next grid's pile.

    python harvest_headroom.py --pilot /expscratch/$USER/acq-3319/bin \
        --horizon 400 --bar 0.5 --prepare-info <study>/results/prepare_info.json

Detail lines first, then a single verdict line last, which is what `preflight.sh`
cases on:

* ``CLEAR`` — the deepest arm stays under the bar at the planned horizon.
* ``OVER``  — it does not.  The pile is too thin for the grid; the line carries
  the positives-per-class that would clear it.
* ``UNKNOWN`` — nothing here rules compression out: no pilot, a pilot that
  stopped short of the horizon, or one too narrow to believe.  **A pilot can
  fail a grid without being able to clear one**: harvest is the most
  category-dependent quantity in this harness, so a one-category pilot that
  comes in over the bar is evidence, and the same pilot coming in under it is
  not (`lessons/2026-09-02-one-pilot-cell-cleared-a-hazard-the-full-wave-hit.md`,
  where one cell read 38% and the wave read 85%).

Exit status mirrors the verdict: 0 CLEAR, 1 OVER, 3 UNKNOWN, 2 usage.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _cells_paths import main_frame_files  # noqa: E402

#: What makes two rows different *cells* rather than two steps of one.  A cell
#: file can hold more than one style, and a grid can run one category under
#: several strategies, so keying on (seed, category) alone silently merges
#: trajectories that harvest at different rates.  Whichever of these the CSV
#: actually carries is used; the rest are ignored.
KEY_COLUMNS = ("dataset", "category", "seed", "strategy", "trainer", "head", "style", "prevalence_arm")

CLEAR, OVER, UNKNOWN = "CLEAR", "OVER", "UNKNOWN"
_EXIT = {CLEAR: 0, OVER: 1, UNKNOWN: 3}


class Cell:
    """One pilot trajectory, read at the last step at or before the horizon."""

    def __init__(self, category: str, t: int, n_good: int, sim_positives: int) -> None:
        self.category = category
        self.t = t
        self.n_good = n_good
        self.sim_positives = sim_positives


class Arm:
    """One pilot arm's harvest, projected onto the planned pile."""

    def __init__(self, name: str, cells: list[Cell], harvests: list[float], bar: float) -> None:
        self.name = name
        self.n_cells = len(cells)
        self.categories = sorted({c.category for c in cells})
        self.t_min = min(c.t for c in cells)
        self.t_max = max(c.t for c in cells)
        self.median_n_good = statistics.median(c.n_good for c in cells)
        self.median_harvest = statistics.median(harvests)
        self.over_bar = sum(1 for h in harvests if h >= bar) / len(harvests)

    def line(self) -> str:
        return "%-10s cells=%-5d cats=%-4d t=%s  median n_good=%-7.1f harvest=%5.1f%%  cells over bar=%4.1f%%" % (
            self.name,
            self.n_cells,
            len(self.categories),
            str(self.t_min) if self.t_min == self.t_max else "%d-%d" % (self.t_min, self.t_max),
            self.median_n_good,
            100 * self.median_harvest,
            100 * self.over_bar,
        )


def read_cells(cells_dir: pathlib.Path, horizon: int) -> list[Cell]:
    """Every cell under *cells_dir*, read at its last step at or before *horizon*.

    Reading at the horizon rather than at the cell's final step is what lets a
    400-step pilot size a 100-step grid: `max_steps` reaches the simulation as a
    loop bound and nothing inside the loop reads it, so a long trajectory is a
    strict extension of the short one (`check_prefix_3547.py`).
    """
    best: dict[tuple[str, ...], Cell] = {}
    for path in main_frame_files(cells_dir):
        if path.stat().st_size == 0:
            continue
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                try:
                    t = int(row["t"])
                    n_good = int(row["n_good"])
                    sim_positives = int(round(float(row["n_haystack"]) * float(row["realized_prevalence"])))
                except (KeyError, TypeError, ValueError):
                    continue  # a truncated or foreign row is not a cell
                if t > horizon:
                    continue
                key = tuple(row.get(col, "") or "" for col in KEY_COLUMNS)
                cur = best.get(key)
                if cur is None or (t, n_good) > (cur.t, cur.n_good):
                    best[key] = Cell(row.get("category", ""), t, n_good, sim_positives)
    return list(best.values())


def arm_dirs(pilot: pathlib.Path, arms: list[str]) -> list[tuple[str, pathlib.Path]]:
    """The pilot's per-arm cell directories.

    *pilot* is either one arm's `cells/` directory or a study's `bin/`-style
    base holding `<arm>/results/cells`.  Which arm is deepest is a fact about
    the launcher, not about the tree, so every arm present is read and the
    **worst** (highest-harvest) one is what the verdict is taken on — the data
    names the deepest arm rather than the caller having to.
    """
    if main_frame_files(pilot):
        parents = pilot.resolve().parents
        name = parents[1].name if len(parents) > 1 else pilot.resolve().name
        return [(name, pilot)]
    found = []
    for sub in sorted(p for p in pilot.iterdir() if p.is_dir()):
        if arms and sub.name not in arms:
            continue
        for cand in (sub / "results" / "cells", sub / "cells", sub):
            if main_frame_files(cand):
                found.append((sub.name, cand))
                break
    return found


def planned_positives(info_path: pathlib.Path | None, sim_fraction: float) -> tuple[dict[str, int], int]:
    """Per-category and thinnest positives in the sim half of the PLANNED pile.

    Read from `prepare_info.json`, the same source `preflight.sh` check 16b
    uses, so the two checks cannot disagree about how deep the pile is.  The
    thinnest selected category is the fallback denominator because it is the one
    that compresses first.
    """
    if info_path is None:
        return {}, 0
    info = json.loads(info_path.read_text())
    per_cat: dict[str, int] = {}
    for embs in (info.get("datasets") or {}).values():
        for rec in (embs or {}).values():
            counts = (rec or {}).get("category_counts") or {}
            for cat in (rec or {}).get("selected_categories") or list(counts):
                if cat in counts and counts[cat]:
                    n = max(1, int(int(counts[cat]) * sim_fraction))
                    per_cat[cat] = min(per_cat[cat], n) if cat in per_cat else n
    return per_cat, (min(per_cat.values()) if per_cat else 0)


def _denominator(cell: Cell, sim_positives: int, per_cat: dict[str, int], thinnest: int) -> int:
    """The positives *this* cell would be harvesting from on the planned pile."""
    if sim_positives:
        return sim_positives
    if cell.category in per_cat:
        return per_cat[cell.category]
    if thinnest:
        return thinnest
    return cell.sim_positives


def size_for(n_good: float, bar: float, sim_fraction: float) -> int:
    """Positives per class that keep *n_good* finds under *bar* of the sim half."""
    return int(math.ceil(math.ceil(n_good / bar) / sim_fraction))


def summarise(arms: list[tuple[str, pathlib.Path]], opts: argparse.Namespace, thinnest: int, per_cat: dict[str, int]):
    """Per-arm harvest on the planned pile, worst arm last."""
    out = []
    for name, cells_dir in arms:
        kept, harvests = [], []
        for cell in read_cells(cells_dir, opts.horizon):
            denom = _denominator(cell, opts.sim_positives, per_cat, thinnest)
            if denom:
                kept.append(cell)
                harvests.append(min(1.0, cell.n_good / denom))
        if kept:
            out.append(Arm(name, kept, harvests, opts.bar))
    out.sort(key=lambda a: a.median_harvest)
    return out


def verdict(opts: argparse.Namespace, arms: list[Arm], thinnest: int, planned_cats: set[str]) -> str:
    """The one line `preflight.sh` reads, given the per-arm summary."""
    sim_pos = opts.sim_positives or thinnest
    if not arms:
        # No pilot at all.  One bound still holds with no assumption whatever:
        # an arm cannot find more positives than it takes clicks.  When even
        # that leaves the arm under the bar, compression is impossible here.
        if not sim_pos:
            return "%s\tno pilot cells and no pile depth (pass --pilot / --sim-positives / --prepare-info)" % UNKNOWN
        bound = min(1.0, opts.horizon / sim_pos)
        if bound < opts.bar:
            return "%s\tno pilot, and none needed: %d clicks cannot exceed %.0f%% of %d sim positives (bar %.0f%%)" % (
                CLEAR,
                opts.horizon,
                100 * bound,
                sim_pos,
                100 * opts.bar,
            )
        return (
            "%s\tno pilot: nothing bounds harvest but the horizon itself (%d clicks, %d sim positives = %.0f%%, bar %.0f%%); pilot the deepest arm, or size for %d positives per class"
            % (
                UNKNOWN,
                opts.horizon,
                sim_pos,
                100 * bound,
                100 * opts.bar,
                size_for(opts.horizon, opts.bar, opts.sim_fraction),
            )
        )

    worst = arms[-1]
    where = "%d sim positives" % sim_pos if sim_pos else "the pilot's own pile"
    if worst.median_harvest >= opts.bar:
        return (
            "%s\t%s harvests %.0f%% of %s at t=%d (bar %.0f%%, %.0f%% of its cells over it); size for %d positives per class"
            % (
                OVER,
                worst.name,
                100 * worst.median_harvest,
                where,
                worst.t_max,
                100 * opts.bar,
                100 * worst.over_bar,
                size_for(worst.median_n_good, opts.bar, opts.sim_fraction),
            )
        )
    if worst.t_max < opts.horizon:
        return (
            "%s\t%s stopped at t=%d of a %d-step horizon; its %.0f%% is a FLOOR, not the answer - run the pilot to the horizon"
            % (
                UNKNOWN,
                worst.name,
                worst.t_max,
                opts.horizon,
                100 * worst.median_harvest,
            )
        )
    missing = sorted(planned_cats - set(worst.categories))
    if missing:
        return (
            "%s\t%s was piloted on %d of %d planned categories (missing %s); harvest is the most category-dependent quantity here, so this cannot CLEAR the grid"
            % (
                UNKNOWN,
                worst.name,
                len(worst.categories),
                len(planned_cats),
                ", ".join(missing[:6]) + (", ..." if len(missing) > 6 else ""),
            )
        )
    if len(worst.categories) < 2:
        return (
            "%s\t%s was piloted on one category (%s); one cell is not a sample of a per-category quantity (#3319 was wrong by 25 points that way)"
            % (
                UNKNOWN,
                worst.name,
                worst.categories[0] or "?",
            )
        )
    return "%s\t%s is the worst of %d piloted arm%s: %.0f%% of %s at t=%d, under the %.0f%% bar" % (
        CLEAR,
        worst.name,
        len(arms),
        "" if len(arms) == 1 else "s",
        100 * worst.median_harvest,
        where,
        worst.t_max,
        100 * opts.bar,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--pilot", type=pathlib.Path, default=None, help="a cells/ dir, or a base holding <arm>/results/cells"
    )
    p.add_argument("--arms", default="", help="comma-separated arm filter; default every arm under --pilot")
    p.add_argument("--bar", type=float, default=None, help="the pre-registered compression bar, as a fraction")
    p.add_argument("--horizon", type=int, default=None, help="planned max_steps (default: $CALIB_MAX_STEPS)")
    p.add_argument("--sim-positives", type=int, default=0, help="positives in the sim half of the PLANNED pile")
    p.add_argument("--prepare-info", type=pathlib.Path, default=None, help="derive --sim-positives from a prepare run")
    p.add_argument("--sim-fraction", type=float, default=None, help="default: $CALIB_SIM_FRACTION, else 0.5")
    opts = p.parse_args(argv)
    if opts.horizon is None:
        opts.horizon = int(os.environ.get("CALIB_MAX_STEPS") or 0)
    if opts.sim_fraction is None:
        opts.sim_fraction = float(os.environ.get("CALIB_SIM_FRACTION") or 0.5)
    if opts.bar is None or not (0 < opts.bar <= 1):
        p.error("--bar must be a fraction in (0, 1] - the study's pre-registered compression bar")
    if opts.horizon <= 0:
        p.error("--horizon (or CALIB_MAX_STEPS) must be a positive number of steps")
    if not (0 < opts.sim_fraction <= 1):
        p.error("--sim-fraction must be in (0, 1]")
    return opts


def main(argv: list[str] | None = None) -> int:
    opts = parse_args(argv)
    per_cat, thinnest = planned_positives(opts.prepare_info, opts.sim_fraction)
    arms: list[tuple[str, pathlib.Path]] = []
    if opts.pilot is not None:
        if not opts.pilot.is_dir():
            print("%s\tno pilot at %s" % (UNKNOWN, opts.pilot))
            return _EXIT[UNKNOWN]
        arms = arm_dirs(opts.pilot, [a for a in opts.arms.split(",") if a])
    summary = summarise(arms, opts, thinnest, per_cat)
    for arm in summary:
        print(arm.line())
    line = verdict(opts, summary, thinnest, set(per_cat))
    print(line)
    return _EXIT[line.split("\t", 1)[0]]


if __name__ == "__main__":
    sys.exit(main())
