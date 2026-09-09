#!/usr/bin/env python3
"""Is a profile from the corrected driver *usable*? (#3521)

The unit tests pin that the fitter withholds a cheap-branch-only cell and prices
a forked step from the runs that did the work. They cannot answer the question
the issue actually asks — whether the resulting profile paces a real progress
bar better than the one dev's driver produces, and better than shipping no
profile at all. That needs measured runs of both branches, which is what
``run_drive_3521.sbatch`` collects.

Three arms, each a way of deciding how to pace a bar:

``old``       fitted from the OLD leg's rows **with the branch field stripped**,
              which is byte-for-byte the profile dev's code writes: its recorder
              does not carry the field and its fitter would not read it.
``new``       fitted from the NEW leg's rows, markers kept.
``shipped``   no profile at all — ``TaskSpec.default_terms`` and, for
              ``dataset_load``, the calibrated ``_load_cost_model`` table.

Each arm is scored against **held-out runs**: the other leg's rows, which are
independent runs of the same workload on the same node. Two scores, because
they answer different questions:

**Step error.** ``|predicted - observed| / observed`` per step, median over
runs. What the fit is usually judged on.

**Bar error.** Half the L1 distance between the predicted weight vector and the
run's observed share of its own total time — the fraction of the bar that sits
in the wrong step. This is the number a user experiences: a bar can be built
from steps that are each 30 % off and still sweep smoothly if the errors share
a direction, and a bar built from one step that is 50x off freezes no matter how
good the others are. Reported per branch, because a profile is not wrong in
general — it is wrong about a branch it never saw.

Writes ``summary.json`` plus ``tables.md`` beside the rows, and prints the
coverage report each profile's own sweep produces.

Run it on a **GPU node**: ``load_step_weights`` reads the device from the
process, not from an argument, so on a login node the shipped arm silently
becomes a CPU row (#3345).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _log(*parts: object) -> None:
    print("[drive3521]", *parts, file=sys.stderr, flush=True)


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def strip_branches(rows: list[dict]) -> list[dict]:
    """The same rows as dev's recorder would have written them.

    Dev stamps no ``branch``, and its fitter reads none. Stripping the field is
    how the OLD arm is made a fair stand-in for dev's output rather than for
    "dev's driver run through the new fitter", which is neither arm.
    """
    return [{k: v for k, v in row.items() if k != "branch"} for row in rows]


def group_runs(rows: list[dict]) -> list[dict]:
    """Collapse per-step rows back into the runs they were emitted from.

    The recorder writes one row per step of one run, in order, and never
    interleaves two runs in one file: a run's rows are contiguous and its first
    step name repeats only when the next run starts. Grouping on that is exact
    here and does not need a run id in the schema.
    """
    runs: list[dict] = []
    current: dict | None = None
    for row in rows:
        key = (row["task"], row.get("media_type", ""), row.get("embedder", ""), row.get("n", 0))
        if current is None or current["key"] != key or row["step"] in current["steps"]:
            current = {
                "key": key,
                "task": row["task"],
                "n": float(row.get("n") or 0.0),
                "media_type": row.get("media_type", ""),
                "embedder": row.get("embedder", ""),
                "size_mb": float(row.get("size_mb") or 0.0),
                "ok": bool(row.get("ok", True)),
                "steps": {},
                "branches": {},
                "rows": [],
            }
            runs.append(current)
        current["rows"].append(row)
        current["steps"][row["step"]] = float(row["seconds"])
        if row.get("branch"):
            current["branches"][row["step"]] = row["branch"]
    return [r for r in runs if r["ok"]]


def run_branch(run: dict) -> str:
    """One label for a run: the branch of whichever step forked, else 'plain'."""
    if not run["branches"]:
        return "plain"
    return "+".join(f"{step}={branch}" for step, branch in sorted(run["branches"].items()))


def profile_cell_key() -> str:
    """The cell key this process will look profiles up under.

    Not the same string as ``resolve_device_name()``: the key carries the cuML
    suffix, and ``cell_keys`` derives that from **live** ``cuml_active()``, so
    the same profile resolves a different cell on a cuML host than on one
    without (#3345). Printing the device alone would hide exactly the mismatch
    that turns every measured arm silently into the shipped one.
    """
    from vtscore.timing.fit import device_key
    from vtscore.timing.profile import cuml_active, resolve_device_name

    return device_key(resolve_device_name(), cuml_active())


def cells_resolve(profile_path: Path, tasks: set[str]) -> dict[str, bool]:
    """Whether each task has a cell this process can actually reach.

    A profile keyed ``cuda+cuml`` read on a host without cuML resolves nothing,
    falls through to the shipped defaults, and produces a table in which the
    measured arms quietly *are* the baseline. Asserted rather than assumed.
    """
    import os

    from vtscore.timing import reload_profile
    from vtscore.timing.profile import active_profile, cell_keys, resolve_device_name

    os.environ["VTSEARCH_TIMING_PROFILE"] = str(profile_path)
    reload_profile()
    profile = active_profile()
    # The media-agnostic key of each candidate device is enough: a task with no
    # cell under any of them cannot be reached at any specificity.
    wanted = set(cell_keys(resolve_device_name(), "", ""))
    return {task: bool(wanted & set(profile.steps.get(task, {}))) for task in sorted(tasks)}


def predict(run: dict, profile_path: Path | None) -> dict[str, float] | None:
    """Predicted seconds per step for *run* under the profile at *profile_path*.

    ``None`` for the shipped arm, which is what "no ``VTSEARCH_TIMING_PROFILE``"
    means and is therefore the honest way to ask for the built-in defaults.
    """
    import os

    from vtscore.timing import reload_profile, step_terms

    if profile_path is None:
        os.environ.pop("VTSEARCH_TIMING_PROFILE", None)
    else:
        os.environ["VTSEARCH_TIMING_PROFILE"] = str(profile_path)
    reload_profile()
    return step_terms(
        run["task"],
        media_type=run["media_type"],
        embedder=run["embedder"],
        n=run["n"],
        size_mb=run["size_mb"],
    )


def bar_error(predicted: dict[str, float], observed: dict[str, float], spec) -> float | None:
    """Fraction of the bar that sits in the wrong step.

    Half the L1 distance between the two normalised weight vectors, summed into
    tracker-step slots exactly as ``step_weights`` does — several cost phases
    can share one step number and the user sees only the slot. Ranges 0 (the
    bar is paced exactly right) to 1 (every second of the bar is budgeted to a
    step other than the one spending it).
    """

    def slots(terms: dict[str, float]) -> list[float] | None:
        weights = [0.0] * spec.tracker_steps
        for step, index in zip(spec.steps, spec.step_index):
            weights[index - 1] += max(0.0, terms.get(step, 0.0))
        total = sum(weights)
        return [w / total for w in weights] if total > 0 else None

    pred, obs = slots(predicted), slots(observed)
    if pred is None or obs is None:
        return None
    return 0.5 * sum(abs(p - o) for p, o in zip(pred, obs))


#: Steps below this many seconds are dropped from the step-error table. A 2 ms
#: step predicted at 20 ms is a 900 % error nobody can see and would dominate a
#: median taken over every step in the sweep; the bar error already accounts for
#: it, correctly, as ~0 % of the bar.
_MIN_OBSERVED_S = 0.05


def score(runs: list[dict], profile_path: Path | None, label: str) -> list[dict]:
    """One record per (run, arm), carrying both errors and the run's branch."""
    from vtscore.timing import task_spec

    out: list[dict] = []
    for run in runs:
        predicted = predict(run, profile_path)
        spec = task_spec(run["task"])
        if predicted is None or spec is None:
            continue
        rec = {
            "arm": label,
            "task": run["task"],
            "media_type": run["media_type"],
            "n": run["n"],
            "branch": run_branch(run),
            "bar_error": bar_error(predicted, run["steps"], spec),
            "steps": {},
            "observed": {},
        }
        for step, observed in run["steps"].items():
            if observed < _MIN_OBSERVED_S:
                continue
            rec["steps"][step] = abs(predicted.get(step, 0.0) - observed) / observed
            rec["observed"][step] = observed
        out.append(rec)
    return out


def split_runs(runs: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split a leg's runs into (fit, held out) within each measured cell.

    The cross-leg comparison below is the headline, but it has a blind spot the
    table would not show: each arm is only ever scored on the branches the
    *other* leg produced, so ``new`` — whose entire purpose is to pace an atlas
    rebuild — is never once asked to predict one. Splitting each leg's own reps
    closes that, at the cost of halving the rows behind each fit.

    Split on the run's identity rather than on file order, so the fit half and
    the held-out half cover the same sizes and the same branches; a positional
    split would put every rep-1 cold model load in one half.
    """
    seen: dict[tuple, int] = defaultdict(int)
    fit: list[dict] = []
    held: list[dict] = []
    for run in runs:
        key = (run["task"], run["media_type"], run["n"], run_branch(run))
        (fit if seen[key] % 2 == 0 else held).append(run)
        seen[key] += 1
    return fit, held


def _median(values: list[float]) -> float | None:
    clean = [v for v in values if v is not None]
    return statistics.median(clean) if clean else None


def _fmt(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", required=True, help="the sweep's results dir")
    args = ap.parse_args()

    from vtscore.timing.fit import coverage_report, fit_profile
    from vtscore.timing.profile import resolve_device_name

    exp = Path(args.exp)
    old_rows = load_rows(exp / "old" / "rows.jsonl")
    new_rows = load_rows(exp / "new" / "rows.jsonl")
    cell_key = profile_cell_key()
    _log(
        f"device={resolve_device_name()}  profile cell key={cell_key}  old={len(old_rows)} rows  new={len(new_rows)} rows"
    )

    # The two profiles under test. OLD is fitted from stripped rows so it is the
    # document dev writes, not dev's driver read through the new fitter.
    profiles = {
        "old": (exp / "profile_old.json", strip_branches(old_rows)),
        "new": (exp / "profile_new.json", new_rows),
    }
    for label, (path, rows) in profiles.items():
        doc = fit_profile(rows, min_samples=2, notes=f"#3521 {label} leg")
        path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _log(f"--- coverage report, {label} leg ---")
        for line in coverage_report(rows, doc):
            print(line, file=sys.stderr)

    # A profile whose cells this host cannot reach makes the measured arms
    # silently identical to the shipped one, and nothing in the table would say
    # so. Check before scoring, not after wondering (#3345's cuML trap).
    measured_tasks = {r["task"] for r in old_rows + new_rows}
    for label, (path, _) in profiles.items():
        reachable = cells_resolve(path, measured_tasks)
        unreachable = sorted(t for t, ok in reachable.items() if not ok)
        if unreachable:
            _log(f"WARNING: {label} profile has no cell reachable as {cell_key} for: {', '.join(unreachable)}")
            _log("         those tasks are scored on the shipped defaults, not on this arm.")
        else:
            _log(f"ok: every measured task resolves a {label} cell under {cell_key}")

    # Each leg is the other's held-out set: independent runs of the same
    # workload on the same node, measured by the same recorder.
    held_out = {"old": group_runs(new_rows), "new": group_runs(old_rows)}
    records: list[dict] = []
    for label, (path, _) in profiles.items():
        records += score(held_out[label], path, label)
    # The shipped arm is scored against every run, since it was fitted from none.
    records += score(group_runs(old_rows) + group_runs(new_rows), None, "shipped")

    # Within-leg holdout: half of each leg's own reps fit, the other half score.
    # This is the only place an arm meets the branches its *own* leg measured —
    # the cross-leg table above never asks `new` to predict an atlas rebuild,
    # which is the branch it exists for.
    self_records: list[dict] = []
    for label, rows in (("old", old_rows), ("new", new_rows)):
        # Split on the labelled runs so the held-out half keeps its branch
        # names; strip only what feeds the fit, since stripping emulates dev's
        # *fitter*, not a held-out run forgetting which branch it took.
        fit_runs, held_runs = split_runs(group_runs(rows))
        fit_rows = [row for run in fit_runs for row in run["rows"]]
        if label == "old":
            fit_rows = strip_branches(fit_rows)
        half = exp / f"profile_{label}_half.json"
        doc = fit_profile(fit_rows, min_samples=2, notes=f"#3521 {label} leg, fit half")
        half.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self_records += score(held_runs, half, label)
    self_records += score(split_runs(group_runs(old_rows))[1] + split_runs(group_runs(new_rows))[1], None, "shipped")

    def roll_up(recs: list[dict]) -> dict[tuple[str, str, str], dict[str, list[float]]]:
        out: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(lambda: {"bar": [], "step": []})
        for rec in recs:
            key = (rec["arm"], rec["task"], rec["branch"])
            out[key]["bar"].append(rec["bar_error"])
            out[key]["step"] += list(rec["steps"].values())
        return out

    by_arm_branch = roll_up(records)
    by_arm_branch_self = roll_up(self_records)

    lines = [
        "# #3521 — does the corrected driver produce a usable profile?",
        "",
        f"Device `{resolve_device_name()}`, profile cell key `{cell_key}` — the arms below are"
        " only meaningful if the profiles' cells resolve under that key (`cell_keys` derives the"
        " cuML suffix from the *live* backend, so a `cuda+cuml` profile read on a host without it"
        " silently becomes the shipped arm). Each profile is scored on the *other* leg's runs;",
        "`shipped` is scored on both. `bar` is the fraction of the progress bar budgeted to",
        "the wrong step (0 is perfect, 1 is every second in the wrong slot); `step` is the",
        "median per-step relative error over steps that took at least",
        f"{_MIN_OBSERVED_S:.2f} s.",
        "",
        "| arm | task | branch measured | runs | bar error | step error |",
        "|---|---|---|---:|---:|---:|",
    ]
    for (arm, task, branch), vals in sorted(by_arm_branch.items()):
        lines.append(
            f"| {arm} | {task} | {branch} | {len(vals['bar'])} | "
            f"{_fmt(_median(vals['bar']))} | {_fmt(_median(vals['step']))} |"
        )

    lines += [
        "",
        "## Within-leg holdout",
        "",
        "Half of each leg's own reps fit the profile, the other half score it. The",
        "cross-leg table above never asks an arm to predict a branch its own leg",
        "measured and the other did not - in particular it never asks `new` to pace an",
        "atlas rebuild, which is the branch it exists for. Fits here stand on half the",
        "rows, so read the ranking, not the third digit.",
        "",
        "| arm | task | branch measured | runs | bar error | step error |",
        "|---|---|---|---:|---:|---:|",
    ]
    for (arm, task, branch), vals in sorted(by_arm_branch_self.items()):
        lines.append(
            f"| {arm} | {task} | {branch} | {len(vals['bar'])} | "
            f"{_fmt(_median(vals['bar']))} | {_fmt(_median(vals['step']))} |"
        )

    summary = {
        "device": resolve_device_name(),
        "cell_key": cell_key,
        "rows": {"old": len(old_rows), "new": len(new_rows)},
        "records": records,
        "self_records": self_records,
        "by_arm_branch_self": {
            "|".join(k): {"runs": len(v["bar"]), "bar_error": _median(v["bar"]), "step_error": _median(v["step"])}
            for k, v in by_arm_branch_self.items()
        },
        "by_arm_branch": {
            "|".join(k): {"runs": len(v["bar"]), "bar_error": _median(v["bar"]), "step_error": _median(v["step"])}
            for k, v in by_arm_branch.items()
        },
    }
    (exp / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (exp / "tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    _log(f"wrote {exp / 'summary.json'} and {exp / 'tables.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
