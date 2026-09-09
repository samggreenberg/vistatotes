# #3547 — state at 2026-09-03 13:40 EDT

**COMPLETE.** Both waves are in, the control settled H2, and the verdicts are
written up in [`REPORT_3547.md`](REPORT_3547.md). Nothing is running.

Branch `claude/acq-deep-3547`, PR **#3598** (open); PR #3584 already MERGED to
dev (the `vg_scale_deep` pile). Worktree `/exp/sgreenberg/projects/vts-acq-3547`.

## What was measured

| wave | pile | cells | where |
|---|---|---:|---|
| deep grid, 7 arms | `vg_scale_deep` | 1344 | `/expscratch/sgreenberg/acq-3547` |
| H2 control, 2 arms | `vg_scale_any` | 384 | `/expscratch/sgreenberg/acq-3547-ctrl` |

0 failures, 0 header-only cells in either.

## Verdicts

* **H3 — the plateau REPLICATES.** Δcost vs `prod` @100: `-1` −0.019, `-3`
  −0.038, `-4` −0.039, `-5` −0.030, `-6` −0.023. The anchor holds, so H1/H2 are
  readable. Falsifier behaved (`+2`, −11 positives).
* **H1 — the optimum does NOT move.** Clean `-4` vs `-3` DiD: cost +0.006
  [−0.002, +0.013], AUC +0.003 [−0.001, +0.007], CTT +2.8 [−5.9, +12]. All
  nulls. **The knob is a CONSTANT, not a schedule.** `-5`/`-6` lean "shallower"
  but are COMPRESSED and excluded as one-sided.
* **H2 — EXHAUSTION, confirmed by the control.** The shallow pile re-run on THIS
  commit reproduced #3319 exactly: 82% harvest, **5.7%** incidence, **all 11
  first-spikes after t=100** (median t=258). The deep pile at the same
  aggression: 1.0%, and **zero of 1344 cells spike after t=100**. Dev drift is
  ruled out; #3319's deep guardrail is an artefact of its own ceiling.
* **The ship is vindicated at depth**: `-4` reaches `prod`'s 400-click answer in
  44 clicks vs 154 (3.5×, matching #3319's 3.2×), now where its tail is NOT
  compressed.

## The mechanism, refined

A deep spike is what a threshold does when it is fit with too few positives, and
there are **two entry points**: too few *yet* (`acq_p2`, first-spike quartiles
21/36/47, `n_good`≈3) and none *left* (control `-3`, quartiles 193/258/276,
`n_good`≈88 and flat). Harvest fraction alone catches only the second.

## Tooling added this session

All committed, all with `--base`/`--csv` so they can be pointed at either study:

* `frontier_3547.py --csv <dir>` — writes 5 tidy CSVs beside the markdown, so
  the figures and the tables cannot drift apart. **Verified output-neutral**:
  regenerating `GENERATED_TABLES_3547.md` after the patch is byte-identical
  apart from a trailing newline.
* `spike_timing_3547.py --base/--arms/--csv` — was hardcoded to the main study.
  Re-run on the main base after parametrising and it reproduced the committed
  `spike_timing_3547.txt` exactly.
* `harvest_3547.py --base/--arms` — was hardcoded to #3319's pile.
* `spike_examples_3547.py` — NEW, prints the literal trajectory rows around a
  cell's first spike. This is where the `n_good`-frozen-at-88 evidence came from.
* `figures_3547.py` — NEW, the five report figures.

## Follow-ups filed

* **#3602** — `analyze_acq.py:218` computes `positives_100` as the trajectory's
  LAST row, not t=100. #3319's "Δ positives@100 = +90.1" is really t=400.
* **#3611** — size a deep grid from its DEEPEST arm; 900 was a supply bound
  checked against a horizon bound, and neither bounds aggression.

## Traps paid for in this session

* `git commit` FAILS rc=1 on ruff/ruff-format **and** on `end-of-file-fixer`;
  read the commit's own rc.
* Never write a `|| git commit -C ORIG_HEAD` fallback — it silently reuses the
  PREVIOUS commit's message. One commit here landed stale and needed `--amend`.
* Preflight refuses a launch with uncommitted tracked changes — commit BEFORE
  launching.
* **Run long analyses under `sbatch`/`nohup`, not a foreground ssh** — a killed
  watcher takes the ssh with it, and 20 min of pandas on a login node is bad
  citizenship. `spike_timing_3547.py` over 7 arms is ~10 minutes.
* Do NOT read `analysis/REPORT_acq.md` as the verdict — its default arm table
  covers 5 of 7 arms and silently drops `acq_m5`/`acq_m6`. Use `frontier_3547.py`.
