# Pre-registration — does the corrected tuning driver produce a usable profile? (#3521)

Written before the sweep of 2026-09-02 (SLURM 609828). Recorded here so the
report can be read against what was expected rather than against what was found.

## Premise under test

`scripts/profiling/tune_timing_profile.py --drive` exercises each task family
and fits the recorded timings into a profile. Several steps fork on a cache,
and a driven sweep hits the cheap side of every fork:

- a demo import whose `EMBEDDINGS_DIR/<demo>.pkl` exists embeds nothing;
- a dataset open whose pickle carries a cached coverage atlas restores it
  instead of rebuilding the hierarchical k-means.

#3345 measured both. The profile it produced gives the coverage step ~0.02 of a
bar that `tasks.py` weights at 0.85.

## Hypotheses

**H1.** The defect reproduces at the driver's own defaults. Driven as dev drives
it, `dataset_load`·`embed` at `--reps 2` yields one real embed and one zero per
demo.

**H2.** Clearing the demo embeddings cache per rep makes every rep measure a
real embed, and the resulting coefficients are reproducible across reps.

**H3.** A profile fitted from a sweep that only ever restored the atlas paces a
*rebuild* badly — worse than shipping no profile at all.

**H4.** Driving the rebuild through the on-demand endpoint yields a coverage
step with a credible slope (r² ≥ 0.9), where the restore-only sweep yields none.

## Design

Two legs of one workload, on one node (#3160: `gres/gpu:v100` resolves to two
different devices, so two nodes would compare hardware). Each leg has its own
data dir; source media are symlinks into shared read-only caches.

- **OLD** — dev's behaviour: shared data dir, `--no-cold-embed`, no
  `--cold-atlas`, `--reps 2` (the default), families in #3345's order.
- **NEW** — `--cold-embed` (the new default) and `--cold-atlas`.

Both legs run on the tree carrying the branch markers, so every row in both is
labelled. The OLD leg's rows are fitted **with the branch field stripped**, which
is the document dev's code writes.

Workload: `caltech101_{s,m,l,a}` (412/838/1704/2954) and `esc50_{s,m,l,a}`
(245/588/1127/1960) — four size tiers per media type, because r² only exists
where the fitter had spread in `n` (#3345).

## Metrics, declared in advance

- **Step error** — `|predicted − observed| / observed` per step, median over
  runs, over steps that took at least 0.05 s.
- **Bar error** — half the L1 distance between the predicted weight vector and
  the run's observed share of its own total time: the fraction of the bar
  budgeted to the wrong step. **This is the primary metric.** A fit can be
  mediocre on every step and still pace smoothly if the errors share a
  direction; one step 50× off freezes the bar however good the others are.

Both reported **per branch**, because the claim is not that a profile is wrong
in general but that it is wrong about a branch it never saw.

## Scoring

Each arm is scored on held-out runs two ways: **cross-leg** (each leg is the
other's held-out set) and **within-leg** (half of each leg's own reps fit, half
score). The second exists because the first never asks an arm to predict a
branch only its own leg produced.

Third arm: **shipped** — no profile at all.

## What would refute the change

- H2 false (cleared cache still measures no embed) would mean the cache is not
  the cause and the fix is aimed at the wrong thing.
- H3 false (`old` paces rebuilds no worse than `shipped`) would mean the
  mispriced profile is harmless and the issue is not worth acting on.
- `new` materially worse than `old` on the branches **both** measured would mean
  the marker costs more than it buys.
