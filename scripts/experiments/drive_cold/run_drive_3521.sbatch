#!/bin/bash
# Does the corrected tuning driver produce a *usable* timing profile (#3521)?
#
# Two legs of the same workload on the same node, differing only in how the
# driver arranges the caches that a step's cost forks on:
#
#   OLD  the behaviour on dev: a shared data dir, no cache clearing, no atlas
#        rebuild, --reps 2 (the DEFAULT, which #3345 dodged by passing 1).
#   NEW  --cold-embed (now the default) and --cold-atlas.
#
# Each leg gets its own data dir so the other's registry and embeddings cannot
# reach it; the *source* media are symlinks into the shared read-only caches, so
# nothing re-downloads and neither leg pays for the other's extraction.
#
# Both legs run on the tree with the branch markers in it, so every row in both
# is labelled with the branch it took. That is what makes each leg a held-out
# truth set for the other: the analyzer fits OLD's rows *with the markers
# stripped* — which is exactly the profile dev's code writes — and then asks it
# to predict the runs NEW measured, and vice versa.
#
# One job, not two: #3160 measured `gres/gpu:v100` resolving to two different
# devices, so two legs on two nodes would compare hardware, not drivers.
#
# Completion is the file $EXP/DONE.run, never a live process (watchers die with
# the VPN).
#SBATCH --job-name=drive3521
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=/expscratch/sgreenberg/drive-3521/slurm-%j.out

set -euo pipefail

EXP=/expscratch/sgreenberg/drive-3521
WT=/exp/sgreenberg/projects/vts-drive-3521
DEMOS=caltech101_s,caltech101_m,caltech101_l,caltech101_a,esc50_s,esc50_m,esc50_l,esc50_a
TUNE="$WT/scripts/profiling/tune_timing_profile.py"

CALTECH=/exp/scale26/datasets/external/vtsearch-resweep/j3-gpu-cuml-on/caltech-101
ESC50=/exp/scale26/datasets/external/vtsearch-demos/ESC-50-master

module load python/3.12.3
source /exp/sgreenberg/projects/VTSearch/.venv/bin/activate
cd "$WT"

for leg in old new; do
  mkdir -p "$EXP/$leg/data"
  ln -sfn "$CALTECH" "$EXP/$leg/data/caltech-101"
  ln -sfn "$ESC50"   "$EXP/$leg/data/ESC-50-master"
done

export VTSEARCH_MODELS_DIR=/exp/scale26/datasets/external/vtsearch-calib-2623/models
export VTSEARCH_TORCH_THREADS="${SLURM_CPUS_ON_NODE:-8}"
unset VTSEARCH_TIMING_PROFILE || true

# Provenance. A load-cost number without its core count is not comparable to any
# other, and the V100 nodes are two different machines (#3160, #3062).
{
  echo "host=$(hostname)"
  echo "commit=$(git rev-parse HEAD)"
  echo "branch=$(git rev-parse --abbrev-ref HEAD)"
  echo "slurm_job=${SLURM_JOB_ID:-none}"
  echo "cpus=${SLURM_CPUS_ON_NODE:-?} torch_threads=$VTSEARCH_TORCH_THREADS"
  echo "cpu_model=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | xargs)"
  nvidia-smi -L 2>/dev/null || echo "no nvidia-smi"
} > "$EXP/env.txt"
python "$WT/scripts/experiments/timing_r2/env_probe.py" >> "$EXP/env.txt" 2>&1 || true
cat "$EXP/env.txt"

step () { echo "=== [$(date +%H:%M:%S)] $* ==="; }

# --- leg OLD: what dev's driver does today --------------------------------
# The families run in the #3345 order and against one shared data dir, because
# that ordering is half the defect: dataset_stage reads the embeddings pkl the
# dataset_load leg before it just wrote.
export VTSEARCH_DATA_DIR="$EXP/old/data"
export VTSEARCH_PROFILE_LOAD="$EXP/old/loadprof.jsonl"
OLDROWS="$EXP/old/rows.jsonl"

step "OLD dataset_load"
python "$TUNE" --drive --allow-mutating --tasks dataset_load --no-cold-embed \
  --demo "$DEMOS" --reps 2 --data-dir "$EXP/old/data" \
  --record "$OLDROWS" --out "$EXP/old/profile_step1.json" 2>&1 | tee "$EXP/old/load.log"

step "OLD dataset_stage"
python "$TUNE" --drive --allow-mutating --tasks dataset_stage --no-cold-embed \
  --demo "$DEMOS" --reps 2 --data-dir "$EXP/old/data" \
  --record "$OLDROWS" --out "$EXP/old/profile_step2.json" 2>&1 | tee "$EXP/old/stage.log"

step "OLD dataset_open,text_sort"
python "$TUNE" --drive --tasks dataset_open,text_sort \
  --reps 2 --data-dir "$EXP/old/data" \
  --record "$OLDROWS" --out "$EXP/old/profile_step3.json" 2>&1 | tee "$EXP/old/readonly.log"

# --- leg NEW: the same workload with the branches arranged ----------------
export VTSEARCH_DATA_DIR="$EXP/new/data"
export VTSEARCH_PROFILE_LOAD="$EXP/new/loadprof.jsonl"
NEWROWS="$EXP/new/rows.jsonl"

step "NEW dataset_load (cold embed)"
python "$TUNE" --drive --allow-mutating --tasks dataset_load \
  --demo "$DEMOS" --reps 2 --data-dir "$EXP/new/data" \
  --record "$NEWROWS" --out "$EXP/new/profile_step1.json" 2>&1 | tee "$EXP/new/load.log"

step "NEW dataset_stage (cold embed)"
python "$TUNE" --drive --allow-mutating --tasks dataset_stage \
  --demo "$DEMOS" --reps 2 --data-dir "$EXP/new/data" \
  --record "$NEWROWS" --out "$EXP/new/profile_step2.json" 2>&1 | tee "$EXP/new/stage.log"

step "NEW dataset_open,text_sort (cold atlas)"
python "$TUNE" --drive --tasks dataset_open,text_sort --cold-atlas \
  --reps 2 --data-dir "$EXP/new/data" \
  --record "$NEWROWS" --out "$EXP/new/profile_step3.json" 2>&1 | tee "$EXP/new/readonly.log"

step "done"
wc -l "$OLDROWS" "$NEWROWS"
touch "$EXP/DONE.run"
