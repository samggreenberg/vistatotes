#!/usr/bin/env bash
# Build the missing pile cells on the GRID: prefetch weights (CPU), then one
# GPU job per dataset so the three run concurrently within the 4-GPU tier.
#
#   bash launch_pile.sh              # prefetch + submit all three dataset jobs
#   bash launch_pile.sh coco_val     # just one dataset's job
#   VTS_GPU_NODE=rack7n03 bash launch_pile.sh visual_genome_m   # pin the device
#   VTS_BUILD_ARGS=--force bash launch_pile.sh vg_scale         # REBUILD, not fill
#   VTS_MAX_BEHIND=0 bash launch_pile.sh vg_scale   # build from an OLD checkout
#
# Weights are prefetched in a separate CPU step because parallel GPU jobs would
# otherwise race to populate the same shared HF cache (see prefetch_models.py).
#
# `VTS_BUILD_ARGS` is passed through to `build_pile.py` verbatim, and it exists
# for one job: `--force`. Filling a gap and *rebuilding a cell that already
# exists* are different operations with different risks -- the second replaces a
# file other studies are reading -- and until #3667 the second had no launcher
# at all, so a rebuild meant hand-writing an sbatch that skipped the canary and
# the CPU-dispatch pin this script exists to apply. A rebuild is exactly the run
# that must not skip them: a cell rebuilt without ATEN_CPU_CAPABILITY is a cell
# whose vectors no longer match its own fingerprint (#3160).
set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER="${USER:-sgreenberg}"
# The checkout the jobs import. DERIVED from this script's own location, never a
# fixed path: the old default pointed at `/exp/$USER/projects/vts-pile`, so
# running `bash launch_pile.sh` from any other worktree submitted jobs that built
# the pile from a DIFFERENT checkout -- 1,420 commits behind dev by 2026-09-06,
# predating `vg_scale` entirely. Nothing in the launch output said so; the build
# would have reported success against code nobody was looking at. Same shape as
# #3269, where a study measured a retired head because its worktree was stale.
REPO="${VTS_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
PILE="${VTS_PILE:-/expscratch/$USER/vts-cache}"
HERE="$REPO/scripts/experiments/pile"
LOGS="${PILE}/logs"
# GPU type is auto-picked from what is free, just before submitting (see below).
# These sweeps peak well under 16G; a fatter request wedges the job off idle
# GPUs whose RAM is already reserved (GRID-PLAYBOOK section 1).
MEM="${VTS_MEM:-24G}"
TIME="${VTS_TIME:-8:00:00}"
CPUS="${VTS_CPUS:-8}"

mkdir -p "$LOGS"

# Name the checkout before anything is submitted, and refuse one that is stale.
# The fixed default above was only half of #3693: the other half is that the
# launch output said `submitted vg_scale -> job NNN` and never named the tree or
# the commit, so a build from a checkout 1,420 commits behind dev had nothing on
# screen to be wrong. This prints both and applies preflight.sh check 4's bar
# (VTS_MAX_BEHIND, default 100) to the pile -- the artifact every study reads.
# shellcheck source=../repo_stamp.sh
source "$SELF_DIR/../repo_stamp.sh"
repo_stamp "$REPO" || exit 1
echo

ENVSET="module load python/3.12.3 && source /exp/$USER/projects/VTSearch/.venv/bin/activate"
ENVSET="$ENVSET && export VTS_REPO=$REPO VTS_PILE=$PILE"
# The commit as it stood at LAUNCH time, carried into the job so the cell's
# provenance can compare it against the commit the build actually resolves. A
# pile job sits in the queue for hours; a worktree that changes branch in the
# meantime builds from code the launch banner never showed anyone.
ENVSET="$ENVSET && export VTS_LAUNCH_COMMIT=${REPO_STAMP_COMMIT:-}"
# Keep HF off /exp: one model download there fills the 50G quota.
ENVSET="$ENVSET && export HF_HOME=$PILE/models VTSEARCH_MODELS_DIR=$PILE/models"
ENVSET="$ENVSET && export VTSEARCH_DATA_DIR=$PILE/datadir && cd $HERE"

# Spend the allocation the build job actually holds. VTSEARCH_TORCH_THREADS
# defaults to 1 (vtscore/config.py), which is right for a constrained container
# and wrong for a job holding $CPUS cores: the image processor's
# resize/normalise runs on the CPU between decode and forward, and leaving it
# single-threaded stalls the GPU behind it. The decode pool sizes itself from
# the cpuset, so it needs no env var here.
BUILDENV="$ENVSET && export VTSEARCH_TORCH_THREADS=$CPUS OMP_NUM_THREADS=$CPUS MKL_NUM_THREADS=$CPUS"

# Pin PyTorch's CPU kernel dispatch. This is what actually makes a pile cell
# reproducible across machines (#3160): the 384px resize in the image processor
# rounds differently under AVX-512 than under AVX2, and an unpinned build
# disagrees with itself across hosts on 12.3% of pixels by one 8-bit level --
# which propagates to 1.5e-04 median 1-cos on siglip2_l, 50x what switching the
# whole forward to fp16 costs. Measured on rack7n03: pinning it made the two
# hosts' vectors agree to 8.9e-16 (from 1.3e-04), and the resize itself got
# *faster* (256 images in 1.75s vs 2.36s), so the determinism is free.
# `avx2` rather than `avx512` because it is the floor every x86 host here can
# reach; a host that cannot do AVX2 would still diverge, and its cells would say
# so in their provenance.
BUILDENV="$BUILDENV ATEN_CPU_CAPABILITY=${VTS_CPU_CAPABILITY:-avx2}"

DATASETS=("${@:-visual_genome_m caltech101_m coco_val}")
read -r -a DATASETS <<< "${DATASETS[@]}"
DS_CSV="$(IFS=,; echo "${DATASETS[*]}")"

# Word-split on purpose: this is a flag list, not one argument.
read -r -a BUILD_ARGS <<< "${VTS_BUILD_ARGS:-}"
if [[ ${#BUILD_ARGS[@]} -gt 0 ]]; then
  echo "build args: ${BUILD_ARGS[*]}"
fi

# --- Stage 1: rebuild canary, then weights (CPU, blocking) ----------------
# The canary runs in front of every launch because that is the only thing that
# exercises the rebuild path on any schedule at all. `--verify` loads the built
# cells and shares no code with the build, so a rebuild path can rot invisibly
# behind a pile that verifies clean -- #3297 did, for eleven days, and surfaced
# only when somebody asked for a rebuild. A purge is the worst possible moment
# to discover that. It costs a fraction of a second, which is what makes here
# the right place for it: a canary expensive enough to skip gets skipped.
#
# Two runs, on purpose. The first reports **every** dataset, so rot under one
# you are not building today still gets seen. The second covers only the
# datasets about to be built, and its exit code is what gates the launch: a
# broken source under a dataset nobody asked for is news, not grounds to refuse
# to submit. It runs inside the srun rather than on the login node because
# build_pile.py asserts it is running against its own checkout's vtscore, which
# needs the venv this ENVSET activates.
echo "=== rebuild canary + weights (CPU) ==="
if ! srun --job-name=pile-prefetch --partition=cpu --cpus-per-task=4 --mem=8G --time=2:00:00 \
  bash -lc "$ENVSET \
    && { python build_pile.py --rebuildable || true; } \
    && python build_pile.py --rebuildable --datasets $DS_CSV >/dev/null \
    && python prefetch_models.py"; then
  echo >&2
  echo "prelaunch FAILED -- nothing submitted." >&2
  echo "  Either the rebuild canary found a source ${DATASETS[*]} cannot be built" >&2
  echo "  from (look for REBUILD-BROKEN above), or the weight prefetch failed." >&2
  exit 1
fi

# --- Stage 2: one GPU job per dataset -------------------------------------
# Pick the GPU type from what is actually free, and do it *here* rather than at
# the top of the script: stage 1 blocks on the queue, so availability measured
# before the prefetch is stale by the time we submit. This used to be a
# hardcoded `v100`, which is how every pile cell built before 2026-08-17 got
# embedded on the slowest GPU on the cluster while L40S/A100 nodes idled --
# 2.3x slower for siglip2_l (issue #3144). VTS_GPU still overrides.
PICK_GPU="$SELF_DIR/../../slurm/pick_gpu.py"

# Rebuilding one existing cell is a different job from building a new pile, and
# it has a constraint the picker cannot express: #3143 measured that a
# `gres/gpu:v100` job can land on either of two devices whose siglip2_l vectors
# differ by 1.5e-04 (#3160). Reproducing a cell therefore means pinning the
# *node* it was built on -- which is in that cell's provenance sidecar:
#
#   python build_pile.py --provenance      # read the node out
#   VTS_GPU_NODE=rack7n03 bash launch_pile.sh visual_genome_m
#
# The type is derived from the node rather than asked for separately: a
# --nodelist that disagrees with --gres pends forever with no explanation.
NODELIST=()
GPU_TYPE=""
if [[ -n "${VTS_GPU_NODE:-}" ]]; then
  NODELIST=(--nodelist="$VTS_GPU_NODE")
  GPU_TYPE="$(sinfo -h -N -n "$VTS_GPU_NODE" -o "%G" | head -1 | sed -E 's/.*gpu:([A-Za-z0-9_.-]+):.*/\1/')"
  if [[ -z "$GPU_TYPE" ]]; then
    echo "VTS_GPU_NODE=$VTS_GPU_NODE: no GPU gres visible on that node -- refusing to guess a type" >&2
    exit 1
  fi
  echo "pinned to node $VTS_GPU_NODE (gpu:$GPU_TYPE); the auto-pick is skipped"
elif [[ -f "$PICK_GPU" ]] && command -v python3 >/dev/null 2>&1; then
  GPU_TYPE="$(python3 "$PICK_GPU" --need "${#DATASETS[@]}" --explain || true)"
fi
# A missing picker (or python) must not sink the launch; l40s is the safe pin
# -- never the slowest type, and the largest pool now that rack4n01 is back.
GPU_TYPE="${GPU_TYPE:-${VTS_GPU:-l40s}}"

for ds in "${DATASETS[@]}"; do
  jid=$(sbatch --parsable \
    --job-name="pile-$ds" \
    --partition=gpu \
    --gres="gpu:${GPU_TYPE}:1" \
    "${NODELIST[@]}" \
    --cpus-per-task="$CPUS" \
    --mem="$MEM" \
    --time="$TIME" \
    --output="$LOGS/pile-$ds-%j.out" \
    --wrap "bash -lc '$BUILDENV && python build_pile.py --datasets $ds ${BUILD_ARGS[*]}'")
  # An empty job id means sbatch silently refused the request -- treat it as a
  # failure rather than reporting a launch that never happened (LESSONS.md).
  if [[ -z "$jid" ]]; then
    echo "FAILED to submit $ds (empty job id)" >&2
    exit 1
  fi
  echo "submitted $ds -> job $jid  (gpu:$GPU_TYPE${VTS_GPU_NODE:+ on $VTS_GPU_NODE}, log: $LOGS/pile-$ds-$jid.out)"
done

echo
echo "built from: $REPO @ ${REPO_STAMP_COMMIT:0:9} (${REPO_STAMP_BRANCH:-unknown})"
echo "watch:   squeue -u $USER"
echo "verify:  cd $HERE && python build_pile.py --verify"
