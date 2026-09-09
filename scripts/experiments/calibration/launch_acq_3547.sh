#!/usr/bin/env bash
# #3547: does the acquisition offset's optimum move DEEPER as a labelling
# session runs on -- asked in a haystack deep enough to answer it.
#
#   bash launch_acq_3547.sh prepare        # stage 0, ONCE (already done)
#   bash launch_acq_3547.sh size [idx] [arm]
#   bash launch_acq_3547.sh arms 16        # the grid
#
# WHAT #3319 LEFT OPEN.  It ran the first 400-click wave this constant has ever
# had and found its aggressive arms harvesting 82-85% of the positives their sim
# half held, against 15% for the control.  The confound is ONE-SIDED, so the win
# survived it -- a ceiling only the aggressive arms reach can compress their
# advantage, not manufacture it.  What it destroyed is the deep question:
# "does the optimum get DEEPER at depth?" is a question about the last quarter
# of a trajectory, and for -3/-4 the last quarter is where the pool runs out.
#
# THE ENVIRONMENT IS THE FIX, and it is `vg_scale_deep` (#3547): the same twelve
# classes, the same COCO-anchored labels, the same human corrections, designated
# BAND-FREE and three times deeper -- 900 positives per class against 11,700
# negatives.  Two properties of that number matter more than its size:
#
#   * PREVALENCE IS HELD at vg_scale's designed 7.1429%, because `k* =
#     -log2((1-pi)/pi)` is the quantity this study is trying to locate.  900
#     against the OLD 3900 negatives would have moved k* from -3.71 to -2.11 --
#     a 1.6-bit shift introduced by the fix, and invisible in the cells.
#   * 900 IS THE DEEPEST VALUE ALL TWELVE CLASSES SUPPORT (`stop sign`, 1006
#     band-free candidates).  1200 drops `kite` and `stop sign`, and a class
#     list differing from #3319's would confound the horizon axis with a
#     vocabulary axis in the one comparison this study exists to make.
#
# `preflight.sh` check 16b is now CLEARED rather than argued around: the sim
# half holds 450 positives against a 400-step horizon.  The prepare confirms it
# per category -- "deep-category filter (>= 400 sim positives): kept 12".
#
# ONE WAVE, BOTH HORIZONS.  `max_steps` reaches the simulation as a loop bound
# and nothing inside the loop reads it (`voting_iterations.py:1633`, its only
# use), so a 400-step trajectory is a strict EXTENSION of the 100-step one.  The
# optimum at t=100 and at t=400 therefore come off the SAME cells, which is what
# makes "does the optimum move?" a within-cell paired question instead of a
# contrast between two waves carrying every difference between the runs.
# CONFIRMED EMPIRICALLY before this grid was written, not read off the loop:
# 6336 cells per arm, all four arms #3319's two waves share, IDENTICAL at t=100
# on cost, n_good, thresholds and acq_pool_percentile (`check_prefix_3547.py`).
# That is why #3319 needed a shallow wave and this needs none.
#
# ARMS.  Seven.  Half steps are NOT run: #3319 measured them as real operating
# points and DECISION-IRRELEVANT (zero of six cost contrasts resolved) and
# retired them as a tuning device.  `-2` is dropped for the same reason `-1` is
# kept -- the grid needs the plateau's EDGES for the H3 shape check, and one
# more interior arm buys nothing `-3` and `-4` do not already carry.
#
#   prod      k =  0   control; clicks-to-target is defined against its final cost
#   acq_m1    k = -1   the pre-#3318 shipped value - the plateau's shallow edge
#   acq_m3    k = -3   the pre-#3319 shipped value; the arm whose guardrail fired
#   acq_m4    k = -4   THE SHIPPED DEFAULT (#3319)
#   acq_m5    k = -5   free on binary at 100 clicks, fails region (#3319)
#   acq_m6    k = -6   the plateau's measured deep edge at 100 clicks (+0.011)
#   acq_p2    k = +2   FALSIFICATION - must make positives worse
#
# KEEP `acq_p2`.  #3319's deep wave omitted it and its analyzer WITHHELD THE
# VERDICT in the generated report ("Falsification arm missing -- verdict
# withheld").  That is the control working, and it is not omitted here.
#
# Design + pre-registered decision rules: `PLAN_3547.md`.  Decision endpoints are
# clicks-to-target and trajectory AUC (#3319's lesson: `final_cost` was flat
# across three bits while speed separated the plateau's edges cleanly); H1 is a
# difference-in-differences across horizons, paired within the cell, rather than
# an argmin over a plateau already known to be unresolvable.
set -uo pipefail
trap 'echo "ABORTED: $0 line $LINENO exited $? -- NOTHING WAS SUBMITTED" >&2' ERR

MODE="${1:-arms}"

export VTS_REPO="${VTS_REPO:-/exp/$USER/projects/vts-acq-3547}"
WT="$VTS_REPO"
HERE="$WT/scripts/experiments/calibration"

# Not /exp: that 50G quota is mostly the venv (GRID-PLAYBOOK section 4).
BASE="${ACQ_BASE:-/expscratch/$USER/acq-3547}"
PREP="${ACQ_PREP:-$BASE/prepare/results}"

ARMS_ALL="${ACQ_ARMS:-prod,acq_m1,acq_m3,acq_m4,acq_m5,acq_m6,acq_p2}"

# arm -> CALIB_ACQ_INCLUSION_OFFSET.  Nothing is left unset: an unset offset
# resolves to the shipped constant, which would silently make one arm a
# duplicate of `acq_m4` the day that constant moves again.
arm_offset() {
  case "$1" in
    prod)    echo "0" ;;
    acq_m1)  echo "-1" ;;
    acq_m3)  echo "-3" ;;
    acq_m4)  echo "-4" ;;
    acq_m5)  echo "-5" ;;
    acq_m6)  echo "-6" ;;
    acq_p2)  echo "2" ;;
    *)       echo "__BAD__" ;;
  esac
}
arm_dir() { printf '%s/bin/%s' "$BASE" "$1"; }

# --- science knobs -----------------------------------------------------------
# The SHIPPED threshold path.  The harness default is the #2781-era unfused
# control and the acquisition cut is taken off the FUSED threshold, so leaving
# this alone would sweep an arm axis that never executes.
export CALIB_SAFE_THRESHOLDS=1
# Every arm chains `noop.py`; the analysis is CROSS-ARM and submitted once.
export CALIB_ANALYZE=noop.py
# CALIB_HEAD unset -> the production linear SVM (#3198).
# CALIB_CALIBRATION_FRACTION unset -> the per-space default (#3290).
# CALIB_EXCLUDE_VOTED unset -> the app's own floor (#3308).
# CALIB_BLEND_SCHEDULE unset -> `production_schedule_for` picks per voting mode.

# --- environment (#2877's, verbatim, except the pile) ------------------------
export CALIB_DATASETS="${CALIB_DATASETS:-vg_scale_deep}"
export CALIB_VGSCALE_DEEP_EMBEDDERS="${CALIB_VGSCALE_DEEP_EMBEDDERS:-siglip}"
export CALIB_REQUIRE_OPENING=text
export CALIB_REQUIRE_SEED_QUERY=1
export CALIB_CATEGORY_MODE="${CALIB_CATEGORY_MODE:-prevalence}"
export CALIB_N_CATEGORIES="${CALIB_N_CATEGORIES:-12}"
# BOTH styles declared, exactly as #3319 and #2877 declare them -- and only
# `whole_image` executes, because the embedder is `siglip` and the harness
# resolves a patch style against the embedder's capability (#3319's `bin` half
# logs `styles=['whole_image']` off this same declaration).  Pinning the bare
# `whole_image` instead is NOT equivalent: preflight check 12 reads it as an
# UNDECLARED divergence from a production default whose style set contains
# `max_patch`, and it is right to -- a study that pins the style axis is
# sweeping it, and this one is not.  So the set is declared and the cell
# resolves it; no patch cell of this dataset exists to run anyway (a
# `dinov3_patch` grid at 22k medias is ~7 GB).
export CALIB_PATCH_STYLES="${CALIB_PATCH_STYLES:-whole_image,max_patch}"
export CALIB_REPOOL_VARIANTS=""
export CALIB_SCHEDULE_VARIANTS=""
export CALIB_MAX_STEPS="${CALIB_MAX_STEPS:-400}"
export CALIB_SIM_FRACTION="${CALIB_SIM_FRACTION:-0.5}"
# A TRIPWIRE, not a filter: every vg_scale_deep cell holds 450 sim positives by
# construction, so this never excludes a category -- it fails loudly if the pile
# is ever rebuilt thinner, rather than letting a 400-click arm quietly average
# over cells whose tail is flat because they ran OUT of positives.  That is
# exactly the failure #3319 shipped, and it is the reason this study exists.
export CALIB_MIN_SIM_POSITIVES="${CALIB_MIN_SIM_POSITIVES:-400}"
# Preflight's floor FOLLOWS the tripwire rather than repeating it. Written as a
# literal `400` it silently contradicts any run that lowers the tripwire on
# purpose -- which the H2 control does, since its whole job is to re-run this
# grid on the SHALLOW pile where 150 sim positives is the fact under test.
# A number restated beside a constant goes stale; #3319 hit that three times.

# Seed-major, and DECLARED at 24 while the study runs a prefix of 16.
export CALIB_CELL_ORDER=seed
export CALIB_N_SEEDS="${CALIB_N_SEEDS:-24}"

# --- ops ---------------------------------------------------------------------
export CALIB_PARTITION=cpu
export CALIB_GRES=none
export CALIB_CPUS=1
export CALIB_ANALYZE_MEM="${CALIB_ANALYZE_MEM:-64G}"
export CALIB_ANALYZE_TIME="${CALIB_ANALYZE_TIME:-3:00:00}"

# MEASURED ON THIS GRID on 2026-09-02 by `size` (job 609833), on the DEEPEST arm
# (k=-6, the one that pays the most per step), NOT scaled from #3319's 10m20s:
# the sim half is 3x deeper and the per-step pool scoring runs over all of it.
#
#   vg_scale_deep x siglip x whole_image, 400 steps:  19m33s, 1.26 GB
#
# The time request is 1h against a measured 20m -- 3x, because an over-request
# costs backfill priority and an under-request costs the CELL, hours in, after
# the array has been running long enough to look healthy.  Memory is 3G against
# a measured 1.26 GB for the same asymmetry (sacct's MaxRSS is SAMPLED, so it
# can miss a short peak).
export CALIB_MEM="${ACQ_MEM:-3G}"
export CALIB_TIME="${ACQ_TIME:-1:00:00}"
# 7 arms x %14 = 98 concurrent.  `cpu_limit` charges 2 cpu per task (240 => 120
# tasks), so this claims ~82% of the task budget and 294G of a 1074G memory
# quota.  At 20 min a cell that puts 1344 cells at roughly 4.6 hours of wall
# clock; a tighter throttle would only lengthen it.
export CALIB_CONC="${ACQ_CONC:-14}"

# Pin the BLAS pools to one thread each: concurrent single-cpu cells each
# spawning a node-sized pool oversubscribe whatever node they land on.  Exported
# HERE so `size` and `arms` measure and run under the SAME environment.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

# Read the pre-embedded pile in place: no re-embed, no GPU, no model download.
# shellcheck disable=SC1091
source "$WT/scripts/experiments/pile/pile_env.sh"

require_jobid() {
  local id="$1" what="$2"
  if ! [[ "$id" =~ ^[0-9]+$ ]]; then
    echo "ERROR: $what was REFUSED by sbatch (no job id came back)." >&2
    echo "       Nothing downstream can run; fix the submission and re-launch." >&2
    exit 1
  fi
}

link_prepare() {
  local rd="$1"
  mkdir -p "$rd/cells"
  [[ -e "$rd/prepare_info.json" ]] || ln -s "$PREP/prepare_info.json" "$rd/prepare_info.json"
  [[ -e "$rd/crops" ]] || ln -s "$PREP/crops" "$rd/crops"
}

# How many (dataset, embedder, category) environments this grid enumerates per
# seed.  Read from the harness rather than assumed: a seed block's width is what
# the array spec is built out of, and hardcoding 12 would silently mis-slice the
# array the day a category is dropped for want of a query.
envs_per_seed() {
  local n
  n=$(cd "$HERE" && CALIB_RESULTS="$PREP" CALIB_N_SEEDS=1 python run_cells.py --print-cells 2>/dev/null | tail -1)
  if ! [[ "$n" =~ ^[0-9]+$ ]] || [[ "$n" -eq 0 ]]; then
    echo "ERROR: could not determine the per-seed environment count (got '$n')" >&2
    exit 1
  fi
  echo "$n"
}

case "$MODE" in
  prepare)
    export CALIB_EXP="$BASE/prepare"
    export CALIB_RESULTS="$PREP"
    mkdir -p "$BASE/prepare/logs" "$PREP/cells" "$PREP/crops"
    ENVX="export CALIB_EXP=$CALIB_EXP CALIB_RESULTS=$CALIB_RESULTS CALIB_VGSCALE_DEEP_EMBEDDERS=$CALIB_VGSCALE_DEEP_EMBEDDERS VTSEARCH_DATA_DIR=$VTSEARCH_DATA_DIR VTSEARCH_MODELS_DIR=$VTSEARCH_MODELS_DIR HF_HOME=$HF_HOME"
    P=$(sbatch --parsable --job-name=acq3547-prep --mem=48G --cpus-per-task=2 \
      --time=3:00:00 --partition=cpu --export=ALL \
      --output="$BASE/prepare/logs/prepare-%j.out" \
      --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && python prepare_data.py")
    require_jobid "$P" "prepare"
    echo "prepare job: $P  ->  $BASE/prepare/logs/prepare-$P.out"
    ;;

  size)
    IDX="${2:-0}"
    ARM="${3:-acq_m6}"
    OFF="$(arm_offset "$ARM")"
    [[ "$OFF" == "__BAD__" ]] && { echo "ERROR: unknown arm '$ARM'" >&2; exit 2; }
    export CALIB_ACQ_INCLUSION_OFFSET="$OFF"
    export CALIB_EXP="$BASE/sizing"
    export CALIB_RESULTS="$PREP"
    SIZING="$BASE/sizing/bin-$ARM"
    mkdir -p "$BASE/sizing/logs" "$SIZING"
    ENVX="export CALIB_EXP=$CALIB_EXP CALIB_RESULTS=$CALIB_RESULTS CALIB_VGSCALE_DEEP_EMBEDDERS=$CALIB_VGSCALE_DEEP_EMBEDDERS CALIB_ACQ_INCLUSION_OFFSET=$CALIB_ACQ_INCLUSION_OFFSET VTSEARCH_DATA_DIR=$VTSEARCH_DATA_DIR VTSEARCH_MODELS_DIR=$VTSEARCH_MODELS_DIR HF_HOME=$HF_HOME"
    S=$(sbatch --parsable --job-name=acq3547-size --mem="$CALIB_MEM" --cpus-per-task="$CALIB_CPUS" \
      --time=6:00:00 --partition=cpu --export=ALL \
      --output="$BASE/sizing/logs/size-%j.out" \
      --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && time python run_cells.py --index $IDX --outdir $SIZING")
    require_jobid "$S" "size"
    echo "size job: $S (cell $IDX, arm $ARM)  ->  $BASE/sizing/logs/size-$S.out"
    echo "read it with: sacct -j $S --format=JobID,JobName%18,MaxRSS,Elapsed,State"
    ;;

  arms)
    if [[ ! -f "$PREP/prepare_info.json" ]]; then
      echo "ERROR: no prepare_info.json at $PREP - run '$0 prepare' first." >&2
      exit 1
    fi
    # Preflight's checks are mostly PYTHON: it imports vtscore to compare every
    # pinned knob against its shipped constant.  A non-interactive login shell
    # has no venv, and the system python is old enough that `X | None` raises at
    # import -- so those checks come back FAIL for a reason unrelated to the run.
    # shellcheck disable=SC1091
    source "$WT/gridenv.sh" >/dev/null 2>&1 || {
      echo "ERROR: could not activate the venv at $WT/gridenv.sh" >&2; exit 1
    }

    SEEDSPEC="${2:-16}"
    if [[ "$SEEDSPEC" == *-* ]]; then
      SLO="${SEEDSPEC%%-*}"; SHI="${SEEDSPEC##*-}"
    else
      SLO=0; SHI=$(( SEEDSPEC - 1 ))
    fi
    if (( SLO < 0 || SHI >= CALIB_N_SEEDS || SLO > SHI )); then
      echo "ERROR: seeds '$SEEDSPEC' fall outside 0..$((CALIB_N_SEEDS-1))" >&2; exit 2
    fi

    # The shipped constant, read from the tree that is about to run - once, and
    # never written down here.  Check 12 compares CALIB_ACQ_INCLUSION_OFFSET
    # against it, so this is what decides which arms have a divergence to
    # declare; a number in this file would go stale silently.  #3319 shipped -4,
    # so `acq_m4` now MATCHES production and the other six declare.
    SHIPPED_K=$(python -c 'from vtscore.training.thresholds import ACQUISITION_INCLUSION_OFFSET as k; print(k)')
    if ! [[ "$SHIPPED_K" =~ ^-?[0-9]+(\.[0-9]+)?$ ]]; then
      echo "ERROR: could not read ACQUISITION_INCLUSION_OFFSET (got '$SHIPPED_K')" >&2; exit 1
    fi
    echo "shipped ACQUISITION_INCLUSION_OFFSET = $SHIPPED_K"

    NENV="$(envs_per_seed)"
    LO=$(( SLO * NENV )); HI=$(( (SHI + 1) * NENV - 1 ))
    NPER=$(( HI - LO + 1 ))
    n_arms=0; for _a in ${ARMS_ALL//,/ }; do n_arms=$((n_arms + 1)); done
    # Preflight's memory check asks "does this claim your whole allowance?", and
    # for a multi-array study the honest answer is about the SUM.  Passing one
    # arm's %N would report a fraction and wave through a study at 90% of quota.
    study_conc=$(( CALIB_CONC * n_arms ))
    echo "=== $NENV envs/seed, seeds ${SLO}..${SHI} of $CALIB_N_SEEDS declared"
    echo "    array ${LO}-${HI} = $NPER cells/arm, $n_arms arms = $(( NPER * n_arms )) cells"
    echo "    $n_arms arms x %$CALIB_CONC x $CALIB_MEM = %$study_conc concurrent"

    DEPS=()
    for ARM in ${ARMS_ALL//,/ }; do
      OFF="$(arm_offset "$ARM")"
      [[ "$OFF" == "__BAD__" ]] && { echo "ERROR: unknown arm '$ARM'" >&2; exit 1; }
      export CALIB_ACQ_INCLUSION_OFFSET="$OFF"
      export CALIB_EXP; CALIB_EXP="$(arm_dir "$ARM")"
      export CALIB_RESULTS="$CALIB_EXP/results"
      mkdir -p "$CALIB_EXP/logs"
      link_prepare "$CALIB_RESULTS"

      JOB_NAME="acq3547-bin-$ARM${ACQ_JOB_TAG:+-$ACQ_JOB_TAG}"
      DIV=()
      [[ "$OFF" != "$SHIPPED_K" ]] && DIV=(--diverges "acq_offset")

      if [[ -x "$WT/scripts/experiments/preflight.sh" ]]; then
        bash "$WT/scripts/experiments/preflight.sh" --exp "$CALIB_EXP" --need-gb 40 \
          --require-min-positives "${ACQ_MIN_POSITIVES:-$CALIB_MIN_SIM_POSITIVES}" \
          --reuse-prepare "$PREP" \
          "${DIV[@]}" \
          --job-name "$JOB_NAME" --mem "$CALIB_MEM" --conc "$study_conc" || {
          echo "preflight FAILED for arm $ARM" >&2
          [[ "${PREFLIGHT_SKIP:-0}" == "1" ]] || exit 1
        }
      fi

      ENVX="export CALIB_EXP=$CALIB_EXP CALIB_RESULTS=$CALIB_RESULTS"
      ENVX="$ENVX CALIB_VGSCALE_DEEP_EMBEDDERS=$CALIB_VGSCALE_DEEP_EMBEDDERS"
      ENVX="$ENVX CALIB_ACQ_INCLUSION_OFFSET=$CALIB_ACQ_INCLUSION_OFFSET"
      ENVX="$ENVX VTSEARCH_DATA_DIR=$VTSEARCH_DATA_DIR VTSEARCH_MODELS_DIR=$VTSEARCH_MODELS_DIR HF_HOME=$HF_HOME"

      J=$(sbatch --parsable --job-name="$JOB_NAME" --array="${LO}-${HI}%${CALIB_CONC}" \
        --mem="$CALIB_MEM" --cpus-per-task="$CALIB_CPUS" --time="$CALIB_TIME" \
        --partition="$CALIB_PARTITION" --export=ALL \
        --output="$CALIB_EXP/logs/cells-%A_%a.out" \
        --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && python run_cells.py")
      require_jobid "$J" "arm $ARM's cells array"
      echo "--- $JOB_NAME (k=$OFF) job=$J -> $CALIB_EXP"
      echo "$J" > "$CALIB_EXP/logs/.cells_jobid"
      DEPS+=("$J")
    done

    # ONE analysis, after every array drains.  `afterany` rather than `afterok`
    # on purpose: an arm that loses cells to a node failure still has to be read
    # and its loss COUNTED, and an analyzer that never runs reports nothing.
    #
    # NOTE it writes to `analysis`, and #3319's launcher-chained analyze
    # SILENTLY CLOBBERED a hand-run one -- a default-scoped table that parsed
    # cleanly and had lost half the arms.  `scancel --name=acq3547-analyze`
    # before reading anything you analysed by hand.
    DEPSTR="$(IFS=:; echo "${DEPS[*]}")"
    ALOGS="$BASE/logs"; mkdir -p "$ALOGS"
    AENVX="export VTSEARCH_DATA_DIR=$VTSEARCH_DATA_DIR VTSEARCH_MODELS_DIR=$VTSEARCH_MODELS_DIR HF_HOME=$HF_HOME"
    A=$(sbatch --parsable --dependency="afterany:$DEPSTR" --job-name=acq3547-analyze \
      --mem="$CALIB_ANALYZE_MEM" --cpus-per-task=4 --time="$CALIB_ANALYZE_TIME" \
      --partition=cpu --export=ALL --output="$ALOGS/analyze-%j.out" \
      --wrap="source $WT/gridenv.sh && $AENVX && cd $HERE && python analyze_acq.py --base $BASE --halves bin --out $BASE/analysis")
    require_jobid "$A" "the cross-arm analyze step"

    echo
    echo "Submitted ${#DEPS[@]} arrays: ${DEPS[*]}"
    echo "cross-arm analyze: $A  ->  $ALOGS/analyze-$A.out"
    echo "report -> $BASE/analysis/REPORT_acq.md"
    echo
    echo "A submission is not a launch: confirm the ids above are numeric and that"
    echo "cells appear under $BASE/bin/*/results/cells before quoting an ETA."
    ;;

  *)
    echo "usage: $0 {prepare|size [cell] [arm]|arms [seeds]}" >&2
    exit 2
    ;;
esac
