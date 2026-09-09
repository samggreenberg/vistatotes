#!/usr/bin/env bash
# Preflight for a GRID experiment launch.
#
# Every check here corresponds to a mistake that actually cost hours on a real
# study.  They are checks, not advice, because advice in a document did not stop
# any of them: the document existed and nobody read it.  Run this immediately
# before submitting arms.
#
#   bash scripts/experiments/preflight.sh --exp /exp/$USER/my-study --arms a,b,c
#
# Exits non-zero if anything is wrong.  `--warn-only` downgrades failures to
# warnings for the cases where you genuinely mean it (resuming a partial run).
#
# See scripts/experiments/GRID-PLAYBOOK.md for the SLURM-resource side of this
# (memory sizing, QOS caps, chunking) and LESSONS.md / lessons/ for the incident log.
set -uo pipefail

EXP=""
ARMS=""
NEED_GB="${PREFLIGHT_NEED_GB:-5}"
WARN_ONLY=0
REPO="${VTS_REPO:-}"
REGION_ARM=""
MIN_POSITIVES=""
REQUIRE_TEXT_SEED=""
MODE_CONTRAST=""
REUSE_PREPARE=""
JOB_NAME=""
MEM_PER_TASK=""
HAS_PATCH_CELLS=0
CONC=""
DIVERGES="${PREFLIGHT_DIVERGES:-}"
HARVEST_BAR=""
PILOT_CELLS=""

# This script's own directory, so check 16c can reach its sibling sizing script
# without depending on VTS_REPO -- which check 4 may already have failed on.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --exp) EXP="$2"; shift 2 ;;
    --arms) ARMS="$2"; shift 2 ;;
    --need-gb) NEED_GB="$2"; shift 2 ;;
    --require-region-voting) REGION_ARM="$2"; shift 2 ;;
    --require-min-positives) MIN_POSITIVES="$2"; shift 2 ;;
    --require-text-seed) REQUIRE_TEXT_SEED=1; shift ;;
    --contrasts-voting-modes) MODE_CONTRAST=1; shift ;;
    --reuse-prepare) REUSE_PREPARE="$2"; shift 2 ;;
    --job-name) JOB_NAME="$2"; shift 2 ;;
    --diverges) DIVERGES="$2"; shift 2 ;;
    --mem) MEM_PER_TASK="$2"; shift 2 ;;
    --patch) HAS_PATCH_CELLS=1; shift ;;
    --conc) CONC="$2"; shift 2 ;;
    --require-harvest-headroom) HARVEST_BAR="$2"; shift 2 ;;
    --pilot-cells) PILOT_CELLS="$2"; shift 2 ;;
    --warn-only) WARN_ONLY=1; shift ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$EXP" ]] || {
  echo "usage: preflight.sh --exp DIR [--arms a,b,c] [--need-gb N]" >&2
  echo "                    [--require-region-voting DATASET:EMBEDDER]" >&2
  echo "                    [--require-text-seed]      # every cell must seed from a TYPED QUERY" >&2
  echo "                    (or declare it once with CALIB_REQUIRE_OPENING=text|known_good|mixed," >&2
  echo "                     which run_cells.py also asserts per cell)" >&2
  echo "                    [--reuse-prepare RESULTS_DIR]" >&2
  echo "                    [--require-harvest-headroom BAR]  # the pre-registered compression bar," >&2
  echo "                    [--pilot-cells DIR]               # sized off a pilot of the DEEPEST arm" >&2
  echo "                    (or declare both once with CALIB_HARVEST_BAR / CALIB_HARVEST_PILOT)" >&2
  echo "                    [--job-name NAME] [--mem 64G] [--conc N] [--patch]" >&2
  echo "                    [--diverges knob1,knob2]   # knobs this study MEANS to pin off-production" >&2
  exit 2
}

# The opening this run asserts, from either half of one declaration: the flag a
# launcher passes here, or `CALIB_REQUIRE_OPENING` -- which `run_cells.py`
# asserts per CELL as well, so the pre-array gate and the per-cell guard cannot
# disagree about what the study meant (#3278).  `mixed` declares a grid that
# deliberately holds both openings, so there is nothing uniform to check here.
WANT_OPENING=""
[[ -n "$REQUIRE_TEXT_SEED" ]] && WANT_OPENING="text"
case "${CALIB_REQUIRE_OPENING:-}" in
  text) WANT_OPENING="text" ;;
  known_good)
    if [[ "$WANT_OPENING" == "text" ]]; then
      echo "--require-text-seed contradicts CALIB_REQUIRE_OPENING=known_good; pick one" >&2
      exit 2
    fi
    WANT_OPENING="known_good" ;;
  mixed|"") ;;
  *) echo "CALIB_REQUIRE_OPENING=${CALIB_REQUIRE_OPENING} is not text|known_good|mixed" >&2; exit 2 ;;
esac

# Check 16c's declaration, from either half of it: the flags a launcher passes
# here, or the two `CALIB_*` names -- which a study can export beside its other
# knobs, so the bar it pre-registered reaches every preflight call in its launch
# loop without the invocation line having to carry it.
HARVEST_BAR="${HARVEST_BAR:-${CALIB_HARVEST_BAR:-}}"
PILOT_CELLS="${PILOT_CELLS:-${CALIB_HARVEST_PILOT:-}}"

FAILED=0
say_fail() {
  if [[ "$WARN_ONLY" == "1" ]]; then echo "  WARN  $*"; else echo "  FAIL  $*"; FAILED=1; fi
}
say_ok() { echo "  ok    $*"; }
say_note() { echo "  note  $*"; }

# Memory strings ("14G", "900M") to MB.  Defined up here beside the other
# helpers because bash resolves a function only once execution reaches its
# definition: while this lived further down, the patch-memory check above it
# called an undefined name, `mem_mb` came back empty, and `(( "" < 12288 ))`
# evaluated the empty string as 0 -- so a correctly-sized `--mem 14G` was
# reported as "(< 12G)".  The gate failed closed, which is the safe direction,
# but it failed for a reason its own message contradicted.
_to_mb() {
  local v="${1^^}"; local n="${v%[GMT]*}"
  case "$v" in *T) awk "BEGIN{print $n*1024*1024}";;
                *G) awk "BEGIN{print $n*1024}";;
                *M) echo "$n";;
                *)  echo "$n";; esac
}

echo "preflight: $EXP"

# --- 1. One experiment, one results dir -------------------------------------
# Two grids pointed at the same CALIB_EXP once.  The resume logic read the other
# grid's cells as "this arm is already complete" and aborted a whole overnight
# batch; had it not aborted, two different grids would have been silently mixed
# in one directory and analysed as one.
#
# Arm roots differ by study: the A/B launchers put arms under `results-ab/`, the
# acquisition and anchor sweeps under `results/`.  Checking only the first meant
# this check silently passed — did nothing at all — for every study of the second
# shape, which is the worse failure of the two: a gate that reports "ok" without
# having looked.
if [[ -n "$ARMS" ]]; then
  for arm in ${ARMS//,/ }; do
    seen=0
    for root in results-ab results; do
      cells="$EXP/$root/$arm/cells"
      [[ -d "$cells" ]] || continue
      seen=1
      n=$(find "$cells" -name 'task_*.csv' ! -name '*sweep*' 2>/dev/null | wc -l)
      if [[ "$n" -gt 0 ]]; then
        say_fail "arm '$arm' already has $n cell files in $cells"
        echo "        -> a fresh study needs its own --exp dir; a resume should pass --warn-only"
      else
        say_ok "arm '$arm' results dir is empty ($root/)"
      fi
    done
    [[ "$seen" == "1" ]] || say_ok "arm '$arm' results dir is new"
  done
fi

# --- 2. Free space on the REAL mount ----------------------------------------
# `df -h /exp` reported 394G free while the actual home, /exp/$USER, was its own
# 50G mount at 100%.  Cells died mid-write for hours on that misread.
# A fresh study's dir does not exist yet - that is the normal case - so stat the
# nearest existing ancestor, which is on the same filesystem.
STAT_PATH="$EXP"
while [[ ! -e "$STAT_PATH" && "$STAT_PATH" != "/" ]]; do
  STAT_PATH=$(dirname "$STAT_PATH")
done
MOUNT=$(df -P "$STAT_PATH" 2>/dev/null | awk 'NR==2 {print $6}')
AVAIL_GB=$(df -PBG "$STAT_PATH" 2>/dev/null | awk 'NR==2 {gsub(/G/,"",$4); print $4}')
if [[ -z "$AVAIL_GB" ]]; then
  say_fail "could not stat $STAT_PATH"
else
  if [[ "$AVAIL_GB" -lt "$NEED_GB" ]]; then
    say_fail "only ${AVAIL_GB}G free on $MOUNT (want >= ${NEED_GB}G)"
  else
    say_ok "${AVAIL_GB}G free on $MOUNT (the mount that actually holds $EXP)"
    # The mistake this guards against: reading a *parent* mount's free space.
    # /exp showed 394G while /exp/$USER was its own 50G mount at 100%.
    parent_mount=$(df -P "$(dirname "$STAT_PATH")" 2>/dev/null | awk 'NR==2 {print $6}')
    if [[ -n "$parent_mount" && "$parent_mount" != "$MOUNT" ]]; then
      echo "        note: $(dirname "$STAT_PATH") is a DIFFERENT mount ($parent_mount) - its free space is irrelevant"
    fi
  fi
fi

# --- 3. Zero-byte and header-only cells from a previous incident ------------
# A cell killed mid-write leaves a 0-byte CSV.  It counts as "present" to the
# resume logic, so it is never re-run, and it crashes or silently shrinks the
# analysis later.
#
# Its quieter twin is the header-only cell: a category that never collects both
# classes writes the CSV header and nothing else.  That file is non-empty and
# parses cleanly, so `-size 0` clears it and every "N/N cells" count calls it
# present.  Check 13 predicts the condition from prepare_info.json *before* a
# launch; this is the after-the-fact half, because a resumed grid inherits
# whatever the previous attempt left behind.  Some are legitimate (a genuinely
# thin category has no trainable step - the #3156 overview ended with 23 such
# cells of 6480), so this REPORTS a count rather than blocking on it.
ZROOTS=()
for root in results-ab results; do [[ -d "$EXP/$root" ]] && ZROOTS+=("$EXP/$root"); done
if [[ "${#ZROOTS[@]}" -gt 0 ]]; then
  z=$(find "${ZROOTS[@]}" -name 'task_*.csv' ! -name '*__*' -size 0 2>/dev/null | wc -l)
  if [[ "$z" -gt 0 ]]; then
    say_fail "$z zero-byte cell files present - delete them or they will never be re-run"
  else
    say_ok "no zero-byte cell files"
  fi
  # `! -name '*__*'` excludes every side frame at once (__sweep, __cutdiag,
  # __cutincl - see _cells_io.SIDE_FRAME_SUFFIXES).  Naming them one at a time
  # is how this check drifted: it still said `! -name '*sweep*'` two side frames
  # later, and each of those is a long-format table that is *legitimately*
  # header-only, so on #3156 the honest answer of 23 came back as 12983.
  #
  # `-size -2k -size +0` narrows to plausible candidates before the per-file
  # `wc -l`, so the fork runs over a handful and not the whole grid: a cell CSV
  # holding even one run is far larger than its ~700-byte header.
  h=$(find "${ZROOTS[@]}" -name 'task_*.csv' ! -name '*__*' -size -2k -size +0 \
        -exec sh -c '[ "$(wc -l < "$1")" -le 1 ]' _ {} \; -print 2>/dev/null | wc -l)
  if [[ "$h" -gt 0 ]]; then
    say_note "$h header-only cell file(s) - present, parse clean, carry no data row (see check 13)"
  else
    say_ok "no header-only cell files"
  fi
fi

# --- 4. The worktree the jobs will actually import ---------------------------
# common.setup_env() puts VTS_REPO at the front of sys.path.  Unset, jobs
# silently import a different (stale) checkout.
if [[ -z "$REPO" ]]; then
  say_fail "VTS_REPO is unset - jobs will import whatever checkout the default points at"
elif [[ ! -d "$REPO/.git" && ! -f "$REPO/.git" ]]; then
  say_fail "VTS_REPO=$REPO is not a git worktree"
else
  say_ok "VTS_REPO=$REPO"
  if git -C "$REPO" rev-parse --abbrev-ref HEAD >/dev/null 2>&1; then
    branch=$(git -C "$REPO" rev-parse --abbrev-ref HEAD)
    git -C "$REPO" fetch -q origin 2>/dev/null
    local_sha=$(git -C "$REPO" rev-parse HEAD)
    if [[ "$branch" == "HEAD" ]]; then
      # Detached, which is how a run pins an exact commit. `--abbrev-ref` returns
      # the literal string "HEAD" there, so the old comparison silently became
      # "must equal origin/HEAD" - i.e. the default branch - and failed every
      # legitimate feature-branch run with "the code you committed is not the
      # code that will run". What actually matters is that the commit EXISTS on
      # origin, so it can be fetched and re-run later.
      if [[ -n "$(git -C "$REPO" branch -r --contains "$local_sha" 2>/dev/null)" ]]; then
        say_ok "detached at ${local_sha:0:8}, which is pushed to origin"
      else
        say_fail "detached at ${local_sha:0:8}, which is NOT on origin"
        echo "        -> the code that will run cannot be recovered from the remote"
      fi
    else
      remote_sha=$(git -C "$REPO" rev-parse "origin/$branch" 2>/dev/null || echo "")
      if [[ -n "$remote_sha" && "$local_sha" != "$remote_sha" ]]; then
        say_fail "worktree is not at origin/$branch (local ${local_sha:0:8}, remote ${remote_sha:0:8})"
        echo "        -> the code you committed is not the code that will run"
      else
        say_ok "worktree matches origin/$branch"
      fi
    fi
    if [[ -n "$(git -C "$REPO" status --porcelain --untracked-files=no)" ]]; then
      say_fail "worktree has uncommitted tracked changes - the run would be unreproducible"
    fi
    # How far behind the integration branch?  Check 12 compares the study's pins
    # against `PRODUCTION_*`, but it reads those constants **out of this same
    # worktree** - so a checkout that predates a production change has a stale
    # pin AND a stale constant, they agree, and check 12 says ok.  That is how
    # #3156's 6480-cell overview trained the retired `linear` head from a base
    # 321 commits behind dev: nothing was pinned wrong, the baseline had simply
    # moved.  Distance from origin/dev is the only signal that catches it, and
    # it is cheap.
    BASE_BRANCH="${PREFLIGHT_BASE_BRANCH:-dev}"
    if git -C "$REPO" rev-parse --verify -q "origin/$BASE_BRANCH" >/dev/null 2>&1; then
      behind=$(git -C "$REPO" rev-list --count "HEAD..origin/$BASE_BRANCH" 2>/dev/null || echo 0)
      if [[ "$behind" -ge "${PREFLIGHT_MAX_BEHIND:-100}" ]]; then
        say_fail "worktree is $behind commits behind origin/$BASE_BRANCH"
        echo "        -> every PRODUCTION_* constant check 12 reads is that old too, so it"
        echo "           cannot see a baseline that moved. Rebase, or set PREFLIGHT_MAX_BEHIND."
      elif [[ "$behind" -gt 0 ]]; then
        say_note "worktree is $behind commits behind origin/$BASE_BRANCH (under the ${PREFLIGHT_MAX_BEHIND:-100} gate)"
      else
        say_ok "worktree is level with origin/$BASE_BRANCH"
      fi
    fi
  fi
fi

# --- 5. …and the checkout `import vtscore` ACTUALLY resolves to --------------
# VTS_REPO being right is not the same as the import being right.  #2846 launched
# from a fresh worktree with a correct VTS_REPO and correct PYTHONPATH and still
# imported the shared vts-calib checkout, via the venv's editable-install finder
# (the `.shadow` shim that neutralises it is untracked, so a new worktree has
# none).  That run only noticed because the branch had *added* a symbol; a branch
# that merely changes behaviour would have produced a clean, plausible, wrong
# table.  So resolve the import the way a job does - through common.setup_env() -
# and check where it landed.
# Checks 6, 7 and 12 are also python, and they import the same tree.  When this
# one fails there is no point running them: each re-derives the identical cause
# and prints it as a raw traceback, so the three of them bury the single line
# that says what to do.  Measured: launching without a venv on a non-interactive
# ssh (where the system python is too old for `X | None` at import time) turned
# one actionable failure into thirty lines of stack.
PY_USABLE=1
if [[ -n "$REPO" && -d "$REPO/vtscore" ]]; then
  RESOLVED=$(CALIB_EXP="$EXP" python - "$REPO" <<'PY' 2>/dev/null
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(sys.argv[1]) / "scripts" / "experiments" / "calibration"))
import common
common.setup_env()
import vtscore
print(pathlib.Path(vtscore.__file__).resolve())
PY
)
  if [[ -z "$RESOLVED" ]]; then
    say_fail "could not resolve 'import vtscore' - is the venv active (source gridenv.sh)?"
    PY_USABLE=0
  else
    REPO_REAL=$(cd "$REPO" && pwd -P)
    case "$RESOLVED" in
      "$REPO_REAL"/*) say_ok "import vtscore -> $RESOLVED" ;;
      *)
        say_fail "import vtscore resolves to $RESOLVED"
        echo "        -> that is NOT $REPO_REAL; the jobs would measure another checkout"
        echo "        -> source this worktree's gridenv.sh (it creates .shadow and pins VTS_REPO)"
        ;;
    esac
  fi
fi

# --- 6. The environment's PREMISE, not the flag that requests it -------------
# #2877 ran a whole study on `visual_genome_m x siglip` believing it was region
# voting.  It was not: `region_voting=True` is a *request*, and the harness
# silently falls back to whole-image training, whole-image scoring and the
# binary blend schedule when the medias carry no `patch_grid`.  Nothing was
# broken, so nothing complained; a report, a PR and a headline recommendation
# had to be corrected.  A flag you passed is not a property you got.
#
# Opt-in, because most studies do not claim region voting — but any study whose
# *rationale* rests on the scoring geometry ("a max over region nodes") should
# pass it.  One pickle open, and it either holds or it does not.
if [[ -n "$REGION_ARM" && "$PY_USABLE" == "0" ]]; then
  say_fail "region-voting premise NOT checked: python cannot import the tree (see above)"
elif [[ -n "$REGION_ARM" ]]; then
  ds="${REGION_ARM%%:*}"; emb="${REGION_ARM##*:}"
  if [[ -z "$ds" || -z "$emb" || "$ds" == "$REGION_ARM" ]]; then
    say_fail "--require-region-voting wants DATASET:EMBEDDER, got '$REGION_ARM'"
  else
    VERDICT=$(CALIB_EXP="$EXP" python - "$REPO" "$ds" "$emb" <<'PY' 2>&1
import pathlib, sys
repo, ds, emb = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, str(pathlib.Path(repo) / "scripts" / "experiments" / "calibration"))
import common
common.setup_env()
import experiment_config as cfg
from _cells_io import load_medias
from vtscore.config import EMBEDDINGS_DIR

# A paired arm (`siglip+dinov3_patch`) carries its patch grid in the LEARN
# half's pickle; naming the file by hand here would look for a pickle that has
# never existed and report the premise as MISSING rather than as held.
pkl = EMBEDDINGS_DIR / cfg.pickle_name(ds, emb)
if not pkl.exists():
    print(f"MISSING {pkl}")
    raise SystemExit(0)
medias = load_medias(pkl)
n = len(medias)
grid = sum(1 for m in medias.values() if m.get("patch_grid") is not None)
print(("HOLDS" if grid == n and n else "FAILS") + f" patch_grid={grid}/{n} {pkl}")
PY
)
    case "$VERDICT" in
      HOLDS*) say_ok "region-voting premise ${ds} x ${emb}: ${VERDICT#HOLDS }" ;;
      FAILS*)
        say_fail "region-voting premise ${ds} x ${emb} does NOT hold: ${VERDICT#FAILS }"
        echo "        -> region_voting=True would silently run BINARY voting here (see #2877)"
        ;;
      *) say_fail "could not check the region-voting premise: $VERDICT" ;;
    esac
  fi
fi

# --- 7. Patch STYLES require box supervision --------------------------------
# Check 6 asks whether the region geometry is present.  This asks whether it is
# usable.  On a boxless dataset a Good vote has no box to pool, so it falls back
# to the image-level vector, while every Bad vote floods the full-image row plus
# ~197 raw patches as negatives.  No patch row is ever positive: the geometry can
# only teach "patch-like => negative", and max-pooling it at inference re-opens
# the asymmetry that produced perfect ranking, zero FPR and catastrophic FNR
# (see the module docstring in vtscore/eval/patch_styles.py).
#
# Reads the study's own config, so it needs no arguments and cannot drift from
# what the run will actually do.
if [[ -n "$REPO" && "$PY_USABLE" == "0" ]]; then
  say_fail "patch styles NOT checked: python cannot import the tree (see above)"
elif [[ -n "$REPO" && -f "$REPO/scripts/experiments/calibration/experiment_config.py" ]]; then
  STYLE_VERDICT=$(CALIB_EXP="$EXP" python - "$REPO" <<'PY' 2>&1
import pathlib, sys
repo = sys.argv[1]
sys.path.insert(0, str(pathlib.Path(repo) / "scripts" / "experiments" / "calibration"))
import experiment_config as cfg

if not hasattr(cfg, "styles_for"):
    print("SKIP config has no dataset-aware styles_for()")
    raise SystemExit(0)

bad = []
for ds in cfg.DATASETS:
    boxed = cfg.BOXED_BY_DATASET.get(ds, False)
    for emb in cfg.embedders_for_dataset(ds):
        styles = [st for st in cfg.styles_for(ds, emb) if st != "whole_image"]
        if styles and not boxed:
            bad.append(f"{ds}x{emb}={','.join(styles)}")
print(("FAILS " + "; ".join(bad)) if bad else "HOLDS")
PY
)
  case "$STYLE_VERDICT" in
    HOLDS*) say_ok "patch styles only on boxed datasets" ;;
    SKIP*)  say_ok "patch-style check skipped (${STYLE_VERDICT#SKIP })" ;;
    FAILS*)
      say_fail "patch styles on a BOXLESS dataset: ${STYLE_VERDICT#FAILS }"
      echo "        -> no Good vote can land on a patch row there; the patch rows are"
      echo "           negatives only, which is the boxless-max_patch failure mode"
      ;;
    *) say_fail "could not check patch styles: $STYLE_VERDICT" ;;
  esac
fi

# --- 7b. A region-voting cell needs region-voting memory ---------------------
# A max_patch cell carries the patch grid and max-pools over it; measured peaks
# are 9-14 GB depending on pool size (GRID-PLAYBOOK.md).  Sizing such an array
# from a whole-image cell is not a near miss -- it is a different order of
# magnitude, and the failure arrives as OUT_OF_MEMORY on most of the arm after
# the array has been running long enough to look healthy (#3156: 74 of 108).
if [[ "$HAS_PATCH_CELLS" == "1" && -n "$MEM_PER_TASK" ]]; then
  mem_mb=$(_to_mb "$MEM_PER_TASK")
  if (( mem_mb < 12288 )); then
    say_fail "patch cells requested with --mem $MEM_PER_TASK (< 12G)"
    echo "        -> measured max_patch peaks are 9-14 GB; see GRID-PLAYBOOK.md"
    echo "        -> size from a cell that actually resolved to a patch style,"
    echo "           not from one that fell back to whole_image"
  else
    say_ok "patch cells with --mem $MEM_PER_TASK (measured peaks 9-14 GB)"
  fi
fi

# --- 8. Your own per-user memory allowance ----------------------------------
# The cluster caps MEMORY per user, not only CPU.  An array that claims the whole
# allowance does not fail - it just parks every later job of YOUR OWN behind it in
# QOSMaxMemoryPerUser, which looks exactly like a busy cluster.  In #3129 this hit
# three times in one evening (a prepare job, a second array throttled to 2 slots,
# and five diagnostic jobs stuck 25 minutes), always self-inflicted.
#
# Size memory from a real cell's MaxRSS, not from a round number:
#   sacct -j <jobid> --format=JobID,MaxRSS,Elapsed
if [[ -n "$MEM_PER_TASK" && -n "$CONC" ]]; then
  req_mb=$(awk "BEGIN{print $(_to_mb "$MEM_PER_TASK") * $CONC}")
  # Two QOS can bind and they disagree: the job's association QOS and the
  # partition's.  In #3129 `squeue %q` said 4gpu_tier while the cpu partition
  # carried cpu_limit (mem=1100000M) - and it was the partition's that produced
  # QOSMaxMemoryPerUser.  Collect every candidate and use the TIGHTEST cap.
  _qos_candidates() {
    [[ -n "${PREFLIGHT_QOS:-}" ]] && { echo "$PREFLIGHT_QOS"; return; }
    scontrol show partition "${PREFLIGHT_PARTITION:-cpu}" 2>/dev/null \
      | grep -o 'QoS=[^ ]*' | cut -d= -f2
    squeue -u "$USER" -h -o %q 2>/dev/null | sort -u
    sacctmgr -n show assoc user="$USER" format=QOS%60 2>/dev/null | tr ',' '\n' | tr -d ' '
  }
  cap_mb=""
  qos=""
  while read -r q; do
    [[ -z "$q" || "$q" == "N/A" || "$q" == "(null)" ]] && continue
    c=$(sacctmgr -n show qos "$q" format=MaxTRESPU%60 2>/dev/null \
        | tr ',' '\n' | grep -o 'mem=[0-9]*' | head -1 | cut -d= -f2)
    [[ -z "$c" ]] && continue
    if [[ -z "$cap_mb" ]] || [[ "$c" -lt "$cap_mb" ]]; then cap_mb="$c"; qos="$q"; fi
  done < <(_qos_candidates | sort -u)

  if [[ -z "$cap_mb" ]]; then
    say_fail "per-user memory allowance: could not read the QOS cap - NOT CHECKED"
    echo "        -> set PREFLIGHT_QOS=<qos> (or --warn-only if you accept the risk)"
    echo "        -> an unreadable cap is not a passing cap; #3129 jammed on this three times"
  else
    pct=$(awk "BEGIN{printf \"%.0f\", 100*$req_mb/$cap_mb}")
    human=$(awk "BEGIN{printf \"%.0f\", $req_mb/1024}")
    caph=$(awk "BEGIN{printf \"%.0f\", $cap_mb/1024}")
    if [[ "$pct" -ge 90 ]]; then
      say_fail "this array claims ${human}G of your ${caph}G allowance under QOS ${qos} (${pct}%)"
      echo "        -> your OWN later jobs will queue in QOSMaxMemoryPerUser behind it"
      echo "        -> size --mem from a real cell (sacct MaxRSS), or lower --conc"
    elif [[ "$pct" -ge 70 ]]; then
      say_ok "per-user memory: ${human}G of ${caph}G (${pct}%) - tight, leaves little for other jobs"
    else
      say_ok "per-user memory: ${human}G of ${caph}G (${pct}%) under QOS ${qos}"
    fi
  fi
fi

# --- 9. A job name you are already using ------------------------------------
# Two arrays submitted as the same --job-name silently break every per-name
# query, including the completion waiter this repo's own skill recommends
# (`squeue -u $USER -h -n JOBNAME`): its counts then span both arrays.
if [[ -n "$JOB_NAME" ]]; then
  n_same=$(squeue -u "$USER" -h -n "$JOB_NAME" -o %i 2>/dev/null | wc -l)
  if [[ "$n_same" -gt 0 ]]; then
    say_fail "job name '$JOB_NAME' already has $n_same task(s) queued/running"
    echo "        -> per-name monitoring will conflate the two runs; pick a distinct name"
  else
    say_ok "job name '$JOB_NAME' is not already in the queue"
  fi
fi

# --- 10. The embeddings the jobs will actually read ---------------------------
# Launchers hardcode a default VTSEARCH_DATA_DIR pointing at whichever study dir
# happened to hold the pickles when they were written (`/exp/$USER/max-patch/
# datadir` in most of them).  Those dirs get archived and deleted; the launcher
# does not notice, because nothing reads the data dir until a cell runs, and a
# cell that cannot find its pickle fails one cell at a time inside a 552-cell
# array - which reads as "a few flaky cells", not "the whole run is pointed at
# nothing".  The pile (scripts/experiments/pile/pile_env.sh) is the durable home;
# this check is what says whether the variable actually points at it.
if [[ -n "${VTSEARCH_DATA_DIR:-}" ]]; then
  if [[ ! -d "$VTSEARCH_DATA_DIR" ]]; then
    say_fail "VTSEARCH_DATA_DIR=$VTSEARCH_DATA_DIR does not exist"
    echo "        -> source scripts/experiments/pile/pile_env.sh, or point it at a real datadir"
  elif [[ ! -d "$VTSEARCH_DATA_DIR/embeddings" ]]; then
    say_fail "VTSEARCH_DATA_DIR=$VTSEARCH_DATA_DIR has no embeddings/ subdirectory"
  else
    npkl=$(find "$VTSEARCH_DATA_DIR/embeddings" -maxdepth 1 -name '*.pkl' 2>/dev/null | wc -l)
    if [[ "$npkl" -eq 0 ]]; then
      say_fail "no .pkl files in $VTSEARCH_DATA_DIR/embeddings - the cells have nothing to read"
    else
      say_ok "VTSEARCH_DATA_DIR holds $npkl embedding pickles"
    fi
  fi
else
  say_ok "VTSEARCH_DATA_DIR unset (jobs will use the repo default)"
fi

# --- 11. A reused prepare whose symlinks still resolve -------------------------
# Reusing a finished study's prepare output is the standard way to skip a GPU
# stage.  But those `crops/` entries are *symlinks* into the study that generated
# them, and when that study is archived the links dangle.  The copy step in the
# launchers resolves them with `readlink -f`, which happily returns a path that
# no longer exists - so the reuse "succeeds", the link is recreated dangling, and
# the failure surfaces much later as missing exemplars.  `-e` follows the link,
# so this is the one-line check that `readlink -f` is not.
if [[ -n "$REUSE_PREPARE" ]]; then
  if [[ ! -f "$REUSE_PREPARE/prepare_info.json" ]]; then
    say_fail "--reuse-prepare $REUSE_PREPARE has no prepare_info.json"
  else
    dangling=0
    for f in "$REUSE_PREPARE"/crops/*; do
      [[ -e "$f" ]] || { dangling=$((dangling + 1)); echo "        dangling: $f -> $(readlink "$f" 2>/dev/null)"; }
    done
    if [[ "$dangling" -gt 0 ]]; then
      say_fail "$dangling crop symlink(s) in $REUSE_PREPARE/crops do not resolve"
      echo "        -> the source study was probably archived; repoint them at the archive's real files"
    else
      say_ok "reused prepare at $REUSE_PREPARE resolves (prepare_info.json + crops)"
    fi
  fi
fi

# --- 12. The knobs this run pins, against what the app actually ships ---------
# The eval framework only measures something if its *unswept* knobs are the
# app's.  They drift the other way round from how it feels: the app moves, and a
# launcher written weeks earlier keeps pinning what used to be production.
# `launch_incl_2865.sh` pinned `CALIB_HEAD=linear` - correct on 2026-08-12, and
# by the time it ran PR #3198 had made the linear SVM the shipped head, so the
# pin would have measured a cut rule on a detector nobody has.  Nothing would
# have broken: the numbers would have been clean, plausible, and about the wrong
# thing.
#
# So every knob with a *named* shipped constant is compared against it, and a
# divergence must be **declared** to pass: `--diverges head,anchor_weight`.  That
# is the whole design - a study is always allowed to pin the axis it sweeps, and
# is never allowed to pin one silently.
if [[ -n "$REPO" && "$PY_USABLE" == "0" ]]; then
  say_fail "pinned knobs NOT compared against production: python cannot import the tree (see above)"
elif [[ -n "$REPO" && -f "$REPO/scripts/experiments/calibration/experiment_config.py" ]]; then
  DIVERGENCE=$(CALIB_EXP="$EXP" python - "$REPO" <<'PYDIV' 2>&1
import os
import pathlib
import sys

repo = sys.argv[1]
sys.path.insert(0, str(pathlib.Path(repo) / "scripts" / "experiments" / "calibration"))
import common  # noqa: E402

common.setup_env()
# Imported with the RUN'S OWN ENVIRONMENT, so every knob below reads the value
# this run will actually use - the env var if it set one, else the harness
# default.  Re-deriving the defaults here as literals is what let #3400's three
# stale ones sit unnoticed: the check compared a launcher's pin against the app
# and never noticed that *not pinning* resolved to a study-era value.  An unset
# knob is only "the shipped arm" if the harness resolves it there, so that is
# what gets compared.
import experiment_config as C  # noqa: E402
from vtscore.eval.voting_iterations import PRODUCTION_HEAD, PRODUCTION_PATCH_STYLE  # noqa: E402
from vtscore.training import thresholds as T  # noqa: E402


def env(name):
    v = os.environ.get(name)
    return v.strip() if v and v.strip() else None


rows = []


def pinned(knob, var, shipped):
    """A scalar knob: unset means the harness resolves it to the shipped value."""
    v = env(var)
    if v is not None and v != str(shipped):
        rows.append((knob, v, str(shipped)))


def must_contain(knob, var, shipped, effective):
    """A set-valued knob: the shipped value has to be IN what the run resolves
    it to, or the run has no arm to compare its challengers against.

    *effective* is the resolved list off ``experiment_config``, so this catches a
    stale harness default exactly as it catches a stale launcher pin - and says
    which of the two it is, because the remedy differs (drop the pin vs. fix the
    default).
    """
    got = [str(x).strip() for x in effective]
    if str(shipped) in got:
        return
    source = "pinned in %s" % var if env(var) else "harness default; %s is unset" % var
    rows.append((knob, "%s (%s)" % (",".join(got), source), "a set containing " + str(shipped)))


pinned("head", "CALIB_HEAD", PRODUCTION_HEAD)
pinned("acq_offset", "CALIB_ACQ_INCLUSION_OFFSET", T.ACQUISITION_INCLUSION_OFFSET)
pinned("calibrate_count", "CALIB_CALIBRATE_COUNT", 2)
# The Train/Calibrate split of each calibration fold (#3287/#3290).  The
# shipped default is no longer one scalar: unset resolves per embedder through
# `production_split_for` (PRODUCTION_SPLIT_BY_SPACE), exactly as the app does,
# so an unset env var IS the production arm.  A pinned scalar can match at
# most one space on a run that mixes them, so - like CALIB_BLEND_SCHEDULE - an
# explicit pin is always a divergence the study must declare.
v = env("CALIB_CALIBRATION_FRACTION")
if v is not None:
    per_space = ", ".join("%s=%g" % (k, f) for k, f in sorted(T.PRODUCTION_SPLIT_BY_SPACE.items()))
    rows.append(("calibration_fraction", v, "<unset> = the app's per-space default (%s)" % per_space))

# The app has no safe-thresholds switch any more (#2799): fusion is always on.
# Read off the resolved config rather than the env var, because until #3400 the
# harness default was 0: an unset var passed this check while the run measured
# the unfused control - the one arm the app can no longer produce.
if not C.SAFE_THRESHOLDS:
    rows.append(("safe_thresholds", env("CALIB_SAFE_THRESHOLDS") or "<unset> = 0", "1 (the app has no switch)"))

# An explicit schedule overrides the app's per-mode default (#2841).
v = env("CALIB_BLEND_SCHEDULE")
if v is not None:
    rows.append(("blend_schedule", v, "<unset> = the app's per-mode default"))

# The #3314 adaptive fold count.  The app has no such thing: `calibrate_count`
# is a constant there, so ANY schedule is a divergence and has to be declared -
# including one whose early phase happens to equal today's constant, since the
# knob's whole effect is that the count stops being one.  Checked separately
# from `calibrate_count` above because the two can be set together and mean
# different arms (the schedule's tail IS `calibrate_count`).
v = env("CALIB_FOLD_COUNT_SCHEDULE")
if v is not None:
    rows.append(("fold_count_schedule", v, "<unset> = a constant calibrate_count, as the app has"))

# The #3308 voted-media exclusion floor, which #3312 sweeps as an arm axis.
# Unset resolves through the app's own `resolve_exclusion_floor`, so an unset
# env var IS the production arm.  Every other value is a divergence - INCLUDING
# a numeric pin that happens to equal today's shipped floor, because pinning it
# freezes the arm against a constant that can move underneath the study.
v = env("CALIB_EXCLUDE_VOTED")
if v is not None and v.strip().lower() not in ("", "default", "app"):
    rows.append(
        (
            "exclusion_floor",
            v,
            "<unset> = the app's own floor (currently %g)" % T.resolve_exclusion_floor(None),
        )
    )

# The anchored/fold-anchored grid (#2852) is emitted only under CALIB_ANCHORED=1
# and is off by default.  Checking its knobs unconditionally makes every study
# that does not use the family declare a divergence it does not have - and a
# declared-but-fictional divergence is worse than no check, because the next
# reader cannot tell the real ones from the noise.  Check them when the family is
# actually on; say plainly that they were skipped when it is not.
if os.environ.get("CALIB_ANCHORED") == "1":
    must_contain("cut_rule", "CALIB_ANCHORED_RULES", T.FOLD_ANCHOR_CUT_RULE, C.ANCHORED_RULES)
    must_contain("fold_combine", "CALIB_ANCHORED_FOLD_COMBINES", T.FOLD_ANCHOR_COMBINE, C.ANCHORED_FOLD_COMBINES)
    must_contain(
        "anchor_weight", "CALIB_ANCHORED_WEIGHTS", "%g" % T.FOLD_ANCHOR_WEIGHT, ["%g" % w for w in C.ANCHORED_WEIGHTS]
    )
else:
    print("SKIPPED\tanchored grid (CALIB_ANCHORED is not 1, so no anchored row is emitted)")
must_contain("patch_style", "CALIB_PATCH_STYLES", PRODUCTION_PATCH_STYLE, C.PATCH_STYLES)

if not rows:
    print("MATCHES")
else:
    for knob, got, want in rows:
        print("DIVERGES\t%s\t%s\t%s" % (knob, got, want))
PYDIV
)
  # Tag-dispatched rather than prefix-matched on the whole blob: the probe emits
  # SKIPPED lines for knob families this run does not enable, and those have to
  # be *reported* (a skipped check is not a passed one) without being mistaken
  # for a divergence or for a broken probe.
  unacked=0
  understood=0
  while IFS=$'\t' read -r tag knob got want; do
    [[ -z "$tag" ]] && continue
    case "$tag" in
      MATCHES)
        say_ok "every pinned knob matches the shipped value"
        understood=1 ;;
      SKIPPED)
        say_ok "knob check skipped: $knob"
        understood=1 ;;
      DIVERGES)
        understood=1
        if [[ ",${DIVERGES}," == *",${knob},"* ]]; then
          say_ok "declared divergence on '$knob' ($got, shipped is $want)"
        else
          say_fail "UNDECLARED divergence from production: $knob = $got, shipped is $want"
          unacked=$((unacked + 1))
        fi ;;
      *)
        say_fail "could not compare this run's knobs against production: $tag $knob $got $want"
        understood=1 ;;
    esac
  done <<< "$DIVERGENCE"
  if [[ "$understood" -eq 0 ]]; then
    say_fail "could not compare this run's knobs against production: $DIVERGENCE"
  fi
  if [[ "$unacked" -gt 0 ]]; then
    echo "        -> if that is the axis this study sweeps, pass --diverges <knob>[,<knob>]"
    echo "        -> if it is not, the run would measure a detector nobody ships"
  fi
fi

# --- 13b. A contrast axis that is confounded with another axis -----------------
# #3115 swept the fold COMBINE rule and reported its headline per "voting mode".
# Its grid held exactly two cells - `siglip x whole_image` and
# `dinov3_patch x max_patch` - so every binary cell was SigLIP and every region
# cell DINOv3.  The sign flip it measured is real; its ATTRIBUTION to voting mode
# is not, because the embedder moved with it.  Check 6 asserts that region voting
# genuinely *happens* on a cell; nothing asserted that a per-mode contrast is
# attributable to the mode.
#
# So: whenever a run will be read per voting mode, both modes need more than one
# embedder between them, or the two axes cannot be told apart.  Opt-in, because
# plenty of studies legitimately report a single cell per mode and never contrast
# across them - the failure is claiming the contrast, not running the grid.
if [[ -n "$MODE_CONTRAST" && -n "$REPO" ]]; then
  if [[ "$PY_USABLE" == "0" ]]; then
    say_fail "mode-contrast confound NOT checked: python cannot import the tree (see above)"
  else
    CONF=$(CALIB_EXP="$EXP" python - "$REPO" <<'PY' 2>&1
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(sys.argv[1]) / "scripts" / "experiments" / "calibration"))
import experiment_config as cfg

by_mode: dict[str, set[str]] = {"binary": set(), "region": set()}
for ds in cfg.DATASETS:
    for emb in cfg.embedders_for_dataset(ds):
        for style in cfg.styles_for(ds, emb):
            mode = "region" if (cfg.region_voting_for(ds, emb) and style != "whole_image") else "binary"
            by_mode[mode].add(emb)
shared = by_mode["binary"] & by_mode["region"]
if not by_mode["binary"] or not by_mode["region"]:
    print("SKIP only one voting mode in this grid; there is no cross-mode contrast to confound")
elif shared:
    print("HOLDS embedder(s) in BOTH modes: " + ", ".join(sorted(shared)))
else:
    print(
        "FAILS binary={%s} region={%s} - disjoint"
        % (",".join(sorted(by_mode["binary"])), ",".join(sorted(by_mode["region"])))
    )
PY
)
    case "$CONF" in
      HOLDS*) say_ok "mode contrast is not embedder-confounded (${CONF#HOLDS })" ;;
      SKIP*)  say_ok "mode-contrast check skipped (${CONF#SKIP })" ;;
      FAILS*)
        say_fail "voting mode is CONFOUNDED with the embedder: ${CONF#FAILS }"
        echo "        -> a per-mode headline from this grid is equally a per-embedder one"
        echo "        -> give one embedder both modes (a patch embedder can run whole_image"
        echo "           too: add it to CALIB_PATCH_STYLES), or do not contrast across modes"
        ;;
      *) say_fail "could not check the mode-contrast confound: $CONF" ;;
    esac
  fi
fi

# --- 13. Categories thin enough to produce no trainable step ------------------
# A cell whose category is too rare never collects both classes, so the
# simulation writes its CSV **header and nothing else**.  That file is
# non-empty, parses cleanly, and passes `find -size 0`, so every "N/N cells"
# count reports it as present - the failure is invisible at exactly the moment
# it matters.  #3115 launched a 208-cell array on `visual_genome_m` and its
# first two completed cells were header-only (`ball`, 51 positives in 4193).
#
# Reads `prepare_info.json`, which already holds `category_counts` and
# `selected_categories`, so this costs no pickle load.  Opt-in with a floor the
# study picks: what counts as "too thin" depends on the horizon and on
# SIM_FRACTION, and a wrong default here would be worse than none.
if [[ -n "$MIN_POSITIVES" ]]; then
  INFO="${CALIB_RESULTS:-$EXP/results}/prepare_info.json"
  if [[ ! -f "$INFO" ]]; then
    say_fail "--require-min-positives: no prepare_info.json at $INFO (run prepare first)"
  elif [[ "$PY_USABLE" == "0" ]]; then
    say_fail "category depth NOT checked: python cannot import the tree (see above)"
  else
    THIN=$(python - "$INFO" "$MIN_POSITIVES" <<'PY' 2>&1
import json
import sys

info, floor = json.load(open(sys.argv[1])), int(sys.argv[2])
thin, seen = [], 0
for ds, embs in info.get("datasets", {}).items():
    for emb, d in embs.items():
        counts = d.get("category_counts") or {}
        for cat in d.get("selected_categories") or []:
            seen += 1
            n = int(counts.get(cat, 0))
            if n < floor:
                thin.append(f"{ds}x{emb}:{cat}={n}")
print(("FAILS " + "; ".join(sorted(thin))) if thin else f"HOLDS {seen} selected cells, all >= {floor} positives")
PY
)
    case "$THIN" in
      HOLDS*) say_ok "category depth: ${THIN#HOLDS }" ;;
      FAILS*)
        say_fail "categories below the $MIN_POSITIVES-positive floor: ${THIN#FAILS }"
        echo "        -> a category this thin can finish with NO trainable step and write a"
        echo "           header-only CSV, which every 'N/N cells' count reports as present"
        ;;
      *) say_fail "could not check category depth: $THIN" ;;
    esac
  fi
fi

# --- 14. Every cell takes the opening this study DECLARED ---------------------
# The autopilot has two documented starts, and which one a cell takes is decided
# silently by whether a query text happens to exist for its (dataset, category)
# and whether its embedder has a text tower.  With a query the seed sort is
# cosine to the typed text; without one the app falls back to three random
# known-good examples.  Both are real user flows, so neither errors and every
# downstream column is populated either way - the same shape as
# `lessons/2026-08-26-the-harness-seeded-from-a-crop.md`, where a parameter was
# fed something other than what its name says it holds.
#
# That is merely untidy for most studies and fatal for one whose arms are
# POSITIONS ON THE SEED SORT (#3267): a cut at the 2nd rank percentile of a text
# sort and the same cut on a known-good sort are cuts on different objects, so a
# grid split across the two is not one experiment.  The lesson above closed this
# for `vg_scale` and left it open for `coco_val` and `vg_box_*` in as many words
# ("Still only advice ... config-only to fix"). This is the control.
#
# Both directions are checkable, because both are real choices (#3278).  A study
# that pins the known-good start - because that flow IS its subject, or because
# it re-runs a completed grid that took it - declares
# `CALIB_REQUIRE_OPENING=known_good`, and this check then fails on a cell that
# *would* text-seed.  A pin nobody can check is a comment, and the sixteen
# launchers #3278 went through are what a comment was worth.
#
# Reads `prepare_info.json` for the categories actually selected, so it sees the
# grid that will run rather than the one the launcher asked for.  `run_cells.py`
# asserts the same declaration per cell, which is what covers a study whose
# array is submitted from a job where no preflight runs.

# The prepare output this check and check 15 read: this run's own, or - when the study
# has not copied it in yet - the one it declared it will reuse.  A launcher that
# preflights BEFORE staging its reused prepare (launch_tail_2881.sh,
# launch_transfer_2883.sh) would otherwise be checking a file that does not exist
# yet, which is the moment its checks are most worth running.
resolve_info() {
  local own="${CALIB_RESULTS:-$EXP/results}/prepare_info.json"
  if [[ -f "$own" ]]; then echo "$own"
  elif [[ -n "$REUSE_PREPARE" && -f "$REUSE_PREPARE/prepare_info.json" ]]; then
    echo "$REUSE_PREPARE/prepare_info.json"
  fi
}

if [[ -n "$WANT_OPENING" ]]; then
  INFO="$(resolve_info)"
  if [[ -z "$INFO" ]]; then
    # An explicit --require-text-seed is a caller asking for the check NOW, so a
    # missing prepare is a failure.  A study-wide `CALIB_REQUIRE_OPENING` is a
    # declaration that `run_cells.py` also asserts per cell, so a launcher that
    # preflights before prepare has run has not lost the guarantee - it has only
    # moved it later.  Saying which of the two happened matters: a skipped check
    # is not a passed one.
    if [[ -n "$REQUIRE_TEXT_SEED" ]]; then
      say_fail "--require-text-seed: no prepare_info.json under ${CALIB_RESULTS:-$EXP/results} (run prepare first)"
    else
      say_note "opening check skipped: no prepare_info.json yet; run_cells.py asserts CALIB_REQUIRE_OPENING=$WANT_OPENING per cell"
    fi
  elif [[ "$PY_USABLE" == "0" ]]; then
    say_fail "seed mode NOT checked: python cannot import the tree (see above)"
  else
    SEEDCHK=$(cd "$REPO/scripts/experiments/calibration" && python - "$INFO" "$WANT_OPENING" <<'PY' 2>&1
import json
import sys

sys.path.insert(0, ".")
import experiment_config as cfg  # noqa: E402

from vtscore.media import get_embedder  # noqa: E402

info, want = json.load(open(sys.argv[1])), sys.argv[2]
bad, seen, held = [], 0, 0
for ds, embs in info.get("datasets", {}).items():
    for emb, d in embs.items():
        # The other half of the seed mode: an embedder with no text tower can
        # never produce a text sort however good the query is (DINOv3).  For a
        # PAIRED arm the tower that matters is the text half's, which is the
        # whole point of pairing - probing the arm name itself would ask the
        # registry for an embedder called "siglip+dinov3_patch" and report a
        # working pair as a broken one.
        text_emb = cfg.text_embedder(emb)
        try:
            has_tower = get_embedder(text_emb).embed_text("probe") is not None
        except Exception as exc:  # noqa: BLE001
            bad.append(f"{ds}x{emb}: embedder {text_emb} failed to load ({type(exc).__name__})")
            continue
        # And a pair can only rank the run's own medias if the text half's
        # pickle holds them.  prepare_data checks this too; checking it again
        # here is what stops an array of thousands of cells from being submitted
        # against a pickle that cannot serve them.
        if cfg.is_paired(emb) and d.get("text_pickle") is None:
            bad.append(f"{ds}x{emb}: prepare recorded no text_pickle (rerun prepare on this branch)")
        for cat in d.get("selected_categories") or []:
            seen += 1
            text = cfg.seed_query_text(ds, cat)
            got = "text" if (text and has_tower) else "known_good"
            if got == want:
                held += 1
            elif want == "text":
                bad.append(f"{ds}x{emb}:{cat}=" + ("no query" if not text else f"no text tower on {text_emb}"))
            else:
                bad.append(f"{ds}x{emb}:{cat}=would text-seed on {text_emb} ({text!r})")
print(("FAILS " + "; ".join(sorted(bad)[:12])) if bad else f"HOLDS {held}/{seen} selected cells open on {want}")
PY
    )
    # Keep the LAST line only.  Loading a SigLIP text tower prints transformers'
    # bos/eos token-id warnings to stderr, and 2>&1 folds them into the verdict,
    # where they match no `case` branch - so a check that HELD reported "could
    # not check".  A gate whose own plumbing can turn a pass into a fail teaches
    # people to pass --warn-only, which is worse than not having the gate.
    SEEDCHK=$(printf '%s\n' "$SEEDCHK" | tail -1)
    case "$SEEDCHK" in
      HOLDS*) say_ok "seed mode: ${SEEDCHK#HOLDS }" ;;
      FAILS*)
        if [[ "$WANT_OPENING" == "text" ]]; then
          say_fail "cells that would NOT seed from a text sort: ${SEEDCHK#FAILS }"
          echo "        -> these take the known-good start instead, so their seed sort is a"
          echo "           different ranking; add the query to EXPERIMENT_QUERIES, pair the arm"
          echo "           with a text tower (siglip+dinov3_patch), or set"
          echo "           CALIB_REQUIRE_SEED_QUERY=1 so prepare never selects them"
        else
          say_fail "cells that would NOT take the known-good start: ${SEEDCHK#FAILS }"
          echo "        -> this study PINS the known-good opening, and these cells can open on"
          echo "           a typed query instead; drop the pin if the text sort is now wanted"
        fi
        ;;
      *) say_fail "could not check seed mode: $SEEDCHK" ;;
    esac
  fi
fi

# --- 15. Every arm in the grid has a prepare entry ----------------------------
# `array_cells` enumerates the grid as `DATASETS x embedders_for_dataset(ds) x
# the categories PREPARE selected for that (dataset, embedder)`.  So an arm with
# no prepare entry contributes **zero cells**, silently: the array is that many
# indices shorter, every index past the gap maps to a different cell than the
# launcher's log says, and the run still comes back complete.
#
# The way to land there is ordinary: reuse a finished study's prepare_info.json
# (the standard way to skip a GPU stage) after renaming an arm.  `dinov3_patch`
# -> `siglip+dinov3_patch` is exactly that rename - #3278 made it in fourteen
# launchers, most of which copy a `$REUSE/prepare_info.json` keyed by the old
# bare name.  Prepare has to run again for the new key; it reuses the cached
# pickle, so it costs no encoder time, and this check is what says so instead of
# an array that quietly drops its region arm.  A skipped or failed embedder
# (`prepare_info["failed"]`, e.g. DINOv3 without HF_TOKEN) lands here too.
INFO15="$(resolve_info)"
if [[ -z "$INFO15" ]]; then
  say_note "grid-vs-prepare check skipped: no prepare_info.json under ${CALIB_RESULTS:-$EXP/results} yet"
elif [[ -z "$REPO" ]]; then
  say_note "grid-vs-prepare check skipped: VTS_REPO unset"
elif [[ "$PY_USABLE" == "0" ]]; then
  say_fail "grid vs prepare NOT checked: python cannot import the tree (see above)"
else
  GRIDCHK=$(cd "$REPO/scripts/experiments/calibration" && python - "$INFO15" <<'PY' 2>&1
import json
import sys

sys.path.insert(0, ".")
import experiment_config as cfg  # noqa: E402

info = json.load(open(sys.argv[1]))
prepared = info.get("datasets", {})
failed = set(info.get("failed") or [])
missing, seen = [], 0
for ds in cfg.DATASETS:
    for emb in cfg.embedders_for_dataset(ds):
        seen += 1
        if emb in prepared.get(ds, {}):
            continue
        why = "prepare FAILED it" if f"{ds}:{emb}" in failed else "prepare has no entry"
        have = ", ".join(sorted(prepared.get(ds, {}))) or "nothing"
        missing.append(f"{ds}x{emb} ({why}; prepared: {have})")
print(("FAILS " + "; ".join(missing[:8])) if missing else f"HOLDS all {seen} grid arms are in prepare_info")
PY
  )
  GRIDCHK=$(printf '%s\n' "$GRIDCHK" | tail -1)
  case "$GRIDCHK" in
    HOLDS*) say_ok "grid vs prepare: ${GRIDCHK#HOLDS }" ;;
    FAILS*)
      say_fail "arms in the grid that prepare never wrote: ${GRIDCHK#FAILS }"
      echo "        -> each contributes ZERO cells, so the array is short and every index"
      echo "           after the gap means a different cell than the launcher printed"
      echo "        -> re-run prepare_data.py for this grid (a cached pickle is reused,"
      echo "           so a renamed arm costs no encoder time), or fix the arm name"
      ;;
    *) say_fail "could not check the grid against prepare: $GRIDCHK" ;;
  esac
fi

# --- 16b. A horizon that outruns its own POSITIVES ----------------------------
# Check 16 asks whether the horizon empties the sim set of MEDIA.  That is the
# wrong ceiling for any study whose endpoint depends on finding positives: a
# 400-step run on a 3873-media sim half passes 16 comfortably and can still
# harvest 85% of the ~150 POSITIVES in it, because an aggressive acquisition arm
# does not sample uniformly - that is the whole point of it.
#
# What that does is not a truncation, it is a CONFOUND, and a one-sided one: the
# aggressive arms hit the ceiling and the control never comes near it, so the
# arms are no longer being compared over the same opportunity.  It is also
# invisible in every summary statistic the analyzer prints.  #3319 measured
# `acq_m3`/`acq_m4` at 82%/85% median harvest at 400 clicks while `prod` sat at
# 15%, after a single pilot cell (57 of ~150, rate still rising) was read as
# clearing the hazard.  One cell is not a sample.
#
# The bound is deliberately crude and conservative: if `max_steps` reaches the
# positives available in the sim half, exhaustion is REACHABLE and the study has
# to have thought about it.  A note, not a failure - running to exhaustion can be
# exactly what a study wants, and it is the *unnoticed* case this exists to stop.
if [[ -n "${CALIB_MAX_STEPS:-}" ]]; then
  PINFO_P=""
  for cand in "${REUSE_PREPARE:-}/prepare_info.json" "$EXP/results/prepare_info.json"; do
    [[ -n "$cand" && -r "$cand" ]] && { PINFO_P="$cand"; break; }
  done
  if [[ -n "$PINFO_P" ]]; then
    POSHZ=$(PINFO="$PINFO_P" python - <<'PY' 2>&1
import json, os
steps = int(os.environ["CALIB_MAX_STEPS"])
frac = float(os.environ.get("CALIB_SIM_FRACTION") or 0.5)
info = json.loads(open(os.environ["PINFO"]).read())
worst = None
for ds, arms in sorted((info.get("datasets") or {}).items()):
    for emb, rec in sorted((arms or {}).items()):
        counts = (rec or {}).get("category_counts") or {}
        sel = (rec or {}).get("selected_categories") or list(counts)
        vals = [int(counts[c]) for c in sel if c in counts and counts[c]]
        if not vals:
            continue
        # The THINNEST selected category is what binds: it exhausts first.
        n_pos_sim = max(1, int(min(vals) * frac))
        if worst is None or n_pos_sim < worst[0]:
            worst = (n_pos_sim, "%s x %s" % (ds, emb), min(vals))
if worst is None:
    print("SKIP\tprepare_info records no category_counts")
elif steps >= worst[0]:
    print("REACH\t%s: thinnest category has ~%d positives in the sim half "
          "(%d total x sim_fraction %g), horizon is %d steps"
          % (worst[1], worst[0], worst[2], frac, steps))
else:
    print("CLEAR\t%s: ~%d sim positives against a %d-step horizon"
          % (worst[1], worst[0], steps))
PY
    )
    POSHZ=$(printf '%s\n' "$POSHZ" | tail -1)
    case "$POSHZ" in
      CLEAR*) say_ok "horizon cannot exhaust the positives - tightest ${POSHZ#CLEAR?}" ;;
      SKIP*)  say_ok "horizon vs positives not checked (${POSHZ#SKIP?})" ;;
      REACH*)
        say_note "the horizon can EXHAUST the positives: ${POSHZ#REACH?}"
        echo "        -> aggressive acquisition arms do not sample uniformly, so they reach"
        echo "           this ceiling while the control does not - the arms then stop being"
        echo "           compared over the same opportunity, one-sidedly"
        echo "        -> it compresses the aggressive arms' measured advantage, so a WIN"
        echo "           survives it; a null or a loss past this point is not interpretable"
        echo "        -> report the realised harvest per arm, and do not size the hazard"
        echo "           from one pilot cell (#3319 did, and was wrong by 25 points)"
        ;;
    esac
  fi
fi

# --- 16c. A DEEP arm that eats its pile before the horizon --------------------
# 16b asks whether the horizon can EXHAUST the positives.  That is the right
# question for a run trying to reach 100% and the wrong one for a grid whose
# contrasts are read off a difference-in-differences, because a tail does not
# have to be empty to be capped: a pre-registered COMPRESSION BAR (#3547 used a
# median harvest of 50%) is the point past which an arm's late gains are limited
# by the pool rather than by the knob under study.
#
# What sets harvest is AGGRESSION, and a grid's aggression is set by its deepest
# arm.  #3547 sized `vg_scale_deep` at 900 positives per class off a SUPPLY bound
# (what twelve classes could furnish) checked against a HORIZON bound (16b, which
# it cleared: 450 sim positives against 400 clicks).  Neither is an aggression
# bound.  Its two deepest arms then harvested 56% and 60%, over the bar, and two
# of its three deep contrasts were excluded as compressed -- leaving one.
# Compression is ONE-SIDED, so the excluded arms were not merely noisy: a capped
# tail biases a DiD toward "no move" or "shallower" and never toward "deeper",
# and all three of that study's "shallower" readings sat on them.
#
# So: SIZE A DEEP GRID FROM ITS DEEPEST ARM, NOT FROM ITS SHIPPED ONE (#3611).
# Opt in with the bar the study pre-registered (as a flag, or once for the whole
# launch loop as `CALIB_HARVEST_BAR` / `CALIB_HARVEST_PILOT`); point the pilot at a short
# pilot wave of the deepest arm (a `cells/` dir, or a base holding
# `<arm>/results/cells`, in which case every arm is read and the worst one is
# what the verdict is taken on).  Without a pilot the only bound left is that an
# arm cannot find more positives than it takes clicks, which clears a deep enough
# pile outright and otherwise says so rather than guessing.
#
# A pilot can FAIL a grid without being able to CLEAR one: harvest is the most
# category-dependent quantity in this harness, so a pilot that stops short of the
# horizon, or that misses planned categories, reports UNKNOWN rather than ok
# (`lessons/2026-09-02-one-pilot-cell-cleared-a-hazard-the-full-wave-hit.md`).
if [[ -n "$HARVEST_BAR" ]]; then
  if [[ -z "${CALIB_MAX_STEPS:-}" ]]; then
    say_fail "--require-harvest-headroom needs CALIB_MAX_STEPS - the horizon it sizes against"
  else
    PINFO_H=""
    for cand in "${REUSE_PREPARE:-}/prepare_info.json" "$EXP/results/prepare_info.json"; do
      [[ -n "$cand" && -r "$cand" ]] && { PINFO_H="$cand"; break; }
    done
    HH_ARGS=(--bar "$HARVEST_BAR" --horizon "$CALIB_MAX_STEPS" --sim-fraction "${CALIB_SIM_FRACTION:-0.5}")
    [[ -n "$PINFO_H" ]] && HH_ARGS+=(--prepare-info "$PINFO_H")
    [[ -n "$PILOT_CELLS" ]] && HH_ARGS+=(--pilot "$PILOT_CELLS")
    HH=$(python "$HERE/calibration/harvest_headroom.py" "${HH_ARGS[@]}" 2>&1)
    HHV=$(printf '%s\n' "$HH" | tail -1)
    hh_detail() { printf '%s\n' "$HH" | sed '$d' | sed 's/^/        /'; }
    case "$HHV" in
      CLEAR*) say_ok "the deepest arm keeps its headroom: ${HHV#CLEAR?}" ;;
      UNKNOWN*)
        say_note "harvest headroom NOT established: ${HHV#UNKNOWN?}"
        hh_detail
        echo "        -> compression is one-sided: it biases a difference-in-differences toward"
        echo "           'no move' or 'shallower', so an arm over the bar cannot be read as either"
        ;;
      OVER*)
        say_fail "the deepest arm is over the compression bar: ${HHV#OVER?}"
        hh_detail
        echo "        -> that arm's deep readings would be reported as compressed and excluded,"
        echo "           which is the grid paying for cells it cannot read (#3547 lost two of"
        echo "           three deep contrasts this way)"
        echo "        -> deepen the pile to the positives-per-class above, shorten the horizon,"
        echo "           or drop the arm from the grid before paying for it"
        ;;
      *) say_fail "could not check harvest headroom: $HH" ;;
    esac
  fi
fi

# --- 16. A horizon that outruns its own haystack ------------------------------
# `sim_fraction` sets the simulation set: the pool the user votes out of AND the
# haystack the threshold's population estimate is fitted on.  `max_steps` is how
# many of those items get voted.  When the second approaches the first the run
# does not fail, it *truncates* - the loop stops when the pool empties - so a
# grid asking for 380 steps on a 200-media sim set silently becomes a ~200-step
# grid, and every arm's horizon becomes a property of the environment rather
# than of the design.  Worse for a study sweeping a *remainder* floor: the last
# steps are fitted on a handful of leftovers, which is the regime where the
# estimator is least trustworthy and most variable.
#
# Only checked when the study moves `sim_fraction` off its 0.5 default, since
# that is the knob that makes the collision possible (#3312).
if [[ -n "${CALIB_SIM_FRACTION:-}" && -n "${CALIB_MAX_STEPS:-}" ]]; then
  PINFO=""
  for cand in "${REUSE_PREPARE:-}/prepare_info.json" "$EXP/results/prepare_info.json"; do
    [[ -n "$cand" && -r "$cand" ]] && { PINFO="$cand"; break; }
  done
  if [[ -z "$PINFO" ]]; then
    say_ok "horizon vs haystack not checked (no prepare_info.json yet)"
  else
  HORIZON=$(PINFO="$PINFO" python - <<'PY' 2>&1
import json, os
frac = float(os.environ["CALIB_SIM_FRACTION"])
steps = int(os.environ["CALIB_MAX_STEPS"])
info = json.loads(open(os.environ["PINFO"]).read())
worst = []
for ds, arms in sorted((info.get("datasets") or {}).items()):
    for emb, rec in sorted((arms or {}).items()):
        n = int((rec or {}).get("n_medias") or 0)
        if not n:
            continue
        n_sim = max(1, int(n * frac))
        worst.append((n_sim - steps, "%s x %s" % (ds, emb), n_sim))
worst.sort()
if not worst:
    print("SKIP\tprepare_info records no n_medias")
elif worst[0][0] <= 0:
    print("FAILS\t%s: sim set is %d media, horizon is %d steps" % (worst[0][1], worst[0][2], steps))
else:
    print("HOLDS\t%s: %d media in the sim set, %d left after %d steps"
          % (worst[0][1], worst[0][2], worst[0][0], steps))
PY
  )
  HORIZON=$(printf '%s\n' "$HORIZON" | tail -1)
  case "$HORIZON" in
    HOLDS*) say_ok "horizon fits its haystack - tightest ${HORIZON#HOLDS?}" ;;
    SKIP*)  say_ok "horizon vs haystack not checked (${HORIZON#SKIP?})" ;;
    FAILS*)
      say_fail "the horizon exhausts the sim set: ${HORIZON#FAILS?}"
      echo "        -> the run does not fail, it TRUNCATES: the loop ends when the pool"
      echo "           empties, so max_steps stops being the design and starts being a"
      echo "           property of the dataset - and differently per dataset"
      echo "        -> raise CALIB_SIM_FRACTION or lower CALIB_MAX_STEPS"
      ;;
    *) say_fail "could not check the horizon against the haystack: $HORIZON" ;;
  esac
  fi
fi

echo
if [[ "$FAILED" == "1" ]]; then
  echo "PREFLIGHT FAILED - fix the above before launching (or --warn-only if deliberate)"
  exit 1
fi
echo "preflight OK"
