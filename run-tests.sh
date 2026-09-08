#!/usr/bin/env bash
# Run tests with minimal output and a clear PASS/FAIL summary.
#
# Usage:
#   ./run-tests.sh              # full gate: every check + all fast tests
#   ./run-tests.sh core         # core group + the cheap gates (see below)
#   ./run-tests.sh sorting      # sorting group + the cheap gates
#   ./run-tests.sh core sorting # core + sorting groups
#   ./run-tests.sh vtscore-clean  # run tests_lib/ with Flask import blocked
#   ./run-tests.sh slides       # slide-deck gates only (~8s, no Python tests)
#   ./run-tests.sh docs         # markdown-only gate (engages by itself; see below)
#
# Available groups: core, api, sorting, datasets, io, detectors,
#                   downloads, integration, cli, converters, projection,
#                   meta (repo/tooling tests: packaging, Dockerfiles, docs,
#                   scripts/, .claude/hooks/, the gates themselves, the test
#                   harness — nothing that ships as vtsearch/vtscore code),
#                   frontend (build + audit + Vitest, no Python tests),
#                   slides (the four gates a deck can trip, no Python
#                   tests — see the note below),
#                   docs (markdown-only: every cheap gate, plus the few tests
#                   that read a doc — see the note below), gpu
#
# Each group is a folder under tests/ AND tests_lib/. Marker assignment is
# automatic: any file at tests[_lib]/<group>/test_*.py gets marked <group>
# by the respective conftest.  tests_lib/ holds Flask-free library tests
# (see vtscore/docs/architecture.md).
#
# NOTE: naming a group puts `-m <group>` on the pytest command line, which
# REPLACES the `-m 'not gpu and not slow'` default in pyproject.toml's
# addopts rather than combining with it. So a group run also picks up any
# slow/gpu tests in that folder (that is how `./run-tests.sh gpu` works).
# To keep the default exclusions, spell the filter out after `--`, e.g.
#   ./run-tests.sh cli -- -m 'cli and not slow'
#
# Extra pytest args can follow a '--':
#   ./run-tests.sh core -- -x --tb=short
#
# ---------------------------------------------------------------------------
# How the run is staged
#
# There is no CI, so a *full* `./run-tests.sh` remains the only real gate and
# still runs every check. What changed is the shape of the run, because the
# stages are independent of each other and the box has more than one core:
#
#   1. Cheap gates, serial, fail-fast (~8s total). Linters, doc invariants,
#      snapshot drift. These are quick enough that stopping at the first
#      failure costs nothing and saves you reading past it.
#   2. The frontend production build, serial. It has to precede pytest because
#      a handful of tests serve the built bundle out of `static/`, and it is
#      also the frontend failure people hit most, so it is worth failing on
#      before spending minutes elsewhere.
#   3. The heavy, mutually independent stages *concurrently*: pyright,
#      pip-audit, the vulture whitelist check, the frontend unit suite, and
#      pytest. Sequentially these were
#      ~338s on a 4-vCPU box; overlapped they land at ~250s, within a few
#      percent of what the machine's cores can physically do.
#
# Stage 3 deliberately does NOT stop at the first failure: every lane runs to
# completion and all failures are reported together. A serial chain that dies
# on the first bad gate makes you pay the whole runtime again per fix.
#
# Group runs (`./run-tests.sh <group>`) run stage 1 and their own tests, and
# skip the stage-3 gates that are not about the code you just changed —
# pyright, pip-audit, the vulture whitelist check, and the frontend suite. That
# keeps the edit/test loop in
# the seconds it should be instead of paying ~105s of whole-repo checks to run
# a five-second group. The skip is announced on every group run, because the
# full run is what actually gates a push. Set VTSEARCH_FULL_GATES=1 to force
# the complete chain on a group run.
#
# `slides` goes further and is the one group that also gates a push, because a
# change confined to slides/ cannot reach anything else in the repo: nothing
# imports slides/build.py (it is a standalone script), pyrightconfig.json does
# not include it, and no test in either tree touches a deck. So the group runs
# only the four gates that can actually observe a deck — ruff (build.py),
# codespell (slide prose), check-docs.py (fragments are tracked markdown), and
# build.py --check — and skips pytest and every whole-repo gate. ~8s against
# ~3.5min. The exemption is self-policing: the group refuses to run when the
# branch touches anything outside slides/, so it cannot be taken by mistake.
#
# `docs` is the same idea one step wider, and it does not wait to be asked.
# A change confined to tracked markdown cannot be seen by pyright (which
# excludes markdown), by pip-audit, by the frontend build or its unit suite, or
# by ~1600 of the ~1603 Python tests. So a *bare* `./run-tests.sh` looks at what
# the branch actually changed and, when the answer is "markdown and nothing
# else", keeps the whole of stage 1 — the gates that genuinely read markdown
# live there — and narrows pytest to the tests that can observe a doc
# (tests_shared/markdown_surface.py names them, and a meta test keeps that list
# honest). ~35s against ~3.5min.
#
# It auto-engages because the thing being fixed is the *default* path: a
# plan-file session pays the full 3.5 minutes precisely by running the command
# everyone runs. A group you have to remember to type would not have caught it.
# The pruning is announced in full, and VTSEARCH_FULL_GATES=1 turns it off.
# Naming the group explicitly (`./run-tests.sh docs`) is the same gate with the
# slides-style guard: it refuses to run when the branch changes anything else.
# ---------------------------------------------------------------------------

set -euo pipefail
cd "$(dirname "$0")"
# Absolute, slash-containing path to this script. `$0` alone can be a bare
# filename (e.g. `run-tests.sh` when invoked as `bash run-tests.sh`), which
# makes `timeout`'s execvp do a PATH-only lookup below and fail with "No
# such file or directory". We've already cd'd into dirname("$0"), so pwd
# joined with basename("$0") is always absolute regardless of how we were
# invoked.
_self="$(pwd)/$(basename "$0")"

# Memory headroom check.
#
# The gates in this script are the memory-hungry part of the repo, not pytest:
# `pyright` runs under node with a ~1.8 GB default heap, and the Angular
# `build:prod` wants a comparable amount. On a small machine the run does not
# fail cleanly — node aborts with "Ineffective mark-compacts near heap limit",
# or the kernel's OOM killer takes whatever it likes, including the shell that
# launched the run. That is not hypothetical either: a 3 GB box died this way
# mid-`pyright`, and because the wrapper below reads 137 as its kill signal, the
# corpse was labelled TESTS TIMED OUT rather than "out of memory".
#
# So say it up front, where it is still actionable. This warns rather than
# blocks: the number below is the observed requirement, not a measured cliff,
# and a machine under it may still finish.
if [[ -z "${_VT_TIMEOUT_WRAPPED:-}" && -r /proc/meminfo ]]; then
    _mem_avail_gb=$(awk '/^MemAvailable:/ {printf "%.1f", $2/1048576}' /proc/meminfo)
    if [[ -n "$_mem_avail_gb" ]] && awk "BEGIN{exit !($_mem_avail_gb < 6)}"; then
        echo "============================================================"
        echo "LOW MEMORY: ${_mem_avail_gb}G available; a full run wants ~6G"
        echo ""
        echo "pyright (node, ~1.8G heap) and the Angular build are what need"
        echo "it. Below this they tend to die as an OOM kill rather than a"
        echo "test failure, and can take the session with them."
        echo ""
        echo "Run the suite on a bigger machine instead — on the GRID:"
        echo "  sbatch --partition=cpu --cpus-per-task=8 --mem=32G \\"
        echo "         --time=02:00:00 --wrap 'source gridenv.sh && ./run-tests.sh'"
        echo "Or run one group at a time here (./run-tests.sh detectors),"
        echo "which skips the frontend gates unless the group is core/frontend."
        echo "============================================================"
        echo ""
    fi
fi

# Wall-clock cap on the whole run.
#
# A healthy full run is a few minutes (see the staging note above), so 30
# minutes is many times the worst legitimate run and will not misfire — but it
# bounds the failure mode where the run wedges and nobody notices. That is not
# hypothetical: an xdist run once sat for 2h12m with three of its four workers
# `<defunct>` and the master idle, because *nothing* in this script had an
# upper bound. A per-test timeout does not cover that case (a dead worker can't
# fire its own timeout), which is why the cap lives out here, wrapping every
# stage — dep install, linters, npm, pytest alike.
#
# Implemented as a one-shot re-exec under `timeout`: the guard variable stops
# the child from re-wrapping itself. Set VTSEARCH_TEST_TIMEOUT=0 to opt out (for
# a deliberately long run, e.g. GPU tests or a full coverage sweep).
VTSEARCH_TEST_TIMEOUT=${VTSEARCH_TEST_TIMEOUT:-1800}

# `-n auto` does NOT mean "one worker per CPU I was given". pytest-xdist asks
# psutil for the MACHINE's physical cores and never looks at the cgroup, so a
# SLURM job with --cpus-per-task=8 on a 40-thread node starts ~20 workers, each
# importing torch. Measured (#3721): 90 oom_kill events in a 32G allocation, the
# workers dying one at a time so the run WEDGES rather than fails, and the only
# thing printed is the wall-clock cap above -- whose message then sends the
# reader hunting for a hung network step. Two runs lost 30 minutes each to it.
#
# The env var beats every detection path in xdist/plugin.py. On an unconstrained
# box the affinity mask is every core, so this is a no-op there; it only bites
# where the cgroup is narrower than the machine, which is exactly the GRID.
if [[ -z "${PYTEST_XDIST_AUTO_NUM_WORKERS:-}" ]]; then
    PYTEST_XDIST_AUTO_NUM_WORKERS=$(python -c 'import os; print(len(os.sched_getaffinity(0)))' 2>/dev/null || echo "")
    [[ -n "$PYTEST_XDIST_AUTO_NUM_WORKERS" ]] && export PYTEST_XDIST_AUTO_NUM_WORKERS
fi
if [[ -z "${_VT_TIMEOUT_WRAPPED:-}" && "$VTSEARCH_TEST_TIMEOUT" != "0" ]] \
    && command -v timeout >/dev/null 2>&1; then
    export _VT_TIMEOUT_WRAPPED=1
    # TERM first so pytest can print what it was doing; KILL 30s later for a
    # process too wedged to answer (exactly the defunct-worker case).
    set +e
    timeout --signal=TERM --kill-after=30 "$VTSEARCH_TEST_TIMEOUT" "$_self" "$@"
    _timeout_status=$?
    set -e
    if [[ $_timeout_status -eq 124 || $_timeout_status -eq 137 ]]; then
        echo ""
        echo "============================================================"
        echo "TESTS TIMED OUT after ${VTSEARCH_TEST_TIMEOUT}s (wall-clock cap)"
        echo ""
        echo "The run wedged rather than failed. Check for dead xdist workers"
        echo "(ps aux | grep defunct) or a hung network/install step. Re-run a"
        echo "single group to narrow it down, or set VTSEARCH_TEST_TIMEOUT=0"
        echo "if this run is legitimately meant to take longer."
        echo "============================================================"
    fi
    exit $_timeout_status
fi

# Install deps if needed
bash .claude/hooks/ensure-test-deps.sh

# vtscore-clean: run only the library-tier tests with Flask blocked.
# Skips the linter / frontend stages because the goal of this mode is
# specifically to verify that the library tier can run independent of
# Flask; we do not re-run the linting we already do in the main path.
if [[ "${1:-}" == "vtscore-clean" ]]; then
    shift
    exec python scripts/check-vtscore-clean.py "$@"
fi

# ---------------------------------------------------------------------------
# Argument parsing
#
# Done up front (it used to sit after the gate chain) because which gates run
# now depends on whether this is a full run or a group run.
# ---------------------------------------------------------------------------
TEST_GROUPS=()
EXTRA_ARGS=()
PAST_SEPARATOR=false

for arg in "$@"; do
    if [[ "$arg" == "--" ]]; then
        PAST_SEPARATOR=true
        continue
    fi
    if $PAST_SEPARATOR; then
        EXTRA_ARGS+=("$arg")
    else
        TEST_GROUPS+=("$arg")
    fi
done

_is_full_run=false
if [[ ${#TEST_GROUPS[@]} -eq 0 ]]; then
    _is_full_run=true
fi

# A group run can be promoted back to the complete chain on demand.
_run_whole_repo_gates=false
if $_is_full_run || [[ "${VTSEARCH_FULL_GATES:-}" == "1" ]]; then
    _run_whole_repo_gates=true
fi

_group_named() {
    local want="$1" g
    for g in ${TEST_GROUPS[@]+"${TEST_GROUPS[@]}"}; do
        [[ "$g" == "$want" ]] && return 0
    done
    return 1
}

# The frontend production build runs for the full suite and for the core /
# frontend groups. Catches compilation errors without needing a browser.
_run_frontend_check=false
if $_is_full_run || _group_named core || _group_named frontend; then
    _run_frontend_check=true
fi

# The frontend Vitest unit suite runs for the full run or an explicit
# `frontend` group, but NOT for `core`: it is heavier than the compile-only
# build check, so it stays off the fast `core` path.
_run_frontend_unit=false
if $_is_full_run || _group_named frontend; then
    _run_frontend_unit=true
fi

# `frontend` is a frontend-only gate; it has no Python tests. If it's the only
# requested group, skip pytest entirely so it doesn't error on an empty
# `-m frontend` selection.
_run_pytest=true
if [[ ${#TEST_GROUPS[@]} -eq 1 && "${TEST_GROUPS[0]}" == "frontend" ]]; then
    _run_pytest=false
fi

# `slides` is the same shape: no Python tests, so pytest is skipped rather than
# asked for an empty `-m slides` selection. Unlike every other group this one
# is also a legitimate pre-push gate; see the staging note at the top and the
# guard immediately below, which is what makes that safe.
_is_slides_run=false
if [[ ${#TEST_GROUPS[@]} -eq 1 && "${TEST_GROUPS[0]}" == "slides" ]]; then
    _is_slides_run=true
    _run_pytest=false
fi

_blocked() {
    echo ""
    echo "============================================================"
    echo "TESTS BLOCKED: $1"
    echo "============================================================"
}

# Everything this branch changes relative to dev — committed, staged, unstaged
# and untracked alike. Untracked files count because a stray new .py would be
# linted by a full run and is exactly the kind of thing that should not ride
# along unchecked on a narrowed one.
#
# Both fast paths below rest on this: `slides` verifies the diff is confined to
# slides/, and the markdown path both detects and verifies. Returns non-zero
# when there is no origin/dev to diff against, in which case neither path can
# establish its premise and both decline to narrow anything.
_branch_changed_paths() {
    local base
    base=$(git merge-base HEAD origin/dev 2>/dev/null || true)
    [[ -z "$base" ]] && return 1
    {
        git diff --name-only "$base" HEAD
        git diff --name-only HEAD
        git ls-files --others --exclude-standard
    } | sort -u
}

# Guard on the `slides` fast path.
#
# The group's whole justification is that a change confined to slides/ cannot
# affect anything the skipped gates check. That premise is about the *diff*,
# not about intent, so it is worth verifying rather than trusting: the failure
# mode this prevents is someone believing their change is slides-only, taking
# the 8s path, and pushing an unchecked edit to real code.
#
# "Confined to slides/" means everything this branch changes relative to dev —
# committed, staged, unstaged, and untracked alike. Untracked files count
# because a stray new .py would be linted by a full run and is exactly the kind
# of thing that should not ride along unchecked.
#
# Strict on purpose: a docs/ or CLAUDE.md edit alongside the deck also fails
# this, because the doc-inventory gate can see markdown that this group skips.
# Run the full suite for a mixed change; it is the honest cost of one.
if $_is_slides_run; then
    if ! _slides_changed=$(_branch_changed_paths); then
        echo "Note: no origin/dev to diff against; skipping the slides-only guard."
    else
        _outside=$(printf '%s\n' "$_slides_changed" | grep -v '^slides/' || true)
        if [[ -n "$_slides_changed" && -n "$_outside" ]]; then
            _blocked "'slides' is a slides-only gate, but this branch changes other files"
            echo ""
            echo "The group skips pytest and every whole-repo gate, which is only"
            echo "sound when nothing outside slides/ has changed. Outside slides/:"
            echo ""
            echo "$_outside" | sed 's/^/  /'
            echo ""
            echo "Run the full suite instead:  ./run-tests.sh"
            exit 1
        fi
    fi
fi

# ---------------------------------------------------------------------------
# The markdown-only fast path.
#
# A change confined to tracked markdown is invisible to most of this script:
# pyright excludes markdown, pip-audit reads the venv, the Angular build only
# *copies* docs/user/*.md into the bundle (frontend/docs-assets is a symlink to
# it) and cannot fail on its contents, and of ~1603 Python tests only the ones
# named in tests_shared/markdown_surface.py ever open a doc. What *can* see a
# doc is stage 1 — check-docs, codespell, the doc-inventory and screenshot
# wiring snapshots, the deck preflight — so a markdown-only run keeps stage 1
# whole and prunes the rest.
#
# Unlike `slides`, this engages on a bare `./run-tests.sh`. The complaint it
# answers is about the default: a session that writes one plan file pays the
# full 3.5 minutes by running the command everybody runs, and a group name you
# have to remember would not have been typed. So the diff decides, the pruning
# is announced in full, and VTSEARCH_FULL_GATES=1 opts out.
#
# Three things keep the narrowing honest rather than optimistic:
#   - it needs a diff to reason about. No origin/dev, or no changes at all, and
#     nothing is pruned.
#   - the pytest selection is single-sourced in tests_shared/markdown_surface.py,
#     and tests_lib/meta/test_markdown_surface.py fails when a test outside that
#     selection learns to read a repo doc — the one way this could rot.
#   - naming the group (`./run-tests.sh docs`) asserts the premise instead of
#     detecting it, so it blocks on a mixed branch rather than quietly running
#     the full suite.
# ---------------------------------------------------------------------------
_is_docs_run=false
_docs_named=false
if [[ ${#TEST_GROUPS[@]} -eq 1 && "${TEST_GROUPS[0]}" == "docs" ]]; then
    _docs_named=true
fi

if $_docs_named || { $_is_full_run && [[ "${VTSEARCH_FULL_GATES:-}" != "1" ]]; }; then
    if ! _md_changed=$(_branch_changed_paths); then
        if $_docs_named; then
            echo "Note: no origin/dev to diff against; skipping the markdown-only guard."
            _is_docs_run=true
        fi
    else
        _md_outside=""
        if [[ -n "$_md_changed" ]]; then
            _md_outside=$(printf '%s\n' "$_md_changed" | grep -v '\.md$' || true)
        fi
        if $_docs_named; then
            if [[ -n "$_md_outside" ]]; then
                _blocked "'docs' is a markdown-only gate, but this branch changes other files"
                echo ""
                echo "The group narrows pytest and skips every whole-repo gate, which is"
                echo "only sound when nothing but tracked markdown has changed. Not markdown:"
                echo ""
                echo "$_md_outside" | sed 's/^/  /'
                echo ""
                echo "Run the full suite instead:  ./run-tests.sh"
                exit 1
            fi
            _is_docs_run=true
        elif [[ -n "$_md_changed" && -z "$_md_outside" ]]; then
            _is_docs_run=true
            echo "============================================================"
            echo "MARKDOWN-ONLY CHANGE: narrowing this run"
            echo ""
            echo "$_md_changed" | sed 's/^/  /'
            echo ""
            echo "Nothing but tracked markdown has changed, so this run keeps every"
            echo "cheap gate (check-docs, codespell, the doc-inventory and screenshot"
            echo "snapshots, the deck preflight) and runs the tests that read a doc,"
            echo "and skips pyright, pip-audit, the vulture whitelist check, the"
            echo "frontend build/audit/unit suite, and the rest of pytest — none of"
            echo "which can observe a markdown file."
            echo ""
            echo "Set VTSEARCH_FULL_GATES=1 to run everything anyway."
            echo "============================================================"
            echo ""
        fi
    fi
fi

_docs_targets=()
if $_is_docs_run; then
    # Nothing outside stage 1 and the markdown surface can see a doc.
    _run_whole_repo_gates=false
    _run_frontend_check=false
    _run_frontend_unit=false
    _run_pytest=true
    # Single-sourced with the gate that proves the list is complete; see
    # tests_shared/markdown_surface.py.
    _md_surface=$(python -c 'from tests_shared.markdown_surface import MARKDOWN_TEST_SURFACE as m; print(" ".join(m))' 2>/dev/null || true)
    if [[ -z "$_md_surface" ]]; then
        _blocked "could not read MARKDOWN_TEST_SURFACE from tests_shared/markdown_surface.py"
        echo ""
        echo "The markdown-only path narrows pytest to that list, so an unreadable"
        echo "list would silently run no tests at all. Run './run-tests.sh' with"
        echo "VTSEARCH_FULL_GATES=1, and fix the import."
        exit 1
    fi
    read -r -a _docs_targets <<< "$_md_surface"
fi

# ---------------------------------------------------------------------------
# Stage 1: cheap gates. Serial and fail-fast — the whole set is ~8s, so
# stopping at the first failure costs nothing.
# ---------------------------------------------------------------------------

# Stale-tree ("phantom base") check.
#
# Runs before every other gate, because a clobbered tree passes all of them by
# construction: when a branch reverts the PRs that merged just before it, the
# feature and its tests go together, so there is nothing left to fail. Five
# PRs have done exactly that (#2741, #2793, #2821, #3184 and the three
# restored in #3206); one reached `main`. Linting the wrong tree cleanly is
# not information, so this asks *whether it is the right tree* first.
#
# Compares against the working tree rather than HEAD: the damage exists
# before it is committed, which is often when tests run. Pure git, ~200ms.
#
# Two signals, because deletions are only the visible half: whole paths that
# vanished, and runs of consecutive commits the branch keeps nothing of (a
# clobber whose reverted hunks sit inside files that survive deletes nothing
# at all). See scripts/check-phantom-base.py for both and their measured
# false-positive rates.
echo "Checking for a stale tree..."
if ! python scripts/check-phantom-base.py ; then
    _blocked "branch reverts work it never touched (stale tree)"
    exit 1
fi

# Ruff lint + format check.
# Fast (~0.5s), and catches mistakes the pytest / frontend stages can't see,
# e.g. F401 unused-import on TYPE_CHECKING imports whose only "use" is inside
# a string-form forward reference.
echo "Running ruff check..."
if ! ruff check . ; then
    _blocked "ruff check failed"
    exit 1
fi
echo "Running ruff format --check..."
if ! ruff format --check . ; then
    _blocked "ruff format --check failed (run 'ruff format .')"
    exit 1
fi

echo "Running codespell..."
if ! codespell --toml pyproject.toml ; then
    _blocked "codespell found typos"
    exit 1
fi

# Documentation drift: relative links, in-page anchors, backticked repo paths,
# leaked absolute machine paths, plan-file citations anywhere in the tree, and
# broken code fences. Pure invariants against the current tree — nothing to
# re-pin — and it imports nothing, so it costs ~0.4s and sits with the linters.
echo "Checking documentation..."
if ! python scripts/check-docs.py ; then
    _blocked "documentation check found drift"
    exit 1
fi

# Skipped on a `slides` run: none of these can see a deck. deptry reads
# imports, the OpenAPI and doc-inventory snapshots are generated from the
# app's registries, and the rest scan Dockerfiles / user-docs screenshots /
# vtscore. The guard above has already established that nothing outside
# slides/ changed, so there is nothing here for them to find.
if ! $_is_slides_run; then
    echo "Running deptry..."
    if ! python -m deptry . ; then
        _blocked "deptry found dependency issues"
        exit 1
    fi

    # OpenAPI snapshot drift check: regenerate the flask-smorest spec from
    # the live app and diff against the checked-in snapshot at
    # frontend/openapi.json. The frontend's generated TS client is built
    # from this snapshot, so a stale file means the generated client lags
    # the real API. Cheap (~2s) and runs every invocation.
    echo "Checking OpenAPI snapshot drift..."
    _openapi_regen=$(mktemp)
    _openapi_dump_log=$(mktemp)
    if ! python scripts/dump_openapi.py > "$_openapi_regen" 2> "$_openapi_dump_log"; then
        _blocked "OpenAPI spec dump failed"
        cat "$_openapi_dump_log"
        rm -f "$_openapi_regen" "$_openapi_dump_log"
        exit 1
    fi
    if ! diff -u frontend/openapi.json "$_openapi_regen" > /dev/null; then
        _blocked "OpenAPI snapshot is stale"
        echo "Run 'npm run regenerate-openapi-snapshot' (or"
        echo "'python scripts/dump_openapi.py > frontend/openapi.json') and"
        echo "commit the result."
        diff -u frontend/openapi.json "$_openapi_regen" | head -80
        rm -f "$_openapi_regen" "$_openapi_dump_log"
        exit 1
    fi
    rm -f "$_openapi_regen" "$_openapi_dump_log"

    # Generated doc-inventory drift check: regenerate the registry-backed
    # tables embedded in the docs (embedders, plugin families, demo datasets,
    # ...) and fail if any committed region is stale. Same shape as the
    # OpenAPI snapshot gate above; see scripts/gen-docs-inventories.py.
    echo "Checking generated doc inventories..."
    if ! python scripts/gen-docs-inventories.py --check ; then
        _blocked "generated doc inventories are stale"
        echo "Run 'python scripts/gen-docs-inventories.py' and commit the"
        echo "result."
        exit 1
    fi

    echo "Checking Dockerfiles..."
    if ! python scripts/check-dockerfiles.py ; then
        _blocked "Dockerfile check failed"
        exit 1
    fi

    # User-docs screenshot wiring: every manifest shot id has both theme PNGs on
    # disk, and every screenshot the user docs embed maps to a manifest id. Cheap,
    # browser-free (see docs/plans/user-docs-screenshots.md); the pixel-diff
    # (check.sh) needs chromium and stays a manual chore.
    echo "Checking user-docs screenshot wiring..."
    if ! python scripts/screenshots/wiring-check.py ; then
        _blocked "user-docs screenshot wiring check failed"
        exit 1
    fi

    # vtscore package docs: every top-level module / sub-package of vtscore/ is
    # covered by a packages/ doc, and no doc cites a file.py:NNN line anchor (they
    # rot on the next edit; cite module-and-symbol instead). Regex sweep, imports
    # nothing, ~0.1s. See scripts/check-vtscore-docs.py for the policy.
    echo "Checking vtscore package docs..."
    if ! python scripts/check-vtscore-docs.py ; then
        _blocked "vtscore package docs check failed"
        exit 1
    fi

    # Extension docs: docs/EXTENDING-*.md and vtscore/docs/extending/ state the
    # contract for the same plugin ABCs in their own words, so each can drift
    # from the code and from the other. This holds both to the class: every
    # member named in a contract table exists, and neither presents a public
    # wrapper as the override point when an _impl hook sits behind it. AST
    # sweep, imports nothing, ~1s. See scripts/check-extension-docs.py.
    echo "Checking extension docs..."
    if ! python scripts/check-extension-docs.py ; then
        _blocked "extension docs check failed"
        exit 1
    fi

    # Calibration script index: every .py/.sh in scripts/experiments/calibration/
    # is filed under exactly one study (or the shared layer) in that directory's
    # README, and every file the index names exists. The directory is flat by
    # decision (#3409), so the table IS the navigation - and an unchecked table
    # decays back into 120 unclassified files. Pure stdlib, ~20ms.
    echo "Checking calibration script index..."
    if ! python scripts/check-calibration-index.py ; then
        _blocked "calibration script index is out of date"
        exit 1
    fi
fi

# Slide decks: every deck manifest names fragments that exist and figures that
# resolve. Marp only *warns* on a missing figure and still exits 0, so a rotted
# deck is otherwise silent until someone rebuilds it the morning of a talk.
# Pure stdlib, reads slides/ only, ~0.05s. See slides/README.md.
echo "Checking slide decks..."
if ! python slides/build.py --check ; then
    _blocked "slide deck preflight failed"
    exit 1
fi

# Eval/app sync: the eval framework reproduces a handful of app surfaces it
# cannot call (the TypeScript autopilot phase machine, the app's default
# resolution). This gate notices when one of those app surfaces changes, so the
# eval default arm can't quietly stop being the shipped algorithm. Parses
# source, imports nothing, ~0.3s.
if ! $_is_slides_run; then
    echo "Checking eval/app sync..."
    if ! python scripts/check-eval-app-sync.py ; then
        _blocked "eval framework is out of sync with the app"
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# Stage 2: frontend production build (serial).
#
# Ahead of the parallel stage on purpose. `tests/core/test_frontend.py` and
# parts of `tests/api/test_dashboard.py` serve the built bundle out of
# `static/`, so building it *while* pytest reads it would race; and a broken
# build is the frontend failure that actually happens, so it is worth learning
# about before the long stage starts.
# ---------------------------------------------------------------------------
if $_run_frontend_check && [ -d "frontend/node_modules" ]; then
    echo "Checking frontend TypeScript build..."
    _fe_log=$(mktemp)
    if (cd frontend && npm run build:prod 2>&1) > "$_fe_log"; then
        # Treat Angular compiler warnings (e.g. NG8107) and budget warnings as
        # errors. Angular colourises its output even when stdout is a file, and
        # it interleaves the escapes *inside* the marker
        # (`ESC[33m▲ ESC[43;33m[ESC[43;30mWARNING…`), so the literal
        # `▲ [WARNING]` never matches the raw log — that blindness let an
        # over-budget initial bundle sail past this gate for months. Match
        # against an ANSI-stripped copy instead.
        _fe_plain=$(mktemp)
        sed -r 's/\x1b\[[0-9;]*m//g' "$_fe_log" > "$_fe_plain"
        if grep -q '▲ \[WARNING\]' "$_fe_plain"; then
            _blocked "Frontend build has warnings (treated as errors)"
            grep -A 10 '▲ \[WARNING\]' "$_fe_plain"
            rm -f "$_fe_log" "$_fe_plain"
            exit 1
        fi
        rm -f "$_fe_plain"
        echo "Frontend build OK"
    else
        _blocked "Frontend build failed"
        cat "$_fe_log"
        rm -f "$_fe_log"
        exit 1
    fi
    rm -f "$_fe_log"
elif $_run_frontend_check && [ ! -d "frontend/node_modules" ]; then
    echo "Skipping frontend build check (node_modules not installed; run: cd frontend && npm install)"
fi

# ---------------------------------------------------------------------------
# Stage 3: the heavy, mutually independent stages, run concurrently.
#
# pytest runs in the foreground so its progress still streams live (it is the
# output people actually watch); everything else runs as a background lane
# writing to its own log, reported after pytest returns. Every lane runs to
# completion even if another fails, so one pass surfaces every problem.
# ---------------------------------------------------------------------------

# pip-audit: scans installed Python packages against the PyPI advisory
# database. Auditing the resolved venv (not requirements files) catches
# transitive vulnerabilities and matches what production will actually run.
#
# `PIP_AUDIT_IGNORE` lists advisory IDs that pip-audit currently reports
# with no fix version available; pinning a "fixed" release isn't an
# option, so the gate would otherwise block indefinitely on upstream
# CVEs that have nothing to do with VTSearch code.  Re-audit the list
# whenever upstream ships a patched release; remove the entry and let
# `ensure-test-deps.sh` upgrade the dep instead of ignoring the CVE.
#   joblib 1.5.3       PYSEC-2024-277             (no upstream fix)
#   pyjwt  2.12.1      PYSEC-2025-183             (no upstream fix)
#   transformers 5.8.1 PYSEC-2025-211..218        (no upstream fix)
#   httplib2 0.20.4    PYSEC-2026-3444            (not a VTSearch dep; pulled in
#                                                  by launchpadlib in the Ubuntu
#                                                  base image, so requirements
#                                                  can't upgrade it)
PIP_AUDIT_IGNORE=(
    --ignore-vuln PYSEC-2024-277
    --ignore-vuln PYSEC-2025-183
    --ignore-vuln PYSEC-2025-211
    --ignore-vuln PYSEC-2025-212
    --ignore-vuln PYSEC-2025-213
    --ignore-vuln PYSEC-2025-214
    --ignore-vuln PYSEC-2025-215
    --ignore-vuln PYSEC-2025-216
    --ignore-vuln PYSEC-2025-217
    --ignore-vuln PYSEC-2025-218
    --ignore-vuln PYSEC-2026-3444
)

_lane_names=()
_lane_pids=()
_lane_logs=()

_start_lane() {
    local name="$1"; shift
    local log
    log=$(mktemp)
    "$@" > "$log" 2>&1 &
    _lane_pids+=("$!")
    _lane_names+=("$name")
    _lane_logs+=("$log")
}

# Pyright: full static type check across vtsearch/ and tests/
# (see `pyrightconfig.json` for the gated scope). The PYRIGHT_PYTHON_FORCE_VERSION
# pin keeps everyone on the same underlying pyright binary regardless of
# what the `pyright` PyPI wrapper would otherwise pull.
_lane_pyright() { PYRIGHT_PYTHON_FORCE_VERSION=1.1.408 pyright; }
_lane_pip_audit() { pip-audit "${PIP_AUDIT_IGNORE[@]}"; }
# Vulture whitelist hygiene. NOT the dead-code audit itself: a vulture hit on a
# public vtscore name is not evidence of anything (out-of-tree extensions import
# names no in-repo grep can see), so the findings stay a human pre-release chore
# per docs/RELEASE.md step 1. What *is* mechanically checkable is the whitelist:
# an entry that suppresses no finding is unfalsifiable and will outlive the
# symbol it was written for. That is how the file rotted before -- 49 of 102
# entries were asserting nothing. Whole-repo scan (~5s), hence a lane.
_lane_vulture_whitelist() { python scripts/vulture-audit.py --check-whitelist; }
# Audits the whole tree, dev deps included. This used to be `--omit=dev`
# because `@angular-devkit/build-angular` dragged in the webpack toolchain
# (webpack-dev-server → sockjs/express/launch-editor), which regularly carried
# advisories with no upstream fix. That package was never used -- angular.json
# names only `@angular/build:*` builders -- so it was swapped for a direct
# `@angular/build` devDependency (#3439), taking ~400 packages and every
# unfixable advisory with it. Keep the gate wide: if a dev-only advisory ever
# has genuinely no fix, pin it in `overrides` or carve it out here explicitly,
# rather than re-blinding the audit to the entire dev tree.
_lane_npm_audit() { (cd frontend && npm audit); }
# `npm run test:ci` regenerates the API client (pretest:ci) then runs
# `ng test --no-watch`, which exits non-zero on any spec failure.
_lane_frontend_unit() { (cd frontend && npm run test:ci); }

if $_run_whole_repo_gates; then
    _start_lane "pyright" _lane_pyright
    _start_lane "pip-audit" _lane_pip_audit
    _start_lane "vulture whitelist" _lane_vulture_whitelist
fi
if $_run_frontend_check && [ -d "frontend/node_modules" ]; then
    _start_lane "npm audit" _lane_npm_audit
fi
if $_run_frontend_unit && [ -d "frontend/node_modules" ]; then
    _start_lane "frontend unit tests (Vitest)" _lane_frontend_unit
elif $_run_frontend_unit && [ ! -d "frontend/node_modules" ]; then
    echo "Skipping frontend unit tests (node_modules not installed; run: cd frontend && npm install)"
fi

if [[ ${#_lane_names[@]} -gt 0 ]]; then
    echo "Running in parallel with the tests: ${_lane_names[*]}"
fi
if $_is_slides_run; then
    # Deliberately not the "a full run is the gate before pushing" notice below:
    # for a change confined to slides/ this *is* the gate. Say what was skipped
    # and why it is sound, so the claim stays auditable rather than folkloric.
    echo "Slides-only run: pytest and every whole-repo gate skipped — nothing"
    echo "outside slides/ changed, and no test or type-check reads a deck."
elif $_is_docs_run; then
    # Also not the "a full run is the gate" notice: for a change confined to
    # tracked markdown this *is* the gate. Stage 1 ran whole; what is left is
    # the tests that can actually open a doc.
    echo "Markdown-only run: every cheap gate ran; pytest is narrowed to the"
    echo "tests that read a repo doc (${_docs_targets[*]}), and pyright,"
    echo "pip-audit, the vulture whitelist check and the frontend lanes are"
    echo "skipped — none of them can see a markdown file."
elif ! $_run_whole_repo_gates; then
    echo "Group run: skipping pyright, pip-audit and the vulture whitelist check."
    echo "A full './run-tests.sh' is the gate before pushing (or set"
    echo "VTSEARCH_FULL_GATES=1 to force them)."
fi

# --- pytest, in the foreground ---------------------------------------------
_pytest_status=0
if $_run_pytest; then
    # Build the pytest marker expression
    if $_is_full_run; then
        # Default: run all fast tests
        MARKER_EXPR=""
    else
        # Combine groups with OR: -m "core or sorting"
        MARKER_EXPR=""
        for g in "${TEST_GROUPS[@]}"; do
            if [[ -n "$MARKER_EXPR" ]]; then
                MARKER_EXPR="$MARKER_EXPR or $g"
            else
                MARKER_EXPR="$g"
            fi
        done
    fi

    # Coverage is opt-in via VTSEARCH_COVERAGE=1. Default off because it adds
    # ~10-20% overhead and the report is most useful when explicitly asked for.
    COV_ARGS=()
    if [[ "${VTSEARCH_COVERAGE:-}" == "1" ]]; then
        COV_ARGS=(--cov=vtsearch --cov-report=term-missing)
    fi

    # Run pytest with:
    #   --tb=short: brief tracebacks (enough to diagnose, not overwhelming)
    #   --no-header: skip the platform/plugin header noise
    #   -q:         quiet mode (dots instead of full test names)
    #   -n auto:    parallel execution via pytest-xdist (one worker per CPU)
    #   --dist loadgroup: like the default load scheduling, except tests marked
    #       @pytest.mark.xdist_group run together on one worker. Used to pin all
    #       real-UMAP-fit tests to a single worker so the ~30s numba JIT compile
    #       of umap-learn's kernels is paid once per run rather than once per
    #       worker (tests_lib/projection/test_umap_projection.py), and to keep
    #       each eval-sweep module together so its memoized sweeps hit the cache
    #       instead of being recomputed on every worker that draws one
    #       (tests_lib/detectors/sweep_cache.py).
    #
    # Both tests/ (app tier) and tests_lib/ (library tier) are passed in.
    # The two trees have independent conftests; pytest's auto-merge picks
    # the right autouse fixtures per test based on file location.
    set +e
    if $_is_docs_run; then
        # Paths, not a marker: the markdown surface is one folder plus one file
        # in another group, and no marker expression names that set. Leaving -m
        # off keeps pyproject.toml's `not gpu and not slow` default in force.
        python -m pytest "${_docs_targets[@]}" -q --tb=short --no-header -n auto --dist loadgroup ${COV_ARGS[@]+"${COV_ARGS[@]}"} ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
    elif [[ -n "$MARKER_EXPR" ]]; then
        python -m pytest tests/ tests_lib/ -q --tb=short --no-header -n auto --dist loadgroup -m "$MARKER_EXPR" ${COV_ARGS[@]+"${COV_ARGS[@]}"} ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
    else
        python -m pytest tests/ tests_lib/ -q --tb=short --no-header -n auto --dist loadgroup ${COV_ARGS[@]+"${COV_ARGS[@]}"} ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
    fi
    _pytest_status=$?
    set -e
fi

# --- collect the background lanes ------------------------------------------
_failed_lanes=()
if [[ ${#_lane_names[@]} -gt 0 ]]; then
    echo ""
    echo "Waiting for: ${_lane_names[*]}"
    for _i in "${!_lane_pids[@]}"; do
        set +e
        wait "${_lane_pids[$_i]}"
        _st=$?
        set -e
        if [[ $_st -eq 0 ]]; then
            echo "  ${_lane_names[$_i]}: OK"
        else
            echo "  ${_lane_names[$_i]}: FAILED"
            _failed_lanes+=("$_i")
        fi
    done
fi

# Report every failing lane, not just the first: the point of running them
# concurrently is that one pass tells you everything that is broken.
for _i in ${_failed_lanes[@]+"${_failed_lanes[@]}"}; do
    _blocked "${_lane_names[$_i]} failed"
    tail -80 "${_lane_logs[$_i]}"
done
for _log in ${_lane_logs[@]+"${_lane_logs[@]}"}; do
    rm -f "$_log"
done

# Final verdict banner. With lanes reporting after pytest, pytest's own
# summary is no longer the last thing printed — this banner is, so "read the
# last ====-bordered block" stays the way to read a run's outcome.
echo ""
echo "============================================================"
if [[ ${#_failed_lanes[@]} -gt 0 || $_pytest_status -ne 0 ]]; then
    _verdict_parts=()
    if [[ $_pytest_status -ne 0 ]]; then
        _verdict_parts+=("pytest")
    fi
    for _i in ${_failed_lanes[@]+"${_failed_lanes[@]}"}; do
        _verdict_parts+=("${_lane_names[$_i]}")
    done
    _joined=$(IFS=", "; echo "${_verdict_parts[*]}")
    echo "RUN FAILED: $_joined"
    echo "============================================================"
    if [[ $_pytest_status -ne 0 ]]; then
        exit $_pytest_status
    fi
    exit 1
fi
if $_is_docs_run; then
    echo "RUN PASSED (markdown-only; this is the full gate for a markdown-only change)"
elif $_run_pytest; then
    echo "RUN PASSED (all gates green; pytest summary above)"
elif $_is_slides_run; then
    echo "RUN PASSED (slides-only; this is the full gate for a slides-only change)"
else
    echo "RUN PASSED (frontend-only; no Python tests in the 'frontend' group)"
fi
echo "============================================================"
exit 0
