#!/usr/bin/env python3
"""Drift gate: the eval framework's default arm vs. the app it is meant to model.

`vtscore.eval` exists to measure *deviations* from the shipped algorithm.  That
only means something if the framework's **default arm** is the shipped
algorithm.  When it isn't, every experiment run against it is measuring a
detector nobody uses, and the damage is silent and retroactive - the numbers
still look fine.

Most of the harness is safe by construction because it *delegates*: the
`max_patch` style calls `pool_box_from_media` / `bad_negative_vecs` /
`media_score_rows` rather than re-deriving them, so it cannot drift.  This gate
covers the parts that can't delegate:

* **ported** - app logic re-implemented in the harness, because the original is
  unreachable (it lives in TypeScript) or unusable (it is wrapped in
  interactive, lock-guarded, single-detector caches).  A copy goes stale the
  moment the original moves.
* **default** - places where the harness resolves "no explicit arm" to whatever
  the app currently defaults to.  When the app's default changes, the harness
  keeps handing out the old one under the name "default".

Each mirror pins a digest of **both** sides.  A copy stays faithful only while
neither half moves without the other, so either half moving is the same event
and both have to be watched:

* `app-changed` - the original moved.  Reconcile the harness copy to it.
* `harness-changed` - the copy moved on its own, with the original standing
  still.  Confirm the copy still says what the app says; a harness edit that
  quietly re-points the default arm is drift that the app side can never
  reveal, and it is the direction the Smart-indicator plumbing actually drifted
  in (#2923).

Either way, once reconciled (or once you have confirmed nothing is owed), re-pin:

    python scripts/check-eval-app-sync.py --update

Digests ignore comments, docstrings and formatting, so only real logic changes
trip the gate.

A few harness sides are too coarse to digest - one function serving several
mirrors and much else besides - and say so in `no_harness_pin=`.  Those keep the
app-side pin alone; the fix, when the blind spot starts to matter, is to extract
the reproduction into its own helper (as #3403 did for the two `*_default`
mirrors) rather than to pin a thousand lines.

Adding a mirror: append a `Mirror(...)` to `MIRRORS` and run `--update`.  The
harness side is pinned by default, so opting out is a decision you have to write
down.  If the harness *intentionally* differs from the app, say so in
`divergence=` - the text is printed whenever the mirror trips, so the next
person reconciling it knows which differences are deliberate.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import re
import sys
import textwrap
import tokenize
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PINS_PATH = REPO_ROOT / "scripts" / "eval-app-sync.pins.json"

AUTOPILOT_TS = "frontend/src/app/services/autopilot-state.service.ts"
LABEL_VIEW_TS = "frontend/src/app/components/label-view/label-view.component.ts"
AUTO_SELECT_TS = "frontend/src/app/utils/auto-select-next.ts"


@dataclass(frozen=True)
class Mirror:
    """One app-side surface the eval harness reproduces rather than calls.

    Attributes:
        id: Stable key into the pins file.  Renaming one re-pins from scratch.
        app: The app-side source. ``py:<dotted.path.to.symbol>`` for Python,
            ``ts:<repo-relative file>::<anchor>`` for a TypeScript block (the
            anchor is matched literally, then brace-matched from the first
            ``{`` after it).
        harness: Where the reproduction lives, as ``<repo-relative file>::<symbol>``,
            or ``::<symbol>,<symbol>`` when the reproduction is spread over
            more than one top-level name.  Each is resolved by parsing, so a
            name surviving only inside a comment does not count as present, and
            deleting or renaming the harness side trips the gate.
        kind: ``ported`` or ``default`` - see the module docstring.
        note: What is reproduced, and what to re-check when either side moves.
        divergence: A *declared, intentional* difference from the app.  Not an
            exemption - the digest is still pinned - but it tells whoever
            reconciles this mirror which differences are on purpose.
        no_harness_pin: Why this mirror's harness side carries no digest of its
            own.  Absent (the default) means it does: a harness symbol that is
            this mirror's dedicated counterpart should be pinned, so opting out
            is a decision that has to be written down rather than defaulted
            into.  The reason is printed with the mirror whenever it trips.
    """

    id: str
    app: str
    harness: str
    kind: str
    note: str
    divergence: str | None = None
    no_harness_pin: str | None = None


MIRRORS: list[Mirror] = [
    # ------------------------------------------------------------------ ported
    Mirror(
        id="autopilot.phase_machine",
        app=f"ts:{AUTOPILOT_TS}::checkPhaseTransition(",
        harness="vtscore/eval/autopilot_flow.py::next_phase",
        kind="ported",
        note=(
            "The phase ordering and every transition trigger of the simulated Autopilot user. "
            "Lives in TypeScript, so there is nothing to import - it is a hand copy. If you "
            "add, remove or reorder a phase, or change what gates one, port the same change."
        ),
    ),
    Mirror(
        id="autopilot.vote_targets",
        app=f"ts:{AUTOPILOT_TS}::const INITIAL_STATE",
        harness="vtscore/eval/autopilot_flow.py::GOOD_TARGET,BAD_TARGET",
        kind="ported",
        note=(
            "goodToStart / badToStart are copied as GOOD_TARGET / BAD_TARGET, which decide how "
            "many votes the simulation spends before its first learned sort. Pinned literally "
            "by tests_lib/detectors/test_autopilot_flow.py::TestPortedConstants. Both harness "
            "constants are named here, because the mirror is the pair: watching only GOOD_TARGET "
            "would leave a change to BAD_TARGET alone as silent as the app-side half used to be."
        ),
    ),
    Mirror(
        id="autopilot.phase_sort_select",
        app=f"ts:{LABEL_VIEW_TS}::onAutopilotStop(",
        harness="vtscore/eval/al_strategies.py::_select_phase_faithful",
        kind="ported",
        note=(
            "Which Sort and which Select each Autopilot phase drives - good=text+top, "
            "bad=text+hard, hard=learned+hard, new=learned+new. The simulated user picks the "
            "next item from exactly this pairing, so re-pointing a phase at a different sort "
            "or select here silently changes what every study's vote order means. The load- "
            "bearing row is `bad`: it is still on the TEXT sort, so the harness must not "
            "consult detector scores there even though a model exists for measurement."
        ),
        divergence=(
            "onAutopilotStop applies the mapping when the user stops Autopilot; the live "
            "mapping is the phase subscription in the same component, which additionally "
            "kicks off the sort request. This anchor is the one place the whole table is "
            "written out in one block, so it is what the digest watches."
        ),
    ),
    Mirror(
        id="autopilot.auto_select_next",
        app=f"ts:{AUTO_SELECT_TS}::export function autoSelectNext(",
        harness="vtscore/eval/al_strategies.py::_hard_pick_by_index",
        kind="ported",
        note=(
            "The app's auto-advance rule - which item each Select mode shows next. `top` takes "
            "the highest-ranked unvoted row; `hard` takes the unvoted row nearest the "
            "acquisition cut BY RANK INDEX, not by score; `new` defers to the coverage atlas. "
            "_hard_pick_by_index is the `hard` branch verbatim, and it is the branch every "
            "simulated vote after the seed phase goes through, so a change to the rule that "
            "does not reach the harness silently re-points every study's vote order. Rank space "
            "is the load-bearing detail: a score-space argmin biases toward whichever side of "
            "the line is denser, which is the whole reason the app measures in indices. Note "
            "the cutoff index is computed over the FULL window, voted rows included, so it does "
            "not slide as votes accumulate - the harness's `ordered` list must stay the full "
            "ranking rather than the pool. Also re-check the tie rule: both sides scan "
            "ascending by index with a strict `<`, so an exact tie takes the higher-ranked row. "
            "Extracted out of the label-view component (#3428) so this digest covers the pick "
            "rule alone rather than the component's sort plumbing; "
            "frontend/src/app/utils/auto-select-next.spec.ts states the same cases executably."
        ),
        divergence=(
            "The harness mirrors only the `hard` branch here. `top` is trivial and is "
            "reproduced inline by the phase-faithful strategy; `new` is not ported at all - "
            "_atlas_next DELEGATES to CoverageAtlas.next_sample, the same call the app's New "
            "pick makes through /api/coverage-atlas/next, so it cannot drift and needs no pin. "
            "The app's guard that a `new` pick still requires a loaded ranking is a UI "
            "precondition (the window steers the probe's scores) with no harness counterpart, "
            "as is `excludeId`: it covers the instant between casting a vote and that vote "
            "landing in goodVotes/badVotes, which a simulation never observes because it "
            "records the vote before asking for the next pick."
        ),
    ),
    Mirror(
        id="autopilot.startup_default",
        app=f"ts:{AUTOPILOT_TS}::const INITIAL_STATE",
        harness="vtscore/eval/startup_schedule.py::PRODUCTION_STARTUP",
        kind="default",
        note=(
            "issue #3267 made the Autopilot opening a parameter, so the harness now has a "
            "spelling of the app's own opening - 'g3@top,b4@mid' - that a study's control arm "
            "runs. If goodToStart/badToStart move, or the opening stops being 'top of the "
            "seed sort then its cutoff', this constant has to move with them or every #3267 "
            "study measures its deviations from an opening nobody ships. "
            "tests_lib/detectors/test_startup_schedule.py pins it two ways: literally against "
            "GOOD_TARGET/BAD_TARGET, and behaviourally by requiring it to reproduce a "
            "default-arm run click for click."
        ),
    ),
    Mirror(
        id="progress.smart_status",
        app="py:vtscore.detectors.labeling_progress._compute_smart_status",
        harness="vtscore/eval/autopilot_flow.py::smart_status",
        kind="ported",
        note=(
            "The Smart indicator - error-cost flatness - one of the three gates the phase "
            "machine reads. Re-check the per-class minimum and the flatness threshold."
        ),
        divergence=(
            "The harness takes the error-cost window as an argument instead of reading a "
            "`_ProgressCache`'s `steps` model cache, which is built for one interactive "
            "detector advancing a vote at a time. The *rules* are copied; only the input "
            "plumbing differs - and only in where the models come from, not in how they are "
            "scored: the caller (`step_trainers._labelset_error_costs`) re-scores the "
            "whole window against the *current* labelset every step, as `_eval_cached_models` "
            "does. Handing in a history of frozen per-step costs instead would silently change "
            "the statistic the slope measures (issue #2923), which is not a declared divergence. "
            "The weighted-cost arithmetic underneath both is no longer a copy at all: since "
            "#3414 both score through `vtscore.training.thresholds.weighted_error_cost` and "
            "price inclusion through `inclusion_cost_weights`, so that half cannot drift and "
            "needs no pin - only the flatness rule above it does. "
            "Two consequences of #3757 are declared here rather than pinned. (1) The app "
            "windows its last SMART_WINDOW *models*, because most of its label steps have no "
            "model - only the ones a learned sort ran against do; the harness trains one per "
            "step, so its last-SMART_WINDOW-steps slice is the identical set and the "
            "`voting_iterations` caller says so at the `del recent_steps[:-SMART_WINDOW]`. "
            "(2) Both sides now score a model in its own serving geometry - the app through "
            "`scoring_rows_for_snap`, the harness through `score_sim_set_with_model` on the "
            "arm's style - rather than on whole-image vectors a patch head was never fitted "
            "on. Those live in the scorers, not in the flatness rule pinned here, so a change "
            "to either scorer trips the `progress.eval_geometry` mirror below instead."
        ),
    ),
    Mirror(
        id="progress.eval_geometry",
        app="py:vtscore.detectors.labeling_progress._score_step",
        harness="vtscore/eval/step_trainers.py::_labelset_error_costs",
        kind="ported",
        note=(
            "How a cached detector is *scored* when the Smart indicator measures it. Both "
            "sides must score a head the way that head is served - max-pooled over the rows "
            "it was fitted against - or the indicator measures a geometry nobody ships, which "
            "is half of issue #3757. If you change which rows either side scores, or the "
            "pooling over them, change both."
        ),
        divergence=(
            "The app reaches the rows through `scoring_rows_for_snap`, which is bound to the "
            "live dataset context and its cached region matrix; the harness has arms rather "
            "than one shipped geometry, so it routes through the arm's style object "
            "(`score_sim_set_with_model`) and keeps the whole-image `predict` for the "
            "whole_image / SVM arms, whose heads are fitted in that space. Both reduce to the "
            "same rule - score a head where it is served - and both delegate the arithmetic "
            "on top to `weighted_error_cost`."
        ),
    ),
    Mirror(
        id="progress.stable_status",
        app="py:vtscore.detectors.labeling_progress._compute_stable_status",
        harness="vtscore/eval/autopilot_flow.py::stable_status",
        kind="ported",
        note=(
            "The Stable indicator - prediction-flip rate. Re-check the per-class minimum, the "
            "minimum history length, and both the rate and max flip thresholds."
        ),
        divergence=(
            "Same input plumbing divergence as progress.smart_status: flip counts are passed "
            "in rather than read from a `_ProgressCache`'s `steps`."
        ),
    ),
    Mirror(
        id="progress.span_status",
        app="py:vtscore.detectors.labeling_progress._compute_span_status",
        harness="vtscore/eval/autopilot_flow.py::span_status",
        kind="ported",
        note=(
            "The Span indicator - coverage-atlas breadth - which drives the new -> done "
            "transition. Re-check the green target and the yellow cutoff."
        ),
        divergence=(
            "The app reads its green target from `CoreConfig.autopilot_goal_diversity`; the "
            "harness takes it per-run so a sweep can vary it, defaulting to the same value."
        ),
    ),
    # ----------------------------------------------------------------- default
    Mirror(
        id="training.train_and_threshold",
        app="py:vtscore.detectors.training.train_and_threshold",
        harness="vtscore/eval/step_trainers.py::_style_train_and_calibrate",
        kind="default",
        note=(
            "The app's canonical train + calibrate pipeline, which the harness reproduces step "
            "for step (fold calibration, full-data fit, fold-anchored threshold). A new stage "
            "here - or a changed fold rule or ordering - has to reach the harness or its "
            "default arm trains a detector the app no longer ships. Note "
            "_app_train_and_calibrate is the single-vector path and reproduces the same shape. "
            "The head is the one knob mirrored by name: this function's `hidden_dim` must equal "
            "`resolve_hidden_dim(step_model.PRODUCTION_HEAD, ...)`, which "
            "tests_lib/detectors/test_harness_linear_head.py pins by training this pipeline for "
            "real - so a head change fails the suite as well as tripping this digest."
        ),
    ),
    Mirror(
        id="training.fused_threshold",
        app="py:vtscore.detectors.training._fused_threshold",
        harness="vtscore/eval/voting_iterations.py::_safe_threshold_for_step",
        kind="default",
        note=(
            "How the cross-calibration cut and the population estimate are fused into the "
            "shipped threshold. The harness's reported operating point is only comparable to "
            "the app's if this rule matches."
        ),
        no_harness_pin=(
            "The harness side is _safe_threshold_for_step, the whole production-threshold path (150 lines, named "
            "by three mirrors, and carrying the arm knobs and per-fold timing that no mirror is about). Its "
            "digest would trip on edits with nothing to do with this mirror, and a pin people re-run --update on "
            "without reading is worse than no pin. The app-side digest still covers the direction that matters "
            "here. When this blind spot starts to matter, the fix is to extract the reproduction into its own "
            "helper - what #3403 did for the two *_default mirrors, giving them a harness side small enough to "
            "pin - not to digest the whole function."
        ),
    ),
    Mirror(
        id="training.calibration_score_rows",
        app="py:vtscore.detectors.training._calibration_score_rows",
        harness="vtscore/eval/voting_iterations.py::_safe_threshold_for_step",
        kind="default",
        note=(
            "Calibrating in *inference* geometry: each voted bag collapses over the rows the "
            "scorer will max-pool, not the rows the fold model trained on. Changing which rows "
            "the app calibrates over silently moves every threshold the harness reports."
        ),
        no_harness_pin=(
            "The harness side is _safe_threshold_for_step, the whole production-threshold path (150 lines, named "
            "by three mirrors, and carrying the arm knobs and per-fold timing that no mirror is about). Its "
            "digest would trip on edits with nothing to do with this mirror, and a pin people re-run --update on "
            "without reading is worse than no pin. The app-side digest still covers the direction that matters "
            "here. When this blind spot starts to matter, the fix is to extract the reproduction into its own "
            "helper - what #3403 did for the two *_default mirrors, giving them a harness side small enough to "
            "pin - not to digest the whole function."
        ),
    ),
    Mirror(
        id="training.blend_schedule_default",
        app="py:vtscore.detectors.training._blend_schedule_for_snap",
        harness="vtscore/eval/voting_iterations.py::_resolve_production_defaults",
        kind="default",
        note=(
            "How the app picks a safe-threshold blend schedule when none is named (per voting "
            "mode). _resolve_production_defaults resolves blend_schedule=None through "
            "production_schedule_for to match; if the app's choice becomes conditional on "
            "something else, that condition has to reach the harness too."
        ),
    ),
    Mirror(
        id="training.split_fraction_default",
        app="py:vtscore.detectors.training.resolve_calibration_fraction",
        harness="vtscore/eval/voting_iterations.py::_resolve_production_defaults",
        kind="default",
        note=(
            "How the app resolves calibration_fraction when the user has no explicit setting "
            "(#3287/#3290): the per-SPACE production split - 0.3 when the detector learns in a "
            "single-vector space, 0.5 on a patch grid, 0.5 when unknown - keyed on the "
            "embedder's supports_patch_regions capability, NOT on the voting mode. "
            "_resolve_production_defaults resolves calibration_fraction=None through the same "
            "production_split_for table, keyed on whether any media carries a patch_grid (the "
            "harness's spelling of 'built by a patch embedder'). The values themselves cannot "
            "drift - both sides read PRODUCTION_SPLIT_BY_SPACE - so what this digest watches is "
            "the *predicate*: if the app's choice starts depending on something patch_grid "
            "presence can't see (the voting mode, the dataset, a per-user knob), that condition "
            "has to reach the harness too."
        ),
    ),
    Mirror(
        id="thresholds.vote_exclusion_floor",
        app="py:vtscore.training.thresholds.anchored.resolve_exclusion_floor",
        harness="vtscore/eval/voting_iterations.py::_safe_threshold_for_step",
        kind="default",
        note=(
            "How the #3308 voted-media exclusion resolves its remainder floor when no override "
            "is given (#3312). The exclusion DECISION itself is not mirrored and must never "
            "become mirrored: both sides call apply_vote_exclusion / drop_voted, so the "
            "all-or-nothing contract and the filtering are shared code rather than two copies. "
            "What this digest watches is the *default*: the harness's exclusion_min_remainder "
            "arm knob defaults to None, and None has to keep meaning 'whatever a live detector "
            "does'. If the app's floor stops being one constant - keyed on the dataset size, "
            "the voting mode, a per-user setting - that resolution has to reach the harness "
            "too, or every arm in the #3312 grid is measured against a baseline nobody runs."
        ),
        divergence=(
            "INTENTIONAL: the harness accepts an explicit floor (0 = exclude unconditionally, "
            "math.inf = the pre-#3308 behaviour) where the app has no such setting. That is the "
            "#3312 arm axis and is exactly the kind of deliberate deviation the harness exists "
            "to measure; the DEFAULT arm still passes None and so resolves here."
        ),
        no_harness_pin=(
            "The harness side is _safe_threshold_for_step, the whole production-threshold path (150 lines, named "
            "by three mirrors, and carrying the arm knobs and per-fold timing that no mirror is about). Its "
            "digest would trip on edits with nothing to do with this mirror, and a pin people re-run --update on "
            "without reading is worse than no pin. The app-side digest still covers the direction that matters "
            "here. When this blind spot starts to matter, the fix is to extract the reproduction into its own "
            "helper - what #3403 did for the two *_default mirrors, giving them a harness side small enough to "
            "pin - not to digest the whole function."
        ),
    ),
    Mirror(
        id="thresholds.fold_anchored_fit_then_cut",
        app="py:vtscore.training.thresholds.anchored.fold_anchored_gmm_threshold",
        harness="vtscore/eval/arms_inclusion.py::_cut_inclusion_arms",
        kind="default",
        note=(
            "The app composes a fold-anchored threshold as fit_fold_anchored_cut(...) then "
            "cut.threshold_at(inclusion). The #2865 sweep re-cuts ONE fit per anchor weight "
            "across the whole (rule, combine, k) grid, so it calls those two halves itself "
            "instead of calling this function per grid point. The two are identical only for as "
            "long as this function has nothing BETWEEN the fit and the cut. If a stage is added "
            "there - a clamp, a smoothing pass, a provenance-dependent substitution - the sweep "
            "silently stops measuring the shipped estimator, and its rows would still look "
            "perfectly reasonable. Re-check that the composition is still fit-then-cut. "
            "Checked at #3166: the empty-interval snap that fixed the saturated-distribution "
            "threshold went INSIDE threshold_at, not between the fit and the cut, precisely so "
            "the sweep picks it up for free and this composition stayed untouched."
        ),
        divergence=(
            "INTENTIONAL: the harness does NOT reproduce this function's terminal fallbacks (the "
            "final model's unanchored midpoint, then its median) when no fold yields a fit. Both "
            "are inclusion-blind by construction, so they would enter the #2865 frame as arms "
            "that trivially lose the knob-liveness comparison while saying nothing about the cut "
            "rule under test; the sweep skips that anchor weight instead. Fitting once and "
            "re-cutting is also what production itself does on an Inclusion slide "
            "(recompute_detector_thresholds_for_inclusion), so the sweep measures the object the "
            "app re-cuts rather than a chain of independent retrains."
        ),
    ),
    Mirror(
        id="thresholds.rate_cut_no_root",
        app="py:vtscore.training.thresholds.gmm._rate_cut",
        harness="vtscore/eval/cut_rules.py::gaussian_cuts",
        kind="default",
        note=(
            "What the 'rate' rule does on a fit whose density crossing has no root between the "
            "component means. This has moved twice already - midpoint (pre-2026-08-06), bare "
            "edge, then continued past the edge at the rule's first-order slope (#2896) - and "
            "each move silently changed what the harness's *_rate arms mean relative to the "
            "app. Pinned so the next move prompts a decision instead of going unnoticed for "
            "days, which is exactly how #2900 happened."
        ),
        divergence=(
            "INTENTIONAL, decided in #2900: gaussian_cuts reports NaN and _safe_gmm_variant_rows "
            "substitutes that fit's MIDPOINT, where production continues past the component "
            "mean. The decomposition family compares cut rules against each other on one fit, so "
            "it wants a neutral, rule-independent stand-in - production's continuation exists "
            "only for 'rate' and would make it incomparable to its cross/priorfree siblings (at "
            "inclusion 0 it would break the rate == priorfree identity every report in "
            "docs/experiments/2026-08-04-gmm-cut/ relies on). The divergence is recorded per row in "
            "cut_fallback_kind ('midpoint' vs 'continued'/'degenerate_midpoint'), so an analysis "
            "that needs the shipped path filters on it. The fold-anchored family calls "
            "gmm_cut_from_fit directly and is the faithful stand-in for the app. When re-pinning "
            "this mirror, re-check that the divergence is still the one you want AND that "
            "cut_fallback still fires on the same fits in both families - the flag being "
            "comparable is what keeps fallback_rate aggregates joinable across them."
        ),
    ),
]


class MirrorError(Exception):
    """A mirror could not be resolved at all - the app or harness side moved."""


# --------------------------------------------------------------------- digests


def _rel(path: Path) -> str:
    """*path* as the repo-relative string an error message should show."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:  # A path from outside the tree - only reachable from tests.
        return str(path)


def _source_slice(lines: list[str], start: tuple[int, int], end: tuple[int, int]) -> str:
    """The literal source text between two `(row, col)` token positions."""
    (srow, scol), (erow, ecol) = start, end
    if srow == erow:
        return lines[srow - 1][scol:ecol]
    parts = [lines[srow - 1][scol:]]
    parts.extend(lines[row - 1] for row in range(srow + 1, erow))
    parts.append(lines[erow - 1][:ecol])
    return "".join(parts)


def _collapse_fstrings(tokens: list[tokenize.TokenInfo], lines: list[str]) -> list[tokenize.TokenInfo]:
    """Re-join PEP 701's split f-string tokens into the one token <=3.11 emits.

    Python 3.12 stopped tokenizing an f-string as a single STRING and started
    emitting `FSTRING_START` / `FSTRING_MIDDLE` / `FSTRING_END` around the real
    tokens of each replacement field.  That is a pure tokenizer change - the
    code means the same thing - but it changes the token *text*, so a digest
    taken on 3.12+ disagreed with one taken on 3.10/3.11 for any mirrored
    function containing an f-string, and `--update` just moved the failure to
    the other half of the supported range instead of converging (issue #3117).

    Splicing the original source back out by token position restores the older
    single-token form on every interpreter, so a pin travels between them.
    """
    start_type = getattr(tokenize, "FSTRING_START", None)
    if start_type is None:  # <=3.11 already emits one STRING token.
        return tokens
    end_type = tokenize.FSTRING_END
    out: list[tokenize.TokenInfo] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.type != start_type:
            out.append(tok)
            i += 1
            continue
        # Nested f-strings are legal on 3.12+, so match by depth, not first END.
        depth = 0
        j = i
        while j < len(tokens):
            if tokens[j].type == start_type:
                depth += 1
            elif tokens[j].type == end_type:
                depth -= 1
                if depth == 0:
                    break
            j += 1
        else:  # pragma: no cover - tokenize raises on an unterminated string first.
            raise MirrorError("unterminated f-string while normalizing source")
        end = tokens[j]
        text = _source_slice(lines, tok.start, end.end)
        out.append(tokenize.TokenInfo(tokenize.STRING, text, tok.start, end.end, tok.line))
        i = j + 1
    return out


def _normalize_python(source: str) -> str:
    """Source text stripped of comments, docstrings and formatting.

    Token-based rather than AST-based on purpose: `ast.unparse` output is not
    guaranteed stable across the Python versions this repo supports (>=3.10),
    which would make the pins fail for whoever is not on the pinning machine's
    interpreter.  Token text is *nearly* stable - see `_collapse_fstrings` for
    the one place it isn't, and how that is normalized away.

    Magic trailing commas are dropped as well, so that `ruff format` wrapping a
    call across lines - which it will do for a change as innocent as a longer
    variable name - doesn't read as a logic change.
    """
    skip = (tokenize.COMMENT, tokenize.NL, tokenize.ENCODING, tokenize.ENDMARKER)
    out: list[str] = []
    prev_meaningful: int | None = None
    text = textwrap.dedent(source) + "\n"
    tokens = _collapse_fstrings(
        list(tokenize.generate_tokens(io.StringIO(text).readline)),
        text.splitlines(keepends=True),
    )
    for i, tok in enumerate(tokens):
        if tok.type in skip:
            continue
        nxt = next((t for t in tokens[i + 1 :] if t.type not in skip), None)
        if tok.type == tokenize.STRING and prev_meaningful in (None, tokenize.NEWLINE, tokenize.INDENT):
            # A string that is an entire statement: a docstring.
            if nxt is not None and nxt.type == tokenize.NEWLINE:
                continue
        if tok.string == "," and nxt is not None and nxt.string in (")", "]", "}"):
            continue
        out.append(tok.string.strip() if tok.type == tokenize.NEWLINE else tok.string)
        prev_meaningful = tok.type
    return "\n".join(s for s in out if s)


_TS_LINE_COMMENT = re.compile(r"//[^\n]*")
_TS_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)


def _normalize_typescript(source: str) -> str:
    """TypeScript block stripped of comments and collapsed whitespace."""
    stripped = _TS_BLOCK_COMMENT.sub(" ", source)
    stripped = _TS_LINE_COMMENT.sub(" ", stripped)
    return re.sub(r"\s+", " ", stripped).strip()


def _ts_block(path: Path, anchor: str) -> str:
    """The brace-delimited block introduced by *anchor* in *path*.

    Comments are removed before brace-matching so a brace inside a comment
    cannot unbalance the scan.
    """
    text = _TS_BLOCK_COMMENT.sub(" ", path.read_text(encoding="utf-8"))
    text = _TS_LINE_COMMENT.sub(" ", text)
    start = text.find(anchor)
    if start < 0:
        raise MirrorError(f"anchor {anchor!r} not found in {_rel(path)}")
    if text.find(anchor, start + len(anchor)) >= 0:
        raise MirrorError(f"anchor {anchor!r} is ambiguous in {_rel(path)} (matches more than once)")
    open_idx = text.find("{", start)
    if open_idx < 0:
        raise MirrorError(f"no block follows anchor {anchor!r} in {_rel(path)}")
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise MirrorError(f"unbalanced block for anchor {anchor!r} in {_rel(path)}")


def _py_symbol_source(path: Path, symbol: str) -> str:
    """The source of top-level *symbol* in *path*, found by parsing not importing.

    Parsing keeps the gate fast and dependency-free: it runs before the test
    suite has installed torch, and reading a module for its text should never
    execute it.
    """
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    lines = text.splitlines()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == symbol:
            start = min([node.lineno, *[d.lineno for d in node.decorator_list]]) - 1
            return "\n".join(lines[start : node.end_lineno])
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id == symbol for t in targets):
                return "\n".join(lines[node.lineno - 1 : node.end_lineno])
    raise MirrorError(f"{_rel(path)} has no top-level {symbol!r} - did it move or get renamed?")


def _app_source(mirror: Mirror) -> str:
    """The normalized app-side source *mirror* is pinned against."""
    kind, _, ref = mirror.app.partition(":")
    if kind == "py":
        module_path, _, symbol = ref.rpartition(".")
        path = REPO_ROOT / (module_path.replace(".", "/") + ".py")
        if not path.exists():
            raise MirrorError(f"module {module_path} ({_rel(path)}) does not exist")
        return _normalize_python(_py_symbol_source(path, symbol))
    if kind == "ts":
        rel, _, anchor = ref.partition("::")
        path = REPO_ROOT / rel
        if not path.exists():
            raise MirrorError(f"{rel} does not exist")
        return _normalize_typescript(_ts_block(path, anchor))
    raise MirrorError(f"unknown app source kind {kind!r} in {mirror.app!r}")


def _harness_source(mirror: Mirror) -> str:
    """The normalized source of the harness symbol(s) *mirror* names.

    Resolved by parsing rather than by searching the file text.  The existence
    check used to be ``symbol in path.read_text()``, which a name surviving only
    inside a comment - or inside the docstring explaining that the reproduction
    was removed - satisfied just as well as the reproduction itself.
    """
    rel, _, symbols = mirror.harness.partition("::")
    path = REPO_ROOT / rel
    if not path.exists():
        raise MirrorError(f"harness file {rel} does not exist")
    names = [s.strip() for s in symbols.split(",") if s.strip()]
    if not names:
        raise MirrorError(f"harness ref {mirror.harness!r} names no symbol")
    parts: list[str] = []
    for name in names:
        try:
            parts.append(_normalize_python(_py_symbol_source(path, name)))
        except MirrorError as exc:
            raise MirrorError(f"harness symbol {name!r} not found in {rel} - was it renamed or removed?") from exc
    return "\n".join(parts)


def _digest(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _digests(mirror: Mirror) -> dict[str, str]:
    """Every side of *mirror* that carries a pin, keyed as the pins file is.

    Resolving the harness side is not conditional on pinning it: a mirror whose
    reproduction has been deleted is unresolvable whether or not its digest is
    recorded, and that is the failure the manifest most wants to hear about.
    """
    harness = _harness_source(mirror)
    pins = {"app": _digest(_app_source(mirror))}
    if mirror.no_harness_pin is None:
        pins["harness"] = _digest(harness)
    return pins


# ----------------------------------------------------------------------- pins


def _load_pins() -> dict[str, dict[str, str]]:
    """The recorded digests, as ``{mirror id: {side: digest}}``."""
    if not PINS_PATH.exists():
        return {}
    raw = json.loads(PINS_PATH.read_text(encoding="utf-8"))
    for key, value in raw.items():
        if not isinstance(value, dict):
            raise SystemExit(
                f"{_rel(PINS_PATH)} records {key!r} as a bare digest, which is the "
                "one-sided format from before the harness side was pinned too. "
                "Run: python scripts/check-eval-app-sync.py --update"
            )
    return raw


def _write_pins(pins: dict[str, dict[str, str]]) -> None:
    PINS_PATH.write_text(json.dumps(pins, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@dataclass
class Drift:
    mirror: Mirror
    reason: str
    detail: str = ""
    extras: list[str] = field(default_factory=list)


def check() -> list[Drift]:
    """Every mirror where either side moved, or whose two sides no longer resolve.

    One Drift per mirror, not per side: both halves of a copy moving together is
    the ordinary shape of a faithful port, and reporting it twice would print
    the mirror's whole note twice for one reconciliation.
    """
    pins = _load_pins()
    drifts: list[Drift] = []
    seen: set[str] = set()
    for mirror in MIRRORS:
        if mirror.id in seen:
            raise SystemExit(f"duplicate mirror id {mirror.id!r} in MIRRORS")
        seen.add(mirror.id)
        try:
            digests = _digests(mirror)
        except MirrorError as exc:
            drifts.append(Drift(mirror, "unresolvable", str(exc)))
            continue
        pinned = pins.get(mirror.id, {})
        reasons: list[str] = []
        details: list[str] = []
        for side, digest in digests.items():
            recorded = pinned.get(side)
            if recorded is None:
                reasons.append(f"{side}-unpinned")
                details.append(f"{side}: no digest recorded yet")
            elif recorded != digest:
                reasons.append(f"{side}-changed")
                details.append(f"{side}: pinned {recorded[:12]}, now {digest[:12]}")
        orphans = sorted(set(pinned) - set(digests))
        if orphans:
            reasons.append("side-not-pinned-anymore")
            details.append(
                "digests recorded for sides this mirror no longer pins: "
                + ", ".join(orphans)
                + " (--update drops them)"
            )
        if reasons:
            drifts.append(Drift(mirror, ", ".join(reasons), "; ".join(details)))
    stale = sorted(set(pins) - {m.id for m in MIRRORS})
    if stale:
        drifts.append(
            Drift(
                MIRRORS[0],
                "stale-pins",
                "pins recorded for mirrors that no longer exist: " + ", ".join(stale),
                extras=stale,
            )
        )
    return drifts


def update() -> int:
    pins: dict[str, dict[str, str]] = {mirror.id: _digests(mirror) for mirror in MIRRORS}
    _write_pins(pins)
    unpinned = [m.id for m in MIRRORS if m.no_harness_pin is not None]
    print(f"Pinned {len(pins)} eval/app mirrors to {_rel(PINS_PATH)}")
    if unpinned:
        print(f"  ({len(unpinned)} app-side only, by declaration: {', '.join(unpinned)})")
    return 0


def _report(drifts: list[Drift]) -> None:
    print("The eval framework and the app code it mirrors no longer agree.")
    print("")
    print("The eval default arm has to BE the app's algorithm - that is the only")
    print("thing that makes a deviation arm meaningful. Reconcile each mirror below,")
    print("then re-pin with:  python scripts/check-eval-app-sync.py --update")
    for drift in drifts:
        mirror = drift.mirror
        print("")
        if drift.reason == "stale-pins":
            print(f"  * {drift.detail}")
            print("    Run --update to drop them.")
            continue
        print(f"  * {mirror.id}  [{mirror.kind}, {drift.reason}]")
        print(f"      app:     {mirror.app.partition(':')[2]}")
        print(f"      harness: {mirror.harness}")
        print(f"      {mirror.note}")
        if mirror.divergence:
            print(f"      DECLARED DIVERGENCE: {mirror.divergence}")
        if mirror.no_harness_pin:
            print(f"      HARNESS SIDE NOT PINNED: {mirror.no_harness_pin}")
        if drift.detail:
            print(f"      {drift.detail}")
    print("")
    print("app-changed:     the original moved. Reconcile the harness copy to it, then --update.")
    print("harness-changed: the copy moved while the original stood still. Re-read the two")
    print("                 against each other: this is the direction a harness edit can")
    print("                 silently re-point the default arm, which is what #2923 was.")
    print("")
    print("If the harness already tracks the change (or is unaffected), --update alone")
    print("is the right answer. If the harness now *intentionally* differs, record why")
    print("in the mirror's divergence= field in scripts/check-eval-app-sync.py.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--update",
        action="store_true",
        help="re-pin every mirror to the current source on both sides, after reconciling them",
    )
    args = parser.parse_args(argv)
    if args.update:
        return update()
    drifts = check()
    if drifts:
        _report(drifts)
        return 1
    print(f"Eval/app sync: {len(MIRRORS)} mirrors up to date.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
