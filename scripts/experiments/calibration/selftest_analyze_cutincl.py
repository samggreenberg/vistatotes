"""Self-test for :mod:`analyze_cutincl` on fabricated cells (no cluster data).

The #2865 sweep answers a two-part question, and both parts are the kind of
quantity a sign error or a bad pairing key moves without ever crashing.  So both
get a **planted answer** here.

What is planted, per arm:

* ``mid`` - the inclusion-blind null.  One threshold for the whole knob, so
  exactly one admitted set: ``knob_yield`` must come back at ``1/len(KS)`` and
  ``dead_step_rate`` at 1.0.  Its regret is planted flat in ``k`` and *worse*
  than the incumbent's away from 0, which is what an inclusion-blind rule costs.
* ``mid_tilt`` - the incumbent.  Live but coarse: it moves the admitted set on
  only some steps of the knob, which is the realistic middle case and the one
  the liveness table has to be able to express (a pass/fail flag could not).
* ``q_tilt`` - live everywhere by construction (every ``k`` its own admitted
  set) and planted with regret **tied** to the incumbent.  This is the arm the
  verdict must recommend: it buys the whole knob for nothing.
* ``rate`` - live everywhere *and* lower regret than everything else at
  ``k <= 0``, but materially **worse** at every ``k > 0``.  This is the trap: an
  arm that wins on pooled regret while being harmful somewhere.  The verdict
  must reject it, or the analyzer would ship a rule that is better on average
  and worse where users actually sit.

The environments differ in separability - one haystack has a wide empty band
between its modes - so the "haystack-imposed ceiling" table has a known ordering
too: no rule can deliver the full knob on the separated environment.

Usage::

    python selftest_analyze_cutincl.py     # exits non-zero on failure
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

KS = [-4, -2, -1, 0, 1, 2, 4]
KAPPA = 0.3
COMBINE = "qmean"

#: (dataset, embedder, style, admitted-granularity).  The second environment is
#: the "cleanly separated haystack" case: its admitted set only changes in big
#: jumps, so even a live rule loses stops off the knob there.
ENVS = [
    ("visual_genome_m", "dinov3_patch", "max_patch", 1),
    ("coco_val", "siglip2", "whole_image", 400),
]
CATEGORIES = ["cat_a", "cat_b", "cat_c"]
SEEDS = [0, 1]
N_TEST = 800
STEPS = [120, 160, 200, 240, 300]


def _arm(rule: str) -> str:
    return f"fold_anchored_w{KAPPA:g}_{rule}_{COMBINE}"


def _regret(rule: str, k: int) -> float:
    """Planted regret curve per rule, on the **rate scale**.

    The fixture writes ``cut_regret`` in the harness's own cost units, i.e. this
    value times ``2**abs(k)`` (see :func:`_cost_scale`), because
    ``inclusion_cost_weights`` doubles one of the two cost weights per step of
    the knob.  Every assertion below is on the analyzer's rate-scale column, so
    the scaling is itself a planted answer: an analyzer that pooled or gated on
    raw cost units would see the ``k=+-4`` rows at sixteen times their weight
    and both the ``rate`` trap and the ``q_tilt`` tie would come out wrong.
    """
    if rule == "mid_tilt":
        return 0.10
    if rule == "q_tilt":
        return 0.10  # exactly tied with the incumbent
    if rule == "mid":
        # Inclusion-blind: fine at 0, progressively wrong away from it.
        return 0.10 + 0.02 * abs(k)
    if rule == "rate":
        # The pooled-average trap: a big win below 0 and a real loss above it,
        # sized so the *mean over the swept k* comes out ahead of every other
        # arm (-0.01) while the loss at k>0 (+0.03) is three times the harm
        # tolerance.  An analyzer that ranked on pooled regret would ship this.
        return 0.10 - 0.04 if k <= 0 else 0.10 + 0.03
    raise AssertionError(rule)


def _cost_scale(k: int) -> float:
    """The larger of the two inclusion cost weights at *k*: ``2**abs(k)``."""
    return 2.0 ** abs(k)


def _admitted(rule: str, k: int, granularity: int) -> int:
    """Planted admitted count.  ``granularity`` coarsens the realized set."""
    if rule == "mid":
        raw = 200  # constant in k: the bug
    elif rule == "mid_tilt":
        # Live but *coarse*: a small per-step displacement, so several adjacent
        # slider stops land on the same admitted set even where the haystack is
        # finely resolved.  This is the realistic middle case, and the one a
        # pass/fail "does the knob move?" flag could not express.
        raw = round((200 - 6 * k) / 10) * 10
    else:
        raw = 200 - 40 * k  # q_tilt / rate: live everywhere
    return int(round(raw / granularity) * granularity)


def _fabricate(results: Path, rng: np.random.Generator) -> None:
    cells = results / "cells"
    cells.mkdir(parents=True, exist_ok=True)
    info: dict = {"datasets": {}}
    idx = 0
    for ds, emb, style, gran in ENVS:
        info["datasets"].setdefault(ds, {})[emb] = {"selected_categories": CATEGORIES}
        for cat in CATEGORIES:
            for seed in SEEDS:
                rows = []
                for t in STEPS:
                    # Step-level offset, drawn once and **shared by every arm**
                    # at that (step, k).  That is how the real frame behaves -
                    # every arm re-cuts the same per-step fit against the same
                    # test scores - and it is the property the analyzer's
                    # pairing exists to exploit: the unpaired means are noisy
                    # (sd 0.03, an order above the planted effects), the paired
                    # differences are not.  An analyzer that lost the pairing
                    # would drown here rather than quietly shift.
                    shared = {k: float(rng.normal(0, 0.03)) for k in KS}
                    for rule in ("mid", "mid_tilt", "rate", "q_tilt"):
                        for k in KS:
                            n_adm = _admitted(rule, k, gran)
                            # Small per-arm jitter so nothing is exactly
                            # degenerate, but far below the planted effects.
                            regret = _regret(rule, k) + shared[k] + float(rng.normal(0, 0.002))
                            # ...written out in the units the harness actually
                            # scores in, which is the rate scale times 2**|k|.
                            scale = _cost_scale(k)
                            rows.append(
                                {
                                    "seed": seed,
                                    "dataset": ds,
                                    "category": cat,
                                    "strategy": "autopilot",
                                    "trainer": "app",
                                    "head": "linear",
                                    "style": style,
                                    "prevalence_arm": "native",
                                    "realized_prevalence": 0.1,
                                    "t": t,
                                    "n_good": t // 3,
                                    "n_bad": t - t // 3,
                                    "phase": "done",
                                    "app_trained": 1,
                                    "embedder": emb,
                                    "arm": _arm(rule),
                                    "cut_rule": rule,
                                    "anchor_weight": KAPPA,
                                    "combine": COMBINE,
                                    "qtilt_step": 0.02 if rule == "q_tilt" else float("nan"),
                                    "inclusion_k": k,
                                    # The quantile always moves, even where the
                                    # realized set cannot: that gap is the
                                    # empty-band signal the report reads.
                                    "fold_quantile": 0.5 - 0.03 * k,
                                    "cut_threshold": 0.6 - 0.01 * k,
                                    "cut_cost": (0.2 + regret) * scale,
                                    "cut_fpr": 0.05,
                                    "cut_fnr": 0.12,
                                    "k_oracle_threshold": 0.55,
                                    "k_oracle_cost": 0.2 * scale,
                                    "cut_regret": regret * scale,
                                    "admitted_frac": n_adm / N_TEST,
                                    "n_admitted": n_adm,
                                    "n_test": N_TEST,
                                }
                            )
                pd.DataFrame(rows).to_csv(cells / f"task_{idx:04d}__cutincl.csv", index=False)
                # An empty main frame beside it, as a real run writes.
                pd.DataFrame([]).to_csv(cells / f"task_{idx:04d}.csv", index=False)
                idx += 1
    (results / "prepare_info.json").write_text(json.dumps(info, indent=2))


def main() -> int:  # noqa: C901 - a linear list of planted-answer assertions
    rng = np.random.default_rng(11)
    with tempfile.TemporaryDirectory() as tmp:
        results = Path(tmp) / "results"
        _fabricate(results, rng)

        os.environ["CALIB_EXP"] = tmp
        os.environ["CALIB_RESULTS"] = str(results)
        os.environ["CALIB_CUT_INCL_KS"] = ",".join(str(k) for k in KS)
        os.environ["CALIB_ANCHORED_WEIGHTS"] = str(KAPPA)
        os.environ["CALIB_ANCHORED_RULES"] = "mid,mid_tilt,rate,q_tilt"
        sys.path.insert(0, str(Path(__file__).parent))

        import analyze_cutincl  # noqa: PLC0415

        rc = analyze_cutincl.main()
        assert rc == 0, f"analyze_cutincl returned {rc}"

        # --- the side frames must not have leaked into a main-frame read ----
        from _cells_io import main_frame_files, side_frame_files  # noqa: PLC0415

        assert side_frame_files(results / "cells", "__cutincl"), "fixture wrote no side frames"
        assert not any("__cutincl" in p.name for p in main_frame_files(results / "cells")), (
            "the __cutincl side frame leaked into the main-frame glob"
        )

        # --- (a) paired regret: sign and shape, per k -----------------------
        reg = pd.read_csv(results / "agg" / "cutincl_regret_vs_incumbent.csv")
        assert set(reg["arm"]) == {_arm(r) for r in ("mid", "rate", "q_tilt")}, set(reg["arm"])

        # q_tilt is planted exactly tied with the incumbent.  The assertion is
        # on the *magnitude* rather than on 14 CIs all covering zero: with the
        # pairing working, the residual is ~1e-4 against planted effects of
        # 2e-2, and demanding 14 independent 95 % intervals all cover would fail
        # ~half the time on a correct analyzer.  That multiplicity is the same
        # reason the analyzer gates on HARM_TOLERANCE rather than on ci_lo > 0.
        q = reg[reg["arm"] == _arm("q_tilt")]
        assert len(q) == len(KS) * len(ENVS), q.to_dict("records")
        assert q["d_regret"].abs().max() < 0.005, q.to_dict("records")
        assert (q["ci_lo"] <= analyze_cutincl.HARM_TOLERANCE).all(), q.to_dict("records")

        # mid is planted worse away from 0 and tied at 0: a V, not a line.
        m = reg[reg["arm"] == _arm("mid")].groupby("inclusion_k")["d_regret"].mean()
        assert abs(m.loc[0]) < 0.02, m.to_dict()
        assert m.loc[4] > 0.05 and m.loc[-4] > 0.05, m.to_dict()
        assert m.loc[4] > m.loc[1] > m.loc[0], m.to_dict()

        # rate is planted better below 0 and significantly worse above it.
        r = reg[reg["arm"] == _arm("rate")].groupby("inclusion_k")["d_regret"].mean()
        assert r.loc[-4] < -0.01 and r.loc[2] > 0.01, r.to_dict()
        harmed = reg[(reg["arm"] == _arm("rate")) & (reg["ci_lo"] > analyze_cutincl.HARM_TOLERANCE)]
        assert not harmed.empty, "the planted +0.03 harm at k>0 was not detected as material"
        assert set(harmed["inclusion_k"]) == {k for k in KS if k > 0}, harmed.to_dict("records")

        # ...and the reported difference IS the rate-scale one: the same rows in
        # raw cost units are 2**|k| times bigger, which is the whole reason the
        # analyzer rescales before pooling or gating.
        r4 = reg[(reg["arm"] == _arm("rate")) & (reg["inclusion_k"] == 4)]
        ratio = (r4["d_regret_cost"] / r4["d_regret"]).abs()
        assert ((ratio - 16.0).abs() < 0.5).all(), r4.to_dict("records")

        # --- (b) knob liveness ---------------------------------------------
        live = pd.read_csv(results / "agg" / "cutincl_liveness.csv").set_index(["arm", "env"])
        for ds, emb, style, _gran in ENVS:
            env = f"{ds}/{emb}/{style}"
            mid = live.loc[(_arm("mid"), env)]
            assert mid["distinct_admitted"] == 1.0, mid.to_dict()
            assert abs(mid["knob_yield"] - 1.0 / len(KS)) < 1e-9, mid.to_dict()
            assert mid["dead_step_rate"] == 1.0, mid.to_dict()
            assert mid["inert_rate"] == 1.0, mid.to_dict()
            assert mid["admitted_span"] == 0.0, mid.to_dict()
            # ...while its *quantile* moved the whole time.  That gap is what
            # separates "the rule is blind" from "the haystack is flat".
            assert mid["quantile_span"] > 0.2, mid.to_dict()

        fine_env = f"{ENVS[0][0]}/{ENVS[0][1]}/{ENVS[0][2]}"
        coarse_env = f"{ENVS[1][0]}/{ENVS[1][1]}/{ENVS[1][2]}"
        # On the well-resolved haystack every stop is distinct for a live rule.
        assert abs(live.loc[(_arm("q_tilt"), fine_env), "knob_yield"] - 1.0) < 1e-9
        # On the separated one it cannot be, whatever the rule does.
        assert live.loc[(_arm("q_tilt"), coarse_env), "knob_yield"] < 1.0
        # The incumbent is live but coarser than q_tilt on the fine haystack.
        assert live.loc[(_arm("mid_tilt"), fine_env), "knob_yield"] < live.loc[(_arm("q_tilt"), fine_env), "knob_yield"]
        assert live.loc[(_arm("mid_tilt"), fine_env), "knob_yield"] > 1.0 / len(KS)

        # The ceiling table must rank the separated environment as the worse one.
        flat = pd.read_csv(results / "agg" / "cutincl_env_flatness.csv")
        assert list(flat["env"]) == [coarse_env, fine_env], flat.to_dict("records")

        # --- the verdict ----------------------------------------------------
        v = json.loads((results / "cutincl_summary.json").read_text())
        assert v["decidable"] is True, v
        assert v["incumbent"] == _arm("mid_tilt"), v
        # q_tilt buys the whole knob at no regret cost -> it is the pick.
        assert v["recommended"] == _arm("q_tilt"), v
        by_arm = {a["arm"]: a for a in v["arms"]}
        # ...and `rate`, despite the best pooled regret, must be rejected for
        # being significantly harmful somewhere on the knob.
        assert by_arm[_arm("rate")]["d_regret_pooled"] < by_arm[_arm("q_tilt")]["d_regret_pooled"]
        assert by_arm[_arm("rate")]["ships"] is False, by_arm[_arm("rate")]
        assert by_arm[_arm("rate")]["no_regret_harm"] is False, by_arm[_arm("rate")]
        # ...and `mid`, the null, must fail on the knob rather than on regret.
        assert by_arm[_arm("mid")]["beats_incumbent_knob"] is False, by_arm[_arm("mid")]

        assert (results / "REPORT_cutincl.md").exists()

    print("selftest_analyze_cutincl: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
