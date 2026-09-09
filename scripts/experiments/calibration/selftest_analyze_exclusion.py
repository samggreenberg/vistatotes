"""Planted-answer self-test for ``analyze_exclusion.py`` (#3312).

The failure modes that matter for this study are not crashes - they are
confident sentences over the wrong evidence:

* two arms whose floors agree above some remainder ARE the same estimator
  there, so the **trap check** must recover ``identical == 1.0``; anything less
  means an arm ran under the wrong environment and the report is worthless;
* the **floor regime** must be reconstructed from ``n_remainder`` alone and must
  land on the planted fractions, since that is what makes a stage-B difference
  attributable to the floor rather than to the arm;
* a difference that is **resolved AND negligible** must be reported as exactly
  that - "real but not worth a decision" - and never as "no effect", which
  would throw away the evidence that the shipped arm is doing measurable work.
  This is the study's most likely production-scale outcome, so it is the one
  sentence that has to be right;
* a difference the grid **cannot resolve** must quote a bound, not a winner;
* a planted harm in one band must trip ``harms_a_band`` even when the pooled
  number looks fine, because an arm can win overall while being worse everywhere
  a short session lives;
* a mislabelled arm directory must be **refused**, not analysed.

Run: ``python selftest_analyze_exclusion.py``
"""

from __future__ import annotations

import math
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze_exclusion  # noqa: E402

FLOORS = {"off": math.inf, "always": 0.0, "app": 60.0, "f250": 250.0}
#: The remainder below which excluding is planted as harmful. Equal to the
#: `app` floor, so only the floorless arm is still excluding there.
DRAINED = 60
GEOMS = (("siglip", "whole_image"), ("siglip+dinov3_patch", "max_patch"))
CATS = ("cat_a", "cat_b")
SEEDS = (0, 1, 2, 3)


def _cell(stage: str, arm: str, emb: str, style: str, cat: str, seed: int, n_hay: int, steps: int, rng):
    """One cell, with the effect planted as a function of whether the arm excluded."""
    t = np.arange(4, steps + 1)
    rem = n_hay - t
    live = (rem > 0) & (rem >= FLOORS[arm])
    # Planted structure, all keyed off `live` so the analyzer can only recover
    # it by reconstructing the floor correctly:
    #   * a small, real benefit from excluding (-0.004, well inside the bound)
    #   * a LARGE harm from excluding once the remainder is drained
    #
    # `DRAINED` must be a remainder the fixture actually reaches - stage B runs
    # 380 steps on a 420-media haystack, so the remainder bottoms out at 40 and
    # a threshold of 30 would plant nothing at all.  It is set at the shipped
    # floor precisely so that only `always` is still excluding there, which is
    # the regime the floor exists to protect.  The size is deliberately far
    # above HARM_TOLERANCE after dilution across the band: a planted answer that
    # lands near the gate tests the noise, not the gate.
    cost = 0.30 + rng.normal(0, 0.004, t.size) - 0.004 * live + 0.30 * (live & (rem < DRAINED))
    return pd.DataFrame(
        {
            "seed": seed,
            "dataset": "vg_scale_any",
            "category": cat,
            "strategy": "autopilot",
            "trainer": "app",
            "head": "linear_svm",
            "style": style,
            "prevalence_arm": "",
            "realized_prevalence": 0.071,
            "t": t,
            "n_good": t // 2,
            "n_bad": t - t // 2,
            "n_haystack": n_hay,
            "n_remainder": rem,
            "phase": "steady",
            "app_trained": 1,
            "startup_schedule": "",
            "acq_threshold": 0.5,
            "acq_pool_percentile": 0.5,
            "report_pool_percentile": 0.5,
            "pool_variant": "max",
            "gmm_variant": "",
            "schedule": "",
            # The trap check reads this: identical wherever the floors agree.
            "threshold": np.where(live, 0.61, 0.60),
            "threshold_provenance": "fold_anchored[2/2]",
            "cost": cost,
            "regret_honest": cost - 0.1,
            "average_precision": 0.7,
            "fpr": 0.1,
            "fnr": 0.2,
            "embedder": emb,
            "seed_mode": "text",
            "seed_query": cat,
            "seed_embedder": "siglip",
            "calibration_fraction": 0.3,
            "sim_fraction": 0.5 if stage == "A" else 0.10,
            "exclusion_arm": "app(f60)" if arm == "app" else arm,
            "exclusion_min_remainder": FLOORS[arm],
        }
    )


def build(base: Path) -> None:
    rng = np.random.default_rng(11)
    spec = {"A": (("off", "app"), 2100, 60), "B": (("off", "always", "app", "f250"), 420, 380)}
    for stage, (arms, n_hay, steps) in spec.items():
        for arm in arms:
            cells = base / f"stage{stage}" / arm / "results" / "cells"
            cells.mkdir(parents=True, exist_ok=True)
            idx = 0
            for emb, style in GEOMS:
                for cat in CATS:
                    for seed in SEEDS:
                        _cell(stage, arm, emb, style, cat, seed, n_hay, steps, rng).to_csv(
                            cells / f"task_{idx:04d}.csv", index=False
                        )
                        idx += 1


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="selftest-excl-"))
    failures: list[str] = []
    try:
        base = tmp / "base"
        build(base)
        out = tmp / "out"
        rc = analyze_exclusion.main(
            ["--base", str(base), "--out", str(out), "--stages", "AB", "--no-figures", "--no-viewer"]
        )
        if rc != 0:
            failures.append(f"analyzer returned {rc}")

        trap = pd.read_csv(out / "agg" / "trap_check.csv")
        regime = pd.read_csv(out / "agg" / "floor_regime.csv")
        vd = pd.read_csv(out / "agg" / "verdict.csv")

        # --- the plumbing check must actually check something, and must pass ---
        checked = trap[trap["checked"] > 0]
        if checked.empty:
            failures.append("trap_check compared no steps at all - it would never catch a bad arm")
        if not checked.empty and float(checked["identical"].min()) < 1.0:
            failures.append("trap_check: arms that share a floor above the line must be IDENTICAL there")
        if not (trap["arm"] == "off").any():
            failures.append("trap_check dropped the `off` arm instead of reporting it as uncheckable")

        # --- the floor must be reconstructed from n_remainder alone -----------
        b = regime[regime["stage"] == "B"].set_index("arm")
        if not np.isclose(b.loc["always", "frac_excluding"], 1.0):
            failures.append("floor_regime: `always` must exclude on every step")
        if not np.isclose(b.loc["off", "frac_excluding"], 0.0):
            failures.append("floor_regime: `off` must never exclude")
        if not (0.90 < b.loc["app", "frac_excluding"] < 1.0):
            failures.append("floor_regime: `app` should stop excluding only in the drained tail")
        if not (0.3 < b.loc["f250", "frac_excluding"] < 0.6):
            failures.append("floor_regime: `f250` should stop excluding around the midpoint")
        if not (b.loc["f250", "frac_excluding"] < b.loc["app", "frac_excluding"]):
            failures.append("floor_regime: a HIGHER floor must switch off EARLIER")

        cost = vd[vd["metric"] == "cost"]

        # --- resolved AND negligible must be reported as exactly that ---------
        # `off` never excludes, so it loses the planted -0.004 benefit: a real
        # effect, comfortably inside the +/-0.01 bound.  The sentence for this
        # case is the one the study's headline depends on.
        a_off = cost[(cost["stage"] == "A") & (cost["arm"] == "off")]
        if a_off.empty:
            failures.append("no stage-A verdict for the `off` arm")
        else:
            for _, r in a_off.iterrows():
                if not bool(r["resolved"]):
                    failures.append("stage A `off`: a planted 0.004 effect should resolve at this n")
                if not bool(r["negligible"]):
                    failures.append("stage A `off`: a 0.004 effect is inside the bound and must read negligible")
                sentence = analyze_exclusion._verdict_sentence(r)
                if "real but negligible" not in sentence:
                    failures.append(f"stage A `off`: wrong sentence for resolved+negligible: {sentence[:80]}")
                if "no effect" in sentence:
                    failures.append("a resolved difference must never be described as 'no effect'")

        # --- a planted harm must trip the pointwise gate ----------------------
        # `always` keeps excluding into the drained tail, where the plant adds
        # +0.08.  Pooled over bands that is diluted; the pointwise gate is what
        # is supposed to see it.
        b_always = cost[(cost["stage"] == "B") & (cost["arm"] == "always")]
        if b_always.empty:
            failures.append("no stage-B verdict for the `always` arm")
        elif not bool(b_always["harms_a_band"].any()):
            failures.append("`always` harms the drained band by +0.08 and the pointwise gate missed it")
        elif bool(b_always["candidate"].any()):
            failures.append("an arm that harms a band must never be a shipping candidate")

        # --- an unresolvable difference must quote a bound, not a winner ------
        flat = cost.iloc[0].copy()
        flat["pooled_delta"], flat["pooled_se"] = 0.0005, 0.004
        flat["ci_lo"], flat["ci_hi"] = -0.0075, 0.0085
        flat["resolved"], flat["negligible"] = False, True
        s = analyze_exclusion._verdict_sentence(flat)
        if "not resolvable" not in s:
            failures.append(f"an unresolved difference must say so: {s[:80]}")

        # --- a mislabelled arm must be refused, not analysed ------------------
        bad = tmp / "bad"
        shutil.copytree(base, bad)
        f = next((bad / "stageB" / "always" / "results" / "cells").glob("task_0000.csv"))
        df = pd.read_csv(f)
        df["exclusion_arm"] = "f250"
        df.to_csv(f, index=False)
        try:
            analyze_exclusion.main(
                ["--base", str(bad), "--out", str(tmp / "out2"), "--stages", "B", "--no-figures", "--no-viewer"]
            )
        except SystemExit:
            pass  # what it is supposed to do
        else:
            failures.append("a cell stamped with the wrong arm was analysed instead of refused")

        # --- cells with no #3312 columns must be refused too ------------------
        old = tmp / "old"
        shutil.copytree(base, old)
        for csv in (old / "stageA").rglob("task_*.csv"):
            d = pd.read_csv(csv).drop(columns=["n_haystack", "n_remainder"])
            d.to_csv(csv, index=False)
        try:
            analyze_exclusion.main(
                ["--base", str(old), "--out", str(tmp / "out3"), "--stages", "A", "--no-figures", "--no-viewer"]
            )
        except SystemExit:
            pass
        else:
            failures.append("cells predating #3312 were analysed instead of refused")

        if failures:
            for x in failures:
                print(f"FAIL: {x}")
            return 1
        print(
            "selftest_analyze_exclusion: OK (trap check, floor regime, resolved-vs-negligible, "
            "harm gate, bound wording, mislabel and stale-column guards)"
        )
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
