"""Self-test for :mod:`analyze_incl_3196` on fabricated cells (no cluster data).

#3196 asks whether a *flat band* opened in the Inclusion knob when the head
changed, and every quantity that answers it - a dead-step rate, a paired
head difference, an offset collapse - is the kind a sign error or a lost pairing
key moves without ever crashing.  So each one gets a planted answer.

What is planted, per head arm:

* ``linear`` (the reference head): the incumbent ``mid_tilt`` moves the admitted
  set at **every** stop of the knob, in both environments.  Dead-step rate 0.
* ``svm`` (the shipped head): ``mid_tilt`` is frozen over a band around
  ``k = 0`` and live outside it - the exact failure the issue describes.  The
  band is **narrow** in the binary environment (H2 must not fire) and **wide**
  in the region one (H2 must fire), so a verdict that collapsed the two
  environments into one number would be caught.

And per rule, in both arms:

* ``mid`` - inclusion-blind: one admitted set for the whole knob.  The
  instrument check must call it inert, and must not be diluted by ``mid_tilt``,
  whose arm name also contains ``_mid_``.
* ``rate`` - planted the way the algebra says it is: the same quantile path as
  ``mid_tilt`` shifted by a **constant**, realized against a coarser admitted
  set.  That is the real shape (the run measured identical `quantile_span` to
  float32 beside dead-step rates 0.08 apart), and it is what separates the
  invariant from the thing that merely looks like it: the check must pass on the
  *quantile* span and must **not** gate on the admitted one.
* ``q_tilt`` at two step sizes - both live everywhere, but only ``s0.02`` is
  regret-tied to the incumbent; ``s0.08`` is planted materially harmful at
  ``k > 0``.  H3 must ship the first and reject the second, or the analyzer
  would buy the knob back with accuracy it did not price.

The acquisition offset falls out of the same plant: inside a flat band the
selector's cut at ``k + offset`` admits exactly what reporting's cut at ``k``
admits, so the collapse rate must be high on ``svm`` and ~0 on ``linear``.

Usage::

    python selftest_analyze_incl_3196.py     # exits non-zero on failure
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

KS = list(range(-10, 11))
KAPPA = 0.3
COMBINE = "qmean"
QTILT_STEPS = (0.02, 0.08)

#: (dataset, embedder, style, flat half-width under the SVM head).  The region
#: environment is the one where the band is wide enough for H2 to fire.
ENVS = [
    ("vg_scale", "siglip", "whole_image", 2),
    ("vg_scale", "siglip+dinov3_patch", "max_patch", 7),
]
CATEGORIES = ["bus@small", "bus@medium", "bus@large"]
SEEDS = [0, 1]
N_TEST = 800
#: Every step is past ``DEEP_VOTES_MIN``, so the deep filter keeps the whole
#: fixture and a failure is never "the analyzer looked at no rows".
STEPS = [110, 140, 170, 200]


def _arm(rule: str, step: float | None = None) -> str:
    base = f"fold_anchored_w{KAPPA:g}_{rule}_{COMBINE}"
    return f"{base}_s{step:g}" if step is not None else base


def _regret(rule: str, step: float | None, k: int) -> float:
    """Planted regret on the **rate scale** (the frame stores it x ``2**|k|``)."""
    if rule in ("mid_tilt", "rate"):
        return 0.10
    if rule == "mid":
        return 0.10 + 0.02 * abs(k)
    if rule == "q_tilt":
        if step == 0.02:
            return 0.10  # exactly tied with the incumbent
        return 0.10 if k <= 0 else 0.10 + 0.03  # harmful above 0
    raise AssertionError(rule)


def _admitted(rule: str, k: int, head: str, flat_half: int) -> int:
    """Planted admitted count: where the knob is alive, and for which head."""
    if rule == "mid":
        return 200  # constant in k: the inclusion-blind null
    if rule in ("mid_tilt", "rate"):
        if head == "linear":
            n = 200 - 8 * k  # live at every stop
        else:
            # The SVM plant: frozen inside the band, live outside it.
            outside = max(abs(k) - flat_half, 0)
            n = 200 - 8 * (1 if k > 0 else -1) * outside
        # `rate` sits a constant quantile below `mid_tilt`, so it travels the
        # same distance in quantile space and realizes it against a different
        # part of the score distribution - here, a coarser one.
        return int(round(n / 20) * 20) if rule == "rate" else n
    return 200 - 8 * k  # q_tilt: live everywhere under both heads


def _fabricate(results: Path, head: str, rng: np.random.Generator) -> None:
    cells = results / "cells"
    cells.mkdir(parents=True, exist_ok=True)
    info: dict = {"datasets": {}}
    idx = 0
    for ds, emb, style, flat_half in ENVS:
        info["datasets"].setdefault(ds, {})[emb] = {"selected_categories": CATEGORIES}
        for cat in CATEGORIES:
            for seed in SEEDS:
                rows = []
                for t in STEPS:
                    # One offset per (step, k), SHARED by every arm - which is
                    # how the real frame behaves (every arm re-cuts the same
                    # per-step fit against the same test scores) and is the
                    # property the pairing exists to exploit.
                    shared = {k: float(rng.normal(0, 0.03)) for k in KS}
                    specs: list[tuple[str, float | None]] = [
                        ("mid", None),
                        ("mid_tilt", None),
                        ("rate", None),
                        *[("q_tilt", s) for s in QTILT_STEPS],
                    ]
                    for rule, step in specs:
                        for k in KS:
                            n_adm = _admitted(rule, k, head, flat_half)
                            regret = _regret(rule, step, k) + shared[k] + float(rng.normal(0, 0.002))
                            scale = 2.0 ** abs(k)
                            rows.append(
                                {
                                    "seed": seed,
                                    "dataset": ds,
                                    "category": cat,
                                    "strategy": "autopilot",
                                    "trainer": "app",
                                    "head": "linear_svm" if head == "svm" else "linear",
                                    "style": style,
                                    "prevalence_arm": "native",
                                    "realized_prevalence": 0.025,
                                    "t": t,
                                    "n_good": t // 3,
                                    "n_bad": t - t // 3,
                                    "phase": "done",
                                    "app_trained": 1,
                                    "embedder": emb,
                                    "arm": _arm(rule, step),
                                    "cut_rule": rule,
                                    "anchor_weight": KAPPA,
                                    "combine": COMBINE,
                                    "qtilt_step": step if step is not None else float("nan"),
                                    "inclusion_k": k,
                                    # The quantile always moves, even where the
                                    # realized set cannot: that gap is the
                                    # empty-band signal the report reads.
                                    # `rate` is `mid_tilt` shifted by a constant
                                    # (#2865, exactly), so the SPAN is identical
                                    # and the position is not.
                                    "fold_quantile": 0.5 - 0.02 * k - (0.07 if rule == "rate" else 0.0),
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
                pd.DataFrame([]).to_csv(cells / f"task_{idx:04d}.csv", index=False)
                idx += 1
    (results / "prepare_info.json").write_text(json.dumps(info, indent=2))


def main() -> int:  # noqa: C901 - a linear list of planted-answer assertions
    rng = np.random.default_rng(3196)
    with tempfile.TemporaryDirectory() as tmp:
        svm = Path(tmp) / "svm" / "results"
        linear = Path(tmp) / "linear" / "results"
        _fabricate(svm, "svm", rng)
        _fabricate(linear, "linear", rng)
        out = Path(tmp) / "analysis"

        os.environ["CALIB_EXP"] = tmp
        os.environ["CALIB_RESULTS"] = str(svm)
        os.environ["CALIB_DATASETS"] = "vg_scale"
        os.environ["CALIB_VGSCALE_EMBEDDERS"] = "siglip,siglip+dinov3_patch"
        os.environ["CALIB_CUT_INCL_KS"] = ",".join(str(k) for k in KS)
        os.environ["CALIB_ANCHORED_WEIGHTS"] = str(KAPPA)
        os.environ["CALIB_ANCHORED_RULES"] = "mid,mid_tilt,rate,q_tilt"
        # `expected_cells` enumerates the grid from the run's own config, so the
        # seed count has to match the fixture or every cell reads as missing -
        # which is the check that catches a real run losing cells to a node.
        os.environ["CALIB_N_SEEDS"] = str(len(SEEDS))
        sys.path.insert(0, str(Path(__file__).parent))

        import analyze_incl_3196 as A  # noqa: PLC0415

        rc = A.main(["--svm", str(svm.parent / "results"), "--linear", str(linear), "--out", str(out)])
        assert rc == 0, rc

        v = json.loads((out / "incl3196_summary.json").read_text())
        incumbent = v["incumbent"]
        assert incumbent == _arm("mid_tilt"), incumbent

        # --- instrument checks: both are algebra, not findings ---------------
        inst = v["instrument"]
        assert inst["mid_is_inert"] is True, inst
        assert inst["mid_arm"] == _arm("mid"), inst
        assert inst["mid_tilt_tracks_rate"] is True, inst
        assert inst["mid_tilt_vs_rate_max_quantile_span_gap"] < 1e-3, inst
        # ...while the realized sets differ, which the check must report and
        # must not fail on.  Gating this would have called the real run's
        # instrument broken over a fact about the haystack's local density.
        assert inst["mid_tilt_vs_rate_max_dead_gap"] > 0.02, inst
        assert inst["ok"] is True, inst

        env_tbl = pd.read_csv(out / "incl3196_per_env.csv")
        binary_env = "vg_scale/siglip/whole_image"
        region_env = "vg_scale/siglip+dinov3_patch/max_patch"
        inc = env_tbl[env_tbl["arm"] == incumbent].set_index(["head_arm", "env"])
        # The plant, recovered: dead under the SVM head, live under the logistic
        # one, and *more* dead where the band is wider.
        assert inc.loc[("linear", region_env), "dead_step_rate"] < 0.01
        assert inc.loc[("linear", binary_env), "dead_step_rate"] < 0.01
        assert inc.loc[("svm", region_env), "dead_step_rate"] > inc.loc[("svm", binary_env), "dead_step_rate"]

        # --- H1: the head moved the knob -------------------------------------
        assert v["H1"]["supported"] is True, v["H1"]
        assert set(v["H1"]["envs_softer_under_svm"]) == {binary_env, region_env}, v["H1"]

        # --- H2: soft in ABSOLUTE terms, and only where planted so ------------
        assert v["H2"]["fires"] is True, v["H2"]
        assert v["H2"]["soft_envs"] == [region_env], v["H2"]

        # --- H3: ship the tied q_tilt, reject the harmful one -----------------
        assert v["H3"]["ships"] == [_arm("q_tilt", 0.02)], v["H3"]
        assert v["H3"]["recommended"] == _arm("q_tilt", 0.02), v["H3"]
        assert _arm("q_tilt", 0.08) in v["H3"]["helps_where_soft"], v["H3"]
        assert _arm("q_tilt", 0.08) not in v["H3"]["regret_clean_arms"], v["H3"]

        # --- H4: the offset collapses exactly where the band is ---------------
        gaps = pd.read_csv(out / "incl3196_offset_gap.csv")
        # The live app constant, not a copy: the analyzer reads
        # `ACQUISITION_INCLUSION_OFFSET` from the app, so a number pinned here
        # asserts what the offset used to be.  This was `-3` and the app moved
        # to `-4`, which failed the selftest without anything being wrong with
        # the analyzer -- the shape the "eval default arm IS the app" rule
        # exists to prevent, one tier down.
        from vtscore.training.thresholds import ACQUISITION_INCLUSION_OFFSET  # noqa: PLC0415

        assert set(gaps["offset"]) == {ACQUISITION_INCLUSION_OFFSET}, gaps["offset"].unique()
        by_head = gaps.groupby(["head_arm", "env"])["collapse_rate"].mean()
        assert by_head[("linear", region_env)] < 0.01, by_head
        assert by_head[("svm", region_env)] > 0.5, by_head
        assert by_head[("svm", region_env)] > by_head[("svm", binary_env)], by_head

        # --- the band breakdown exists and names the bands --------------------
        bands = pd.read_csv(out / "incl3196_by_band.csv")
        assert set(bands["band"]) == {"small", "medium", "large"}, bands["band"].unique()

        # --- every cell is accounted for, in both arms ------------------------
        for row in v["cells"]:
            assert row["n_missing"] == 0, row
            assert row["n_unreadable"] == 0 and row["n_empty"] == 0, row

        assert (out / "REPORT_incl3196.md").exists()

    print("selftest_analyze_incl_3196: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
