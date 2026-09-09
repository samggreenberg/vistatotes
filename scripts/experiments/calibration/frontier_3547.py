"""#3547: does the acquisition offset's optimum move DEEPER through a session?

Reads the raw per-cell trajectories rather than ``agg/trajectories.csv``,
because the shared aggregate is FINAL-ONLY (plus positives at t=50/100) and
every question here is about the same cell at two horizons.

**Why the horizons come off one wave.** ``max_steps`` reaches the simulation as
a loop bound and nothing inside the loop reads it, so a 400-step trajectory is a
strict EXTENSION of the 100-step one.  Confirmed empirically against #3319's two
independent waves before this study relied on it -- 6336 cells per arm,
identical at t=100 on cost, ``n_good``, thresholds and ``acq_pool_percentile``
(``check_prefix_3547.py``).  So t=100 and t=400 here are the SAME trajectory,
paired within the cell, rather than two runs compared through their summaries.

**Why a difference-in-differences and not an argmin.** #3319 established that
``final_cost`` is flat across three bits of this plateau, so an argmin over it
is noise -- it would report a "move" from sampling error alone.  The DiD asks
the question directly and is paired on both axes at once:

    DiD = [m(deep, 400) - m(shallow, 400)] - [m(deep, 100) - m(shallow, 100)]

Negative means the deep arm gains ground as the session runs on, i.e. the
optimum moves deeper.

Usage::

    python frontier_3547.py --base /expscratch/$USER/acq-3547
    python frontier_3547.py --base ... --markdown REPORT_3547_frontier.md
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pandas as pd

from _cells_io import load_arm

#: The horizons compared. t=100 is every prior environment behind this constant;
#: t=400 is the session this study exists to measure.
HORIZONS = (100, 400)

#: `analyze_spikes.py`'s deep-spike rule, restated here rather than imported so
#: this file states the guardrail it reports. Kept in sync by
#: `test_frontier_3547_spike_rule_matches_analyze_spikes`.
WARM_T, DEEP_COST, DEEP_EXCESS = 20, 0.25, 0.20

#: Cost-regression tolerance, as in every study behind this constant.
TOL = 0.010

ARM_K = {
    "prod": 0.0,
    "acq_m1": -1.0,
    "acq_m3": -3.0,
    "acq_m4": -4.0,
    "acq_m5": -5.0,
    "acq_m6": -6.0,
    "acq_p2": 2.0,
}
CONTROL = "prod"
PAIR_KEYS = ("category", "seed")

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--base", default="/expscratch/sgreenberg/acq-3547")
ap.add_argument("--embedder", default="siglip")
ap.add_argument("--style", default="whole_image")
ap.add_argument("--shallow", default="acq_m3", help="the DiD's shallow reference arm")
# `acq_m4` leads the list deliberately. The pile was sized so the SHIPPED arm's
# tail is interpretable, and it is (harvest ~35% against #3319's 85%) -- but
# `-5` and `-6` out-harvest what it was sized for and land above the 50% bar.
# So the -4/-3 contrast is the one DiD in this grid where NEITHER side is
# compressed; it is reported first and the deeper ones are MARKED beside it,
# because a compressed deep arm can only understate a deepening optimum, never
# manufacture one.
ap.add_argument("--deep", default="acq_m4,acq_m5,acq_m6", help="comma-separated DiD deep arms")
ap.add_argument("--markdown", default=None)
ap.add_argument("--boot", type=int, default=10000)
ap.add_argument("--csv", default=None, help="directory for the tidy CSVs the figures read")
args = ap.parse_args()

BASE = pathlib.Path(args.base)
rng = np.random.default_rng(0)
lines: list[str] = []


def out(s: str = "") -> None:
    print(s)
    lines.append(s)


def boot_ci(x: np.ndarray, n: int) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean of a paired difference."""
    if len(x) < 2:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, len(x), size=(n, len(x)))
    means = x[idx].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (arm, category, seed) x horizon, from a raw trajectory frame.

    `sim_pos` is the positives the cell's sim half actually held, recovered from
    the harness's own `n_haystack` x `realized_prevalence` rather than assumed
    from the pile's design -- the point of the study is that the two can differ.
    """
    rows = []
    for (cat, seed), g in df.groupby(list(PAIR_KEYS), dropna=False):
        g = g.sort_values("t")
        t = g["t"].to_numpy()
        cost = g["cost"].to_numpy(dtype=float)
        oracle = g["oracle_cost"].to_numpy(dtype=float)
        sim_pos = float(g["n_haystack"].iloc[0]) * float(g["realized_prevalence"].iloc[0])
        for H in HORIZONS:
            m = t <= H
            if not m.any() or t.max() < H:
                continue
            # The step nearest H from below is the horizon's endpoint; a cell
            # that skipped a step must not silently borrow a later one.
            end = np.where(m)[0][-1]
            warm = m & (t >= WARM_T)
            excess = cost - oracle
            deep = warm & (cost >= DEEP_COST) & (excess >= DEEP_EXCESS)
            rows.append(
                {
                    "category": cat,
                    "seed": seed,
                    "horizon": H,
                    "cost": cost[end],
                    "auc": float(np.trapezoid(cost[m], t[m]) / (t[m][-1] - t[m][0])) if m.sum() > 1 else np.nan,
                    "positives": float(g["n_good"].to_numpy()[end]),
                    "ap": float(g["average_precision"].to_numpy()[end]),
                    "sim_pos": sim_pos,
                    "harvest": float(g["n_good"].to_numpy()[end]) / sim_pos if sim_pos else np.nan,
                    "has_deep": bool(deep.any()),
                    # Carried so clicks-to-target can be computed against this
                    # cell's own control, which needs the whole curve.
                    "_t": t[m],
                    "_cost": cost[m],
                }
            )
    return pd.DataFrame(rows)


def clicks_to_target(row: pd.Series, target: float) -> float:
    """First click at which this cell's cost reaches *target*, else NaN.

    NaN is a real outcome ("never got there"), reported as a miss rate beside
    the median rather than filled in -- #3319's honesty note: `prod` reaches its
    own target in 100% of cells by construction while arms miss in 8-22%.
    """
    hit = np.where(row["_cost"] <= target)[0]
    return float(row["_t"][hit[0]]) if len(hit) else np.nan


# --- load ---------------------------------------------------------------------
per_arm: dict[str, pd.DataFrame] = {}
prov_lines: list[str] = []
for arm in ARM_K:
    d = BASE / "bin" / arm / "results"
    if not (d / "cells").is_dir():
        continue
    raw, prov = load_arm(d)
    if raw.empty:
        continue
    raw = raw[(raw["embedder"] == args.embedder) & (raw["style"] == args.style)]
    if raw.empty:
        continue
    per_arm[arm] = summarise(raw)
    prov_lines.append(
        f"| `{arm}` | {prov['n_files']} | {prov['n_read']} | {len(prov['zero_byte'])} | "
        f"{len(prov['unreadable'])} | {prov.get('header_only', prov.get('headless', 'n/a'))} |"
    )

if CONTROL not in per_arm:
    raise SystemExit(f"no {CONTROL} arm under {BASE}/bin -- nothing to pair against")

arms = [a for a in ARM_K if a in per_arm]

# Tidy rows for `--csv`, accumulated as each table is built rather than
# recomputed, so a figure and its table cannot disagree.
csv_harvest: list[dict] = []
csv_speed: list[dict] = []
csv_delta: list[dict] = []
csv_did: list[dict] = []
csv_spikes: list[dict] = []
out(f"# #3547 — does the optimum move deeper? — `{args.embedder} x {args.style}`\n")
out(f"Arms: {', '.join(f'`{a}` (k={ARM_K[a]:g})' for a in arms)}")
out(f"Horizons: {', '.join(str(h) for h in HORIZONS)} clicks, off ONE wave (see the module docstring)")
out(f"Cost-regression tolerance: ±{TOL:g}\n")

out("## Cells read\n")
out("| arm | files | read | zero-byte | unreadable | header-only |")
out("|---|---:|---:|---:|---:|---:|")
for ln in prov_lines:
    out(ln)
out()

# --- realised harvest: the number that says whether a tail is interpretable ---
# #3547's explicit ask, and a first-class column rather than a footnote: #3319's
# deep wave was read as a result for a quarter of a trajectory in which the
# aggressive arms had run out of positives.
out("## Realised harvest — is the tail interpretable?\n")
out("An arm that has consumed most of its positives is no longer being compared")
out("over the same opportunity as the control. Pre-registered bar: **a median")
out("above 50% means that arm's deep readings are reported as compressed.**\n")
out("| arm | k | sim positives | median positives @400 | median harvest | cells >50% | cells >80% |")
out("|---|---:|---:|---:|---:|---:|---:|")
harvest_flag: list[str] = []
for a in arms:
    s = per_arm[a]
    s4 = s[s["horizon"] == 400]
    if s4.empty:
        continue
    h = s4["harvest"].to_numpy(dtype=float)
    med = float(np.nanmedian(h))
    if med > 0.50:
        harvest_flag.append(a)
    csv_harvest.append(
        {
            "arm": a,
            "k": ARM_K[a],
            "sim_pos": float(np.nanmedian(s4["sim_pos"])),
            "positives_400": float(np.nanmedian(s4["positives"])),
            "harvest_med": med,
            "frac_over_50": float(np.nanmean(h > 0.50)),
            "frac_over_80": float(np.nanmean(h > 0.80)),
            "compressed": med > 0.50,
        }
    )
    out(
        f"| `{a}` | {ARM_K[a]:g} | {float(np.nanmedian(s4['sim_pos'])):.0f} | "
        f"{float(np.nanmedian(s4['positives'])):.0f} | {100 * med:.1f}% | "
        f"{100 * float(np.nanmean(h > 0.50)):.1f}% | {100 * float(np.nanmean(h > 0.80)):.1f}% |"
    )
out()
if harvest_flag:
    out(
        f"**COMPRESSED TAIL: {', '.join('`' + a + '`' for a in harvest_flag)} exceed the 50% bar.** "
        "Their deep readings are a lower bound on the offset's value, not a measurement of it.\n"
    )
else:
    out(
        "**No arm exceeds the 50% bar** — the deep tail is interpretable for every arm, "
        "which is what `vg_scale_deep` was built to buy and what #3319 could not say.\n"
    )


# --- paired helpers ------------------------------------------------------------
def paired(metric: str, a: str, b: str, H: int) -> dict:
    """Mean paired difference `a - b` on *metric* at horizon *H*, with a CI."""
    ka = per_arm[a]
    kb = per_arm[b]
    x = ka[ka["horizon"] == H].set_index(list(PAIR_KEYS))[metric]
    y = kb[kb["horizon"] == H].set_index(list(PAIR_KEYS))[metric]
    j = pd.concat([x.rename("a"), y.rename("b")], axis=1).dropna()
    if j.empty:
        return {"n": 0, "mean": np.nan, "lo": np.nan, "hi": np.nan}
    d = (j["a"] - j["b"]).to_numpy(dtype=float)
    lo, hi = boot_ci(d, args.boot)
    return {"n": len(d), "mean": float(d.mean()), "lo": lo, "hi": hi}


def fmt(p: dict, prec: int = 4) -> str:
    if not p["n"]:
        return "—"
    return f"{p['mean']:+.{prec}f} [{p['lo']:+.{prec}f}, {p['hi']:+.{prec}f}]"


# --- clicks-to-target ----------------------------------------------------------
# Per cell, against THAT CELL's own control final cost at the horizon in
# question -- #3319's construction, and the endpoint that separated the
# plateau's edges when `final_cost` could not.
ctl = {H: per_arm[CONTROL][per_arm[CONTROL]["horizon"] == H].set_index(list(PAIR_KEYS)) for H in HORIZONS}
for a in arms:
    s = per_arm[a]
    ctt, miss = [], []
    for _, r in s.iterrows():
        c = ctl[r["horizon"]]
        key = (r["category"], r["seed"])
        tgt = float(c.loc[key, "cost"]) if key in c.index else np.nan
        v = clicks_to_target(r, tgt) if np.isfinite(tgt) else np.nan
        ctt.append(v)
        miss.append(not np.isfinite(v))
    s["ctt"] = ctt
    s["ctt_miss"] = miss

out("## Speed — clicks to the answer the control ends its session with\n")
out("| arm | k | median CTT @100 | never reached | median CTT @400 | never reached |")
out("|---|---:|---:|---:|---:|---:|")
for a in arms:
    s = per_arm[a]
    cells = []
    for H in HORIZONS:
        sh = s[s["horizon"] == H]
        cells.append(f"{float(np.nanmedian(sh['ctt'])):.1f}" if len(sh) else "—")
        cells.append(f"{100 * float(np.mean(sh['ctt_miss'])):.0f}%" if len(sh) else "—")
        if len(sh):
            csv_speed.append(
                {
                    "arm": a,
                    "k": ARM_K[a],
                    "horizon": H,
                    "ctt_median": float(np.nanmedian(sh["ctt"])),
                    "miss_frac": float(np.mean(sh["ctt_miss"])),
                }
            )
    out(f"| `{a}` | {ARM_K[a]:g} | " + " | ".join(cells) + " |")
out()
out("`prod` reaches its own target in 100% of cells **by construction**; a miss")
out("rate elsewhere is a real outcome and is why the median is reported beside it.\n")

# --- H3: does the t=100 frontier replicate #3319's shape? ---------------------
out("## H3 — does the plateau replicate at t=100?\n")
out("The anchor connecting this pile to the shipped constant. If it fails, the")
out("dataset change did more than add depth and **every** deep reading below is")
out("about the dataset rather than the horizon.\n")
out("| arm | k | Δ cost vs `prod` @100 [95% CI] | Δ cost @400 [95% CI] | Δ positives @400 | Δ AP @400 |")
out("|---|---:|---|---|---:|---:|")
for a in arms:
    if a == CONTROL:
        continue
    c100, c400 = paired("cost", a, CONTROL, 100), paired("cost", a, CONTROL, 400)
    p400, ap400 = paired("positives", a, CONTROL, 400), paired("ap", a, CONTROL, 400)
    out(f"| `{a}` | {ARM_K[a]:g} | {fmt(c100)} | {fmt(c400)} | {p400['mean']:+.1f} | {ap400['mean']:+.3f} |")
    for metric, H, pr in (("cost", 100, c100), ("cost", 400, c400), ("positives", 400, p400), ("ap", 400, ap400)):
        csv_delta.append(
            {
                "arm": a,
                "k": ARM_K[a],
                "metric": metric,
                "horizon": H,
                "mean": pr["mean"],
                "lo": pr["lo"],
                "hi": pr["hi"],
                "n": pr["n"],
            }
        )
out()

# --- the falsification arm -----------------------------------------------------
out("## The falsification arm\n")
if "acq_p2" in per_arm:
    p = paired("positives", "acq_p2", CONTROL, 400)
    ok = p["n"] and p["hi"] < 0
    out(
        f"`acq_p2` (k=+2) vs `prod`, positives @400: **{p['mean']:+.1f}** "
        f"[{p['lo']:+.1f}, {p['hi']:+.1f}] over {p['n']} pairs.\n"
    )
    if ok:
        out("**BEHAVED** — the lever is a lever.\n")
    else:
        out("**DID NOT DEGRADE — the run is VOID**, not explained: if sampling against")
        out("the evidence does not cost positives, nothing else here is interpretable.\n")
else:
    out("**MISSING — verdict withheld.** #3319's deep wave omitted this arm and its")
    out("analyzer refused a verdict for exactly this reason; the refusal is inherited.\n")

# --- H1: the difference-in-differences ----------------------------------------
out("## H1 — does the optimum move DEEPER through the session?\n")
out(
    f"`DiD = [m(deep,400) − m(deep,100)] − [m(shallow,400) − m(shallow,100)]`, "
    f"paired within the cell, against `{args.shallow}`.\n"
)
out("A **negative** DiD on cost (or on AUC, or on clicks-to-target) means the")
out("deeper arm gains ground as the session runs on — the optimum moves deeper,")
out("as the likelihood-ratio reading predicts. A **positive** one is #2910's")
out("reading: the benefit fades where scarcity ends.\n")
out("| deep arm | k | metric | DiD [95% CI] | pairs | tail | reading |")
out("|---|---:|---|---|---:|---|---|")
did_verdicts = []
compressed = set(harvest_flag)
for dp in [x for x in args.deep.split(",") if x in per_arm]:
    for metric, prec in (("cost", 4), ("auc", 4), ("ctt", 1)):
        A_ = per_arm[dp].set_index(list(PAIR_KEYS) + ["horizon"])[metric]
        B_ = per_arm[args.shallow].set_index(list(PAIR_KEYS) + ["horizon"])[metric]
        j = pd.concat([A_.rename("a"), B_.rename("b")], axis=1)
        try:
            d400 = (j.xs(400, level="horizon")["a"] - j.xs(400, level="horizon")["b"]).dropna()
            d100 = (j.xs(100, level="horizon")["a"] - j.xs(100, level="horizon")["b"]).dropna()
        except KeyError:
            continue
        did = (d400 - d100).dropna().to_numpy(dtype=float)
        if len(did) < 2:
            continue
        lo, hi = boot_ci(did, args.boot)
        mean = float(did.mean())
        if hi < 0:
            verdict = "**deeper** ✓"
        elif lo > 0:
            verdict = "**shallower** ✗"
        else:
            verdict = "no move"
        clean = dp not in compressed and args.shallow not in compressed
        did_verdicts.append((dp, metric, verdict, clean))
        csv_did.append(
            {
                "deep_arm": dp,
                "k": ARM_K[dp],
                "shallow_arm": args.shallow,
                "metric": metric,
                "mean": mean,
                "lo": lo,
                "hi": hi,
                "pairs": len(did),
                "clean": clean,
                "verdict": verdict.replace("*", "").replace(" ✓", "").replace(" ✗", "").strip(),
            }
        )
        out(
            f"| `{dp}` | {ARM_K[dp]:g} | {metric} | {mean:+.{prec}f} [{lo:+.{prec}f}, {hi:+.{prec}f}] | "
            f"{len(did)} | {'clean' if clean else '**compressed**'} | {verdict} |"
        )
out()
# The verdict is taken on the UNCOMPRESSED contrasts. A compressed deep arm has
# its late gains capped, which biases its DiD toward "no move" or "shallower" --
# one-sided, and in the direction that would falsify H1. So a compressed
# contrast can CORROBORATE a "deeper" finding and cannot produce one, and it can
# never be the evidence for "shallower".
clean_v = [v for v in did_verdicts if v[3]]
judged = clean_v or did_verdicts
if not clean_v and did_verdicts:
    out("**No uncompressed contrast available** -- every DiD below rests on an arm")
    out("above the harvest bar, so a null or a 'shallower' reading is not")
    out("interpretable. A 'deeper' reading would survive, being one-sided.\n")
deeper = [v for v in judged if "deeper" in v[2]]
shallower = [v for v in judged if "shallower" in v[2]]
if deeper and not shallower:
    out(f"**H1 SUPPORTED** on {len(deeper)} of {len(judged)} judged contrasts, none opposing.\n")
elif shallower and not deeper:
    out(
        f"**H1 FALSIFIED** — the optimum moves SHALLOWER ({len(shallower)} contrasts). "
        "That is #2910's reading, against the likelihood-ratio prediction.\n"
    )
elif not did_verdicts:
    out("**No DiD computed** — arms or horizons missing.\n")
elif deeper and shallower:
    out(
        "**MIXED — no verdict.** Contrasts disagree in sign, which is not a small "
        "effect but an inconsistent one; report it as such rather than pooling.\n"
    )
else:
    out(
        "**H1 NOT SUPPORTED and NOT FALSIFIED** — the optimum does not move on the "
        "range this grid covers. That retires the question rather than deferring it.\n"
    )

# --- H2: is the deep guardrail aggression, or exhaustion? ---------------------
out("## H2 — is the deep guardrail aggression, or exhaustion?\n")
out("#3319 measured `acq_m3`'s deep-spike incidence at **0.5% → 5.7% (p=0.006)**")
out("between 100 and 400 clicks, on a pile where that arm had consumed 82% of its")
out("positives. #2790 traced deep spikes to positive **starvation**. If that was")
out("the mechanism, the incidence should collapse here, where harvest is far lower")
out("at the same aggression.\n")
out("| arm | k | deep spikes @100 | deep spikes @400 | median harvest @400 |")
out("|---|---:|---:|---:|---:|")
for a in arms:
    s = per_arm[a]
    r = []
    for H in HORIZONS:
        sh = s[s["horizon"] == H]
        r.append(f"{100 * float(np.mean(sh['has_deep'])):.1f}%" if len(sh) else "—")
    s4 = s[s["horizon"] == 400]
    hv = f"{100 * float(np.nanmedian(s4['harvest'])):.1f}%" if len(s4) else "—"
    out(f"| `{a}` | {ARM_K[a]:g} | {r[0]} | {r[1]} | {hv} |")
    for H in HORIZONS:
        sh = s[s["horizon"] == H]
        if len(sh):
            csv_spikes.append(
                {
                    "arm": a,
                    "k": ARM_K[a],
                    "horizon": H,
                    "spike_frac": float(np.mean(sh["has_deep"])),
                    "n_cells": int(len(sh)),
                    "harvest_med": float(np.nanmedian(s4["harvest"])) if len(s4) else float("nan"),
                }
            )
out()
out("**Power, stated rather than discovered.** At a 0.5% baseline 192 cells expects")
out("~1 event, so this table cannot rank arms against each other for safety — which")
out("is exactly why #3319 recorded a hazard rather than evidence. It CAN resolve")
out("H2's contrast, which is large: 5.7% against ≤1% is ~11 events against ~2.\n")

if args.csv:
    cdir = pathlib.Path(args.csv)
    cdir.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("harvest", csv_harvest),
        ("speed", csv_speed),
        ("delta_vs_prod", csv_delta),
        ("did", csv_did),
        ("spikes", csv_spikes),
    ):
        pd.DataFrame(rows).to_csv(cdir / f"frontier_3547_{name}.csv", index=False)
    print(f"wrote 5 CSVs to {cdir}")

if args.markdown:
    pathlib.Path(args.markdown).write_text("\n".join(lines) + "\n")
    print(f"\nwrote {args.markdown}")
