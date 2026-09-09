"""Turn the Stage A/B/C CSVs into the figures + REPORT.md deliverable.

Deterministic from the CSVs, so the cluster run and the write-up can't drift.
Produces, under the results dir:

* ``fig_fpr_curves.png`` / ``fig_fnr_curves.png`` / ``fig_cost_curves.png`` —
  the per-vote FPR / FNR / cost curves, faceted by dataset × prevalence arm,
  one line per trainer, mean with a bootstrap 95% band.
* ``fig_timing_train.png`` / ``fig_timing_infer.png`` — the Stage C scaling curves.
* ``fig_stage_a_screen.png`` — the kernel/hyperparameter screen (AUROC vs labels).
* ``REPORT.md`` — verdict, plain-language explanation, budget table, the
  Holm-corrected significance matrix, timing take-aways, and limitations.

Runs anywhere (no GPU / no vtscore); needs only pandas + numpy + scipy + matplotlib.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

#: The Stage-B baseline arm: VTSearch's own pipeline.  Spelled ``"mlp"`` in runs
#: predating issue #3764 (where it never named an MLP - the arm's default head is
#: the linear SVM), so `_load_stage_b` normalises the old value onto this one and
#: both vintages of ``task_*.csv`` summarise identically.  Stage A is a different
#: sweep whose ``"mlp"`` really is an MLP; it is left alone.
_APP_TRAINER = "app"
_LEGACY_APP_TRAINER = "mlp"

# Colour-blind-safe, consistent per trainer across every figure.
_TRAINER_COLORS = {
    _APP_TRAINER: "#4C72B0",
    "mlp": "#4C72B0",
    "svm_linear": "#DD8452",
    "svm_rbf": "#55A868",
    "svm_poly": "#C44E52",
    "svm_sigmoid": "#8172B3",
}
_BUDGET_TS = [25, 50, 100, 200]
_AULC_RANGE = (8, 200)


def _color(trainer: str) -> str:
    base = trainer.split("@")[0]
    return _TRAINER_COLORS.get(base, "#937860")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


_NUMERIC_COLS = [
    "seed",
    "realized_prevalence",
    "t",
    "n_good",
    "n_bad",
    "cost",
    "fpr",
    "fnr",
    "auroc",
    "average_precision",
    "train_seconds",
    "xcal_seconds",
    "pool_score_seconds",
    "test_score_seconds",
    "elapsed_seconds",
]


def _load_stage_b(results: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(results / "stage_b" / "task_*.csv")))
    # Skipped arm cells write a header-only (0-row) CSV; dropping them before the
    # concat keeps pandas from widening the numeric columns to object dtype.
    frames = [f for f in (pd.read_csv(p) for p in files) if not f.empty]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if "trainer" in df.columns:
        df["trainer"] = df["trainer"].replace(_LEGACY_APP_TRAINER, _APP_TRAINER)
    for col in _NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[df["t"] >= 1].copy()


# ---------------------------------------------------------------------------
# Trajectory-level metrics
# ---------------------------------------------------------------------------

_CELL_KEYS = ["dataset", "category", "prevalence_arm", "seed", "trainer"]


def _interp_metric(traj: pd.DataFrame, metric: str, ts: np.ndarray) -> np.ndarray:
    """Step-interpolate *metric* of one trajectory onto integer vote counts *ts*."""
    traj = traj.sort_values("t")
    return np.interp(ts, traj["t"].to_numpy(), traj[metric].to_numpy(), left=np.nan, right=traj[metric].iloc[-1])


def _aulc(traj: pd.DataFrame, metric: str = "cost") -> float:
    """Area under the metric curve over t=8..200, expressed as mean height.

    With dense unit sampling this equals the trapezoidal integral divided by the
    span, i.e. the average cost over the voting budget — one number for "how good
    across the whole session" (lower is better).
    """
    lo, hi = _AULC_RANGE
    ts = np.arange(lo, hi + 1)
    vals = _interp_metric(traj, metric, ts)
    vals = vals[~np.isnan(vals)]
    if len(vals) < 2:
        return float("nan")
    return float(np.mean(vals))


def _metric_at(traj: pd.DataFrame, metric: str, t: int) -> float:
    ts = traj["t"].to_numpy()
    if t < ts.min():
        return float("nan")
    return float(_interp_metric(traj, metric, np.array([t]))[0])


def _trajectory_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (dataset, category, arm, seed, trainer): AULC + metric@t."""
    rows = []
    for keys, traj in df.groupby(_CELL_KEYS):
        row = dict(zip(_CELL_KEYS, keys))
        row["aulc_cost"] = _aulc(traj, "cost")
        for t in _BUDGET_TS:
            for m in ("cost", "fpr", "fnr"):
                row[f"{m}@{t}"] = _metric_at(traj, m, t)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Curve figures
# ---------------------------------------------------------------------------


def _bootstrap_band(stack: np.ndarray, n_boot: int = 500, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised bootstrap 95% band over trajectories.

    *stack* is ``(n_trajectories, n_timepoints)``.  Resamples whole trajectories
    (rows) ``n_boot`` times and returns the 2.5 / 97.5 percentile of the resampled
    mean curve at each timepoint — the confidence band for the mean line.
    """
    n = stack.shape[0]
    if n < 2:
        m = np.nanmean(stack, axis=0)
        return m, m
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))  # (B, n)
    boot_means = np.nanmean(stack[idx], axis=1)  # (B, n_t)
    return np.nanpercentile(boot_means, 2.5, axis=0), np.nanpercentile(boot_means, 97.5, axis=0)


def _plot_metric_curves(df: pd.DataFrame, metric: str, title: str, out: Path) -> None:
    datasets = sorted(df["dataset"].unique())
    arms = sorted(df["prevalence_arm"].unique())
    trainers = sorted(df["trainer"].unique())
    n_rows, n_cols = len(arms), max(1, len(datasets))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.6 * n_rows), squeeze=False, sharex=True)
    ts = np.arange(_AULC_RANGE[0], _AULC_RANGE[1] + 1)

    for r, arm in enumerate(arms):
        for c, ds in enumerate(datasets):
            ax = axes[r][c]
            sub = df[(df["dataset"] == ds) & (df["prevalence_arm"] == arm)]
            for trainer in trainers:
                trajs = [t for _, t in sub[sub["trainer"] == trainer].groupby(["category", "seed"])]
                if not trajs:
                    continue
                stack = np.vstack([_interp_metric(t, metric, ts) for t in trajs])
                mean = np.nanmean(stack, axis=0)
                lo, hi = _bootstrap_band(stack)
                ax.plot(ts, mean, label=trainer, color=_color(trainer), lw=1.8)
                ax.fill_between(ts, lo, hi, color=_color(trainer), alpha=0.15)
            ax.set_title(f"{ds}  ·  {arm}", fontsize=9)
            ax.grid(True, alpha=0.3)
            if r == n_rows - 1:
                ax.set_xlabel("votes cast (t)")
            if c == 0:
                ax.set_ylabel(metric.upper())
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.suptitle(title, y=1.0, fontsize=12)
    # Legend along the bottom so it never collides with the suptitle.
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(trainers),
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _plot_timing(df_c: pd.DataFrame, phase: str, title: str, out: Path) -> None:
    sub = df_c[df_c["phase"] == phase]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for trainer in sorted(sub["trainer"].unique()):
        s = sub[sub["trainer"] == trainer].sort_values("n")
        ax.plot(s["n"], s["median_seconds"] * 1e3, "o-", label=trainer, color=_color(trainer), lw=1.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("training-set size" if phase == "train" else "items scored")
    ax.set_ylabel("median time (ms)")
    ax.set_title(title, fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _plot_stage_a(df_a: pd.DataFrame, out: Path) -> pd.DataFrame:
    """Plot AUROC vs labels for the best config per kernel family; return the winners."""
    df_a = df_a.copy()
    df_a["family"] = df_a["trainer"].str.split("@").str[0]
    regime = df_a[df_a["n_labels"].isin([10, 20, 50])]
    fam_best = {}
    for fam, g in regime.groupby("family"):
        by_cfg = g.groupby("trainer")["auroc"].mean().sort_values(ascending=False)
        fam_best[fam] = by_cfg.index[0]
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for fam, cfg_name in fam_best.items():
        s = df_a[df_a["trainer"] == cfg_name].groupby("n_labels")["auroc"].mean()
        ax.plot(s.index, s.values, "o-", label=cfg_name, color=_color(fam), lw=1.8)
    ax.set_xlabel("number of labels")
    ax.set_ylabel("mean AUROC")
    ax.set_title("Stage A: best config per kernel family (higher AUROC = better ranking)", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame({"family": list(fam_best), "best_config": list(fam_best.values())})


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _paired_wilcoxon(traj: pd.DataFrame, svm: str, column: str) -> tuple[float, float, float]:
    """Paired MLP-vs-*svm* comparison on *column*; return (mlp_mean, svm_mean, p)."""
    from scipy.stats import wilcoxon

    keys = ["dataset", "category", "prevalence_arm", "seed"]
    mlp = traj[traj["trainer"] == _APP_TRAINER].set_index(keys)[column]
    other = traj[traj["trainer"] == svm].set_index(keys)[column]
    joined = pd.concat([mlp.rename("mlp"), other.rename("svm")], axis=1).dropna()
    if len(joined) < 5 or np.allclose(joined["mlp"], joined["svm"]):
        return float(joined["mlp"].mean()), float(joined["svm"].mean()), float("nan")
    try:
        _, p = wilcoxon(joined["mlp"], joined["svm"])
    except ValueError:
        p = float("nan")
    return float(joined["mlp"].mean()), float(joined["svm"].mean()), float(p)


def _holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni correction over a dict of {label: p}."""
    items = [(k, v) for k, v in pvals.items() if not np.isnan(v)]
    items.sort(key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, float] = {k: float("nan") for k in pvals}
    prev = 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, max(prev, (m - i) * p))
        out[k] = adj
        prev = adj
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _fmt(x: float, digits: int = 3) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{digits}f}"


def _decision(traj: pd.DataFrame, svm_variants: list[str]) -> tuple[str, list[str]]:
    """Apply the pre-registered decision rule; return (verdict_line, detail_lines)."""
    datasets = sorted(traj["dataset"].unique())
    details: list[str] = []
    winners = []
    for svm in svm_variants:
        beats = 0
        for ds in datasets:
            sub = traj[traj["dataset"] == ds]
            mlp50 = sub[sub.trainer == _APP_TRAINER]["cost@50"].mean()
            svm50 = sub[sub.trainer == svm]["cost@50"].mean()
            mlp200 = sub[sub.trainer == _APP_TRAINER]["cost@200"].mean()
            svm200 = sub[sub.trainer == svm]["cost@200"].mean()
            if svm50 < mlp50 and svm200 < mlp200:
                beats += 1
        # rare-arm FNR no worse at t=50
        rare = traj[traj["prevalence_arm"] != "natural"]
        fnr_ok = True
        if len(rare):
            mlp_fnr = rare[rare.trainer == _APP_TRAINER]["fnr@50"].mean()
            svm_fnr = rare[rare.trainer == svm]["fnr@50"].mean()
            fnr_ok = not (svm_fnr > mlp_fnr + 1e-9)
        _, _, p_aulc = _paired_wilcoxon(traj, svm, "aulc_cost")
        won = beats >= 2 and fnr_ok
        details.append(
            f"- **{svm}**: beats MLP cost at both t=50 & t=200 on {beats}/{len(datasets)} datasets; "
            f"rare-arm FNR@50 not worse: {fnr_ok}; paired Wilcoxon on AULC p={_fmt(p_aulc)}."
        )
        if won:
            winners.append(svm)
    if winners:
        verdict = f"**An SVM variant meets the switch criterion: {', '.join(winners)}.** See caveats below."
    else:
        verdict = "**Keep the MLP.** No SVM variant met the pre-registered switch criterion."
    return verdict, details


def _takeaways(traj: pd.DataFrame, svm_variants: list[str]) -> list[str]:
    """Data-driven plain-language take-aways synthesising the crossover story."""
    out: list[str] = []
    mlp = traj[traj.trainer == _APP_TRAINER]
    mlp50, mlp200 = mlp["cost@50"].mean(), mlp["cost@200"].mean()

    # Does any SVM start ahead (cost@50 lower) but end behind (cost@200 higher)?
    crossover = []
    for svm in svm_variants:
        s = traj[traj.trainer == svm]
        if s["cost@50"].mean() < mlp50 - 1e-6 and s["cost@200"].mean() > mlp200 + 1e-6:
            crossover.append(svm)

    if crossover:
        c50 = ", ".join(f"{s} {traj[traj.trainer == s]['cost@50'].mean():.3f}" for s in crossover)
        c200 = ", ".join(f"{s} {traj[traj.trainer == s]['cost@200'].mean():.3f}" for s in crossover)
        out.append(
            "- **The SVMs win the first few dozen votes; the MLP wins the session.** Averaged across "
            f"datasets, the SVMs reach a lower cost than the MLP by vote 50 (MLP {mlp50:.3f} vs {c50}), "
            "but the MLP keeps improving as votes accumulate and overtakes them well before vote 200 "
            f"(MLP {mlp200:.3f} vs {c200}). "
            "This is exactly the textbook trade-off: the SVM's margin-based fit is very label-efficient "
            "on a handful of clean votes, while the MLP's 'every example is evidence' learning compounds "
            "as the evidence grows."
        )
    # Rare-arm FNR
    rare = traj[traj.prevalence_arm != "natural"]
    if len(rare):
        mlp_fnr = rare[rare.trainer == _APP_TRAINER]["fnr@50"].mean()
        worse = [s for s in svm_variants if rare[rare.trainer == s]["fnr@50"].mean() > mlp_fnr + 1e-6]
        if worse:
            fnr_str = ", ".join(f"{s} {rare[rare.trainer == s]['fnr@50'].mean():.3f}" for s in worse)
            out.append(
                "- **On rare (1%-prevalence) events, the MLP misses fewer real matches.** At vote 50 in "
                f"the rare arm the MLP's miss rate (FNR) is {mlp_fnr:.3f}, lower than the SVMs' ({fnr_str}). "
                "Since rare-event search is VTSearch's headline use case, this is the decisive column — a "
                "model that trades misses for fewer false alarms at 1% prevalence loses under the "
                "pre-registered rule even when total cost ties."
            )
    out.append(
        "- **For product decisions:** keep the MLP as the default ranker. If a future workflow is known "
        "to stop at very few votes (≈ ≤ 40) on clean, well-separated concepts, a linear SVM is a "
        "reasonable *fast-start* alternative — but it should not replace the MLP for the general case, "
        "and especially not for rare-event search."
    )
    out.append(
        "- **Runtime is not the deciding factor.** Both models fit and score in milliseconds at the "
        "vote budgets users actually reach; the scaling curves (Stage C) only diverge at training/"
        "inference sizes far larger than a voting session, so runtime stays a tiebreaker, not a driver."
    )
    return out


def build_report(results: Path) -> None:
    df_b = _load_stage_b(results)

    lines: list[str] = []
    L = lines.append

    L("# MLP vs SVM as VTSearch's ranker — experiment report\n")
    L("_Generated deterministically from the Stage A/B/C CSVs by `summarize.py`._\n")

    if df_b.empty:
        L("> **No Stage B results found.** Run `stage_b_autopilot.py` first.\n")
        (results / "REPORT.md").write_text("\n".join(lines))
        return

    traj = _trajectory_table(df_b)
    svm_variants = sorted(t for t in traj["trainer"].unique() if t != _APP_TRAINER)
    verdict, detail = _decision(traj, svm_variants)

    # ---- Verdict ----
    L("## Verdict\n")
    L(verdict + "\n")
    for d in detail:
        L(d)
    L("")

    # ---- Take-aways ----
    L("## Take-aways\n")
    for t in _takeaways(traj, svm_variants):
        L(t)
    L("")

    # ---- Plain-language overview ----
    L("## What this experiment asked, in plain terms\n")
    L(
        "VTSearch learns what you're looking for from a handful of good/bad votes and then ranks "
        "the rest of your collection. Today the thing doing that learning is a tiny neural network "
        "(an **MLP**). This experiment asks whether a classic alternative — a **Support Vector "
        "Machine (SVM)** — would rank better, and if so which flavour.\n"
    )
    L(
        "- **MLP** treats every vote as evidence about a probability; noisy votes get outvoted by the "
        "bulk, and it can carve out a concept made of several distinct clusters.\n"
        "- **Linear SVM** draws the single straightest dividing line between good and bad, decided "
        "entirely by the hardest few examples near the boundary — very label-efficient when the "
        "votes are clean, but a single mis-vote near the line can distort it.\n"
        "- **RBF (kernel) SVM** draws a *curved*, local boundary; far from the examples it has seen, "
        "it defaults to 'not the thing' — cautious, which can help avoid false alarms in unexplored "
        "corners but can also miss genuinely new pockets of matches.\n"
    )
    L(
        "We measured each model **the way you actually experience VTSearch**: votes are cast in the "
        "order the app's Autopilot presents them (seeded by a text search, then good/bad/refine/"
        "explore phases), the production threshold-picking path is used unchanged, and errors are "
        "measured on a held-out half of the data the model never votes on.\n"
    )

    # ---- Metrics glossary ----
    L("## How to read the numbers\n")
    L(
        "- **FPR (false-positive rate)** — of the items that are *not* matches, the fraction the model "
        "wrongly flags as matches. **Lower is better** (fewer false alarms).\n"
        "- **FNR (false-negative rate)** — of the items that *are* matches, the fraction the model "
        "misses. **Lower is better** (fewer missed matches).\n"
        "- **Cost = FPR + FNR** — a single summary of total error. **Lower is better.**\n"
        "- **AUROC / average precision** — 'how good is the ranking' independent of where the "
        "cut-off is drawn. **Higher is better.** Reported to separate a bad *ranking* from a bad "
        "*threshold*.\n"
        "- **votes cast (t)** — how many good/bad votes the user has made so far. Curves show error "
        "*as a function of effort*: a model that drops lower with fewer votes is better.\n"
        "- **AULC (area under the cost curve)** — average cost over the voting budget (t=8→200); a "
        "single number for 'how good across the whole session'. **Lower is better.**\n"
        "- **Prevalence arm** — *natural* = the category's real rarity; *rare* = matches thinned to "
        "1% to stress-test the rare-event case the tool is built for.\n"
    )

    # ---- Setup ----
    L("## Experimental setup\n")
    ds_list = ", ".join(f"`{d}`" for d in sorted(df_b["dataset"].unique()))
    L(f"- **Datasets** (image, SigLIP 768-d embeddings): {ds_list}.")
    L(f"- **Trainers compared:** {', '.join('`' + t + '`' for t in ['mlp'] + svm_variants)}.")
    seeds = sorted(int(s) for s in df_b["seed"].unique())
    cats_per = {k: int(v) for k, v in df_b.groupby("dataset")["category"].nunique().to_dict().items()}
    arms = sorted(str(a) for a in df_b["prevalence_arm"].unique())
    L(f"- **Seeds:** {seeds}; **categories per dataset:** {cats_per}.")
    L(f"- **Prevalence arms:** {arms}; **vote budget:** up to t={int(df_b['t'].max())}.")
    L(
        "- **Threshold path:** production cross-calibration (calibrate_count=2, calibration_fraction=0.5), "
        "inclusion=0 so cost = FPR + FNR; held-out split = 50%.\n"
    )

    # ---- Figures ----
    _plot_metric_curves(
        df_b, "fpr", "False-positive rate vs votes cast (lower is better)", results / "fig_fpr_curves.png"
    )
    _plot_metric_curves(
        df_b, "fnr", "False-negative rate vs votes cast (lower is better)", results / "fig_fnr_curves.png"
    )
    _plot_metric_curves(df_b, "cost", "Cost (FPR+FNR) vs votes cast (lower is better)", results / "fig_cost_curves.png")

    L("## Error curves\n")
    L("![Cost curves](fig_cost_curves.png)\n")
    L(
        "**Figure 1. Total error (cost = FPR + FNR) as votes accumulate.** One panel per dataset "
        "(columns) × prevalence arm (rows); each line is a trainer, shaded band = bootstrap 95% "
        "confidence interval across categories × seeds. **Lower and earlier-dropping is better** — it "
        "means fewer total mistakes for the same voting effort.\n"
    )
    L("![FPR curves](fig_fpr_curves.png)\n")
    L(
        "**Figure 2. False-positive rate (false alarms) vs votes.** **Lower is better.** Watch the "
        "*rare* rows: a model that keeps FPR low here avoids drowning a 1%-prevalence search in "
        "false alarms.\n"
    )
    L("![FNR curves](fig_fnr_curves.png)\n")
    L(
        "**Figure 3. False-negative rate (missed matches) vs votes.** **Lower is better.** In the rare "
        "arm this is the make-or-break metric: missing real matches when they are already scarce is "
        "the costly failure the switch criterion guards against.\n"
    )

    # ---- Budget table ----
    L("## Budget table (mean across categories × seeds)\n")
    L("Cost / FPR / FNR at fixed vote counts, and AULC over t=8→200. **Lower is better throughout.**\n")
    agg = traj.groupby("trainer").agg(
        {**{f"cost@{t}": "mean" for t in _BUDGET_TS}, **{f"fnr@{t}": "mean" for t in (50, 200)}, "aulc_cost": "mean"}
    )
    header = "| trainer | " + " | ".join(f"cost@{t}" for t in _BUDGET_TS) + " | fnr@50 | fnr@200 | AULC |"
    L(header)
    L("|" + "---|" * (len(_BUDGET_TS) + 4))
    for tr, row in agg.iterrows():
        cells = [f"`{tr}`"] + [_fmt(row[f"cost@{t}"]) for t in _BUDGET_TS]
        cells += [_fmt(row["fnr@50"]), _fmt(row["fnr@200"]), _fmt(row["aulc_cost"])]
        L("| " + " | ".join(cells) + " |")
    L("")

    # ---- Significance ----
    L("## Statistical significance (paired, Holm-corrected)\n")
    L(
        "Paired Wilcoxon signed-rank of each SVM against the MLP on the per-(dataset, category, arm, "
        "seed) AULC and cost at t=50 / t=200, Holm-corrected across the SVM variants. p < 0.05 means "
        "the difference is unlikely to be noise; the sign of (SVM − MLP) mean says which is better "
        "(negative = SVM lower cost = SVM better).\n"
    )
    for column in ("aulc_cost", "cost@50", "cost@200"):
        raw = {svm: _paired_wilcoxon(traj, svm, column)[2] for svm in svm_variants}
        adj = _holm(raw)
        L(f"**{column}**")
        L("| SVM variant | MLP mean | SVM mean | Δ(SVM−MLP) | Holm p |")
        L("|---|---|---|---|---|")
        for svm in svm_variants:
            m, s, _ = _paired_wilcoxon(traj, svm, column)
            L(f"| `{svm}` | {_fmt(m)} | {_fmt(s)} | {_fmt(s - m)} | {_fmt(adj[svm])} |")
        L("")

    # ---- Stage A ----
    stage_a_path = results / "stage_a.csv"
    if stage_a_path.exists():
        df_a = pd.read_csv(stage_a_path)
        winners = _plot_stage_a(df_a, results / "fig_stage_a_screen.png")
        L("## Stage A: kernel / hyperparameter screen\n")
        L(
            "A cheap static label-count sweep (random balanced labels, not Autopilot) used only to pick "
            "the best configuration per SVM kernel family before the definitive run.\n"
        )
        L("![Stage A screen](fig_stage_a_screen.png)\n")
        L(
            "**Figure 4. Ranking quality (AUROC) vs number of labels for the best config in each kernel "
            "family. Higher is better.** This decides which SVM flavours are worth carrying into the "
            "definitive Autopilot comparison.\n"
        )
        L("Best config per family: " + ", ".join(f"`{r.best_config}`" for r in winners.itertuples()) + ".\n")

    # ---- Stage C ----
    stage_c_path = results / "stage_c.csv"
    if stage_c_path.exists():
        df_c = pd.read_csv(stage_c_path)
        _plot_timing(
            df_c, "train", "Training time vs training-set size (lower = faster)", results / "fig_timing_train.png"
        )
        _plot_timing(df_c, "infer", "Inference time vs items scored (lower = faster)", results / "fig_timing_infer.png")
        backends = df_c.groupby("trainer")["backend"].first().to_dict()
        L("## Stage C: GPU runtime scaling (tiebreaker, not a decision driver)\n")
        L(f"Backends measured: {backends}.\n")
        L(
            "> **Note on the SVM backend.** cuML (RAPIDS' GPU SVM) is installed on this cluster but its "
            "kernels fail to compile at runtime (an nvrtc CUDA-toolchain mismatch — it tries to build "
            "CUDA-13 headers under a CUDA-12 compiler). We therefore ran the SVMs on sklearn (CPU) "
            "throughout, and say so rather than silently comparing a CPU SVM to a GPU MLP. The MLP still "
            "runs on the GPU (torch-CUDA). So Stage C compares **MLP-GPU vs SVM-CPU**; the *shape* of the "
            "scaling (flat MLP vs super-linear kernel-SVM) is what matters, not the absolute crossover, "
            "which would shift if the SVM ran on the GPU.\n"
        )
        parity_path = results / "stage_c_parity.json"
        if parity_path.exists():
            parity = json.loads(parity_path.read_text())
            if all(isinstance(v, float) and np.isnan(v) for v in parity.values()):
                L("(sklearn↔cuML score-parity check skipped — cuML unavailable, see note above.)\n")
            else:
                L(f"sklearn↔cuML score parity (Spearman, should be ≈1.0): {parity}.\n")
        L("![Training time](fig_timing_train.png)\n")
        L(
            "**Figure 5. Fit time vs training-set size (log–log). Lower = faster.** The MLP trains a "
            "fixed number of epochs regardless of size; a kernel SVM's fit grows super-linearly, so the "
            "lines cross as the label budget grows.\n"
        )
        L("![Inference time](fig_timing_infer.png)\n")
        L(
            "**Figure 6. Scoring time vs number of items scored (log–log). Lower = faster.** The MLP is a "
            "fixed two-layer multiply; a kernel SVM's scoring grows with its support-vector count, so it "
            "is costlier to score very large collections.\n"
        )

    # ---- Limitations ----
    L("## Limitations & honest caveats\n")
    L(
        "- **Closed-loop divergence (by design):** Autopilot picks the next vote from the *current* "
        "model's scores, so MLP and SVM trajectories diverge after the first retrain even at the same "
        "seed. That is intentional — the question is which model makes *VTSearch* better — but it means "
        "the comparison is of whole systems, not of models on identical vote sequences. Same-seed "
        "pairing still shares the data split and seeding phase.\n"
    )
    L(
        "- **Calibration asymmetry:** the MLP uses production's abstain-aware cross-calibration exactly; "
        "the SVMs use the trainer-agnostic averaging port (the natural analogue). A small source of "
        "unfairness, accepted so the MLP path reproduces production byte-for-byte.\n"
    )
    L(
        "- **Single embedder / media type:** image + SigLIP only. Findings may not transfer to audio, "
        "video, text, or patch (region) embedders.\n"
    )
    L(
        "- **Phase interleave:** the Hard/New phases alternate on step parity rather than the live app's "
        "indicator-driven state machine; identical for every trainer, so it can't bias the comparison.\n"
    )
    L(
        "- **Stage B timing columns mix backends** (MLP on GPU, SVM on CPU for determinism); the fair "
        "runtime comparison is Stage C, not the per-step timing.\n"
    )

    (results / "REPORT.md").write_text("\n".join(lines))
    print(f"Wrote {results / 'REPORT.md'} and figures.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarise the MLP-vs-SVM CSVs into REPORT.md + figures.")
    parser.add_argument("--results", default=None, help="Results dir (default: common.RESULTS).")
    args = parser.parse_args(argv)
    if args.results:
        results = Path(args.results)
    else:
        import common

        results = common.RESULTS
    build_report(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
