#!/usr/bin/env python3
"""Would an ALL-PROVABLE negative pool hand a head a provenance shortcut?

#3670 declines to draw every negative from the COCO-anchored half because the
positives are only 57% COCO-sourced, so provenance would correlate with the
label: under that composition an off-COCO image is a positive with certainty.
``provenance_probe.py`` shows provenance is linearly readable at AUC 0.53-0.56 --
weakly, but not at chance. This asks the question that actually decides the
composition: does that readability become a shortcut a head will USE?

Method is #3667's, deliberately, so the answer lands on the same scale as the
effect that justified the last rebuild:

1. fit a head on positives against **provable negatives only** -- the all-provable
   composition, as the benchmark would pose it;
2. pin a threshold at 5% FPR on **held-out provable** negatives;
3. score the **silent** (off-COCO) negatives, which the head never saw, at that
   threshold.

A ratio of 1 means provenance buys nothing. #3667's cross-class shortcut measured
**1.88 +/- 0.19** on this scale, and that was worth rebuilding eleven cells for.

**The forward ratio is an UPPER bound and it came back 1.46 +/- 0.10, so it did
NOT settle the question.** The silent negatives also carry genuine VG-silence
contamination: some really do hold the class, and a head SHOULD score those high.
At #3666's measured 1.40% [0.68, 2.86] pooled over the shipped twelve, and a TPR
near 0.7 at this threshold, contamination alone predicts
``1 + 0.0140 x (0.70/0.05 - 1)`` = **1.18** -- much of what was measured. So the
forward arm alone cannot tell a provenance shortcut from a dirty stratum. (An
earlier version of this docstring used 2.5%, the top of #3635's *predicted*
per-class range, and got 1.32; that overstated the contamination and was what
made the trade look decisive.)

Two further arms separate them, and neither needs new labelling:

* **reverse** -- fit on positives against the SILENT negatives, pin the threshold
  on held-out silent, and score the PROVABLE ones. A ratio at 1 says provenance
  buys nothing and a ratio below 1 is the shortcut with its sign reversed.

  **This arm is NOT contamination-free, and an earlier version of this docstring
  said it was.** The threshold is pinned at the 95th percentile of a stratum that
  holds hidden positives, which pushes it UP, so clean negatives then fire at
  under 5% -- contamination depresses the reverse arm exactly as it inflates the
  forward one. The two are one route counted twice, not two.

  What that buys instead is better than what it cost: under PURE contamination
  the two arms must satisfy **forward + reverse = 2**, for any contamination rate
  and any TPR (`contamination_identity.py`: 2.00-2.02 for rates up to 2.5%). The sum
  is therefore a **contamination-free diagnostic**. It does not size the
  asymmetry -- `1/reverse` is a lower bound, and a Gaussian shift model that
  reproduces the forward arm gives reverse 0.65 against a measured 0.895, so the
  arms are not reconcilable under the simplest model either. Sizing it needs the
  clean-only arm below, at a sample the existing labels cannot supply.
* **clean-only forward** -- the forward arm with the silent negatives restricted
  to those a human reviewed and confirmed absent (`corrections.json` carries a row
  for every reviewed pair, agreeing or not). Contamination is removed by
  selection rather than by arithmetic.

They fail differently -- the reverse arm trains on a dirty stratum and is
depressed by it, the clean-only arm is limited by review coverage -- so
agreement between them is worth more than either alone. **Agreement between
forward and reverse is worth nothing**, for the reason above.

**Read `ratio` and `ratio_reverse` PER CLASS, never pooled**, and read the
EXCESS over 2 rather than the sum -- as sums the rows below read 2.65 against
2.07 and look like a 28% spread. `forward + reverse - 2` runs `bus` **0.65** and
`bicycle` **0.60** against `knife` **0.07** and `boat` **0.13**, the last two
indistinguishable from pure contamination. A pooled number was read as evidence
that the artifact is uniform across classes, and that reading is what #3670's
composition was justified on (#3702).

Run on `clip` as well as the shipped `siglip`: CLIP reads provenance most
strongly of the five columns, so it is the adversarial case.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts/experiments/pile")
sys.path.insert(0, "scripts/experiments/calibration")

import pile_config as pc  # noqa: E402
from _cells_io import load_medias  # noqa: E402
from pilebuild.corrections import load_corrections  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("provenance_shortcut.json")
EMBEDDERS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["siglip", "clip"]
#: Where the cells live. Overridable because this measurement needs a pool with
#: BOTH strata in it, and the composition it argues for leaves only one -- so
#: once #3670 lands, the only place to reproduce it is the archived pre-change
#: cell, not the live pile. A measurement whose own result destroys its input
#: has to say where the input went.
CELLS = Path(sys.argv[3]) if len(sys.argv) > 3 else pc.EMBEDDINGS
TARGET_FPR = 0.05
SEED = 0


def unit(a: np.ndarray) -> np.ndarray:
    return a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-8, None)


def main() -> None:
    report: dict[str, dict] = {}
    corrections = load_corrections()

    for embedder in EMBEDDERS:
        pkl = CELLS / f"vg_scale__{embedder}.pkl"
        if not pkl.exists():
            report[embedder] = {"error": "cell missing"}
            continue
        m = load_medias(pkl)

        ids = [i for i in m if (m[i].get("embeddings") or {}).get(embedder) is not None]
        X = unit(np.asarray([m[i]["embeddings"][embedder] for i in ids], dtype=np.float32))
        by_id = {i: k for k, i in enumerate(ids)}

        idx_to_id = {k: i for i, k in by_id.items()}
        # {image_id: {class, ...}} a human reviewed and confirmed ABSENT. A row
        # exists for every reviewed pair whether or not the human disagreed, so
        # this is review coverage, not just the disagreements.
        reviewed_absent: dict[int, set[str]] = {}
        for (iid, cls), row in corrections.items():
            if not row.get("present"):
                reviewed_absent.setdefault(iid, set()).add(cls)

        neg_ids = [i for i in ids if not m[i].get("categories")]
        prov = np.asarray([by_id[i] for i in neg_ids if m[i].get("labels_exhaustive")])
        silent = np.asarray([by_id[i] for i in neg_ids if not m[i].get("labels_exhaustive")])

        per_class = {}
        for c in pc.SCALE_CLASSES:
            # `categories` holds CELL names (`bus@small`), never the bare class,
            # so a bare `in` test silently matches nothing and every class is
            # skipped by the size guard below.
            cells_c = {pc.scale_cell(c, b) for b in pc.BOX_BANDS}
            pos = np.asarray([by_id[i] for i in ids if cells_c & set(m[i].get("categories") or [])])
            if len(pos) < 30 or len(prov) < 100 or len(silent) < 100:
                continue

            # Fold over BOTH positives and provable negatives so the held-out
            # provable set that sets the threshold is never trained on.
            idx = np.concatenate([pos, prov])
            lab = np.concatenate([np.ones(len(pos), np.int8), np.zeros(len(prov), np.int8)])
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

            fprs, ratios = [], []
            for tr, te in cv.split(idx, lab):
                clf = LogisticRegression(max_iter=2000, C=1.0)
                clf.fit(X[idx[tr]], lab[tr])
                held_neg = idx[te][lab[te] == 0]
                if len(held_neg) < 20:
                    continue
                # Threshold pinned on held-out PROVABLE negatives, then applied
                # unchanged to the silent ones. Re-pinning per stratum would
                # define the difference away.
                thr = float(np.quantile(clf.decision_function(X[held_neg]), 1 - TARGET_FPR))
                fpr_silent = float((clf.decision_function(X[silent]) > thr).mean())
                fprs.append(fpr_silent)
                ratios.append(fpr_silent / TARGET_FPR)

            # --- reverse arm: fit against SILENT, score PROVABLE -----------
            ridx = np.concatenate([pos, silent])
            rlab = np.concatenate([np.ones(len(pos), np.int8), np.zeros(len(silent), np.int8)])
            rev = []
            for tr, te in cv.split(ridx, rlab):
                clf = LogisticRegression(max_iter=2000, C=1.0)
                clf.fit(X[ridx[tr]], rlab[tr])
                held = ridx[te][rlab[te] == 0]
                if len(held) < 20:
                    continue
                thr = float(np.quantile(clf.decision_function(X[held]), 1 - TARGET_FPR))
                rev.append(float((clf.decision_function(X[prov]) > thr).mean()) / TARGET_FPR)

            # --- clean-only forward: silent negatives a human cleared ---------
            clean_only = np.asarray(
                [j for j in silent if reviewed_absent.get(idx_to_id[j], set()) and c in reviewed_absent[idx_to_id[j]]]
            )
            clean_ratios = []
            if len(clean_only) >= 30:
                for tr, te in cv.split(idx, lab):
                    clf = LogisticRegression(max_iter=2000, C=1.0)
                    clf.fit(X[idx[tr]], lab[tr])
                    held_neg = idx[te][lab[te] == 0]
                    if len(held_neg) < 20:
                        continue
                    thr = float(np.quantile(clf.decision_function(X[held_neg]), 1 - TARGET_FPR))
                    clean_ratios.append(float((clf.decision_function(X[clean_only]) > thr).mean()) / TARGET_FPR)

            if ratios:
                per_class[c] = {
                    "n_pos": int(len(pos)),
                    "fpr_silent": round(float(np.mean(fprs)), 4),
                    "ratio": round(float(np.mean(ratios)), 3),
                    "ratio_se": round(float(np.std(ratios, ddof=1) / np.sqrt(len(ratios))), 3),
                    "ratio_reverse": round(float(np.mean(rev)), 3) if rev else None,
                    "n_clean_only": int(len(clean_only)),
                    "ratio_clean_only": round(float(np.mean(clean_ratios)), 3) if clean_ratios else None,
                }

        if not per_class:
            raise SystemExit(f"{embedder}: no class cleared the size guard -- check the label keying")

        def pooled(key: str) -> tuple[float | None, float | None]:
            xs = [v[key] for v in per_class.values() if v.get(key) is not None]
            if not xs:
                return None, None
            se = float(np.std(xs, ddof=1) / np.sqrt(len(xs))) if len(xs) > 1 else None
            return round(float(np.mean(xs)), 3), (round(se, 3) if se is not None else None)

        vals = [v["ratio"] for v in per_class.values()]
        report[embedder] = {
            "n_provable_negatives": int(len(prov)),
            "n_silent_negatives": int(len(silent)),
            "target_fpr": TARGET_FPR,
            "per_class": per_class,
            "ratio_mean": round(float(np.mean(vals)), 3) if vals else None,
            "ratio_se": round(float(np.std(vals, ddof=1) / np.sqrt(len(vals))), 3) if len(vals) > 1 else None,
            "ratio_reverse_mean": pooled("ratio_reverse")[0],
            "ratio_reverse_se": pooled("ratio_reverse")[1],
            "ratio_clean_only_mean": pooled("ratio_clean_only")[0],
            "ratio_clean_only_se": pooled("ratio_clean_only")[1],
            "n_clean_only_total": int(sum(v.get("n_clean_only", 0) for v in per_class.values())),
        }
        r = report[embedder]
        print(
            f"{embedder}: forward {r['ratio_mean']} +/- {r['ratio_se']} | "
            f"reverse {r['ratio_reverse_mean']} +/- {r['ratio_reverse_se']} | "
            f"clean-only {r['ratio_clean_only_mean']} +/- {r['ratio_clean_only_se']} "
            f"(n={r['n_clean_only_total']}) over {len(vals)} classes"
        )

    OUT.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
