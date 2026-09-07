#!/usr/bin/env python3
"""The clean-only arm: does the forward ratio SURVIVE removing contamination?

`provenance_shortcut.py` measured a forward ratio of 1.46 and a reverse of 0.895,
and the pure-contamination model predicts `forward + reverse = 2` for any rate --
the two arms are one threshold shift seen from both sides. The data sit at 2.35,
so contamination alone is refuted; what it cannot say is how big the remaining
provenance effect IS.

This identifies it. #3156's slates reviewed the shipped twelve's own negatives,
and 519 of those verdicts are **off-COCO images a human confirmed absent** and
that sat in the pre-#3670 pool -- negatives where provenance applies and
contamination does not. Fit against provable negatives exactly as before, then
score:

* every silent negative  -> the original forward ratio (contamination + provenance)
* the human-cleared ones  -> provenance alone

**The strata are reported separately and the RANDOM one is the estimate.** 363 of
the 519 come from the `boundary` stratum, which was text-ranked to be hard on
purpose; measuring a false-positive rate on images selected for looking like the
class would inflate it for a reason that has nothing to do with provenance. That
is the finding/estimating separation #3686 is built on, applied to its own study.

Runs against the ARCHIVED pre-#3670 pickle, which is the only one that still
contains a silent stratum.
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
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

ARCHIVE = Path("/expscratch/sgreenberg/archive/pre-3670-negpool")
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("clean_arm.json")
TARGET_FPR = 0.05
SEED = 0


def unit(a):
    return a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-8, None)


def main() -> None:
    cleared = json.load(open("/expscratch/sgreenberg/negpool-3670/clean_silent.json"))
    by_class: dict[str, dict[str, set[int]]] = {}
    for r in cleared:
        d = by_class.setdefault(r["class"], {"random": set(), "boundary": set()})
        if r["stratum"] in d:
            d[r["stratum"]].add(int(r["image_id"]))

    roster = json.load(open(ARCHIVE / "vg_scale_roster.json"))
    old_pool = {int(i) for i in roster.get("negatives", [])}
    report: dict[str, dict] = {}

    for embedder in ("siglip", "clip"):
        pkl = ARCHIVE / f"vg_scale__{embedder}.pkl"
        if not pkl.exists():
            report[embedder] = {"error": "archived cell missing"}
            continue
        m = load_medias(pkl)
        ids = [i for i in m if (m[i].get("embeddings") or {}).get(embedder) is not None]
        X = unit(np.asarray([m[i]["embeddings"][embedder] for i in ids], dtype=np.float32))
        k = {i: n for n, i in enumerate(ids)}
        exh = {i for i in ids if m[i].get("labels_exhaustive")}

        prov = np.asarray([k[i] for i in ids if i in old_pool and i in exh])
        silent = np.asarray([k[i] for i in ids if i in old_pool and i not in exh])

        per_class = {}
        for c, strata in sorted(by_class.items()):
            cells_c = {pc.scale_cell(c, b) for b in pc.BOX_BANDS}
            pos = np.asarray([k[i] for i in ids if cells_c & set(m[i].get("categories") or [])])
            rand = np.asarray([k[i] for i in strata["random"] if i in k])
            bound = np.asarray([k[i] for i in strata["boundary"] if i in k])
            if len(pos) < 30 or len(rand) < 20:
                continue

            idx = np.concatenate([pos, prov])
            lab = np.concatenate([np.ones(len(pos), np.int8), np.zeros(len(prov), np.int8)])
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

            got: dict[str, list[float]] = {"all_silent": [], "cleared_random": [], "cleared_boundary": []}
            for tr, te in cv.split(idx, lab):
                clf = LogisticRegression(max_iter=2000, C=1.0).fit(X[idx[tr]], lab[tr])
                held = idx[te][lab[te] == 0]
                if len(held) < 20:
                    continue
                thr = float(np.quantile(clf.decision_function(X[held]), 1 - TARGET_FPR))
                got["all_silent"].append(float((clf.decision_function(X[silent]) > thr).mean()) / TARGET_FPR)
                got["cleared_random"].append(float((clf.decision_function(X[rand]) > thr).mean()) / TARGET_FPR)
                if len(bound) >= 20:
                    got["cleared_boundary"].append(float((clf.decision_function(X[bound]) > thr).mean()) / TARGET_FPR)

            per_class[c] = {
                "n_pos": int(len(pos)),
                "n_random": int(len(rand)),
                "n_boundary": int(len(bound)),
                **{f"ratio_{key}": (round(float(np.mean(v)), 3) if v else None) for key, v in got.items()},
                "se_cleared_random": (
                    round(float(np.std(got["cleared_random"], ddof=1) / np.sqrt(len(got["cleared_random"]))), 3)
                    if len(got["cleared_random"]) > 1
                    else None
                ),
            }
            p = per_class[c]
            print(
                f"{embedder:7s} {c:9s} all-silent {p['ratio_all_silent']}  "
                f"cleared-RANDOM {p['ratio_cleared_random']} +/- {p['se_cleared_random']} (n={p['n_random']})  "
                f"cleared-boundary {p['ratio_cleared_boundary']} (n={p['n_boundary']})"
            )
        report[embedder] = {"n_provable": int(len(prov)), "n_silent": int(len(silent)), "per_class": per_class}

    OUT.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
