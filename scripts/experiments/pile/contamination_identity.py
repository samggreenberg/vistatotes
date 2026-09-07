#!/usr/bin/env python3
"""Why `forward + reverse = 2` under pure contamination, and what that buys.

``provenance_shortcut.py`` runs two arms against #3667's FPR scale, and #3670's
report first read their agreement as two independent routes converging on one
number. **They are not independent**, and this is the demonstration.

Both arms pin a threshold at the ``1 - FPR`` quantile of one stratum and score
the other at it unchanged:

* **forward** -- threshold on the clean (COCO-scored) negatives, score the silent
  ones. A fraction *c* of those are really positives, found at the TPR rather
  than at the false-positive rate, so the ratio rises to ``1 + c(TPR/FPR - 1)``.
* **reverse** -- threshold on the *silent* negatives, score the clean ones. The
  same hidden positives sit at the top of the distribution the quantile is taken
  from, so they push the threshold **up**, and clean negatives then fire at under
  ``FPR``. The ratio falls.

One cause, two arms, opposite signs. To first order in *c* the two displacements
are equal and opposite, so **the sum is 2 whatever the contamination rate and
whatever the TPR** -- which makes ``forward + reverse - 2`` a *contamination-free
diagnostic*, and ``1/reverse`` a lower bound on any real asymmetry rather than an
estimate of one.

This script is the check on that claim, over rates and separations spanning what
this dataset plausibly has. It plants the answer -- there is no provenance effect
in the simulation at all, the two strata are drawn from the same distribution --
so any departure from 2 is the approximation's error and nothing else.

Measured on the real cells the sums are **2.35 / 2.36 / 2.42**, far outside what
this produces, which is what refutes contamination-alone. See
``docs/experiments/2026-09-06-provable-negatives-3670/REPORT.md`` §3 and #3702.

Usage::

    python contamination_identity.py            # the table the report quotes
"""

from __future__ import annotations

import numpy as np

#: The operating point `provenance_shortcut.py` pins at.
TARGET_FPR = 0.05
#: Contamination rates worth checking: #3666 measured 1.40% [0.68, 2.86] pooled
#: over the shipped twelve, and 5% is included only to show where the first-order
#: approximation starts to visibly bend.
RATES = (0.005, 0.014, 0.025, 0.05)
#: Separation between the positive and negative score distributions, in SDs.
#: These give a TPR of roughly 0.44 / 0.64 / 0.83 at the 5% threshold, bracketing
#: the ~0.70 the report assumes.
SEPARATIONS = (1.5, 2.0, 2.6)
N = 2_000_000
SEED = 0


def arms(rng: np.random.Generator, c: float, sep: float) -> tuple[float, float, float]:
    """``(forward, reverse, tpr)`` with NO provenance effect present.

    The two negative strata are drawn from the same distribution on purpose: the
    only thing that differs is that a fraction *c* of the silent one is really
    positive. Anything the arms then show is contamination and nothing else.
    """
    positives = rng.normal(sep, 1.0, N)
    provable = rng.normal(0.0, 1.0, N)
    hidden = int(round(c * N))
    silent = np.concatenate([rng.normal(0.0, 1.0, N - hidden), rng.normal(sep, 1.0, hidden)])

    t_forward = float(np.quantile(provable, 1 - TARGET_FPR))
    t_reverse = float(np.quantile(silent, 1 - TARGET_FPR))
    return (
        float((silent > t_forward).mean()) / TARGET_FPR,
        float((provable > t_reverse).mean()) / TARGET_FPR,
        float((positives > t_forward).mean()),
    )


def main() -> None:
    rng = np.random.default_rng(SEED)
    print(f"{'c':>7}{'sep':>6}{'TPR':>7}{'forward':>9}{'reverse':>9}{'sum':>8}{'excess':>9}")
    worst = 0.0
    for c in RATES:
        for sep in SEPARATIONS:
            fwd, rev, tpr = arms(rng, c, sep)
            worst = max(worst, abs(fwd + rev - 2.0))
            print(f"{c:>7.3f}{sep:>6.1f}{tpr:>7.2f}{fwd:>9.3f}{rev:>9.3f}{fwd + rev:>8.3f}{fwd + rev - 2:>9.3f}")
    print(f"\nlargest departure from 2 anywhere in this grid: {worst:.3f}")
    print("measured on the real cells: 2.35 (siglip), 2.36 (clip), 2.42 (siglip2_l)")


if __name__ == "__main__":
    main()
