# 2026-09-03 — a deep grid sized from its shipped arm, not its deepest (#3547)

**Study:** #3547, the deep-session acquisition offset — does the optimum move
*deeper* through a 400-click session? **Cost:** two of the study's three deep
contrasts, unreadable on a 1344-cell wave that had already been paid for.

`vg_scale_deep` was built at **900 positives per class**, and that number came
from two real constraints:

* a **supply** bound — the deepest value all twelve classes could furnish
  band-free (`stop sign` tops out at 1006);
* a **horizon** bound — `preflight.sh` check 16b, which it cleared comfortably:
  450 positives in the sim half against a 400-step horizon.

Neither of those is an **aggression** bound, and aggression is what sets
harvest. What the wave measured at t=400:

| arm | k | median harvest |
|---|---:|---:|
| `acq_m3` | −3 | 19% |
| `acq_m4` | −4 | 36% |
| `acq_m5` | −5 | **56%** |
| `acq_m6` | −6 | **60%** |

The plan pre-registered a 50% median harvest as the compression bar, so `-5`
and `-6` were reported as compressed and excluded — leaving exactly one clean
deep contrast (`-4` vs `-3`) out of three.

**This is worse than losing cells, because compression is one-sided.** A capped
tail biases a difference-in-differences toward "no move" or "shallower" and
never toward "deeper". The excluded arms therefore do not merely add noise, they
lean in a known direction — and all three of the study's "shallower" readings
landed on exactly those arms. A study that had not tracked harvest per arm would
have reported them as a finding.

**The generalisable rule: size a deep grid from its DEEPEST arm, not from its
shipped one.** Take the most aggressive arm in the grid, estimate what it will
harvest at the planned horizon, and pick the positive count so *that* arm stays
under the bar; every shallower arm is then comfortably under it too. The
arithmetic is one line:

    positives per class  >=  (positives the deepest arm finds by the horizon) / bar / sim_fraction

**Prevented.** `preflight.sh` check **16c**, opt-in with the bar the study
pre-registered:

```bash
bash scripts/experiments/preflight.sh --exp "$CALIB_EXP" \
  --require-harvest-headroom 0.5 --pilot-cells /expscratch/$USER/<pilot>/bin
```

It reads a short pilot wave of the deep arms
(`calibration/harvest_headroom.py`), projects each pilot cell's positives onto
the *planned* pile's depth, and refuses the launch when the worst arm's median
is at or over the bar — naming the positives-per-class that would clear it. With
no pilot it falls back to the one bound that needs no data at all (an arm cannot
find more positives than it takes clicks), which clears a deep enough pile
outright and otherwise says so instead of guessing.

A launcher preflights once per arm, so the study can declare it once instead —
`export CALIB_HARVEST_BAR=0.5 CALIB_HARVEST_PILOT=...` beside its other knobs —
and every call in the loop picks it up.

Its verdicts are deliberately asymmetric, which is the other half of
[the #3319 lesson](2026-09-02-one-pilot-cell-cleared-a-hazard-the-full-wave-hit.md):
**a pilot can fail a grid without being able to clear one.** Over the bar at
t=100 settles the matter, because harvest only goes up. Under the bar at t=100,
or on a pilot that skipped some of the planned categories, settles nothing —
harvest is the most category-dependent quantity in this harness — so the check
reports `UNKNOWN` rather than `ok`.

Check 16b is not made redundant by this and is not a substitute for it: 16b asks
whether the horizon can *exhaust* the pile, which is the right question for a
run trying to reach 100% and the wrong one for a grid whose contrasts are read
off a difference-in-differences. A tail does not have to be empty to be capped.
