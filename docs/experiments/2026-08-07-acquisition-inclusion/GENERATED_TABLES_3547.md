# #3547 — does the optimum move deeper? — `siglip x whole_image`

Arms: `prod` (k=0), `acq_m1` (k=-1), `acq_m3` (k=-3), `acq_m4` (k=-4), `acq_m5` (k=-5), `acq_m6` (k=-6), `acq_p2` (k=2)
Horizons: 100, 400 clicks, off ONE wave (see the module docstring)
Cost-regression tolerance: ±0.01

## Cells read

| arm | files | read | zero-byte | unreadable | header-only |
|---|---:|---:|---:|---:|---:|
| `prod` | 192 | 192 | 0 | 0 | n/a |
| `acq_m1` | 192 | 192 | 0 | 0 | n/a |
| `acq_m3` | 192 | 192 | 0 | 0 | n/a |
| `acq_m4` | 192 | 192 | 0 | 0 | n/a |
| `acq_m5` | 192 | 192 | 0 | 0 | n/a |
| `acq_m6` | 192 | 192 | 0 | 0 | n/a |
| `acq_p2` | 192 | 192 | 0 | 0 | n/a |

## Realised harvest — is the tail interpretable?

An arm that has consumed most of its positives is no longer being compared
over the same opportunity as the control. Pre-registered bar: **a median
above 50% means that arm's deep readings are reported as compressed.**

| arm | k | sim positives | median positives @400 | median harvest | cells >50% | cells >80% |
|---|---:|---:|---:|---:|---:|---:|
| `prod` | 0 | 450 | 20 | 4.4% | 0.0% | 0.0% |
| `acq_m1` | -1 | 450 | 30 | 6.6% | 0.0% | 0.0% |
| `acq_m3` | -3 | 450 | 84 | 18.7% | 17.2% | 0.0% |
| `acq_m4` | -4 | 450 | 160 | 35.7% | 41.7% | 0.0% |
| `acq_m5` | -5 | 450 | 251 | 55.8% | 57.8% | 0.0% |
| `acq_m6` | -6 | 450 | 269 | 59.8% | 66.1% | 0.0% |
| `acq_p2` | 2 | 450 | 8 | 1.8% | 0.0% | 0.0% |

**COMPRESSED TAIL: `acq_m5`, `acq_m6` exceed the 50% bar.** Their deep readings are a lower bound on the offset's value, not a measurement of it.

## Speed — clicks to the answer the control ends its session with

| arm | k | median CTT @100 | never reached | median CTT @400 | never reached |
|---|---:|---:|---:|---:|---:|
| `prod` | 0 | 50.0 | 0% | 153.5 | 0% |
| `acq_m1` | -1 | 28.0 | 18% | 82.0 | 7% |
| `acq_m3` | -3 | 26.0 | 7% | 53.0 | 3% |
| `acq_m4` | -4 | 22.0 | 9% | 43.5 | 7% |
| `acq_m5` | -5 | 23.0 | 10% | 45.5 | 9% |
| `acq_m6` | -6 | 21.0 | 13% | 42.5 | 17% |
| `acq_p2` | 2 | 18.0 | 70% | 160.5 | 77% |

`prod` reaches its own target in 100% of cells **by construction**; a miss
rate elsewhere is a real outcome and is why the median is reported beside it.

## H3 — does the plateau replicate at t=100?

The anchor connecting this pile to the shipped constant. If it fails, the
dataset change did more than add depth and **every** deep reading below is
about the dataset rather than the horizon.

| arm | k | Δ cost vs `prod` @100 [95% CI] | Δ cost @400 [95% CI] | Δ positives @400 | Δ AP @400 |
|---|---:|---|---|---:|---:|
| `acq_m1` | -1 | -0.0189 [-0.0254, -0.0122] | -0.0250 [-0.0307, -0.0194] | +11.8 | +0.044 |
| `acq_m3` | -3 | -0.0375 [-0.0446, -0.0304] | -0.0471 [-0.0536, -0.0406] | +102.6 | +0.100 |
| `acq_m4` | -4 | -0.0385 [-0.0470, -0.0301] | -0.0421 [-0.0487, -0.0355] | +160.8 | +0.107 |
| `acq_m5` | -5 | -0.0300 [-0.0390, -0.0210] | -0.0340 [-0.0401, -0.0277] | +196.5 | +0.105 |
| `acq_m6` | -6 | -0.0225 [-0.0318, -0.0131] | -0.0263 [-0.0331, -0.0196] | +219.1 | +0.107 |
| `acq_p2` | 2 | +0.0728 [+0.0616, +0.0841] | +0.0656 [+0.0583, +0.0730] | -11.2 | -0.076 |

## The falsification arm

`acq_p2` (k=+2) vs `prod`, positives @400: **-11.2** [-11.9, -10.5] over 192 pairs.

**BEHAVED** — the lever is a lever.

## H1 — does the optimum move DEEPER through the session?

`DiD = [m(deep,400) − m(deep,100)] − [m(shallow,400) − m(shallow,100)]`, paired within the cell, against `acq_m3`.

A **negative** DiD on cost (or on AUC, or on clicks-to-target) means the
deeper arm gains ground as the session runs on — the optimum moves deeper,
as the likelihood-ratio reading predicts. A **positive** one is #2910's
reading: the benefit fades where scarcity ends.

| deep arm | k | metric | DiD [95% CI] | pairs | tail | reading |
|---|---:|---|---|---:|---|---|
| `acq_m4` | -4 | cost | +0.0060 [-0.0016, +0.0134] | 192 | clean | no move |
| `acq_m4` | -4 | auc | +0.0031 [-0.0013, +0.0073] | 192 | clean | no move |
| `acq_m4` | -4 | ctt | +2.8 [-5.9, +11.8] | 159 | clean | no move |
| `acq_m5` | -5 | cost | +0.0057 [-0.0022, +0.0133] | 192 | **compressed** | no move |
| `acq_m5` | -5 | auc | +0.0082 [+0.0040, +0.0123] | 192 | **compressed** | **shallower** ✗ |
| `acq_m5` | -5 | ctt | +12.9 [+1.5, +25.5] | 157 | **compressed** | **shallower** ✗ |
| `acq_m6` | -6 | cost | +0.0059 [-0.0029, +0.0147] | 192 | **compressed** | no move |
| `acq_m6` | -6 | auc | +0.0106 [+0.0064, +0.0146] | 192 | **compressed** | **shallower** ✗ |
| `acq_m6` | -6 | ctt | +8.4 [-3.3, +20.1] | 143 | **compressed** | no move |

**H1 NOT SUPPORTED and NOT FALSIFIED** — the optimum does not move on the range this grid covers. That retires the question rather than deferring it.

## H2 — is the deep guardrail aggression, or exhaustion?

#3319 measured `acq_m3`'s deep-spike incidence at **0.5% → 5.7% (p=0.006)**
between 100 and 400 clicks, on a pile where that arm had consumed 82% of its
positives. #2790 traced deep spikes to positive **starvation**. If that was
the mechanism, the incidence should collapse here, where harvest is far lower
at the same aggression.

| arm | k | deep spikes @100 | deep spikes @400 | median harvest @400 |
|---|---:|---:|---:|---:|
| `prod` | 0 | 0.5% | 0.5% | 4.4% |
| `acq_m1` | -1 | 0.5% | 0.5% | 6.6% |
| `acq_m3` | -3 | 1.0% | 1.0% | 18.7% |
| `acq_m4` | -4 | 2.1% | 2.1% | 35.7% |
| `acq_m5` | -5 | 1.0% | 1.0% | 55.8% |
| `acq_m6` | -6 | 2.6% | 2.6% | 59.8% |
| `acq_p2` | 2 | 7.3% | 7.3% | 1.8% |

**Power, stated rather than discovered.** At a 0.5% baseline 192 cells expects
~1 event, so this table cannot rank arms against each other for safety — which
is exactly why #3319 recorded a hazard rather than evidence. It CAN resolve
H2's contrast, which is large: 5.7% against ≤1% is ~11 events against ~2.
