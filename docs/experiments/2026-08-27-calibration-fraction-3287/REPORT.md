# What Train/Calibrate split should a detector use? (#3287)

Of the votes a user has cast, what share should **train** each calibration fold's
model, and what share should be held out to **read that fold's threshold**?
`calibration_fraction` decides it, it has been `0.5` since it was introduced, and
it had never been measured — the obvious default, not a result. It sits on the
shipped threshold path, so it is priced on every detector anyone trains.

Design and pre-registered decision rules: [`PLAN.md`](PLAN.md). Every number here
comes from [`agg/`](agg/), written by
[`analyze_calfrac.py`](../../../scripts/experiments/calibration/analyze_calfrac.py).

## The answer, in five lines

- **On the shipped default configuration (`siglip`, whole-image/binary voting), 0.5
  is not optimal.** Spending more of the votes on Train is worth **−0.013 ± 0.003**
  in cost at 0.4, and it wins in *every* vote band — there is no regime where it
  costs anything.
- **That result does not generalise, and the reason matters:** it is a property of
  the **embedder**, not of the voting mode. On `dinov3_patch` — in *either* voting
  mode — nothing beats 0.5, and 0.3 is significantly *worse*.
- **So the per-mode default the issue proposed is not what the data supports.** A
  `PRODUCTION_SPLIT_BY_MODE` written from the per-mode table would be reading an
  average across a disagreement.
- **A follow-up run settles "is it SigLIP specifically?" — no.** `siglip2_l` behaves
  like `siglip`: 0.3 beats 0.5 by **−0.013 ± 0.003**, winning in every band. Two of
  two single-vector arms want less than 0.5; `dinov3_patch` in two of two styles
  does not. [Below](#follow-up-siglip2_l-behaves-like-siglip-not-like-dinov3).
- **A second follow-up settles "is it the SigLIP *family*?" — also no.** Two CLIP
  checkpoints, a different lineage entirely, both want 0.3: **−0.011 ± 0.003**
  (ViT-B/32) and **−0.016 ± 0.003** (ViT-L/14), negative in every band. Running two
  capacities is what rules out the alternative reading — the effect is not the
  family, not the vector width, and not the encoder's size.
  [Below](#follow-up-clip-is-not-the-family-not-the-width-not-the-capacity).

No production change is proposed here. This is the evidence; the decision is the
owner's.

## What was run

Five **full runs**, one per fraction, not five paired arms: the fraction sets the
threshold, the threshold sets the acquisition cut, and the cut sets what Autopilot
samples next — so an arm at 0.3 has collected different votes by its second
trained step. Everything except the fraction is the app's own behaviour: the fused
threshold path, the production linear-SVM head, `calibrate_count=2`, the per-mode
blend schedule, and the text-sort opening a user gets by typing a query.

| | |
|---|---|
| dataset | `vg_scale_any` — 12 hand-checked classes, 300 positives each against one shared 3900-image negative pool, so **prevalence is identical in every cell** |
| geometries | `siglip/whole_image`, `dinov3_patch/whole_image`, `dinov3_patch/max_patch` |
| seeds / steps | 4 seeds, 150 clicks |
| grid | 5 arms × 96 cells = **480 cells, 480 COMPLETED, 0 failures** |
| frame | 21,177 rows per arm; **0 unreadable, 0 zero-byte, 0 starved cells** |

Uniform prevalence is the instrument, not a nicety: a threshold **is** a quantile
of the calibration set, and this study's subject is how big that set should be, so
a grid whose calibration sets differed 60-fold in size would confound the swept
axis with itself.

The third geometry is what makes the headline readable. `siglip/whole` vs
`dinov3/whole` is the **embedder** at fixed voting mode; `dinov3/whole` vs
`dinov3/max_patch` is the **voting mode** at fixed embedder. Without that corner,
mode and embedder move together and a per-mode claim cannot be distinguished from
a per-embedder one — which is exactly the distinction that turned out to matter.

The knob is doing what it says: `n_cal_scores` on the production row comes out at
`2 × round(t × fraction)` — 90, 150 and 210 held-out scores at t=150 for 0.3, 0.5
and 0.7 — so the sweep really is resizing the set the conformal quantile is taken
over.

## The result, per geometry

Paired within `(dataset, category, seed, geometry)`; **cost**, pooled
inverse-variance across bands, bootstrapped over **cells** (never steps —
consecutive steps of one trajectory share a model).

| geometry | 0.3 | 0.4 | 0.6 | 0.7 |
|---|---|---|---|---|
| `siglip/whole_image` | **−0.012 ± 0.003** | **−0.013 ± 0.003** | +0.003 ± 0.003 | +0.008 ± 0.003 |
| `dinov3_patch/whole_image` | +0.015 ± 0.005 | −0.001 ± 0.004 | +0.005 ± 0.005 | +0.020 ± 0.005 |
| `dinov3_patch/max_patch` | −0.002 ± 0.002 | −0.001 ± 0.002 | −0.001 ± 0.003 | +0.008 ± 0.003 |

Negative is better than 0.5. **Bold** = resolved at more than 2 SE *and* not worse
than 0.5 in any band.

`regret_honest`, against the cross-fitted reference, says the same thing
independently — `siglip/whole_image` at 0.3 (−0.0093 ± 0.0015) and 0.4
(−0.0068 ± 0.0015) are the only arms that beat the incumbent on both metrics.

Absolute deep-band levels, for scale:

| geometry | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 |
|---|---|---|---|---|---|
| `siglip/whole_image` | 0.30 | 0.31 | 0.31 | 0.31 | 0.31 |
| `dinov3_patch/whole_image` | 0.39 | 0.37 | 0.36 | 0.37 | 0.37 |
| `dinov3_patch/max_patch` | 0.23 | 0.23 | 0.22 | 0.22 | 0.23 |

## Why this is not a per-mode result

The per-mode table reports "binary: 0.4 is a candidate, −0.0085 ± 0.0024". That row
is **entirely `siglip`'s**. The other binary geometry disagrees, and at 0.3 it
disagrees in sign: −0.012 ± 0.003 on `siglip/whole_image` against **+0.015 ± 0.005**
on `dinov3_patch/whole_image`. Pooling them produces a number that describes
neither.

Compare the two legs directly:

- **embedder, at fixed voting mode** (`siglip/whole` vs `dinov3/whole`): the 0.3 arm
  moves by 0.027 in cost, and flips sign.
- **voting mode, at fixed embedder** (`dinov3/whole` vs `dinov3/max_patch`): the same
  arm moves by 0.017, and both of the two are within 2 SE of the incumbent at 0.4
  and 0.6.

The embedder leg is the larger one, and it is the one carrying the only resolved
effect in the study. #3115 reached the same shape one level up — what looked like a
law about voting mode was two cells that happened to agree — and this run is that
lesson applied before the claim rather than after it.

That is also why the study is not reported as "binary wants 0.4". It is reported as
"the configuration most users are actually in wants less than 0.5, and we cannot yet
say what the knob follows."

## Across vote bands

Cost, paired vs 0.5, per band:

| geometry | band | 0.3 | 0.4 | 0.6 | 0.7 |
|---|---|---|---|---|---|
| `siglip/whole_image` | early 1–25 | −0.018 | −0.018 | +0.006 | +0.020 |
| | mid 26–60 | −0.007 | −0.013 | +0.011 | +0.011 |
| | late 61–100 | −0.009 | −0.011 | +0.001 | +0.001 |
| | deep 101–150 | −0.013 | −0.009 | −0.003 | −0.002 |
| `dinov3_patch/whole_image` | early 1–25 | −0.006 | −0.010 | +0.004 | +0.031 |
| | mid 26–60 | +0.019 | −0.002 | +0.010 | +0.026 |
| | late 61–100 | +0.019 | −0.004 | +0.002 | +0.019 |
| | deep 101–150 | +0.028 | +0.010 | +0.004 | +0.011 |
| `dinov3_patch/max_patch` | early 1–25 | −0.012 | −0.009 | −0.007 | +0.000 |
| | mid 26–60 | −0.005 | −0.004 | +0.006 | +0.008 |
| | late 61–100 | −0.003 | −0.002 | −0.003 | +0.010 |
| | deep 101–150 | +0.004 | +0.002 | +0.001 | +0.010 |

The predicted shape is there, and it is clearest on `max_patch`: **more Train is
worth most when votes are scarce** (−0.012 at 1–25) and the advantage decays to
nothing, then reverses, by 150 clicks (+0.004). That is the trade-off the knob sits
on, visible directly: early, the fold models are the scarce thing; late, the
quantile's resolution is.

`siglip/whole_image` is the exception that makes the decision easy — it is negative
in *every* band, so the gain there is not bought from a regime someone else lives
in.

## Where the pointwise gate decided nothing

The pre-registered rule refuses an arm that is worse than the incumbent by more
than 0.01 in any band. Three arms landed on that line rather than either side of
it, and the analyzer now says so rather than printing a boolean:

| mode | fraction | worst band | Δ | SE | margin vs 0.01 |
|---|---|---|---|---|---|
| binary | 0.4 | deep 101–150 | 0.00998 | 0.0069 | **−0.00002** |
| binary | 0.6 | mid 26–60 | 0.01051 | 0.0074 | +0.00051 |
| region | 0.7 | late 61–100 | 0.01018 | 0.0049 | +0.00018 |

Binary 0.4 "passes" the harm gate by **2 × 10⁻⁵**, a margin roughly four hundred
times smaller than its own standard error. That is not a pass; it is a coin
landing on its edge. It does not change the conclusion — 0.4's case rests on
`siglip/whole_image`, where the worst band is *negative* (−0.0089) and the gate is
not close — but a report that printed `candidate=True` and moved on would have been
claiming a decision the data did not make.

## Threshold stability

`sd(threshold)` across the 4 seeds at a fixed (category, step), averaged per band,
ranges 0.014–0.032 with **no consistent ordering in the fraction**. The one hint is
in `max_patch`'s deep band, where it rises with the fraction (0.023 at 0.3 →
0.032 at 0.7) — the opposite of the "more calibration data ⇒ a steadier quantile"
intuition — but it is one band in one geometry and is not resolved. Full table:
[`agg/sd_threshold.csv`](agg/sd_threshold.csv).

## The decomposition trap, measured

The issue warned that `rule_inefficiency` and `calibration_shift` are not
independent effects of this knob, because `calibration_shift` is measured against a
`cal_oracle_cost` estimated *from the calibration set the knob resizes*. That is
exactly what the data shows, and the size of it is worth recording:

Across all 12 (geometry × band) cells:

| | |
|---|---|
| corr(`rule_inefficiency`, `calibration_shift`) across arms | **−0.60 to −0.999**, and below −0.94 in 8 of 12 |
| \|sum − `regret`\| | **< 4×10⁻⁸** (identically zero) |
| spread of `rule_inefficiency` across arms | 0.013 – 0.135 |
| spread of `calibration_shift` across arms | 0.013 – 0.097 |
| spread of `regret` across arms | 0.005 – 0.038 |

The two decomposed terms move **1.3–8.2× more than the quantity they decompose**,
in opposite directions, with their sum pinned by construction. Anyone reading them
per-term would have reported effects several times the real one and given them a
mechanism. #2897 did exactly that. No per-term claim is made here; levels are read
off `regret` / `regret_honest`, which are referenced to something the knob does not
move. Table: [`agg/trap_check.csv`](agg/trap_check.csv).

## Figures

![cost over clicks](figures/cost_vs_clicks.png)

One panel per **geometry** (there is one dataset, and `max_patch` must never be
averaged with `whole_image`); colour is the fraction; the band is the inter-quartile
range over cells. **Click 0 is the free text sort** — what typing the query was
worth before any clicking — so the distance from the left marker to the right end is
what the clicking bought. A line is dashed wherever it describes fewer than 95% of
that arm's cells; only a solid segment is a level worth quoting. It does *not*
license comparing across panels: the three geometries have different absolute costs
for reasons that have nothing to do with this knob.

![average precision over clicks](figures/average_precision_vs_clicks.png)

Per-run versions — every seed as its own line, one file per geometry — are in
[`figures/`](figures/); the spread there is routinely the finding, and two arms with
the same mean can be "every run is mediocre" and "half are excellent".

**Every other slice — per category, per seed, other metrics — is in the interactive
viewer: [`viewer.html`](viewer.html).**

## What this does not license

- **It is one dataset.** `vg_scale_any` was chosen because uniform prevalence is
  what makes a calibration question answerable, and that same choice means the
  result has not been seen at any other prevalence.
- **The embedder finding rests on two embedders**, one of which (`dinov3_patch`)
  has no text tower and reaches its opening through the `siglip+dinov3_patch` pair.
  "The optimum follows the embedder" is the most economical description of these
  three geometries, not an established law — it is the same over-generalisation
  this report criticises the per-mode reading for, one axis over.
- **Nothing here was run at a prevalence, horizon or class count a real user picks.**
  150 clicks is a long session.

## Follow-up: `siglip2_l` behaves like `siglip`, not like `dinov3`

Run immediately after the main grid, identical except for the embedder:
`vg_scale_any × siglip2_l`, whole-image, same 12 classes, 4 seeds, 150 clicks, 5
fractions. **240/240 cells, zero failures, 7,051 rows per arm, nothing dropped.**
Artifacts in [`siglip2l/`](siglip2l/), viewer at
[`siglip2l/viewer.html`](siglip2l/viewer.html).

| metric | 0.3 | 0.4 | 0.6 | 0.7 |
|---|---|---|---|---|
| cost | **−0.013 ± 0.003** | −0.005 ± 0.003 | +0.006 ± 0.003 | +0.006 ± 0.003 |
| `regret_honest` | **−0.007 ± 0.002** | −0.001 ± 0.002 | +0.006 ± 0.002 | +0.012 ± 0.002 |

0.3 beats 0.5 on both metrics and its worst band is **negative** (−0.0090) — it wins
at every vote band, exactly as `siglip`'s 0.3 and 0.4 arms do:

| band | 0.3 | 0.4 | 0.6 | 0.7 |
|---|---|---|---|---|
| early 1–25 | −0.018 | −0.010 | +0.005 | +0.019 |
| mid 26–60 | −0.018 | −0.013 | −0.006 | −0.011 |
| late 61–100 | −0.011 | −0.001 | +0.007 | −0.001 |
| deep 101–150 | −0.009 | −0.001 | +0.013 | +0.008 |

**So "SigLIP specifically" is refuted.** Two of two single-vector arms want
materially less than 0.5; `dinov3_patch` in two of two styles does not. The optimum
is a property of the representation the detector learns in — not of the voting mode,
and not of one model.

### What this still cannot separate

The two arms that want 0.3 are also exactly the two where the **opening and the
learning happen in the same space**. `siglip` and `siglip2_l` rank their own text
sort and learn in that same space; both `dinov3_patch` arms reach their opening
through the `siglip+dinov3_patch` pair, so they open in SigLIP space and learn in
DINOv3 space. Embedder and *space-match* are therefore confounded in this design,
and a mismatch plausibly changes which items get voted on — which is the labelset
the split then divides.

The discriminator is cheap and specific: a **`siglip+siglip2_l` pair** — open in
SigLIP, learn in `siglip2_l`. If it still wants 0.3 the mismatch is not the driver
and the effect belongs to the embedder; if it moves to 0.5 the story is about
opening-vs-learning space and not about the embedder at all. That is another
single-vector grid: ~12 minutes.

Until that runs, the defensible statement is: **the shipped default is measurably
not optimal for either single-vector embedder in the pile, the effect is not about
voting mode, and one alternative explanation remains open.**

## Follow-up: CLIP is not the family, not the width, not the capacity

`siglip2_l` refuted "SigLIP specifically", but `siglip` and `siglip2_l` are the same
lineage, so what the study had established was closer to *"the SigLIP family wants
0.3"* than *"single-vector embedders want 0.3"* — and #3290 proposes to ship the
second. CLIP is the cheapest way to tell them apart: single-vector and
language-aligned like SigLIP, but a genuinely different model lineage — OpenAI's
softmax/InfoNCE contrastive objective against SigLIP's pairwise sigmoid loss,
different data, different recipe.

**Two CLIP checkpoints were run, not one.** Leaving SigLIP changes the family *and*
the capacity at once, which is the confound the pile's own `siglip → siglip2_l` note
warns about and that #3115 turned into a wrong law. Two capacities of one lineage
separate them:

| arm | checkpoint | encoder | dim |
|---|---|---|---|
| `clip` | `openai/clip-vit-base-patch32` | ViT-B/32 | 512 |
| `clip_l` | `openai/clip-vit-large-patch14` | ViT-L/14 | **768 — matches `siglip`** |

`clip_l` is dimension-matched to `siglip` on purpose, so a difference cannot be
"CLIP's vectors are narrower". Artifacts in [`clip/`](clip/) and
[`clip_l/`](clip_l/); viewers at [`clip/viewer.html`](clip/viewer.html) and
[`clip_l/viewer.html`](clip_l/viewer.html).

**480/480 cells across the two grids, zero failures, zero zero-byte outputs, 7,068
production rows in every one of the ten arms — nothing dropped.** The comparison to the
SigLIP rows is paired at the dataset level and was checked rather than assumed: all
five `vg_scale_any` cells — `siglip`, `siglip2_l`, `dinov3_patch`, `clip`, `clip_l` —
carry the **same 7,747 medias and the identical parent-label digest**
(`b973e7398e15…`), so the CLIP arms ran on the same images, labels, boxes and #3281
corrections as the arms they are being read beside. That is the check #3115 lost a
run to: a derived cell inherits its parent silently, and a `vg_scale_any` built after
a parent rebuild is not the same dataset as one built before it. Every metric row
opened on a typed query in the arm's own space (`seed_mode=text`,
`seed_embedder=clip`/`clip_l`, no blank queries), so neither arm fell back to the
known-good opening that would have made it incomparable to the SigLIP rows (#3278).

### The result

Cost and `regret_honest`, paired vs 0.5, pooled inverse-variance across bands:

| geometry | metric | 0.3 | 0.4 | 0.6 | 0.7 |
|---|---|---|---|---|---|
| `clip/whole_image` | cost | **−0.011 ± 0.003** | **−0.0087 ± 0.0031** | +0.0065 ± 0.0035 | +0.010 ± 0.0033 |
| `clip/whole_image` | `regret_honest` | **−0.0062 ± 0.0013** | **−0.0040 ± 0.0013** | +0.0029 ± 0.0014 | +0.0077 ± 0.0015 |
| `clip_l/whole_image` | cost | **−0.016 ± 0.003** | **−0.014 ± 0.003** | +0.0058 ± 0.0034 | +0.021 ± 0.0041 |
| `clip_l/whole_image` | `regret_honest` | **−0.0082 ± 0.0016** | **−0.0083 ± 0.0014** | +0.0075 ± 0.0017 | +0.0096 ± 0.0019 |

Negative is better than 0.5. **Bold** = resolved at more than 2 SE *and* not worse
than 0.5 in any band.

Per band, on cost — the shape is what matters, and it is monotone in both arms:

| band | `clip` 0.3 | 0.4 | 0.6 | 0.7 | `clip_l` 0.3 | 0.4 | 0.6 | 0.7 |
|---|---|---|---|---|---|---|---|---|
| early 1–25 | −0.012 | −0.010 | +0.011 | +0.014 | −0.021 | −0.014 | +0.010 | +0.033 |
| mid 26–60 | −0.014 | −0.0042 | +0.0036 | +0.0006 | −0.015 | −0.0094 | +0.0010 | +0.022 |
| late 61–100 | −0.013 | −0.011 | +0.0034 | +0.0054 | −0.013 | −0.012 | +0.0027 | +0.012 |
| deep 101–150 | −0.0083 | −0.0082 | +0.0034 | +0.0096 | −0.014 | −0.017 | +0.0029 | +0.011 |

**Every cell of that table has the same sign structure**: 0.3 and 0.4 below zero in
all eight band-arms, 0.6 and 0.7 above it in all eight. Individual band SEs run
0.0045–0.011, so most single cells are not resolved on their own — the finding is the
consistency, not any one of them. `clip_l`'s 0.3 arm is the exception that is also
resolved band by band, in all four.

Absolute deep-band cost, for scale — monotone increasing in the fraction, both arms:

| geometry | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 |
|---|---|---|---|---|---|
| `clip/whole_image` | 0.33 | 0.33 | 0.34 | 0.35 | 0.35 |
| `clip_l/whole_image` | 0.31 | 0.31 | 0.32 | 0.33 | 0.33 |

### Figures

![cost over clicks, clip](clip/figures/cost_vs_clicks.png)
![cost over clicks, clip_l](clip_l/figures/cost_vs_clicks.png)

One panel per arm; colour is the fraction; the band is the inter-quartile range over
cells. **Click 0 is the free text sort** — 0.45 for `clip`, 0.47 for `clip_l` — so the
drop from the left marker to the right end is what clicking bought, and the ordering
of the five lines at the right end is this study's whole subject. Dashed where fewer
than 95% of that arm's cells are measured; coverage reaches 100% by click 4 and stays
there, so every level quoted above is on solid segments. Do **not**
read across the two panels: `clip` and `clip_l` have different absolute costs for
reasons that have nothing to do with this knob.

Every arm beats the free text sort, and the cheaper splits get there sooner and end
lower — the crossover click, and the deep-band level it reaches:

| arm | `clip` crossover | final cost | `clip_l` crossover | final cost |
|---|---|---|---|---|
| 0.3 | click 16 | 0.32 | click 13 | 0.30 |
| 0.4 | click 14 | 0.33 | click 13 | 0.30 |
| 0.5 | click 20 | 0.33 | click 14 | 0.31 |
| 0.6 | click 20 | 0.34 | click 16 | 0.32 |
| 0.7 | click 18 | 0.35 | click 19 | 0.32 |

Per-run versions — every seed its own line — are in [`clip/figures/`](clip/figures/)
and [`clip_l/figures/`](clip_l/figures/), and `average_precision_vs_clicks.png` beside
each. **Every other slice is in the interactive viewers**, linked above.

### What it settles

The issue pre-registered the reading, and the answer is the first row:

> **CLIP wants 0.3** → "single-vector ⇒ 0.3" survives a change of family; #3290's
> gate is on much firmer ground and can ship as written.

It survives it **twice, at two capacities and two vector widths**, which is more than
was asked for and rules out the three alternative explanations that a single CLIP arm
would have left open:

- **not the family** — a different lineage, objective, data and recipe still wants 0.3;
- **not the vector width** — 512-d and 768-d agree, and the 768-d arm is the *stronger*
  of the two, so the effect does not track dimensionality;
- **not the capacity** — ViT-B/32 and ViT-L/14 agree, so it is not an artifact of
  small or large encoders.

Four of four single-vector arms now want materially less than 0.5. `dinov3_patch`, in
two of two styles, does not.

### What it does not settle

**The space-match confound is untouched, and CLIP makes the count worse rather than
better.** Like `siglip` and `siglip2_l`, both CLIP arms have a text tower and open on
their own text sort, so their opening and their learning happen in the *same* space;
both `dinov3_patch` arms open through the `siglip+dinov3_patch` pair and learn
elsewhere. So the split is now **four space-matched arms wanting 0.3 against two
space-mismatched arms wanting 0.5** — and "space-match" still explains all six
geometries exactly as well as "single-vector" does. Adding a fourth arm on the same
side of that line cannot break it; only an arm that crosses it can.

The discriminator remains the one the main report already names: a **`siglip+siglip2_l`
pair** — open in SigLIP, learn in `siglip2_l` — which is single-vector *and*
space-mismatched, the one combination the pile has never run. It is another ~20-minute
grid, and it is now the single highest-value cell in this design. **Tracked in #3559.**

**So the honest statement of #3290's status is:** its constant is right for every
configuration measured, and the predicate it is written on (`is_patch_embedder`) is
still not known to be the predicate doing the work.

### Keeping the arm out of production

`clip_l` is a research arm and nothing has evaluated it *for the app*. `MediaEmbedder`
therefore gained an `eval_only` property, and the two app-facing enumerations —
`embedders_for_type` (every picker and every per-media-type default) and
`all_embedders_dict` (what `GET /api/embedders` serialises) — withhold it. Resolution
by name stays open, or a pile cell embedded by an eval arm could not load.

A note on the issue's premise: it assumed no CLIP existed in the tree. `clip`
(`clip-vit-base-patch32`) has shipped for a while and *is* app-selectable, so the
"keep it out of the picker" instruction was written about a state that no longer
held. That embedder is left exactly as it is — removing a shipped choice is a
production decision, not an experiment's to make — and only the new `clip_l` is
marked eval-only.

## Reproducing

```bash
cd /exp/$USER/projects/vts-calfrac-3287/scripts/experiments/calibration
python selftest_analyze_calfrac.py        # planted answer, before the array
bash launch_calfrac_3287.sh prepare       # once, shared by every arm
bash launch_calfrac_3287.sh baseline      # the click-0 anchor
bash launch_calfrac_3287.sh arms          # 5 arrays + one cross-arm analyze
```

Measured over all 480 cells (the distribution, not the first few — quoting an
early sample is how #3129 produced a 90-minute error):

| cell | n | elapsed mean | elapsed range | RSS mean | RSS max |
|---|---|---|---|---|---|
| binary (`siglip/whole_image`) | 240 | 3.5 min | 1.9 – 5.4 | 0.71 GB | 0.94 GB |
| region (`dinov3`, both styles) | 240 | 21.8 min | 10.4 – 41.1 | 5.33 GB | **7.71 GB** |

12G per task is not negotiable down — preflight check 7b enforces it, and the
7.71 GB peak is why. Sizing from a binary cell would have picked a limit eight
times too small. Five arms at `%16` each (80 concurrent × 12G = 89% of the
per-user allowance) drained in **3h 43m**, 23:24 → 03:07.

The `siglip2_l` follow-up is a different sizing problem, and the launcher now derives
it rather than being told: with no patch cells the grid is **CPU-bound, not
memory-bound** (0.94 GB peak measured, so 4G is 4× headroom), which lifts concurrency
from 80 to the QOS's CPU cap of 120. 240 cells at ~3.1 min drained in **~12 minutes**,
launch to report. The same test removes `--patch`, `--require-region-voting` and
`--contrasts-voting-modes` from its preflight — the last *should* fail on a
single-vector grid, since nothing is in both voting modes, and a check that cannot
apply is a claim in the run's record that nobody verified.

### The CLIP follow-up's sizing

```bash
cd /exp/$USER/projects/vts-clip-3292/scripts/experiments/calibration
bash run_3292.sh chain     # pile build + both grids, one SLURM chain
bash run_3292.sh status    # progress, read off files rather than a live process
```

Chained on `afterok` rather than driven from a terminal, because every laptop-side
watcher this project has run has eventually died with the VPN. Measured over all 480
array cells:

| | |
|---|---|
| pile build (`vg_scale` + derived `vg_scale_any`, both encoders, one v100) | 8m 24s |
| per cell | **3.1 min** (1.8 – 3.9) |
| RSS | **0.64 GB mean, 1.16 GB max** |
| per study, prepare → report | ~19 min |
| end to end, both studies | **47 min** (07:13 → 08:00) |

**The memory sizing was wrong in the safe direction, and the measurement says so.**
This run asked for 8G per task rather than the launcher's single-vector default of 4G,
reasoning that `run_cells.py` embeds the text query per cell so CLIP ViT-L/14's weights
would sit in every cell's RSS. The measured peak is **1.16 GB** — the 4G default had
4× headroom all along, and the concurrency halved to %12 bought nothing. Next
single-vector grid: take the default. The reasoning was sound and the number was not,
which is the argument for `size` over arithmetic.
