# The shared pre-embedded pile

A grid of `(dataset, embedder)` cells that every study reads instead of
embedding its own copy. One cell = one `<dataset>__<embedder>.pkl` of media
dicts carrying vectors (and `patch_grid` for patch embedders) but **no pixels**,
so the pile stays small relative to its sources.

```bash
source scripts/experiments/pile/pile_env.sh   # point a study at it
python build_pile.py --list                   # what exists
python build_pile.py                          # build whatever is missing
python build_pile.py --verify                 # check every cell is usable
python build_pile.py --rebuildable            # check every cell could be REBUILT
python build_pile.py --bands                  # voted-box scale bands (boxed datasets)
```

**Rebuilding a cell that already exists is a different operation from filling a
gap**, and it goes through the launcher so it keeps the rebuild canary and the
`ATEN_CPU_CAPABILITY` pin — a cell rebuilt without the pin no longer matches its
own fingerprint (#3160):

```bash
VTS_BUILD_ARGS=--force VTS_GPU_NODE=<node> bash launch_pile.sh vg_scale
```

Pin the node from the cell's own provenance (`--provenance`). Two things to know
before you do it, both learned in #3667:

- **A rebuild is from `dev`, not from the commit that built the cell.** It picks
  up every `pile_config` ruling merged in between, so the membership can move
  even when you changed nothing relevant. `vg_scale`'s September rebuild dropped
  41 positives and 40 came in, across five merged rulings.
- **Copy the cells first, and check your glob.** `vg_scale__*` does not match
  `vg_scale_deep__*`.

## Where the code lives

`build_pile.py` is the CLI and the per-cell build loop; everything it does
*inside* a build lives in `pilebuild/`, one module per question:

| Module | Answers |
|---|---|
| `pilebuild/loaders/<kind>.py` | how a `DATASETS[ds]["kind"]` is built (`load`) **and** what a rebuild of it reads (`check`) |
| `pilebuild/vgsource.py`, `boxscan.py` | reading the VG source; choosing a band's categories from the box scan |
| `pilebuild/corrections.py` | human verdicts, and the one place their boxes cross from normalised into pixel space |
| `pilebuild/geometry.py` | geometry no honest region box can have; the derived-label digest |
| `pilebuild/provenance.py` | which machine produced a cell, and its vector hash |
| `pilebuild/audit.py`, `manifest.py`, `provenance_report.py` | the read-only modes |

**Both halves of a dataset live in one module on purpose.** They used to be two
`kind` switches a thousand lines apart, and the drift that invites is #3299: the
rebuild canary checked `COCO_IMAGES` while the builder opened `val2017.zip`
inline, and reported `coco_val` REBUILD-BROKEN against a staging area that was
entirely intact. A new dataset kind therefore adds one module carrying both, and
a kind with no module fails at dispatch instead of falling through to the demo
loader. `tests_lib/meta/test_pile_loaders.py` pins that.

The `vg_scale` build is eight named passes rather than one long function, because
two of them are where this pile's expensive bugs have lived — `apply_corrections`
(the single normalised→pixel crossing, #3281) and `designate_cells` (whether a
rebuild keeps the images a human reviewed). Both are ordinary functions taking
what they read and returning what they produce, so
`tests_lib/meta/test_pile_vg_scale.py` exercises them without the VG source.

**VG's vocabulary is free text, and the read matches an object's primary name
only** — so a class is built from one spelling out of several, and on the ~52% of
VG that COCO does not annotate the others become *negatives* for their own class,
because there VG's silence is the only evidence of absence. `bicycle` shipped
that way: the VG name `bike` carries 638 of COCO's 3,683 `bicycle` boxes against
the `bicycle` spelling's 775 (#3605). There is no cheaper fix in the reader —
every one of VG's 2,516,939 objects carries a `names` list of length **one**, so
there is no synonym to read instead (#3618).

Two config tables decide what happens to a spelling:

| table | meaning | effect (`vg_scale.py`) |
|---|---|---|
| `SCALE_VG_NAMES` | the name is the class, **and its box is the object** | `canonicalise` folds the boxes onto the class name |
| `SCALE_VG_AMBIGUOUS` | the class may be present; this box cannot be its positive | `lift_ambiguous` withholds the image from the class's bands **and** from the shared negative pool |

Suppression applies only where the spelling is the last word: an image COCO
annotates, or one a reviewer has ruled on, already answers the question. That is
why `lift_ambiguous` runs after `anchor_to_coco` and `apply_corrections`.

**Both tables are measured, and by the test that matches what the entry claims**
(#3618). Which table a name goes in is *derived* by `name_evidence.py` from two
numbers, not drafted:

| number | what it decides | how it is measured |
|---|---|---|
| **repair precision** | act on this name at all | over the VG∩COCO overlap, take the images carrying the name and **not** the class name — the state that becomes a false negative on the other half — and ask COCO whether the class is present. Read it as a price, in the units of #3635: `1 / precision - 1` is **good hard negatives destroyed per contaminated negative retired** — not images withheld from the pool, which is 18x over-subscribed. Cut at 1/3, on the **Wilson lower bound**. |
| **box agreement** | fold it, or only withhold it | the share of the name's boxes landing on a COCO box of the class (`coco_folds.py`'s fold-in, per name). Cut at 0.5 over ≥ 20 boxes, because folding claims *every* box under the name is the object and a band is a claim about one object's size (#3616). |

The candidates come from two searches, and neither finds what the other does:
`coco_folds.py` finds names that land on COCO's boxes, and `vg_name_families.py`
enumerates a class's **head-noun family** over the whole of VG, which is where a
spelling used mostly off the COCO half shows up. A family is a list of names to
*measure*, never a list to fold: the largest member of `dog`'s is **`hot dog`**,
405 images, which scores 0 of 181.

**Box overlap between two names (`scan_name_overlap.py`) is not the instrument
for this and cannot be.** It needs both names on one image, and an annotator who
writes `back pack` does not also write `backpack`: 846 of 1,740 candidate pairs
never co-occur, and `backpack`/`backpacks` scores **0.000 both ways**. It keeps
its own job — the case where two names really do sit on one box
(`clock`/`clock face`, 0.562/0.701) and refuting a lookalike, which is how `bus`
survived matching 80 images annotated `bush`.

**What a name withholds is never a random slice of the pool, and that is what
the price is really counting** (#3635). `pool_contamination.py` measures the
unconditioned question a name-conditioned rate cannot reach — *of the images that
would enter the shared negative pool on VG's evidence alone, what share actually
hold the class?* — by running the loader's own passes over the overlap with
`exhaustive=set()`, i.e. as if those images were off-COCO, and holding COCO back
as the answer key. It also prices the two exclusion rules against each other:
today one ambiguous name costs **all twelve** classes the image, though
`evaluable_categories` could make it cost one (#3655).

`withheld_difficulty.py` then asks whether the withheld images are the pool's
*hard* negatives, by ranking the drawn pool with the class's own text query. The
answer is yes — for **every** ambiguous name, which is why concentration cannot
discriminate between a good entry and a bad one and the ratio above must. `bike`
takes 17.9x its base rate of the top 50 and `sign` 6.8x; what separates them is
that `bike` destroys 31 good negatives to retire 30, and `sign` destroys 435 to
retire 37.

`name_coverage.py` then prices a proposed table before it ships: coverage against
COCO, images repaired and withheld on the non-COCO half, and the **band ledger** —
a fold merges boxes, so it can push an image the class already banded past the
scatter guard and out of every band (248 across *C*; `clock` nets −16).

**That is the right outcome, and the build now says so.** #3637 scored the three
readings of it against COCO's exhaustive boxes: on the images they disagree
about, the class really is scattered 88% of the time, and no cell is anywhere
near needing the images back. `canonicalise` reports the count on the same line
as the boxes folded, `SCALE_FOLD_MODE` selects the arm, and `band_fold.py`
re-measures it. Full study:
[`docs/experiments/2026-09-05-band-fold-3637/`](../../../docs/experiments/2026-09-05-band-fold-3637/REPORT.md).

`SCALE_VG_NAMES_AUDITED` records which classes have actually had this measured,
because "no spelling is listed" and "no spelling exists" are the same empty
table. All twelve as of #3618, listed one by one rather than derived from
`SCALE_CLASSES` so that a newly added class is not marked audited by arithmetic.
A build names the classes that have not been, since a rebuild is the moment the
fix is cheap. Full study:
[`docs/experiments/2026-09-04-vg-name-coverage/`](../../../docs/experiments/2026-09-04-vg-name-coverage/REPORT.md).

**A class's review rule is part of its definition, so it lives in the config
too.** A reviewer votes on bare images — the slates name files by image id
alone — which makes the dataset name the entire brief. For a class whose plain
English name does not settle the question, `pile_config.SCALE_CLASS_RULES`
carries both halves of the rule, and `review_name(cls, pass)` is what every
slate maker builds its `detector` column from:

| field | meaning |
|---|---|
| `name` | the few words the reviewer reads (`cell phone not landlines`) |
| `test` | what that abbreviates — the near-misses two words cannot settle |

Both are recorded because a name is not a definition. `book` split on the half
that was never written down: COCO annotates magazines as `book`, the human pass
read it narrowly, and 21 verdicts landed on one meaning against 49 on another.
`cell phone`'s first slate then split on the `test` rather than the `name` — it
read "anything with a cord or a base station is Bad", which discriminates on a
base being *present* when what it means is that the handset is not itself the
whole device, and so rejected a mobile phone sitting in a charging dock (#3612).
A class absent from the table is its own definition and keeps the bare class
name, which is what the manifests held before the table existed.

## Why this exists

Before it, each study embedded its own datadir and then later studies
symlinked back to whichever one happened to have the pair they needed — the
chain rooted at `max-patch/datadir`, an artifact named after a finished
experiment, split across two study dirs on a chronically full 50G mount.
Embedders got re-run because the cache had no home of its own.

## The grid

Eight datasets x five embedders, complete — 40 of 40 cells built as of
2026-08-28.

| embedder | dim | note |
|---|---:|---|
| `siglip` | 768 | the shipped default |
| `siglip2_l` | 1152 | the premium end |
| `dinov3_patch` | 768 | the only patch-capable one, so the only region-voting column |
| `clip` | 512 | a different pretraining *family*, at base capacity |
| `clip_l` | 768 | the same family at large capacity |

Differing dims mean galleries are **not** interchangeable across columns.

`siglip` -> `siglip2_l` moves *generation* (1 -> 2) and *capacity* (base ->
SO400M) together, so a difference between those two cannot be attributed to
either alone. That is what the CLIP columns are for: `clip`/`clip_l` change the
pretraining family at two capacities, which is the axis #3292 needed and could
not get from the SigLIP pair. The middle SigLIP columns (`siglip_l`, `siglip2`)
are still deliberately absent — a study learns little from interpolating
between endpoints — and `build_pile.py --embedders siglip2` rebuilds one if a
result ever needs that split.

| dataset | medias | boxed | note |
|---|---:|:--:|---|
| `visual_genome_m` | 4193 | yes | demo dataset; ground-truth regions |
| `caltech101_m` | 838 | no | demo dataset; whole-image labels only |
| `coco_val` | 4952 | yes | assembled from the staged val2017 zip |
| `vg_box_small` | 12000 | yes | box-banded VG: union box **below one patch** |
| `vg_box_medium` | 12000 | yes | box-banded VG: patch → HAC leaf |
| `vg_box_large` | 12000 | yes | box-banded VG: leaf → 80% of the image |
| `vg_scale` | 7747 | yes | one class list held fixed across every box-size band |
| `vg_scale_any` | 7747 | yes | derived from `vg_scale`, band collapsed away (#3115) |

The six `vg_box_* x {clip, clip_l}` cells were the last gap, and they were
unbuildable rather than merely unbuilt: band selection died before the embedder
was ever reached (#3297). They were built on 2026-08-28 once that was repaired.
Unlike their `siglip2_l` siblings from 2026-08-12 they carry the
`ATEN_CPU_CAPABILITY=avx2` pin, but no comparison rests on that: the CPU-dispatch
divergence #3160 measured is in the **384px** resize, and both CLIP columns are
224px models, where the resize is bit-identical either way.

### The box-banded VG sets

`vg_box_small/medium/large` exist because **`visual_genome_m`'s `_m` is a
dataset size tier, not a box size** — it is a `slice_frac` window over the
source, and `caltech101_m` (boxless) carries the same suffix. To vary box scale
you need datasets built for it.

They are drawn from the **whole** VG source — all 108k images across `VG_100K`
and `VG_100K_2`, with the full free-text vocabulary in `objects.json` — not the
demo pipeline's 100 curated categories on a 4% slice. That matters: the demo
vocabulary puts **5** categories in the sub-patch band; the full source has
**643**. A vocabulary chosen for recognisability is not a sample of scales.

Each band takes 40 categories, stratified *within* the band (support correlates
with size, so taking the best-supported would cluster them at one end and the
band would silently be a point), and up to 12000 images carrying them.
Categories are restricted to **concrete countable objects**: attributes (`red`),
frame relations (`front`), placeholders (`object`, `group`) and mass nouns /
unbounded surfaces (`sky`, `grass`, `floor`) are excluded by
`pile_config.is_object_category`, which matches on the **head noun** so
`blue sky` is dropped while `blue jeans` and `tennis ball` survive.

Rebuild the scan behind them with `python scan_vg_boxes.py` (writes
`vg_box_scale.json`; caches image dims, since `objects.json` stores boxes in
pixels and carries no image dimensions).

**Banding by median puts each category in exactly one band**, so these three
sets carry disjoint vocabularies and a small-vs-large difference confounds box
size with class identity. The scan therefore also emits each category's full
per-band histogram, and `shortlist_scale_classes.py` ranks the categories with
real support at *every* size — the input to a construction that holds the class
list fixed and varies only scale.

Supply alone does not qualify a class: `pile_config.scale_study_exclusion`
additionally rejects **parts** (a "small nose" is a distant face, and "no nose
here" is unverifiable wherever a person is), **places** (no principled box
extent), bare **polysemous** names, and **pervasive** classes. The shortlist
prints those with reasons rather than dropping them quietly. And
`scan_name_overlap.py` settles whether two names denote one object by box IoU
rather than by string similarity — the trap that made the benchmark's error
report match `bush` for `bus`. See
[`docs/plans/vg-scale-bands-and-corrections.md`](../../../docs/plans/vg-scale-bands-and-corrections.md).

Verified separation, measured with `--bands`: 38/40 of `vg_box_small`'s
categories fall in `sub_patch`, 40/40 of `vg_box_medium` in `patch_to_leaf`,
33/40 of `vg_box_large` in `leaf_to_4x`. The handful of strays are a
measurement difference — band membership was assigned on the full-VG median
voted area, while `--bands` recomputes it on the 12000-image sample.

## Region voting needs both halves

A region-voting arm drags a ground-truth box and pools it over a patch grid. It
therefore needs a **boxed dataset** *and* a **patch embedder**. Pair a boxed
dataset with a single-vector embedder and it does not error — it silently runs
as binary voting, because there is no `patch_grid` to pool and no
`patch_regions` to max-pool.

That mis-specification has cost three studies (#2877, #2897, #2905), so
capability is stated per *cell* (`pile_config.region_capable`) rather than per
dataset, and `--verify` asserts the geometry is physically present instead of
trusting the arm table. Only `dinov3_patch` is patch-capable, so the pile's
region-voting cells are `visual_genome_m x dinov3_patch` and
`coco_val x dinov3_patch` — deliberately two, so a region result can be
separated from the environment it was measured in.

COCO is built from the staged images rather than the #2790 vector cache, which
stores HAC region vectors but not the raw patch grid and so can never carry a
region arm.

## Boxes arrive in two spaces, and the file has to say which

VG's and COCO's boxes are in **pixels**. A correction box is the reviewer's
`region_box` from the app, already **normalised** to [0, 1]. The builder merges
all three and normalises on the way into the pickle, so a correction box merged
unconverted is normalised *twice*: divided by ~500 a second time and parked on
the frame origin. That is #3281 — 130 boxes, and with them 97 images filed into
`@small` whose object is medium or large, on the one axis `vg_scale` exists to
measure.

Three things now stop it, because none of them alone would have:

- `corrections.json` rows carry `box_space`, and `build_pile.py` refuses a row
  whose boxes contradict it. Inference cannot do this job: a normalised box and
  a pixel box are the same numbers for a box in the top-left corner of a 1×1
  image, which is precisely the shape the bug produced.
- The conversion happens **once**, against the same `(W, H)` the region write
  divides by, so the round trip is exact rather than close.
- `--verify` (and the build, before the GPU hours) checks boxes against the
  **frame**: a sub-pixel side is a failure outright, and the share crushed into
  the top-left 1% of the frame is a failure as a rate. The older check — box
  against the band its cell name claims — passed happily through all of this,
  because the band is *derived from* the box and moved with it. A consistency
  check between two values computed from one source is not a check.

`vg_scale_any` is a relabel of the built `vg_scale` pickle and shares its
vectors, so a parent rebuild used to leave it holding the parent's previous
labels with a perfectly healthy media count. It now stamps a digest of the
parent's labels, `--verify` compares that against the live parent, and a run
that rebuilds `vg_scale` pulls the derived dataset in with it.

## A rebox can change the band, and that is a correction

An image sits in `class@band` because of the box it arrived with, so a reviewer
who redraws that box onto a different, more prominent instance of the same class
moves it to another cell and vacates the one it was sampled to fill. Six of the
first thirteen redrawn boxes did, two of them leaving `small` (#3616).

The move is **kept**. VG's recall over *C* is 0.61, so an image holding a small
annotated bowl and a large unannotated one was banded by the only box anyone had
written down, and the reviewer is the first person to have seen the other one —
the sampled band was the error, not the redraw. What was wrong is that it
happened silently, so `verdicts_to_corrections.py` now prints every
band-changing rebox with the band it left and the band it lands in.

`audit_band_drift.py` asks how much of the same error the *un*-reviewed half is
still hiding, without spending a human on it. The COCO-anchored half has both
readings available — VG's boxes and COCO's exhaustive ones — so banding each
anchored image twice and counting the disagreements measures the rate directly,
and the roster says how many un-anchored seats that rate applies to:

```
python audit_band_drift.py                          # small + medium
python audit_band_drift.py --bands small --out drift_small.json
```

Only the bottom bands are audited by default: the defect can only push an image
*up* (a band can hide a larger instance), and `large` has nowhere to go. A
disagreement in the other direction is an extent error in VG's box, which is a
different problem and is counted separately.

## Voted-box scale bands (`--bands`)

Orthogonal to the `_s`/`_m`/`_l` suffix, which is a **dataset size tier** (a
`slice_frac` window over the source), *not* a box-size subset — `caltech101_m`
is boxless and still carries an `_m`.

Box size enters as a **category-selection** axis: `select_categories_by_scale`
in `../calibration/experiment_config.py` bins categories by the median area of
the box a Good vote drags (`category_scale_stats` in `vtscore/eval/labels.py`)
and takes 6 per band, preferring low `union_inflation` (categories that are one
clean object per image rather than scattered instances whose union box is far
bigger than anything a user would drag).

Band edges are anchored to the patch embedder's geometry, which is the point:

| band | range | meaning |
|---|---|---|
| `sub_patch` | 0 – 0.51% | below **one DINOv3 patch** (1/196) — unresolvable |
| `patch_to_leaf` | 0.51 – 8.33% | patch to smallest **HAC leaf** (1/12) |
| `leaf_to_4x` | 8.33 – 33.3% | a few leaves |
| `above_4x` | 33.3 – 101% | most of the image |

**On `visual_genome_m` and `coco_val`, `sub_patch` is starved and tuning cannot
fix it.** It holds 5 candidate categories on VG and 1 on COCO, unchanged at
every `min_count` from 5 to 30 — the filter is not the binding constraint, so
lowering it recovers nothing. Widening the band edge would inflate the count
with objects the grid *can* resolve, destroying what the band means.

The real cause is the **vocabulary**, not the band: those are the demo
pipeline's 100 curated categories (and COCO's 80 object-level classes, which
have no analogue for VG's part annotations like `eye`, `nose`, `cap`). Measured
against the full VG source the same band holds **643** categories. So the fix is
to use `vg_box_small` — built for exactly this — rather than to re-cut the band
on a dataset that was never sampled for scale.

## Rebuilding

Scratch is treated as purgeable, so every cell must rebuild from sources that
are not on scratch: demo datasets from the shared demo cache, COCO from the
staged zip plus flattened annotations. `build_pile.py` is idempotent — it skips
cells that exist, so it doubles as the resume path for a partial SLURM run.

**A demo cell will not build if its source is missing from the datadir.** The
downloaders read a missing extraction dir as "not downloaded yet" and refetch,
which once substituted a partial re-download and produced a healthy-looking
`visual_genome_m` cell holding 1662 of 4193 medias. `require_demo_source`
blocks that, and `--verify` cross-checks that a dataset's cells agree on media
count.

**`--verify` does not tell you the pile is rebuildable; `--rebuildable` does.**
The two paths share no code, so a cell can load perfectly while the code that
would produce it again is broken. That is not hypothetical: `scan_vg_boxes.py`
grew a `{"meta": …, "categories": …}` envelope on 2026-08-17, the scan file on
scratch stayed pre-envelope, and every `vg_box_*` rebuild died with
`KeyError: 'categories'` for eleven days behind a pile that verified clean
(#3297). `--rebuildable` runs each dataset's *selection* step — really choosing
`vg_box_*`'s categories, confirming everything else's sources are present and
readable — and embeds nothing, so it costs seconds. Run it after changing
anything a build reads, and before trusting scratch to be purgeable.

The reader now accepts **both** scan shapes, which is deliberate: re-running
`scan_vg_boxes.py` would produce a current-format file, but with per-image
compact filtering (`10239c24e`) and per-band supply (`fb4f4ec03`) that qualify
categories differently — silently redefining three datasets whose numbers are
published in #3129 and #3156. The envelope was the only incompatibility; the
selector reads `voted_area`, `n_images` and `union_inflation` and nothing else,
all three present in the 2026-08-12 file.

**Where a band is already built, `--rebuildable` also asks whether a rebuild
would produce *that*.** "Selection runs" and "selection picks the same thing"
come apart in the direction that hurts: both candidate repairs for #3297 made
the selector run again, and only one kept choosing the categories the published
sets hold — the other would have redefined three datasets with the right media
count, the right vectors and nothing visible to say so. So the canary compares
today's selection against the vocabulary the smallest built cell carries and
reports `REBUILD-BROKEN` on any difference. Verified against the live pile on
2026-08-28: all three bands reproduce exactly, 40/40 categories, agreeing
across all three cells present at the time (#3299).

## Building on the GRID

```bash
bash launch_pile.sh              # canary + weights (CPU), then one GPU job per dataset
bash launch_pile.sh coco_val     # just one dataset
```

`--rebuildable` runs in front of every launch. That is the answer to "what runs
the canary periodically": every build already touches the pile, the check costs
a fraction of a second, and a purge is the worst possible moment to learn the
rebuild path rotted. It reports **all** datasets — rot under one you are not
building today still gets seen — but only the datasets being launched gate the
submission, since a broken source under a dataset nobody asked for is news
rather than grounds to refuse.

Weights are prefetched in the same CPU stage because parallel GPU jobs would
otherwise race on the shared HF cache, and because the embedders load with
`cache_dir=<VTSEARCH_MODELS_DIR>` — prefetching to the HF default instead leaves
weights the jobs cannot see.

The GPU **type** is not pinned. `launch_pile.sh` calls
[`pick_gpu.py`](../../slurm/pick_gpu.py) after the prefetch returns (availability
measured before a blocking queue wait is stale) and requests the fastest type
with enough free GPUs for the jobs it is about to submit. This used to be a
hardcoded `v100`, which is why every cell built before 2026-08-17 was embedded on
the slowest GPU on the cluster — 2.3× slower for `siglip2_l` than the L40S nodes
sitting idle beside it. Set `VTS_GPU` to pin a type anyway; see
[`docs/SETUP.md`](../../../docs/SETUP.md#which-gpu-type-gets-requested).

Until 2026-08-28 that query was answering from a field this cluster does not
emit, so it read every GPU as free and always returned the first candidate —
a hardcoded `a100` wearing a query, which sent the #3299 build into a 24-hour
queue with 109 V100s idle. It now reads `AllocTRES` where `GresUsed` is absent
and refuses to count a node whose usage it cannot read; see
[the lesson](../lessons/2026-08-28-the-gpu-picker-reported-every-gpu-free.md).

## Considering a new class for `vg_scale` (#3588)

Three scripts, in the order they answer questions. None of them changes
`SCALE_CLASSES`; they produce the evidence for doing so.

```bash
python shortlist_scale_classes.py --compact --floor 100 --n 80   # what VG supports
python coco_folds.py --classes cup,bowl --out folds.json         # what the name MEANS
python make_class_slate.py --supply-only                         # what a build would get
python make_class_slate.py --out .../slates                      # the review material
python import_slates.py --slates .../slates                      # datasets + empty detectors
```

## Auditing a class's VG names (#3618)

Four scripts, in the order they answer questions. The first two search, the
third decides, the fourth prices what it decided.

```bash
python coco_folds.py --min-count 1 --out folds.json               # names on the class's COCO boxes
python vg_name_families.py --min-images 3 --out families.json     # names sharing its head noun
python name_evidence.py --candidates cands.json \
    --propose-out proposal.json --out evidence.json               # precision, box, verdict
python name_coverage.py --propose proposal.json --out cov.json    # repaired, withheld, band ledger
```

Two more when the question is the **pool** rather than a name (#3635) — the first
needs no proposal at all, and the shipped tables are worth scoring with it about
once a rebuild:

```bash
python pool_contamination.py --out contam.json                    # per-class pool false-negative rate
python pool_contamination.py --propose prop.json --out c2.json    # what a proposed name buys and costs
python pool_contamination.py --drop bicycle:bike --out c3.json    # the counterfactual for an entry that SHIPS
python withheld_difficulty.py --class "stop sign" --names sign \
    --out hard.json                                               # are the withheld images the hard ones?
```

`--drop` exists because `--propose` can only add: scoring `bike` means comparing
the pool *without* it against the shipped pool, and without that the control in
#3635 would have been an estimate rather than a measurement.

What a **human** says about the same pool is `shipped_pool_error.py` (#3666),
which reads the negative pass back out per class and scores it against
`pool_contamination.py`'s prediction:

```bash
python shipped_pool_error.py                  # the twelve, from the committed verdicts
python shipped_pool_error.py --rebank         # re-distil verdicts.csv from the passes on scratch
python shipped_pool_error.py --figures        # + the report's figures (needs the VG pixels)
```

**A group pass is not a per-class rate**, and that is the whole reason this
script is not three lines. A *clean* verdict on "none of these four" is a
negative for every member; a *present* verdict names no member. Attributing the
group finds is what turns one pass into twelve numbers, and it cost 9 images
because COCO settles every find on its own half for free.

Its `ADJUDICATION` table carries the study's actual result: what each find is,
and whether the class's own names would ever have admitted it. Six of nine
were boundary calls on rules that do not exist for the shipped twelve (#3673),
which at a 1% rate moves the estimate further than another 3,000 draws per class
would.

Four when the question is **what a change to the evaluation frame is worth**
(#3667). They are ordered by what they can answer, and the order matters — each
can only be checked by the next:

```bash
python cross_class_negatives_effect.py                            # PRICE it, off the shipped cell, before any GPU
python cross_class_negatives_rebuilt.py --json out.json           # what the rebuild actually moved, vs that price
python cross_class_negatives_difficulty.py --json d.json          # are the new negatives nearer the class?
python cross_class_negatives_shortcut.py --json s.json            # ...or was the OLD contrast a shortcut?
```

The *price* reads `categories`, because that is all a cell pickle carries; the
*build* reads the labels, and an image can hold a class without being designated
a positive for it, so the two disagree by design (0.8% here). The last two are a
pair on purpose: a **text query cannot learn a shortcut** and a **trained head
can**, so the difference between what each loses on the new negatives is the
size of the shortcut. Running only one of them halves the effect and invites the
wrong conclusion.

Run `name_coverage.py` with no `--propose` to score the tables that are actually
shipped, which is what says whether `pile_config` still does what its comment
claims. Every cut is a flag (`--min-precision`, `--min-box`, `--min-sole`), so a
different appetite for withheld negatives is a re-run, not a re-argument.

**`coco_folds.py` is the one that is easy to skip and should not be.** It asks,
over the ~51k images that are both VG and COCO, which VG names land on a COCO
class's boxes (fold-in: what a reviewer must accept) and which COCO class sits
under a VG box of a given name (fold-out: what the VG name denotes). Run against
`book` it prints `magazine` and `magazines` — i.e. it would have caught the split
that cost the `book` pass 49 verdicts, before a human saw one image.

Its fold-out column doubles as a **definition-risk score**: the share of a name's
boxes landing on *no* COCO class, on images COCO annotated exhaustively. The
mechanical floor is ~7-15%; `book`, the class that actually broke, scores 43%.
Anything near that is a class whose rule has to be settled before review, not
during it.

`--classes` scopes the *question*, never the COCO vocabulary the answer is read
against: fold-out always tests a VG box against every COCO class, so the score
for a name does not move when you ask about it alongside different company. It
did once — a class nobody named carried no boxes, so `bike` read 100% "means
nothing" against a recorded 40.1% (#3640) — and 100% is the reading that sends a
good spelling to `SCALE_VG_AMBIGUOUS` and costs the class half its positives.

That rule then travels in `pile_config.SCALE_CLASS_RULES`, whose value is the
**dataset and detector name** — the only string the app shows while voting. A
rule in a manifest is a rule the reviewer never reads.

**All twenty-five classes now carry one, and the long form for the shipped
twelve is [`ANNOTATION-GUIDE.md`](ANNOTATION-GUIDE.md)** (#3673). Read it before
issuing any slate of those classes: #3666 measured that six of the nine pool
errors found in the negative pass were boundary calls on rules that did not
exist, and at a ~1% rate one ruling moves a class further than 3,000 extra
uniform draws would. It also records the trap that nearly wrote two of those
rules the wrong way — a fold-in count is a *box* test and the pool asks an
*image* question, so `watch` (11% against a 4.5% base) and `canopy` (7% against
3.7%) are refused despite landing on 35 and 32 COCO boxes.

A candidate's measured spellings go in `SCALE_CANDIDATE_VG_NAMES`, not in the
`SCALE_VG_NAMES` table above: that one widens the `vg_scale` **read** and is
folded on every build, so an entry there for a class outside *C* would change the
built dataset before anything has been decided. A candidate promoted into *C*
moves its row across.

`make_class_slate.py` differs from `make_audit_slate.py` in what it can assume:
the audit slate reads a class the pickle already holds, while a candidate has
neither banded positives nor a checked negative pool. Positives come from the VG
source through the loader's own `band_candidates`, so the banding is the one a
build would use; negatives come from the built pickle's shared pool, **minus any
image that holds the candidate**. That subtraction is not hygiene: the shared
pool was drawn as "holds none of the current twelve", so 34% of it holds at least
one of the thirteen candidates, and the count is the rebuild cost (#3604).
