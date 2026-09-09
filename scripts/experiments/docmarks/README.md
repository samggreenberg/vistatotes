# DocMarks — eval data for stamp detection

Labelled data for evaluating **structural similarity search on small marks in
scanned documents** — finding a given stamp, seal or letterhead logo in a pile
of pages. Built for the feature the structural embedder is heading toward, and
for the experiments that will decide how it should work.

The 2026-07-13 study found the first configuration where structural search beats
the deep embedder on a real corpus (SuperPoint+LightGlue, AP 0.395/0.481 vs
SigLIP's 0.204/0.235), then ran out of road: its two document corpora are 259
and 1,088 pages, with as few as 9 instances per class, and their class
identities were derived by that study rather than verified by anyone.

```bash
python build_corpus.py --probe                       # can I reach every source?
python build_corpus.py --sources spods               # cluster into candidates
python shortlist.py --corpus <dir> --write-roster     # rank them, draft a roster
$EDITOR <dir>/roster.json                             # pick your two dozen
python build_corpus.py --sources spods --roster <dir>/roster.json

python make_audit_slate.py --task merge               # which classes are one mark?
python make_audit_slate.py --task membership          # verify every instance
$EDITOR <dir>/audit/merge/merges.txt                  # one line per same-mark group
python audit_to_corrections.py --task merge --apply
python audit_to_corrections.py --task membership --apply

source ../pile/pile_env.sh
python embed_corpus.py --tier s --embedders sift_vlad,siglip
```

## The design in one idea: a curated roster, not an inventory

The corpus holds two populations with completely different standards of
evidence, and keeping them apart is the whole point:

- **Roster classes** — a small, named, checked-in set of about two dozen. Every
  instance is adjudicated in or out by hand; every confusable pair is
  adjudicated same or different. Nothing enters by heuristic.
- **Distractors** — everything else, unlabelled and unexamined, in whatever
  quantity the tier budget allows. They need no labels, only to be *safe to
  score against*.

That split is what buys a trustworthy eval at a cost a person can pay.
Verifying 24 classes exhaustively is an afternoon; verifying 400 is not, and a
benchmark whose labels nobody checked is a benchmark whose numbers nobody should
quote. Without a `--roster`, the builder emits candidate classes for
`shortlist.py` to rank — those are *proposals*, and the build says so.

## Both directions of the ground truth

An eval for "find this mark" needs two kinds of label, and clustering can only
ever propose one:

- **same** — a shared `class_id`. These instances must all be found.
- **different** — a recorded separation. These must be told apart.

The second is what usually goes missing, and its absence is invisible: without
it, the only thing keeping two similar marks in separate classes is where a
distance threshold happened to land, so nudging the threshold silently rewrites
the ground truth. Measured on the fixture corpus at a loose threshold, three
distinct marks collapse into one class — unless the separations are on disk, in
which case all three survive.

So a `different` verdict is stored permanently in `separations.json`, keyed on
**page ids** (which survive a re-cluster; class ids do not) and enforced as a
cannot-link constraint on every future run. The constraint propagates, so two
separated marks cannot be reunited through some third ambiguous crop.

## What each source ships

**SPODS** — 1,088 scanned pseudo-official pages, direct download, no
registration. Confirmed by walking the RAR headers:

```
SPODS_Dataset/image (1..1088).png
Ground truth (GT1)/{logo,signature,stamp,text}/image (1..1088).png
```

Four **binary pixel masks per page**, one per category. Note what is *not*
there: any notion of *which* logo. A previous study reported "64 logo/stamp
classes" for SPODS with names like `logo_14` — those identities were derived by
that study and never verified. Since class identity is the entire ground truth
of an instance benchmark, that inventory was a hypothesis with unmeasured error
bars. Here the derivation is explicit (`cluster_marks.py`), flagged
(`provenance="clustered"`), and only becomes ground truth after the membership
audit.

SPODS is **not offline**, despite having been recorded that way: its own page
advertises `www.facweb.iitkgp.ernet.in`, a decommissioned host that 503s, while
`facweb.iitkgp.ac.in` serves the same 2.94 GB file. The authors' Scanned
Document Degradation Tool sits beside it (`sddt.zip`, 689 MB).

**StaVer** — 400 scanned dummy bills, pixel-accurate stamp GT, per-file `info`
text (stamp count, colour, overlap). Kaggle mirror; DFKI's original was
unreachable. Locations, no identities. The recorded stamp count is used as an
independent check on the mask decomposition — a page the dataset says has one
stamp that decomposes to four means the merge gap is wrong.

**Tobacco800** — 1,290 real scanned business documents, 412 with a logo, GEDI
XML ground truth from UMD's LAMP. Its published protocol keeps the 21 logo
categories with ≥2 occurrences, which cannot support a train-and-search eval.

**UCSF IDL** — an open Solr index; measured live 2026-08-25:

| query | count |
|---|---:|
| tobacco, 1–2 pages, has `collection` | 13,216,456 |
| tobacco, 1 page, `type:letter` | 1,802,100 |
| `author:"RJR"` 1-page letters | 162,197 |
| `author:"PHILIP MORRIS"` | 73,320 |

**`author` is a candidate pool, not a class.** The field asserts a page is
*from* a company; it has never looked at the mark. Making it a class id writes
two guaranteed errors into the ground truth: a company that redesigned its
letterhead yields one class holding two artworks (so a detector is punished for
telling them apart), and two subsidiaries sharing artwork yield two classes
holding one mark (so it is punished for recognising it). Those are exactly the
errors the eval exists to measure. So the author narrows millions of pages to a
high-yield pool, each candidate gets a coarse top-of-page band to locate the
mark, and identity is settled by clustering plus adjudication like everything
else. `documentdate` is recorded and never enters a class id: era is a fact
about the calendar, not about the mark.

Distractors only: `--ucsf-letterhead-per-author 0`.

**Synth** — real artwork (LogoDet-3K, or any `--synth-pool-dir`) pasted onto
held-out real scans at known `(x, y, scale, rotation)` with scanner-style
degradation. Exact ground truth, and the only stratum that can be *swept*: size,
rotation and count are inputs, so an experiment can locate the ~32px floor or
the inlier-count working point instead of straddling it. The rule that comes
with it: **synthetic numbers quantify a mechanism, real numbers size the
effect.** A finding that appears only in `synth` is a hypothesis about the
pipeline, not a claim about documents.

## Masks to marks: decompose, merge, *then* filter

SPODS and StaVer ship pixel masks, not boxes, so `sources/_common.py` has to
decide what counts as one mark. The order of those three steps is the whole
game, and getting it wrong is silent.

A mark's mask is **not one connected component**. A rubber stamp is a ring, the
text inside it, and however many broken arcs the ink left behind; a script stamp
is one component per pen stroke. Each of those is individually tiny. So an area
floor applied to *raw* components deletes eleven fragments of the dozen as
speckle, the merge that exists to reassemble them never sees them, and the one
or two chunkiest survivors get promoted to classes of their own. That is how a
class called `spods/stamp_00129_1` came to be 38 instances of the word **New**,
clipped out of a three-line "Dy.Manager / NewEastZone" stamp (issue #3361).

So: **decompose, merge, then filter**, with the floor applied to the merged
group's *ink* (a ring is mostly hole, so its box area proves nothing). The only
filter that runs before the merge is an absolute few-pixel one, because true
single-pixel scan noise carries no evidence either way but can bridge the gap
between two marks that should stay apart.

The merge gap is a fraction of the page's longest side (`MERGE_GAP_FRAC`), not a
pixel count — SPODS pages are A4 at 300 DPI (~2,476×3,480) and StaVer's are not.
0.035 is read off a sweep of all 1,088 SPODS pages: **from gap 90 px through gap
300 px the output is byte-identical** — one mark per page that has one, median
428 px, largest box 3.5% of the page — because nothing else lies within 300 px
of a mark. Below 90 the merge under-runs and splits real stamps across their own
line spacing. StaVer's recorded stamp count is the independent check on the
other side; it is the source that actually carries two stamps on one page.

Two consequences worth stating plainly:

- **The `text` mask yields no marks.** It is the page body — a property of the
  page, not a thing on it. Filtered, what survived was not even words: it was
  whichever headings and ruled tables had an underline welding their glyphs into
  one component, about 1.1 per page. They were never query classes, but they
  were real entries in `page.marks` and leaked into everything that read it
  without a kind filter, starting with the synthetic-background selector. The
  mask is kept as `meta["text_frac"]` and `meta["text_components"]`; the
  documented non-queryable negative control is `signature`, which is a localised
  mark and stays one.
- **A box covering more than `MAX_MARK_AREA_FRAC` of its page is rejected**,
  with a warning naming the page. Nothing in SPODS trips it after the above —
  the largest surviving mark is 3.5% — which is the point: it is a tripwire for
  the next source, not a filter this one needs.

## Three kinds of negative

Not all distractors are equal, and the manifest keeps them distinct:

- **known negative** — a page from a source exhaustively checked for this class,
  so its *absence* of the mark is verified. These are the valuable ones: same
  scanner, same paper, same era, known clean. A SPODS page carrying a different
  mark is the hardest possible negative for a SPODS class, and the membership
  audit is what makes it usable instead of a contamination risk.
- **presumed negative** — from a contamination-safe source nobody checked
  individually. Fine in bulk, and the only way to reach 200k.
- **excluded** — a contamination risk, never scored.

The trap that last category exists for: RVL-CDIP, Tobacco800 and UCSF's Tobacco
industry all descend from IIT-CDIP, so an American Tobacco letterhead is
*certain* to appear in an RVL-CDIP "distractor" pool. Unlabelled positives don't
make a benchmark slightly noisy — they make a correct retrieval count as a false
positive, so the metric punishes the model for being right. No hand pass fixes
that at 200k pages; `CONTAMINATES` in `docmarks_config.py` fixes it by
construction, and each class records its resolved
`eligible_distractor_sources`.

## Tiers

`s`=5k, `m`=50k, `l`=200k pages, **nested**: every page in `s` is in `m` is in
`l`, sharing class ids, so a result on one is comparable to a result on another.
Roster positives are in every tier — a tier keeping 3 of a class's 30 instances
measures a different and harder problem, not the same one more cheaply.
Distractors get a stable hash rank and tiers are prefixes of it.

Two stability promises are on offer and they genuinely conflict:

- **within a build** (default): exact budgets, nested. Run on `s`, then `l`, no
  rebuild.
- **across builds** (`--pin-tiers <earlier build_report.json>`): membership fixed
  by absolute rank cutoff, so a grown source pool cannot evict a page from a
  tier it was already in. Budgets drift instead.

Without pinning, a build over a different page set is a **new corpus version**.
Both behaviours are pinned by tests, including the negative one.

## The human passes

In the order you run them. Only the first two are needed for a first eval.

1. **`merge`** — the whole class list on a handful of numbered contact sheets,
   ordered so near-identical classes sit next to each other, plus explicit
   side-by-side sheets for the closest pairs. The answer is a list of index sets
   in `merges.txt` — `12 37 41` means those three are one mark — and nothing at
   all for the classes that are already right. See **The slate** below.
2. **`membership`** — every instance of every roster class, numbered on contact
   sheets. Verdict is `ok` or the indices that are *not* this mark (`3,17`), so
   a 30-crop class is one line. Afterwards no positive is unexamined, which is
   what lets a miss be blamed on the detector rather than the label. A rejected
   crop keeps its box and stays on its page — it becomes a known negative.
3. **`confusable`** — the same question as `merge`, asked one pair per sheet.
   Correct, and the form to use on a roster small enough that the full matrix is
   a sitting; past a couple of dozen classes prefer the slate, which compiles to
   exactly these verdicts. `same` sends you to `merge_into:` on the cluster task;
   `different` writes a permanent separation.
4. **`cluster`** — is a class one mark at all? Mostly useful while choosing a
   roster. `split` is productive: it re-clusters that class alone at half the
   threshold and re-sheets the pieces, disturbing nothing else.
5. **`distinctive`** — mark vs shape. A plain warning triangle or ruled box is a
   *shape*: "find this rectangle" is not a well-posed retrieval query. The prior
   study's worst classes (`warning_diamond` at 17 keypoints, `hospital_cross`)
   are exactly this. Generic classes are kept and labelled, never deleted.
6. **`letterhead`** — for the later UCSF expansion: sample bands per candidate
   author and count how many carry a printed mark at all. Decides whether that
   pool is worth clustering.

Query crops come from each class's largest boxed instance automatically (the
prior study measured a 2.2× AP advantage for a clean query over a small in-scene
crop). Band-located classes get none — auto-cropping the strip would hand the
query a banner of letterhead plus address plus rule line and call it a logo,
which is worse than no crop because it looks like ground truth. They are listed
in `build_report.json` under `needs_hand_crop`.


## The slate: ask for the partition, not the pairs

The pairwise pass is right about what the corpus needs and wrong about what a
person can deliver. Adjudicating a matrix means one sheet per pair, and pairs
are quadratic: the 24-class roster the design was written around is 276 pairs,
which is a sitting; **v2's 60 admitted classes are 1,770**, which is not. Worse,
the answer it asks for is 1,770 independent binary verdicts, ~1,750 of them
`different` between marks nobody could confuse.

That is the wrong shape for the information a reviewer actually holds. What they
know after looking at the classes is a **partition** — these three are one mark,
everything else is already right — and almost all of it is the trivial part. So
the slate elicits the partition directly:

```
python make_audit_slate.py --task merge
$EDITOR <corpus>/audit/merge/merges.txt
python audit_to_corrections.py --task merge --apply
```

`slate_*.png` is every class as a numbered 3-up strip of its own instances,
4x6 to a sheet. `merges.txt` is the answer, and the format is the whole of it:

```
12 37 41
3 8        # same elephant stamp, blue and red ink
REVIEWED-ALL
```

One line per group of classes that are the same mark. A 60-class slate that
over-split three times is four sheets and three lines. Nothing is written for a
class that is already right, groups that share a member are unioned rather than
refused (sameness is transitive; the file is allowed to be redundant about it),
and any token that is not a resolvable index is refused rather than guessed at
— each of those is a typo whose silent interpretation would write a permanent
merge between classes nobody looked at.

**Three instances per cell, not one.** A single exemplar fits every class on one
page and is tempting for exactly that reason, but then a merge decision rests
entirely on one crop being representative of its class — which is the assumption
the membership pass exists because we do not trust.

**Similarity order, plus an appendix.** Classes are laid out by a greedy
nearest-neighbour seriation of the same descriptor the clustering uses, so an
over-split shows up as two adjacent cells that look alike — a thing people are
good at. But a 1-D order cannot preserve a metric that is not 1-D, and the row
wrap breaks adjacency every four cells regardless, so the ordering is an aid to
scanning and not a guarantee. The guarantee is `pairs_*.png`: the
`MERGE_SLATE_NEAR_PAIRS` closest pairs, explicitly side by side, so no pair
where a wrong call costs anything depends on the layout having been kind.

### The descriptor the audit asks with is not the one it was built with

`phash` clusters 200k pages for nothing and is the reason the corpus exists at
this size. It is a poor judge of its own output. Measured on v2 (#3600): the one
literal duplicate on the slate — two classes of the same `DY.Secretary` rubber
stamp — ranked **83rd of 120** in the near-pair appendix, behind 82 pairs of
stamps nobody would confuse, while two internally-mixed classes took 37% of the
appendix between them. A perceptual hash of blue ink on white paper measures ink
layout, and two different stamps in the same typeface at the same size have
nearly the same ink layout. The identical failure is on record for UCSF
letterhead bands, where no threshold produced classes at all.

The audit can afford what the build cannot — it runs over ~1,300 crops, not
200k pages — so `siglip_audit.py --embed` embeds every class instance once with
`siglip2_l` and caches the vectors. Only that step needs a card; the slate
render, the analysis and every re-render afterwards read the cache on `cpu`:

```bash
bash launch_docmarks.sh siglip                                   # GPU, ~7 min
python make_audit_slate.py --task merge   --descriptor siglip2_l # re-ordered slate
python make_audit_slate.py --task cluster --descriptor siglip2_l # split proposals
```

Three questions get better answers, and each was checked against a human verdict
before being trusted rather than after:

- **Which classes are nearest** — by class *centroid*, not by one query crop, so
  an unrepresentative exemplar cannot place a whole class (#3599). On v2 the
  closest pair went from 0.11 (two unrelated stamps) to 0.030 (two scans of the
  same B&W wordmark).
- **Which classes hold more than one mark** — each class's own instances
  clustered by average linkage, swept over `AUDIT_SPLIT_SWEEP` and reported as a
  sweep, never as a single verdict. Average linkage rather than single: single
  linkage's failure mode is one ambiguous crop bridging two marks, which is the
  defect the pass exists to find. On v2 it proposed `15 / 8 / 1` for the StaVer
  class and `22 / 2` for `spods/stamp_00489_1` — both exactly the boundaries the
  reviewer had already drawn by eye, and both found without being told. A class
  whose two most distant instances sit further apart than the loosest threshold
  in the sweep (`AUDIT_MIXED_MAX_WITHIN`) is flagged `mixed`.
- **Whether a class's query crop retrieves its own class** — the question the
  eval will ask, stated as the *rank* of the class's own centroid rather than a
  distance whose scale nobody knows. It flags 3 of 59 on v2, including the one
  `phash` scored second-*best* of 60. Beside the rank, **how many of the class's
  own instances that crop reaches** before the nearest other class's centroid,
  which is the part a rank cannot see.

#### A screen for a crop is not a screen for a class

Both query-crop numbers are properties of one *crop*; `mixed` is a property of a
*class*, and #3610 is the case that separates them.
`staver/stamp_stampds-00156_0` holds **five** marks across 27 instances — a
16-strong blue routing box, eight `DFKI Empfang`, and three singletons — and the
rank scores it 0 at distance 0.078, the healthiest tier of all 59 classes. That
score is *correct on its own terms*: its query crop is a good instance of the
16-strong mark, so it retrieves its own class first, and nothing about a rank is
sensitive to the class being five marks in a trench coat. Its `max_within` is
0.433, second worst of 59. Gate on both, or the mixed classes the `cluster`
sheets exist to adjudicate arrive pre-certified healthy.

And the sample the questions are asked of is **spread** over the class, not
taken off the head of its page-id list. Page ids sort by source and number, so a
head sample is the scanner's order: the two classes #3610 was filed over are 27
and 30 instances against `AUDIT_MAX_PER_CLASS` = 24, and between them five marks
live only in the tail. Note that a plain `[::step]` stride does not fix this —
`27 // 24` is 1, so the stride hands back the first 24 — which is why
`sources/_common.spread` spaces indices over the whole range instead.

One caveat for the reviewer working these sheets: a **wide crop catches
neighbouring page furniture**, and at 150 px that is indistinguishable from a
second mark. Two crops in the StaVer class show the routing box with a black
`EINGEGANGEN AM 05. JAN. 2011` stamp above it; others catch `* Artikel mit 7%`
or `Bankverbindung:`. Check at full size before splitting.

The proposals are hypotheses on a contact sheet. The verdict vocabulary does not
change, `split` still means a person looked, and a false proposal is expected:
on v2 two of the four proposed splits were scan quality varying more than the
mark does, which is obvious on the sheet and invisible in the number.

### `REVIEWED-ALL` is the closed world, and it is deliberately narrow

A partition asserts both directions at once: within a group is `same`, across
groups is `different`. Taken literally, one reviewed slate would adjudicate all
1,770 pairs — the complete "different" half of the ground truth, from one
sitting, which is the half this README elsewhere says usually goes missing.

It is not taken literally, because it would be a lie. A reviewer scanning sixty
thumbnails has genuinely compared the cells next to each other and every pair on
the appendix sheets. They have not compared the far end of the distance ranking.
Recording those as adjudicated would put a decision nobody made into
`adjudications.json`, which every future re-cluster is then bound by — the exact
failure the separations exist to prevent, committed by the mechanism meant to
prevent it.

So `REVIEWED-ALL` separates **the appendix pairs and only those**. Everything
else stays unadjudicated: still separated by the threshold, still liable to move
if the threshold moves, and honestly labelled as such. `MERGE_SLATE_NEAR_PAIRS`
is that honesty budget — raise it to buy more of the matrix, and pay for it in
sheets to work through. Leave the line off entirely and only the merges are
recorded; nothing is assumed about what was looked at.

The slate is an input *format*, not a second path through the ground truth:
`merge_verdicts` compiles it into the same `same`/`different` rows the pairwise
pass produces, and `apply_confusable` records them. A group becomes a star
rather than a clique (sameness is transitive and the classes are merged
outright, so n-1 rows state a group of n), every `same` is emitted before every
`different` (the applier merges as it goes and follows the chain afterwards, so
a separation pinned first would name a class about to stop existing), and a pair
that a merge put inside one group is never also separated — `save_adjudications`
refuses a pair ruled both ways, so that is a correctness gate rather than
tidiness.

### What it does not do

It does not check instances. `merge` fixes the *partition* — which proposals are
one mark; `membership` fixes the *instances* — whether each crop really is that
mark. Both are needed before the classes stop being proposals, they are one
sitting rather than two, and `launch_docmarks.sh slate` renders both into one
bundle for that reason.

## Output

```
corpus.jsonl        one record per page: path, size, marks, provenance, tier
classes.json        per class: instances, distinct_from, caveats, eligible distractors, audit state
roster.json         the hand-picked classes an eval runs on
separations.json    adjudicated "different mark" pairs, keyed on page ids
queries/            one query crop per box-located roster class
shortlist.png/json  ranked candidates for choosing a roster
cluster_report.json what clustering did, and how many separations it honoured
audit/merge/       the slate: slate_*.png, pairs_*.png, index.json, merges.txt
audit/membership/  numbered instance sheets + verdicts.jsonl
build_report.json   counts, survival curve, tier cutoffs, rejections, warnings
```

Every mark carries a `provenance`: `gt` (a box shipped by the source),
`clustered` (identity derived here), `clustered_band` (identity derived from a
coarse strip, so the box locates a region and not the mark), `candidate` (pool
member, no identity yet) or `synthetic` (true by construction). A class also
carries `audit.membership_verified` — **false means it is still a proposal**.
Do not aggregate across provenances without saying so.

## Embedding cells

`embed_corpus.py` writes `docmarks_<tier>__<embedder>.pkl` into the shared
pile's `embeddings/` dir, in the pile's format, via its pickle IO. It is
deliberately *not* a `pile_config.DATASETS` entry: the pile builds the full
dataset × embedder cross-product, so adding DocMarks and `sift_vlad` there would
silently schedule `sift_vlad` cells for all six existing datasets, on a mount
the playbook already calls chronically full.

## Strict partition, cheap merge

The two clustering errors do not cost the same, so the threshold is not set
where it is "most accurate":

- An **over-split** shows up in the audit as one obvious pair of near-identical
  classes. One click.
- An **over-merge** shows up nowhere. The class quietly means two things for as
  long as the corpus lives, and every number computed on it is wrong in a
  direction nobody can see.

So the threshold runs **strict**, the partition over-splits on purpose, and the
repair is done by hand. Both directions of every hand decision are recorded in
`adjudications.json` as page-id pairs and replayed on every future re-cluster —
`same` becomes a must-link, `different` a cannot-link — so an afternoon of
merging is not undone the next time a number moves. A pair ruled both ways is
refused rather than resolved by whichever is applied last.

Measured on 2,054 real SPODS marks with the 256-bit hash, after the mask
decomposition was fixed:

| threshold | classes | largest component | share | classes with ≥10 |
|---:|---:|---:|---:|---:|
| 0.02 | 1,261 | 31 | 1.5% | 31 |
| 0.08 | 800 | 31 | 1.5% | 40 |
| **0.10** | **672** | **31** | **1.5%** | **44** |
| 0.12 | 523 | 166 | 8.1% | 47 |
| 0.14 | 428 | 354 | 17.2% | 43 |
| 0.16 | 310 | 653 | 31.8% | 36 |
| 0.22 | 138 | 1,288 | 62.7% | 22 |

Read the **share** column, not the class count. 0.10 is the top of the flat
region: the most the clustering assembles before it starts assembling things
that do not belong together. Note that 0.16 — the value read off the *previous*
sweep, and perfectly defensible against those marks — now chains a third of the
corpus while still reporting a plausible 310 classes. **The threshold is a
property of the marks, so a change in what a mark *is* invalidates it.** Fixing
the decomposition (issue #3361) replaced "the one chunkiest fragment of each
stamp" with "the whole stamp", which is a different object to hash; the sweep
had to be re-read from scratch, and the class count moving the *right* way (36
usable classes at 0.16 before, 44 at 0.10 now) is not what tells you so.

## The descriptor had to be widened to get here

The first real run merged a red **book** stamp (5 instances) and a blue
**elephant** stamp (27) into one class, and no threshold separated them — a
single pair at Hamming 2/64 bridged the set, so it was one group at 0.04 and 21
fragments at 0.03.

The mechanism is frequency. A stamp's border ring is big, smooth and
low-frequency; the interior that says *which* stamp is not. An 8×8 DCT block
keeps almost nothing but the ring, so a 64-bit hash encoded "is a round stamp".
The hash is now 16×16 (256 bits) with a soft radial taper toward the crop's own
mean, and on that class it gives exactly {27 elephants} + {5 books}.

It stays **greyscale** on purpose: the same elephant appears in blue on 26
pages and red on one, and belongs in a single class.

Clustering on SigLIP crop vectors would have separated them trivially and was
rejected: SigLIP is one arm of the eval, so letting it define the classes would
tilt the comparison toward it. The proposal step stays model-free.

Re-run `tune_clustering.py` whenever the source set or the descriptor changes.
The number is a property of the data and does not travel — it moved from 0.05
to 0.16 when the hash went from 64 to 256 bits.

## Looking at what you built

```bash
python make_report.py --corpus <dir> --out docs/experiments/<date>-docmarks/report.html
```

One self-contained HTML page: counts per source and provenance, whole pages with
marks boxed in situ (the only way to see how small the target is), every class
as a strip of its own instances, the distractor pool, and the mark-size
distribution against the 32px structural floor. Images are inlined, so the file
survives being archived or opened on a machine with no access to `/expscratch`.

## Running it at full scale

A tier-`s` SPODS-only build fits on a laptop. Tiers `m` and `l` need the
cluster — see **[`GRID-RUNBOOK.md`](GRID-RUNBOOK.md)** for sizing, staging, the
resume story and what to check afterwards.

`python build_corpus.py --probe` first, wherever you run. Every source fails
differently — a decommissioned hostname, a missing Kaggle token, an absent RAR
extractor — and finding out which costs seconds now and a queue slot later.
SPODS needs one of `bsdtar` / `7z` / `unar` / `unrar`; StaVer and Tobacco800
need a Kaggle token. The probe is metadata-only — it fetches no source bytes —
so asking it repeatedly is free.
