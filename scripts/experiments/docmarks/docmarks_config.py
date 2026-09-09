"""Configuration for the DocMarks corpus — stamps and logos in scanned documents.

DocMarks is one corpus assembled from several sources, in three strata:

* **anchor** — real ground truth from SPODS, StaVer and Tobacco800.  These carry
  per-mark boxes; SPODS and StaVer need an identity-clustering pass on top (see
  :mod:`cluster_marks`) because neither ships instance labels.
* **haystack** — real scanned pages with no marks of interest, pulled from the
  UCSF Industry Documents Library.  Distractors, plus (optionally) weakly
  labelled letterhead classes keyed on the document's ``author`` field.
* **synth** — real mark artwork pasted onto held-out real scans at known
  ``(x, y, scale, rotation)``.  Instance ground truth by construction; used for
  statistical power, never quoted on its own.

The corpus is emitted as one manifest with **nested tiers**, so a 5k experiment
and a 200k experiment read the same file and the same class ids.

Every knob here is overridable by environment variable so a GRID job can be
re-pointed without editing the tree.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

#: Where source archives are downloaded and unpacked.  Big; keep it off the
#: 50G mount (see scripts/experiments/GRID-PLAYBOOK.md).
RAW = Path(os.environ.get("VTS_DOCMARKS_RAW", "/expscratch/{u}/docmarks/raw".format(u=os.environ.get("USER", "user"))))

#: Where the assembled corpus lands: ``images/``, ``queries/``, ``corpus.jsonl``,
#: ``classes.json``.  This is what a study reads and what the pile embeds.
OUT = Path(
    os.environ.get("VTS_DOCMARKS_OUT", "/expscratch/{u}/docmarks/corpus".format(u=os.environ.get("USER", "user")))
)

# --------------------------------------------------------------------------
# Tiers — nested by construction
# --------------------------------------------------------------------------

#: ``tier -> total page budget``.  Tiers are *nested*: every page in ``s`` is in
#: ``m``, every page in ``m`` is in ``l``.  Positives are in every tier (a class
#: with 30 instances is useless if a tier keeps 3 of them); only distractors are
#: subsampled, by a stable seeded rank on the page id, so growing a tier never
#: reshuffles the smaller one.
TIERS: dict[str, int] = {
    "s": int(os.environ.get("VTS_DOCMARKS_TIER_S", "5000")),
    "m": int(os.environ.get("VTS_DOCMARKS_TIER_M", "50000")),
    "l": int(os.environ.get("VTS_DOCMARKS_TIER_L", "200000")),
}

#: Ordered smallest-to-largest.  Used for the nesting invariant.
TIER_ORDER: tuple[str, ...] = ("s", "m", "l")

#: Salt for the deterministic distractor rank.  Change it and every tier
#: membership is reshuffled — so don't, unless you mean to.
TIER_SALT = os.environ.get("VTS_DOCMARKS_TIER_SALT", "docmarks-v1")

# --------------------------------------------------------------------------
# Class admission
# --------------------------------------------------------------------------

#: A class needs at least this many instances to be admitted as a *query* class.
#: Tobacco800's published "21 logo classes" uses a >=2 bar, which cannot support
#: a train-and-search eval: with two instances there is nothing left to retrieve
#: once one is the query.  ``build_corpus.py`` prints the survival curve over
#: every threshold so this can be set from data rather than taste.
MIN_INSTANCES = int(os.environ.get("VTS_DOCMARKS_MIN_INSTANCES", "10"))

#: Per-source instance bar.  10 is right where a source has classes that deep;
#: applied to one that does not, it does not raise quality, it just empties the
#: source.  Measured on the 200k build, queryable classes surviving each bar:
#:
#:            >=2   >=5   >=8  >=10
#:   spods     174    70    49    44
#:   staver     12     1     1     1     <- 10 admits ONE class
#:   tobacco800 28     5     3     3
#:
#: A roster drawn under a flat 10 is ~85% SPODS, so the eval measures SPODS and
#: reports a number about documents.  StaVer and Tobacco800 drop to 5: still
#: enough instances to both query and retrieve, and every one of them is
#: hand-adjudicated in the membership pass, which is what actually makes a class
#: trustworthy -- the instance count is a proxy for that, not a substitute.
MIN_INSTANCES_BY_SOURCE: dict[str, int] = {
    "staver": 5,
    "tobacco800": 5,
}


def min_instances_for(source: str) -> int:
    """Instance bar for *source*.  An explicit env override wins for all."""
    if os.environ.get("VTS_DOCMARKS_MIN_INSTANCES"):
        return MIN_INSTANCES
    return MIN_INSTANCES_BY_SOURCE.get(source, MIN_INSTANCES)


#: Marks smaller than this (longest side, px, at the page's native scan
#: resolution) are recorded but never promoted to a query class.  The
#: 2026-07-13 study found a hard floor around 32 px below which no structural
#: pipeline recovers anything; a class made of sub-floor instances measures the
#: floor, not the method.
MIN_MARK_PX = int(os.environ.get("VTS_DOCMARKS_MIN_MARK_PX", "32"))

#: A *merged* mark carrying less ink than this fraction of the page is dropped
#: as mask speckle.  It is applied after the merge, never before: a stamp's
#: fragments are each individually below it, so filtering first deletes the
#: evidence that the stamp is there (issue #3361).
MIN_MARK_AREA_FRAC = float(os.environ.get("VTS_DOCMARKS_MIN_MARK_AREA_FRAC", "0.0002"))

#: A mark covering at least this fraction of its page is rejected as a mask
#: artefact, with a warning naming the page.  A mark is a thing *on* a page,
#: not the page: the case this catches is a ruled table whose borders weld the
#: whole grid into one connected component (``spods/00975`` reached 45.9%).
#: The bar is deliberately loose — the observed median mark covers 0.76% of its
#: page and p90 is near 2%, so 25% is more than ten times the largest mark
#: anyone has looked at and agreed with.
MAX_MARK_AREA_FRAC = float(os.environ.get("VTS_DOCMARKS_MAX_MARK_AREA_FRAC", "0.25"))

# --------------------------------------------------------------------------
# Contamination — which sources may serve as distractors for which classes
# --------------------------------------------------------------------------
#
# The trap this exists to avoid: RVL-CDIP and Tobacco800 are both drawn from
# IIT-CDIP, so an American Tobacco letterhead is *certain* to appear in an
# RVL-CDIP "distractor" pool.  Unlabelled positives in the distractor set do not
# make the benchmark slightly noisy, they make a correct retrieval count as a
# false positive — the metric punishes the model for being right.  No amount of
# hand annotation fixes that at 200k pages, so it is fixed by construction here.
#
# Read as: "a class from source K may be scored against distractors from any
# source NOT listed in CONTAMINATES[K]".

CONTAMINATES: dict[str, frozenset[str]] = {
    # Indian pseudo-official documents authored for the dataset.  Their marks
    # exist nowhere else on earth, so every other source is a safe distractor.
    "spods": frozenset({"spods"}),
    # Stamps on European scanned invoices.  Likewise self-contained.
    "staver": frozenset({"staver"}),
    # IIT-CDIP tobacco litigation documents.  UCSF's Tobacco industry is the
    # *same underlying archive*, so it is excluded; the other UCSF industries
    # are different companies and are admitted.
    "tobacco800": frozenset({"tobacco800", "ucsf:Tobacco"}),
    # Weakly-labelled UCSF letterhead classes contaminate all of UCSF: the same
    # company's letterhead recurs across industries (Philip Morris reaches Food
    # through Kraft), and the label is metadata-derived rather than observed.
    "ucsf": frozenset({"ucsf"}),
    # Synthetic pastes contaminate only their own backgrounds, which
    # build_corpus.py holds out of every other stratum.
    "synth": frozenset({"synth"}),
}

#: UCSF industries pulled for the distractor pool.  Tobacco is deliberately
#: *first* and deliberately excluded from Tobacco800's eligible distractors by
#: ``CONTAMINATES`` above — it is pulled because it is the richest source of
#: scanned letterhead for the weakly-labelled classes, not despite the clash.
UCSF_INDUSTRIES: tuple[str, ...] = ("Tobacco", "Opioids", "Chemical", "Fossil Fuel", "Drug", "Food")

#: Fraction of page height treated as the letterhead band on a UCSF candidate
#: page.  UCSF ships no boxes, and a mark nobody can see cannot be adjudicated;
#: a letterhead is at the top of the page by definition, so the top strip is a
#: coarse but honest locator to cluster on.  It is never a ground-truth box —
#: the tight box comes from the hand-drawn query crop after adjudication.
LETTERHEAD_BAND_FRAC = float(os.environ.get("VTS_DOCMARKS_LETTERHEAD_BAND_FRAC", "0.22"))

#: Companies whose single-page letters form the letterhead **candidate pool**.
#: ``author`` (who wrote it), not ``collection`` (whose files it sat in): a
#: letter *in* the Philip Morris collection is as likely to be incoming mail on
#: a law firm's letterhead.  Live counts of single-page ``type:letter``
#: documents per author, measured 2026-08-25, are in the README.
#:
#: An author is a pool, never a class.  See ``sources/ucsf.py`` for why turning
#: one into a class id would write two guaranteed errors into the ground truth.
UCSF_LETTERHEAD_AUTHORS: tuple[str, ...] = (
    "PHILIP MORRIS",
    "RJR",
    "LOR, LORILLARD",
    "AMERICAN TOBACCO",
    "BROWN & WILLIAMSON",
    "BATCO",
    "COUNCIL FOR TOBACCO RESEARCH",
    "TOBACCO INSTITUTE",
)

# --------------------------------------------------------------------------
# Identity clustering
# --------------------------------------------------------------------------

#: Default backend for turning per-mark crops into identity classes.  ``phash``
#: is cheap, deterministic and runs with no models — good enough to build the
#: audit slate a human then corrects.  ``siglip`` is the quality option and
#: needs the pile's models dir.
CLUSTER_BACKEND = os.environ.get("VTS_DOCMARKS_CLUSTER_BACKEND", "phash")

#: Agglomerative merge threshold, in the backend's own distance units
#: (normalised Hamming for ``phash``, cosine distance for ``siglip``).
#:
#: **Deliberately strict, because the two errors do not cost the same.** An
#: over-split shows up in the audit as one obvious pair of near-identical
#: classes and costs one merge click; an over-merge is invisible, and quietly
#: makes a class mean two things for as long as the corpus lives.  So the
#: threshold is set below where merging starts and the repair is done by hand,
#: with every merge recorded in ``adjudications.json`` and replayed on each
#: re-cluster so the work is done once.
#:
#: 0.10 is read off a sweep of the real corpus (2,054 SPODS marks, 256-bit
#: hash).  From 0.02 to 0.10 the largest component is pinned at 31 marks (1.5%)
#: — the size of a legitimate class here — while usable classes climb from 31 to
#: 44; it breaks at 0.12 (166 marks, 8.1%), reaches 31.8% at 0.16 and chains
#: outright by 0.22.  0.10 is the top of the flat region — the most the
#: clustering can assemble before it starts assembling things that do not belong
#: together — and it sits in the valley of a now cleanly bimodal distance
#: histogram (a within-class mode below 0.06, the between-class bulk above 0.18).
#:
#: Re-run ``tune_clustering.py`` whenever the source set or the descriptor
#: changes; this number is a property of the data, and it does not travel.  It
#: moved from 0.05 when the hash went from 64 to 256 bits, and from 0.16 when
#: the mask decomposition was fixed (issue #3361) — that is the point.  The
#: marks the descriptor sees are different objects now: whole stamps instead of
#: the one chunkiest fragment of each, so 0.16 chained 653 marks (31.8%) into a
#: single class while still reporting a plausible-looking 310.
CLUSTER_THRESHOLD = float(os.environ.get("VTS_DOCMARKS_CLUSTER_THRESHOLD", "0.10"))

#: Per-source override, because the paragraph above ends "this number is a
#: property of the data, and it does not travel" -- and it does not travel
#: between SOURCES either, which the single global value quietly assumed.
#:
#: Swept per source on the 200k build (2026-09-02).  The three disagree sharply,
#: and 0.10 was serving SPODS while damaging the others:
#:
#:   spods       0.10  largest component flat at 1.5% to 0.10, 8.1% at 0.12.
#:   staver      0.04  already 5.8% at 0.02 and 22% by 0.10 -- StaVer's stamps
#:                     are near-duplicates of each other, so it chains early.
#:   tobacco800  0.18  the opposite problem: only 14.7% at 0.18, and usable
#:                     classes PEAK there at 9 against 3 at 0.10.  Its logos are
#:                     printed artwork rather than inked impressions, so the
#:                     within-class distances are wider.
#: UCSF is absent on purpose -- see CLUSTERED_SOURCES.  It has no usable
#: threshold, which the sweep says plainly rather than by omission.
#:
#: Re-run `tune_clustering.py --source <s>` per source, not once for the corpus:
#: a sweep over everything is dominated by whichever source has the most marks
#: and reports its optimum as the corpus's.
CLUSTER_THRESHOLD_BY_SOURCE: dict[str, float] = {
    "spods": 0.10,
    "staver": 0.04,
    "tobacco800": 0.18,
}


#: Sources whose marks are clustered into candidate classes.
#:
#: **UCSF is not one, and the reason is the descriptor rather than the knob.**
#: Its letterhead "mark" is a fixed-geometry crop -- the top 22% of every page --
#: so a perceptual hash of it is dominated by page layout, not by the logo
#: inside it, and two unrelated companies' letterheads at the same position hash
#: alike.  Swept on 3,000 marks (2026-09-02) there is no flat region anywhere:
#: the largest component is already **12.4% at 0.02**, the lowest threshold on
#: the grid, and climbs monotonically -- 36% at 0.04, 81% at 0.10, 99% at 0.22 --
#: while 85% of marks stay singletons.  SPODS sits pinned at 1.5% across that
#: same range.  There is no percolation *transition* because it is percolated
#: from the start, so no threshold exists to choose.
#:
#: At 0.10 this produced a single 12,706-instance "class" which, being made of
#: admitted-class pages, pinned 13,874 pages into tier `s` (budget 5,000) and
#: measured nothing.
#:
#: README's own audit list already gates this: the `letterhead` pass exists to
#: "sample bands per candidate author and count how many carry a printed mark at
#: all -- decides whether that pool is worth clustering", and it has never been
#: run.  Auto-admitting band classes did the thing that pass is there to gate.
#:
#: So UCSF's 197k pages stay in the corpus as distractors, which is what 92% of
#: them were for anyway, and its letterhead candidates keep their band marks with
#: `class_id=None` so the human pass can still use them.  Making bands usable
#: needs a descriptor that looks at the mark rather than the strip -- the
#: `siglip` backend is the obvious candidate and wants its own sweep.
CLUSTERED_SOURCES: tuple[str, ...] = ("spods", "staver", "tobacco800")


def cluster_threshold_for(source: str) -> float:
    """Merge threshold for *source*.  An explicit env override wins for all."""
    if os.environ.get("VTS_DOCMARKS_CLUSTER_THRESHOLD"):
        return CLUSTER_THRESHOLD
    return CLUSTER_THRESHOLD_BY_SOURCE.get(source, CLUSTER_THRESHOLD)


#: Sources whose pages are ALWAYS in every tier, whatever the budget.
#:
#: These carry the corpus's known negatives -- "same scanner, same paper, same
#: era, known clean", per README, and the hardest negatives a class can be
#: scored against.  The 2026-09-01 build dropped 129 of them over the tier
#: budget to make room for UCSF distractors, which trades the hardest negatives
#: for the easiest ones and is exactly backwards.  There are only ~2,650 of
#: them; they fit in every tier including `s`.
ANCHOR_SOURCES: frozenset[str] = frozenset({"spods", "staver", "tobacco800"})

# --------------------------------------------------------------------------
# Synthesis (layer 3)
# --------------------------------------------------------------------------

#: Instances generated per synthetic class.
SYNTH_INSTANCES_PER_CLASS = int(os.environ.get("VTS_DOCMARKS_SYNTH_PER_CLASS", "30"))

#: Longest-side pixel sizes the pasted mark is drawn from, log-uniformly.  The
#: band spans the 2026-07-13 study's measured cliff (nothing works below ~32 px;
#: 128-256 px is where SIFT recovers) so a sweep can locate it rather than
#: straddle it.
SYNTH_SIZE_PX = (24, 320)

#: Rotation range in degrees.  Scanned marks are near-upright but not exactly:
#: a rubber stamp is applied by hand, a letterhead is not.
SYNTH_ROTATION_DEG = (-8.0, 8.0)

#: Seed for every random choice in synthesis.  One seed, one corpus.
SYNTH_SEED = int(os.environ.get("VTS_DOCMARKS_SYNTH_SEED", "20260825"))


def eligible_distractor(class_source: str, page_source: str, page_industry: str | None = None) -> bool:
    """Is *page_source* safe to score as a distractor for a *class_source* class?

    ``page_industry`` qualifies UCSF pages, so that Tobacco800 classes can use
    UCSF's non-tobacco industries while excluding the tobacco archive they
    overlap with.
    """
    banned = CONTAMINATES.get(class_source, frozenset())
    if page_source in banned:
        return False
    if page_source == "ucsf" and page_industry and f"ucsf:{page_industry}" in banned:
        return False
    return True


# --------------------------------------------------------------------------
# The merge slate (human pass)
# --------------------------------------------------------------------------
#
# The `confusable` pass adjudicates one pair per sheet, which is correct and
# unusable at scale: 60 admitted classes is 1,770 pairs, so the reviewer is
# handed 1,770 PNGs to open.  The `merge` slate asks the same question in the
# shape a person can actually answer -- every class on a few contact sheets,
# similarity-ordered and numbered, answered as a list of index sets -- and
# compiles that answer back into exactly the same same/different verdicts.

#: Instances shown per class cell on the slate.  One exemplar is denser and
#: fits every class on a single page, but a merge call then rests entirely on
#: one crop being representative of its class -- which is the assumption the
#: membership pass exists because we do not trust.  Three is the smallest
#: number that shows within-class variation.
MERGE_SLATE_INSTANCES = int(os.environ.get("VTS_DOCMARKS_MERGE_INSTANCES", "3"))

#: Cells per slate sheet.  4x6 at a 3-up cell is ~1,500x1,400 px: legible at
#: 100% on a desktop, which is the only place these are ever looked at.
MERGE_SLATE_COLS = 4
MERGE_SLATE_ROWS = 6

#: How many of the nearest class pairs get their own explicit side-by-side
#: sheet, and thereby become eligible to be recorded as adjudicated.
#:
#: This number is the honesty budget for the closed-world rule.  A reviewer who
#: works a whole slate has genuinely compared the pairs that sit next to each
#: other and the pairs on the appendix; they have *not* compared all 1,770, and
#: recording the far ones as adjudicated would assert a decision nobody made.
#: So only these pairs are separated on a `REVIEWED-ALL` slate.  Raise it to
#: buy more of the matrix at the cost of more sheets to work through.
MERGE_SLATE_NEAR_PAIRS = int(os.environ.get("VTS_DOCMARKS_MERGE_NEAR_PAIRS", "120"))

#: Pairs per appendix sheet.
MERGE_PAIRS_PER_SHEET = 12

# --------------------------------------------------------------------------- #
# The audit's second opinion
# --------------------------------------------------------------------------- #
#
# `phash` is the right descriptor for *clustering* 200k pages and the wrong one
# for *auditing* the result, and #3600 measured the gap: on corpus v2 the one
# literal duplicate on the slate ranked 83rd of 120 in the near-pair appendix,
# behind 82 pairs of stamps nobody would confuse, while two internally-mixed
# classes took 37% of the appendix between them.  A perceptual hash of a blue
# rubber stamp on white paper measures ink layout, and two different stamps of
# the same size in the same typeface have nearly the same ink layout -- the
# failure already on record for UCSF letterhead bands.
#
# The audit can afford what the build cannot: it runs over ~1.5k crops, not
# 200k pages, so a GPU embedder is minutes.  Vectors are cached, so only the
# embed step needs a card and every render afterwards stays on `cpu`.

#: The embedder the audit's similarity questions are asked with.  Not the
#: clustering's descriptor: changing that would change what the corpus *is*,
#: and the roster's classes were admitted under `phash`.
AUDIT_EMBEDDER = os.environ.get("VTS_DOCMARKS_AUDIT_EMBEDDER", "siglip2_l")

#: Instances embedded per class.  A centroid stops moving long before a class's
#: 30th instance, and the cap is recorded beside the vectors so a later reader
#: knows the centroid is over a sample.
AUDIT_MAX_PER_CLASS = int(os.environ.get("VTS_DOCMARKS_AUDIT_MAX_PER_CLASS", "24"))

#: Cosine-distance thresholds the within-class split proposal is swept over.
#: Reported as a sweep and never as a single verdict: the operating point is a
#: property of this corpus and this embedder, and quietly picking one is how
#: `CLUSTER_THRESHOLD`'s 0.16 outlived the mark decomposition it was measured
#: on (#3366).
AUDIT_SPLIT_SWEEP = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40)

#: The within-class spread at which a class is reported as internally **mixed**.
#:
#: The rank of a class's own centroid answers "is this query crop an outlier",
#: and #3610 is the case where nothing answered "is this class more than one
#: mark": `staver/stamp_stampds-00156_0` holds five marks and scores rank 0,
#: distance 0.078 -- correctly, because its query crop is a good instance of the
#: 16-strong mark it is drawn from.  A rank cannot see the other four.
#:
#: Not a fresh magic number: it is the loosest threshold in `AUDIT_SPLIT_SWEEP`,
#: so "mixed" means *the class's two most distant instances sit further apart
#: than the loosest cut we would ever call one mark*.  Tying it to the sweep is
#: what stops it drifting away from the sweep the way `CLUSTER_THRESHOLD`'s 0.16
#: drifted away from its decomposition (#3366).
AUDIT_MIXED_MAX_WITHIN = float(os.environ.get("VTS_DOCMARKS_AUDIT_MIXED_MAX_WITHIN", str(max(AUDIT_SPLIT_SWEEP))))

#: Which descriptor the slate orders itself by.  `phash` stays the default so a
#: slate renders with no cache and no card; `siglip2_l` requires
#: `siglip_audit.py --embed` to have run, and refuses rather than silently
#: falling back -- a slate whose ordering is not the one asked for is a slate
#: whose appendix means something other than it says.
SLATE_DESCRIPTOR = os.environ.get("VTS_DOCMARKS_SLATE_DESCRIPTOR", "phash")
