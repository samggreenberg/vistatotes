"""Pre-registered grid for the calibration study (issue #2781).

One place the prepare stage, the SLURM array indexer, and the analyzer all agree
on.  See ``docs/plans/calibration-experiment.md`` for the design.

Arms (each an ``(embedder, style)`` pair).  An embedder is either a single
registered name or a **paired** ``"<text>+<learn>"`` name (see :data:`PAIR_SEP`),
which opens on one space's text sort and learns in another's:

* ``visual_genome_m`` (boxed; ground-truth regions):
  ``siglip`` / ``siglip_l`` × ``whole_image`` (row-wise conformal), and
  ``dinov3_patch`` × ``max_patch`` (grouped bag calibration).  The raw-patch
  tree geometry ``max_patch_pca_hac`` and its ``topk`` / ``pnorm`` re-pools were
  #2781 arms and are off by default now that both questions are closed; see
  :data:`PATCH_STYLES` and :data:`REPOOL_VARIANTS`.

  **Only the ``dinov3_patch`` arms actually region-vote.** Region voting needs a
  stored ``patch_grid`` to pool the dragged box and ``patch_regions`` to
  max-pool at scoring time; the single-vector embedders have neither, so
  :data:`REGION_VOTING_BY_DATASET` degrades to whole-image training *and*
  whole-image scoring for them, and they blend under the **binary** schedule.
  This docstring previously called the whole set "region voting", which is how
  #2877 came to report a binary-voting environment as a region-voting one.
* ``caltech101_m`` (binary voting; boxless): ``siglip`` / ``siglip_l`` ×
  ``whole_image`` only — the ordinary row-wise conformal path most users hit.

Category selection (scale-band on the boxed VG set, prevalence-spread on the
boxless Caltech set) is copied from the Max-Patch runner so the two studies
select the *same* categories and their pickles are interchangeable.
"""

from __future__ import annotations

import math
import os
import zlib

# --- Datasets and their embedders (arms differ per dataset) ---
DATASETS = os.environ.get("CALIB_DATASETS", "visual_genome_m,caltech101_m").split(",")

#: Text a user would type, for datasets that exist only inside this experiment.
#:
#: ``vtscore.eval.config.EVAL_DATASETS`` is the app's demo-dataset query table and
#: is asserted to hold only real demo datasets, so purpose-built fixtures like
#: ``vg_scale`` cannot live there.  They still need a query: Autopilot's Good
#: phase seeds from a **text sort**, and seeding it any other way measures a flow
#: no user has (see lessons/2026-08-26-the-harness-seeded-from-a-crop.md).
#:
#: Keyed ``dataset -> {category: text}``.  For banded datasets the band is a
#: property of the *cell*, not the query: someone hunting a distant boat and
#: someone hunting a close one both type "boat", so one text serves all three
#: bands and only the labels differ.
_VG_SCALE_TEXTS = {
    "backpack": "a backpack",
    "bicycle": "a bicycle",
    "bird": "a bird",
    "boat": "a boat on the water",
    "book": "a book",
    "bus": "a bus",
    "clock": "a clock",
    "dog": "a dog",
    "kite": "a kite in the sky",
    "knife": "a knife",
    "stop sign": "a stop sign",
    "umbrella": "an umbrella",
    # The thirteen #3588 added, taken VERBATIM from `_COCO_TEXTS` below -- which
    # is where the twelve above came from too: every one of them is byte-identical
    # to its COCO entry, scene qualifiers included (`a boat on the water`, `a kite
    # in the sky`). Reusing rather than re-writing is what keeps the opening from
    # becoming a second, uncontrolled axis: a hand-tuned query for one class and a
    # plain one for another is an arm-dependent seeding difference, which is the
    # confound #3278 added the region pair to remove.
    #
    # **The query is what a USER TYPES; the class rule is what a REVIEWER
    # APPLIES.** They deliberately do not match, and two of these show why: `cup`
    # is `cup` U `wine glass` (SCALE_CLASS_MERGES) and `truck` excludes SUVs
    # (SCALE_CLASS_RULES), but nobody hunting either types the boundary into the
    # search box. Encoding the ruling here would measure an opening no user has.
    "truck": "a truck",
    "car": "a car on the street",
    "fork": "a fork",
    "spoon": "a spoon",
    "cup": "a cup",
    "bowl": "a bowl",
    "bottle": "a bottle",
    "vase": "a vase",
    "bench": "a bench",
    "chair": "a chair",
    "sink": "a sink",
    "cell phone": "a cell phone",
    "fire hydrant": "a fire hydrant",
}

#: COCO-2017-val's 80 categories as **typed queries**.
#:
#: ``coco_val`` is assembled by ``build_coco_pickle.py`` and so is an experiment
#: fixture, not a demo dataset - ``vtscore.eval.config.EVAL_DATASETS`` is
#: asserted to hold only real demo datasets, so its query table has to live
#: here.  Without one, ``_seed_query_text`` returns "" and the autopilot takes
#: its *other* documented start (three random known-goods), which is the gap
#: ``lessons/2026-08-26-the-harness-seeded-from-a-crop.md`` closed for
#: ``vg_scale`` and explicitly left open here.  #3267 is a study **about the
#: text sort**, so the gap is load-bearing rather than cosmetic.
#:
#: The text is what a user would plausibly type, not the raw label: COCO's
#: category strings are terse ("tv", "remote", "skis") and several are ambiguous
#: out of context ("mouse", "orange", "remote"), where a bare noun would rank a
#: different concept and the study would measure the query rather than the
#: opening.
_COCO_TEXTS = {
    "person": "a person",
    "bicycle": "a bicycle",
    "car": "a car on the street",
    "motorcycle": "a motorcycle",
    "airplane": "an airplane",
    "bus": "a bus",
    "train": "a train",
    "truck": "a truck",
    "boat": "a boat on the water",
    "traffic light": "a traffic light",
    "fire hydrant": "a fire hydrant",
    "stop sign": "a stop sign",
    "parking meter": "a parking meter",
    "bench": "a bench",
    "bird": "a bird",
    "cat": "a cat",
    "dog": "a dog",
    "horse": "a horse",
    "sheep": "a sheep",
    "cow": "a cow",
    "elephant": "an elephant",
    "bear": "a bear",
    "zebra": "a zebra",
    "giraffe": "a giraffe",
    "backpack": "a backpack",
    "umbrella": "an umbrella",
    "handbag": "a handbag",
    "tie": "a person wearing a necktie",
    "suitcase": "a suitcase",
    "frisbee": "a frisbee",
    "skis": "a pair of skis",
    "snowboard": "a snowboard",
    "sports ball": "a sports ball",
    "kite": "a kite in the sky",
    "baseball bat": "a baseball bat",
    "baseball glove": "a baseball glove",
    "skateboard": "a skateboard",
    "surfboard": "a surfboard",
    "tennis racket": "a tennis racket",
    "bottle": "a bottle",
    "wine glass": "a wine glass",
    "cup": "a cup",
    "fork": "a fork",
    "knife": "a knife",
    "spoon": "a spoon",
    "bowl": "a bowl",
    "banana": "a banana",
    "apple": "an apple",
    "sandwich": "a sandwich",
    "orange": "an orange fruit",
    "broccoli": "broccoli",
    "carrot": "a carrot",
    "hot dog": "a hot dog",
    "pizza": "a pizza",
    "donut": "a donut",
    "cake": "a cake",
    "chair": "a chair",
    "couch": "a couch",
    "potted plant": "a potted plant",
    "bed": "a bed",
    "dining table": "a dining table",
    "toilet": "a toilet",
    "tv": "a television screen",
    "laptop": "a laptop computer",
    "mouse": "a computer mouse",
    "remote": "a tv remote control",
    "keyboard": "a computer keyboard",
    "cell phone": "a cell phone",
    "microwave": "a microwave oven",
    "oven": "an oven",
    "toaster": "a toaster",
    "sink": "a sink",
    "refrigerator": "a refrigerator",
    "book": "a book",
    "clock": "a clock",
    "vase": "a vase",
    "scissors": "a pair of scissors",
    "teddy bear": "a teddy bear",
    "hair drier": "a hair dryer",
    "toothbrush": "a toothbrush",
}


EXPERIMENT_QUERIES: dict[str, dict[str, str]] = {
    "vg_scale": {
        f"{cls}@{band}": text for cls, text in _VG_SCALE_TEXTS.items() for band in ("small", "medium", "large")
    },
    "coco_val": _COCO_TEXTS,
    # `vg_scale` with the band collapsed away (#3115): the same verified images
    # under the bare class name, so the same twelve texts serve it.  Without this
    # entry every `vg_scale_any` cell falls back to the known-good start -- which
    # is what it silently did until #3278 paired its region arm and the pair's
    # own guard refused to run, since a pair exists FOR the text sort.
    "vg_scale_any": dict(_VG_SCALE_TEXTS),
    # `vg_scale_deep` (#3547) is `vg_scale_any`'s construction designated
    # band-free and 3x deeper, on the SAME twelve class names -- so it takes the
    # same texts. Sharing `_VG_SCALE_TEXTS` rather than copying it is the point:
    # a query that drifted between the two would confound the pile axis with a
    # seeding axis in the one comparison the deep study exists to make.
    "vg_scale_deep": dict(_VG_SCALE_TEXTS),
    # The box-size bands are built from the FULL Visual Genome vocabulary, so
    # their categories are VG's, not COCO's.  A band is a property of the cell -
    # someone hunting a small bus and someone hunting a large one both type
    # "a bus" - so the same text serves all three.
    "vg_box_small": dict(_VG_SCALE_TEXTS),
    "vg_box_medium": dict(_VG_SCALE_TEXTS),
    "vg_box_large": dict(_VG_SCALE_TEXTS),
}


#: Whether the simulated user's opening query is embedded through the
#: embedder's ``description_wrappers`` ensemble ("Enrich Sort Descriptions") or
#: plainly.
#:
#: This must track the app's shipped default for ``enrich_descriptions``
#: (``vtsearch/settings_models.py``), which is ``False``: the opening and the
#: click-0 anchor are supposed to be the sort a real user sees, and a harness
#: that quietly embeds plainly while the app enriches is measuring a different
#: opening than the one it reports.  Every ``embed_text_query`` call in this
#: harness passes it explicitly rather than leaning on the library default, so
#: that if the app's default ever moves there is exactly one line to change and
#: it is grep-able (#3341).
#:
#: Note that under #3341 enrichment is per-embedder: ``siglip`` -- this study's
#: text half -- returns no wrappers at all, so flipping this to ``True`` is a
#: no-op there rather than a silent re-ranking.
SEED_ENRICH = os.environ.get("CALIB_SEED_ENRICH", "0") == "1"


def seed_query_text(dataset: str, category: str) -> str:
    """The text a user would type to find *category* in *dataset*, or "".

    One implementation, read by three callers that must agree: ``run_cells.py``
    (which seeds the run), ``prepare_data.py`` (which may filter selection to
    categories that have one), and ``preflight.sh`` (which refuses to launch a
    text-sort study whose cells would silently take the known-good start).  Two
    copies of this lookup is how a preflight check comes to pass while the run
    does something else.

    :data:`EXPERIMENT_QUERIES` wins over the app's demo-dataset table so a
    fixture can override, but neither is required.
    """
    local = EXPERIMENT_QUERIES.get(dataset) or {}
    if category in local:
        return local[category]

    from vtscore.eval.config import EVAL_DATASETS  # noqa: PLC0415

    info = EVAL_DATASETS.get(dataset)
    if not info:
        return ""
    for query in info["queries"]:
        if query.target_category == category:
            return query.text
    return ""


DATASET_EMBEDDERS: dict[str, list[str]] = {
    "visual_genome_m": os.environ.get("CALIB_VG_EMBEDDERS", "siglip,siglip2_l,dinov3_patch").split(","),
    "caltech101_m": os.environ.get("CALIB_CALTECH_EMBEDDERS", "siglip,siglip2_l,dinov3_patch").split(","),
    # COCO-2017-val, assembled from the #2790 sweep cache by
    # ``build_coco_pickle.py`` (issue #2841).  Whole-image embedders only: that
    # cache holds each image's whole vector and its HAC region vectors but not
    # the raw patch grid, so no region-voting style can be built from it.
    "coco_val": os.environ.get("CALIB_COCO_EMBEDDERS", "siglip,siglip2_l,dinov3_patch").split(","),
    # Box-size-banded VG (PR #3123, still unmerged).  Only the embedder table is
    # needed: the pile already holds every cell, and prepare_data reads a cell
    # pickle in place when it exists.  Registering them here rather than basing
    # the run on that branch keeps the code at dev HEAD.
    "vg_box_small": os.environ.get("CALIB_VGBOX_EMBEDDERS", "siglip,siglip2_l,dinov3_patch").split(","),
    "vg_box_medium": os.environ.get("CALIB_VGBOX_EMBEDDERS", "siglip,siglip2_l,dinov3_patch").split(","),
    "vg_box_large": os.environ.get("CALIB_VGBOX_EMBEDDERS", "siglip,siglip2_l,dinov3_patch").split(","),
    # #3156's verified scale dataset with the box-size band collapsed away
    # (#3115): 12 hand-checked classes, 300 positives each against one shared
    # 3900-image negative pool, so **prevalence is identical in every cell**.
    # That is what a calibration study wants and what `visual_genome_m` is not -
    # its selected categories run 25 to 1645 positives, and the thin ones
    # produce cells with no trainable step at all.  Both embedders are in the
    # pile, so one dataset supplies BOTH voting modes and the mode contrast
    # stops being confounded with the dataset.
    # The region arm is the PAIR, for the reason spelled out on `vg_scale` below:
    # bare `dinov3_patch` cannot open on a text sort, and `EXPERIMENT_QUERIES`
    # gives this dataset a query for every category precisely so it can.
    "vg_scale_any": os.environ.get("CALIB_VGSCALE_EMBEDDERS", "siglip,siglip+dinov3_patch").split(","),
    # #3547's deep pile. Its own env var, defaulting to `siglip` ALONE, because
    # only the binary half was built: the deep question is about the shipped arm
    # (`siglip x whole_image`) and a `dinov3_patch` cell at 22k medias is ~7 GB.
    # Defaulting to the pair here would enumerate cells whose pickle does not
    # exist, which fails late and per-cell rather than at prepare.
    "vg_scale_deep": os.environ.get("CALIB_VGSCALE_DEEP_EMBEDDERS", "siglip").split(","),
    # The same-class-across-bands set (#3156): one class list at three box
    # scales, so a small-vs-large difference is about size rather than about
    # which words happen to live at which size. Its categories are
    # band-suffixed (`bus@small`), and every one of them is the experiment --
    # see CATEGORY_MODE "all".
    #
    # The region-voting arm is the PAIR `siglip+dinov3_patch` rather than bare
    # `dinov3_patch`: DINOv3 has no text tower, so on its own it opens on three
    # random known-goods while the whole-image arms open on a text sort, and the
    # voting-mode contrast carries a seeding contrast inside it. See PAIR_SEP.
    #
    # The DEFAULT carries the pair, not just the launchers. Every caller that
    # sets `CALIB_VGSCALE_EMBEDDERS` already names it, so the bare fallback was
    # reachable only by the callers that name nothing -- a direct
    # `run_cells.py --index`, a preflight run without the env -- which are
    # exactly the ones with no launcher comment to warn them.
    "vg_scale": os.environ.get("CALIB_VGSCALE_EMBEDDERS", "siglip,siglip+dinov3_patch").split(","),
}

#: Region voting (drag the ground-truth box) only makes sense on a boxed dataset.
#: COCO *is* boxed, but its cached vectors cannot feed a patch style (see above),
#: so it runs as a second binary-voting dataset - which is exactly the axis
#: #2841 asks about separately from region voting.
#:
#: NOTE: this flag is necessary but **not sufficient**.  It is per-*dataset*,
#: while whether region voting actually happens is per-*embedder*: a boxed
#: dataset paired with a single-vector embedder silently runs as binary voting
#: (no ``patch_grid`` to pool, no ``patch_regions`` to max-pool).
#: ``simulate_voting_iterations`` now warns when that combination is requested.
REGION_VOTING_BY_DATASET: dict[str, bool] = {
    "visual_genome_m": True,
    "caltech101_m": False,
    "coco_val": False,
}

#: Whether a dataset carries ground-truth region boxes.  This is the *dataset*
#: half of region voting; the other half is the embedder (it must emit a patch
#: grid).  Kept separate from :data:`REGION_VOTING_BY_DATASET` - which older
#: analyzers read as a per-dataset label - so that marking a dataset boxed here
#: cannot silently change another study's control selection.
#:
#: ``coco_val`` is boxed and, since the pile gained ``coco_val__dinov3_patch``,
#: is now genuinely region-capable - the second region-voting environment the
#: #2905 confound needed.  The older map still says False because it predates
#: that cell.
BOXED_BY_DATASET: dict[str, bool] = {
    "visual_genome_m": True,
    "caltech101_m": False,
    "coco_val": True,
    "vg_box_small": True,
    "vg_box_medium": True,
    "vg_box_large": True,
    # The same-class-across-bands set (#3156). Omitting it here does not fail
    # loudly: `styles_for` reads a missing entry as boxless and silently falls a
    # patch embedder back to whole_image, which is the right behaviour for a
    # genuinely boxless dataset and the wrong one for a boxed dataset the table
    # has simply never heard of. That cost a full 108-cell arm, and it is the
    # same shape as #2877/#2897/#2905: the premise has to be asserted per cell,
    # and here the premise itself lived in a second registry that drifted from
    # `pile_config.DATASETS`.
    "vg_scale": True,
    "vg_scale_any": True,
    # Boxed like its sibling -- the boxes are `vg_scale`'s, carried through the
    # same passes. Necessary but not sufficient: no patch cell is built, so in
    # practice every `vg_scale_deep` arm binary-votes (see the note above).
    "vg_scale_deep": True,
}


def region_voting_for(dataset: str, embedder: str) -> bool:
    """Region voting needs **both** halves: boxes (dataset) and a patch grid (embedder).

    Asserting the premise per *cell* rather than per dataset is the control for
    the mis-specification behind #2877, #2897 and #2905, where a boxed dataset
    paired with a single-vector embedder was reported as a region-voting arm
    while it silently ran as binary voting.
    """
    return BOXED_BY_DATASET.get(dataset, False) and is_patch_embedder(embedder)


# --- Styles per embedder kind ---
#: Patch geometries every patch-capable arm runs.  **The shipped one, alone**
#: (kept a literal rather than importing ``PRODUCTION_PATCH_STYLE``, because
#: this module is imported by tooling that has no vtscore tree; preflight
#: check 12 is what holds the two together).
#:
#: This default used to be ``max_patch,max_patch_pca_hac``, because the #2781
#: study wanted that contrast - and a study default is not a shipped default
#: (``lessons/2026-08-12-a-study-default-is-not-a-shipped-default.md``).
#: ``max_patch_pca_hac`` lost the Max-Patch study at the operating point
#: (PR #2749) and #2886 removed the HAC tree it delegates to from ingest, so
#: carrying it here doubled the GPU cost of every patch cell to measure a
#: geometry production does not have.  A study that wants the tree arm back
#: adds it explicitly and declares the divergence.
PATCH_STYLES = os.environ.get("CALIB_PATCH_STYLES", "max_patch").split(",")
SINGLE_STYLES = ["whole_image"]

#: The style whose per-node scores get re-pooled into the remedial arms.
REPOOL_STYLE = "max_patch_pca_hac"
#: The #2781 re-pool arms, **off by default: the question is closed.**  Both
#: pre-registered fixed re-pools failed (``docs/plans/set-scorer-experiment.md``:
#: ``topk`` made regret worse, sign-corrected ``pnorm`` closed ~21% of the gap),
#: and every analyzer filters ``pool_variant`` back down to the base rows
#: (``_cells_io.BASE_POOL_VARIANTS``), so the arms cost a re-calibration and a
#: re-pool per step per cell to produce rows nothing reads.  The live version of
#: the question is learned set-pooling, not another fixed rule.
REPOOL_VARIANTS = [v for v in os.environ.get("CALIB_REPOOL_VARIANTS", "").split(",") if v]
REPOOL_TOPK = int(os.environ.get("CALIB_REPOOL_TOPK", "4"))

#: Inclusion values the fold orderings are re-thresholded at for the budget sweep.
INCLUSION_SWEEP_KS = [int(k) for k in os.environ.get("CALIB_SWEEP_KS", "-4,-2,-1,0,1,2,4").split(",")]

# --- Sizing knobs ---
SEEDS = list(range(int(os.environ.get("CALIB_N_SEEDS", "4"))))
MAX_STEPS = int(os.environ.get("CALIB_MAX_STEPS", "150"))
EXEMPLAR_CANDIDATES = int(os.environ.get("CALIB_EXEMPLAR_CANDIDATES", "8"))

# --- Production-faithful fixed choices (pre-registered) ---
INCLUSION = 0
#: Share of a cell's medias that become the **simulation set** - the pool the
#: user votes out of AND the haystack the threshold's population estimate is
#: fitted on; the rest is the held-out test set every metric is scored against.
#:
#: 0.5 for every study before #3312, where it was a constant rather than a knob.
#: It became one because it is that study's *instrument*: the #3308 voted-media
#: exclusion is bounded in size by the votes' share of the haystack, so the
#: only way to place cells on both sides of the effect - and on both sides of
#: the ``EXCLUSION_MIN_REMAINDER`` floor - is to move the haystack's size while
#: holding the horizon fixed.  At 0.5 on ``vg_scale_any`` a 150-click run votes
#: ~7% of a ~2100-media haystack and the floor never binds; at 0.08 it votes
#: ~45% of ~340 and the floor decides most of the run.
#:
#: Shrinking it cuts both ways and the trade is deliberate: the test set grows
#: (tighter metrics) while the sim set's positive count falls with it, so a
#: small fraction needs ``CALIB_MIN_SIM_POSITIVES`` to keep cells that cannot
#: seed out of the frame.  The split is a plain permutation, not stratified.
SIM_FRACTION = float(os.environ.get("CALIB_SIM_FRACTION", "0.5"))
if not 0.0 < SIM_FRACTION < 1.0:
    raise ValueError(f"CALIB_SIM_FRACTION={SIM_FRACTION} must lie strictly in (0, 1)")
#: Number of cross-calibration folds.  Production is 2, which is why it was a
#: constant - but 2 folds make the fold-anchored ``qmean``/``qmedian`` combine
#: arms byte-identical, so the combine question cannot be asked without moving
#: it.  Changing this changes the *trajectory* (different splits, different
#: per-fold models), so a folds contrast is a run-level A/B, not a paired arm.
CALIBRATE_COUNT = int(os.environ.get("CALIB_CALIBRATE_COUNT", "2"))
#: Share of each calibration fold's labelset held out to READ the threshold from;
#: the rest trains that fold's model.  0.5 since it was introduced, and never
#: measured - the obvious default, not a result (issue #3287).
#:
#: It is a genuine trade-off rather than a "more is better" knob.  More Train
#: gives better fold models, so their held-out orderings are a closer proxy for
#: the final model's, but leaves fewer anchors and a coarser conformal quantile.
#: More Calibrate gives finer quantiles but drifts the fold models further from
#: the final model, which always trains on ALL votes - so the scale the cut is
#: read on is less like the scale it is applied on.
#:
#: Like :data:`CALIBRATE_COUNT`, moving this moves the *trajectory* (different
#: splits, different fold models, a different threshold and therefore a
#: different Hard pick), so a fraction contrast is a run-level A/B and NOT a
#: paired arm re-cut inside one run.
#: ``None`` (env unset) = the app's per-space default: the harness resolves it
#: per cell through ``production_split_for`` (0.3 single-vector / 0.5 patch,
#: issue #3290), exactly as a live detector does.  Pinning a scalar here is a
#: divergence preflight check 12 requires the study to declare.
_CALIBRATION_FRACTION_ENV = os.environ.get("CALIB_CALIBRATION_FRACTION", "").strip()
CALIBRATION_FRACTION: float | None = float(_CALIBRATION_FRACTION_ENV) if _CALIBRATION_FRACTION_ENV else None
if CALIBRATION_FRACTION is not None and not 0.0 < CALIBRATION_FRACTION < 1.0:
    raise ValueError(f"CALIB_CALIBRATION_FRACTION={CALIBRATION_FRACTION} must lie strictly in (0, 1)")
#: The #3312 arm axis: the minimum unlabeled remainder at which the #3308
#: voted-media exclusion still applies.  One scalar spans the whole axis, so
#: the arms are ordered and no sentinel is needed:
#:
#:   ``off``    -> ``math.inf`` - the exclusion never fires (pre-#3308 baseline)
#:   ``always`` -> ``0``        - unconditional exclusion, no floor
#:   ``<int>``  -> that floor   - e.g. ``60``, the shipped constant
#:   unset      -> ``None``     - resolve through the app's own
#:                               ``resolve_exclusion_floor``, i.e. whatever a
#:                               live detector does.  This is the DEFAULT and
#:                               is what keeps the harness's default arm equal
#:                               to production; pinning anything else is a
#:                               divergence preflight check 12 requires the
#:                               study to declare.
#:
#: Like every other knob upstream of the threshold, moving this moves the
#: *trajectory* - a different cut is a different acquisition rank, which is a
#: different next vote - so an exclusion contrast is a run-level A/B and NOT a
#: paired arm re-cut inside one run.
_EXCLUDE_VOTED_ENV = os.environ.get("CALIB_EXCLUDE_VOTED", "").strip().lower()
if _EXCLUDE_VOTED_ENV in ("", "default", "app"):
    EXCLUSION_MIN_REMAINDER: float | None = None
elif _EXCLUDE_VOTED_ENV in ("off", "never", "inf"):
    EXCLUSION_MIN_REMAINDER = math.inf
elif _EXCLUDE_VOTED_ENV in ("always", "0"):
    EXCLUSION_MIN_REMAINDER = 0.0
else:
    try:
        EXCLUSION_MIN_REMAINDER = float(_EXCLUDE_VOTED_ENV)
    except ValueError:
        raise ValueError(
            f"CALIB_EXCLUDE_VOTED={_EXCLUDE_VOTED_ENV!r} is not one of "
            "'off' / 'always' / a non-negative number / unset (= the app's default)"
        ) from None
    if EXCLUSION_MIN_REMAINDER < 0:
        raise ValueError(f"CALIB_EXCLUDE_VOTED={_EXCLUDE_VOTED_ENV!r} must not be negative")


def exclusion_arm_name() -> str:
    """Short label for this run's exclusion arm, for logs and the cell column."""
    if EXCLUSION_MIN_REMAINDER is None:
        from vtscore.training.thresholds import resolve_exclusion_floor

        return f"app(f{resolve_exclusion_floor(None):g})"
    if EXCLUSION_MIN_REMAINDER == math.inf:
        return "off"
    if EXCLUSION_MIN_REMAINDER == 0.0:
        return "always"
    return f"f{EXCLUSION_MIN_REMAINDER:g}"


#: The **shipped** threshold path: fuse the haystack into the cut.  `docs/ML.md`:
#: "Every trained threshold fuses the haystack into the cut.  There is no
#: setting for this."  The app has had no switch since #2799, so an unset knob
#: has to resolve to the fused path or the harness measures a detector nobody
#: has.
#:
#: This defaulted to ``"0"`` until #3400 - the #2781-era pre-registered control,
#: which was a shipped default when #2781 ran and stopped being one when #2799
#: shipped.  Twenty-one launchers papered over it with an explicit
#: ``CALIB_SAFE_THRESHOLDS=1``; the ones that (correctly) set no behavioural knob
#: at all - ``launch_bench.sh`` and ``launch_scale.sh`` among them, both of which
#: say in their own headers that they ride shipped defaults on purpose - silently
#: measured the unfused control.  Same failure family as
#: ``lessons/2026-08-12-a-study-default-is-not-a-shipped-default.md``.
#:
#: A study that wants the unfused control sets ``=0`` and declares the
#: divergence to preflight (``--diverges safe_thresholds``).
SAFE_THRESHOLDS = os.environ.get("CALIB_SAFE_THRESHOLDS", "1") == "1"
MEDIA_TYPE = "image"

#: The #2852 anchored-mixture study (design + pre-registered decision rules:
#: ``docs/plans/population-anchored-calibration.md``) flips this on via
#: ``CALIB_ANCHORED=1``; every step then additionally emits the label-anchored,
#: fold-anchored ("cross-LabeledGMM"), and rank-transfer arm rows.  Requires
#: ``CALIB_SAFE_THRESHOLDS=1`` (the anchored arms ride the variant-row path).
ANCHORED = os.environ.get("CALIB_ANCHORED", "0") == "1"
#: Anchor-weight grid: each labelled score counts as this many haystack scores
#: in the anchored EM.  Log-spaced from "one label = one haystack point" to
#: "labels dominate the fit" - the fusion knob the sweep exists to place.
#:
#: **The shipped weight comes first.**  The grid was ``1,3,10,30,100`` when the
#: #2852 sweep placed the knob; #2861 read it, and ``FOLD_ANCHOR_WEIGHT`` shipped
#: at 0.3 - below the whole grid, so a re-run swept challengers with no shipped
#: arm to pair them against.  Log-spaced from the shipped value upward.
ANCHORED_WEIGHTS = [float(w) for w in os.environ.get("CALIB_ANCHORED_WEIGHTS", "0.3,1,3,10,30").split(",") if w]
#: #3329: emit the goodness-of-fit side frame (``__fitq.csv``).  Off by default
#: - it costs one extra unanchored EM fit per emitted step per geometry, which
#: is pure overhead for every study that is not asking whether the mixture fits.
FIT_QUALITY = os.environ.get("CALIB_FIT_QUALITY", "0") == "1"
#: Emit a goodness-of-fit row every this many steps (the first three always
#: emit).  5 keeps ~20 points on a 100-click horizon, which resolves the
#: trajectory at a fifth of the fit cost of every step.
FIT_QUALITY_STRIDE = int(os.environ.get("CALIB_FIT_QUALITY_STRIDE", "5"))

#: Cut rules re-cutting each anchored fit: the **shipped** rule first
#: (``FOLD_ANCHOR_CUT_RULE``, today ``mid_tilt``), then the plain midpoint and
#: the rate-optimal crossing (well-founded on an anchored fit, where the
#: components *are* the classes - the #2836 identification term is gone).
#: ``mid`` was the shipped rule when #2852 registered this grid and stopped
#: being one when ``mid_tilt`` shipped, which left the default grid with no
#: production arm in it - preflight check 12 has flagged it ever since.
ANCHORED_RULES = [r for r in os.environ.get("CALIB_ANCHORED_RULES", "mid_tilt,mid,rate").split(",") if r]
#: Fold-anchored + rank-transfer arms cost one sim-set scoring pass per
#: calibration fold per step; disable to keep only the cheap final-model arms.
ANCHORED_FOLD_ARMS = os.environ.get("CALIB_ANCHORED_FOLD_ARMS", "1") == "1"
#: How the fold arms combine per-fold cuts in quantile space.
ANCHORED_FOLD_COMBINES = [c for c in os.environ.get("CALIB_ANCHORED_FOLD_COMBINES", "qmean,qmedian").split(",") if c]
#: Vote-count checkpoints the anchored analyzer windows on (the plan's deep
#: regime; each window is (previous checkpoint, checkpoint]).
ANCHORED_CHECKPOINTS = [
    int(c) for c in os.environ.get("CALIB_ANCHORED_CHECKPOINTS", "20,50,100,200,300").split(",") if c
]

#: Inclusion values the **fold-anchored cut rules** are swept over (issue
#: #2865), into the ``__cutincl.csv`` side frame.  Empty (the default) = off,
#: and every other study runs exactly as before.
#:
#: Not to be confused with :data:`INCLUSION_SWEEP_KS`, which sweeps the
#: *conformal* rule's ``alpha(k)`` budget.  This one asks a different question -
#: which cut rule should answer the Inclusion knob at all - so its rows are
#: scored at their own ``k``, not at :data:`INCLUSION`.
#:
#: The arms are :data:`ANCHORED_WEIGHTS` x :data:`ANCHORED_RULES` x
#: :data:`ANCHORED_FOLD_COMBINES`, so a run that wants the #2865 candidate set
#: sets ``CALIB_ANCHORED_RULES=mid,mid_tilt,rate,cross_tilt,q_tilt``.
CUT_INCLUSION_KS = [int(k) for k in os.environ.get("CALIB_CUT_INCL_KS", "").split(",") if k.strip()]

#: Step sizes the eval-only ``q_tilt`` rule expands over - its free parameter,
#: in combined-fold-quantile units per inclusion step.  Every other rule ignores
#: this.  Empty = the single placeholder default in
#: ``vtscore.training.thresholds.FOLD_ANCHOR_QTILT_STEP``, which is a
#: placeholder and not a measurement; a run that means to *place* the parameter
#: has to pass a grid here.
CUT_INCLUSION_QTILT_STEPS = [float(s) for s in os.environ.get("CALIB_CUT_INCL_QTILT_STEPS", "").split(",") if s.strip()]

#: Calibration fold counts to score **counterfactually** at every step (issue
#: #2897), on top of whatever :data:`CALIBRATE_COUNT` the run lives under.
#: Empty (the default) = off, and every other study runs exactly as before.
#:
#: This is nearly free per *K* relative to what it buys, and exact rather than
#: approximate, because the folds are nested: each is an independent stratified
#: draw off one ``RandomState(42)`` at a size that does not depend on the count,
#: so the K folds a live ``calibrate_count=K`` run would train are the first K of
#: the Kmax folds this run trains.  One run therefore measures every K's regret
#: *and* every K's wall clock, paired within the step - which is why the fold
#: count, alone among the knobs here, does not need one full run per value to be
#: screened.  It still needs the A/B runs to close: K also steers acquisition.
#:
#: Cost: ``max(FOLD_COUNTS) - CALIBRATE_COUNT`` extra fold fits per step, so the
#: grid's *maximum* sets the price, not its length.  Size it from a real cell.
FOLD_COUNTS = [int(k) for k in os.environ.get("CALIB_FOLD_COUNTS", "").split(",") if k.strip()]

#: The **live** fold count as a function of the vote count (issue #3314):
#: ``"K@N"`` = ``K(n_votes) = K while n_votes < N, else`` :data:`CALIBRATE_COUNT`.
#: Empty (the default) = off, and every other study runs exactly as before -
#: this is the whole reason it is a separate knob rather than a widening of
#: :data:`CALIBRATE_COUNT`, which stays a scalar that no other launcher has to
#: learn a new grammar for.
#:
#: Unlike :data:`FOLD_COUNTS` this is NOT counterfactual: it moves the threshold
#: the app would have shown, which moves the acquisition cut, which moves the
#: votes.  It therefore needs a full run per schedule, exactly like
#: :data:`CALIBRATE_COUNT` itself - the screen cannot see it.
FOLD_COUNT_SCHEDULE = os.environ.get("CALIB_FOLD_COUNT_SCHEDULE", "").strip() or None

#: **Supervised-skyline** arms to measure once per run (issue #3322), splitting
#: ``oracle_cost`` into a learnability floor plus the headroom the interactive
#: loop left on the table.  Empty (the default) = off, and every other study runs
#: exactly as before.
#:
#: ``CALIB_SKYLINE_ARMS=skyline_train_full`` is the headline: the same head,
#: through the same trainer, on the entire sim split with full ground-truth
#: labels.  Add ``skyline_test_xfit`` for the cross-fitted test-side bracket
#: partner.  Both are vote-independent, so the price is one extra fit per arm per
#: cell rather than one per click - and both are scoped to the whole-image column
#: in v1 (a patch column's skyline needs a supervision decision that is still
#: open on #3321; the harness warns and skips there rather than improvising one).
SKYLINE_ARMS = [a.strip() for a in os.environ.get("CALIB_SKYLINE_ARMS", "").split(",") if a.strip()]

#: Which head each step trains (``vtscore.eval.voting_iterations.HEADS``).
#: Unset (the default) hands ``head=None`` to the harness, which resolves it to
#: the head the live detector actually trains — ``linear_svm``.  That is the
#: only setting a study's headline numbers can be read off, because questions
#: like #2799's ("should safe_thresholds be forced on for every VTSearch
#: user?") are answerable only on the shipped head.  Set ``CALIB_HEAD=linear``
#: for the logistic head the SVM replaced (#2790/#2809), or ``CALIB_HEAD=mlp``
#: for the historical auto-sized-MLP arm (#2781).
HEAD = os.environ.get("CALIB_HEAD") or None

#: Which safe-threshold mix-in schedule the run *lives* under (issue #2841).
#: This steers the trajectory - the blended threshold feeds Autopilot's Hard
#: pick - so an A/B between schedules needs one full run per value here.
BLEND_SCHEDULE = os.environ.get("CALIB_BLEND_SCHEDULE") or None


#: Extra schedules to score *counterfactually* on this run's trajectory, one
#: metric row each (tagged ``schedule``).  Free relative to the simulation, but
#: blind to acquisition feedback - the screen, not the verdict.  ``"all"``
#: expands to the whole registry.
def _schedule_variants() -> list[str]:
    raw = os.environ.get("CALIB_SCHEDULE_VARIANTS", "").strip()
    if not raw:
        return []
    if raw == "all":
        from vtscore.training.blend_schedules import schedule_names  # noqa: PLC0415

        return schedule_names()
    return [s.strip() for s in raw.split(",") if s.strip()]


SCHEDULE_VARIANTS = _schedule_variants()


def _opt_float(name: str) -> float | None:
    raw = os.environ.get(name, "").strip()
    return float(raw) if raw else None


#: Acquisition-side cut, as an **offset** from :data:`INCLUSION`.  The threshold
#: does two unrelated jobs - it is the reported decision line *and* the rank
#: position Autopilot's ``hard`` pick samples around.  This knob moves only the
#: second; reporting and every metric stay at :data:`INCLUSION`, so the arms
#: remain comparable.
#:
#: Direction is the opposite of the intuition from the cost weights, because the
#: pick reads the threshold as a **rank position**: a *negative* offset raises
#: the cut, moves it *up* the ranking, and returns *more* positives.
#:
#: Unset = whatever ``vtscore.training.thresholds`` currently ships, so an
#: unconfigured run measures what users actually get.  **Do not restate the
#: value here** - read it from the constant.  This comment previously named it,
#: went stale across two moves of the value, and was still claiming "-1 today"
#: three ships later; that is exactly how a study comes to mis-state its own
#: baseline.  ``0`` is the pre-#2876 control, one threshold doing both jobs.
#: **Fractional offsets are meaningful and are read as such** (issue #3319).
#: Inclusion is a log2 scale - one step doubles the price of a false alarm
#: relative to a miss - so a half step is a factor of sqrt(2), and every rule
#: that consumes it (the conformal quantiles, ``FoldAnchoredCut.threshold_at``)
#: is continuous in ``k``.  Parsing this as an int would silently refuse the
#: half-step grid rather than fail, so it is a float.
ACQ_INCLUSION_OFFSET = _opt_float("CALIB_ACQ_INCLUSION_OFFSET")
if ACQ_INCLUSION_OFFSET is None:
    from vtscore.training.thresholds import ACQUISITION_INCLUSION_OFFSET

    ACQ_INCLUSION_OFFSET = ACQUISITION_INCLUSION_OFFSET

#: The ``rank_pin`` arm: place the acquisition cut at this quantile of the
#: simulation-set scores directly, rather than by naming an inclusion.  Requires
#: ``CALIB_ACQ_INCLUSION_OFFSET=0``; the two name the same cut.
ACQ_RANK_PERCENTILE = _opt_float("CALIB_ACQ_RANK_PERCENTILE")

#: The **Autopilot opening** this arm runs (issue #3267), in the grammar of
#: :mod:`vtscore.eval.startup_schedule` - e.g. ``"n6@k-6,n6@k-2,n6@k0"``.
#:
#: Unset = the app's own opening (three positives off the top of the seed sort,
#: then four negatives at its cutoff), which is what every study before #3267
#: ran and what the `prod` control arm must keep running.  Do **not** write the
#: production spelling in here as a "default": a schedule string frozen in this
#: file goes stale the moment the app's opening moves, and the control arm would
#: then quietly stop being the control.  ``PRODUCTION_STARTUP`` exists for a run
#: that wants to name it explicitly, and is pinned against the app.
STARTUP_SCHEDULE = os.environ.get("CALIB_STARTUP_SCHEDULE", "").strip() or None

#: Emit the per-click pick log (``task_*__picks.csv``).  On by default for a
#: #3267 run and harmless everywhere else - one small row per vote.  It is the
#: only frame that records the **opening**, which emits no main row because no
#: detector exists yet, so an arm's mining behaviour is invisible without it.
EMIT_PICKS = os.environ.get("CALIB_EMIT_PICKS", "1") not in ("", "0")

#: Minimum positives a category must have **in the simulation half** to be kept.
#: A long-horizon run (#2841 follow-up: does pure x-cal ever overtake the blend?)
#: is bounded by positives, not pool size: once autopilot has exhausted them,
#: every further vote is a negative and the conformal positive-quantile stops
#: improving, so the tail of the curve would measure nothing.  0 disables the
#: filter, which is the behaviour of every run before the follow-up.
MIN_SIM_POSITIVES = int(os.environ.get("CALIB_MIN_SIM_POSITIVES", "0"))

# --- Category-selection parameters (copied from the Max-Patch runner) ---
_MIN_CATEGORY_COUNT = int(os.environ.get("CALIB_MIN_CAT_COUNT", "20"))
N_CATEGORIES = int(os.environ.get("CALIB_N_CATEGORIES", "6"))  # prevalence-spread count (Caltech)
N_PER_BAND = int(os.environ.get("CALIB_N_PER_BAND", "6"))  # scale-band count (VG)
MAX_VOTED_AREA = float(os.environ.get("CALIB_MAX_VOTED_AREA", "0.80"))

PATCH_AREA = 1 / 196  # one DINOv3 patch, ~0.51 % of the image
LEAF_AREA = 1 / 12  # smallest HAC leaf, ~8.3 %
SCALE_BANDS: list[tuple[str, float, float]] = [
    ("sub_patch", 0.0, PATCH_AREA),
    ("patch_to_leaf", PATCH_AREA, LEAF_AREA),
    ("leaf_to_4x", LEAF_AREA, 4 * LEAF_AREA),
    ("above_4x", 4 * LEAF_AREA, 1.01),
]


#: Separator in a **paired embedder** name, ``"<text>+<learn>"``.
#:
#: Autopilot asks an embedding space for two different things, and nothing says
#: they must be the same space:
#:
#: * the **opening** is a text sort - the user types a query and votes down the
#:   cosine ranking - which needs a *text tower*;
#: * everything after it - training the detector, pooling a dragged box, and
#:   re-sorting by the trained model - happens in the *media* space, which needs
#:   a *patch grid* for region voting and no text tower at all.
#:
#: ``dinov3_patch`` is the only patch-capable embedder in the pile and it has no
#: text tower, so on its own it can never take the app's real opening: every
#: DINOv3 cell falls back to the three-random-known-goods start.  That made the
#: voting-mode contrast a *seeding* contrast as well - the binary arms opened on
#: a text sort and the region arm did not - which is a confound in the axis the
#: study exists to measure, not a detail.
#:
#: ``siglip+dinov3_patch`` removes it: SigLIP (the shipped default embedder, and
#: the one whose text sort users actually see) ranks the typed query, and DINOv3
#: does every piece of learning - vector learning, region learning, and the
#: learn-sort.  Production would have to keep two vectors per image to ship it;
#: the pile already does, which is why it is measurable here first.
PAIR_SEP = "+"


def split_embedder(embedder: str) -> tuple[str, str]:
    """``"<text>+<learn>"`` -> ``(text_embedder, learn_embedder)``.

    A plain name is both of its own halves, so callers can split
    unconditionally instead of branching on whether a name is paired.
    """
    text, sep, learn = embedder.partition(PAIR_SEP)
    return (text, learn) if sep else (embedder, embedder)


def learn_embedder(embedder: str) -> str:
    """The space the detector trains, scores and sorts in - i.e. which pickle."""
    return split_embedder(embedder)[1]


def text_embedder(embedder: str) -> str:
    """The space the typed query is embedded in - i.e. which opening."""
    return split_embedder(embedder)[0]


def is_paired(embedder: str) -> bool:
    """True when the opening and the learning run in *different* spaces."""
    return PAIR_SEP in embedder


def is_patch_embedder(embedder: str) -> bool:
    """True for embedders that produce a patch grid + HAC tree.

    Reads the **learn** half: ``siglip+dinov3_patch`` region-votes because
    DINOv3 supplies the patches, whatever supplies the opening.
    """
    return learn_embedder(embedder).endswith("_patch")


def styles_for_embedder(embedder: str) -> list[str]:
    """The style arms an embedder participates in."""
    return PATCH_STYLES if is_patch_embedder(embedder) else SINGLE_STYLES


def styles_for(dataset: str, embedder: str) -> list[str]:
    """The style arms one ``(dataset, embedder)`` cell runs.

    A patch embedder only earns its patch styles where the dataset can supply
    box supervision.  On a **boxless** dataset a Good vote has no box to pool,
    so it falls back to the image-level vector, while every Bad vote floods the
    full-image row **plus ~197 raw patches** as negatives.  No patch row is ever
    positive, so the patch geometry teaches only "patch-like => negative", and
    max-pooling it at inference buys nothing while re-opening the asymmetry
    behind the boxless-``max_patch`` failure (perfect ranking, zero FPR,
    catastrophic FNR -- see :mod:`vtscore.eval.patch_styles`).

    If the user can only answer in booleans about whole images, then the Bad
    pile and the haystack should be whole images too.
    """
    if is_patch_embedder(embedder) and not BOXED_BY_DATASET.get(dataset, False):
        return SINGLE_STYLES
    return styles_for_embedder(embedder)


def embedders_for_dataset(dataset: str) -> list[str]:
    return DATASET_EMBEDDERS.get(dataset, [])


def pickle_name(dataset: str, embedder: str) -> str:
    """The cell pickle an arm loads its medias from.

    A paired arm loads the **learn** half's pickle, because that is where the
    vectors the detector is trained and scored on live.  The text half's pickle
    is opened separately, once, and only to rank the opening - see
    :func:`text_pickle_name` and ``run_cells._text_seed_scores``.
    """
    return f"{dataset}__{learn_embedder(embedder)}.pkl"


def text_pickle_name(dataset: str, embedder: str) -> str:
    """The pickle whose vectors the typed query is ranked against.

    Equal to :func:`pickle_name` for an unpaired embedder, which is what makes
    the paired path a generalisation of the ordinary one rather than a branch.
    """
    return f"{dataset}__{text_embedder(embedder)}.pkl"


def crops_basename(dataset: str, embedder: str) -> str:
    return f"{dataset}__{learn_embedder(embedder)}__crops"


def category_rng_seed(category: str) -> int:
    """Deterministic (process-stable) RNG seed for a category's exemplar draw."""
    return zlib.crc32(category.encode("utf-8")) & 0x7FFFFFFF


def select_categories_by_prevalence(category_counts: dict[str, int], n: int = N_CATEGORIES) -> list[str]:
    """Pick *n* categories spanning common->rare (boxless datasets)."""
    usable = sorted(
        ((c, n_) for c, n_ in category_counts.items() if n_ >= _MIN_CATEGORY_COUNT),
        key=lambda kv: kv[1],
        reverse=True,
    )
    if len(usable) <= n:
        return [c for c, _ in usable]
    idx = [round(i * (len(usable) - 1) / (n - 1)) for i in range(n)]
    return [usable[i][0] for i in sorted(set(idx))]


def select_categories_by_scale(
    medias: dict,
    category_counts: dict[str, int],
    n_per_band: int = N_PER_BAND,
) -> tuple[list[str], dict]:
    """Pick categories stratified by voted-box scale (boxed datasets)."""
    from vtscore.eval.labels import category_scale_stats  # noqa: PLC0415

    stats: dict[str, dict] = {}
    dropped_large: list[tuple[str, float]] = []
    for cat, count in category_counts.items():
        if count < _MIN_CATEGORY_COUNT:
            continue
        s = category_scale_stats(medias, cat)
        if s is None:
            continue
        if s["voted_area"] > MAX_VOTED_AREA:
            dropped_large.append((cat, s["voted_area"]))
            continue
        stats[cat] = s

    selected: list[str] = []
    report: dict = {
        "dropped_above_max_voted_area": sorted(dropped_large),
        "max_voted_area": MAX_VOTED_AREA,
        "bands": {},
    }
    for name, lo, hi in SCALE_BANDS:
        in_band = sorted(
            (c for c, s in stats.items() if lo <= s["voted_area"] < hi),
            key=lambda c: (stats[c]["union_inflation"], c),
        )
        picks = in_band[:n_per_band]
        selected.extend(picks)
        report["bands"][name] = {
            "range": [lo, hi],
            "target": n_per_band,
            "n_candidates": len(in_band),
            "under_populated": len(picks) < n_per_band,
            "selected": picks,
            "not_selected": in_band[n_per_band:],
            "scales": {c: stats[c] for c in picks},
        }
    return sorted(selected), report


#: Force a category-selection mode instead of inferring it from the medias.
#: ``"prevalence"`` is what the already-box-banded ``vg_box_*`` sets want: they
#: are a box-size axis by construction, so re-banding *within* one leaves most
#: bands empty (wave 2 of #3129 collapsed to 5/4/2 categories out of 40).
#: ``"scale"`` forces banding; unset infers as before.
CATEGORY_MODE = os.environ.get("CALIB_CATEGORY_MODE", "").strip().lower()

#: Restrict category selection to categories that have a typed query (#3267).
#:
#: The autopilot's opening is a walk down the **seed sort**, and where that sort
#: comes from is a property of the cell: with a query the app ranks by cosine to
#: the typed text, without one it falls back to three random known-goods.  A
#: study that sweeps the opening on a text sort cannot have half its cells on
#: the other start - the arms' cuts are positions in a ranking, and the two
#: rankings are not the same object.
#:
#: This filter runs BEFORE selection rather than after, so a dropped category is
#: replaced by the next eligible one instead of shrinking the grid.  0 (the
#: default) is the behaviour of every run before #3267.
REQUIRE_SEED_QUERY = os.environ.get("CALIB_REQUIRE_SEED_QUERY", "0") == "1"


#: The opening this study **declares**, asserted per cell (#3278).
#:
#: Autopilot has two real starts and which one a cell takes is decided silently:
#: a text sort when the (dataset, category) has a query *and* the embedder's text
#: half has a tower, three random known-goods otherwise.  Since #3269 the harness
#: takes the first wherever it can, which means a grid mixing SigLIP and DINOv3
#: arms now opens two different ways along one axis - the confound
#: ``lessons/2026-08-27-the-region-arm-could-not-open-the-way-the-app-does.md``
#: describes, and the reason :data:`PAIR_SEP` exists.
#:
#: Values, all of them a *declaration* rather than a switch - none of them
#: changes how a cell seeds, only what it is allowed to have seeded from:
#:
#: * ``"text"`` - every cell must open on a typed query.  A cell that falls back
#:   raises instead of running, for the reason the paired-arm guard in
#:   ``run_cells`` does: a mislabelled cell is invisible where a missing one is
#:   not.
#: * ``"known_good"`` - every cell must open on three random known-goods.  This
#:   is the pin for a study whose subject *is* that flow (or one re-running a
#:   finished grid that took it), and it fails if an arm silently gains a text
#:   tower or a query.
#: * ``"mixed"`` - the grid deliberately holds both openings, e.g. a re-runner
#:   mirroring a completed study's arms.  Nothing is asserted per cell; the
#:   declaration is what stops the mix reading as an oversight, and the analyzer
#:   guard (``_cells_io.assert_one_opening``) is what stops two openings being
#:   pooled into one number.
#: * ``""`` (unset) - no assertion, the behaviour of every run before #3278.
REQUIRE_OPENING = os.environ.get("CALIB_REQUIRE_OPENING", "").strip().lower()
_OPENINGS = ("", "text", "known_good", "mixed")
if REQUIRE_OPENING not in _OPENINGS:
    raise ValueError(f"CALIB_REQUIRE_OPENING={REQUIRE_OPENING!r} is not one of {_OPENINGS[1:]}")


def select_categories(
    medias: dict, category_counts: dict[str, int], dataset: str | None = None
) -> tuple[list[str], dict]:
    """Scale-stratified when boxed, else prevalence-spread.

    ``CALIB_CATEGORY_MODE`` overrides the inference in either direction.
    ``CALIB_REQUIRE_SEED_QUERY=1`` additionally drops categories with no typed
    query *before* selecting, so the surviving grid is entirely text-seeded.
    """
    dropped_no_query: list[str] = []
    if REQUIRE_SEED_QUERY and dataset is not None:
        eligible = {c: n for c, n in category_counts.items() if seed_query_text(dataset, c)}
        dropped_no_query = sorted(set(category_counts) - set(eligible))
        category_counts = eligible

    selected, report = _select_categories_inner(medias, category_counts)
    if REQUIRE_SEED_QUERY and dataset is not None:
        report["require_seed_query"] = True
        report["dropped_no_seed_query"] = dropped_no_query
    return selected, report


def _select_categories_inner(medias: dict, category_counts: dict[str, int]) -> tuple[list[str], dict]:
    if CATEGORY_MODE == "all":
        # A designated dataset carries its design in its category list: the
        # cells were built to be run, prevalence is identical across them by
        # construction, and re-banding an already-banded set would discard the
        # experiment. Take every category the pickle holds.
        return sorted(category_counts), {
            "mode": "all",
            "reason": "dataset's categories are the experimental design (designated cells)",
        }
    if CATEGORY_MODE == "prevalence":
        return select_categories_by_prevalence(category_counts), {
            "mode": "prevalence",
            "reason": "forced by CALIB_CATEGORY_MODE (the dataset has no scale axis to stratify on)",
        }
    selected, report = select_categories_by_scale(medias, category_counts)
    if selected:
        report["mode"] = "scale_bands"
        return selected, report
    return select_categories_by_prevalence(category_counts), {
        "mode": "prevalence",
        "reason": "dataset carries no ground-truth region boxes; no scale axis to stratify",
    }


#: Which index varies fastest in :func:`array_cells`.
#:
#: ``category`` (the default, and every run before #3267) walks a category's
#: whole seed block before moving on.  ``seed`` walks every environment at seed
#: 0, then every environment at seed 1, and so on.
#:
#: The difference only shows up when an array does **not** finish, and then it
#: decides what the run loses.  A SLURM array dispatches roughly in index order,
#: so under ``category`` a truncated run is missing its last *categories
#: entirely* - whole environments, gone, and the prevalence axis the analysis
#: bands on is short at one end.  Under ``seed`` it is missing its last *seeds*,
#: uniformly across every environment: the design is intact and only the
#: standard errors are wider, which is a thing a report can simply state.
#:
#: That makes it the right ordering for any run against a wall-clock deadline -
#: it converts "ran out of time" from a design failure into a power one.
CELL_ORDER = os.environ.get("CALIB_CELL_ORDER", "category").strip().lower()


def array_cells(categories_by_dataset: dict[str, dict[str, list[str]]]) -> list[dict]:
    """Enumerate ``(dataset, embedder, category, seed)`` cells for the SLURM array.

    Each cell runs **all styles** for its embedder inside one task (they share
    the loaded pickle), so an embedder's arms are paired on identical data,
    splits, and exemplar.  Deterministic order -> a task index maps to a stable
    cell across submissions.  :data:`CELL_ORDER` chooses which index varies
    fastest; see the note there for why that matters to a truncated run.
    """
    envs: list[tuple[str, str, str]] = []
    for ds in DATASETS:
        per_emb = categories_by_dataset.get(ds, {})
        for emb in embedders_for_dataset(ds):
            for cat in per_emb.get(emb, []):
                envs.append((ds, emb, cat))

    cells: list[dict] = []
    if CELL_ORDER == "seed":
        for seed in SEEDS:
            for ds, emb, cat in envs:
                cells.append({"dataset": ds, "embedder": emb, "category": cat, "seed": seed})
    else:
        for ds, emb, cat in envs:
            for seed in SEEDS:
                cells.append({"dataset": ds, "embedder": emb, "category": cat, "seed": seed})
    return cells
