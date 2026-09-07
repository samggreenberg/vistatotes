"""The shared pre-embedded pile: which ``(dataset, embedder)`` cells exist and where.

A *cell* is one ``<dataset>__<embedder>.pkl`` under the pile's
``embeddings/`` dir — the per-pair artifact every study loads instead of
re-embedding. Studies point ``VTSEARCH_DATA_DIR`` at the pile and read the
cells in place; nothing here is study-specific.

**Reproducibility.** The pile lives on scratch, which is treated as purgeable,
so every cell must be rebuildable from sources that are *not* on scratch:

* ``visual_genome_m`` / ``caltech101_m`` are VTSearch demo datasets, downloaded
  into the shared demo cache (``DEMO_CACHE``) and loaded by ``load_demo_dataset``.
* ``coco_val`` is not a demo dataset; it is assembled from the COCO-2017-val
  images and the flattened annotations staged under ``COCO_ROOT``.

Because ``_cells_io.dump_medias`` drops ``media_bytes``, a cell holds vectors
(plus ``patch_grid`` for patch embedders) and no pixels — so the pile is small
relative to its sources and a rebuild always re-reads the staged originals.

**Region voting.** Only patch embedders emit ``patch_grid``; a boxed dataset
paired with a single-vector embedder silently degrades to binary voting. That
mis-specification has burned three studies (#2877, #2897, #2905), so
:func:`region_capable` states it per *cell* rather than per dataset, and
``build_pile.py --verify`` asserts the geometry is actually present.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import NamedTuple

USER = os.environ.get("USER", "sgreenberg")

#: Root of the shared pile. Everything below is derived from it.
PILE = Path(os.environ.get("VTS_PILE", f"/expscratch/{USER}/vts-cache"))
DATADIR = PILE / "datadir"
EMBEDDINGS = DATADIR / "embeddings"
MODELS = PILE / "models"

#: Shared, non-scratch sources the pile is rebuilt from.
DEMO_CACHE = Path(os.environ.get("VTS_DEMO_CACHE", "/exp/scale26/datasets/external/vtsearch-demos"))
COCO_ROOT = Path(os.environ.get("VTS_COCO_ROOT", "/exp/scale26/datasets/external/COCO"))
#: The zip the builder reads pixels out of. This, not an extracted directory,
#: is what a `coco_val` rebuild actually depends on -- the staging area holds
#: `val2017.zip` and has never held `val2017/`. Named here because it was
#: previously spelled inline in the builder while :data:`COCO_IMAGES` named the
#: directory, and the rebuild canary checked the directory: it reported
#: `coco_val` REBUILD-BROKEN against a source that was present and fine (#3299).
COCO_VAL_ZIP = COCO_ROOT / "images" / "val2017.zip"
#: Where the images live *if* somebody extracts them. Optional, and not part of
#: the rebuild path: nothing depends on this directory existing. `box_sheets.py`
#: prefers it (a loose JPEG is cheaper than a zip member) but falls back to
#: :data:`COCO_VAL_ZIP`, which is where the pixels have always actually been --
#: it used to read only this path and drew empty sheets instead (#3305).
COCO_IMAGES = COCO_ROOT / "images" / "val2017"
COCO_ANNOTATIONS = COCO_ROOT / "derived" / "objects_flat_val2017.jsonl.gz"

#: Datasets in the pile. ``boxed`` means the medias carry ground-truth region
#: boxes, which is what a region-voting arm drags — necessary but not
#: sufficient (the embedder must also be patch-capable; see region_capable).
#: ``source_dir`` is the demo extraction dir the loader treats as "already
#: downloaded" (vtscore/datasets/downloader/*.py). It must be present in the
#: datadir before a demo cell is built — see :func:`require_demo_source`.
DATASETS: dict[str, dict] = {
    "visual_genome_m": {"boxed": True, "kind": "demo", "source_dir": "visual_genome"},
    "caltech101_m": {"boxed": False, "kind": "demo", "source_dir": "caltech-101"},
    "coco_val": {"boxed": True, "kind": "coco"},
    # Box-size-banded VG, drawn from the WHOLE source (all 108k images, full
    # free-text vocabulary) rather than the demo pipeline's 100 curated
    # categories on a 4% slice.  The `_s`/`_m`/`_l` on `visual_genome_*` is a
    # dataset *size* tier and says nothing about boxes; these are the box bands.
    "vg_box_small": {"boxed": True, "kind": "vg_band", "band": "small"},
    "vg_box_medium": {"boxed": True, "kind": "vg_band", "band": "medium"},
    "vg_box_large": {"boxed": True, "kind": "vg_band", "band": "large"},
    # The same-class-across-bands set (#3156). One pickle, one class list, one
    # negative pool; the band lives on the category name (`bus@small`). Not a
    # replacement for `vg_box_*` -- those measured what they measured and stay
    # reproducible -- but the two are not comparable: disjoint vocabularies
    # against a fixed one.
    #
    # Drawn from the half of VG that COCO sourced, and labelled from COCO's
    # exhaustive annotation rather than VG's free text, because VG's own labels
    # cannot support the construction: measured on this pool its recall over C
    # is 0.76, and 1.4% of the images it calls negative actually hold the object
    # (`coco_anchor.py`). At 80 positives per cell that would be ~54 hidden
    # positives sitting in the negatives.
    "vg_scale": {"boxed": True, "kind": "vg_scale", "labels": "coco"},
    # `vg_scale` with the box-size band collapsed away (#3115): the same images,
    # boxes and corrections, keyed on the bare class.  A calibration study wants
    # uniform prevalence across cells and does not care how big the box is;
    # `visual_genome_m` gives neither (25 to 1645 positives, and its thin
    # categories produce cells with no trainable step at all).
    #
    # DERIVED from the built `vg_scale` pickle, so it must be listed AFTER it -
    # and so it inherits whatever that cell currently holds.  #3252 changed how
    # `vg_scale` selects and corrects its cells, which means a `vg_scale_any`
    # built before that commit is NOT the same dataset as one built after it.
    # Rebuild it whenever `vg_scale` is rebuilt.  That used to be a rule nobody
    # could check: `--force` on `vg_scale` alone left this cell holding the old
    # labels with the right media count and the right vectors, so it looked
    # healthy (#3281 shipped a box repair to one study and not the other).  It
    # is now enforced twice -- `build_pile.py` pulls this dataset into any run
    # that rebuilds its parent, and `--verify` compares the parent-label digest
    # stamped on each derived media against the parent's live one.
    "vg_scale_any": {"boxed": True, "kind": "vg_scale_any"},
    # `vg_scale_any`'s construction with the BAND DROPPED FROM SELECTION rather
    # than from the key, and sized for a long labelling session (#3547).
    #
    # `vg_scale_any` collapses `class@band` after the fact, so it inherits
    # `vg_scale`'s per-band designation and is capped by the THINNEST band:
    # `bus@small` has 138 candidates, which is why 100/band was the ceiling and
    # 300 positives per class the result.  A study that never asks about box
    # size does not need that cap.  Designating band-free off the same
    # COCO-anchored labels takes the binding class from 414 candidates to 1006
    # (`stop sign`), which is what makes a 400-click horizon measurable at all:
    # #3319's deep wave harvested 82-85% of its ~150 sim positives.
    #
    # Prevalence is held at `vg_scale`'s designed 7.14% BY CONSTRUCTION rather
    # than inherited by accident -- `SCALE_DEEP_N_NEG` is derived from
    # `SCALE_DEEP_N_POS`, not set beside it.  That is the whole point: the
    # optimum this dataset exists to locate is `k* = -log2((1-pi)/pi)`, so
    # adding positives against a FIXED negative pool would move the answer
    # (300->900 against 3900 shifts pi to 18.8% and k* by a full bit) while
    # appearing to be nothing but "a deeper haystack".
    #
    # `on_request`: this one is 3x `vg_scale`'s media count, so it stays OUT of
    # the default sweep. A bare `build_pile.py` would otherwise quietly add five
    # cells nobody asked for, one of them a ~7 GB `dinov3_patch` grid. Name it
    # to build it: `--datasets vg_scale_deep --embedders siglip`.
    "vg_scale_deep": {"boxed": True, "kind": "vg_scale_deep", "on_request": True},
}

#: Box-size bands, as a fraction of image area, anchored to the patch
#: embedder's geometry (the same anchors the calibration harness bands on):
#: one DINOv3 patch is 1/196 of the image and the smallest HAC leaf is 1/12.
#: ``small`` is therefore "below what the patch grid can resolve at all".
#: Whether an image that is a positive for one class may serve as a NEGATIVE
#: for a class it does not hold (#3667).
#:
#: The construction originally said: positive for its own cells, negative only
#: if it holds nothing in *C*, excluded otherwise. The "excluded otherwise" is
#: right for the SAME class at another size -- scoring a large-bus image as a
#: small-bus negative penalises a detector for finding a real bus -- but it was
#: applied to every OTHER class too, where the reason does not hold. The cost
#: was 41.9% of the pile dropped from every class's evaluation, and negatives
#: that contain none of twelve common objects while positives contain one, so a
#: detector could score by learning "is this a scene with stuff in it".
#:
#: Gated on ``labels_exhaustive``: COCO annotates all eighty of its classes on
#: any image it annotates, so absence is a fact there. On the other half absence
#: is VG's silence, measured wrong 0.5-2.5% of the time per class (#3588), and
#: importing that into the negatives is a different decision from this one.
SCALE_CROSS_CLASS_NEGATIVES = True

#: The upper cut mirrors ``MAX_VOTED_AREA``: a box covering >80% of the image
#: is not a region, it is the image.
PATCH_AREA = 1 / 196
LEAF_AREA = 1 / 12
MAX_VOTED_AREA = 0.80
BOX_BANDS: dict[str, tuple[float, float]] = {
    "small": (0.0, PATCH_AREA),
    "medium": (PATCH_AREA, LEAF_AREA),
    "large": (LEAF_AREA, MAX_VOTED_AREA),
}

#: How many categories each banded dataset draws, and the image cap.  Categories
#: are stratified *within* the band so a band is not silently all one size.
BAND_N_CATEGORIES = int(os.environ.get("VTS_BAND_N_CATEGORIES", "40"))
BAND_MAX_IMAGES = int(os.environ.get("VTS_BAND_MAX_IMAGES", "12000"))
#: Categories whose union box is much larger than a single instance are
#: scattered instances, not a region a user would drag.
BAND_MAX_INFLATION = float(os.environ.get("VTS_BAND_MAX_INFLATION", "1.5"))
BAND_MIN_IMAGES = int(os.environ.get("VTS_BAND_MIN_IMAGES", "50"))

#: VG is annotated with free text, so its vocabulary is not a list of objects.
#: A detector asked to find "red" or "front" is measuring nothing, so these are
#: excluded from the banded datasets. The policy is **concrete countable
#: objects only**, which drops three kinds of name:
#:
#: * colours and other attributes -- properties, not things;
#: * frame relations and abstractions -- "front", "group", "object": either a
#:   position in the image or a placeholder the annotator reached for;
#: * mass nouns and unbounded surfaces -- "grass", "sky", "floor": real, but
#:   *stuff* rather than an object with an extent a user would drag a box around.
#:
#: The third group is the aggressive part of the policy and it costs coverage
#: in the large band specifically, because scene-scale stuff is exactly what
#: large boxes are made of. Countable landforms and structures ("tree",
#: "mountain", "building") are deliberately kept.
NON_OBJECT_CATEGORIES: frozenset[str] = frozenset(
    # attributes
    """red blue green yellow orange purple pink brown black white gray grey tan beige
    silver gold golden dark light bright colorful clear blurry shiny""".split()
    # frame relations, abstractions, placeholders
    + """front back side top bottom left right middle center centre corner edge end
    part section area region spot place row line lines stripe stripes pattern design
    shape size distance background foreground surface object objects thing things item
    items stuff group bunch pile set collection image picture photo photograph view
    scene display something other""".split()
    # mass nouns, unbounded surfaces and scene regions
    + """water snow sand dirt mud grass gravel concrete pavement asphalt sky smoke steam
    fog haze shade shadow shadows reflection glare sunlight ice foam liquid air weather
    ground floor flooring wall walls ceiling road roadway street sidewalk pathway path
    field beach ocean sea lake river land terrain lawn grassy lot traffic""".split()
)


def is_object_category(name: str) -> bool:
    """False for VG names that are not concrete countable objects.

    Matches on the **head noun** (the last token), not the whole string, because
    VG's vocabulary is full of modified compounds. Whole-string matching lets
    ``blue sky`` and ``table top`` through while head-noun matching drops them;
    matching *any* token would wrongly drop ``blue jeans``, ``tennis ball`` and
    ``left eye``, whose heads are perfectly good objects.
    """
    tokens = name.replace("-", " ").split()
    if not tokens:
        return False
    return tokens[-1] not in NON_OBJECT_CATEGORIES and name not in NON_OBJECT_CATEGORIES


#: Extra exclusions for a **same-class-across-scale-bands** study, keyed by head
#: noun and carrying the reason. Deliberately separate from
#: :func:`is_object_category`, which defines the published ``vg_box_*`` sets and
#: must keep defining them; this is a stricter policy layered on top for the new
#: construction, so the old numbers stay reproducible.
#:
#: The extra bar exists because a scale study asks two things of a class that
#: mere objecthood does not:
#:
#: * **Its size must be its own.** A part's box is set by its host, so a "small
#:   nose" is just a distant face -- banding it measures the host's distance,
#:   not the object's scale, and the arm silently becomes a different experiment.
#: * **Its absence must be checkable.** The negative pool is ~95% of the images
#:   and rests on "no instance here". For a part that is unverifiable at any
#:   scale: every image with a person has a nose whether or not VG annotated
#:   one, so the negatives are poisoned by construction and no amount of review
#:   fixes it. That is the worst case for the correction pass, not a candidate
#:   for it.
#:
#: Curated, not inferred -- and reported rather than applied silently, because
#: silent automated judgements about VG's vocabulary are what #3156 is about.
SCALE_STUDY_EXCLUSIONS: dict[str, str] = {
    # Individuated only by a host object. Size tracks the host; absence is
    # unverifiable wherever the host appears.
    **dict.fromkeys(
        """nose ear ears eye eyes face head hair mouth lip lips chin cheek forehead eyebrow
        eyebrows neck chest shoulder shoulders arm arms hand hands finger fingers thumb leg legs
        foot feet knee elbow wrist ankle waist hip tail paw paws hoof hooves horn horns tusk tusks
        beak snout mane fur skin tooth teeth tongue mustache moustache beard sideburns""".split(),
        "part",
    ),
    # Parts of artefacts: same two failures, non-anatomical.
    **dict.fromkeys(
        """collar sleeve sleeves cuff pocket zipper hem waistband strap straps buckle handle knob
        spout lid rim brim blade tread stem tip base""".split(),
        "part",
    ),
    # A location rather than a thing: the box has no principled extent (where
    # does an intersection begin?), so its area is an annotator choice and the
    # band it lands in is noise.
    **dict.fromkeys(
        """court courtyard intersection station runway walkway crossing crosswalk driveway alley
        parking lot yard park playground platform entrance exit doorway hallway corridor stairway
        staircase kitchen bathroom bedroom room office restaurant store shop market""".split(),
        "place",
    ),
    # Parts of a plant or structure: same failure as anatomy.
    **dict.fromkeys("""trunk branch twig root roof chimney railing banister step steps""".split(), "part"),
}

#: One string, several objects: "find the trunk in the middleground" is not one
#: question, so the class cannot be scored as one. Matched on the **whole name**
#: rather than the head noun, because a modifier is precisely what resolves the
#: ambiguity -- bare ``bat`` is unusable, ``baseball bat`` is a perfectly good
#: class. (Head-noun matching would reject both, and misreport the reason for
#: ``tree trunk``, which is unfit for being a *part*, not for being ambiguous.)
POLYSEMOUS_NAMES: frozenset[str] = frozenset(
    """trunk bat mouse pitcher crane tie nail bow plate glass iron seal pen""".split()
)

#: A class annotated on more than this share of all images is treated as
#: pervasive: its negative pool is both thin and least trustworthy, since a
#: ubiquitous thing is exactly what an annotator stops bothering to mark. `sky`
#: is the worked example -- 18.8% prevalent as annotated, plainly higher in
#: truth (`docs/experiments/2026-08-12-overview-bench/REPORT.md`). Measured, not listed,
#: because which names are pervasive is a property of the corpus.
PERVASIVE_PREVALENCE = float(os.environ.get("VTS_PERVASIVE_PREVALENCE", "0.10"))


def scale_study_exclusion(name: str) -> str | None:
    """Why *name* is unfit for a scale-band study, or ``None`` if it is fit.

    Head-noun matched, like :func:`is_object_category`, so ``left eye`` and
    ``bus station`` are caught while ``eyeglasses`` and ``gas station wall``
    are judged on their own heads.
    """
    if not is_object_category(name):
        return "non_object"
    if name in POLYSEMOUS_NAMES:
        return "polysemous"
    tokens = name.replace("-", " ").split()
    return SCALE_STUDY_EXCLUSIONS.get(tokens[-1]) or SCALE_STUDY_EXCLUSIONS.get(name)


#: The class list *C* for the same-class-across-bands study (issue #3156).
#:
#: Chosen by the owner on 2026-08-17 from the measured shortlist
#: (``shortlist_scale_classes.py --compact --floor 100``), out of the 24
#: candidates that were simultaneously: supported at >= 100 images in all three
#: bands, free of a measured alias partner and of plural-form ambiguity, and
#: **also a COCO-2017 class**. That last property is what makes the correction
#: pass affordable: COCO val2017 is exhaustively annotated over these names, so
#: VG's miss rate -- and our own annotators' accuracy -- can be scored against it
#: with no extra human review.
#:
#: Deliberately *not* derived at build time from the scan. Which classes a human
#: can annotate consistently is a judgement, and re-deriving it would silently
#: change what the study measures whenever the scan is re-run.
SCALE_CLASSES: tuple[str, ...] = (
    "clock",
    "bird",
    "boat",
    "umbrella",
    "kite",
    "book",
    "dog",
    "backpack",
    "knife",
    "bicycle",
    "bus",
    "stop sign",
)

#: VG spellings that ARE a class in *C*, beyond the class name itself.
#:
#: VG's vocabulary is free text and :func:`pilebuild.vgsource.vg_boxes_by_name`
#: matches an object's PRIMARY name only, so a class built from one spelling
#: silently drops every other. That is not merely a supply loss: on the ~52% of
#: VG that COCO does not annotate, VG's silence is the only evidence of absence,
#: so an instance annotated under an unlisted spelling becomes a **negative** for
#: its own class (#3605). There is no cheaper fix available in the reader --
#: every one of the 2,516,939 objects in this release of VG carries a ``names``
#: list of length **one**, and of the 18,897 objects named by an entry in either
#: table below, **zero** carry the class name further down that list (#3618).
#:
#: Every entry is measured, and folding one makes two claims, so two things are
#: measured (``name_evidence.py``):
#:
#: 1. **the class is present** when this name is its only evidence -- the
#:    *repair precision*: over the VG-COCO overlap, the share of images carrying
#:    the name and NOT the class name where COCO says the class is there anyway.
#:    Read it as a price, in the right units: ``1 / precision - 1`` is how many
#:    **good hard negatives are destroyed per contaminated negative retired**.
#:    Not pool membership -- 77,119 images are eligible against a 4,200-image
#:    draw -- but the images a name withholds are the ones hardest to tell from
#:    the class, which is what makes the ratio the thing to cut on (#3635). The
#:    cut is 1/3 -- two destroyed per repair -- taken on the **Wilson lower
#:    bound**, so a name measured on five images cannot outrank one measured on
#:    two thousand.
#: 2. **this box is the object**: at least half of the name's boxes land on a
#:    COCO box of the class, over at least 20 boxes. A band is a claim about one
#:    object's size (#3616), so a name that passes (1) and fails (2) goes to
#:    :data:`SCALE_VG_AMBIGUOUS` instead -- the safe side, since a wrong
#:    ambiguous costs a few pool images and a wrong alias injects a mis-banded
#:    positive.
#:
#: **Box overlap between two names is not the instrument here** and cannot be.
#: ``scan_name_overlap.py`` needs the two names on one image, and an annotator
#: who writes `back pack` does not also write `backpack`: 846 of 1,740
#: class-vs-candidate pairs never co-occur at all, and where a singular and its
#: plural do co-occur they are deliberately different boxes -- `backpack` /
#: `backpacks` scores **0.000 both ways** over 10 co-images and is called
#: *distinct*. That test keeps its own job, which is the case where two names
#: really do sit on one box (`clock`/`clock face`, 0.562/0.701 over 286) and
#: refuting a lookalike, which is how `bus` survived matching 80 images
#: annotated `bush`.
SCALE_VG_NAMES: dict[str, tuple[str, ...]] = {
    "backpack": ("back pack",),
    "bicycle": ("bicycles",),
    # COCO annotates ducks, geese and gulls as `bird`, and so does VG under its
    # own species names: each of these is above the cut on both tests.
    "bird": ("duck", "goose", "ostrich", "owl", "parrot", "pigeon", "seagull", "swan"),
    "boat": ("boats", "canoe", "kayak", "raft", "sailboats", "ship"),
    # COCO has no magazine class and annotates magazines as `book`, which is the
    # reading this dataset already took -- see SCALE_CLASS_RULES["book"].
    "book": ("magazine",),
    # The subtype family pools to 83% over 18 sole images and 82% over 74 boxes,
    # which carries the four route/deck spellings no one of which reached the
    # five-image floor on its own (#3636).
    "bus": ("buses", "city bus", "double-decker bus", "passenger bus", "school bus", "tour bus"),
    # The face is the clock: 89% of `clock face` boxes land on COCO's clock box,
    # over 184 of them -- the best-supported fold in the study. `clockface` is
    # the same word without the space, and pooled with it scores 89% over 19
    # sole images and 89% over 194 boxes (#3636).
    "clock": ("clock face", "clockface", "clocks"),
    # `dalmation` (VG's spelling) and `lab` come from the breed group, `white dog`
    # from the colour one: 74% and 91% pooled, both folding on box agreement (#3636).
    "dog": ("black dog", "brown dog", "dalmation", "dogs", "lab", "puppy", "white dog"),
    # COCO's `kite` covers parasails and parachutes, and VG names them so.
    # `para sail` is `parasail` with a space; the pair scores 83% over 24 sole
    # images and 84% over 74 boxes (#3636).
    "kite": ("kites", "para sail", "parachute", "parasail"),
    "knife": ("butter knife",),
    # Two groups, both folding (#3636). The colour family -- eight spellings, one
    # hypothesis -- pools to 97% over 35 sole images and 80% over 122 boxes, so
    # the four nobody could measure alone join the two that could. The subtype
    # family (`beach`, `patio`, `closed`, `open`) pools to 88% and 69%.
    "umbrella": (
        "beach umbrella",
        "blue umbrella",
        "closed umbrella",
        "green umbrella",
        "open umbrella",
        "orange umbrella",
        "parasol",
        "patio umbrella",
        "red umbrella",
        "white umbrella",
        "yellow umbrella",
    ),
}

#: What :func:`pilebuild.loaders.vg_scale.canonicalise` does when an alias box
#: lands on an image where the class already has one of its own -- see that
#: function's ``FOLD_MODES``.
#:
#: ``fold`` is the reading #3637 measured and kept, and the margin is not close:
#: over the VG-COCO overlap, on the 225 images where the three modes disagree,
#: COCO's exhaustive boxes say the class is **not** a single-band positive on 199
#: of them, and folding names the right answer 88% of the time against 6.7% for
#: keeping the class's own band. The structural reason is bigger than that
#: number: the COCO half is already banded off COCO's own exhaustive box set, so
#: ``anchor_to_coco`` un-bands 17% of the cleanly-banded images there on evidence
#: of exactly the same kind. A mode that protected the un-anchored half from it
#: would make the two halves of one dataset disagree about what a positive is.
#:
#: The alternatives are kept so the arm can be re-measured (``band_fold.py``),
#: not because either is a candidate default.
SCALE_FOLD_MODE = os.environ.get("VTS_SCALE_FOLD_MODE", "fold")

#: VG spellings that are evidence the class MAY be present, and cannot be its box.
#:
#: `bike` is the case that named this table. Over the 51,411-image VG-COCO
#: overlap it carries **638 of COCO's 3,683 `bicycle` boxes** against the
#: `bicycle` spelling's 775 -- so `bicycle` built from one spelling is missing
#: roughly half its positives on the non-COCO half. It cannot simply be merged:
#: on the images where it is the only evidence, COCO finds a bicycle **47%** of
#: the time, and `bike` is a measured alias of `motorcycle` too (box IoU 0.38
#: over 388 co-images).
#:
#: A box under one of these names is treated as **evidence of neither presence
#: nor absence**: it is not a positive (we cannot say the object is there) and it
#: bars the image from the shared negative pool (we cannot say it is not). That
#: is the ``excluded`` third state the construction already has -- see
#: :func:`pilebuild.loaders.vg_scale.lift_ambiguous`. It removes the contaminated
#: negatives; recovering the missing positives needs a human pass.
#:
#: **Three kinds of name land here, and they share one treatment because they
#: share one answer** -- *this image cannot serve as a negative, and this box
#: cannot serve as a positive* (#3618):
#:
#: * a spelling that may denote something else: `bike` (47% precision),
#:   `tricycle`, `silverware`.
#: * a **collective**: `books` (89% precision, 34% box agreement), `birds`,
#:   `umbrellas`, `knives`. The box is a pile; a band is a claim about one
#:   object's size.
#: * a **part or container**, whose box is not the object at all: `beak` (86%),
#:   `bookshelf` (81%), `knife block` (79%), `stop` (70% -- the lettering on the
#:   sign). Named apart in the report because they cost differently: a spelling
#:   withholds the images that spell one class oddly, while a scene word
#:   withholds a whole scene type from **every** class's pool.
#:
#: Suppression applies only where it is the sole evidence. On an image COCO
#: annotates, or one a reviewer has ruled on, the answer is already known and
#: the ambiguous spelling is ignored.
SCALE_VG_AMBIGUOUS: dict[str, tuple[str, ...]] = {
    "backpack": ("black backpack", "black bag", "bookbag", "duffle bag"),
    "bicycle": ("bicyclist", "bike", "bike tire", "bikes", "tricycle"),
    # `black bird` / `white bird` pool to 100% over 6 sole images but only 15
    # boxes -- above the precision cut, below the box floor, which is the safe
    # side. `pigeons` inherits from `pigeon`, not from the species family:
    # a plural is a collective whatever its singular does (#3636).
    "bird": (
        "beak",
        "birds",
        "black bird",
        "dove",
        "ducks",
        "feather",
        "feathers",
        "geese",
        "peacock",
        "pigeons",
        "seagulls",
        "white bird",
    ),
    "boat": ("barge", "bouy", "sail boat", "sailboat"),
    "book": (
        "binder",
        "black book",
        "book case",
        "book shelf",
        "bookcase",
        "books",
        "bookshelf",
        "dvd",
        "dvds",
        "games",
        "library",
        "magazines",
        "notebook",
        "white book",
    ),
    # `busses` is VG's other plural of `bus`. `buses` folds on its own measured
    # box agreement; `busses` has none of its own, and a plural with no
    # measurement behind it is a collective until shown otherwise (#3636).
    "bus": ("blue bus", "busses"),
    # `numerals` and `clock faces` each inherit from their own singular (#3636).
    "clock": ("alarm clock", "clock faces", "numeral", "numerals", "roman numerals"),
    "dog": ("bulldog", "poodle"),
    "knife": ("butterknife", "knife block", "knives", "silverware"),
    # `sign` is the largest fold-in column anywhere in C -- 473 of COCO's 1,016
    # `stop sign` boxes, 46.6% -- and it is NOT here, because a VG `sign` box is
    # a stop sign 7.9% of the time: listing it would withhold 12.7 images from
    # the pool per contaminated negative removed. This class's missing positives
    # need a human pass, not a name (#3618).
    "stop sign": ("octagon", "stop"),
    "umbrella": ("an umbrella", "black umbrella", "pink umbrella", "umbrellas"),
}


#: How a name is written, once, so ``vg_name_families.py`` and the grouping
#: below cannot drift apart on what a name's head noun is.
#:
#: Trailing characters VG annotators leave on a name (`umbrella.`, `"clock"`).
NAME_PUNCT = ".,;:!?'\"()[]"


def name_head(name: str) -> str:
    """The final token of *name*, stripped of punctuation and a possessive.

    `umbrella's` and `umbrella.` are the same word as `umbrella` with an
    annotator's typing on the end, and there is no sense in which they denote
    something else.
    """
    tokens = name.replace("-", " ").split()
    if not tokens:
        return ""
    tok = tokens[-1].strip(NAME_PUNCT)
    return tok[:-2] if tok.endswith("'s") else tok


def name_singulars(token: str) -> set[str]:
    """Candidate singular forms of *token*, over-generating on purpose.

    Over-generation is safe here because the result is only ever used to *test*
    membership against one known class name -- `buses` proposing both `bus` and
    `buse` costs nothing, and missing `bus` would cost a whole spelling.
    """
    out = {token}
    if token.endswith("ies"):
        out.add(token[:-3] + "y")
    if token.endswith("ves"):
        out.update({token[:-3] + "fe", token[:-3] + "f"})
    if token.endswith("ses"):  # busses -> bus
        out.add(token[:-3])
    if token.endswith("es"):
        out.add(token[:-2])
    if token.endswith("s"):
        out.add(token[:-1])
    return out


def name_skeleton(name: str) -> str:
    """*name* with whitespace, hyphens and annotator punctuation removed.

    Two names with one skeleton are one word typed two ways -- `back pack` /
    `backpack`, `clock face` / `clockface`, `row boat` / `rowboat`. This is what
    the ``spelling`` construction groups on, and it is an equivalence relation
    rather than a lexicon, so it needs no vocabulary to maintain.
    """
    return "".join(ch for ch in name.lower() if ch.isalnum())


class Construction(NamedTuple):
    """A productive way of writing a name that does not change what it denotes.

    ``modifiers`` is the vocabulary that may stand in front of the class's head
    noun. It is a **declared list, not a regex**: which words leave a denotation
    alone is exactly the judgement #3636 is about, so it is written down here
    beside the tables it fills rather than inferred at run time.

    ``foldable`` is whether a member may reach :data:`SCALE_VG_NAMES` at all. A
    ``count`` compound never can, however well it scores: `two birds` names a
    *set* of that many, and a band is a claim about one object's size -- the
    same reason `books` and `umbrellas` sit in the ambiguous table (#3618).
    """

    key: str
    modifiers: frozenset[str]
    foldable: bool
    why: str


#: The constructions ``name_evidence.py --pooled`` adjudicates as one hypothesis.
#:
#: #3618 scored every candidate name alone, against a floor of five images where
#: the name is the class's only evidence, and **76 of 626 fell below it** --
#: recorded `unmeasured`, neither acted on nor refuted. They are not noise: they
#: carry 312 non-COCO images between them.
#:
#: Most are not independent hypotheses. `blue umbrella`, `red umbrella`,
#: `green umbrella`, `orange umbrella` and `yellow umbrella` are one hypothesis
#: five times over -- *a colour word in front of the class name does not change
#: what the name denotes* -- and that hypothesis is testable at the sample size
#: of the whole family rather than one colour at a time.
#:
#: A name joins a construction for class *c* when its head noun is *c*'s head
#: noun (a plural allowed) and every remaining token is in the vocabulary. So
#: `black clock` is a `clock` colour compound and `black face` is not, which is
#: the distinction that keeps the group measuring one thing.
SCALE_VG_CONSTRUCTIONS: tuple[Construction, ...] = (
    Construction(
        key="colour",
        modifiers=frozenset(
            {
                "beige",
                "black",
                "blue",
                "brown",
                "colorful",
                "colourful",
                "dark",
                "gold",
                "golden",
                "gray",
                "green",
                "grey",
                "multicolored",
                "orange",
                "pink",
                "purple",
                "red",
                "silver",
                "tan",
                "white",
                "yellow",
            }
        ),
        foldable=True,
        why="a colour word does not change what the name denotes",
    ),
    Construction(
        key="size",
        modifiers=frozenset({"big", "giant", "huge", "large", "little", "long", "small", "tall", "tiny"}),
        foldable=True,
        why="a size word does not change what the name denotes; the band is read off the box, not the word",
    ),
    Construction(
        key="typing",
        modifiers=frozenset({"a", "an", "the"}),
        foldable=True,
        why="a determiner is the annotator typing, not a distinction",
    ),
    Construction(
        key="count",
        modifiers=frozenset(
            {"two", "three", "four", "five", "six", "several", "many", "some", "multiple", "group", "bunch", "pair"}
        ),
        foldable=False,
        why="a numeral names a SET of that many: the box is a pile, so a member can be evidence but never a band",
    ),
    #: Not a lexicon -- membership is the equivalence class under
    #: :func:`name_skeleton`, so `back pack`/`backpack` and `clock face`/`clockface`
    #: group with no vocabulary to keep current.
    Construction(
        key="spelling",
        modifiers=frozenset(),
        foldable=True,
        why="one word typed two ways: same letters, different whitespace",
    ),
    #: Also not a lexicon: one group per singular form, so `pigeon`/`pigeons`
    #: is a hypothesis and `pigeons`/`ducks` is not. That is the pairwise
    #: question worth asking -- *does this plural denote what its own singular
    #: denotes* -- and for `pigeons` it is far better evidence than the species
    #: group, which knows only that a pigeon is a bird.
    #:
    #: **Never foldable**, and that is the shipped rule rather than caution:
    #: `books`, `birds`, `umbrellas`, `knives`, `ducks`, `geese` and `seagulls`
    #: are all in :data:`SCALE_VG_AMBIGUOUS` because the box is a pile. The
    #: plurals #3618 *did* fold (`boats`, `clocks`, `dogs`, `kites`) each earned
    #: it on their own measured box agreement, which an individual measurement
    #: still delivers -- inheritance is for names that have none, and a plural
    #: with no measurement of its own is a collective until shown otherwise.
    #: `buses` keeps the alias it earned; `busses` inherits withheld.
    Construction(
        key="plural",
        modifiers=frozenset(),
        foldable=False,
        why="a plural names a SET: the box is a pile, so a member can be evidence but never a band",
    ),
)


class NameGroup(NamedTuple):
    """A set of names a human asserts denote the same kind of thing.

    Where a :class:`Construction` is productive -- a vocabulary that applies to
    every class -- a group is a judgement about *this* class's vocabulary, and
    the judgement is the ``criterion``. That string is load-bearing, not a
    comment: it is what makes membership auditable, and it is the only defence
    against fitting the group to the answer. **Every candidate name meeting the
    criterion is listed, including the ones known to score badly** -- `crane`
    (the machine) is in `bird`/`species` and `jet ski` is in `boat`/`vessel`,
    because a group whose losers were quietly left out is not a measurement.

    ``foldable=False`` marks a group whose members may be evidence the class is
    present but can never carry a band, exactly as for a construction.
    """

    key: str
    criterion: str
    names: tuple[str, ...]
    foldable: bool = True


#: Hand-declared groups, per class, each with the criterion that defines it.
#:
#: These reach what a construction cannot. #3618's residue is mostly *hyponyms*
#: -- `yacht`, `ferry`, `flamingo`, `grandfather clock` -- and no modifier
#: vocabulary groups those, because the whole name is different. What groups
#: them is a person saying "these all name a kind of watercraft", which is a
#: hypothesis with a sample size like any other.
#:
#: This is **not** the head-noun fold #3618 refuted. That was mechanical: every
#: name sharing a head noun, which puts `hot dog` (405 images, 0 of 181) in with
#: `puppy`. Three things separate a group from it: the criterion is stated and
#: excludes `hot dog` on meaning rather than on its score; the group is
#: adjudicated before anything is inherited; and a member measurable on its own
#: keeps its own verdict either way (``name_evidence.py``).
SCALE_VG_GROUPS: dict[str, tuple[NameGroup, ...]] = {
    "bicycle": (
        NameGroup(
            key="part",
            criterion="a VG name for a part of a bicycle",
            names=(
                "bars",
                "bicycle tire",
                "bike tire",
                "frame",
                "front wheel",
                "rack",
                "tire",
                "tires",
                "wheel",
                "wheels",
            ),
        ),
    ),
    "bird": (
        NameGroup(
            key="species",
            criterion="a VG name denoting a species or kind of bird",
            names=(
                "chicken",
                "chickens",
                "crane",
                "dove",
                "duck",
                "ducks",
                "eagle",
                "flamingo",
                "geese",
                "goose",
                "hen",
                "ostrich",
                "owl",
                "parrot",
                "peacock",
                "pelican",
                "penguin",
                "pigeon",
                "pigeons",
                "rooster",
                "seagull",
                "seagulls",
                "swan",
                "turkey",
            ),
        ),
    ),
    "boat": (
        NameGroup(
            key="vessel",
            criterion="a VG name denoting a kind of watercraft",
            names=(
                "barge",
                "boats",
                "canoe",
                "cruise ship",
                "ferry",
                "jet ski",
                "kayak",
                "motorboat",
                "raft",
                "row boat",
                "rowboat",
                "sail boat",
                "sailboat",
                "sailboats",
                "ship",
                "vessel",
                "yacht",
            ),
        ),
        NameGroup(
            key="mooring",
            criterion="a VG name for a place where vessels are moored (not open water or a shoreline)",
            names=("dock", "harbor", "marina"),
        ),
        NameGroup(
            key="part",
            criterion="a VG name for a part of a vessel",
            names=("bow", "cabin", "hull", "mast", "oar", "sail", "sails"),
        ),
    ),
    "book": (
        NameGroup(
            key="part",
            criterion="a VG name for a part of a book",
            names=("binding", "book cover", "cover", "page", "pages", "spine", "title"),
        ),
    ),
    "bus": (
        NameGroup(
            key="subtype",
            criterion="a VG name for a kind of bus, by its route or its deck",
            names=("city bus", "double decker", "double-decker bus", "passenger bus", "school bus", "tour bus"),
        ),
        NameGroup(
            key="part",
            criterion="a VG name for a part of a bus specifically (not a part any vehicle has)",
            names=("bus front", "top level", "upper level"),
        ),
    ),
    "clock": (
        NameGroup(
            key="subtype",
            criterion="a VG name for a kind of clock, by its mechanism or its mounting",
            names=("alarm clock", "digital clock", "grandfather clock"),
        ),
        NameGroup(
            key="dial",
            criterion="a VG name for a clock's dial taken as a whole (not a marking on it)",
            names=("clock face", "clock faces", "clockface", "dial", "dials"),
        ),
        NameGroup(
            key="marking",
            criterion="a VG name for a marking on a clock face",
            names=("black numbers", "numeral", "numerals", "roman numerals"),
        ),
        NameGroup(
            key="part",
            criterion="a VG name for a part of a clock other than its dial or its markings",
            names=("clock frame", "hands"),
        ),
    ),
    "dog": (
        NameGroup(
            key="breed",
            criterion="a VG name for a dog breed or life stage",
            names=("bulldog", "dalmation", "lab", "poodle", "puppy"),
        ),
        NameGroup(
            key="part",
            criterion="a VG name for a part of a dog",
            names=("dog's head", "fur"),
        ),
    ),
    "kite": (
        NameGroup(
            key="part",
            criterion="a VG name for a part of a kite",
            names=("kite tail", "long tail", "string", "strings", "tail", "tails"),
        ),
    ),
    "knife": (
        NameGroup(
            key="subtype",
            criterion="a VG name for a kind of knife, by what it cuts",
            names=("butter knife", "butterknife", "cake server", "cutter"),
        ),
        NameGroup(
            key="part",
            criterion="a VG name for a part of a knife",
            names=("blade", "handle", "handles", "knife blade", "tip"),
        ),
    ),
    "stop sign": (
        NameGroup(
            key="sign-type",
            criterion="a VG name for a kind of road or street sign, by what it says",
            names=(
                "arrow sign",
                "construction sign",
                "direction sign",
                "dollar sign",
                "electric sign",
                "handicapped sign",
                "no parking sign",
                "number sign",
                "one way sign",
                "street sign",
            ),
        ),
    ),
    "umbrella": (
        NameGroup(
            key="subtype",
            criterion="a VG name for a kind of umbrella, by its use or its state",
            names=("beach umbrella", "closed umbrella", "open umbrella", "parasol", "patio umbrella"),
        ),
    ),
}


def scale_vg_groups_for(cls: str, candidates: list[str]) -> dict[str, list[str]]:
    """Which pooled groups *cls*'s *candidates* fall into, keyed by group.

    Constructions are matched here rather than in the caller so that the
    vocabularies above are the only place the rule is written. A name may belong
    to at most one construction (the first that accepts it) and to any declared
    group, so `parasol` is both `umbrella`/`subtype` and -- were it spelled two
    ways -- a ``spelling`` member.
    """
    head_wanted = name_head(cls)
    skeletons: dict[str, list[str]] = {}
    for n in candidates:
        skeletons.setdefault(name_skeleton(n), []).append(n)

    out: dict[str, list[str]] = {}
    for con in SCALE_VG_CONSTRUCTIONS:
        if con.key == "spelling":
            continue
        members = []
        for n in candidates:
            tokens = n.replace("-", " ").split()
            if len(tokens) < 2 or head_wanted not in name_singulars(name_head(n)):
                continue
            if all(t.strip(NAME_PUNCT) in con.modifiers for t in tokens[:-1]):
                members.append(n)
        if members:
            out[con.key] = sorted(members)

    # `spelling` is ONE GROUP PER SKELETON, not one per class. The hypothesis is
    # "`clockface` denotes what `clock face` denotes", and it is pairwise: pooling
    # every respelt name in a class would ask instead whether bookcase-ish images
    # are book images, which is a different question with a different answer. A
    # skeleton shared with the class name alone is dropped -- there is no second
    # rate to pool with, since the class name is never its own sole evidence.
    for skel, ns in sorted(skeletons.items()):
        if len(ns) > 1:
            out[f"spelling:{skel}"] = sorted(ns)

    # `plural`: group by singular form, the same shape as `spelling`. A name's
    # key is the shortest of its own singular forms that is itself a candidate
    # (or the class name), so `ducks` keys on `duck`, `buses` and `busses` both
    # key on `bus`, and `clock faces` on `clock face`.
    known = set(candidates) | {cls}
    by_singular: dict[str, list[str]] = {}
    for n in candidates:
        forms = sorted(name_singulars(n) & known, key=len)
        by_singular.setdefault(forms[0] if forms else n, []).append(n)
    for key, ns in sorted(by_singular.items()):
        if len(ns) > 1:
            out[f"plural:{key}"] = sorted(ns)

    for grp in SCALE_VG_GROUPS.get(cls, ()):
        members = sorted(set(grp.names) & set(candidates))
        if members:
            out[grp.key] = members
    return out


def scale_vg_group_foldable(cls: str, key: str) -> bool:
    """Whether a member of group *key* may reach :data:`SCALE_VG_NAMES`."""
    base = key.split(":", 1)[0]
    for con in SCALE_VG_CONSTRUCTIONS:
        if con.key == base:
            return con.foldable
    for grp in SCALE_VG_GROUPS.get(cls, ()):
        if grp.key == key:
            return grp.foldable
    return True


def scale_vg_group_why(cls: str, key: str) -> str:
    """The declared reason group *key* is one hypothesis -- printed with it."""
    base = key.split(":", 1)[0]
    for con in SCALE_VG_CONSTRUCTIONS:
        if con.key == base:
            return con.why
    for grp in SCALE_VG_GROUPS.get(cls, ()):
        if grp.key == key:
            return grp.criterion
    return ""


#: Classes in *C* whose VG-name coverage has actually been measured.
#:
#: Written down because "no alternate spelling is listed" and "no alternate
#: spelling exists" are the same empty table, and the first is what shipped:
#: `bicycle` was built from one spelling for the whole of #3156 with every
#: structural check passing. A class is added here once its names have been
#: adjudicated against COCO -- ``coco_folds.py`` for the fold-in column,
#: ``vg_name_families.py`` for the spellings COCO's half barely sees, and
#: ``name_evidence.py`` for the verdict.
#:
#: All twelve as of #3618. Listed one by one rather than derived from
#: :data:`SCALE_CLASSES`, because deriving it would mark a *newly added* class
#: audited without anyone having looked -- which is the exact failure this flag
#: exists to make visible. :func:`pilebuild.loaders.vg_scale.load` names the
#: unaudited classes on every build, since the rebuild is when this stops being
#: cheap to fix (#3605).
SCALE_VG_NAMES_AUDITED: frozenset[str] = frozenset(
    {
        "backpack",
        "bicycle",
        "bird",
        "boat",
        "book",
        "bus",
        "clock",
        "dog",
        "kite",
        "knife",
        "stop sign",
        "umbrella",
    }
)


class ClassRule(NamedTuple):
    """One class's review definition: the ``name`` a reviewer sees, and the ``test``."""

    #: What the slate's dataset/detector is called. This is the ONLY thing a
    #: reviewer sees while voting -- files are named by image id alone -- so it
    #: has to carry the discrimination on its own, in a few words.
    name: str
    #: The full wording the name abbreviates: what counts as Good, what counts
    #: as Bad, and the near-miss the short name does not settle. Read by whoever
    #: builds the slate and whoever adjudicates it, so both apply one definition.
    #:
    #: Empty means *not written down yet*, not *no boundary case*: the rule was
    #: measured as a name before :class:`ClassRule` existed and its wording has
    #: never been recorded. Fill one in when a slate of that class is issued --
    #: an unwritten test is the state #3612 exists to end.
    test: str = ""


#: Per-class review definitions, for the classes whose plain English name is not
#: the whole question.
#:
#: A class whose meaning differs between the halves of a dataset is not noisy, it
#: is two classes wearing one name (``make_definition_reslate.py``). The rule
#: that separates them travels in the **dataset name**, because a reviewer
#: cannot see a manifest while voting -- and until this table existed the rule
#: was typed by hand at slate time and written down nowhere, so the wording a
#: re-review used was whatever the next person remembered.
#:
#: Two things are therefore recorded, not one. The ``name`` is what the reviewer
#: reads; the ``test`` is what the name abbreviates, and is what settles the
#: near-misses a two-word name cannot. A rule whose ``test`` lives only in a
#: session transcript is a rule that will be re-derived differently.
#:
#: Classes absent from this table are their own definition, and
#: :func:`review_name` falls back to the bare class name for them. A class need
#: not be in :data:`SCALE_CLASSES` to appear here: `cell phone` is a #3588
#: candidate whose first slate is already voted.
SCALE_CLASS_RULES: dict[str, ClassRule] = {
    # Candidates from #3588, each rule measured with `coco_folds.py` before it
    # was written: the fold-in names the boundary case a reviewer will actually
    # meet. Long form, with the counts, in the annotation guide.
    "truck": ClassRule(
        name="truck incl vans not SUVs",
        test=(
            "Good: pickups, box trucks, semis and tractor units, flatbeds, tow, fire and "
            "food trucks, full-size cargo and panel vans. Bad: SUVs, crossovers and "
            "passenger minivans (those are `car`), and a detached trailer with no cab. "
            "Three tests, in order. (1) Is it a self-propelled road vehicle, or the "
            "powered unit of one? No -> neither Car nor Truck, whatever it is carrying: "
            "that excludes a detached trailer, a bike trailer, a handcart, a caravan "
            "under tow, and it keeps a bobtail tractor unit as the powered half. Not "
            "`does it have a cab`, which fails on any open driving position. "
            "(2) Does the body CARRY a load down a road, or PERFORM WORK at a site? "
            "Carrying is a Truck -- fire engine (fire truck 35 / fire engine 11 / "
            "firetruck 12, none on car), ambulance 19, dump truck 18, tow truck 7, "
            "garbage truck 4, cement mixer 3. Working is neither: a CRANE, and likewise "
            "tractor 24, forklift 2, bulldozer, excavator, backhoe. Plant machinery is "
            "the real reason those are excluded; the old `towed and pushed things` "
            "rationale was wrong about a tractor. A crane on a road-going lorry chassis "
            "is a Truck, a tracked or lattice-boom one is not, and `crane` occurs twice "
            "in the whole overlap. (3) Use the BODY, not the badge, and ask what it was "
            "BUILT FOR: goods is a Truck, people is a Car. Cargo space is the cue, not "
            "the definition -- a bobtail tractor has no cargo space and is a Truck (its "
            "fifth wheel says so), while a car with a tow hitch is still a Car. A "
            "two-seater sportscar is a Car (sports car/coupe/convertible/hatchback are "
            "0 Truck to 15 Car). Accessories change nothing. Never squint inside to "
            "count rows. A long-exposure night shot where traffic is only headlight "
            "streaks has no locatable instance at all: vote Good with NO box, which "
            "excludes the image rather than filing a photograph of traffic as confirmed "
            "no-Car. Same for a photo taken from INSIDE a car -- the Car contains the "
            "camera, so it has no box, and boxing the sun visor would band the visor; "
            "MAX_VOTED_AREA says the same thing, over 80% is not a region but the image. "
            "Car vs BUS is barely a boundary (car->bus 15, bus->car 19, ~0.5% each way) "
            "-- do not spend time there; the case that needs a call is a minibus, which "
            "is boarded through its own door rather than entered by row. A TAXI is not "
            "the hard case either: `taxi` lands on bus ONCE against 62 on car. Service "
            "is a borrowing, not a property -- built-as beats used-as, the same rule "
            "that makes a jar of flowers a Bottle -- so a saloon cab is a Car and a "
            "minibus running as a shared taxi is a Bus. `van` names THREE vehicles and "
            "COCO splits it 261 truck / 318 car / 37 bus; `suv` "
            "does not (62 / 222, an SUV is a Car) and neither does `minivan` (5 / 51). "
            "Disagreement on vans is a known cost of this class, not a mistake."
        ),
    ),
    "car": ClassRule(
        name="car incl SUVs and minivans",
        test=(
            "Good: sedans, hatchbacks, coupes, estates, SUVs, crossovers, passenger "
            "minivans, taxis and cabs -- COCO folds every passenger body into `car`. "
            "Bad: pickups and cargo vans, which are `truck`."
        ),
    ),
    "fork": ClassRule(
        name="fork incl plastic",
        test=(
            "Good: metal, plastic and disposable forks, and serving, carving and fondue "
            "forks. Bad: spatulas, tongs, whisks, skewers. Vote Good only when the boxed "
            "object IS a fork, not when a fork sits somewhere inside a `silverware` or "
            "`utensil` box covering a whole place setting. When only the handle shows and "
            "the food gives nothing away, read the GRIP: a fist closed to stab is a fork, "
            "a spoon is never held that way. Known bad positive: 2322780 boxes a steam "
            "locomotive's cow-catcher as a fork (fork@medium, so rejectable)."
        ),
    ),
    "spoon": ClassRule(
        name="spoon incl plastic not spatulas",
        test=(
            "Good: teaspoons, tablespoons, soup, wooden, plastic, disposable and serving "
            "spoons, and ladles -- a ladle is a spoon with a deep bowl. Bad: spatulas, "
            "slotted turners, scoops, whisks, tongs. Judge the object, not the drawer. "
            "When only the HANDLE shows, read the food: a handle out of cereal is a "
            "spoon, a handle out of a salad is a fork. The one rule here that infers "
            "from surroundings rather than the object, because the alternative deletes "
            "every partly buried spoon."
        ),
    ),
    "cup": ClassRule(
        name="cup incl mugs glasses and stemware",
        test=(
            "Good: a plain drinking glass IS a cup, as are mugs, teacups, paper and "
            "plastic cups, tumblers, pints -- and STEMWARE, which this class was merged "
            "with (see SCALE_CLASS_MERGES): a wine glass, champagne flute, martini glass "
            "or snifter counts. A glass holding cut flowers is still a cup, since `vase` "
            "is only a vessel made as one. Bad: a JAR however it is drunk from (a jar is "
            "a `bottle`; 25 jar boxes are COCO cups), a can, a tin, a carton, and "
            "anything that serves MORE THAN ONE -- a pitcher, jug, carafe, teapot or "
            "thermos is a `bottle`; a bucket is a general-purpose container made for "
            "nothing in particular. The test is portion, not shape: A CUP IS HAND-HELD "
            "AND A SINGLE SERVING."
        ),
    ),
    # `bowl` is the class whose plain name misleads most: `container` is its
    # fourth-largest fold-in (143 boxes), ahead of `pot` and `basket`, and the
    # first name -- "incl plates and dishes" -- said nothing about it. Renamed
    # mid-slate once that showed up.
    "bowl": ClassRule(
        name="bowl incl plates and food containers not wrappers",
        test=(
            "Good: bowls, plates (a paper plate is a plate), saucers, dishes, serving "
            "pots, baskets that hold food, disposable food containers, and a dog's water "
            "bowl. A paper food boat with turned-up sides is a bowl, flimsy or not. "
            "CONTAINING food does not make something a food container: a 5-gallon bucket "
            "of apples is not a bowl, nor is a shopping cart, a grocery store, or a car "
            "boot with the shopping in it. It has to be MADE to hold food. "
            "Bad: flat wrappers and sleeves, cups and mugs (`cup`), sink basins (`sink`), "
            "toilet bowls, feed troughs, planters (`vase`), ashtrays and carafes. "
            "Judge the vessel, not the food -- which answers WHAT TO BOX. Contents "
            "answer WHICH CLASS when the vessel alone is ambiguous: full of soup is a "
            "Bowl whatever its shape; empty, the ladder decides (Bowl 0.66 h/w, Cup 1.26). "
            "No single object is both a Cup and a Bowl, though an image may hold one of each."
        ),
    ),
    "bottle": ClassRule(
        name="bottle incl jars",
        test=(
            "Good: water, wine, beer, soda and spirit bottles, jars ALWAYS and whatever is "
            "in them (a jar of flowers is a bottle, not a vase), jugs and pitchers -- a "
            "pouring vessel serving more than one is a bottle, not a cup -- soap and "
            "shampoo dispensers, shakers, spray bottles, baby bottles, condiment bottles, "
            "vacuum flasks, and the seasoning shelf -- shakers including SALT (16) and "
            "PEPPER (11) shakers, condiment, ketchup, mustard and oil bottles "
            "(181 boxes for the family), and a SQUEEZABLE TUBE -- toothpaste, suntan "
            "lotion, shower gel -- which is reasoned rather than measured (the toiletries "
            "family is 110 boxes but the tube shape itself only ~6). Bad: cans, cartons, boxes, a stemmed "
            "glass of wine, and a "
            "FUEL TANK. An integral component of a larger object is not an instance of a "
            "container class: a mouth is not a food container and a stomach is not a "
            "bottle. Same test as the feed trough in `bench`. "
            "Judge the container, not its contents; do not judge it by its neck, since "
            "`jar` (120) and `jug` (28) fold in and barely have one."
        ),
    ),
    "vase": ClassRule(
        name="vase incl pots and planters",
        test=(
            "Good: only a vessel MADE as one -- vases, flower pots, planters, urns, "
            "pottery. Against an ornamental BOWL, use the box: a vase is TALLER THAN "
            "WIDE (median h/w 1.58, 84% of boxes) and a bowl is wider than tall (0.66, "
            "13%); the middle halves do not overlap. Size does not help -- bowl's median "
            "box is the larger. "
            "A potted plant's pot is a vase; vote the vessel, not the plant. "
            "Bad: a cooking pot on a stove, a plain bowl, and any BORROWED vessel however "
            "it is used -- a jar of cut flowers is a `bottle`, a glass of them a `cup`. "
            "A pitcher or jug of them is a `bottle`: COCO split them (pitcher to cup 30, "
            "jug to bottle 28), so the call is made on portion instead. "
            "Costs 192 boxes, 8.2% of COCO vase, which is the largest narrowing here."
        ),
    ),
    # The guide first named `chair` (53 boxes) as this class's confusion. It is
    # third: `seat` (64) and `table` (58) both outrank it, and each turns on a
    # question COCO's annotators do not ask.
    "bench": ClassRule(
        name="bench not chairs",
        test=(
            "Good: any backed or backless seat BUILT AS SEATING for two or more -- park, "
            "bus-stop and station benches, church pews, picnic-table benches, and a "
            "rowboat's thwart. Bad: a single chair, a sofa, a judge's bench (that is a "
            "table; the seating is the chairs behind it), and a concrete planter wall or "
            "ledge people merely sit on. Two tests: seating or surface, and built as "
            "seating or merely sittable."
        ),
    ),
    "chair": ClassRule(
        name="chair incl stools not couches",
        test=(
            "Good: dining, office, folding and deck chairs, armchairs, high chairs, "
            "stools and bar stools, and one seat within a row of stadium or theatre "
            "seating; `seat`/`seats` (269) is the third largest fold-in here. One seat "
            "is a Chair, two or more a couch -- upholstered is NOT the test, so a club "
            "chair or recliner counts. A lifeguard station counts (built as seating for "
            "one). Bad: couches and sofas (`couch`), benches, a TOILET (separate COCO "
            "class, zero confusions), and a CAR SEAT -- a component is not an instance, "
            "and counting them would fire on every street scene. A car seat REMOVED from "
            "the car is free-standing, so it counts; so do a motorcycle's seat and a "
            "saddle NOT (both are part of the vehicle or the tack, and both are zero "
            "boxes in COCO). Someone clearly sitting on an INVISIBLE chair: vote Good and "
            "draw NO box -- present, no size measured, which excludes the image from the "
            "negative pool without making it a positive. A PART INHERITS THE RULING OF ITS "
            "WHOLE: a chair back or leg is evidence of a Chair and you box the Chair, but "
            "a headrest in a car is part of a car seat, so it is not one."
        ),
    ),
    "sink": ClassRule(
        name="sink basin not counter",
        test=(
            "Good: kitchen sinks, bathroom sinks, pedestal basins, utility sinks, vessel "
            "basins; a double sink in one unit is one sink. Bad: bathtubs, showers, "
            "toilets, urinals. This class's risk is the BOX, not membership: box the "
            "basin and its tap, never the vanity or the run of counter."
        ),
    ),
    "fire hydrant": ClassRule(
        name="fire hydrant not standpipes",
        test=(
            "Good: street fire hydrants in any colour or design, including ones wrapped, "
            "repainted or half-buried in snow. Bad: building standpipes and wall-mounted "
            "siamese connections, bollards, water valves, parking meters, utility posts. "
            "Also Bad: a FIRE TRUCK (that is a `Truck`, another of the thirteen), and "
            "busted street plumbing whose break is hidden under water -- you cannot "
            "confirm what you cannot see. Good-with-no-box requires certainty of "
            "PRESENCE; unsure whether anything is there at all is Bad. "
            "The cleanest class measured -- a call that feels hard here usually means the "
            "object is something else."
        ),
    ),
    # COCO has no magazine class, so its annotators put magazines in `book`
    # while the human pass applied the narrower English reading -- leaving 21
    # verdicts on one definition and 49 on another. The dataset takes COCO's,
    # since that is the half with an exhaustive reference.
    # The class's whole risk is landlines: VG `phone` lands on no COCO class
    # 46.2% of the time, worse than `book`'s 43.3%. The first slate's test read
    # "anything with a cord or a base station is Bad", which discriminates on a
    # base being PRESENT when what it means is that the handset is not itself
    # the whole device -- so it rejected 2387021, a mobile phone in a charging
    # dock (#3612).
    "cell phone": ClassRule(
        name="cell phone not landlines",
        test=(
            "Bad if the handset needs the base to work -- landline handsets, desk phones, "
            "payphones, wall phones, intercoms. A mobile phone resting in a charging dock "
            "or cradle is still Good."
        ),
    ),
    # ------------------------------------------------------------------
    # The SHIPPED twelve (#3673). #3666 measured what their absence costs:
    # six of the nine pool-error finds in the negative pass were boundary
    # calls on rules that did not exist, and at a 1% rate one ruling moves a
    # class further than 3,000 extra uniform draws would.
    #
    # Every entry below was measured before it was written, with BOTH tests,
    # because the cheaper one gets two of them wrong. `coco_folds.py` gives the
    # box test -- which VG names land on a COCO box of the class -- and it says
    # COCO's annotators call a wristwatch a `clock` 35 times and a `canopy` or
    # `tent` an `umbrella` 58 times, more than `parasol`. Read alone it would
    # have folded both in. `name_evidence.py` gives the image test the pool
    # actually asks -- where the name is the SOLE evidence, does COCO find the
    # class? -- and refutes both: `watch` 11% against a 4.5% base, `canopy` 7%
    # and `tent` 10% against 3.7%, all under the 1/3 cut, all verdict
    # `neither`. A fold-in tail is COCO's inconsistency; it is not a definition.
    "clock": ClassRule(
        name="clock not watches",
        test=(
            "Good: a device whose job is showing the time and which stands, hangs or is "
            "mounted -- wall, tower, station, mantel, alarm and desk clocks, analogue or "
            "digital, and a bare clock face on a building. Bad: a WRISTWATCH or a watch on "
            "a table (`watch` was measured for this class and refused: over the 970 overlap "
            "images where it is the only evidence COCO finds a clock 11% of the time "
            "against a 4.5% base, with 3% box agreement -- the 35 COCO clock boxes a VG "
            "`watch` box lands on are a tail, and admitting them would define the class one "
            "way on the COCO half and another way on the half VG names alone). Also Bad: a "
            "departure board or scoreboard that happens to show the time (`display` scores "
            "2%), a clock drawn on a screen or printed on a page -- the depiction rule "
            "applies to every class -- and a sundial. The near-miss this settles is a "
            "wristwatch worn by a bystander, which is what the negative pass found (#3666)."
        ),
    ),
    "umbrella": ClassRule(
        name="umbrella incl parasols not canopies",
        test=(
            "Good: a hand-held umbrella open or furled, a parasol, and a beach or patio "
            "umbrella -- one central pole carrying a round canopy. Bad: a pop-up CANOPY or "
            "market stall, a tent, an awning over a shopfront, a sunshade sail. The test is "
            "the frame, not the shade it casts: ONE POLE AND A ROUND TOP is an umbrella, "
            "FOUR LEGS OR A WALL FIXING is not. Measured, and the box test disagrees with "
            "the image test here: COCO's annotators land `canopy` on a COCO umbrella box 32 "
            "times and `tent` 26, together more than `parasol`'s 38 -- but over the images "
            "where those names are the only evidence COCO finds an umbrella 7% and 10% of "
            "the time against a 3.7% base (`awning` 4%, `shade` 1%), all verdict `neither`. "
            "The near-miss this settles is a rank of pop-up canopies at a skate park (#3666)."
        ),
    ),
    "backpack": ClassRule(
        name="backpack not handbags or luggage",
        test=(
            "Good: a bag made to be carried on the back on shoulder straps -- rucksacks, "
            "daypacks, school bags, hiking packs -- whether worn, held or set down. Bad: a "
            "handbag, a shoulder or messenger bag, a suitcase, a duffel, a camera bag. COCO "
            "carries `handbag` and `suitcase` as their own classes, so this line is COCO's "
            "too. Two straps over two shoulders is the cue; a single diagonal strap is a "
            "shoulder bag. `bookbag` is on the ambiguous list (85% precision, 88% box) and "
            "`pack` is not a name for anything (38%). The near-miss this settles is the "
            "hump under a motorcyclist's leathers, which the pass could not call (#3666)."
        ),
    ),
    "stop sign": ClassRule(
        name="stop sign not other signs",
        test=(
            "Good: the octagonal red STOP sign, on a post, on a school bus arm, or held; "
            "from behind ONLY when the octagon is readable in the silhouette. Bad: every "
            "other traffic and street sign -- yield, one way, speed limit, street names -- "
            "a stop sign painted on the road, a pictogram, and a blank sign back whose "
            "shape you cannot read. `sign` is deliberately NOT a name for this class even "
            "though it carries 46.6% of COCO's stop-sign boxes: a VG `sign` box is a stop "
            "sign 7.9% of the time, which would withhold 12.7 pool images per contaminated "
            "negative retired (#3618, #3635). The near-miss this settles is the blank "
            "aluminium back of a sign on a street-name pole (#3666)."
        ),
    ),
    "book": ClassRule(
        name="book incl magazines",
        test=(
            "Good: a bound book, and also a magazine, a notebook and a bound pamphlet -- "
            "COCO has no magazine class and annotates magazines as `book`, which is the "
            "reading this dataset uses. Bad: newspapers, loose paper, letters, posters, "
            "menus, printouts, and a screen showing text. ONE TEST: IS IT BOUND ALONG A "
            "SPINE? Bound is a book; folded or loose sheets are not. Measured on the "
            "overlap, and read against `book`'s own 13% self-match rather than against "
            "100%: `magazines` 11% and `magazine` 10% land on a COCO book box at the same "
            "rate as `book` itself, `newspaper` 3% at a quarter of it, `menu` and `paper` "
            "at ~1%. This class is the study's calibration failure -- 43% of its VG boxes "
            "land on no COCO class, the worst of the twenty-five -- and the bound test "
            "narrows it without repairing that."
        ),
    ),
    "bird": ClassRule(
        name="bird any species not cooked",
        test=(
            "Good: any live bird, wild or domestic, of any species -- ducks, geese, gulls, "
            "pigeons, swans, parrots, ostriches, owls, eagles, flamingos, peacocks, hens "
            "and roosters. Bad: a COOKED bird on a plate, a feather or a wing on its own, a "
            "bird figurine, a bird on a sign or a logo. The cooked clause is not "
            "hypothetical: `chicken` names 428 overlap images and COCO finds a bird on 10% "
            "of them, `turkey` 53 images at 12% -- in VG both words are usually food. "
            "`crane` is the other trap and it is a machine: 308 images, 2%. None of the "
            "three can be a name for this class, but a reviewer looking at a live one "
            "should vote Good."
        ),
    ),
    "boat": ClassRule(
        name="boat any watercraft",
        test=(
            "Good: anything built to travel on water and carrying its own hull -- ships, "
            "ferries, yachts, sailboats, canoes, kayaks, rafts, rowboats, gondolas, barges, "
            "tugs, pedal boats, jet skis. On a trailer or in dry dock still counts. Bad: a "
            "surfboard or paddleboard (COCO carries `surfboard` separately), a sail or a "
            "mast on its own, a dock, a buoy, a boat on a sign. `sailboat`, `canoe`, "
            "`kayak`, `raft` and `ship` are already folded in; `yacht`, `ferry`, `rowboat` "
            "and `barge` all measure 88-100% precision on small samples and are candidates "
            "for the same treatment."
        ),
    ),
    "bus": ClassRule(
        name="bus incl coaches not trams",
        test=(
            "Good: a road vehicle built to carry passengers in rows and boarded through "
            "its own door -- city buses, coaches, school buses, double-deckers, minibuses, "
            "tour buses, trolleybuses on tyres. Bad: a TRAM or train on rails, a cargo van, "
            "a truck, an RV or camper, a bus shelter. The boundary that costs verdicts is "
            "the van: `van` names three different vehicles and COCO splits it 261 truck / "
            "318 car / 37 bus, so read the BODY -- rows of seats and a passenger door is a "
            "Bus, a cargo box is a Truck. Two VG words are traps for a NAME and not for a "
            "reviewer: `coach` is usually a person -- 0% precision over its 50 sole images -- "
            "and `trolley` is usually a shopping cart (31% over 32)."
        ),
    ),
    "bicycle": ClassRule(
        name="bicycle incl trikes not motorcycles",
        test=(
            "Good: a human-powered pedal cycle -- road, mountain, BMX, folding, cargo and "
            "children's bicycles, ridden, parked or on a rack; a tricycle counts (71% "
            "precision, and COCO boxes six as `bicycle`). Bad: a MOTORCYCLE, moped or "
            "scooter (COCO carries `motorcycle` separately, and `motorcycle` measured as an "
            "alias of `bike` at 0.38 box IoU), an exercise bike, a wheel or a bike rack "
            "alone, and a bicycle PICTOGRAM on a road sign -- three of the ten "
            "`bicycle@small` positives in #3156 are exactly that, boxed as `bicycle` by "
            "COCO, and the depiction rule wins over COCO's box (#3614). The class is built "
            "from the spelling `bicycle` alone while `bike` carries 638 of COCO's 3,683 "
            "boxes against `bicycle`'s 775, so it is missing roughly half its positives on "
            "the non-COCO half (#3605)."
        ),
    ),
    "kite": ClassRule(
        name="kite incl parasails and parachutes",
        test=(
            "Good: a kite flown on a line, and -- this is the surprise, and it is COCO's "
            "reading, not ours -- a PARASAIL, a paraglider and a PARACHUTE: `parasail` "
            "lands on a COCO kite box 57 times and `parachute` 26, and both are already "
            "folded into this class. Bad: a flag, a banner, a balloon, a bird, a windsock, "
            "a kite tail or string on its own. A kite lying on the ground still counts."
        ),
    ),
    "knife": ClassRule(
        name="knife incl butter knives and servers",
        test=(
            "Good: a bladed cutting or spreading implement at the table or in the kitchen "
            "-- table, steak, butter, bread, chef's, paring and pocket knives, cleavers, "
            "and cake or pizza servers. Bad: SCISSORS (COCO carries its own class, and VG "
            "`scissors` finds a COCO knife on 3% of its 196 sole images), a spatula, a "
            "peeler, a knife block or a drawer with nothing visible, and a whole "
            "`silverware` or `utensil` box covering a place setting -- vote Good only when "
            "the boxed object IS the knife, the same rule `fork` carries. Where only the "
            "handle shows, read the blade line, not the food."
        ),
    ),
    # The remaining rules were measured as names before ``test`` existed
    # (#3588): `coco_folds.py` asks which VG names land on a COCO class's boxes
    # over the ~51k-image overlap, which enumerates a class's boundary cases
    # before a human meets one -- run against `book` it prints `magazine` (79)
    # and `magazines` (30). Each name below states the boundary case that
    # measurement found, and the long form now lives in each entry's ``test``
    # above -- filled in as each class was slated (#3588).
}


def review_name(cls: str, suffix: str = "") -> str:
    """The dataset/detector name a slate of *cls* is reviewed under.

    The class's rule name where it has one, else the bare class name, plus an
    optional *suffix* naming the pass (``positives``, ``audit``). Every slate
    maker builds its ``detector`` column from this, so a class's rule reaches
    the reviewer whichever pass they are voting -- the first pass included,
    which is where a definition split does its damage.
    """
    rule = SCALE_CLASS_RULES.get(cls)
    return f"{rule.name if rule else cls}{f' {suffix}' if suffix else ''}"


def scale_vg_wanted() -> set[str]:
    """Every VG name the ``vg_scale`` read must match, spellings included.

    The class names themselves plus both name tables. Read with this rather than
    ``set(SCALE_CLASSES)``: a spelling absent from the read is invisible later,
    since an image holding it then looks like an image holding nothing.
    """
    wanted = set(SCALE_CLASSES)
    for table in (SCALE_VG_NAMES, SCALE_VG_AMBIGUOUS):
        for names in table.values():
            wanted.update(names)
    return wanted


#: Images per ``(class, band)`` cell, and the shared negative pool every cell
#: draws from. The pool is the whole of VG, labelled by VG and repaired from
#: COCO where an exhaustive reference exists, so the binding supply is the union
#: of both halves; the builder logs any cell it cannot fill.
#:
#: Cells are **designated**, not inferred: each is exactly these positives plus
#: this negative pool, and every other image in the pickle is excluded from it.
#: Prevalence is therefore identical in all 36 cells by construction, which is
#: what makes small-vs-large a paired comparison rather than two datasets with
#: different difficulty. Unequal prevalence between arms is what made wave 1 and
#: wave 2 of the overview benchmark non-comparable.
SCALE_N_POS = int(os.environ.get("VTS_SCALE_N_POS", "100"))
SCALE_N_NEG = int(os.environ.get("VTS_SCALE_N_NEG", "9900"))
#: Extra negatives drawn into the pickle but designated into no cell. A human
#: verdict can retire a contaminated negative later; re-designating from a spare
#: is a relabel, while drawing a fresh one would mean re-embedding every cell.
SCALE_N_NEG_SPARE = int(os.environ.get("VTS_SCALE_N_NEG_SPARE", "300"))

#: What the shared negative pool is made of (#3670).
#:
#: ``provable``
#:     Every designated negative is COCO-scored, so "holds no bus" is a FACT --
#:     COCO annotates all eighty of its classes on any image it touches -- rather
#:     than VG's silence, which #3666 measured wrong **1.40%** [0.68, 2.86] of
#:     the time pooled over the shipped twelve, and #3635 predicts between
#:     **0.28% and 2.87%** depending on the class.
#: ``matched``
#:     The negatives' COCO share is matched to the positives' own (~57%), so
#:     provenance carries no information about the label. Keeps the
#:     contamination, and keeps the whole negative review (#3670 measured that
#:     price: `provable` rules 513 of 743 reviewed negatives ineligible).
#:
#: **Chosen on the SPREAD, not on the magnitude, and the difference matters.**
#: Both compositions distort, and on #3667's FPR-inflation scale the two are not
#: separable:
#:
#: * an all-provable pool hands a head a provenance shortcut, because off-COCO
#:   then implies positive -- **1.1x** (`provenance_shortcut.py`; 1.06-1.12 over
#:   two embedders and two independent routes, a reverse arm and the residual
#:   after subtracting predicted contamination);
#: * a mixed pool carries contamination -- **1.18x** at #3666's pooled 1.40%,
#:   with a 95% interval of [1.09, 1.37] that contains the first number.
#:
#: So this is NOT "all-provable is less distorted": at the pooled rate the two
#: overlap, and an earlier reading of this trade quoted 1.32x by taking the top
#: of a *predicted* per-class range that has since been measured lower. What
#: separates them is that the provenance shortcut is **uniform across classes**
#: while contamination is not -- 1.04x to 1.37x, class by class. `vg_scale`
#: exists to compare one class against another and one band against another, and
#: a distortion that varies per class is the one that makes those comparisons
#: unreadable. A uniform one moves every cell together and cancels in the
#: contrast. (#3667's cross-class shortcut, for scale, was 1.88x and justified
#: rebuilding eleven cells.)
#:
#: Switching back to ``matched`` needs the off-COCO stratum EMBEDDED, which the
#: all-provable build does not carry -- it is a rebuild, not a relabel.
SCALE_NEG_COMPOSITION = os.environ.get("VTS_SCALE_NEG_COMPOSITION", "provable")

#: The **designed** prevalence of a `vg_scale` cell: 100 positives per band x 3
#: bands against the shared negatives, which #3670 took from 3,900 to 9,900
#: so a cell's prevalence is 1%. Named because `vg_scale_deep` has to
#: REPRODUCE it rather than re-derive it, and because every k* this family of
#: studies quotes is computed from it (`-log2((1-pi)/pi) = -3.71`).
#:
#: DESIGNED, not realised, and since #3667 the two differ. The harness scores
#: the *evaluable pool*, which grew ~45% when each class gained the other
#: eleven's COCO-exhaustive positives as negatives -- no positive was added, so
#: prevalence fell: `vg_scale_any` to **4.99%** and `vg_scale_deep` to **5.09%**
#: (k* -4.25 and -4.22, against the -3.70 this constant gives). The two cells
#: moved TOGETHER, 0.03 bits apart, so the "only depth changed" premise below
#: survives; what does not survive is quoting -3.71 as the dataset's own
#: optimum. See #3681 and
#: `docs/experiments/2026-09-06-cross-class-negatives-3667/REPORT.md`.
SCALE_PREVALENCE = (3 * SCALE_N_POS) / (3 * SCALE_N_POS + SCALE_N_NEG)

#: `vg_scale_deep`'s positives per class (#3547). 900 is the deepest value all
#: twelve classes support band-free -- `stop sign`, the thinnest, has 1006
#: candidates (`measure_supply.py`) -- and it is chosen against `preflight.sh`
#: check 16b, which clears only when the sim half holds MORE positives than the
#: horizon has steps: at `SIM_FRACTION` 0.5 that is 450 against 400.
#:
#: Going deeper costs classes, not money: 1200 drops `kite` and `stop sign`,
#: and a class list that differs from #3319's would confound the horizon axis
#: with a vocabulary axis in the one comparison this dataset exists to make.
SCALE_DEEP_N_POS = int(os.environ.get("VTS_SCALE_DEEP_N_POS", "900"))
#: `vg_scale_deep` does NOT follow #3670's expansion, and the pin is deliberate.
#: Deriving its pool from the live `SCALE_PREVALENCE` would have taken it from
#: 11,700 negatives to 29,700 as a silent side effect of a change to a DIFFERENT
#: dataset. Deep exists for one comparison -- the #3319/#3547 acquisition horizon
#: -- and moving its prevalence mid-stream would confound that axis with a
#: prevalence axis, which is the argument `SCALE_DEEP_N_POS` already makes about
#: holding the class list fixed. Whether deep should follow is #3690.
#:
#: Still DERIVED, never set: a negative pool written as a literal beside a
#: positive count is how prevalence drifts. What is pinned is the `vg_scale`
#: pool size deep's prevalence refers to, not the pool itself.
SCALE_DEEP_PIN_N_NEG = int(os.environ.get("VTS_SCALE_DEEP_PIN_N_NEG", "3900"))
SCALE_DEEP_PREVALENCE = (3 * SCALE_N_POS) / (3 * SCALE_N_POS + SCALE_DEEP_PIN_N_NEG)
SCALE_DEEP_N_NEG = round(SCALE_DEEP_N_POS * (1 - SCALE_DEEP_PREVALENCE) / SCALE_DEEP_PREVALENCE)
SCALE_DEEP_N_NEG_SPARE = int(os.environ.get("VTS_SCALE_DEEP_N_NEG_SPARE", "300"))


#: How far a VG copy's aspect ratio may drift from the COCO original before its
#: boxes are considered untransferable. Normalised coordinates survive a rescale
#: but not a re-crop or a rotation, and 49 of the 51,497 overlaps are one of
#: those -- small enough to ignore by accident, which is why it is a constant
#: with a check rather than an assumption.
MAX_ASPECT_DRIFT = float(os.environ.get("VTS_MAX_ASPECT_DRIFT", "0.01"))


#: The coordinate space a correction box is recorded in. VG's and COCO's boxes
#: arrive in **pixels**; a correction box comes from the app's ``region_box``,
#: which is already **normalised** to [0, 1]. The builder divides every box by
#: (W, H) on the way into the pickle, so a correction box merged in unconverted
#: is normalised twice: it lands on the frame origin, sub-pixel, and takes its
#: band with it (#3281 -- 130 boxes, and 97 images filed in ``@small`` whose
#: object is medium or large). The space is therefore *declared* in the file and
#: converted once at read, never inferred: the two spaces are indistinguishable
#: for a box in the top-left corner of a 1x1 image, which is exactly the shape
#: the bug produced.
CORRECTION_BOX_SPACE = "normalised"

#: Below this normalised side length a box is sub-pixel on any image the pile
#: holds -- VG's largest copy is 1280 px wide -- so it cannot describe anything
#: that was observed. Zero legitimate boxes are anywhere near it; the 130
#: double-normalised ones were all under 1e-3.
MIN_BOX_SIDE = float(os.environ.get("VTS_MIN_BOX_SIDE", "0.000244"))  # 1/4096

#: "Crushed to the origin": both corners inside the top-left square holding this
#: fraction of the frame area. Unlike the sub-pixel rule this one has genuine
#: hits -- a small object really can sit in the top-left corner, 43 of 3470
#: healthy boxes do -- so it gates on the *rate*, not on any single box.
CORNER_AREA_FRAC = float(os.environ.get("VTS_CORNER_AREA_FRAC", "0.01"))

#: The share of a cell's boxes that may be crushed to the origin before the
#: build is refused. The measured healthy rate is 1.2% and the defect put it at
#: 100% of the affected images, so anything in between separates them.
MAX_CORNER_RATE = float(os.environ.get("VTS_MAX_CORNER_RATE", "0.05"))


#: Which images each cell currently holds. Selection is hash-stable, but a
#: roster is what carries membership across a CHANGE of selection rule -- and
#: across the corrections that are the whole point of the review, since a review
#: is only worth what it still covers after the next rebuild.
ROSTER = Path(os.environ.get("VTS_SCALE_ROSTER", str(PILE / "vg_scale_roster.json")))

#: `vg_scale_deep`'s own roster. Separate from `ROSTER` on purpose: the two
#: datasets designate different cells from the same candidates, and one file
#: holding both would let a `vg_scale_deep` rebuild retire images `vg_scale`'s
#: review is pinned to.
DEEP_ROSTER = Path(os.environ.get("VTS_SCALE_DEEP_ROSTER", str(PILE / "vg_scale_deep_roster.json")))


def scale_cell(category: str, band: str) -> str:
    """The band-suffixed category name a harness cell is keyed on.

    One pickle carries all three bands, distinguished by this suffix, because a
    cell is already ``(dataset, category)`` -- so the bands need no harness
    change, embedding is done once instead of three times, and the bands are
    paired on identical negatives.
    """
    return f"{category}@{band}"


#: Embedders in the pile. ``patch`` embedders attach ``patch_grid`` and are the
#: only ones that can carry a region-voting arm. ``batch`` is the GPU forward
#: batch size (``VTSEARCH_EMBED_BATCH_SIZE``); the app's default of 32 is sized
#: for a modest card and wastes a build GPU on a base-sized encoder, while a
#: SO400M/384 model at 32 is already the heavy end. Sizes are per model, not per
#: run, so a fatter card only means the whole table can move up.
#:
#: Batch size does not change what is embedded: in fp32 it shifts vectors by
#: ~1e-7 through kernel selection, orders of magnitude below anything the
#: studies resolve.
#: Deliberately three, not five. ``siglip`` is the shipped default and
#: ``siglip2_l`` the premium end; the middles (``siglip_l``, ``siglip2``) were
#: dropped because a study learns little from interpolating between them, and
#: the compute is better spent on more runs of the endpoints.
#:
#: The cost of that: ``siglip`` -> ``siglip2_l`` moves generation (1 -> 2) and
#: capacity (base -> SO400M) at the same time, so a difference between them
#: cannot be attributed to either alone. Rebuild a middle column if a result
#: ever needs that split -- ``build_pile.py --embedders siglip2`` restores one.
#: The two CLIP columns are **evaluation only** (#3292) and exist to test whether
#: #3287's `calibration_fraction` optimum follows single-vector geometry or just
#: the SigLIP lineage.  Both are run, not one, because a single CLIP arm cannot
#: separate the two things that change when you leave SigLIP:
#:
#:   `clip`   ViT-B/32, 512-d - the checkpoint the app already ships
#:   `clip_l` ViT-L/14, 768-d - dimension-matched to `siglip`, so a difference
#:                              cannot be "CLIP's vectors are narrower"
#:
#: Agreement between them is what licenses reading their verdict as CLIP's
#: lineage rather than CLIP's capacity.  Neither is selectable in the app
#: (`MediaEmbedder.eval_only`); `clip_l` is not a production candidate at all.
EMBEDDERS: dict[str, dict] = {
    "siglip": {"patch": False, "batch": 128},
    "siglip2_l": {"patch": False, "batch": 32},
    "clip": {"patch": False, "batch": 128},
    # ViT-L/14 at 224px: ~3x the base encoder's activation, so half the batch.
    "clip_l": {"patch": False, "batch": 64},
    # Patch embedders hold an (N, H, W, D) grid per image, not one vector, so
    # they carry far more activation memory per item than their backbone size
    # alone suggests.
    "dinov3_patch": {"patch": True, "gated": True, "batch": 64},
}


def embed_batch_size(embedder: str) -> int | None:
    """This embedder's ``VTSEARCH_EMBED_BATCH_SIZE``, or ``None`` for the default."""
    val = EMBEDDERS.get(embedder, {}).get("batch")
    return int(val) if val else None


def cells() -> list[tuple[str, str]]:
    """Every ``(dataset, embedder)`` cell in the full grid."""
    return [(ds, emb) for ds in DATASETS for emb in EMBEDDERS]


def pickle_name(dataset: str, embedder: str) -> str:
    return f"{dataset}__{embedder}.pkl"


def cell_path(dataset: str, embedder: str) -> Path:
    return EMBEDDINGS / pickle_name(dataset, embedder)


def provenance_path(dataset: str, embedder: str) -> Path:
    """Sidecar recording *which machine* produced this cell (#3160).

    Beside the pickle rather than inside it: a cell built before this existed
    stays loadable, and the sidecar can be read (or backfilled) without paying
    to unpickle a 900 MB file.
    """
    return EMBEDDINGS / f"{dataset}__{embedder}.provenance.json"


def is_patch_embedder(embedder: str) -> bool:
    return bool(EMBEDDERS.get(embedder, {}).get("patch"))


def region_capable(dataset: str, embedder: str) -> bool:
    """True when this *cell* can actually region-vote.

    Both halves are required: ground-truth boxes to drag (dataset) and a patch
    grid to pool them over (embedder). Stated per cell precisely because the
    per-dataset flag alone reads as "this arm region-votes" and does not.
    """
    return bool(DATASETS.get(dataset, {}).get("boxed")) and is_patch_embedder(embedder)


def require_demo_source(dataset: str) -> None:
    """Fail loudly if a demo dataset's source is not staged in the datadir.

    The demo downloaders treat a *missing* extraction dir as "not downloaded
    yet" and go fetch it. On a datadir that lost its symlink into the shared
    demo cache, that silently substitutes a partial re-download for the real
    dataset: the build still succeeds, but the cell holds a truncated subset
    and disagrees with its sibling cells. Cheaper to block than to detect.
    """
    name = DATASETS.get(dataset, {}).get("source_dir")
    if not name:
        return
    src = DATADIR / name
    if not src.exists():
        raise SystemExit(
            f"{dataset}: demo source {src} is missing, so the loader would re-download it.\n"
            f"  Link the shared cache in first, e.g.\n"
            f"    ln -s {DEMO_CACHE}/{name} {src}"
        )
    if not any(src.iterdir()):
        raise SystemExit(f"{dataset}: demo source {src} is empty (an empty dir reads as 'download complete')")


def setup_env() -> None:
    """Point vtscore + HF at the pile. Call before importing anything vtscore."""
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    import _expcommon  # noqa: PLC0415

    # Default to the checkout this file lives in, rather than requiring VTS_REPO.
    # Depending on the env var is a live hazard: with it unset, ``import vtscore``
    # falls through to the venv's editable install, which points at the *main*
    # checkout -- 592 commits stale at the time of writing, and missing embedders
    # this pile uses. A build that resolved there would embed against different
    # code with no error. (This is how the shadow-module trap actually bites:
    # `VAR=x cmd1 && cmd2` applies VAR to cmd1 only, so the second command
    # silently ran against the wrong tree.)
    repo = os.environ.get("VTS_REPO") or str(Path(__file__).resolve().parents[3])
    os.environ["VTS_REPO"] = repo  # so calibration's common.py agrees with us
    # `_expcommon.setup_env` puts `repo` first on sys.path and drops the venv's
    # editable-install finder, so `import vtscore` resolves to this checkout
    # rather than whichever clone that install points at.
    _expcommon.setup_env(repo=repo, datadir=DATADIR, models_dir=MODELS, hf_home=MODELS)


# --------------------------------------------------------------------------
# #3588: candidate additions to C, and the definition each is reviewed under
# --------------------------------------------------------------------------

#: Candidates for an expanded *C*, measured rather than proposed.
#:
#: Issue #3588 asks for the class list to sample *context exclusivity* on
#: purpose instead of by accident. Its own proposal does not survive the gate
#: it correctly specifies: of the classes it names, `airplane` (85 small),
#: `train` (34), `zebra` (27), `elephant` (20), `giraffe` (14), `cat` (44),
#: `suitcase` (54) and `potted plant` (1) all miss the 100-per-band floor, and
#: `motorcycle`, `surfboard`, `snowboard` and `skateboard` each carry a
#: measured alias partner (`bike` 0.38; `board` 0.45/0.50/0.52 --
#: `scan_name_overlap.py`). `traffic light` fails twice: 58 in the large band,
#: and its head noun `light` is already barred by :func:`scale_study_exclusion`.
#:
#: What survives is listed here. It is *not* the symmetric design the issue
#: asked for, and the asymmetry is structural rather than a sampling accident:
#: every scene-exclusive class the issue wanted (train, zebra, giraffe,
#: elephant) fails on the SMALL band specifically, because an animal or vehicle
#: that owns its scene is photographed filling the frame -- `giraffe` has 14
#: small-band images against 1,279 large. Context exclusivity and small-band
#: supply are anti-correlated in VG, so the easy end of the axis cannot be
#: widened with this source at this floor. These additions widen the hard end
#: and add same-scene partners; see the report for what that costs the design.
SCALE_CANDIDATES_3588: tuple[str, ...] = (
    # Tier A -- a habitat partner of a class already in C, so the negative pool
    # is shared and the contrast is same-scene, different-object.
    "truck",  # partner of `bus`      -- street, large vehicle
    "car",  # partner of `bus`/`truck` -- street, and the generic-clutter end
    "fork",  # partner of `knife`     -- table setting
    "spoon",  # partner of `knife`    -- table setting
    # Tier C -- objects whose surroundings ARE the negative pool.
    "cup",
    "bowl",
    "bottle",
    "vase",
    "bench",
    "chair",
    "sink",
    "cell phone",
    "fire hydrant",
)

#: VG names that denote one of the CANDIDATES. VG's vocabulary is free text and
#: :func:`pilebuild.vgsource.vg_boxes_by_name` matches the PRIMARY name only, so
#: a class built from one spelling silently drops the others.
#:
#: Only merges measured as aliases are listed. `fire hydrant` / `hydrant` is one
#: object under two spellings (box IoU 0.77/0.74, `scan_name_overlap.py`), and
#: `hydrant` accounts for 266 of the 835 COCO `fire hydrant` boxes on the
#: overlap -- taking `fire hydrant` alone would throw away a third of the class.
#: `phone` is listed for `cell phone` on the same evidence (541 boxes) **and is
#: the single riskiest entry here**: see `SCALE_CLASS_RULES`.
#:
#: **Deliberately not :data:`SCALE_VG_NAMES`**, which is the same measurement for
#: a class already in *C*. The two cannot share a table because they are read at
#: different times by different code: `SCALE_VG_NAMES` widens the ``vg_scale``
#: READ and is folded by :func:`pilebuild.loaders.vg_scale.canonicalise` on every
#: build, so an entry there for a class outside *C* would change the built
#: dataset -- and nothing here has been decided yet (#3604). This table is read
#: only by `make_class_slate.py`, which bands a candidate without touching the
#: pickle. A candidate promoted into *C* moves its row across, minus the class
#: name itself: the entries here list the class name too, because the slate
#: builder has no separate class-name read to add it to.
SCALE_CANDIDATE_VG_NAMES: dict[str, tuple[str, ...]] = {
    "fire hydrant": ("fire hydrant", "hydrant"),
    "cell phone": ("cell phone", "phone", "cellphone"),
    # The stemware half of the merged `cup` (see SCALE_CLASS_MERGES). Measured
    # against COCO `wine glass` boxes: `wine glasses` 16, `wineglass` 9,
    # `goblet` 8, `champagne glass` 5, `champagne flute` 5; `mug` 238 is the
    # cup half's own missing spelling.
    #
    # `glass` is NOT here, and the reasoning that briefly put it here is worth
    # keeping because it was subtly wrong. Its fold-out is 62.2% onto the merged
    # class (1,146 of 3,224 VG boxes on COCO `cup`, 861 on `wine glass`), well
    # clear of `bike`'s 40.1%, which reads as a usable alias.
    #
    # But a fold-out rate is NOT a positive-precision rate. Fold-out is measured
    # only on the COCO-annotated half, where a reference exists; POSITIVES are
    # drawn from all of VG, including the half with no check, and there the 35%
    # of `glass` boxes that land on no COCO class -- windowpanes, eyeglasses --
    # arrive as positives unexamined. The review measured the damage: the merged
    # `cup` slate rejected 9 of 30 boxed positives, 30%, against 0-17% for every
    # other class, and worst in the LARGE band at 40%, which is where a
    # windowpane lands. `glass` is in SCALE_CANDIDATE_VG_AMBIGUOUS instead.
    "cup": (
        "cup",
        "mug",
        "wine glass",
        "wine glasses",
        "wineglass",
        "goblet",
        "champagne glass",
        "champagne flute",
    ),
}

#: Candidate-class spellings that MAY denote the class but also denote something
#: else -- :data:`SCALE_VG_AMBIGUOUS` for classes not yet in *C*.
#:
#: Same three-valued treatment, for the same reason: a box under one of these is
#: evidence in neither direction, so it is dropped from the bands *and* bars its
#: image from the shared negative pool. :func:`pilebuild.loaders.vg_scale.lift_ambiguous`
#: does both, and exempts any image COCO annotates exhaustively or a reviewer has
#: ruled on -- there the question is already answered and the spelling is ignored.
#:
#: `glass` is the entry that named this table. It is 62.2% good by fold-out, which
#: is why it was first tried as a plain alias, and the 35% that is windowpanes and
#: eyeglasses still cost `cup` 30% of its boxed positives -- the measurement is in
#: the comment on that table above.
SCALE_CANDIDATE_VG_AMBIGUOUS: dict[str, tuple[str, ...]] = {
    "cup": ("glass",),
}


#: Classes this project defines as the UNION of several COCO classes.
#:
#: Distinct from every other table here, and the distinction is the whole point:
#: an alias merge (:data:`SCALE_CANDIDATE_VG_NAMES`) says two *names* denote one
#: object, which is a measurement. This says we are choosing a class boundary
#: COCO did not draw, which is a decision.
#:
#: It is available at all only because both halves are COCO classes. The scored
#: subset -- the fifth of each slate carrying a COCO answer, and the only reason
#: a reviewer's residual error is a number rather than a hope -- survives a union
#: of exhaustively annotated classes, since "COCO annotated a cup or a wine glass
#: here" is as well defined as either half. That is NOT true of a category COCO
#: lacks entirely, which is what the toy and fuel-tank rulings turn on; running
#: the two together is a mistake this file made for one commit.
#:
#: `cup` ∪ `wine glass` buys +8,180 boxes (+38%), +1,469 images, and **+35% in
#: the small band** -- the binding constraint on class supply everywhere here
#: (#3603). It costs a negative-pool redraw at build time for those 1,469, and
#: it makes `cup` the first class in this study that is not a plain COCO class.
SCALE_CLASS_MERGES: dict[str, tuple[str, ...]] = {
    "cup": ("cup", "wine glass"),
}


def coco_classes_for(cls: str) -> set[str]:
    """The COCO classes whose boxes count as *cls*, merges applied."""
    return set(SCALE_CLASS_MERGES.get(cls, (cls,)))


def scale_class_dataset_name(category: str) -> str:
    """Deprecated alias for :func:`review_name`.

    A thin spelling of :func:`review_name` with no pass suffix, kept because
    ``make_class_slate.py`` bands a *candidate* rather than issuing a voted
    slate and so has no pass to name.

    This used to read a second ``SCALE_CLASS_RULES`` of its own, declared later
    in this module and therefore shadowing the first: #3588 and #3612 each gave
    the same table the same name from opposite ends of one branch, and the
    survivor -- the string one -- left :func:`review_name` reading ``.name`` off
    a ``str``. Both rule sets now live in the one table above.
    """
    return review_name(category)
