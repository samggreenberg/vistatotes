"""Canonical registry of long-running task families and their ordered steps.

Every task that drives a progress bar with a ``step``/``total_steps`` structure
registers here. The registry is the shared vocabulary between three parties that
would otherwise drift apart:

- the **task code**, which paces its bar with a weight vector one entry per
  tracker step;
- the **recorder** (:mod:`vtscore.timing.recorder`), which needs to label a
  measured duration with the name of the step it belongs to;
- the **tuning script**, which fits ``a + b · n`` per step and writes the
  profile JSON keyed by these same names.

Adding a new long-running task means adding a :class:`TaskSpec` here, then
calling :func:`vtscore.timing.step_weights` at the task's entry point instead of
writing a literal vector.

**Phases vs tracker steps.** Usually they are the same thing and
``step_index`` is just ``(1, 2, 3, …)``. A task may model a step as several
cost *phases* that scale differently — ``dataset_load``'s step 1 covers both the
network transfer (scales with archive bytes) and the archive unpack (scales with
archive bytes at a very different rate) — in which case several phases share one
tracker step and :func:`vtscore.timing.step_weights` sums their terms back into
that step's slot.

**Default terms** reproduce the hand-tuned vectors these tasks shipped with
before the profile existed, so an instance with no ``VTSEARCH_TIMING_PROFILE``
paces exactly as it did. They are *pseudo-seconds*: only their ratios are
meaningful, because nobody measured them. A profile replaces them with real
seconds, which is what makes the ETA stop drifting. One vector is no longer a
transcription — ``dataset_stage``'s was re-derived from measured rows once its
step boundary was corrected (#3593); its comment below says from which.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    """Declares one long-running task family's step structure.

    Attributes:
        name: Stable identifier used as the profile JSON's task key, in recorded
            JSONL rows, and in ``--tasks`` selections. Never rename one of these
            without migrating the profiles admins have already generated.
        steps: Ordered cost-phase names. Profile coefficients are keyed by these.
        step_index: 1-based tracker step each phase reports against, parallel to
            ``steps``. Several phases may share one step (see module docstring).
        tracker_steps: How many step numbers the task reports — the length of
            the weight vector ``set_step_weights`` expects.
        scale: Human description of what the ``n`` scale variable counts, for
            the tuning script's ``--help`` and the profile's self-documentation.
        default_terms: Shipped fallback pseudo-seconds, parallel to ``steps``.
            Empty means "this task has its own richer default model" — only
            ``dataset_load``, whose calibrated table lives in
            :mod:`vtscore.datasets.stages._load_cost_model`.
        byte_scaled: Steps whose cost tracks downloaded **bytes** rather than
            item count. The tuning script fits these as a per-MB rate instead of
            regressing them against ``n``, because a 2 GB archive of 500 videos
            and a 20 MB archive of 500 texts take wildly different times to
            fetch for reasons ``n`` cannot see.
        loads_encoder: Whether a run of this task can pay a **cold encoder
            load** — the first time a process needs a given ``(media_type,
            embedder)`` it downloads/instantiates the model, and every later run
            finds it resident and pays nothing. Only tasks that declare this
            participate in the recorder's residency ledger, because the ledger's
            key is shared process-wide: a ``dataset_open`` that never touches an
            encoder must not claim ``(image, siglip)`` and leave the genuinely
            cold ``text_sort`` behind it stamped warm. Default ``False`` — the
            safe direction, since an unmarked task simply fits as it does today.
    """

    name: str
    steps: tuple[str, ...]
    step_index: tuple[int, ...]
    tracker_steps: int
    scale: str
    default_terms: tuple[float, ...] = ()
    byte_scaled: tuple[str, ...] = ()
    loads_encoder: bool = False

    def __post_init__(self) -> None:
        if len(self.step_index) != len(self.steps):
            raise ValueError(f"{self.name}: step_index must be parallel to steps")
        if self.default_terms and len(self.default_terms) != len(self.steps):
            raise ValueError(f"{self.name}: default_terms must be parallel to steps")


def _linear(
    name: str,
    steps: tuple[str, ...],
    scale: str,
    terms: tuple[float, ...],
    *,
    loads_encoder: bool = False,
) -> TaskSpec:
    """Build a spec whose phases map 1:1 onto tracker steps (the common case)."""
    return TaskSpec(
        name=name,
        steps=steps,
        step_index=tuple(range(1, len(steps) + 1)),
        tracker_steps=len(steps),
        scale=scale,
        default_terms=terms,
        loads_encoder=loads_encoder,
    )


#: Every registered task family, keyed by :attr:`TaskSpec.name`.
#:
#: The default terms below are transcribed from the literal vectors these tasks
#: carried before the timing profile existed; each site's original reasoning is
#: preserved in the comments here rather than in six scattered constants.
TASKS: dict[str, TaskSpec] = {
    # Importing a dataset: acquire the source, read/convert it into medias,
    # embed every item, then dedup + coverage-atlas + registry save. Deliberately
    # carries no default terms — its shipped model is the measured affine table
    # in ``_load_cost_model``, which is already ``n``-aware per (device, media,
    # embedder) and far better than any flat vector could be here.
    "dataset_load": TaskSpec(
        name="dataset_load",
        steps=("download", "extract", "load", "embed", "finalize"),
        step_index=(1, 1, 2, 3, 4),
        tracker_steps=4,
        scale="media items embedded",
        byte_scaled=("download", "extract"),
        loads_encoder=True,
    ),
    # Re-opening an already-imported dataset from its pkl. Step 1 (pickle read +
    # convert + the near-instant exact-dedup) is seconds at most; step 2 is the
    # coverage atlas, ~10 ms when the cached atlas restores and a hierarchical
    # k-means rebuild when it does not.
    #
    # That rebuild is *seconds*, not minutes, at every size anybody has swept.
    # Driven cold on a V100 with cuML active it fits 0.0026 s/item (r^2 0.95)
    # over n = 412..2954 -- 0.98 s to 7.7 s. The same fit reaches ~26 s at
    # n = 10 000 and ~131 s (2.2 min) only near COVERAGE_ATLAS_AUTO_THRESHOLD
    # (50 000), so "minutes" is true near the threshold and off by two orders
    # of magnitude below ~3000 items. Both of those figures extrapolate a fit
    # whose largest point is 2954, across a 17x gap nothing has measured, and
    # hierarchical k-means need not be linear there
    # (docs/experiments/2026-09-03-drive-cold-3521/REPORT.md section 2, #3595).
    #
    # The 0.85 below is a direction, not a measurement: the rebuild does
    # dominate step 1 whenever it runs, but the weight was never fitted and is
    # roughly 100x too generous at the small end. Re-deriving it waits on a
    # sweep past 2954 (#3595) -- and no single weight can pace both branches
    # anyway, since the restore is ~700x cheaper at n = 2954 (#3594).
    "dataset_open": _linear(
        "dataset_open",
        ("items", "coverage"),
        "media items in the pkl",
        (0.15, 0.85),
    ),
    # Promoting a staged subset into a real dataset. The atlas's hierarchical
    # k-means dominates, then embedding serialization; the registry write is
    # trivial.
    "dataset_promote": _linear(
        "dataset_promote",
        ("coverage", "serialize", "registry"),
        "media items in the promoted subset",
        (6.0, 3.5, 0.5),
    ),
    # Staging an import: run the importer, embed what it produced, serialize the
    # result to a staging pkl. Deliberately *not* modelled as a ``dataset_load``:
    # staging stops before dedup, the coverage atlas, and the registry write (a
    # later promote pays those), so folding its runs into the load fit would
    # teach that finalize is free. No byte-scaled phases here — the staging path
    # is never told an archive size, so a per-MB rate would have nothing to
    # divide by.
    #
    # These are the one set of default terms not transcribed from a pre-profile
    # hand-tuned vector. The shipped ``(0.30, 0.60, 0.10)`` budgeted 60 % of the
    # bar to a step that measured 0.000–0.002 s on every run ever recorded,
    # because the importer's embedding was landing under ``acquire``
    # (``_STAGE_STATUS_TO_STEP`` in ``vtscore/datasets/load_pipeline.py`` is the
    # fix). Re-derived from #3521 §5's image rows, reading the old ``acquire``
    # slope as the embed it actually was: embed ``0.0136 s/item``, serialize
    # ``~0.0042 s/item`` — 76:24, which is the 0.72:0.23 below. Acquire measured
    # near zero there (a demo's acquisition is a cached local read, and its fresh
    # rows leave no residual once the embed line is subtracted); it gets 0.05
    # rather than 0.02 because the only importer these rows cover is the demo
    # one, and a server-folder or upload import spends real I/O in that step.
    "dataset_stage": _linear(
        "dataset_stage",
        ("acquire", "embed", "serialize"),
        "media items staged",
        (0.05, 0.72, 0.23),
        loads_encoder=True,
    ),
    # Loading a saved detector: read its labelset, pull the label examples back
    # into the active dataset, retrain the MLP. Training dominates; the other
    # two are quick I/O.
    "detector_load": _linear(
        "detector_load",
        ("restore_labels", "seed_examples", "train"),
        "labels in the detector's labelset",
        (0.15, 0.15, 0.70),
        loads_encoder=True,
    ),
    # Text search: load the embedder, embed the one-line query, score every
    # media by cosine similarity. The model load dominates on a cold start
    # (seconds to pull CLAP / SigLIP weights); embedding one short query is
    # trivial; scoring scales with the dataset but is vectorised.
    "text_sort": _linear(
        "text_sort",
        ("load_model", "embed_query", "score"),
        "medias scored",
        (0.75, 0.05, 0.20),
        loads_encoder=True,
    ),
    # Running saved detectors across saved datasets. Scoring dominates; loading
    # datasets from pkl is moderate; preparing detector configs is quick.
    "find": _linear(
        "find",
        ("prepare", "load", "score"),
        "medias scored across all selected datasets",
        (0.10, 0.30, 0.60),
        loads_encoder=True,
    ),
    # Train-and-score against the active dataset: resolve the detector, train
    # its MLP, score every media, apply the resulting labels. Train + score
    # carry the cost; resolve/apply are quick.
    "train_and_score": _linear(
        "train_and_score",
        ("resolve", "train", "score", "apply"),
        "medias scored",
        (0.10, 0.45, 0.40, 0.05),
        loads_encoder=True,
    ),
}


#: Branch names meaning "this run took the step's **cheap** path" — a cached
#: artefact stood in for work a first run has to do. They are recorded by
#: :func:`vtscore.timing.note_branch` at the site that makes the decision, and
#: read by the fitter, which will not price a step from cheap runs alone.
#:
#: ``cached``    a demo import satisfied itself from the embeddings pkl, so it
#:               downloaded nothing, embedded nothing, and loaded no encoder.
#: ``restored``  a dataset open adopted the coverage atlas cached in its pickle
#:               instead of rebuilding the hierarchical k-means.
#: ``deferred``  a dataset open past ``COVERAGE_ATLAS_AUTO_THRESHOLD`` skipped
#:               the atlas entirely, leaving it to the on-demand endpoint.
CHEAP_BRANCHES = frozenset({"cached", "restored", "deferred"})

#: Branch names meaning "this run did the work" — the branch somebody waits on,
#: and the only population a step's coefficients may be fitted from.
#:
#: ``fresh``     the import really downloaded, embedded, and finalised.
#: ``rebuilt``   the coverage atlas was built from scratch.
DEAR_BRANCHES = frozenset({"fresh", "rebuilt"})


def task_spec(task: str) -> TaskSpec | None:
    """Return the :class:`TaskSpec` for *task*, or ``None`` if unregistered.

    Unregistered is not an error: a caller that passes an unknown task simply
    gets no profile-derived weights and keeps whatever fallback it supplied.
    """
    return TASKS.get(task)
