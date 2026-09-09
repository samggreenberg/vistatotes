"""The per-step ranker the voting-iterations eval trains, and its primitives.

:class:`StepModel` is the trainer-agnostic contract every trainer in
:mod:`vtscore.eval.step_trainers` returns and every scorer in
:mod:`vtscore.eval.voting_iterations` consumes: a ``predict`` callable plus the
provenance the result rows record.  The rest of this module is the handful of
primitives that sit on both sides of that contract - the head sentinels and the
two functions that turn a media into a training vector or a simulation set into
scores.  The Inclusion cost weights are *not* here: there is one eval-tier
spelling of them, :func:`vtscore.eval.calibration_metrics.inclusion_weights`,
over the shipped :func:`vtscore.training.thresholds.inclusion_cost_weights`.

They live here rather than in the harness module because the trainers and the
loop both need them, and a module that is *only* the contract keeps that
dependency one-directional: ``step_trainers`` and ``voting_iterations`` both
import from here, and nothing here imports either of them.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    import numpy as np

from vtscore.embedding.media_vectors import media_embedding
from vtscore.eval.labels import region_box_for_category
from vtscore.training.mlp import LINEAR_HEAD, LINEAR_SVM_HEAD, _auto_hidden_dim


@dataclass
class StepModel:
    """A trained per-step ranker plus the metadata the eval loop records.

    ``predict`` maps an ``(N, D)`` numpy embedding matrix to per-row
    ``P(positive)`` scores in ``[0, 1]`` — the trainer-agnostic scoring contract
    (identical to :data:`vtscore.eval.sweep_trainers.PredictFn`).  ``torch_model``
    is set only on the app-pipeline path (:data:`APP_TRAINER`), where
    region-aware datasets need the raw torch module to max-pool over patch
    regions; it is ``None`` for the standalone-SVM path (which the experiment
    only ever runs on single-vector, region-free datasets).
    ``backend``/``device`` are recorded on every result row so the report can
    say which engine produced each number.
    """

    predict: Callable[[Any], "np.ndarray"]
    torch_model: Optional[Any]
    backend: str
    device: str


#: Head choices for the app-pipeline trainer (:data:`APP_TRAINER`), all three
#: reached through the same ``hidden_dim`` sentinel production threads.  This is
#: *which classifier head the shipped pipeline fits*, a different question from
#: the ``trainer`` knob beside it, which selects the pipeline itself.  ``"linear_svm"`` is the
#: head the live detector trains (:data:`~vtscore.training.mlp.LINEAR_SVM_HEAD`,
#: a single ``Linear(d, 1)`` fitted to the maximum-margin boundary), so a
#: ``"linear_svm"`` run measures the shipped detector.  ``"linear"`` is the same
#: architecture fitted with balanced BCE — logistic regression, the head shipped
#: between #2790/#2809 and the SVM switch — and ``"mlp"`` is the older harness
#: candidate, a hidden layer auto-sized from the vote count
#: (:func:`~vtscore.training.mlp._auto_hidden_dim`).  The choice is threaded into
#: the calibration folds too, exactly as production threads one sentinel through
#: ``_train_and_score_xy``.
HEADS: tuple[str, ...] = ("mlp", "linear", "linear_svm")


#: The voting simulation's ``trainer`` value naming the **app's own pipeline**:
#: :func:`vtscore.eval.step_trainers._app_train_and_calibrate` (or its style-aware
#: sibling), i.e. :func:`vtscore.training.mlp.train_model` plus production fold
#: calibration.  Which *head* that pipeline fits is a separate knob (``head=``,
#: see :data:`HEADS`), defaulting to :data:`PRODUCTION_HEAD`.
#:
#: It was spelled ``"mlp"`` until issue #3764, which was doubly misleading: the
#: arm trains no MLP by default (its head is the linear SVM), and the *other*
#: eval registry, :data:`vtscore.eval.sweep_trainers.SWEEP_TRAINERS`, uses the
#: same string for an arm that genuinely is one.  The old spelling is still
#: accepted as an input alias (:func:`resolve_trainer_name`) so archived launch
#: scripts keep running; result rows record the new name.
APP_TRAINER: str = "app"

#: Pre-#3764 spelling of :data:`APP_TRAINER`, accepted on input only.
LEGACY_APP_TRAINER: str = "mlp"


def resolve_trainer_name(trainer: str) -> str:
    """Normalise a voting-simulation ``trainer`` value.

    Maps the retired ``"mlp"`` spelling onto :data:`APP_TRAINER` and passes
    every other name (the ``svm_*`` standalone estimators) through untouched, so
    exactly one spelling reaches the dispatch, the guards, and the result rows.
    """
    return APP_TRAINER if trainer == LEGACY_APP_TRAINER else trainer


#: The head the **app** trains, and therefore the harness's default arm:
#: ``vtscore.detectors.training.train_and_threshold`` pins ``hidden_dim =
#: LINEAR_SVM_HEAD`` on every production fit.  ``head=None`` resolves to this,
#: the way ``style=None`` and ``blend_schedule=None`` resolve to the app's
#: geometry and blend schedule — an eval default that isn't the app default
#: measures a detector nobody ships (see the "Eval Default Arm IS the App"
#: rule).  If the shipped head ever changes, move this with it:
#: ``test_harness_linear_head`` pins the two against each other by training the
#: real app pipeline, so the suite fails rather than letting the default arm
#: drift silently.
PRODUCTION_HEAD: str = "linear_svm"


def resolve_hidden_dim(head: str, n_votes: int) -> int:
    """``hidden_dim`` sentinel for *head* at *n_votes* votes.

    The two linear heads return their sentinels (they have no width to size);
    only ``"mlp"`` consults the vote count.
    """
    if head == "linear_svm":
        return LINEAR_SVM_HEAD
    if head == "linear":
        return LINEAR_HEAD
    if head == "mlp":
        return _auto_hidden_dim(n_votes)
    raise ValueError(f"unknown head {head!r}; expected one of {HEADS}")


def good_training_vec(
    media: dict[str, Any],
    target_category: str,
    region_voting: bool,
) -> np.ndarray:
    """Return the training vector for one Good vote on *media*.

    With *region_voting* the simulated user drags the ground-truth box around
    the object: when *media* carries a stored ``patch_grid`` and an annotated
    region for *target_category*, the box is pooled on-the-fly via
    :func:`vtscore.detectors.training.pool_box_from_media` (the same path the
    live region-vote flow uses).  Falls back to the whole-image embedding when
    region voting is off, the media has no patch grid (single-vector
    embedders), or no box is annotated for this category - exactly an
    image-level Good vote.
    """
    if region_voting:
        from vtscore.detectors.training import pool_box_from_media  # noqa: PLC0415

        pooled = pool_box_from_media(media, region_box_for_category(media, target_category))
        if pooled is not None:
            return pooled
    return media_embedding(media)


def score_sim_set_with_model(
    model: Any,
    region_aware: bool,
    sim_clips: dict[int, dict[str, Any]] | None,
    X_all_clips: Any,
    sim_ids: list[int],
    style_obj: Any = None,
) -> tuple[list[int], list[float]]:
    """``(ids, scores)`` for the simulation set under an arbitrary torch *model*.

    The same scorer the test set uses, so the population estimator sees the
    distribution the threshold will actually cut: through the detection style
    when one is given, else region max-pool on a patch dataset, else the
    pre-computed whole-image matrix *X_all_clips* (stacked over
    ``sorted(sim_ids)`` - that ordering is preserved).
    """
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    if style_obj is not None:
        assert sim_clips is not None
        score_map = style_obj.score_media(model, sim_clips)
        ids = list(score_map.keys())
        return ids, [float(score_map[cid]) for cid in ids]
    if region_aware:
        from vtscore.detectors.training import score_media_with_model  # noqa: PLC0415

        assert sim_clips is not None
        scored = score_media_with_model(model, sim_clips)
        return [int(r["id"]) for r in scored], [float(r["score"]) for r in scored]
    with torch.no_grad():
        t = torch.tensor(np.asarray(X_all_clips), dtype=torch.float32).to(next(model.parameters()).device)
        scores = torch.sigmoid(model(t)).squeeze(1).cpu().numpy()
    return sorted(sim_ids), [float(s) for s in scores]
