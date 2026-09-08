"""Schemas for the eval / labeling-progress API.

Covers four routes in ``vtsearch/routes/eval.py``:

* ``POST /api/labeling-progress``                -> :class:`LabelingProgressResponseSchema`
* ``GET  /api/labeling-status``                  -> :class:`LabelingStatusResponseSchema`
* ``GET  /api/indicator-score-history``          -> :class:`IndicatorScoreHistoryQuerySchema`
                                                    -> :class:`IndicatorScoreHistoryResponseSchema`
* ``POST /api/eval/train-and-score``             -> :class:`EvalTrainAndScoreRequestSchema`
                                                    -> :class:`EvalTrainAndScoreResponseSchema`
* ``GET  /api/eval/train-and-score/result``      -> :class:`EvalTrainAndScoreResultQuerySchema`
                                                    -> :class:`EvalTrainAndScoreResponseSchema`

The per-step ``error_cost`` / ``stability`` / ``diversity`` lists are
enumerated as nested schemas (:class:`ErrorCostPointSchema`,
:class:`StabilityPointSchema`, :class:`DiversityPointSchema`).  Each shape is
built in exactly one place inside ``vtscore.detectors.labeling_progress``, so
the key sets are fixed and the frontend charts can consume the generated
models instead of a hand-written mirror.

``GET /api/indicator-score-history`` is the one exception: its ``history``
array holds *whichever* of the three point shapes matches the requested
``metric``, so it stays an opaque list of dicts rather than committing to one
of them.  Expressing that as an OpenAPI ``oneOf`` would mean hand-writing
component ``$ref`` strings in field metadata, which silently rots the moment
a schema class is renamed; the single client-side cast is the cheaper trade.

The two train-and-score endpoints share a single response schema that
declares every metric-specific data key (``error_cost`` / ``stability`` /
``diversity``) and fills in whichever one the requested metric produced;
its ``unknown = "include"`` only widens what the schema accepts on load.
"""

from __future__ import annotations

from marshmallow import Schema, fields, validate

from vtsearch.schemas.common import PluginExtrasSchema


_METRIC_VALIDATOR = validate.OneOf(["smart", "stable", "diverse"])


# ---------------------------------------------------------------------------
# /api/labeling-progress
# ---------------------------------------------------------------------------


class ErrorCostPointSchema(Schema):
    """One step of the error-cost series (``_eval_cached_models``).

    One point per label step the app **trained a detector for**, not per label
    step: a step whose label set no learned sort ran against carries no model
    and is absent from the series, so ``num_labels`` is not contiguous across
    points (issue #3757)."""

    num_labels = fields.Integer(required=True)
    error_cost = fields.Float(
        required=True,
        metadata={"description": "``fpr_weight * fpr + fnr_weight * fnr`` for the model trained at this step."},
    )
    time_index = fields.Integer(metadata={"description": "Zero-based index of this step in the label history."})
    fpr = fields.Float(metadata={"description": "False-positive rate on the held-out eval set."})
    fnr = fields.Float(metadata={"description": "False-negative rate on the held-out eval set."})


class StabilityPointSchema(Schema):
    """One step of the prediction-stability series.

    Absent for the first step, which has no prior predictions to compare
    against.
    """

    num_labels = fields.Integer(required=True)
    num_flips = fields.Integer(
        required=True,
        metadata={
            "description": (
                "How many still-unlabeled items changed predicted class between the previous step's model and this one."
            )
        },
    )
    time_index = fields.Integer(metadata={"description": "Zero-based index of this step in the label history."})
    num_unlabeled = fields.Integer(
        metadata={"description": "Size of the monitored pool the flip count was measured over."}
    )


class DiversityPointSchema(Schema):
    """One step of the coverage-diversity series (from the coverage atlas)."""

    num_labels = fields.Integer(required=True)
    diversity_level = fields.Float(
        required=True, metadata={"description": "Coverage level reached by the labels so far."}
    )
    depth = fields.Integer(required=True, metadata={"description": "Total nodes in the coverage atlas."})


class LabelingProgressResponseSchema(Schema):
    """Response for ``POST /api/labeling-progress``."""

    error_cost_over_time = fields.List(fields.Nested(ErrorCostPointSchema), required=True)
    stability_over_time = fields.List(fields.Nested(StabilityPointSchema), required=True)
    diversity_level_over_time = fields.List(fields.Nested(DiversityPointSchema), required=True)
    total_labels = fields.Integer(required=True)
    total_medias = fields.Integer(required=True)


# ---------------------------------------------------------------------------
# /api/labeling-status
# ---------------------------------------------------------------------------


class StatusIndicatorSchema(PluginExtrasSchema):
    """One ``smart`` / ``stable`` / ``span`` indicator in the labeling-status
    response.  ``status`` is the red/yellow/green flag every indicator emits;
    ``reason`` is the human-readable explanation.  Metric-specific keys
    (``cost``, ``flips``, ``diversity_level``, ``avg_flip_rate``, …) flow
    through unchanged via
    :class:`~vtsearch.schemas.common.PluginExtrasSchema`; the
    :mod:`vtscore.detectors.labeling_progress` analyzer remains the
    source of truth for that shape."""

    status = fields.String(required=True)
    reason = fields.String()


class LabelingStatusResponseSchema(Schema):
    """Response for ``GET /api/labeling-status``."""

    good_count = fields.Integer(required=True)
    bad_count = fields.Integer(required=True)
    total_count = fields.Integer(required=True)
    smart = fields.Nested(StatusIndicatorSchema, required=True)
    stable = fields.Nested(StatusIndicatorSchema, required=True)
    span = fields.Nested(StatusIndicatorSchema, required=True)
    # ``true`` while a background worker is advancing the per-step cache: the
    # ``smart`` / ``stable`` indicators reflect the last-computed snapshot (or a
    # transient "computing" placeholder), not the current labelset yet. Cleared
    # on the first poll after the refresh lands. Optional so existing clients /
    # mocks that omit it stay valid. See issue #2397.
    stale = fields.Boolean()


# ---------------------------------------------------------------------------
# /api/indicator-score-history
# ---------------------------------------------------------------------------


class IndicatorScoreHistoryQuerySchema(Schema):
    """Query for ``GET /api/indicator-score-history``."""

    metric = fields.String(
        required=True,
        validate=_METRIC_VALIDATOR,
        metadata={"description": "Which metric history to return: ``smart``, ``stable``, or ``diverse``."},
    )

    class Meta:
        # Tolerate unrelated query params (e.g. dataset/detector ids
        # added by the request-context middleware on some clients).
        unknown = "exclude"


class IndicatorScoreHistoryResponseSchema(Schema):
    """Response for ``GET /api/indicator-score-history``."""

    metric = fields.String(required=True, validate=_METRIC_VALIDATOR)
    #: Whichever of ``ErrorCostPoint`` / ``StabilityPoint`` / ``DiversityPoint``
    #: matches ``metric``; kept opaque here (see the module docstring).
    history = fields.List(fields.Dict(), required=True)
    # ``false`` when the per-step cache does not yet cover the whole label
    # history, in which case ``history`` is empty and the client should fall
    # back to the async ``POST /api/eval/train-and-score`` job.  The route
    # never advances the cache itself; see its docstring.
    complete = fields.Boolean(required=True)


# ---------------------------------------------------------------------------
# /api/eval/train-and-score (start) and /result (poll)
# ---------------------------------------------------------------------------


class EvalTrainAndScoreRequestSchema(Schema):
    """Body for ``POST /api/eval/train-and-score``."""

    metric = fields.String(required=True, validate=_METRIC_VALIDATOR)
    wait = fields.Boolean(
        load_default=False,
        metadata={
            "description": (
                "If true, block until the job completes and return the metric data inline. "
                "Used by tests; production clients poll ``/result`` instead."
            )
        },
    )


class EvalTrainAndScoreResultQuerySchema(Schema):
    """Query for ``GET /api/eval/train-and-score/result``."""

    job_id = fields.String(required=True)

    class Meta:
        unknown = "exclude"


class EvalTrainAndScoreResponseSchema(Schema):
    """Combined response for the start + poll train-and-score routes.

    The response shape varies with status (``running`` vs ``done`` vs
    ``error`` vs ``cancelled``) and metric (``error_cost`` vs
    ``stability`` vs ``diversity`` data keys). Declared as a permissive
    schema so the metric-specific data key flows through unchanged.
    """

    job_id = fields.String(required=True)
    status = fields.String(
        required=True,
        validate=validate.OneOf(["running", "done", "error", "cancelled", "missing"]),
    )
    metric = fields.String()
    current = fields.Integer()
    total = fields.Integer()
    error_cost = fields.List(fields.Nested(ErrorCostPointSchema))
    stability = fields.List(fields.Nested(StabilityPointSchema))
    diversity = fields.List(fields.Nested(DiversityPointSchema))
    error = fields.String()

    class Meta:
        # A future metric's data key is tolerated on ``load`` rather than
        # raising.  Load-only: ``dump`` emits the declared fields and
        # nothing else, so a new metric still needs a field declared here
        # (or this schema needs
        # :class:`~vtsearch.schemas.common.PluginExtrasSchema`) before its
        # data reaches the client.
        unknown = "include"


class EvalTrainAndScoreCancelResponseSchema(Schema):
    """Response for ``POST /api/eval/train-and-score/cancel/<job_id>``."""

    ok = fields.Boolean(required=True)


__all__ = [
    "EvalTrainAndScoreCancelResponseSchema",
    "EvalTrainAndScoreRequestSchema",
    "EvalTrainAndScoreResponseSchema",
    "EvalTrainAndScoreResultQuerySchema",
    "IndicatorScoreHistoryQuerySchema",
    "IndicatorScoreHistoryResponseSchema",
    "LabelingProgressResponseSchema",
    "LabelingStatusResponseSchema",
    "StatusIndicatorSchema",
]
