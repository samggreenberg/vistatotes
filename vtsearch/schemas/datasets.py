"""Schemas for dataset-management routes.

These cover the read-only / display-oriented dataset blueprints: listings
(media types, embedders, clippers, converters, importers), status / cancel,
and UI helpers (demo dataset list, file browser, dashboard). The heavier
modules (``load``, ``staging``, ``registry``) involve multipart upload,
binary streaming, and plugin-field-shaped bodies and are migrated
separately.

Every ``to_dict()`` payload here is enumerated as a nested schema so the
generated OpenAPI client carries a real typed model and the frontend can
delete its hand-written mirror.  Two flavours, depending on whether the
plugin family can add keys of its own:

* **Media types, embedders, converters, importers, picker tabs, registry
  entries** have a *fixed* key set (no subclass overrides ``to_dict``
  beyond the documented base), so their schemas are strict: an undeclared
  key is a bug, and dropping it on dump is the correct signal.
* **Clippers and cleaners** genuinely vary — concrete clippers append their
  own keys (``duration``, ``top_db``, ``box``, …) on top of the fixed base.
  Their schemas enumerate the base and inherit
  :class:`~vtsearch.schemas.common.PluginExtrasSchema`, so the fixed fields
  get real types while the plugin extras still reach the client verbatim.
  The generated model picks up an index signature from
  ``additionalProperties: true``, which is exactly the shape the frontend
  used to hand-maintain.
"""

from __future__ import annotations

from marshmallow import Schema, fields, validate

from vtsearch.schemas.common import PluginExtrasSchema, list_of_strings
from vtsearch.schemas.file_browser import BrowseDirectoryEntrySchema, BrowseFileEntrySchema

#: Upper bound on user-supplied dataset names, mirroring
#: ``vtsearch.schemas.detectors.MAX_NAME_LENGTH``.  A name past this is already
#: unusable for display, and capping it here stops an absurdly long name from
#: ever reaching a filesystem path (and the uncaught ``OSError`` /
#: absolute-path leak that would follow).
MAX_NAME_LENGTH = 128


# ---------------------------------------------------------------------------
# /api/media-types, /api/embedders, /api/clippers, /api/converters
# (read-only listings)
# ---------------------------------------------------------------------------


class _MediaTypeFilterQuerySchema(Schema):
    """Query string shared by ``GET /api/embedders`` and ``/api/clippers``."""

    media_type = fields.String(
        load_default="",
        metadata={
            "description": (
                "Optional ``type_id`` (e.g. ``image``) or "
                "``folder_import_name`` (e.g. ``images``). When provided, "
                "only matching entries are returned."
            )
        },
    )


class EmbeddersListQuerySchema(_MediaTypeFilterQuerySchema):
    """Query for ``GET /api/embedders``."""


class ClippersListQuerySchema(_MediaTypeFilterQuerySchema):
    """Query for ``GET /api/clippers``."""


class CleanersListQuerySchema(_MediaTypeFilterQuerySchema):
    """Query for ``GET /api/cleaners``."""


class ConvertersListQuerySchema(Schema):
    """Query for ``GET /api/converters``.

    ``target`` and ``source`` are mutually exclusive filters; if both are
    supplied the handler prefers ``target``.
    """

    target = fields.String(load_default="")
    source = fields.String(load_default="")


class MediaTypeInfoSchema(Schema):
    """One ``MediaType.to_dict()`` payload (see ``vtscore/media/base.py``).

    Fixed shape: every media type emits the same keys, so this is a real
    nested schema rather than an opaque ``fields.Dict()``.  Only ``type_id``
    and ``name`` are guaranteed present enough to mark ``required``; the rest
    mirror the frontend's optional fields.
    """

    type_id = fields.String(required=True)
    name = fields.String(required=True)
    icon = fields.String()
    folder_import_name = fields.String()
    loops = fields.Boolean(
        metadata={"description": "Whether this media type's rendered form loops (e.g. short video/audio)."}
    )
    file_extensions = fields.List(
        fields.String(),
        metadata={"description": 'Glob patterns for files this media type claims, e.g. ``["*.jpg", "*.png"]``.'},
    )
    has_thumbnail = fields.Boolean(
        metadata={
            "description": (
                "Whether items of this type have a browsable thumbnail (image/video/document, and audio via "
                "its waveform PNG). Drives the VTSBrowse square-vs-hex bin shape and thumbnail painting."
            )
        }
    )
    importable = fields.Boolean(
        metadata={
            "description": (
                "Whether this type is a first-class ingestion category the user picks when importing (folder "
                "scan, file upload). ``false`` for a convert-in half type like ``face``."
            )
        }
    )
    embeddable = fields.Boolean(
        metadata={
            "description": (
                "Whether this type can be embedded (and therefore sorted / browsed / text-queried) on its own. "
                "``false`` for a convert-out half type like ``document`` that must be converted first."
            )
        }
    )
    converts_to = fields.List(
        fields.String(),
        metadata={
            "description": (
                "Embeddable target type_ids a non-embeddable type can convert into (first = default). "
                '``["image", "text"]`` for ``document``; empty for a directly-embeddable type.'
            )
        },
    )


class EmbedderInfoSchema(Schema):
    """One ``MediaEmbedder.to_dict()`` payload (see ``vtscore/media/embedder.py``).

    Fixed shape across all embedders (no subclass overrides ``to_dict``), so
    the payload is fully enumerated here instead of ``fields.Dict()``.
    """

    name = fields.String(required=True)
    display_name = fields.String(
        metadata={
            "description": (
                'Human-readable label shown in pickers, e.g. ``"SigLIP (general images)"``. Falls back to '
                "``name`` for legacy embedders that don't supply a friendlier label."
            )
        }
    )
    model_id = fields.String(
        allow_none=True,
        metadata={
            "description": (
                "Concrete pretrained-model identifier the embedder loads - usually a HuggingFace repo id "
                "(or a direct weights URL). ``null`` for embedders with no single downloadable model id "
                "(e.g. the classical SIFT/VLAD structural embedder)."
            )
        },
    )
    media_type_id = fields.String(required=True)
    is_default = fields.Boolean(
        metadata={
            "description": (
                "Whether this embedder is the recommended default for its media type (exactly one per media "
                'type). The dropdown surfaces this entry under a "Recommended" optgroup.'
            )
        }
    )
    supports_text = fields.Boolean(
        metadata={
            "description": (
                "Whether this embedder can embed text queries into the same vector space as its media. "
                "``false`` for vision-only encoders (DINOv3, Perception Encoder)."
            )
        }
    )
    supports_patch_regions = fields.Boolean(
        metadata={
            "description": (
                "Whether this embedder produces patch-level vectors and a hierarchical region tree per image. "
                "``true`` for patch-based encoders (DINOv2, DINOv3, EUPE)."
            )
        }
    )
    supports_geometric_verification = fields.Boolean(
        metadata={
            "description": (
                "Whether this embedder produces local features (keypoints + descriptors) for instance "
                "matching. ``true`` for structural embedders (SIFT/VLAD); ``false`` for every semantic embedder."
            )
        }
    )
    license_notice = fields.String(
        allow_none=True,
        metadata={
            "description": (
                "User-facing licence warning to show before the user picks this embedder. ``null`` for "
                "embedders with no special licensing constraints. Advisory only."
            )
        },
    )


class ImporterFieldSchema(Schema):
    """One ``PluginField.to_dict()`` payload (see ``vtscore/plugins/__init__.py``).

    Shared by every plugin family that renders a form: dataset importers,
    converters, label importers/exporters.  Fixed shape - ``PluginField`` is
    a dataclass and ``to_dict`` emits all of its wire fields unconditionally.

    Optionality mirrors what the frontend treats as optional rather than what
    the backend always emits (same convention as
    :class:`MediaTypeInfoSchema`): only ``key`` and ``field_type`` are the
    fields a consumer can never do without.
    """

    key = fields.String(required=True)
    field_type = fields.String(
        required=True,
        validate=validate.OneOf(
            ["file", "folder", "url", "text", "password", "email", "number", "select", "server_path", "checkbox"]
        ),
    )
    label = fields.String()
    description = fields.String()
    accept = fields.String(
        metadata={"description": 'For ``file`` fields: comma-separated extensions, e.g. ``".pkl"``.'}
    )
    options = fields.List(
        fields.String(),
        metadata={
            "description": (
                "For ``select`` fields: the statically-declared allowed values. Runtime-computed options come "
                "from ``POST /api/dataset/import/<importer>/options`` instead and carry their own "
                "``(value, label)`` shape; see ``ImporterFieldOptionsResponseSchema``."
            )
        },
    )
    default = fields.String()
    required = fields.Boolean()
    placeholder = fields.String(metadata={"description": "Hint shown as placeholder text inside the input widget."})
    hint = fields.String(
        metadata={
            "description": (
                "Inline format-hint text rendered as a visible chip below the input. Distinct from "
                "``description`` (which feeds the placeholder): the hint stays visible after the user starts "
                "typing, so it is the right place for accepted extensions or a short sample file layout."
            )
        }
    )
    dynamic_options = fields.Boolean(
        metadata={
            "description": (
                "When true, ``options`` is computed at runtime by calling "
                "``POST /api/dataset/import/<importer>/options`` with the current field values. The frontend "
                "re-fetches whenever any field listed in ``depends_on`` changes."
            )
        }
    )
    depends_on = fields.List(
        fields.String(), metadata={"description": "Field keys whose values this field's options depend on."}
    )
    allow_free_text = fields.Boolean(
        metadata={
            "description": (
                "For ``select`` fields: when true, render as a combobox the user can type an arbitrary value "
                "into. When the options refresh, a typed value absent from the new list is kept; a strict "
                "select clears it."
            )
        }
    )
    min = fields.String(metadata={"description": "For ``number`` fields: minimum allowed value (empty = no min)."})
    max = fields.String(metadata={"description": "For ``number`` fields: maximum allowed value (empty = no max)."})
    step = fields.String(
        metadata={"description": 'For ``number`` fields: step increment (empty / ``"any"`` = unconstrained).'}
    )
    clears = fields.List(
        fields.String(),
        metadata={
            "description": (
                "Field keys this field is mutually exclusive with. Entering a non-empty value here blanks each "
                "listed field (and they list this one back), so only one of the set is ever active at a time."
            )
        },
    )
    template_vars = fields.List(
        fields.String(),
        metadata={
            "description": (
                "Framework-substituted template variable names (e.g. ``detector_name``, ``YYYYMMDD``) replaced "
                "in this field's value before the plugin runs. Advisory for clients; substitution is server-side."
            )
        },
    )


class ConverterInfoSchema(Schema):
    """One ``MediaConverter.to_dict()`` payload (see ``vtscore/converters/base.py``).

    Fixed shape across all converters.
    """

    name = fields.String(required=True)
    source_type = fields.String(required=True)
    target_type = fields.String(required=True)
    display_name = fields.String()
    description = fields.String()
    summary_template = fields.String(
        metadata={
            "description": (
                "One-line preview with ``{key}`` placeholders for each field. The native row of the importer "
                "source-specs picker substitutes the current field values. Falls back to ``description`` when empty."
            )
        }
    )
    #: Python attribute renamed to avoid shadowing ``marshmallow.fields``;
    #: mapped back to the ``fields`` wire key.
    field_list = fields.List(fields.Nested(ImporterFieldSchema), attribute="fields", data_key="fields")


class ClipperParameterSchema(Schema):
    """One entry of a clipper's ``parameters`` / ``creation_questions`` list.

    See ``MediaClipper.parameters`` in ``vtscore/media/clipper.py`` for the
    descriptor contract.  ``min`` / ``max`` / ``step`` are only meaningful for
    ``number`` parameters.
    """

    key = fields.String(required=True)
    label = fields.String(required=True)
    description = fields.String()
    type = fields.String(required=True, validate=validate.OneOf(["number", "string"]))
    default = fields.Raw(required=True, metadata={"oneOf": [{"type": "number"}, {"type": "string"}]})
    min = fields.Float()
    max = fields.Float()
    step = fields.Float()


class ClipperInfoSchema(PluginExtrasSchema):
    """One ``MediaClipper.to_dict()`` payload (see ``vtscore/media/clipper.py``).

    The base key set is fixed, but concrete clippers append their own
    resolved parameter values (``duration``, ``top_db``, ``box``, …), so this
    enumerates the base and lets the extras through untouched.  See the
    module docstring for why that beats either an opaque ``fields.Dict()`` or
    a strict schema that would silently drop those keys.
    """

    name = fields.String(required=True)
    media_type = fields.String(required=True)
    display_name = fields.String()
    description = fields.String()
    summary_template = fields.String(
        metadata={
            "description": (
                "One-line preview with ``{key}`` placeholders for each parameter. The native row of the "
                "importer source-specs picker substitutes the current parameter values, so the user sees a live "
                "summary of what the clipper will do. Falls back to ``description`` when empty."
            )
        }
    )
    parameters = fields.List(fields.Nested(ClipperParameterSchema))
    creation_questions = fields.List(
        fields.Nested(ClipperParameterSchema),
        metadata={
            "description": (
                "Parameters to ask about when the user first picks this clipper. Defaults to ``parameters``."
            )
        },
    )


class CleanerInfoSchema(ClipperInfoSchema):
    """One ``MediaCleaner.to_dict()`` payload (see ``vtscore/media/cleaner.py``).

    ``MediaCleaner`` subclasses ``MediaClipper``, so this is the clipper
    descriptor plus the cleaner-only ``default_enabled`` flag.
    """

    default_enabled = fields.Boolean(
        metadata={
            "description": (
                "Whether the import form checks this cleaner by default. True only for cleaners that fix an "
                "outright representation bug (EXIF orientation), where leaving it off ships known-wrong vectors."
            )
        }
    )


class ImporterInfoSchema(Schema):
    """One ``ImporterBase.to_dict()`` payload (see ``vtscore/datasets/importers/base/core.py``).

    Fixed shape: ``PluginBase.to_dict()`` emits the shared plugin metadata
    and ``ImporterBase.to_dict()`` appends ``picker_view`` / ``category`` /
    ``available_converters_by_media_type``.  No concrete importer overrides
    ``to_dict``, so this schema is strict.
    """

    name = fields.String(required=True)
    display_name = fields.String()
    description = fields.String()
    icon = fields.String()
    #: Python attribute renamed to avoid shadowing ``marshmallow.fields``;
    #: mapped back to the ``fields`` wire key.
    field_list = fields.List(fields.Nested(ImporterFieldSchema), attribute="fields", data_key="fields")
    ui_mode = fields.String()
    hidden_from_picker = fields.Boolean()
    picker_view = fields.String(
        metadata={
            "description": (
                "Which view the dataset-importer modal opens for this card: ``form`` (default), ``demo``, "
                "``server_folder``, ``local_folder``, or ``local_files``."
            )
        }
    )
    category = fields.String(
        metadata={
            "description": (
                "Picker tab this importer belongs to. One of ``services``, ``server``, ``local``, ``demo``, "
                'or ``""`` (uncategorised).'
            )
        }
    )
    available_converters_by_media_type = fields.Dict(
        keys=fields.String(),
        values=fields.List(fields.Nested(ConverterInfoSchema)),
        metadata={
            "description": (
                "Map of output media-type id -> converters whose ``target_type`` matches that id. Drives the "
                '"Include rows" UI without an extra round-trip to ``/api/converters``.'
            )
        },
    )
    enabled = fields.Boolean(
        metadata={
            "description": (
                "Only set on ``combine_datasets`` by ``GET /api/dataset/all-importers``: false when fewer than "
                "two saved datasets share a media type, so there is nothing to combine."
            )
        }
    )


class ImporterPickerTabSchema(Schema):
    """One entry of the ``tabs`` array from ``GET /api/dataset/all-importers``.

    Registered by plugins via
    :func:`vtscore.datasets.importers.tabs.register_picker_tab`; ``id``
    matches the importers' ``category``.
    """

    id = fields.String(required=True)
    label = fields.String(required=True)
    icon = fields.String(metadata={"description": "``vt-icon`` type name."})
    order = fields.Integer(metadata={"description": "Lower values render first."})


class MediaTypesListResponseSchema(Schema):
    """Response for ``GET /api/media-types``."""

    media_types = fields.List(fields.Nested(MediaTypeInfoSchema), required=True)


class EmbeddersListResponseSchema(Schema):
    """Response for ``GET /api/embedders``."""

    embedders = fields.List(fields.Nested(EmbedderInfoSchema), required=True)


class ClippersListResponseSchema(Schema):
    """Response for ``GET /api/clippers``."""

    clippers = fields.List(fields.Nested(ClipperInfoSchema), required=True)


class CleanersListResponseSchema(Schema):
    """Response for ``GET /api/cleaners``."""

    cleaners = fields.List(fields.Nested(CleanerInfoSchema), required=True)


class ConvertersListResponseSchema(Schema):
    """Response for ``GET /api/converters``."""

    converters = fields.List(fields.Nested(ConverterInfoSchema), required=True)


class DatasetImportersListResponseSchema(Schema):
    """Response for ``GET /api/dataset/importers``."""

    importers = fields.List(fields.Nested(ImporterInfoSchema), required=True)


class DatasetAllImportersListResponseSchema(Schema):
    """Response for ``GET /api/dataset/all-importers``.

    ``tabs`` is the picker-tab layout; the ``combine_datasets`` importer is
    annotated with an ``enabled`` flag by the handler.
    """

    importers = fields.List(fields.Nested(ImporterInfoSchema), required=True)
    tabs = fields.List(fields.Nested(ImporterPickerTabSchema), required=True)


# ---------------------------------------------------------------------------
# /api/dataset/status, /api/dataset/cancel
# ---------------------------------------------------------------------------


class DatasetStatusResponseSchema(Schema):
    """Response for ``GET /api/dataset/status``."""

    loaded = fields.Boolean(required=True)
    num_medias = fields.Integer(required=True)
    has_votes = fields.Boolean(required=True)
    media_type = fields.String(allow_none=True, required=True)
    display_name = fields.String(required=True)
    num_dupes = fields.Integer(required=True)


class CancelDatasetLoadResponseSchema(Schema):
    """Response for ``POST /api/dataset/cancel`` and ``/cancel/<task_id>``.

    Cancellation is cooperative, so ``ok`` reports whether the flag actually
    reached something that can act on it — not merely that it was set.  The
    lists say which operations did what; see
    :func:`vtscore.concurrency.progress.cancel_dataset_progress`.  A request
    that reached nothing answers ``409`` with ``ok: false``.
    """

    ok = fields.Boolean(required=True)
    message = fields.String(required=True)
    #: Everything that claimed to be working when the cancel arrived.
    targets = fields.List(fields.String(), required=True)
    #: Targets that reached a terminal state within the grace period.
    acknowledged = fields.List(fields.String(), required=True)
    #: Targets still running, whose live worker will observe the flag.
    pending = fields.List(fields.String(), required=True)
    #: Targets whose progress claimed work no live thread was doing. Stale
    #: trackers, now cleared — not operations that were stopped.
    unresponsive = fields.List(fields.String(), required=True)


# ---------------------------------------------------------------------------
# /api/dataset/demo-list, /api/dataset/demo-categories/<name>
# ---------------------------------------------------------------------------


class _DemoDatasetEntrySchema(Schema):
    """One entry in the ``GET /api/dataset/demo-list`` ``datasets`` array."""

    name = fields.String(required=True)
    label = fields.String(required=True)
    status = fields.String(required=True, validate=validate.OneOf(["ready", "needs_embedding", "needs_download"]))
    ready = fields.Boolean(required=True)
    num_files = fields.Integer(required=True)
    download_size_mb = fields.Float(required=True)
    description = fields.String(required=True)
    media_type = fields.String(required=True)
    num_categories = fields.Integer(required=True)
    available_converters = fields.List(fields.Nested(ConverterInfoSchema), required=True)
    pkl_embedder = fields.String(required=True)
    pkl_clipper = fields.String(required=True)


class DemoDatasetListQuerySchema(Schema):
    """Query string for ``GET /api/dataset/demo-list``.

    All three fields are optional cache-key filters: when supplied, a cached
    pkl is only considered ``"ready"`` if it was produced with the same
    embedder / clipper / converter.  ``converter`` names a convert-on-load
    step (e.g. ``document2image`` for the Document demo tab) and only affects
    demos whose media type matches the converter's source type — those are
    cached under the ``{name}__{converter}`` pickle key.
    """

    embedder = fields.String(load_default="")
    clipper = fields.String(load_default="")
    converter = fields.String(load_default="")


class DemoDatasetListResponseSchema(Schema):
    """Response for ``GET /api/dataset/demo-list``."""

    datasets = fields.List(fields.Nested(_DemoDatasetEntrySchema), required=True)


class DemoCategoriesResponseSchema(Schema):
    """Response for ``GET /api/dataset/demo-categories/<name>``."""

    categories = fields.List(fields.String(), required=True)


# ---------------------------------------------------------------------------
# /api/browse-media-files, /api/browse-media-files/select
# ---------------------------------------------------------------------------


class BrowseMediaFilesQuerySchema(Schema):
    """Query for ``GET /api/browse-media-files``."""

    source = fields.String(
        load_default="",
        metadata={"description": "One of ``demo:<name>``, ``folder``, or ``server_fs``."},
    )
    path = fields.String(
        load_default="",
        metadata={"description": "Relative sub-path within the source root."},
    )


class BrowseMediaFilesResponseSchema(Schema):
    """Response for ``GET /api/browse-media-files``.

    The directory- and file-entry shapes are identical to the ones
    used by ``GET /api/browse`` (see ``vtsearch.schemas.file_browser``),
    so reuse those nested schemas; registering distinct
    ``_BrowseDirectoryEntry`` / ``_BrowseFileEntry`` schemas alongside
    the public names made the generated TS client emit duplicate
    identifiers (ng-openapi-gen strips the leading underscore).
    """

    directories = fields.List(fields.Nested(BrowseDirectoryEntrySchema), required=True)
    files = fields.List(fields.Nested(BrowseFileEntrySchema), required=True)
    root_path = fields.String(required=True)
    default_path = fields.String(
        load_default="",
        dump_default="",
        metadata={
            "description": (
                "Suggested initial relative sub-path for this source, for example "
                "the server user's home directory when ``source=server_fs``. Empty "
                "for sources where the root is already the right starting point."
            )
        },
    )


class BrowseMediaFilesSelectRequestSchema(Schema):
    """Body for ``POST /api/browse-media-files/select``."""

    source = fields.String(required=True, validate=validate.Length(min=1))
    path = fields.String(required=True, validate=validate.Length(min=1))


class BrowseMediaFilesSelectResponseSchema(Schema):
    """Response for ``POST /api/browse-media-files/select``."""

    filename = fields.String(required=True)
    original_name = fields.String(required=True)


class DetectMediaTypeQuerySchema(Schema):
    """Query for ``GET /api/dataset/detect-media-type``.

    The ``limit`` field is capped to ``[1, 500]`` by the handler; the
    schema only narrows the type. Invalid integer strings fall back to
    the default rather than rejecting the request, preserving the
    pre-migration permissiveness for this hint endpoint.
    """

    source = fields.String(
        load_default="folder",
        metadata={"description": "One of ``demo:<name>`` or ``folder`` (matches ``/api/browse-media-files``)."},
    )
    path = fields.String(load_default="")
    recursive = fields.Boolean(load_default=True)
    limit = fields.Integer(load_default=50)

    class Meta:
        unknown = "exclude"


class DetectMediaTypeResponseSchema(Schema):
    """Response for ``GET /api/dataset/detect-media-type``.

    Mirrors :func:`vtscore.datasets.media_type_detection.detect_media_types_in_folder`'s
    return value.
    """

    sample_size = fields.Integer(required=True)
    counts_by_type = fields.Dict(keys=fields.String(), values=fields.Integer(), required=True)
    extensions = fields.Dict(keys=fields.String(), values=fields.Integer(), required=True)
    dominant = fields.String(allow_none=True, required=True)
    truncated = fields.Boolean(required=True)


# ---------------------------------------------------------------------------
# /api/dashboard/disk-usage
# ---------------------------------------------------------------------------


class DashboardDiskUsageResponseSchema(Schema):
    """Response for ``GET /api/dashboard/disk-usage``."""

    total = fields.Integer(required=True)
    used = fields.Integer(required=True)
    free = fields.Integer(required=True)
    path = fields.String(required=True)


class DashboardRamUsageResponseSchema(Schema):
    """Response for ``GET /api/dashboard/ram-usage``."""

    total = fields.Integer(required=True)
    used = fields.Integer(required=True)
    free = fields.Integer(required=True)


# ---------------------------------------------------------------------------
# /api/dataset/load-* and /api/dataset/clear (vtsearch/routes/datasets/load.py)
# ---------------------------------------------------------------------------


class DatasetLoadDemoRequestSchema(Schema):
    """Body for ``POST /api/dataset/load-demo``.

    ``name`` must be a key of ``DEMO_DATASETS``; the handler returns 400
    if it isn't (the set isn't known at schema-build time).
    """

    name = fields.String(required=True, validate=validate.Length(min=1))
    embedder = fields.String(load_default="")
    embedders = fields.List(
        fields.String(),
        load_default=None,
        metadata={
            "description": (
                "v3 trio of create-time embedder picks (text / patch / structural). "
                "When set, every name is embedded so a multi-embedder dataset is produced; "
                "omitted falls back to the single `embedder`."
            )
        },
    )
    clipper = fields.String(load_default="")
    clipper_params = fields.Dict(
        load_default=None,
        metadata={
            "description": (
                'Optional parameter overrides for `clipper` (e.g. `{"duration": 5.0}`). '
                "Only applied when `clipper` names a real, non-default clipper."
            )
        },
    )
    cleaners = fields.List(
        fields.Dict(),
        load_default=None,
        metadata={
            "description": (
                "Cleanup gates to run on each finished unit before embedding, as "
                '`[{"name": "image_exif_orient", "params": {}}]`. Order is ignored; '
                "cleaners always run last, after the clipper / converter chain."
            )
        },
    )
    converter = fields.String(load_default="")
    dataset_name = fields.String(load_default="")
    build_projection = fields.String(
        load_default="false",
        metadata={"description": "When 'true', compute + persist the 2-D Browse projection at ingest."},
    )
    merge_near_duplicates = fields.String(
        load_default="false",
        metadata={"description": "When 'true', collapse near-duplicate media into dupe sets at ingest."},
    )


class DatasetLoadSourceRequestSchema(Schema):
    """Body for ``POST /api/dataset/load-source``.

    ``source`` is a raw origin dict (``{"importer": ..., "params": {...}}``)
    whose inner shape varies per importer and is validated by the
    handler (via ``can_reload_from_origin`` / ``reload_from_origin``).
    """

    source = fields.Dict(
        required=True,
        metadata={"description": "Origin dict as stored on medias."},
    )


class DatasetLoadStartedResponseSchema(Schema):
    """Response for ``POST /api/dataset/load-demo`` / ``load-file`` /
    ``load-source`` / ``import-local-folder`` and for
    ``POST /api/dataset/combine`` / ``promote`` in ``staging.py``.

    ``task_id`` is the background-task tracker id (string) used by the
    SSE progress stream; it may be empty when the load completes
    synchronously (rare).
    """

    ok = fields.Boolean(required=True)
    message = fields.String(required=True)
    task_id = fields.String(required=True)


class DatasetClearResponseSchema(Schema):
    """Response for ``POST /api/dataset/clear``."""

    ok = fields.Boolean(required=True)


# ---------------------------------------------------------------------------
# Staging routes (vtsearch/routes/datasets/staging.py)
# ---------------------------------------------------------------------------


class _AvailableDatasetFileSchema(Schema):
    """One ``.pkl`` file listed by ``GET /api/dataset/available-files``."""

    name = fields.String(required=True)
    path = fields.String(required=True)
    size_mb = fields.Float(required=True)


class DatasetAvailableFilesResponseSchema(Schema):
    """Response for ``GET /api/dataset/available-files``."""

    files = fields.List(fields.Nested(_AvailableDatasetFileSchema), required=True)


class DatasetCombineResolutionSchema(Schema):
    """One per-embedder-type conflict resolution in a combine request.

    ``action`` is ``"reembed"`` (re-embed every source dataset to *embedder* so
    the whole combined dataset shares that concrete embedder) or ``"drop"``
    (leave that embedder type out of the combined dataset entirely).  ``embedder``
    is required for ``reembed`` and ignored for ``drop``.
    """

    action = fields.String(required=True, validate=validate.OneOf(["reembed", "drop"]))
    embedder = fields.String(load_default="")


class DatasetCombineRequestSchema(Schema):
    """Body for ``POST /api/dataset/combine``."""

    datasets = fields.List(
        fields.String(),
        required=True,
        validate=validate.Length(min=2),
        metadata={"description": "At least two server-side pickle file paths to merge."},
    )
    name = fields.String(load_default="")
    #: Per-embedder-type conflict resolutions, keyed by embedder type
    #: (``semantic`` / ``patch_semantic`` / ``structural``).  Present only when
    #: the sources bind conflicting embedders of the same type; the combine route
    #: refuses (400) a conflict left unresolved here.
    resolutions = fields.Dict(
        keys=fields.String(),
        values=fields.Nested(DatasetCombineResolutionSchema),
        load_default=dict,
        metadata={"description": "Embedder-type -> {action, embedder} conflict resolutions."},
    )


class DatasetPromoteRequestSchema(Schema):
    """Body for ``POST /api/dataset/promote``.

    Promotes a set of media items from the active dataset into a brand-new
    saved dataset (e.g. the Find "Goods" pile). The items keep their
    original origins and embeddings; the new dataset gets a fresh
    ``created_at`` but inherits the source dataset's ``expires_at``.
    """

    name = fields.String(
        required=True,
        validate=validate.Length(min=1),
        metadata={"description": "Display name for the new dataset."},
    )
    media_ids = fields.List(
        fields.Integer(),
        required=True,
        validate=validate.Length(min=1),
        metadata={"description": "IDs of the media items (in the active dataset) to promote."},
    )


class DatasetStageFileResponseSchema(Schema):
    """Response for ``POST /api/dataset/stage-file`` (multipart upload).

    ``count`` and ``media_type`` are derived from a cheap pickle peek and
    fall back to ``0`` / ``"unknown"`` when the file can't be inspected.
    ``error`` carries the reason peek failed (empty string on success), so
    the UI can distinguish "valid pickle with 0 medias" from "couldn't
    parse this file".
    """

    path = fields.String(required=True)
    name = fields.String(required=True)
    count = fields.Integer(required=True)
    media_type = fields.String(required=True)
    error = fields.String(load_default="", dump_default="")


class DatasetStagingStartedResponseSchema(Schema):
    """Response for ``POST /api/dataset/stage-demo/<name>`` and the
    plugin-field staging routes that haven't migrated yet.

    ``task_id`` is the background staging-task tracker id (string) used by
    the ``loading-tasks`` SSE channel to poll progress and pick up the final
    ``staging_result``; it may be empty when no task was started."""

    ok = fields.Boolean(required=True)
    message = fields.String(required=True)
    task_id = fields.String(required=True)


class DatasetStageDemoRequestSchema(Schema):
    """Body for ``POST /api/dataset/stage-demo/<name>``.

    ``name`` is supplied via the URL path; the optional ``converter`` /
    ``dataset_name`` override the demo's defaults.
    """

    converter = fields.String(load_default="")
    dataset_name = fields.String(load_default="")


class ClearStagingResponseSchema(Schema):
    """Response for ``DELETE /api/dataset/staging``."""

    ok = fields.Boolean(required=True)


class ImporterFieldOptionsRequestSchema(Schema):
    """Body for ``POST /api/dataset/import/<importer_name>/options``."""

    field_key = fields.String(required=True, validate=validate.Length(min=1))
    values = fields.Dict(load_default=dict)


class FieldOptionsSchema(Schema):
    """A single dropdown option for a dynamic-options field.

    ``value`` is what the form submits; ``label`` is the friendly text
    shown in the dropdown.  For plain-string options the two coincide; for
    ``(value, label)`` tuple options they differ so a dropdown can submit
    an opaque id while displaying a human-readable name.
    """

    value = fields.String(required=True)
    label = fields.String(required=True)


class ImporterFieldOptionsResponseSchema(Schema):
    """Response for ``POST /api/dataset/import/<importer_name>/options``."""

    options = fields.List(fields.Nested(FieldOptionsSchema), required=True)


class ImporterSuggestedNameRequestSchema(Schema):
    """Body for ``POST /api/dataset/import/<importer_name>/suggested-name``."""

    values = fields.Dict(load_default=dict)


class ImporterSuggestedNameResponseSchema(Schema):
    """Response for ``POST /api/dataset/import/<importer_name>/suggested-name``.

    ``dataset_name`` is what the importer would name a dataset built from
    the supplied form values, i.e. the value the Dataset Name box is
    prefilled with while the user is still typing.
    """

    dataset_name = fields.String(required=True)


# ---------------------------------------------------------------------------
# Registry routes (vtsearch/routes/datasets/registry.py)
# ---------------------------------------------------------------------------


class DatasetRegistryEntrySchema(Schema):
    """One entry of ``GET /api/datasets/registry``.

    The persisted registry record (written in exactly one place,
    :func:`vtscore.datasets.registry.register_dataset`) plus the fields the
    route derives per request: ``loaded``, ``embedders_by_type``, the
    ``clipper`` display-name resolution, and the legacy fallbacks for
    ``bound_embedders`` / ``embedder_types``.  The writer's key set is closed,
    so this is a strict schema.

    One persisted key is deliberately **not** here: ``coverage_branch``, the
    memo the load route writes recording which path that dataset's coverage
    atlas took last time (see ``vtsearch/routes/datasets/registry.py``).  It
    exists to pace the next load's progress bar before the pickle is read, and
    nothing outside the server has any use for it, so it stays out of the wire
    format rather than becoming a field the frontend must ignore.

    Optionality mirrors what the frontend treats as optional (same convention
    as :class:`MediaTypeInfoSchema`), not what the current writer always
    emits: entries persisted by older versions can legitimately lack the
    newer fields.
    """

    id = fields.String(required=True)
    name = fields.String(required=True)
    media_type = fields.String(required=True)
    loaded = fields.Boolean(metadata={"description": "Whether this dataset is currently resident in memory."})
    num_items = fields.Integer(metadata={"description": "Item count after near-duplicate collapsing."})
    num_dupes = fields.Integer(metadata={"description": "How many items were collapsed as near-duplicates."})
    pkl_path = fields.String(metadata={"description": "Server-side path of the dataset pickle."})
    origin = fields.String(metadata={"description": "Name of the importer that produced this dataset."})
    source = fields.Dict(
        allow_none=True,
        metadata={
            "description": (
                "The importer's origin dict for this dataset (importer name plus its field values). Inner keys "
                "are importer-specific, so this stays an opaque map."
            )
        },
    )
    clipper = fields.String(
        metadata={
            "description": (
                'Resolved clipper display name, or ``"-"`` when the dataset used its media type\'s default clipper.'
            )
        }
    )
    embedder = fields.String(
        metadata={"description": "Name of the embedder this dataset's media were vectorised with."}
    )
    embedder_types = fields.List(
        fields.String(),
        metadata={
            "description": (
                "The embedder *types* this dataset supplies (``semantic`` / ``patch_semantic`` / ``structural``); "
                "a v3 trio dataset can supply several. Drives the detector/dataset compatibility gate."
            )
        },
    )
    bound_embedders = fields.List(
        fields.String(), metadata={"description": "Every concrete embedder this dataset binds (primary first)."}
    )
    embedders_by_type = fields.Dict(
        keys=fields.String(),
        values=fields.String(),
        metadata={
            "description": (
                'One concrete embedder per type it binds, e.g. ``{"semantic": "siglip", "patch_semantic": '
                '"dinov3_patch"}``. Drives the Combine-Datasets conflict detector.'
            )
        },
    )
    readers = fields.List(
        fields.String(),
        metadata={
            "description": (
                'Usernames granted read access. Empty means creator-only; ``["*"]`` means visible to everyone.'
            )
        },
    )
    created_by = fields.String()
    created_at = fields.Float(metadata={"description": "Unix timestamp (seconds) at which the dataset was registered."})
    file_type_counts = fields.Dict(
        keys=fields.String(),
        values=fields.Integer(),
        metadata={"description": "Per-extension file counts observed during ingest."},
    )
    ingest_started_at = fields.Float(allow_none=True)
    ingest_finished_at = fields.Float(allow_none=True)
    expires_at = fields.Float(
        allow_none=True,
        metadata={
            "description": (
                "Unix timestamp (seconds) at which this dataset ages off and is automatically removed; "
                "``null``/absent means it never expires."
            )
        },
    )


class DatasetsRegistryListResponseSchema(Schema):
    """Response for ``GET /api/datasets/registry``."""

    datasets = fields.List(fields.Nested(DatasetRegistryEntrySchema), required=True)


class DatasetRegistryLoadResponseSchema(Schema):
    """Response for ``POST /api/datasets/registry/<id>/load``.

    Successful kickoff returns ``task_id``; the "already loaded" path
    returns the same envelope with an empty ``task_id``.
    """

    ok = fields.Boolean(required=True)
    message = fields.String(required=True)
    task_id = fields.String(load_default="")


class DatasetRegistryOkResponseSchema(Schema):
    """Bare ``{"ok": true}`` response (unload, delete)."""

    ok = fields.Boolean(required=True)


class DatasetRegistryPreloadEmbedderResponseSchema(Schema):
    """Response for ``POST /api/datasets/registry/<id>/preload-embedder``.

    ``embedder`` is the name of the embedder being warmed in the
    background, or ``""`` when no embedder could be resolved (e.g. the
    dataset's media type has no registered embedder).
    """

    ok = fields.Boolean(required=True)
    embedder = fields.String(required=True)


class DatasetRegistryRenameRequestSchema(Schema):
    """Body for ``PUT /api/datasets/registry/<id>/rename``."""

    name = fields.String(required=True, validate=validate.Length(min=1, max=MAX_NAME_LENGTH))


class DatasetRegistryRenameResponseSchema(Schema):
    """Response for ``PUT /api/datasets/registry/<id>/rename``."""

    ok = fields.Boolean(required=True)
    name = fields.String(required=True)


class DatasetRegistryReadersRequestSchema(Schema):
    """Body for ``PUT /api/datasets/registry/<id>/readers``.

    Declared as ``fields.Raw`` with a custom validator (rather than
    ``fields.List(fields.String())``) so that numeric or other
    non-string items are rejected as 422 instead of being silently
    coerced to strings by ``fields.String``'s deserializer.
    """

    readers = fields.Raw(
        required=True,
        validate=list_of_strings,
        metadata={
            "description": 'List of usernames; ``["*"]`` makes the dataset public.',
            "type": "array",
            "items": {"type": "string"},
        },
    )


class DatasetRegistryReadersResponseSchema(Schema):
    """Response for ``PUT /api/datasets/registry/<id>/readers``."""

    ok = fields.Boolean(required=True)
    readers = fields.List(fields.String(), required=True)


class DatasetDomainShiftResponseSchema(Schema):
    """Response for ``GET /api/datasets/registry/<id>/domain-shift``.

    Reports how typical the *active* dataset's items look under the named
    reference dataset's coverage atlas.  ``frac_atypical`` is the observed
    fraction of items with typicality p-value below ``alpha`` (roughly the
    shifted proportion); under no shift it stays near ``expected_atypical``.
    ``shifted`` is True when the excess is both statistically clear and
    practically large — a detector trained on the reference dataset should
    not be trusted on the active one without hands-on verification.
    """

    reference_dataset_id = fields.String(required=True)
    n_items = fields.Integer(required=True)
    alpha = fields.Float(required=True)
    frac_atypical = fields.Float(required=True)
    expected_atypical = fields.Float(required=True)
    z_score = fields.Float(required=True)
    median_pvalue = fields.Float(required=True)
    shifted = fields.Boolean(required=True)


class DatasetRegistryStatsResponseSchema(Schema):
    """Response for ``GET /api/datasets/registry/<id>/stats``.

    A superset of the Dashboard grid row: ``name``, ``media_type``,
    ``num_items``, ``created_at``, ``expires_at``, ``created_by`` and
    ``readers`` are the grid's own columns, so the Stats window can show
    everything the grid does while it covers the grid up.
    """

    name = fields.String(required=True)
    media_type = fields.String(required=True)
    num_items = fields.Integer(required=True)
    num_dupes = fields.Integer(required=True)
    file_type_counts = fields.Dict(
        keys=fields.String(),
        values=fields.Integer(),
        required=True,
        metadata={
            "description": (
                "File type → item count. The type is the item's filename extension, or the format "
                "sniffed from its bytes when it has none (a service importer may name items after an "
                "opaque content id). Items no signal could type land in a parenthesised "
                '"(unknown)" bucket.'
            )
        },
    )
    created_at = fields.Raw(allow_none=True)
    expires_at = fields.Raw(allow_none=True)
    created_by = fields.String(required=True)
    readers = fields.List(fields.String(), required=True)
    ingest_started_at = fields.Raw(allow_none=True)
    ingest_finished_at = fields.Raw(allow_none=True)
    origin = fields.String(required=True)
    source = fields.Dict(required=True)
    clipper = fields.String(required=True)
    embedder = fields.String(required=True)


class DuplicateSetMemberSchema(Schema):
    """One member of a collapsed duplicate set (its pre-collapse provenance)."""

    md5 = fields.String(required=True)
    filename = fields.String(required=True)
    category = fields.String(required=True)
    origin_name = fields.String(required=True)
    importer = fields.String(required=True)


class DuplicateSetSchema(Schema):
    """One collapsed duplicate set: its display name and every member."""

    name = fields.String(required=True)
    members = fields.List(fields.Nested(DuplicateSetMemberSchema), required=True)


class DatasetRegistryDuplicatesResponseSchema(Schema):
    """Response for ``GET /api/datasets/registry/<id>/duplicates``."""

    duplicate_sets = fields.List(fields.Nested(DuplicateSetSchema), required=True)


__all__ = [
    "BrowseMediaFilesQuerySchema",
    "BrowseMediaFilesResponseSchema",
    "BrowseMediaFilesSelectRequestSchema",
    "BrowseMediaFilesSelectResponseSchema",
    "CancelDatasetLoadResponseSchema",
    "CleanersListQuerySchema",
    "CleanersListResponseSchema",
    "ClearStagingResponseSchema",
    "ClippersListQuerySchema",
    "ClippersListResponseSchema",
    "ConvertersListQuerySchema",
    "ConvertersListResponseSchema",
    "DashboardDiskUsageResponseSchema",
    "DashboardRamUsageResponseSchema",
    "DatasetAllImportersListResponseSchema",
    "DatasetAvailableFilesResponseSchema",
    "DatasetClearResponseSchema",
    "DatasetCombineRequestSchema",
    "DatasetCombineResolutionSchema",
    "DatasetDomainShiftResponseSchema",
    "DatasetImportersListResponseSchema",
    "DatasetLoadDemoRequestSchema",
    "DatasetLoadSourceRequestSchema",
    "DatasetLoadStartedResponseSchema",
    "DatasetRegistryDuplicatesResponseSchema",
    "DatasetRegistryLoadResponseSchema",
    "DatasetRegistryReadersRequestSchema",
    "DatasetRegistryReadersResponseSchema",
    "DatasetRegistryRenameRequestSchema",
    "DatasetRegistryRenameResponseSchema",
    "DatasetRegistryStatsResponseSchema",
    "DatasetStageDemoRequestSchema",
    "DatasetStageFileResponseSchema",
    "DatasetStagingStartedResponseSchema",
    "DatasetStatusResponseSchema",
    "DatasetsRegistryListResponseSchema",
    "DemoCategoriesResponseSchema",
    "DemoDatasetListQuerySchema",
    "DemoDatasetListResponseSchema",
    "EmbeddersListQuerySchema",
    "EmbeddersListResponseSchema",
    "DatasetRegistryOkResponseSchema",
    "DatasetRegistryPreloadEmbedderResponseSchema",
    "FieldOptionsSchema",
    "ImporterFieldOptionsRequestSchema",
    "ImporterFieldOptionsResponseSchema",
    "MediaTypesListResponseSchema",
]
