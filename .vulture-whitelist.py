"""Whitelist for vulture's dead-code detector.

Vulture finds defined-but-never-referenced names. This file lists symbols
that VTSearch DOES use, but only reflectively - so static analysis can't
see the reference.

**Do not run vulture by hand.** ``scripts/vulture-audit.py`` owns the
invocation (scan paths, excludes, ignore lists, confidence floor) and
applies this file:

    python scripts/vulture-audit.py                    # the pre-release audit
    python scripts/vulture-audit.py --check-whitelist  # the run-tests.sh gate

The scan covers **every tier that defines or consumes first-party Python**:
``vtsearch/``, ``app.py``, ``tests/``, ``vtscore/``, ``tests_lib/``, and
``scripts/``. It used to cover only the first three, which is how this file
rotted: half its entries were written for symbols living in ``vtscore/`` and
so could never fire, and another twenty suppressed nothing because the real
caller lived in a directory nobody was scanning.

**Every entry below must suppress a real finding.** The
``--check-whitelist`` gate fails the build when one doesn't, because an
entry that suppresses nothing makes a claim about the codebase that nothing
can ever check - it will quietly outlive the symbol it was written for. If
the gate names your entry, the symbol is either gone or has acquired a real
caller: delete the entry and its comment.

**A vulture hit on a public ``vtscore`` name is not evidence of anything.**
Out-of-tree extensions import ``vtscore`` symbols no in-repo grep can see,
so the library tier surfaces many public names with no *internal* caller.
That is expected output. Such a name gets an entry here with a true reason,
never a deletion - removing one is a deliberate library break needing an
``[Unreleased]`` note in ``vtscore/CHANGELOG.md`` and the owner's sign-off.
See CLAUDE.md, "Dead code in ``vtscore/`` is a claim you cannot verify by
grepping".

Why the excludes / ignores (all configured in ``scripts/vulture-audit.py``):

* ``vtsearch/schemas/*`` and ``vtsearch/settings_models.py`` are pure
  framework-managed declaration files. Every field assignment in a
  marshmallow ``Schema`` subclass or a pydantic ``BaseModel`` looks
  "unused" to vulture because both frameworks collect fields via
  metaclass at class-creation time. There is no way to tell vulture
  about that linkage short of listing every field by hand, so we just
  skip the directories. Dead schemas, if any, will surface as unused
  imports in the route files that consume them.
* ``scripts/experiments/`` is scanned but not *reported on*: the drivers
  import ``vtscore``, so scanning them resolves ten ``vtscore`` findings
  that would otherwise read as dead, but an unused constant inside an
  archival record of a run that already happened is not a liability.
* ``--ignore-names Meta,model_config`` covers the same metaclass-managed
  inner-class / config-attribute patterns for any stragglers that aren't
  in the excluded paths.
* ``--ignore-names _keys_to_ignore_on_load_unexpected`` is HuggingFace's
  convention for telling ``transformers.PreTrainedModel`` to skip
  certain weight keys when loading; the attribute is read by the base
  class via reflection.
* Flask route handlers are decorated with ``@bp.route(...)`` etc., which
  vulture doesn't follow back to a call site - the decorator filter
  covers ``@*.route``, ``@*.before_request``, ``@*.errorhandler``,
  ``@bp.*``, ``@app.*``, and the rest of the Flask lifecycle hooks.
* Test-only names: ``test_*`` and ``Test*`` are pytest-discovered, the
  ``setup_method``/``teardown_method``/``setup_class``/``teardown_class``
  hooks are pytest fixture lifecycle, ``pytest_*`` and ``pytestmark`` are
  framework reserved.
* Python protocol dunders (``__enter__``, ``__exit__``, ``__package__``)
  are called by the runtime, not by user code - vulture sees the
  assignment (e.g. on a ``MagicMock`` instance for a context-manager
  test) but no read.

Whitelist entries below cover the remaining individual symbols that
vulture flags but are actually used reflectively or as the public API
surface of a module.
"""

# ---------------------------------------------------------------------------
# Plugin sentinels - each ``<FAMILY> = SomeClass`` line at the bottom of a
# plugin module is discovered by the ``PluginRegistry`` scanner via the
# matching attribute name. Vulture sees the assignment but no reference;
# discovery happens at import time through ``getattr``.
# ---------------------------------------------------------------------------
EXPORTER  # noqa: F821
CONVERTER  # noqa: F821
SETTINGS_SOURCE  # noqa: F821
LABELSET_SOURCE  # noqa: F821
DATASOURCE_IMPORTER  # noqa: F821 - vtscore.datasource_importers, sentinel= in its registry

# ---------------------------------------------------------------------------
# argparse.Action.__call__ requires the ``option_string`` parameter even
# when the action ignores it.
# ---------------------------------------------------------------------------
option_string  # noqa: F821

# ---------------------------------------------------------------------------
# Flask reads ``app.secret_key`` via attribute access for session signing.
# Setting it is the side-effecting "use".
# ---------------------------------------------------------------------------
secret_key  # noqa: F821

# ---------------------------------------------------------------------------
# Public-API forwarders in ``vtscore.concurrency.progress`` that mirror
# their ``update_<tracker>_progress`` partners. The corresponding
# trackers (``sort_progress``, ``find_progress``) are imported and read
# directly in tests / routes; the helper wrappers stay for API symmetry
# and are tabulated in ``vtscore/docs/packages/concurrency.md``.
# ---------------------------------------------------------------------------
get_sort_progress  # noqa: F821
get_eval_progress  # noqa: F821

# ---------------------------------------------------------------------------
# Public module constants - referenced by callers via ``module.NAME`` or
# settings lookup, which vulture treats as the only assignment.
# ---------------------------------------------------------------------------
SAVED_DATASETS_DIR  # noqa: F821 - vtscore.datasets.registry default dir
DETECTORS_DIR  # noqa: F821 - vtscore.detectors.store default dir
SAMPLE_VIDEOS_DOWNLOAD_SIZE_MB  # noqa: F821 - downloader size budget constant
CUT_FALLBACK_KINDS  # noqa: F821 - vtscore.eval.cut_rules, the enumerated fallback reasons

# ---------------------------------------------------------------------------
# Public ``vtscore`` API with no *internal* caller. Each is exported from
# its package ``__init__`` and/or tabulated in ``vtscore/docs/packages/``,
# which makes it importable by out-of-tree extensions this repo cannot see.
# Per CLAUDE.md these keep their names; deleting one is a library break.
# ---------------------------------------------------------------------------
recreate_model_at_time  # noqa: F821 - vtscore.detectors.labeling_progress
clear_dataset  # noqa: F821 - vtscore.datasets.load_pipeline
is_request_missing_context  # noqa: F821 - vtscore.state.core; docs/packages/state.md
build_md5_lookup  # noqa: F821 - vtscore.state.media_lookup; docs/packages/state.md
get_find_scores  # noqa: F821 - vtscore.state.votes; docs/packages/state.md
set_dataset_display_name  # noqa: F821 - vtscore.state; docs/packages/state.md
make_plugin_route_schema  # noqa: F821 - vtscore.plugins.schema; kept for out-of-tree app tiers that mint a route per plugin (see the docstring)

# ---------------------------------------------------------------------------
# ``SplgMatcher`` is the SuperPoint + LightGlue structural backend that the
# structural-embedder design reserves alongside the shipped SIFT one: a
# StructuralMatcher-conformant alternative, evaluated in the 2026-07-13
# iconography study and wired in by choosing it, not by being called from
# here. A zero-registrant extension point is the shape of a working one.
# ---------------------------------------------------------------------------
SplgMatcher  # noqa: F821

# ---------------------------------------------------------------------------
# Public context managers exported from ``vtsearch.state`` for callers
# that need explicit, scoped switching of the active dataset/detector
# without going through the per-request middleware.
# ---------------------------------------------------------------------------
with_dataset_context  # noqa: F821
with_detector_context  # noqa: F821

# ---------------------------------------------------------------------------
# TYPE_CHECKING-only accessor stubs in ``vtsearch.settings``. The actual
# runtime functions are generated by ``make_accessors`` at module-bottom;
# the ``if TYPE_CHECKING:`` block exists so pyright can resolve
# ``settings.get_<key>()`` / ``settings.set_<key>()`` from other modules.
# The ``set_<key>`` runtime versions are also called reflectively via
# ``getattr(settings, f"set_{key}", None)`` in
# ``vtsearch/routes/settings/api.py``. Vulture sees the stub but can't
# connect it to the dynamic definition or the runtime callers.
# ---------------------------------------------------------------------------
get_audio_playing  # noqa: F821
get_hide_autopilot  # noqa: F821
get_autopilot_resort_interval  # noqa: F821
get_browse_panel_width  # noqa: F821
get_browse_colormap  # noqa: F821
get_browse_icon_size  # noqa: F821
get_browse_thumbnail_border  # noqa: F821
set_audio_playing  # noqa: F821
set_show_animations  # noqa: F821
set_hide_autopilot  # noqa: F821
set_browse_panel_width  # noqa: F821
set_browse_colormap  # noqa: F821
set_browse_icon_size  # noqa: F821
set_browse_thumbnail_border  # noqa: F821
get_browse_mouse_zooms_per_level  # noqa: F821
set_browse_mouse_zooms_per_level  # noqa: F821
get_browse_signposts  # noqa: F821
set_browse_signposts  # noqa: F821
get_browse_graphics  # noqa: F821
set_browse_graphics  # noqa: F821
set_browse_signpost_captioner  # noqa: F821
set_autopilot_resort_interval  # noqa: F821
set_projection_n_neighbors  # noqa: F821
set_projection_min_dist  # noqa: F821
get_bin_details_docked  # noqa: F821
set_bin_details_docked  # noqa: F821
get_browse_details_panel_width  # noqa: F821
set_browse_details_panel_width  # noqa: F821
get_browse_details_metadata_width  # noqa: F821
set_browse_details_metadata_width  # noqa: F821
get_grid_icon_size_popup  # noqa: F821
set_grid_icon_size_popup  # noqa: F821
get_popup_metadata_shown  # noqa: F821
set_popup_metadata_shown  # noqa: F821
get_popup_preview_size  # noqa: F821
set_popup_preview_size  # noqa: F821

# ---------------------------------------------------------------------------
# Per-side setting validators generated by ``_make_per_side_setting`` in
# ``vtsearch/settings.py``. Each ``validate_<key>_left`` / ``_right`` is
# looked up reflectively by ``validate_setting(key, value)`` via
# ``globals().get(f"validate_{key}")``, which is what ``PUT /api/settings``
# uses to validate a multi-key body up front. Vulture sees the tuple
# unpack but not the dynamic caller.
# ---------------------------------------------------------------------------
validate_grid_icon_size_left  # noqa: F821
validate_grid_icon_size_right  # noqa: F821
validate_focus_mode_left  # noqa: F821
validate_focus_mode_right  # noqa: F821
validate_panel_pct_left  # noqa: F821
validate_panel_pct_right  # noqa: F821

# ---------------------------------------------------------------------------
# ``verification_classifier`` is a declared ``DetectorContext`` slot (listed
# in the context's ``__slots__`` table in ``vtscore/state/core.py``) that the
# structural-similarity trainer writes. It is an in-memory head cache, so
# the write is the point and no reader lives in this repo; the slot is part
# of the context's shape either way.
# ---------------------------------------------------------------------------
verification_classifier  # noqa: F821

# ---------------------------------------------------------------------------
# Third-party protocol members: the framework calls or reads these by name,
# so the definition here is the whole contract and there is no in-repo
# caller to find.
# ---------------------------------------------------------------------------
objtype  # noqa: F821 - descriptor protocol: __get__(self, obj, objtype=None)
ir_version  # noqa: F821 - onnx ModelProto field, read by onnx.checker / the runtime
layerdrop  # noqa: F821 - transformers AutoConfig field, read when building the module graph
log_logit_scale  # noqa: F821 - torch.nn.Parameter; lives in the checkpoint's state_dict
padding_side  # noqa: F821 - transformers tokenizer setting, read during batched generation
show_progress_bar  # noqa: F821 - sentence-transformers ``encode()`` shape that toponymy calls
supports_system_prompts  # noqa: F821 - toponymy LLMWrapper property, read by the base class
init_poolmanager  # noqa: F821 - requests HTTPAdapter override, called by requests
_pool_connections  # noqa: F821 - the three attributes that make a requests adapter picklable
_pool_maxsize  # noqa: F821
_pool_block  # noqa: F821
__getattr__  # noqa: F821 - PEP 562 module-level hook in ``vtsearch/settings.py``
__wrapped__  # noqa: F821 - functools convention, read by ``inspect.unwrap``
create_module  # noqa: F821 - importlib Loader protocol, called by the import machinery
trust_env  # noqa: F821 - requests.Session attribute, read by requests when sending

# ---------------------------------------------------------------------------
# Flask's ``@after_this_request`` registers the decorated function as a
# response callback on the current request; the name is never referenced
# again. Vulture's ``--ignore-decorators`` matches ``@*.after_request``
# (the blueprint/app form) but not the bare ``@after_this_request`` call.
# ---------------------------------------------------------------------------
_cache_tile  # noqa: F821

# ---------------------------------------------------------------------------
# Pytest fixture parameters whose side effects are the point - the body
# never references the argument name, but the fixture must be requested
# by name in the signature. Vulture flags each parameter at 100 %
# confidence because the local name has no read.
# ---------------------------------------------------------------------------
stub_run_eval  # noqa: F821
stub_pipeline  # noqa: F821
stub_extractor_factory  # noqa: F821
stub_localizer_factory  # noqa: F821
stubbed_resolver  # noqa: F821
reset_state  # noqa: F821
fake_caps  # noqa: F821 - monkeypatches the embedder capability table
registered_cleaners  # noqa: F821 - installs media cleaners into the registry
clean_seen_embedders  # noqa: F821 - clears the load profiler's seen-embedder set
restore_pil_limit  # noqa: F821 - saves/restores PIL's decompression-bomb ceiling
restore_provider  # noqa: F821 - saves/restores the active login provider
as_alice  # noqa: F821 - registers a confining provider and runs as user ``alice``
single_user  # noqa: F821 - forces single-user (unconfined) file access
wide_video  # noqa: F821 - stubs a wide video for the frame sampler
queried  # noqa: F821 - captures the queries a paired embedder issues
converged_training  # noqa: F821 - pins training hyperparameters to convergence
degenerate_gmm  # noqa: F821 - forces the GMM to degenerate
restore_resolvers  # noqa: F821 - re-binds the resolver globals via monkeypatch so teardown restores them
clean_paths  # noqa: F821 - saves/restores sys.path and sys.meta_path around setup_env
restore_stdlib  # noqa: F821 - re-installs the stdlib packages_distributions via monkeypatch for the test

# ---------------------------------------------------------------------------
# Mock function signatures that must match a real API but whose body
# ignores certain kwargs. Callers pass the kwarg by name, so the parameter
# must exist in the mock's signature even though the mock discards it.
# ---------------------------------------------------------------------------
repo_type  # noqa: F821 - huggingface_hub.snapshot_download(repo_id, repo_type=..., ...)
ignore_patterns  # noqa: F821 - same call's file filter
pretrained  # noqa: F821 - facenet-pytorch InceptionResnetV1(pretrained=...)

# ---------------------------------------------------------------------------
# Test-local names that exist only to document the shape of an unpacked
# tuple. They are unused after binding by design.
# ---------------------------------------------------------------------------
first_chunk_medias  # noqa: F821
cov_base  # noqa: F821 - the (base, span) pair for the finalize "coverage" slot
reg_base  # noqa: F821 - ditto for "registry"
orig_text  # noqa: F821 - vtscore.datasets.importers.combine_datasets binding triple
cost_k  # noqa: F821 - vtscore.eval.voting_iterations operating-cost triple

# ---------------------------------------------------------------------------
# Autouse fixture imported for side effects in both conftests (the ``import``
# is what registers it with pytest). The local alias goes through ``as
# _allow_test_tmp_paths`` so a reader immediately sees ``F401`` and knows the
# import is deliberate; the ``noqa: F401`` at the import site silences ruff,
# and this entry silences vulture.
# ---------------------------------------------------------------------------
_allow_test_tmp_paths  # noqa: F821

# ---------------------------------------------------------------------------
# ``StreamProgress.readable`` overrides ``io.IOBase.readable`` on a stream
# wrapper the container reader hands to ``pickle.load``. Python's io / pickle
# machinery queries ``.readable()`` reflectively before pulling bytes; there
# is no in-repo caller by design.
# ---------------------------------------------------------------------------
readable  # noqa: F821 - vtscore.datasets.container.StreamProgress, io.IOBase override

# ---------------------------------------------------------------------------
# ``ErrorSchema.error_code`` is a marshmallow field on the flask-smorest error
# schema that documents the ``error_code`` slug we surface (``dataset_not_loaded``,
# ``auth_required``, …). Marshmallow collects fields via metaclass at class-
# creation time, so vulture cannot see the linkage. The blanket ``vtsearch/schemas/*``
# exclude does not cover this file — the schema lives in ``vtsearch/errors.py``
# alongside the handlers that populate the field.
# ---------------------------------------------------------------------------
error_code  # noqa: F821

# ---------------------------------------------------------------------------
# ``VTSearchApi.ERROR_SCHEMA`` overrides flask-smorest's ``Api.ERROR_SCHEMA``
# so ``/api/openapi.json`` documents *our* error envelope, not the library's.
# flask-smorest reads ``self.ERROR_SCHEMA`` reflectively in
# ``flask_smorest/spec/__init__.py`` when building the default-error response;
# there is no in-repo caller.
# ---------------------------------------------------------------------------
ERROR_SCHEMA  # noqa: F821
