# Changelog - `vtscore`

All notable changes to the `vtscore` library are documented here. The
project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

`vtscore` versions are tracked manually in `vtscore/__init__.py` and move
only when a release is cut - there is no auto-bump on commit. (The companion
[`vtsearch`](../README.md) application uses a git-derived timestamp version
instead, since every commit on `dev` is effectively a new app release.)

### Added

- **`vtscore.utils.import_metadata`** (issue #3715) - `seed_packages_distributions()`
  installs a stat-free stand-in for `importlib.metadata.packages_distributions`,
  which `transformers` calls at module import. The stdlib version stats every
  file recorded by every installed distribution (~85k on a torch + RAPIDS venv);
  on an NFS install with a cold cache that was 16 minutes of silent startup.
  `initialize_models()` now seeds it before anything can import transformers.
  Purely additive: the seed is idempotent and best-effort, and the replacement
  reports every entry the stdlib does.

- **A timing profile can price a forked step per branch** (issue #3594).
  `vtscore.timing.step_weights` and `step_terms` take an optional `branch=` -
  a branch name from `CHEAP_BRANCHES` / `DEAR_BRANCHES`, or a `{step: branch}`
  mapping - and price any step the profile measured on that branch from its own
  coefficients. `vtscore.timing.fit.fit_branches` produces them, stored under
  the step's new `branches` key, and `TimingProfile` gained a parallel
  `branches` table to hold them.

  Purely additive in both directions. The step's top-level coefficients still
  describe the dear branch, so a caller that passes no `branch` (every existing
  call site), a profile written before this existed, and a build that has never
  heard of `branches` all behave exactly as they did - which is also why the
  profile schema version did not move. A step whose runs all took one branch
  gets no split at all.

  The problem it solves is that a profile cell is keyed
  `(device, media_type, embedder)`, so a step that forks on a cache had one set
  of coefficients for two code paths measured 110-700x apart: an admin picking a
  profile was picking which branch to be wrong about, at up to 0.94 of a
  progress bar.

  The sibling of `skip_steps` below, and the half of the problem it does not
  cover: a step that does not run costs nothing and needs no measurement, while
  a step that runs either way and costs two different things needs one
  measurement per path.

- **`skip_steps` on `vtscore.timing.step_weights()` / `step_terms()`, and
  `MediaEmbedder.models_loaded()`** (issue #3596). A timing profile prices how
  long a step takes; it has no way to say a step will not run at all. That gap
  is what left `text_sort` unpaced by anything: its `load_model` step is
  seconds on a process's first sort and exactly zero on the next 47, so every
  profile fitted for it - and the shipped defaults - put 0.80-0.85 of the bar
  in the wrong step. `skip_steps` names the steps this particular run will
  skip and prices them at zero, which needs no measurement; the remaining
  steps share the whole bar. Purely additive - an omitted `skip_steps` paces
  exactly as before.

  `models_loaded()` is the accompanying public read on `MediaEmbedder`: "would
  `load_models()` do any work?", answered without doing it. The default reads
  the same private model attribute `load_models()` and `loaded_backbone()`
  already rely on, so an embedder built the usual way needs no change; one
  that holds its backbone elsewhere should override both together.

- **`slot_embedders_for_snap(snap)` and `keying_embedder_for_type(type, snap)`
  on `vtscore.embedding.binding`** (issue #3386). Purely additive companions to
  the two existing snapshot resolvers, added so the near-synonymous private
  wrappers scattered across `vtscore/detectors/` could collapse onto them.
  `slot_embedders_for_snap` is `derive_binding_from_names` applied to a
  snapshot's first media, returning the raw `(text, patch, structural)` triple
  without `score_marker_embedder_for_snap`'s collapse-and-fall-back-to-primary
  step. `keying_embedder_for_type` is `keying_embedder_for_snap`'s body taking
  the detector's embedder *type* as a plain string, for callers holding a
  serialised detector rather than a live context - four of them were conjuring
  a throwaway `SimpleNamespace(embedder_type=...)` carrier to satisfy the
  `det_ctx` parameter. `keying_embedder_for_snap` now delegates to it, so the
  two cannot disagree. No resolved embedder name changes.

  `vtscore.detectors.training.detector_score_embedder` keeps its name, its
  signature and its `None`-for-empty-snap normalisation; only its docstring
  moved. The four underscore-prefixed wrappers around it are private and were
  inlined or collapsed.

- **`vtscore.host_seams` snapshots the host seams** (issue #3385). The eight
  callbacks a host application installs into the library
  (`register_core_config_builder`, the context resolvers, the setting
  persisters, the achievement recorders, …) are each a process global declared
  beside the code that calls them, so nothing knew the complete list — and a
  test that installed one leaked it into every test that followed.
  `capture_host_seams()` / `restore_host_seams()` snapshot and put back the
  whole set. Deliberately snapshot-and-restore rather than reset-to-default, so
  an integration whose tests run against its real wiring keeps the seams under
  test. See `docs/integration.md`.

  `register_plugin_family` is deliberately **not** covered: the library
  registers its own families at import time, so it is a plugin extension point
  rather than a host seam, and restoring it would drop the built-ins.

- **The Toponymy signpost fit counts the library warnings it suppresses**
  (issue #3512). `signpost_build` keeps Toponymy's per-topic naming warnings
  off the CLI (issue #2558), but it used to `filterwarnings("ignore")` them,
  which discarded the count along with the output — nothing anywhere could
  tell zero warnings from a flood, so a library bump that re-broke the
  `KeyphraseNamer` prompt parse (fixed in issue #2567) would have surfaced
  only as an unexplained slow build. The fit now tallies them and logs one
  line when it returns (`warning` level when the count is non-zero or a topic
  fell back to the literal name `"unnamed"`, `debug` otherwise), with the
  per-message breakdown and the disambiguation-pass count. Other modules'
  warnings still reach `showwarning` untouched, exactly as before.

  `make_keyphrase_namer(on_name=None, counts=None)` gained the optional
  second parameter — a `Counter` the namer tallies its `"named"` /
  `"unnamed"` / `"disambiguation"` calls into. Purely additive; existing
  callers are unaffected.

- **All four `autodetect_*_main` entry points accept `stream_results` and
  `keep_negatives`** (issue #3432). The two keyword-only flags used to exist
  only on the chunked pair, so a whole-dataset run had no way to hand a
  streaming-capable exporter a lazy record iterator even though the pipeline
  underneath supported it. Purely additive: the four names, their positional
  parameters and their defaults are unchanged, and the flags default to
  `False` (the previous behaviour). Internally the four are now thin shims
  over one private `_autodetect(spec, ...)` driven by a `_SourceSpec`, which
  is what let the matrix drift in the first place.

- **`AsyncJob` is backed by a `ProgressTracker`** (issue #3380). Every job now
  owns one (`job.progress`) instead of carrying its own copy of the tracker's
  field set, so background jobs get the parts a hand-rolled copy never had: a
  smoothed, coarsened `eta_seconds`, the whole-job `overall` /
  `overall_step_end` fractions with optional per-phase weights
  (`job.progress.set_step_weights(...)`), and `subscribe()` for pushing
  snapshots rather than polling them. `job.progress.get()` returns the whole
  set at once.

  Cancellation collapses to one flag in the same motion: `AsyncJob.cancel_event`
  **is** the tracker's event, so `job.cancel()`, `job.progress.cancel()`,
  `job.is_cancelled`, `check_job_cancelled()` and
  `job.progress.check_cancelled()` all act on a single
  `threading.Event` and raise a single `CancelledError`.

  **For library consumers:** additive at every documented surface.
  `current` / `total` / `message` / `step` / `total_steps` and `cancel_event`
  remain readable and writable under their old names — they are now properties
  over the tracker rather than dataclass fields, which changes only two things:
  they can no longer be passed to the `AsyncJob(...)` constructor (nothing in
  the tree did), and a write publishes a snapshot to subscribers instead of
  mutating an attribute in place.

- **`ProgressTracker.cancel_event`** — the tracker's cancellation flag, exposed
  so a holder that must publish exactly one cancel signal can hand out the
  tracker's event rather than keeping a second one in sync. Prefer `cancel()` /
  `check_cancelled()` / `is_cancelled` for ordinary use.

- **`PROGRESS_COMMON_EXTRAS`** — the extras every long-running operation
  declares (`step`, `total_steps`, `error`, `eta_seconds`, `overall`,
  `overall_step_end`), promoted from the private `_PROGRESS_COMMON_EXTRAS` so a
  caller constructing its own `ProgressTracker` can opt into the same payload
  shape the frontend already renders. The private name is gone; it was never
  contract.

- **`MediaEmbedder.loaded_backbone()` - a supported way to reach an embedder's
  raw model** (issue #3395). Returns `(model, processor)`, loading the model
  first if needed; the second element is `None` for backbones that need no
  separate processor. The default implementation reads the `_model` /
  `_processor` attribute convention that `load_models()` itself relies on, so
  every embedder built the usual way gets it for free, and an embedder that
  holds its backbone elsewhere overrides this one documented method. It raises
  `RuntimeError` rather than returning `None` when no backbone is resident
  after loading.

  **For plugin authors:** purely additive - nothing is required of an existing
  embedder. The change worth knowing about is on the *consumer* side: the
  `vtscore.embedding.loader` getters `get_clap_model`, `get_xclip_model` and
  `get_e5_model` keep their names, signatures and return shapes, but now go
  through `loaded_backbone()` instead of reaching into each subclass's private
  `_get_model_and_processor()` / `_get_model()` via `cast(Any, emb)`. Those six
  private helpers are gone; nothing outside the getters called them, and a
  private name was never contract. If you were calling one anyway, call
  `loaded_backbone()` instead.

### Changed

- **`JOB_MANAGERS` is now the single registry of every module-level
  `JobManager`, with visibility carried by the manager** (issue #3404). It
  previously held only the managers surfaced by `/api/jobs/active`, which
  forced `reset_all_async_jobs_for_tests()` to re-list the hidden ones by
  hand — and that second list went stale, leaving `archive_thumbnail_jobs`
  unreset between tests. `JobManager.__init__` gained a keyword-only
  `user_visible: bool = True`, `list_active_pairs()` filters on it, and the
  reset helper walks the whole registry. Registering a new manager is now one
  edit in one place, and its visibility is declared at its own definition.

  **For library consumers:** the `JobManager(...)` change is purely additive —
  the default keeps existing constructions user-visible. `JOB_MANAGERS` itself
  now enumerates the internal managers too (`labeling-status`,
  `signpost-relabel`, `archive-thumbnail-warm`), so code that iterated it
  expecting only user-facing work should read `list_active_pairs()` or filter
  on `mgr.user_visible`. The keys of the visible entries are unchanged.

- **`register_setting_persister` rejects an unrecognised key** (issue #3385).
  It previously stored a persister under any key, but only `inclusion`,
  `calibrate_count` and `calibration_fraction` are ever fired, so any other key
  was a typo in the host's wiring that sat there silently never firing — the
  failure mode `achievements_hooks.KNOWN_EVENTS` already existed to catch. The
  key set is now public as `vtscore.state.KNOWN_SETTING_KEYS`, and an unknown
  key raises `ValueError`. This is an extension-facing behaviour change, but no
  working integration can be relying on it: a rejected key could never have
  persisted anything.

- **`vtscore.training.evt_mixture` moved to `vtscore.eval.evt_mixture`** (issue
  #3396). The Gumbel/Normal score mixture is a research arm from the #2836 /
  #2846 cut studies, not a shipped threshold rule: nothing in the app or the
  library's production path fits a Gumbel, and its only consumers are the eval
  harness's cut families and the calibration studies under
  `scripts/experiments/calibration/`. Sitting in `vtscore/training/` put it
  beside the `thresholds/` package that *is* offered to library consumers, with
  nothing to tell the two apart.

  **For library consumers:** this is a breaking move of a module path, taken
  deliberately rather than shimmed. `GumbelNormalFit1D`,
  `fit_gumbel_normal_mixture_state`, `gaussian_mixture_mean_loglik`,
  `CROSSING_REASONS` and the rest are unchanged and keep their names; only the
  module they live in changed, so an importer updates one line. There is no
  re-export at the old path on purpose: `vtscore.training` importing from
  `vtscore.eval` would invert the package layering and drag the eval package's
  matplotlib import into the training tier. The Gaussian threshold rules that
  production actually ships (`vtscore.training.thresholds`) are untouched.

- **A broken `vtscore.datasets` install now fails loudly instead of silently
  disabling origin resolution** (issue #3397).
  `vtscore.detectors.resolver` used to defer its imports of
  `vtscore.datasets.sources` / `vtscore.datasets.importers` into a first-use
  auto-wire step wrapped in `except ImportError: pass`. Because both packages
  ship in this same distribution, that `except` could only ever fire on a
  genuinely broken install - and when it did, the module carried on with *no*
  resolvers registered, so every label silently failed to resolve and the only
  symptom was an "N of M labels resolved" warning. Both imports are now
  module-level, and the defaults are bound at import time rather than on first
  use, so the underlying `ImportError` reaches the caller with its real
  traceback.

  **For library consumers:** `register_source_resolver`,
  `register_importer_resolver`, `SourceResolver` and `ImporterResolver` are
  unchanged, and a registered resolver still replaces the default. The only
  visible difference is *when* an already-fatal misconfiguration surfaces.

- **`vtscore.config` is now a package, not a single module** (issue #3375). The
  933-line file is split into `paths`, `runtime`, `models`, `device`,
  `processor_backend` and `core_config`, layered so each reads only from the ones
  before it. **Nothing about the import surface changes:** every public name is
  re-exported from `vtscore/config/__init__.py` and listed in its `__all__`, so
  `vtscore.config.X` and `from vtscore.config import X` resolve exactly as before,
  and the old file path was never a documented import path anyway.

  **For plugin authors:** the one behaviour that moved is *reloading*.
  `importlib.reload(vtscore.config)` now only re-runs the re-exports - the
  submodules are already in `sys.modules`, so the environment variables are not
  re-read. Call `vtscore.config._reload_all()` for the old whole-module reload.
  Likewise, stubbing a name on the package reaches callers outside the package
  but not the package's own functions, which resolve their module globals: patch
  `vtscore.config.device` / `vtscore.config.runtime` for those. Private names are
  deliberately not re-exported, so an attempt to stub one on the package raises
  instead of being silently ignored.

- **`MediaClipper.resolve_for_durations` is documented as reserved and inert**
  (issue #3395). The method is unchanged and still part of the ABC - removing
  it would turn a third-party clipper's silently-inert override into a hard
  error - but the docstring and all three clipper guides claimed it was "called
  once per dataset at load time", which stopped being true when auto-routing
  moved to the per-item `resolve_for_media`. Nothing invokes it. **If you
  override it, your override never runs**; move the logic to
  `resolve_for_media`.

- **`apply_converter_to_demo`'s `embedder_name` is documented as accepted and
  ignored** (issue #3395). Conversion changes the media type, so an embedder
  chosen for the source type does not apply to the outputs - the framework
  embed stage resolves the target type's embedder itself. The parameter is
  kept, since out-of-tree callers may still pass it.

- **`vtscore.timing.profile_covers` keeps its export; its docstring no longer
  claims callers it does not have** (issue #3395). It named the tuning script's
  coverage report and the dataset-load path, neither of which calls it. It is
  public API with no in-repo caller, and is documented as such.

### Deprecated

- **`vtscore.state.coverage_atlas` and `vtscore.state.near_dupes` have moved**
  (issue #3391). Both were pure algorithms filed under `state/`: neither
  references `DatasetContext` or `_state_lock`, and the Coverage Atlas sat
  beside its own wiring module (`state/coverage.py`) as a near-homograph pair
  in which only one of the two was actually state.

  - `vtscore.state.coverage_atlas` → **`vtscore.coverage`** (a new package;
    the module itself is `vtscore.coverage.atlas`). Exports `CoverageAtlas`,
    `auto_max_depth`, `domain_shift_report` and the `COVERAGE_ATLAS_*`
    partition defaults.
  - `vtscore.state.near_dupes` → **`vtscore.media.near_dupes`**, beside the
    media types whose bytes it hashes.

  **Nothing breaks yet.** Both old module paths remain as aliases that
  re-export the new location and raise a `DeprecationWarning` on import;
  attribute access falls through to the new module, so even private names
  keep resolving. They will be removed in a future release — update imports
  to the new paths.

  The `vtscore.state` re-exports are **unchanged**: `build_coverage_atlas*`,
  `coverage_atlas_*`, `collapse_near_duplicates`, `phash_image` and
  `simhash_text` all still import from `vtscore.state` exactly as before, as
  do their `vtsearch.state` counterparts. `vtscore.state.sort_results_cache`
  has **not** moved: it exports a process-global mutable singleton with its
  own lock, which is state by any reading.

### Removed

- **The global dataset-progress system: `dataset_progress`, `get_progress()`
  and `check_dataset_cancelled()`** (issue #3376). Dataset and import progress
  now lives entirely in the per-task `loading_tasks` registry, one
  `ProgressTracker` per operation. The SSE `dataset` channel these fed is gone
  with them; per-task progress rides `loading-tasks`, which the dashboard has
  read for some time.

  A process-wide progress sink has no owner, and that turned out to be the
  whole bug class: nothing could say when the work it was narrating had ended,
  so a finished import and a wedged one produced identical output (#3167). The
  workarounds it needed were the tell — a `_park_global_progress_if_orphaned()`
  sweep on the last load out of the door, a synthetic terminal tick appended to
  every unscoped `load_models()`, and a `LEGACY_PROGRESS_TARGET` special case
  threaded through cancellation. All are deleted rather than fixed. Removed
  alongside them: `MediaEmbedder._orphan_progress` (the terminal-tick wrapper),
  `LoadingTasksTracker.any_worker_alive()`, and `LEGACY_PROGRESS_TARGET`.

  `cancel_dataset_progress()` keeps its name and contract and now cancels
  exactly the active loading tasks. Its `targets` list no longer contains the
  string `"dataset_progress"` — every entry is a real task id — so a client
  that special-cased that value can drop the branch. Cancellation also stopped
  needing a `reset_cancel()` guard on each new load: a per-task flag starts
  clear, and a cancel aimed at an earlier load stays with it.

  **For plugin authors:** if you read `dataset_progress` directly or called
  `get_progress()`, switch to `loading_tasks` — `list_tasks()` for a snapshot
  of every operation, `get_tracker(task_id)` for one. If you polled
  `check_dataset_cancelled()`, call `check_cancelled()` on the tracker your
  operation owns; reporting progress through the thread's sink usually does it
  for you, since the callbacks the load pipeline binds check cancellation
  before recording each tick.

### Fixed

- **A matrix built for one embedding space can no longer be filled with
  another space's vectors** (issue #3650). `scoreable_snapshot()`,
  `get_embedding_matrix()` and `get_embedding_matrix_for_snap()` decide whether
  an explicitly named embedder *is* the snapshot's primary - and so may reuse
  the cached primary path, which reads whatever vector each media happens to
  carry. That test sampled **one** media. On a homogeneous snapshot that is
  right and is the single-embedder optimisation it was written for; on a
  mixed-type snapshot it was a sampling error, because media #1's primary
  picked the path for all N.

  Ask for space `A` on a snapshot whose first media is an `A` media and whose
  rest are `B` media, and every media contributed its own vector: `B`-space
  rows stacked into an `A`-space matrix and scored through an `A`-space head.
  Nothing raised - the only guard was the width check, so this was reachable
  whenever the two spaces share a dimension, which 512-d and 768-d encoders
  routinely do. Reordering the same snapshot flipped the answer between "all N
  rows, some of them wrong" and "only the `A` rows".

  The collapse now requires **every** media to share that primary (a
  short-circuiting scan, memoised per `media_revision` on `DatasetContext` so
  the cached hot path stays O(1)). A snapshot whose medias disagree keeps the
  name and takes the named path, which reads each media's vector *in the
  requested space* - so `scoreable_snapshot()` drops the media that have none
  and `get_embedding_matrix*()` raises on them, per their existing contracts.
  Homogeneous snapshots are byte-for-byte unchanged.

  `scoreable_snapshot()` additionally logs one `WARNING` per call naming the
  requested embedder, the spaces the dropped media live in, and how many were
  dropped. Dropping stays the policy - a mixed dataset scored for one space
  *should* leave out the media that live in another - but a silently short
  haystack is what hid this.

### Added

- **`ProgressCallback`, `noop_progress()` and `resolve_progress_callback()` are
  exported from `vtscore.concurrency.progress`** (issue #3392). The
  `(status, message, current, total)` type alias had been redeclared in ten
  modules and the "use the thread's callback, else a default" resolution copied
  byte-identically into eight; both now have one definition.
  `resolve_progress_callback()` returns the calling thread's progress callback,
  or `noop_progress` when none is bound. `vtscore.media.base` re-exports the
  alias and the no-op, so existing imports from there keep working.

### Changed

- **`update_progress()` reports into the calling thread's tracker instead of a
  global one** (issue #3376). It stays public and stays the documented way for
  an importer to report progress without accepting an `on_progress` argument —
  it is now the free-function spelling of `resolve_progress_callback()`.

  This **fixes** out-of-tree importers that followed
  `docs/extending/dataset-importers.md`. Writing straight to the global meant
  their ticks landed on a channel the dashboard does not render, so the row for
  their import never moved, and the `"embedding"` status that swaps the
  download concurrency-gate for the embed gate never reached the pipeline.
  Both work now with no change to the calling code. Two in-tree importers
  (`recaller`, and `DatasetImporter`'s default record loop) had the same bug.

  Two narrower changes to its signature: the `staging_result=` parameter is
  gone (that field belongs to the staging task's own tracker, declared via
  `create_task(..., extra_fields={"staging_result": None})`), and the remaining
  extras (`error`, `step`, `total_steps`) are forwarded only when the bound
  sink's signature accepts them — the four-argument callbacks the load pipeline
  installs get the four positional arguments alone. Acceptance is decided by
  inspecting the signature, not by catching `TypeError` from the call, so a
  `TypeError` raised *inside* a sink still propagates.

- **`description_wrappers` is now a per-embedder, measured choice, and four
  built-in embedders return `[]`** (issue #3341, following #3127). Issue
  #3127 measured wrapper-averaged text embedding on/off across 22 eval
  datasets and 560 paired queries (paired Δ in text-sort average precision,
  SEs clustered on (corpus, category)). The result splits by *model*, not by
  media type: `clap_general` +0.014 ± 0.009 and `xclip` +0.008 ± 0.014, but
  `siglip` −0.001 ± 0.002, `clap` −0.010 ± 0.008, `e5` −0.057 ± 0.009 and
  `bge` −0.059 ± 0.009 (the two text embedders lost on 45 of 45 categories,
  and every individual template was negative on all four). Those four now
  override `description_wrappers` to return `[]`, so `embed_text_enriched`
  degrades to `embed_text` and the app's `enrich_descriptions` setting is a
  no-op for them. `clap_general`, `xclip` and the unmeasured cross-modal
  siblings are unchanged.

  **For plugin authors:** the base-class default is still `[]` and nothing in
  the ABC moved, so no out-of-tree embedder is affected. But the default is
  now documented as a *measured answer* rather than an unfilled slot — do not
  add wrappers to your own embedder without measuring that they beat the typed
  query on your checkpoint, because a sibling model's templates are not
  evidence (`clap` and `clap_general` disagree on the identical five).

- **Voted media are excluded from the calibrated threshold's haystacks**
  (issue #3308). The fold-anchored threshold estimator drops the voted
  items from every population sample it touches - each calibration fold
  model's corpus scores and the final model's realization sample - because
  those models were trained on the votes, so the votes' own scores under
  them are optimistically shifted (and the calibration votes previously sat
  in the haystack twice: once as free points, once as anchors). All the
  distributions in the quantile transfer now cover one identical
  population, the unlabeled remainder. New optional `voted_ids` parameter
  on `vtscore.detectors.training.train_and_threshold` (and the internal
  `_train_and_score_xy` / `_fused_threshold`); `train_and_score` and the
  labelset/model-loading pipelines pass it automatically, and omitting it
  keeps the historical include-everything behaviour. The exclusion switches
  off entirely when it would leave fewer than
  `vtscore.training.thresholds.EXCLUSION_MIN_REMAINDER` (60) scores - a
  drained remainder is too coarse and too selection-biased to be a
  population estimate. Thresholds move only where votes are a nontrivial
  share of the corpus (small datasets); on large corpora the change is
  bounded by the votes' share of the ≤50k haystack sample.

- **`calibration_fraction` defaults are now per-embedder, and `None` means
  "resolve it".** The shipped Train/Calibrate split of each calibration fold
  is keyed on the space the detector learns in (issue #3287):
  `PRODUCTION_SPLIT_BY_SPACE` in `vtscore.training.thresholds` maps
  `single_vector` → 0.3 and `patch` → 0.5, with `PRODUCTION_SPLIT = 0.5` as
  the unknown-space fallback, resolved by `production_split_for(patch_space=…)`
  (a three-state contract mirroring `production_schedule_for`) and, with the
  explicit-setting precedence, by
  `vtscore.detectors.training.resolve_calibration_fraction`. Accordingly the
  `calibration_fraction` parameters of `train_and_score`,
  `labelset_train_and_score`, `train_detector_from_origins`,
  `simulate_voting_iterations`, `run_voting_iterations_eval[_from_pickles]`,
  `eval_learned_sort`, and `run_eval` changed from `float = 0.5` to
  `float | None = None`, where `None` resolves to the per-space production
  default (the eval keys it on `patch_grid` presence, matching the app).
  `CoreConfig.calibration_fraction` is now `float | None` (`None` = no
  explicit user setting), and `vtscore.state.get_calibration_fraction()` /
  `set_calibration_fraction()` pass that tri-state through. Callers that
  passed an explicit fraction are unaffected; callers that *relied* on the
  implicit 0.5 default and want it back should pass `0.5` explicitly.

- **Train/Calibrate split sizes are dithered rather than rounded.**
  `calibration_folds` / `calibration_folds_cached` (and the grouped,
  bag-aware path behind them) now round a fractional split size up with
  probability equal to its fractional part, instead of calling `round`.
  The count is unbiased either way; what changes is that a *tie* no longer
  resolves the same way for every labelset of a given size. `round` is
  round-half-to-even, so at the default `calibration_fraction=0.5` the odd
  label's destination alternated with the label count - Train at
  `n % 4 == 1`, Calibrate at `n % 4 == 3` - and every threshold read off
  the fold models inherited that period-4 seesaw. Harmless for one
  detector, but any study that advances one label at a time saw it
  phase-locked across every run, where it survived averaging as a
  spurious 4-label ripple (issue #3286). Thresholds remain a pure,
  reproducible function of the labelset: the tie-break RNG is seeded from
  a digest of the labels and training vectors, so the same votes still
  give the same cut, and the calibration cache stays valid.

- **Results exporters declare which payload kinds they handle, instead of
  sniffing the dict shape.** `LabelsetExporter` is now `ResultsExporter`
  (the old name stays as a permanent module-level alias), and it exposes
  `export_find_results()` and `export_labelset()` alongside the existing
  `export_cli_detectors()`. `supported_payloads` is derived from which of
  those a subclass overrides, so each picker offers an exporter only for the
  kinds it can actually read and the export route answers 400 - rather than
  letting an exporter be handed a shape it doesn't understand and deliver an
  empty export while reporting success. `email_smtp` gained a labelset mode
  it never had.

  **Existing exporters keep working with no changes.** The default
  `export_find_results()` / `export_labelset()` both delegate to `export()`,
  so a plugin written against the single-method contract still runs and still
  does its own `if "labels" in results` check; it is credited with both
  payload kinds and logs one line at import pointing at the named methods.
  Migrating is a mechanical split of that `if` into two methods, and buys
  accurate picker filtering. See
  [`vtscore/docs/extending/results-exporters.md`](docs/extending/results-exporters.md).

- **`RemoteUnreachableError` now also covers a retryable HTTP status that
  outlives the retry budget.** `download_file_with_progress` and
  `fetch_text_with_retry` used to end a run of 500/502/503/504/429 responses by
  calling `raise_for_status()`, so the caller got a raw `requests.HTTPError`
  naming whichever CDN node the redirect landed on. Both now raise
  `RemoteUnreachableError` with a sentence naming the host and the status.
  Non-retryable statuses (404 and friends) still raise `HTTPError` unchanged,
  and gated 401/403 still raise `GatedResourceError`.
- **A multi-file demo download tolerates individual files the host refuses.**
  The per-file sets (Apollo 11, the Nixon tapes) set a failed file aside, retry
  it once after the rest of the set, then skip it with a `notify()` warning,
  failing the download only when more than a quarter of the set is missing.

- **`LINEAR_SVM_HEAD` is the production detector head**, replacing
  `LINEAR_HEAD`. Both sentinels build the same `Linear(D, 1)`; the new one is
  fitted by `vtscore.training.svm.fit_linear_svm_head` (squared hinge + L2 via
  liblinear, `class_weight="balanced"`) rather than by `train_model`'s balanced
  BCE loop. The fit delegates to `train_svm(kernel="linear")` — the very call
  the eval harness scores as its `svm_linear` arm — so the shipped head and the
  measured arm cannot drift apart, and `vtscore.eval.voting_iterations`'
  `PRODUCTION_HEAD` moves to `"linear_svm"` with it. `head="linear"` and
  `head="mlp"` remain as named eval arms. Callers that passed `LINEAR_HEAD` to
  match production must now pass `LINEAR_SVM_HEAD`; scores from a detector
  trained on the same labels will differ.
- **`train_svm` accepts `sample_weight`** (`decision_sigmoid` calibration
  only), mirroring `train_model`'s contract: per-row weights replace the
  `class_weight` balance rather than stacking on it. This is how region
  flooding weights a Bad image's many region rows down to one image's worth.

### Added

- **The detector-JSON write lock and the cross-dataset labelset merge are now
  public, and re-exported from the `labelset_ops` façade** (issue #3398).
  `vtscore.detectors.label_sync._label_sync_write_lock` and
  `_merge_labelsets_across_datasets` are now
  `label_sync_write_lock` and `merge_labelsets_across_datasets`, exported from
  `vtscore.detectors.labelset_ops` alongside the rest of the detector-labelset
  surface. Neither is new behaviour — the lock's contract has always bound
  out-of-module writers (any caller doing its own read-modify-write of a
  detector JSON file must hold it for the whole pass, acquired *before*
  `_state_lock`), and `sync_labels_to_loaded_detector` and the app's four route
  writers all merge with the same function. The private names simply said
  otherwise, which made the seam unenforceable and the symbols un-renameable.
  Both old names are gone rather than aliased, per this repo's rule that
  `_`-prefixed symbols carry no compatibility promise; an out-of-tree writer
  that was reaching for them should import from `labelset_ops`.

- **`vtscore.utils.hits.hit_custom_metadata(media)`** — the `custom_metadata`
  sanitiser `build_media_hit` already applied is now a public export (issue
  #3368). It returns a fresh copy of a media's importer metadata with the
  `embedding` key stripped, and exists as a public name because every surface
  that hands that dict to an outside caller — hits, the app's detector and
  processor scoring routes, `POST /api/medias/batch`, the label-export blob —
  has to agree on what it strips. A top-level key filter cannot: a
  pre-computed vector shipped via `custom_metadata_map` rides *inside*
  `custom_metadata`, where it would balloon the payload or fail JSON encoding
  outright. Behaviour of `build_media_hit` is unchanged.

- **`PluginBase.get_field_options(field_key, current_values)`** — the
  dynamic-select hook now lives on the shared plugin base instead of on the
  importer families alone, so every plugin family inherits it and a results
  exporter can compute its options at runtime (issue #3360). The default
  raises `NotImplementedError` naming the field, matching what the importer
  bases already did; existing overrides are unaffected. A dynamic select's
  declared `options` are also no longer used as the `choices` of its
  auto-generated CLI flag — they are a seed for the first render, not the
  allowed set, so pinning argparse to them rejected every value the plugin
  resolved at runtime.

- `vtscore.eval.seed_scores.build_seed_scores` accepts an `enrich` keyword
  (default `False`, matching the app's shipped `enrich_descriptions`
  default), so a simulated-user study can open on the same sort a real user
  sees instead of silently always embedding the query plainly (issue #3341).

- **`vtscore.concurrency.notifications`** - `notify()`, `Notification` and
  `NotificationBroker`: a fan-out channel for one-off user-facing messages.
  The progress trackers publish *state* and an exception ends the operation;
  neither fits "we skipped 3 unreadable files but the other 900 imported
  fine". `notify()` is the third option — keep going, and say so. Consumers
  subscribe to the process-wide `notifications` broker (the app's SSE stream
  and the CLI printer both do); with no subscriber the message is still
  logged at a severity matching its level. `PluginBase.notify()` wraps it
  with the plugin's `display_name` as the source. The call never raises: bad
  levels degrade, long messages truncate, broken subscribers are swallowed.

- **`vtscore.security.login`** - the `LoginProvider` ABC,
  `DefaultLoginProvider`, the process-wide `set_login_provider` /
  `get_login_provider` registry, `get_user_data_dir()` and
  `is_safe_username()`, moved down from the `vtsearch` app tier. Path
  confinement (`vtscore.security.path_validation`) asks the active provider
  where the current user's data lives, so the abstraction had to be reachable
  in a process with no Flask in it — previously `get_file_access_base_dir()`
  raised `ImportError` there. An embedder can now opt into per-user
  confinement by registering a provider whose `get_user_data_dir()` returns a
  per-user subtree; the default stays single-user and unconfined.

- **`beats` audio embedder** - Microsoft's BEATs iter3+ AudioSet-2M
  self-supervised encoder, exposed as a 768-d audio-only embedder
  (`supports_text=False`). There is no `transformers` implementation, so the
  architecture is vendored in `vtscore.media.audio._beats_model` (DeepNorm
  residuals, a shared gated relative position bias, a weight-normalised
  convolutional position embedding) and the released MIT-licensed checkpoint
  is loaded onto it. Its Kaldi `compute-fbank-feats` front-end is ported from
  torchaudio's pure-PyTorch implementation rather than taken as a dependency,
  since torchaudio's wheels are built against a pinned torch.

### Removed

- **The four never-implemented integration plugins** (issue #3451): the
  `holder` results exporter (`vtscore.exporters.holder`), the `holder` label
  importer (`vtscore.labels.importers.holder`), the `recaller` dataset
  importer (`vtscore.datasets.importers.recaller`), and the `pullwrest` media
  source (`vtscore.datasets.sources.pullwrest`). Every I/O entry point in all
  four raised `NotImplementedError("TODO: implement ...")`, so no code path
  through them had ever executed; they were registered but
  `hidden_from_picker`. **Breaking:** `get_exporter("holder")`,
  `get_label_importer("holder")` and `get_importer("recaller")` now return
  `None`, `get_source_for_origin({"importer": "recaller", ...})` returns
  `None`, and the modules no longer import. Their API routes
  (`/api/dataset/import/recaller`, `/api/dataset/stage-import/recaller`,
  `/api/label-importers/import/holder`,
  `/api/detectors/{name}/import-labels/holder`,
  `/api/detectors/registry/from-labelset/holder`) are gone with them. If you
  copied one as a starting point, the shapes they demonstrated are now
  documented directly: service-style dataset importers and the bulk-fetch
  hook in `docs/EXTENDING-plugins.md` and
  [`extending/dataset-importers.md`](docs/extending/dataset-importers.md),
  labelset-only exporters in
  [`extending/results-exporters.md`](docs/extending/results-exporters.md).

### Fixed

- **A pre-computed vector nested in `custom_metadata` no longer reaches a
  labelset** (issue #3368). `LabelSet.from_clips_and_votes` copied a media's
  `custom_metadata` onto every `LabeledElement` verbatim, so a dataset
  imported through `custom_metadata_map`'s vector channel wrote a numpy array
  into the detector JSON - a hard `json.dump` failure, and the vector
  persistence the no-persisted-vectors rule forbids. The dict now goes through
  `hit_custom_metadata` (see Added above); `LabeledElement.metadata` is `None` when
  the vector was its only key. No other importer metadata is affected.

### Changed

- **`clap_general` is the default audio embedder**, replacing `clap`.
  Measured on the full ESC-50 (2000 clips, all 50 categories),
  `laion/larger_clap_general` wins every comparison against
  `laion/clap-htsat-unfused`: text-sort mAP 0.869-0.895 vs 0.850-0.866,
  learned-sort mean F1 0.523-0.564 vs 0.457-0.529, and leave-one-out 1-NN
  accuracy 0.973 vs 0.958. `embedders_for_type("audio")[0]` and any caller
  that passes `embedder=""`/`None` now resolve to `clap_general`.
  `clap` is **not** removed - it stays a first-class explicit choice, ~2.1x
  faster and ~20% smaller, and existing pickles and detector JSONs recording
  `embedder: "clap"` keep resolving. Both are 512-d, so vectors from the two
  are dimension-compatible but not interchangeable.
- **The two general CLAP display names are now distinguishable.** `clap` reads
  "CLAP (general, faster)" (was "CLAP (general audio)") and `clap_general`
  reads "CLAP (general, larger)" (was "CLAP (general 2024)"); their progress
  labels are "CLAP Fast" and "CLAP General".
- **`fold_anchored_gmm_threshold` is the shipped decision threshold.** Per
  calibration fold, a semi-supervised 2-component mixture is fitted to that
  fold model's scores over the whole collection with the fold's *held-out*
  labels clamped to their component; each fold's cut is carried to the final
  model as a quantile and the folds are combined in quantile space. Anchor mass
  is 0.3 (each vote counts as three tenths of a haystack point) and the cut
  rule is `mid_tilt`: at Inclusion 0 the midpoint between the fitted component
  means — the interior optimum of a six-environment κ sweep — and away from 0
  that midpoint's combined quantile shifted by the rate-optimal cut's own
  displacement from its inclusion-0 position, so the fused threshold answers
  the Inclusion knob monotonically while reproducing the measured midpoint arm
  exactly where it was measured. The
  label-count-scheduled blend (`calculate_safe_threshold`) is now only the
  fallback for label sets too small to form calibration folds.
- **`gmm_cut_from_fit(rule="rate")` continues past the component means instead
  of falling back to the midpoint** when the density crossing has no root
  between them. The cut is read as the highest score at which the low component
  still out-densities the high one under the cost tilt, which makes it monotone
  in the Inclusion knob (and so keeps the included sets nested); it equals the
  old root wherever a crossing exists, including at every equal-weight cut.
  Once the crossing runs off the inter-mean interval the cut keeps moving,
  continuing past the edge by the log-cost excess times the mixture-weighted
  variance over the mean gap - the equal-variance crossing's own slope, so for
  equal-variance fits the continuation extends the interior crossing line
  seamlessly. Returning the bare edge there (the first form of this change)
  made the cut *constant* in the cost ratio, which flattened the composed
  `mid_tilt` quantile over whole bands of the Inclusion knob and silently
  collapsed the acquisition offset to a no-op inside them.
- **`vtscore.eval` defaults `safe_thresholds=True`**, matching the app; `False`
  is the no-fusion control arm. `eval_learned_sort` / `run_eval` lost the
  parameter entirely - they delegate to the production trainer, which has no
  such mode.
- **`gmm_cut_from_fit` returns `(cut, kind)` instead of `(cut, flag)`**, where
  *kind* is one of `CUT_KIND_INTERIOR` (`""`), `CUT_KIND_CONTINUED` or
  `CUT_KIND_DEGENERATE_MIDPOINT`. It is empty exactly when the old flag was 0,
  so `bool(kind)` is the previous "no interior stationary point" boolean; the
  non-empty values distinguish a cut *continued* past a component mean (still
  moving with the cost tilt, still the rate rule) from a fit too degenerate to
  express a boundary at all (a midpoint, constant in the tilt). Those two were
  indistinguishable before, which made a fallback countable but not
  attributable. **Breaking:** a caller comparing the second element to `0`/`1`
  should compare to `""` or wrap in `bool()`.

### Removed

- **`CoreConfig.safe_thresholds`**, `vtscore.state.get_safe_thresholds` /
  `set_safe_thresholds`, and the `safe_thresholds` parameter of
  `train_and_score`, `labelset_train_and_score`, and `run_learned_sort`. The
  fused threshold measured better than the alternative at every label count, so
  the switch was deleted rather than kept as a way to opt into a worse cut.
  **Breaking:** construct `CoreConfig` without the field, and drop the keyword
  from any `train_and_score` call.
- **`vtscore.training.thresholds.xcal_is_discarded`** - it existed to skip the
  fold training where the blend schedule zeroed the cross-cal cut, and the
  fold-anchored estimator needs those fold models at every label count.
- **`cross_calibration_threshold_cached`** - superseded by
  `calibration_folds_cached`, which returns the fold models alongside the
  orderings (plus `threshold_from_folds` for the cross-calibration cut).

### Added

- **`vtscore.state.current_user`** - Flask-free resolution of "who is this
  work for": a pluggable request-user resolver
  (`register_request_user_resolver`), the `thread_user` thread-local scope
  background jobs use to inherit a requester's identity, and the `"default"`
  fallback. Library code that needs a username now calls
  `vtscore.state.current_user.get_current_user()` instead of importing
  `vtsearch.auth`, which made `JobManager.start()` (and label sync, dataset
  loading, exporters, plugin templating) hard-require Flask at call time.
  `vtsearch.auth` re-exports every name, so there is still exactly one
  thread-local behind the app.
- **`FoldAnchoredCut`** / **`fit_fold_anchored_cut`** - the fitted estimator,
  split from the cut so a new Inclusion value can be re-cut arithmetically
  without refitting or re-scoring.
- **`inclusion_cost_weights`** - the single definition of what an Inclusion
  value costs in `(fpr_weight, fnr_weight)`, read by both the shipped threshold
  rule and the eval harness.

- **`LabelsetExporter.opens_url`** and the `"open_url"` response key - an
  exporter can return an `http(s)` URL for the frontend to open in a new
  browser tab, which is how a third-party site with no ingest API receives a
  labelset. Setting `opens_url = True` advertises it on `to_dict()` so the UI
  can label the button before the export runs.
- **`vtscore.security.url_validation.validate_browser_url`** - scheme
  allowlist for URLs the *user's browser* opens. Deliberately not the
  `validate_url` SSRF guard: no server-side request is made, so private hosts
  are legitimate targets and only the scheme is dangerous.
- **`open_url` exporter** - formats the labelset into a user-supplied URL
  template (`{ids}`, `{count}`), URL-encoding the joined identifiers,
  truncating to `max_items`, and refusing a URL past the ~2000-character
  practical limit.

## [0.1.0] - Initial release

The library was carved out of the `vtsearch` monolith and shipped as a
separate package. The 0.1.0 release captures that work as the first
publishable snapshot. See [`docs/architecture.md`](docs/architecture.md)
for the seven seams the refactor cut between vtscore and vtsearch.

### Added

- **`vtscore.config.CoreConfig`** - dataclass for the knobs library code reads
  (`safe_thresholds`, `calibrate_count`, `calibration_fraction`,
  `enrich_descriptions`, `data_dir`, `saved_datasets_dir`, `detectors_dir`,
  `autopilot_goal_diversity`, `max_concurrent_dataset_downloads`,
  `max_concurrent_dataset_embeddings`, `inclusion`). `CoreConfig.from_settings()`
  is the app-side bridge; library-only consumers construct `CoreConfig`
  directly.
- **`vtscore.datasets`** - `Origin`, `LabeledElement`, `LabelSet`,
  `DatasetImporter`, folder / pickle / demo dataset loaders, the
  `MediaSource` abstraction (local_folder, http_archive, pullwrest), the
  `IMPORTER`-sentinel auto-discovery scanner, and bidirectional split / dedup
  helpers.
- **`vtscore.media`** - `MediaType`, `MediaEmbedder`, `MediaClipper`,
  `Processor` / `Detector` / `Localizer` / `Extractor` ABCs, the audio /
  image / text / video / document plugins, and `MediaResponse` (framework-
  agnostic HTTP response wrapper).
- **`vtscore.embedding`** - Lazy embedder loader (LAION-CLAP, SigLIP, X-CLIP,
  E5, DINOv2/v3, BGE, EUPE, LanguageBind), torch device selector, smart
  preload scheduler, cached `(N, D)` embedding matrix.
- **`vtscore.training`** - Generic learned-sort primitives: MLP build / train /
  weight-serialise, GMM and cross-calibration threshold solvers, safe-
  threshold blending, region-similarity scoring, SVM prototype.
- **`vtscore.detectors`** - Detector registry, JSON-backed store, vote-aware
  online training (`train_and_score`), origin resolver, label-sync,
  cross-dataset label restoration, labelset materialisation, and the
  labeling-session analyzer (`analyze_labeling_progress` + helpers).
- **`vtscore.eval`** - Offline text-sort / learned-sort evaluation, voting-
  iteration simulator, metric dataclasses (`QueryMetrics`,
  `LearnedSortMetrics`, `DatasetResult`), and `format_results_json`.
- **`vtscore.converters`** - `MediaConverter` ABC and the seven built-in
  converters: audio↔image / text, video→audio / image, document→image / text,
  image→text.
- **`vtscore.exporters`** - `LabelsetExporter` ABC and built-ins
  (`server_json_file`, `server_csv_file`, `webhook`, `email_smtp`, `gui`,
  `holder`).
- **`vtscore.labels`** - `LabelImporter` + `LabelsetSource` ABCs, registries,
  and bidirectional sync helpers.
- **`vtscore.plugins`** - `PluginRegistry` with sentinel-based discovery,
  eager construction by default, and `importlib.metadata` entry-point
  support so third-party packages can register plugins without
  monkey-patching.
- **`vtscore.concurrency`** - `AsyncJob` / `JobManager`,
  `cap_workers_by_memory`, the long-running-operation progress trackers
  (`ProgressTracker`, `LoadingTasksTracker`, dataset / sort / eval / find
  variants), and the per-thread progress hook (`set_thread_progress`).
- **`vtscore.state`** - Per-dataset `DatasetContext` and per-detector
  `DetectorContext`, context registries, pluggable
  `register_*_context_resolver()` hooks for app integration,
  `with_dataset_context` / `with_detector_context` thread-local bindings.
- **`vtscore.sync`** - `SyncSource[L,S]` ABC shared by labelset sources
  (library) and settings sources (app-side).
- **`vtscore.security`** - Path validation (`validate_server_filepath`,
  symlink-aware globbing), SSRF guard (`validate_url`), allowlist-based
  `safe_pickle_load` + `peek_pickle_dataset_summary`.
- **`vtscore.utils`** - `build_media_hit` (the canonical scored-media hit
  dict) and offline synthetic-media generators
  (`generate_audio_dataset` / `generate_image_dataset` /
  `generate_video_dataset`).
- **`vtscore.cli`** - Flask-free CLI entry points: `autodetect_main`,
  `autodetect_importer_main`, plus chunked variants. Reads `CoreConfig`,
  not `vtsearch.settings`.

### Architecture invariants

- **No Flask imports** anywhere under `vtscore/`. Verified by the
  `./run-tests.sh vtscore-clean` mode (installs a meta-path import hook that
  refuses `flask` / `werkzeug` / `flask_smorest` before collection).
- **No `vtsearch.settings` imports** in library-candidate modules.
  Configuration arrives via `CoreConfig` or a context object.
- **No hardcoded `data/` paths.** Every reference routes through
  `vtscore.config.DATA_DIR` (honouring `$VTSEARCH_DATA_DIR`), which is
  snapshotted into `CoreConfig.data_dir`.
- **No persisted embeddings or MLP weights.** Origins are the canonical
  persisted form; the library re-derives `origin → file → embedding → MLP`
  on demand. Detector JSON files store only `LabeledElement`s; dataset
  pickles are the one sanctioned vector store.

[Unreleased]: https://github.com/samggreenberg/vtsearch/compare/vtscore-0.1.0...HEAD
[0.1.0]: https://github.com/samggreenberg/vtsearch/releases/tag/vtscore-0.1.0
