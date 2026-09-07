"""Model loading and initialisation - delegates to the media type registry.

All embedding models are now owned by their respective
:class:`~vtscore.media.base.MediaType` instances and are loaded **lazily**
on first use (the first call to ``embed_media``, ``embed_text``, or a getter
function such as ``get_clap_model``).

This module keeps its original public API (``initialize_models``,
``get_clap_model``, etc.) as thin wrappers so that existing callers continue
to work unchanged.
"""

import gc
import logging
import os
import re
import sys

_TIME_SUFFIX_RE = re.compile(r"\s*\(\d+s(?:,\s*\d+\s+modules)?\)$")

#: Fixed column width for the label printed before a console progress bar.
#: Padding every label to this width makes the ``[####…]`` bars line up
#: vertically.  Sized to fit the longest common preload label
#: ("Importing sentence_transformers…", 32 chars).  Labels longer than this
#: are truncated with an ellipsis (see :func:`_fit_label`) so the bar's open
#: bracket always lands in the same column and the bars stay aligned.
_LABEL_WIDTH = 32

#: 24-bit ("truecolor") RGB endpoints for the bar-fill gradient, interpolated
#: continuously by :func:`_bar_rgb`: reddest at 0%, yellowest at 50%, greenest
#: at 100%.  Only the fill (the ``#`` characters already drawn) is colored;
#: the ``.`` characters marking the not-yet-filled remainder keep the
#: terminal's default text color, and so does the rest of the line.
#: ``_ANSI_RESET`` restores the default color after the fill.
_RGB_RED = (205, 0, 0)
_RGB_YELLOW = (255, 215, 0)
_RGB_GREEN = (0, 205, 0)
_ANSI_RESET = "\033[0m"


def _bar_rgb(pct: int) -> tuple[int, int, int]:
    """Return the interpolated ``(r, g, b)`` fill color for a bar at *pct* (0-100).

    Linearly interpolates red -> yellow over ``[0, 50]`` and yellow -> green
    over ``[50, 100]``, so every percentage gets a distinct shade rather than
    snapping between three fixed colors.
    """
    pct = max(0, min(100, pct))
    if pct <= 50:
        start, end, t = _RGB_RED, _RGB_YELLOW, pct / 50
    else:
        start, end, t = _RGB_YELLOW, _RGB_GREEN, (pct - 50) / 50
    return tuple(round(start[i] + (end[i] - start[i]) * t) for i in range(3))  # type: ignore[return-value]


def _bar_color(pct: int) -> str:
    """Return the 24-bit ANSI foreground escape for a bar at *pct* percent (0-100)."""
    r, g, b = _bar_rgb(pct)
    return f"\033[38;2;{r};{g};{b}m"


def _strip_time_suffix(msg: str) -> str:
    return _TIME_SUFFIX_RE.sub("", msg) if msg else msg


def _fit_label(base: str) -> str:
    """Fit *base* to exactly :data:`_LABEL_WIDTH` columns.

    Short labels are left-padded with spaces; labels longer than the width are
    truncated and given a trailing ``…``.  Either way the result is always
    ``_LABEL_WIDTH`` columns wide, so every bar's open bracket lands in the same
    column and successive bars line up vertically regardless of label length.
    """
    if len(base) > _LABEL_WIDTH:
        return base[: _LABEL_WIDTH - 1] + "…"
    return f"{base:<{_LABEL_WIDTH}}"


from vtscore.config import MODELS_CACHE_DIR, resolve_device
from vtscore.media.base import ProgressCallback
from vtscore.media.torch_setup import ensure_torch_configured
from vtscore.utils.import_metadata import seed_packages_distributions

logger = logging.getLogger(__name__)


def get_torch_device():
    """Return the preferred ``torch.device`` for MLP training / scoring.

    Resolves :data:`vtscore.config.DEVICE` (``VTSEARCH_DEVICE``, default
    ``"auto"``) to a concrete device - ``cuda`` when available, ``mps`` on
    Apple silicon, or ``cpu``.  Imports torch lazily.
    """
    import torch  # noqa: PLC0415

    return torch.device(resolve_device())


# Concurrency defaults are clamped into this range, matching the
# ``_clamp(1, 16)`` on the settings fields they feed (see
# ``vtsearch/settings_models.py``). Bounding here means neither the env
# override nor the hardware probe can hand the settings layer a value it would
# only reject.
_CONCURRENCY_MIN = 1
_CONCURRENCY_MAX = 16


def _env_concurrency_override(var: str) -> int | None:
    """Explicit concurrency override read from environment variable *var*.

    Lets a launcher pin concurrency *without* writing ``data/settings.json`` -
    which would otherwise freeze the value at first-run hardware and defeat the
    per-startup autodetect. The HLTCOE Grid launcher uses this to give a fat
    single-GPU SLURM node more parallelism than the conservative auto caps,
    while a laptop (var unset) keeps its hardware default - so ``python app.py``
    stays the launch command on both.

    Returns the clamped override, or ``None`` when the var is unset/blank so the
    hardware probe runs. A non-integer value also returns ``None`` (logged once,
    never fatal - a launcher typo must not block startup); an out-of-range
    integer is clamped into ``[_CONCURRENCY_MIN, _CONCURRENCY_MAX]``.
    """
    raw = os.environ.get(var)
    if raw is None or not raw.strip():
        return None
    try:
        value = int(raw.strip())
    except ValueError:
        logger.warning("Ignoring non-integer %s=%r; falling back to hardware autodetect", var, raw)
        return None
    clamped = max(_CONCURRENCY_MIN, min(_CONCURRENCY_MAX, value))
    if clamped != value:
        logger.warning("Clamped %s=%d into [%d, %d] -> %d", var, value, _CONCURRENCY_MIN, _CONCURRENCY_MAX, clamped)
    return clamped


def default_concurrent_downloads() -> int:
    """Default for ``max_concurrent_dataset_downloads``.

    Honours ``VTSEARCH_MAX_CONCURRENT_DOWNLOADS`` when set; otherwise derives
    from hardware. The download phase is bandwidth- and disk-bound. Allowing a
    handful of parallel downloads usually saturates a home connection without
    thrashing the disk; capped at 4 to keep memory and FD pressure reasonable on
    small boxes.
    """
    override = _env_concurrency_override("VTSEARCH_MAX_CONCURRENT_DOWNLOADS")
    if override is not None:
        return override
    return max(1, min(4, os.cpu_count() or 1))


# A single CPU embed job holds an embedder model plus an N x D fp32 working
# set; budgeting ~4 GiB of total RAM per concurrent job keeps memory-starved
# boxes at one worker while letting roomy workstations run a few in parallel.
# (Only consulted on CPU hosts; on an accelerator the embed default is 1 - see
# ``default_concurrent_embeddings``.)
_RAM_BYTES_PER_CPU_EMBED = 4 * 1024 * 1024 * 1024

# Cores budgeted per concurrent embed job on a CPU host.
_CPUS_PER_CPU_EMBED = 4

# Upper bound on the auto-derived embed default regardless of how big the box
# is. The ``VTSEARCH_MAX_CONCURRENT_EMBEDDINGS`` env override (and a hand edit
# in ``data/settings.json``) can still go higher, up to the settings clamp of
# 16 - that env override is the lever for fat cluster nodes that want more
# parallelism than this conservative auto cap.
_MAX_CPU_EMBED_DEFAULT = 4


def _total_memory_bytes() -> int:
    """Best-effort total physical RAM in bytes, or ``0`` if it can't be read.

    Uses ``MemTotal`` (Linux ``/proc/meminfo``) with an ``SC_PHYS_PAGES``
    sysconf fallback. Total (not *available*) RAM is the right signal for a
    startup default: it's stable, whereas free memory swings with whatever
    else happens to be running when the setting is first resolved.
    """
    try:
        with open("/proc/meminfo", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page_size > 0:
            return pages * page_size
    except (OSError, ValueError):
        pass
    return 0


def default_concurrent_embeddings() -> int:
    """Default for ``max_concurrent_dataset_embeddings``.

    Honours ``VTSEARCH_MAX_CONCURRENT_EMBEDDINGS`` when set; otherwise derives
    from hardware.

    **On an accelerator the default is 1.** Embedders are now device-aware (see
    ``to_compute_device`` / ``resolve_device``), so on a CUDA/MPS host they share
    a single GPU. Running several embed jobs at once would multiply resident model
    weights on that one device and court OOM with no throughput win - the global
    ``_embed_lock`` serialises forward passes on the device anyway. Power users
    with VRAM headroom (or genuinely multi-GPU nodes) raise it via the env
    override.

    **On a CPU host the embed phase is CPU- and RAM-bound**, so the default
    scales with the scarcer of cores and RAM and ignores GPU count: roughly one
    job per :data:`_CPUS_PER_CPU_EMBED` cores and one per
    ``_RAM_BYTES_PER_CPU_EMBED`` of total RAM, whichever is smaller, capped at
    :data:`_MAX_CPU_EMBED_DEFAULT`. Constrained machines (few cores or little
    RAM) resolve to 1 - preserving the old fully-serial behaviour where a second
    concurrent embed would thrash or OOM - while workstations and fat cluster
    nodes get genuine parallel embedding with no config change. When total RAM
    can't be read we fall back to 1 rather than guess generously. Set
    ``VTSEARCH_MAX_CONCURRENT_EMBEDDINGS`` past the auto cap to go further.
    """
    override = _env_concurrency_override("VTSEARCH_MAX_CONCURRENT_EMBEDDINGS")
    if override is not None:
        return override

    # One embed job per GPU: the device is shared and forward passes serialise on
    # it, so extra jobs only add VRAM pressure. resolve_device() returns "cpu"
    # when torch is missing or no usable accelerator is present.
    from vtscore.config import resolve_device  # noqa: PLC0415

    if resolve_device() != "cpu":
        return 1

    by_cpu = (os.cpu_count() or 1) // _CPUS_PER_CPU_EMBED
    total_ram = _total_memory_bytes()
    by_ram = total_ram // _RAM_BYTES_PER_CPU_EMBED if total_ram else 1
    return max(1, min(_MAX_CPU_EMBED_DEFAULT, by_cpu, by_ram))


def _warm_threadpool_controller() -> None:
    """Build sklearn's cached ``ThreadpoolController`` while single-threaded.

    The first sklearn fit (e.g. KMeans in coverage-atlas building) constructs
    a ``threadpoolctl.ThreadpoolController``, which scans every loaded shared
    library via ``dl_iterate_phdr``.  That C call holds glibc's loader lock
    while invoking a *Python* callback per library - and running Python means
    acquiring the GIL.  If another thread holds the GIL and is itself inside
    native code that needs the loader lock (libgcc's ``_Unwind_Find_FDE``
    during stack unwinding does ``dl_iterate_phdr`` too, and numpy-heavy
    embedding work hits it), the two threads deadlock: loader lock <-> GIL.
    Observed in production as a frozen server when a detector load and a
    dataset load ran concurrently.

    sklearn caches the controller in a module global
    (``sklearn.utils.parallel._get_threadpool_controller``), so building it
    once here - at startup, before any load threads exist - means the
    dangerous scan never runs under concurrency.  Best-effort: sklearn may be
    absent (minimal installs) and the helper is private, so fall back through
    public threadpoolctl (which at least pre-loads the library handles) and
    swallow failures rather than block startup.
    """
    try:
        from sklearn.utils.parallel import _get_threadpool_controller  # noqa: PLC0415

        _get_threadpool_controller()
    except Exception:
        try:
            import threadpoolctl  # noqa: PLC0415

            threadpoolctl.ThreadpoolController()
        except Exception:
            pass


def _install_transformers_logging_bridge() -> None:
    """Route transformers' logs through the app's handler, when the app is present.

    A module-level function rather than a closure inside
    :func:`initialize_models` so tests have a *library-tier* seam to patch.
    Patching it saves the ~0.7s ``transformers`` import in suites that stub
    every embedder; without the seam the only patchable name was the app-tier
    ``vtsearch.logging_config`` symbol, which forced ``tests_lib/`` (whose whole
    contract is "no ``vtsearch`` imports") to reach across the tier boundary.

    The ``vtsearch`` import is optional by design - see the ``embedding/loader.py``
    entry in ``tests_lib/meta/test_library_layering.py``'s allowlist - so the
    library keeps working when the app package is absent.
    """
    try:
        from vtsearch.logging_config import install_transformers_logging_bridge  # noqa: PLC0415

        install_transformers_logging_bridge()
    except Exception:
        pass


def initialize_models(on_progress: ProgressCallback | None = None) -> None:
    """Prepare the runtime environment for embedding models.

    Creates the model cache directory, configures PyTorch thread count
    **if torch is already imported**, and warms sklearn's threadpool
    controller while still single-threaded (see
    :func:`_warm_threadpool_controller`).  When torch has not been imported
    yet (e.g. during fast test startup) the thread-count configuration is
    deferred until ``ensure_torch_configured`` is called by the first code
    path that actually imports torch.

    Models themselves are **not** loaded here.

    The two heavy first-time imports this triggers - scikit-learn (pulled in
    by the threadpool warm-up) and transformers (pulled in by the logging
    bridge) - dominate the wall-clock cost and can take ~10s combined on a
    cold start, during which the process looks frozen.  Pass *on_progress* to
    render a live elapsed-time progress bar for each, matching the embedder
    preload bars printed later in startup.  When ``None`` (the default, used
    by tests and the eval CLI) the imports run silently as before.
    """
    # Before anything can import transformers: transformers builds
    # ``importlib.metadata.packages_distributions()`` at module import, and the
    # stdlib version of that stats every file recorded by every installed
    # distribution.  On an NFS venv with a cold cache that is minutes of silent
    # startup (issue #3715); the seed makes it ~224 small reads instead.
    seed_packages_distributions()

    MODELS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ensure_torch_configured()

    def _warm() -> None:
        _warm_threadpool_controller()

    if on_progress is None:
        _warm()
        _install_transformers_logging_bridge()
    else:
        from vtscore.media.embedder import IMPORT_MODULE_ESTIMATES, timed_progress  # noqa: PLC0415

        console_cb = _make_console_progress(on_progress)
        with timed_progress(
            console_cb, "loading", "Importing scikit-learn…", est_modules=IMPORT_MODULE_ESTIMATES["sklearn"]
        ):
            _warm()
        # The transformers import here is only the lightweight logging bridge,
        # not the full model classes, so it pulls in far fewer modules than an
        # embedder's ``from transformers import …``.
        with timed_progress(
            console_cb,
            "loading",
            "Importing transformers…",
            est_modules=IMPORT_MODULE_ESTIMATES["transformers_logging"],
        ):
            _install_transformers_logging_bridge()
        console_cb.flush()  # type: ignore[attr-defined]

    gc.collect()


def _make_console_progress(original_callback):
    """Wrap *original_callback* to also print status to the terminal.

    Used during startup preloading so the user sees intermediate status
    messages and download progress bars in the console while models are
    being loaded.

    Consecutive progress events sharing a base message (the same text with
    any trailing ``(Ns)`` elapsed-time suffix stripped) overwrite the same
    terminal line, so ``timed_progress`` ticker updates animate in place
    instead of stacking new lines.  The progress line is terminated with a
    newline only when a different base message arrives, a phase message
    arrives, or the caller invokes ``cb.flush()``.

    Every bar's label is fitted to a fixed width (:data:`_LABEL_WIDTH`) via
    :func:`_fit_label` - short labels padded, long labels truncated with an
    ellipsis - so successive bars line up vertically regardless of label
    length, and the elapsed-time/status suffix is rendered after the
    percentage so its changing length (``1s`` → ``10s``) never shifts the bar
    mid-progress.
    """
    _last_msg: list[str | None] = [None]
    _last_base: list[str | None] = [None]
    _on_progress_line: list[bool] = [False]

    _FULL_BAR = "#" * 30

    def _complete_bar() -> None:
        """Overwrite the current progress line with a 100% bar."""
        if _last_base[0]:
            label = _fit_label(_last_base[0])
            sys.stdout.write(f"\r    {label} [{_bar_color(100)}{_FULL_BAR}{_ANSI_RESET}] 100%\033[K")

    def _flush() -> None:
        if _on_progress_line[0]:
            _complete_bar()
            sys.stdout.write("\n")
            sys.stdout.flush()
            _on_progress_line[0] = False
            _last_base[0] = None
            _last_msg[0] = None

    def _callback(status: str, message: str = "", current: int = 0, total: int = 0) -> None:
        original_callback(status, message, current, total)

        if total > 0:
            base = _strip_time_suffix(message)
            # If we're already on a progress line for a different task, complete it
            # at 100% before starting the new bar.
            if _on_progress_line[0] and _last_base[0] is not None and _last_base[0] != base:
                _complete_bar()
                sys.stdout.write("\n")
            pct = min(100, current * 100 // total)
            filled = pct * 30 // 100
            # Color only the '#' fill characters with the continuous R->Y->G
            # gradient; reset before the '.' placeholders so the not-yet-filled
            # remainder keeps the terminal's default text color.
            bar = f"{_bar_color(pct)}{'#' * filled}{_ANSI_RESET}{'.' * (30 - filled)}"
            # Pad the base label to a fixed width so every bar starts at the
            # same column and the bars line up vertically.  The elapsed-time /
            # status suffix (e.g. "(3s, 247 modules)") goes *after* the
            # percentage, where its changing length can't shift the bar.
            suffix = message[len(base) :].strip()
            tail = f" {suffix}" if suffix else ""
            # \033[K clears from cursor to end of line so a shorter message
            # doesn't leave trailing chars from a longer prior render.
            line = f"\r    {_fit_label(base)} [{bar}] {pct:>3}%{tail}\033[K"
            sys.stdout.write(line)
            sys.stdout.flush()
            _on_progress_line[0] = True
            _last_base[0] = base
            _last_msg[0] = message
        elif message and message != _last_msg[0]:
            # New phase message - print on its own line; complete any active bar first.
            if _on_progress_line[0]:
                _complete_bar()
                sys.stdout.write("\n")
                _on_progress_line[0] = False
                _last_base[0] = None
            sys.stdout.write(f"    {message}\n")
            sys.stdout.flush()
            _last_msg[0] = message

    _callback.flush = _flush  # type: ignore[attr-defined]
    return _callback


def predict_embedders_to_preload(
    extra_media_types: list[str] | None = None,
    extra_embedders: list[str] | None = None,
) -> list[str]:
    """Predict which embedders are likely to be needed next, from active metadata.

    Walks the dataset registry and detector registry and returns the unique
    list of embedder names the user is likely to need:

    - For each registered dataset and detector: ``entry["embedder"]`` if set
      and recognised, otherwise the default embedder for
      ``entry["media_type"]``.  A set-but-unrecognised ``embedder`` (e.g.
      the embedder was renamed or removed) also falls back to the media
      type's default rather than being silently dropped - losing the
      optimisation entirely on a typo is worse than warming the wrong
      embedder, which the user can still override.
    - For every media type in *extra_media_types*: the default embedder
      for that type. Used by the solo-mediaType streamlined mode (see
      :func:`vtsearch.settings.get_effective_solo_media_type`) so the
      admin-chosen type has its default embedder warm at startup even
      when no datasets or detectors are registered yet.
    - For every embedder name in *extra_embedders*: the embedder itself
      (if it exists in the registry). Used by the solo-mediaEmbedder
      mode (``--solo-embedder TYPE=EMB``) so the CLI-pinned embedder is
      warm even when its mediaType is not in *extra_media_types*.

    Detector entries written before the ``embedder`` field existed have
    ``entry["embedder"] == ""``, so they fall through to the media type's
    default - matching the previous behaviour for unmigrated state.

    Order reflects discovery order (extras first so a solo-mode user
    isn't stuck behind unrelated dataset warmups, then datasets, then
    detectors), and is stable across runs.
    """
    from vtscore.datasets.registry import list_datasets
    from vtscore.detectors.registry import list_detectors
    from vtscore.media import all_embedders, embedders_for_type

    valid = {e.name for e in all_embedders()}

    def _default_for(media_type: str) -> str:
        if not media_type:
            return ""
        opts = embedders_for_type(media_type)
        return opts[0].name if opts else ""

    def _resolve(entry: dict) -> str:
        emb = (entry.get("embedder") or "").strip()
        if emb and emb in valid:
            return emb
        return _default_for(entry.get("media_type", "") or "")

    candidates: list[str] = []
    candidates.extend(extra_embedders or ())
    candidates.extend(_default_for(mt) for mt in (extra_media_types or ()))
    candidates.extend(_resolve(entry) for entry in list_datasets())
    candidates.extend(_resolve(entry) for entry in list_detectors())

    predictions: list[str] = []
    seen: set[str] = set()
    for name in candidates:
        if name and name in valid and name not in seen:
            seen.add(name)
            predictions.append(name)
    return predictions


def preload_predicted_embedders(
    extra_media_types: list[str] | None = None,
    extra_embedders: list[str] | None = None,
) -> list[str]:
    """Eagerly load embedding models predicted by :func:`predict_embedders_to_preload`.

    Calls :meth:`~vtscore.media.base.MediaEmbedder.load_models` on each
    predicted embedder so it is warm before the user opens the GUI.
    Prints intermediate status messages and download progress bars to
    the console while models load.

    *extra_media_types* is forwarded to :func:`predict_embedders_to_preload`
    so the caller (e.g. ``initialize_server`` honoring the
    ``--solo-media-type`` CLI fallback) can ensure a specific mediaType's
    default embedder is warm even with empty registries.

    Returns the list of embedder names that were preloaded.
    """
    from vtscore.media import get_embedder

    targets = predict_embedders_to_preload(
        extra_media_types=extra_media_types,
        extra_embedders=extra_embedders,
    )
    if not targets:
        return []

    print(f"  Predicted embedders to preload: {', '.join(targets)}", flush=True)
    preloaded: list[str] = []
    for emb_name in targets:
        try:
            emb = get_embedder(emb_name)
            print(f"  Preloading {emb_name} embedder...", flush=True)
            console_cb = _make_console_progress(emb._on_progress)
            try:
                with emb.progress_scope(console_cb):
                    emb.load_models()
            finally:
                console_cb.flush()  # type: ignore[attr-defined]
            preloaded.append(emb_name)
        except Exception as exc:
            print(f"  Warning: failed to preload {emb_name}: {exc}", flush=True)
    return preloaded


def smart_preload_in_background() -> None:
    """Kick a daemon thread that warms any predicted embedders not yet loaded.

    Idempotent: embedders whose model is already in memory are skipped.
    Failures are swallowed because this is a best-effort optimisation -
    the real load path will retry on first use.
    """
    import threading

    def _run() -> None:
        from vtscore.media import get_embedder

        for emb_name in predict_embedders_to_preload():
            try:
                emb = get_embedder(emb_name)
                if getattr(emb, "_model", None) is not None:
                    continue
                # A speculative warm-up is not an operation the user started,
                # so it gets no progress surface: unscoped it would narrate
                # itself on the app's global dataset channel and read as an
                # import that never ends (#3167).
                with emb.silent_progress():
                    emb.load_models()
            except Exception:
                pass

    threading.Thread(target=_run, name="smart-preload", daemon=True).start()


def predict_embedder_for_dataset(dataset_id: str) -> str:
    """Predict the embedder name needed by *dataset_id*.

    Mirrors the per-dataset half of :func:`predict_embedders_to_preload`
    for a single registry entry: returns ``entry["embedder"]`` if set and
    recognised, otherwise the default embedder for ``entry["media_type"]``
    (also the fallback when ``embedder`` is set but unrecognised).
    Returns ``""`` when the dataset is unknown or has no resolvable
    embedder.
    """
    from vtscore.datasets.registry import get_dataset
    from vtscore.media import all_embedders, embedders_for_type

    entry = get_dataset(dataset_id)
    if entry is None:
        return ""

    valid = {e.name for e in all_embedders()}
    emb = (entry.get("embedder") or "").strip()
    if emb and emb in valid:
        return emb
    media_type = entry.get("media_type", "") or ""
    if not media_type:
        return ""
    opts = embedders_for_type(media_type)
    return opts[0].name if opts else ""


def preload_embedder_for_dataset(dataset_id: str) -> str:
    """Warm the embedder needed by *dataset_id* in a background daemon thread.

    Used by the dashboard when the user selects a dataset row so the
    embedder is ready by the time they click Train. Idempotent: if the
    embedder is already loaded, the worker exits immediately. Returns
    the embedder name being warmed, or ``""`` when no embedder can be
    resolved for the given dataset.
    """
    import threading

    emb_name = predict_embedder_for_dataset(dataset_id)
    if not emb_name:
        return ""

    def _run() -> None:
        from vtscore.media import get_embedder

        try:
            emb = get_embedder(emb_name)
            if getattr(emb, "_model", None) is not None:
                return
            # Silent for the same reason as smart_preload_in_background: the
            # user selected a row, they did not start an import.
            with emb.silent_progress():
                emb.load_models()
        except Exception:
            pass

    threading.Thread(target=_run, name=f"preload-ds-{dataset_id[:8]}", daemon=True).start()
    return emb_name


# ---------------------------------------------------------------------------
# Backward-compatible getter functions
#
# These return the model instances held by their respective embedder objects.
# Existing callers that import these functions directly continue to work.
#
# They go through the public :meth:`MediaEmbedder.loaded_backbone` accessor
# rather than each subclass's own private helper, so an embedder that is
# reimplemented (or replaced out-of-tree) either keeps working via the ABC's
# default or overrides one documented method - instead of breaking silently
# the moment a private helper is renamed.
# ---------------------------------------------------------------------------


def get_clap_model():
    """Return ``(clap_model, clap_processor)`` from the CLAP embedder."""
    from vtscore.media import get_embedder

    return get_embedder("clap").loaded_backbone()


def get_xclip_model():
    """Return ``(xclip_model, xclip_processor)`` from the X-CLIP embedder."""
    from vtscore.media import get_embedder

    return get_embedder("xclip").loaded_backbone()


def get_e5_model():
    """Return the E5 ``SentenceTransformer`` from the E5 embedder."""
    from vtscore.media import get_embedder

    return get_embedder("e5").loaded_backbone()[0]
