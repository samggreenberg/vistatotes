"""Session bootstrap and per-test reset of ``vtscore``'s process-global state.

Both suites run thousands of tests in a handful of long-lived xdist worker
processes, so every module-level cache, registry and progress tracker in
``vtscore`` is shared between tests and has to be put back before each one.
The sequence below is the whole of that reset for the library tier; each
conftest wraps it with its own tier-specific extras (the app tier's autorun
processors, login provider and settings isolation; the library tier's
``CoreConfig`` builder and temp registry paths).
"""

from __future__ import annotations

import pytest

from vtscore.host_seams import HostSeams, restore_host_seams

#: Training epochs for the whole test session (production default is 200; 30 is
#: enough for the tiny MLP heads to converge on the small test fixtures).
#: Re-asserted per test by :func:`reset_shared_state` because
#: ``test_torch_config.py`` reloads ``vtscore.config``, which resets every
#: module-level constant to its import-time value — and a leaked 200 silently
#: retrains every later detector fixture on a different budget, which is what
#: made a fully seeded threshold fixture order-dependent (issue #3101).
TEST_TRAIN_EPOCHS = 30

#: Ids of the contexts :func:`reset_shared_state` installs for each test.
_TEST_DATASET_CONTEXT_ID = "_test_default"
_TEST_DETECTOR_CONTEXT_ID = "_test_default_det"


def pin_training_budget() -> None:
    """Pin ``TRAIN_EPOCHS`` to the test budget for the rest of the session.

    Written to the ``vtscore.config`` package *and* to the submodule that
    defines it: ``vtscore.training.mlp`` reads it off the package at call time,
    so the package write is the load-bearing one, but a reload re-executes the
    submodule and leaving the two disagreeing would be a trap for any future
    in-package reader.  Called once at each conftest's import time and again per
    test from :func:`reset_shared_state`.
    """
    import vtscore.config as config
    from vtscore.config import runtime as config_runtime

    config.TRAIN_EPOCHS = config_runtime.TRAIN_EPOCHS = TEST_TRAIN_EPOCHS


def install_startup_contexts() -> None:
    """Create the import-time dataset/detector contexts.

    ``init_medias()`` needs a dataset context to write into, and vote proxies
    need a detector context to delegate to, before the per-test fixture has run
    even once.
    """
    import vtscore.state.core as state_core

    startup_ctx = state_core.DatasetContext("_startup")
    state_core.register_context(startup_ctx)
    state_core.set_thread_dataset_context(startup_ctx)
    startup_det = state_core.DetectorContext("_startup_det")
    state_core.register_detector_context(startup_det)
    state_core.set_thread_detector_context(startup_det)


#: Host seams as the conftest left them at the end of its bootstrap.  Filled by
#: :func:`capture_startup_host_seams` and put back before every test by
#: :func:`reset_shared_state`.  ``None`` until captured, so a suite that never
#: calls the bootstrap simply skips the seam restore rather than wiping seams it
#: never recorded.
_startup_host_seams: HostSeams | None = None


def capture_startup_host_seams() -> None:
    """Record the host seams the conftest has just finished installing.

    Must run *after* the tier has wired its seams - for the app tier that means
    after ``import app`` has run its ``register_app_*`` calls, for the library
    tier after the conftest's ``register_core_config_builder``.  Both conftests
    call it beside :func:`freeze_startup_heap`, which is already at that point.

    Deliberately a snapshot rather than a reset: the app tier's tests run
    *against* the real ``vtsearch`` wiring, so resetting the slots to their
    library defaults before each test would strip the seams under test.
    """
    global _startup_host_seams

    from vtscore.host_seams import capture_host_seams

    _startup_host_seams = capture_host_seams()


def freeze_startup_heap() -> None:
    """Exclude the import-time heap from garbage-collection scans.

    Torch, the registries and the test medias all live for the whole session
    anyway.  Production code sprinkles ``gc.collect()`` through the dataset-load
    pipeline for memory hygiene on huge datasets; with the multi-hundred-MB
    startup heap unfrozen, each of those calls costs ~0.3s of pure scan time in
    tests that load several tiny datasets (combine, staging, promote).
    """
    import gc

    gc.collect()
    gc.freeze()


def reset_shared_state(medias_snapshot) -> None:
    """Reset every ``vtscore`` global that leaks between tests.

    *medias_snapshot* is the deep copy of the generated test medias taken at
    each conftest's import time; it is passed in rather than imported because
    generating them is a tier-specific fixture concern.

    The live mapping to refill is **not** a parameter: it is the ``medias`` dict
    of the default context this function has just registered, so resolving it
    here is both simpler and safer than accepting one.  A caller can only pass a
    mapping it resolved *before* the reset - i.e. the previous test's context -
    unless it hands over a lazily-resolving proxy, and the only such proxy in the
    repo is app-tier (``vtsearch.state_proxies``), which ``tests_lib/`` may not
    import.
    """
    import vtscore.config as config
    import vtscore.state.core as core

    # Put the host seams back first: two of them (the dataset / detector context
    # resolvers) decide which context the rest of this reset - and the test that
    # follows - actually lands on, so a resolver leaked by the previous test
    # would misdirect the contexts registered just below.
    if _startup_host_seams is not None:
        restore_host_seams(_startup_host_seams)

    core.clear_all_contexts()
    default_ctx = core.DatasetContext(_TEST_DATASET_CONTEXT_ID)
    core.register_context(default_ctx)
    core.set_thread_dataset_context(default_ctx)

    core.clear_all_detector_contexts()
    default_det = core.DetectorContext(_TEST_DETECTOR_CONTEXT_ID)
    core.register_detector_context(default_det)
    core.set_thread_detector_context(default_det)

    default_ctx.medias.update({k: dict(v) for k, v in medias_snapshot.items()})

    from vtscore.security.hf_auth import clear_credential

    clear_credential()

    # Drain background jobs (joining any live worker) BEFORE clearing the
    # progress cache: a labeling-status refresh from the previous test writes
    # the status snapshot at the end of its run, so it must be stopped first or
    # its late write would survive the clear and leak into this test.
    from vtscore.concurrency.async_jobs import reset_all_async_jobs_for_tests

    reset_all_async_jobs_for_tests()

    from vtscore.state.sort_results_cache import sort_results_cache

    sort_results_cache.reset_for_tests()

    from vtscore.detectors.labeling_progress import clear_progress_cache

    clear_progress_cache()

    from vtscore.embedding.helpers import clear_text_query_cache

    clear_text_query_cache()

    # ``resolve_device`` is lru_cached for the life of the process.  A test that
    # resolves it under mocked CUDA (``test_torch_config.py``, which shares the
    # worker process) would otherwise leak a cached "cuda" into every later
    # ``train_model`` on a CPU-only box.  ``vtscore.embedding.loader`` binds the
    # function at import, and the ``importlib.reload`` in those tests can leave
    # it holding a *different* function object than the current module
    # attribute — clear both.
    import vtscore.embedding.loader as emb_loader

    config.resolve_device.cache_clear()
    emb_loader.resolve_device.cache_clear()

    # Same reload hazard, for a value that doesn't crash when it leaks.
    pin_training_budget()

    from vtscore.concurrency.progress import (
        clear_thread_progress,
        detector_loading_tasks,
        eval_progress,
        find_progress,
        loading_tasks,
        sort_progress,
    )

    # A test that bound a per-thread progress sink must not leak it into the
    # next one: with the global fallback gone, resolve_progress_callback() reads
    # this and nothing else.
    clear_thread_progress()
    find_progress.update("idle", "", 0, 0, step=None, total_steps=None, error=None)
    sort_progress.update("idle", "", 0, 0, step=None, total_steps=None, error=None)
    eval_progress.update("idle", "", 0, 0, step=None, total_steps=None, error=None)
    # Cancels each task and joins its worker before dropping the entry, so a
    # load still running at the end of the previous test is stopped rather than
    # orphaned.
    loading_tasks.reset_for_tests()
    detector_loading_tasks.reset_for_tests()

    # ...and hand this test fresh load gates regardless, so a worker that
    # outlived the join above holds a permit on a gate nothing will look at
    # again.  Without this the gates' counts are whatever the previous test in
    # the worker process left behind, which made the assertions in
    # ``TestLoadingGates`` a lottery over xdist scheduling (issue #3613).
    from vtscore.datasets.load_pipeline import reset_load_gates_for_tests

    reset_load_gates_for_tests()

    # Cancel any debounced labelset-source push left over from the previous test
    # so its captured contexts don't fire after this test's reset has dropped
    # them.
    from vtscore.labels.sync import reset_label_sync_for_tests

    reset_label_sync_for_tests()

    # Drop the TTL-cached detector-file mtimes so a stale entry from a prior
    # test can't suppress a rehydrate in the next one.
    from vtscore.detectors.dataset_sync import reset_mtime_cache_for_tests

    reset_mtime_cache_for_tests()

    # Forget which encoders the timing recorder has seen loaded: the ledger is
    # process-global and decides whether a recorded run is written cold or warm,
    # so one test's run would otherwise label the next test's.
    from vtscore.timing.recorder import reset_seen_models_for_tests

    reset_seen_models_for_tests()

    # Reset CLI progress format so a test that flips it to "json" can't leak the
    # choice into the next test.
    from vtscore import cli_progress

    cli_progress.set_format("text")

    # Drop notification subscribers: they are process-global, so a test that
    # subscribes a collector (or drives the CLI, which subscribes a printer)
    # would otherwise keep receiving every later test's notifications.
    from vtscore.concurrency.notifications import notifications

    notifications.clear_subscribers()

    from vtscore.datasets.registry import reset_for_tests as reset_dataset_registry
    from vtscore.detectors.registry import reset_for_tests as reset_detector_registry

    reset_dataset_registry()
    reset_detector_registry()


@pytest.fixture(autouse=True)
def allow_test_tmp_paths(monkeypatch):
    """Widen ``validate_server_filepath`` to also accept the system temp tree.

    With ``base_dir=None`` (single-user / no-auth mode) ``validate_server_filepath``
    is already unrestricted, so system temp paths pass unchanged.  This wrapper
    only matters for the ``base_dir=None`` path historically; when a specific
    ``base_dir`` is given (e.g. multi-user mode) the restriction is honoured.
    """
    import tempfile
    from pathlib import Path

    import vtscore.security.path_validation as paths_mod

    original = paths_mod.validate_server_filepath

    def _permissive(filepath_str, base_dir=None):
        try:
            return original(filepath_str, base_dir)
        except ValueError:
            # When a specific base_dir is given (multi-user mode) we must honour
            # that restriction; only widen for the unrestricted default case.
            if base_dir is not None:
                raise
            return original(filepath_str, Path(tempfile.gettempdir()))

    monkeypatch.setattr(paths_mod, "validate_server_filepath", _permissive)
