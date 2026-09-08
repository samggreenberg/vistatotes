"""The stat-free ``packages_distributions`` seed (issue #3715).

``transformers`` calls ``importlib.metadata.packages_distributions()`` at
module import; the stdlib implementation stats every file recorded by every
installed distribution, which on a cold NFS venv is minutes of silent startup.
These tests pin both halves of the fix: the replacement mapping is built
without touching the recorded files, and startup installs it before anything
can import transformers.
"""

import importlib.metadata
from unittest.mock import patch

import pytest

from vtscore.utils.import_metadata import (
    fast_packages_distributions,
    original_packages_distributions,
    seed_packages_distributions,
)


@pytest.fixture
def restore_stdlib(monkeypatch):
    """Put the stdlib implementation back for the duration of a test.

    Both conftests call ``initialize_models()`` at import, so by the time any
    test runs the seed is already installed process-wide.
    """
    monkeypatch.setattr(importlib.metadata, "packages_distributions", original_packages_distributions)


@pytest.fixture
def synthetic_site(tmp_path, monkeypatch):
    """A site-packages dir with two dist-infos whose recorded files don't exist."""
    site = tmp_path / "site"
    declared = site / "foo-1.0.dist-info"
    declared.mkdir(parents=True)
    (declared / "METADATA").write_text("Metadata-Version: 2.1\nName: foo\nVersion: 1.0\n")
    (declared / "top_level.txt").write_text("foo\n_foo_ext\n")

    inferred = site / "bar-2.0.dist-info"
    inferred.mkdir(parents=True)
    (inferred / "METADATA").write_text("Metadata-Version: 2.1\nName: bar\nVersion: 2.0\n")
    (inferred / "RECORD").write_text(
        "bar/__init__.py,sha256=x,10\n"
        "bar/sub/mod.py,sha256=x,10\n"
        "barmod.py,sha256=x,10\n"
        "bar-2.0.dist-info/METADATA,sha256=x,10\n"
        "../../include/bar.h,sha256=x,10\n"
    )
    monkeypatch.syspath_prepend(str(site))
    return site


class TestFastPackagesDistributions:
    def test_reads_declared_top_level_names(self, synthetic_site):
        mapping = fast_packages_distributions()
        assert mapping["foo"] == ["foo"]
        assert mapping["_foo_ext"] == ["foo"]

    def test_infers_top_level_names_from_record(self, synthetic_site):
        mapping = fast_packages_distributions()
        assert mapping["bar"] == ["bar"]
        assert mapping["barmod"] == ["bar"]
        # The ``.dist-info`` row and the ``../`` data-path escape are not import
        # names, and neither is the nested module.
        assert "bar-2" not in mapping
        assert ".." not in mapping
        assert "mod" not in mapping

    def test_does_not_stat_recorded_files(self, synthetic_site):
        """None of the recorded files exist, yet their names still map.

        The stdlib drops them: ``Distribution.files`` filters through
        ``os.path.exists()``, and that filter *is* the 85k-stat walk.
        """
        assert not (synthetic_site / "bar").exists()
        assert "bar" in fast_packages_distributions()

    def test_never_touches_distribution_files(self, synthetic_site):
        def boom(self):
            raise AssertionError("Distribution.files stats every recorded path")

        with patch.object(importlib.metadata.Distribution, "files", property(boom)):
            mapping = fast_packages_distributions()
        assert "bar" in mapping

    def test_loses_nothing_the_stdlib_reports_on_the_live_venv(self):
        """Every stdlib entry survives, with the same distributions behind it.

        Equality is deliberately not asserted: skipping the missing-file filter
        can only *add* names, and which extras appear depends on what the venv
        happens to record (and on the stdlib's own per-version inference).
        Losing an entry would be a real break; gaining one is not.

        ``__pycache__`` is the one name held out, and it is not an import name.
        A single-module distribution records ``__pycache__/<mod>.cpython-*.pyc``
        beside its ``.py``, so the stdlib's inference reports ``__pycache__`` as
        a top-level name of every such package -- three of them on the GRID venv
        (`typing_extensions`, `threadpoolctl`, `matplotlib`), and none in the
        container this test was written in, which is why it passed there and
        fails here. :func:`_inferred_top_level` drops the name on purpose,
        because nothing can import it; asserting it survives would pin a stdlib
        artifact as a contract.
        """
        fast = fast_packages_distributions()
        real = {n: d for n, d in original_packages_distributions().items() if n != "__pycache__"}
        assert {name: fast.get(name) for name in real} == real


class TestSeed:
    def test_installs_and_is_idempotent(self, restore_stdlib):
        assert importlib.metadata.packages_distributions is original_packages_distributions
        assert seed_packages_distributions() is True
        assert importlib.metadata.packages_distributions is fast_packages_distributions
        assert seed_packages_distributions() is True
        assert importlib.metadata.packages_distributions is fast_packages_distributions

    def test_reports_failure_when_the_function_is_absent(self, monkeypatch):
        monkeypatch.delattr(importlib.metadata, "packages_distributions", raising=False)
        assert seed_packages_distributions() is False


class TestStartupSeedsBeforeTransformers:
    def test_initialize_models_installs_the_seed(self, restore_stdlib):
        from vtscore.embedding.loader import initialize_models

        called = []
        monkeyed = importlib.metadata.packages_distributions

        def tripwire():
            called.append(True)
            return monkeyed()

        with (
            patch.object(importlib.metadata, "packages_distributions", tripwire),
            patch("vtscore.embedding.loader._warm_threadpool_controller"),
            patch("vtscore.embedding.loader._install_transformers_logging_bridge"),
        ):
            initialize_models()
            assert importlib.metadata.packages_distributions is fast_packages_distributions

        assert called == [], "startup invoked the stat-walking stdlib implementation"
