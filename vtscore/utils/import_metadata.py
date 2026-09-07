"""Stat-free replacement for :func:`importlib.metadata.packages_distributions`.

``transformers`` builds a top-level-name -> distribution mapping at *module
import* time, unconditionally and with no env guard::

    # transformers/utils/import_utils.py
    PACKAGE_DISTRIBUTION_MAPPING = importlib.metadata.packages_distributions()

The stdlib implementation asks every installed distribution for its file list,
and ``Distribution.files`` filters that list through ``os.path.exists()`` - one
``stat()`` per *file recorded by every installed package*.  On a venv carrying
torch + onnx + RAPIDS that is ~85,000 stats across ~224 distributions.  Locally
that is a page-cache hit and costs milliseconds; on an NFS-mounted venv whose
dentries have been evicted it is 85,000 serialized round trips.  Measured on the
GRID (issue #3715): 78 stats/sec, so **16 minutes** of startup during which the
process sits in ``D`` state printing nothing and looking hung.

:func:`seed_packages_distributions` swaps in an implementation that reads each
distribution's ``top_level.txt`` (falling back to parsing ``RECORD`` as text)
and never touches the recorded files at all - ~224 small reads instead of 85k
stats.  Call it once, early, **before anything imports transformers**; the
module-level call above then finds our function under the name it looks up and
the walk never happens.
"""

import collections
import csv
import importlib.metadata
import inspect
from collections.abc import Iterable, Mapping
from pathlib import PurePosixPath

__all__ = ["fast_packages_distributions", "seed_packages_distributions"]

#: The stdlib function displaced by :func:`seed_packages_distributions`, kept so
#: callers (and tests) can tell whether the seed is installed and can put the
#: original back.
original_packages_distributions = importlib.metadata.packages_distributions


def _declared_top_level(dist: importlib.metadata.Distribution) -> list[str]:
    """Top-level names a distribution declares in ``top_level.txt`` (may be empty)."""
    try:
        return (dist.read_text("top_level.txt") or "").split()
    except Exception:
        return []


def _inferred_top_level(dist: importlib.metadata.Distribution) -> set[str]:
    """Top-level names inferred from ``RECORD``, without stat-ing the files.

    Mirrors the stdlib's ``_top_level_inferred`` - first path component for a
    nested path, module name for a top-level file - but reads ``RECORD`` as
    text rather than going through ``Distribution.files``, whose
    ``skip_missing_files`` filter is the ``stat()`` storm this module exists to
    avoid.  Dropping that filter can only *add* names (those of files recorded
    but since deleted), which is harmless for the mapping's consumers.
    """
    try:
        record = dist.read_text("RECORD") or ""
    except Exception:
        return set()
    names: set[str] = set()
    for row in csv.reader(record.splitlines()):
        if not row or not row[0]:
            continue
        parts = PurePosixPath(row[0]).parts
        name = parts[0] if len(parts) > 1 else inspect.getmodulename(row[0])
        # A dotted name is not an import name: it is a ``*.dist-info`` directory,
        # a ``../`` escape to a data path, or a file with an unrecognised suffix.
        if name and "." not in name and name != "__pycache__":
            names.add(name)
    return names


def fast_packages_distributions() -> Mapping[str, list[str]]:
    """Return ``{import_name: [distribution_name, ...]}`` without stat-ing files.

    Drop-in for :func:`importlib.metadata.packages_distributions`.
    """
    pkg_to_dist: dict[str, list[str]] = collections.defaultdict(list)
    for dist in importlib.metadata.distributions():
        try:
            dist_name = dist.metadata["Name"]
        except Exception:
            continue
        if not dist_name:
            continue
        names: Iterable[str] = _declared_top_level(dist) or _inferred_top_level(dist)
        for pkg in names:
            pkg_to_dist[pkg].append(dist_name)
    return dict(pkg_to_dist)


def seed_packages_distributions() -> bool:
    """Install :func:`fast_packages_distributions` over the stdlib's version.

    Idempotent and best-effort: returns ``True`` when the fast implementation is
    in place afterwards, ``False`` when the swap could not be made (a Python
    without the function, or an ``importlib.metadata`` that refuses attribute
    assignment).  Nothing downstream depends on the return value - a ``False``
    only means startup pays the original cost.
    """
    current = getattr(importlib.metadata, "packages_distributions", None)
    if current is fast_packages_distributions:
        return True
    if current is None:
        return False
    try:
        importlib.metadata.packages_distributions = fast_packages_distributions  # type: ignore[assignment]
    except Exception:
        return False
    return True
