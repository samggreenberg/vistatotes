"""Dataset importer registry with auto-discovery.

Any package placed directly under this directory is automatically registered
if it exposes a module-level ``IMPORTER`` attribute that is a
:class:`~vtsearch.datasets.importers.base.DatasetImporter` instance.

Usage::

    from vtsearch.datasets.importers import get_importer, list_importers

    importer = get_importer("folder")
    for imp in list_importers():
        print(imp.name, imp.display_name)
"""

from __future__ import annotations

import importlib
import pkgutil
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vtsearch.datasets.importers.base import DatasetImporter

_registry: dict[str, DatasetImporter] = {}


def _discover() -> None:
    """Scan sub-packages for ``IMPORTER`` objects and register them."""
    package_dir = Path(__file__).parent
    for _, module_name, is_pkg in pkgutil.iter_modules([str(package_dir)]):
        if not is_pkg:
            continue
        try:
            mod = importlib.import_module(f"vtsearch.datasets.importers.{module_name}")
            importer = getattr(mod, "IMPORTER", None)
            if importer is not None:
                _registry[importer.name] = importer
        except Exception as exc:  # pragma: no cover
            warnings.warn(
                f"Failed to load dataset importer '{module_name}': {exc}",
                stacklevel=2,
            )


def _ensure_discovered() -> None:
    if not _registry:
        _discover()


def get_importer(name: str) -> DatasetImporter | None:
    """Return the registered importer with *name*, or ``None`` if not found."""
    _ensure_discovered()
    return _registry.get(name)


def list_importers() -> list[DatasetImporter]:
    """Return all registered importers in discovery order."""
    _ensure_discovered()
    return list(_registry.values())


__all__ = [
    "get_importer",
    "list_importers",
]
