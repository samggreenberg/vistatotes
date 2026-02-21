"""Label importer registry with auto-discovery.

Any package placed directly under this directory is automatically registered
if it exposes a module-level ``LABEL_IMPORTER`` attribute that is a
:class:`~vtsearch.labels.importers.base.LabelImporter` instance.

Usage::

    from vtsearch.labels.importers import get_label_importer, list_label_importers

    importer = get_label_importer("csv")
    for imp in list_label_importers():
        print(imp.name, imp.display_name)
"""

from __future__ import annotations

import importlib
import pkgutil
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vtsearch.labels.importers.base import LabelImporter

_registry: dict[str, LabelImporter] = {}


def _discover() -> None:
    """Scan sub-packages for ``LABEL_IMPORTER`` objects and register them."""
    package_dir = Path(__file__).parent
    for _, module_name, is_pkg in pkgutil.iter_modules([str(package_dir)]):
        if not is_pkg:
            continue
        try:
            mod = importlib.import_module(f"vtsearch.labels.importers.{module_name}")
            importer = getattr(mod, "LABEL_IMPORTER", None)
            if importer is not None:
                _registry[importer.name] = importer
        except Exception as exc:  # pragma: no cover
            warnings.warn(
                f"Failed to load label importer '{module_name}': {exc}",
                stacklevel=2,
            )


def _ensure_discovered() -> None:
    if not _registry:
        _discover()


def get_label_importer(name: str) -> LabelImporter | None:
    """Return the registered label importer with *name*, or ``None`` if not found."""
    _ensure_discovered()
    return _registry.get(name)


def list_label_importers() -> list[LabelImporter]:
    """Return all registered label importers in discovery order."""
    _ensure_discovered()
    return list(_registry.values())
