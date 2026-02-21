"""Processor-importer registry with auto-discovery.

Any package placed directly under this directory is automatically registered
if it exposes a module-level ``PROCESSOR_IMPORTER`` attribute that is a
:class:`~vtsearch.processors.importers.base.ProcessorImporter` instance.

Usage::

    from vtsearch.processors.importers import get_processor_importer, list_processor_importers

    importer = get_processor_importer("detector_file")
    for imp in list_processor_importers():
        print(imp.name, imp.display_name)
"""

from __future__ import annotations

import importlib
import pkgutil
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vtsearch.processors.importers.base import ProcessorImporter

_registry: dict[str, ProcessorImporter] = {}


def _discover() -> None:
    """Scan sub-packages for ``PROCESSOR_IMPORTER`` objects and register them."""
    package_dir = Path(__file__).parent
    for _, module_name, is_pkg in pkgutil.iter_modules([str(package_dir)]):
        if not is_pkg:
            continue
        try:
            mod = importlib.import_module(f"vtsearch.processors.importers.{module_name}")
            importer = getattr(mod, "PROCESSOR_IMPORTER", None)
            if importer is not None:
                _registry[importer.name] = importer
        except Exception as exc:  # pragma: no cover
            warnings.warn(
                f"Failed to load processor importer '{module_name}': {exc}",
                stacklevel=2,
            )


def _ensure_discovered() -> None:
    if not _registry:
        _discover()


def get_processor_importer(name: str) -> ProcessorImporter | None:
    """Return the registered processor importer with *name*, or ``None`` if not found."""
    _ensure_discovered()
    return _registry.get(name)


def list_processor_importers() -> list[ProcessorImporter]:
    """Return all registered processor importers in discovery order."""
    _ensure_discovered()
    return list(_registry.values())
