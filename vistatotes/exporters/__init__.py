"""Results-exporter registry with auto-discovery.

Any package placed directly under this directory is automatically registered
if it exposes a module-level ``EXPORTER`` attribute that is a
:class:`~vistatotes.exporters.base.ResultsExporter` instance.

Usage::

    from vistatotes.exporters import get_exporter, list_exporters

    exporter = get_exporter("file")
    for exp in list_exporters():
        print(exp.name, exp.display_name)
"""

from __future__ import annotations

import importlib
import pkgutil
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vistatotes.exporters.base import ResultsExporter

_registry: dict[str, ResultsExporter] = {}


def _discover() -> None:
    """Scan sub-packages for ``EXPORTER`` objects and register them."""
    package_dir = Path(__file__).parent
    for _, module_name, is_pkg in pkgutil.iter_modules([str(package_dir)]):
        if not is_pkg:
            continue
        try:
            mod = importlib.import_module(
                f"vistatotes.exporters.{module_name}"
            )
            exporter = getattr(mod, "EXPORTER", None)
            if exporter is not None:
                _registry[exporter.name] = exporter
        except Exception as exc:  # pragma: no cover
            warnings.warn(
                f"Failed to load results exporter '{module_name}': {exc}",
                stacklevel=2,
            )


def _ensure_discovered() -> None:
    if not _registry:
        _discover()


def get_exporter(name: str) -> ResultsExporter | None:
    """Return the registered exporter with *name*, or ``None`` if not found."""
    _ensure_discovered()
    return _registry.get(name)


def list_exporters() -> list[ResultsExporter]:
    """Return all registered exporters in discovery order."""
    _ensure_discovered()
    return list(_registry.values())
