"""Flask blueprints for organizing application routes."""

from vistatotes.routes.clips import clips_bp
from vistatotes.routes.datasets import datasets_bp
from vistatotes.routes.detectors import detectors_bp
from vistatotes.routes.exporters import exporters_bp
from vistatotes.routes.main import main_bp
from vistatotes.routes.sorting import sorting_bp

__all__ = [
    "main_bp",
    "clips_bp",
    "sorting_bp",
    "detectors_bp",
    "datasets_bp",
    "exporters_bp",
]
