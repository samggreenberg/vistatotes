"""Flask blueprints for organizing application routes."""

from vectorytones.routes.clips import clips_bp
from vectorytones.routes.datasets import datasets_bp
from vectorytones.routes.main import main_bp
from vectorytones.routes.sorting import sorting_bp

__all__ = [
    "main_bp",
    "clips_bp",
    "sorting_bp",
    "datasets_bp",
]
