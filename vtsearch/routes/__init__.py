"""Flask blueprints for organizing application routes."""

from vtsearch.routes.clips import clips_bp
from vtsearch.routes.datasets import datasets_bp
from vtsearch.routes.detectors import detectors_bp
from vtsearch.routes.exporters import exporters_bp
from vtsearch.routes.label_importers import label_importers_bp
from vtsearch.routes.main import main_bp
from vtsearch.routes.sorting import sorting_bp

__all__ = [
    "main_bp",
    "clips_bp",
    "sorting_bp",
    "detectors_bp",
    "datasets_bp",
    "exporters_bp",
    "label_importers_bp",
]
