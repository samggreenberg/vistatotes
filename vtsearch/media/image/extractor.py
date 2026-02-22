"""Image class extractor using YOLO object detection."""

from __future__ import annotations

import io
from typing import Any, Optional

from PIL import Image

from vtsearch.media.base import Extractor


class ImageClassExtractor(Extractor):
    """Extracts bounding boxes of a specific YOLO class from images.

    Each instance is configured with a target class name (e.g. ``"person"``,
    ``"car"``, ``"dog"``) and a confidence threshold.  Running :meth:`extract`
    on an image clip returns a list of dicts, one per detected object of the
    target class whose confidence meets the threshold::

        [
            {
                "confidence": 0.92,
                "bbox": [x1, y1, x2, y2],
                "label": "car",
            },
            ...
        ]

    Coordinates are in pixel space (float) matching the original image size.
    """

    def __init__(self, name: str, target_class: str, threshold: float = 0.25, model_id: str = "yolo11n.pt") -> None:
        """Create an extractor that finds *target_class* objects.

        Args:
            name: Unique name for this extractor instance.
            target_class: YOLO class name to look for (e.g. ``"person"``).
            threshold: Minimum confidence to count a detection (0–1).
            model_id: YOLO model weight file passed to ``ultralytics.YOLO()``.
        """
        self._name = name
        self._target_class = target_class
        self._threshold = threshold
        self._model_id = model_id
        self._model: Optional[Any] = None

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @property
    def media_type(self) -> str:
        return "image"

    @property
    def target_class(self) -> str:
        return self._target_class

    @property
    def threshold(self) -> float:
        return self._threshold

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        if self._model is not None:
            return
        from ultralytics import YOLO

        self._model = YOLO(self._model_id)

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def extract(self, clip: dict[str, Any]) -> list[dict[str, Any]]:
        """Detect ``target_class`` objects in *clip* and return bounding boxes.

        The *clip* dict must contain ``"clip_bytes"`` (raw image bytes).

        Returns a list of dicts, each with keys ``"confidence"``, ``"bbox"``
        (``[x1, y1, x2, y2]`` in pixels), and ``"label"``.
        """
        self.load_model()
        assert self._model is not None

        clip_bytes = clip.get("clip_bytes")
        if clip_bytes is None:
            return []

        image = Image.open(io.BytesIO(clip_bytes)).convert("RGB")
        results = self._model(image, verbose=False)

        hits: list[dict[str, Any]] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                conf = float(boxes.conf[i])
                cls_id = int(boxes.cls[i])
                label = result.names[cls_id]
                if label != self._target_class:
                    continue
                if conf < self._threshold:
                    continue
                bbox = boxes.xyxy[i].tolist()
                hits.append(
                    {
                        "confidence": round(conf, 4),
                        "bbox": [round(c, 2) for c in bbox],
                        "label": label,
                    }
                )

        hits.sort(key=lambda h: h["confidence"], reverse=True)
        return hits

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["extractor_type"] = "image_class"
        d["config"] = {
            "target_class": self._target_class,
            "threshold": self._threshold,
            "model_id": self._model_id,
        }
        return d

    @classmethod
    def from_config(cls, name: str, config: dict[str, Any]) -> "ImageClassExtractor":
        """Reconstruct an ``ImageClassExtractor`` from a saved config dict."""
        return cls(
            name=name,
            target_class=config["target_class"],
            threshold=config.get("threshold", 0.25),
            model_id=config.get("model_id", "yolo11n.pt"),
        )
