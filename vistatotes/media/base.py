"""Abstract base class for media types.

To add a new media type:

1. Create a subdirectory under ``vistatotes/media/`` (e.g. ``vistatotes/media/code/``).
2. Add a ``requirements.txt`` listing any pip packages your embedder needs.
3. Implement a subclass of :class:`MediaType` in ``media_type.py``.
4. Register it in ``vistatotes/media/__init__.py``::

       from vistatotes.media.code.media_type import CodeMediaType
       register(CodeMediaType())

That is all.  The rest of the application (routing, dataset loading, model
initialisation, demo listing) picks up your new type automatically through
the registry.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from flask import Response


@dataclass
class DemoDataset:
    """Metadata describing one demo dataset that belongs to a media type."""

    id: str
    """Unique key used throughout the app (e.g. ``"nature_sounds"``)."""

    label: str
    """Human-readable display name (e.g. ``"Animal & Nature Sounds"``)."""

    description: str
    """Long-form description shown in the UI."""

    categories: list
    """Category names used to filter the raw source data."""

    source: str = ""
    """Identifier for the raw data source (e.g. ``"cifar10_sample"``, ``"ucf101"``).
    Leave empty for sources that don't require an explicit identifier."""


class MediaType(ABC):
    """Abstract base class that every media type must implement.

    A *media type* bundles together everything the application needs to work
    with a particular kind of media:

    * How to embed a file into a fixed-size vector (:meth:`embed_media`).
    * How to embed a text query into the **same** vector space (:meth:`embed_text`).
    * Human-readable identity: :attr:`name` and :attr:`icon`.
    * Which file extensions to scan when importing a folder (:attr:`file_extensions`).
    * Whether the viewer should loop (:attr:`loops`).
    * Which demo datasets are available (:attr:`demo_datasets`).
    * How to serve a clip over HTTP (:meth:`clip_response`).
    * How to load media-specific clip fields from a file (:meth:`load_clip_data`).
    * An optional folder-import alias (:attr:`folder_import_name`) for the
      ``/api/dataset/load-folder`` endpoint.

    Adding a new media type
    -----------------------
    See the module docstring above for the four-step process.
    """

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def type_id(self) -> str:
        """Unique internal identifier, e.g. ``"audio"``, ``"video"``."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable display name shown in the UI, e.g. ``"Audio"``."""

    @property
    @abstractmethod
    def icon(self) -> str:
        """Icon for the UI (emoji or icon name), e.g. ``"🔊"``."""

    # ------------------------------------------------------------------
    # File import
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def file_extensions(self) -> list:
        """Glob patterns for importable files, e.g. ``["*.wav", "*.mp3"]``."""

    @property
    def folder_import_name(self) -> str:
        """Alias used by the ``/api/dataset/load-folder`` endpoint.

        Defaults to :attr:`type_id`.  Override if your type uses a legacy
        plural name (e.g. ``"sounds"``, ``"videos"``).
        """
        return self.type_id

    # ------------------------------------------------------------------
    # Viewer behaviour
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def loops(self) -> bool:
        """``True`` if the viewer should loop (audio/video); ``False`` otherwise."""

    # ------------------------------------------------------------------
    # Demo datasets
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def demo_datasets(self) -> list:
        """List of :class:`DemoDataset` objects available for this media type."""

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    @abstractmethod
    def load_models(self) -> None:
        """Load (and cache) the embedding models for this media type.

        Called once at application startup for every registered media type.
        Implementations must be idempotent — a second call should be a no-op.
        """

    @abstractmethod
    def embed_media(self, file_path: Path) -> Optional[np.ndarray]:
        """Return a fixed-size embedding vector for the media file at *file_path*.

        Returns ``None`` if the file cannot be embedded (model not loaded,
        corrupt file, etc.).
        """

    @abstractmethod
    def embed_text(self, text: str) -> Optional[np.ndarray]:
        """Return an embedding of *text* in the **same vector space** as :meth:`embed_media`.

        This is used for text-query sorting: the resulting vector is compared
        against media embeddings via cosine similarity.

        Returns ``None`` if the model is not loaded or encoding fails.
        """

    # ------------------------------------------------------------------
    # Clip data
    # ------------------------------------------------------------------

    @abstractmethod
    def load_clip_data(self, file_path: Path) -> dict:
        """Load and return media-specific fields for a clip dict.

        The returned dict is merged into the *base* clip dict (which already
        contains ``id``, ``type``, ``file_size``, ``md5``, ``embedding``,
        ``filename``, and ``category``).  You must include at minimum a
        ``"duration"`` key.

        Example return value for audio::

            {"wav_bytes": b"...", "duration": 3.2}

        Example return value for images::

            {"image_bytes": b"...", "duration": 0, "width": 32, "height": 32}
        """

    # ------------------------------------------------------------------
    # HTTP serving
    # ------------------------------------------------------------------

    @abstractmethod
    def clip_response(self, clip: dict) -> Response:
        """Return a Flask :class:`~flask.Response` that serves *clip*'s media content.

        Use ``flask.send_file`` for binary media or ``flask.jsonify`` for
        text/structured data.
        """


class Extractor(ABC):
    """Abstract base class for extractors.

    While a *Detector* answers "is this clip Good?" (True/False), an
    *Extractor* answers "what Good things are inside this clip, and where?"
    by returning structured details for each occurrence found.

    For example an image extractor might return bounding boxes and class
    labels; a video extractor might return start/stop timestamps of events.

    Each concrete ``Extractor`` operates on exactly **one** media type
    (declared via :attr:`media_type`), just like Detectors.

    Subclasses must implement:

    * :attr:`name` — unique identifier for this extractor.
    * :attr:`media_type` — which media type it works on.
    * :meth:`extract` — run extraction on a single clip dict and return a
      list of result dicts.

    Subclasses *may* override:

    * :meth:`load_model` — called once before first use to load heavy
      resources (model weights, etc.).  Default is a no-op.
    """

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this extractor, e.g. ``"yolo_person"``."""

    @property
    @abstractmethod
    def media_type(self) -> str:
        """The media ``type_id`` this extractor operates on (e.g. ``"image"``)."""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Load any heavyweight resources (model weights, etc.).

        Called lazily before the first :meth:`extract` call.  The default
        implementation is a no-op — override in subclasses that need
        one-time model loading.
        """

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    @abstractmethod
    def extract(self, clip: dict[str, Any]) -> list[dict[str, Any]]:
        """Run extraction on *clip* and return a list of result dicts.

        Each dict in the returned list describes **one occurrence** of the
        thing the extractor is looking for.  The schema of these dicts is
        extractor-specific, but every dict **must** include a ``"confidence"``
        key with a float in ``[0, 1]``.

        Returns an empty list when nothing is found.

        Example return value for an image bounding-box extractor::

            [
                {"confidence": 0.92, "bbox": [x1, y1, x2, y2], "label": "car"},
                {"confidence": 0.87, "bbox": [x1, y1, x2, y2], "label": "car"},
            ]

        Example return value for a video timestamp extractor::

            [
                {"confidence": 0.85, "start": 1.2, "end": 3.4, "label": "explosion"},
            ]
        """

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary of this extractor's metadata."""
        return {
            "name": self.name,
            "media_type": self.media_type,
        }
