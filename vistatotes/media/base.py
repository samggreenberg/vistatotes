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
from dataclasses import dataclass, field
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
