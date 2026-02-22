"""Abstract base classes for media types and processors.

To add a new media type:

1. Create a subdirectory under ``vtsearch/media/`` (e.g. ``vtsearch/media/code/``).
2. Add a ``requirements.txt`` listing any pip packages your embedder needs.
3. Implement a subclass of :class:`MediaType` in ``media_type.py``.
4. Register it in ``vtsearch/media/__init__.py``::

       from vtsearch.media.code.media_type import CodeMediaType
       register(CodeMediaType())

That is all.  The rest of the application (routing, dataset loading, model
initialisation, demo listing) picks up your new type automatically through
the registry.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

# Type alias for progress callbacks.  Modules that accept an ``on_progress``
# parameter use this signature so callers can report status without depending
# on ``vtsearch.utils.progress``.
ProgressCallback = Callable[[str, str, int, int], None]

__all__ = [
    "DemoDataset",
    "Detector",
    "Extractor",
    "MediaResponse",
    "MediaType",
    "Processor",
    "ProgressCallback",
]


def _noop_progress(status: str, message: str = "", current: int = 0, total: int = 0) -> None:
    """Default no-op progress callback used when no real reporter is set."""


@dataclass
class MediaResponse:
    """Framework-agnostic representation of media content for HTTP serving.

    This decouples media type implementations from Flask so they can be used
    as standalone libraries.  The Flask route layer converts this into a real
    ``flask.Response`` via :func:`media_response_to_flask`.

    Attributes:
        data: The payload — ``bytes`` for binary media, ``dict`` for JSON.
        mimetype: MIME type string (e.g. ``"audio/wav"``, ``"application/json"``).
        download_name: Suggested filename for the ``Content-Disposition`` header.
    """

    data: bytes | dict
    mimetype: str
    download_name: str = ""


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

    required_folder: Optional[Path] = None
    """Local directory that must exist for a cached ``.pkl`` to be usable.

    Audio and video datasets store references to external media files rather
    than inlining the bytes, so a stale ``.pkl`` left behind after the source
    directory was removed would incorrectly appear ready.  Set this to the
    directory that the importer places the source files into (e.g.
    ``DATA_DIR / "ESC-50-master" / "audio"``).  Leave ``None`` for datasets
    whose ``.pkl`` is entirely self-contained (images, text)."""


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

        Called lazily the first time this media type needs to embed something
        (i.e. on the first ``embed_media``, ``embed_text``, or getter call).
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

    @property
    def description_wrappers(self) -> list[str]:
        """Return wrapper templates for enriching sort descriptions.

        Each template is a format string containing ``{text}`` where the
        user's description will be inserted.  Override in subclasses to
        provide media-specific wrappers that improve embedding quality.

        The default returns an empty list (no wrappers — plain embedding only).
        """
        return []

    def embed_text_enriched(self, text: str) -> Optional[np.ndarray]:
        """Embed *text* using the average over all description wrappers.

        For each wrapper in :attr:`description_wrappers`, formats the wrapper
        with *text*, embeds the result, and returns the mean of all resulting
        vectors (L2-normalised).  Falls back to :meth:`embed_text` if no
        wrappers are defined or all wrapper embeddings fail.

        Returns ``None`` only if :meth:`embed_text` also returns ``None``.
        """
        wrappers = self.description_wrappers
        if not wrappers:
            return self.embed_text(text)

        embeddings = []
        for wrapper in wrappers:
            wrapped = wrapper.format(text=text)
            vec = self.embed_text(wrapped)
            if vec is not None:
                embeddings.append(vec)

        if not embeddings:
            return self.embed_text(text)

        avg = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(avg)
        if norm > 0:
            avg = avg / norm
        return avg

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

            {"clip_bytes": b"...", "duration": 3.2}

        Example return value for images::

            {"clip_bytes": b"...", "duration": 0, "width": 32, "height": 32}
        """

    # ------------------------------------------------------------------
    # HTTP serving
    # ------------------------------------------------------------------

    @abstractmethod
    def clip_response(self, clip: dict) -> MediaResponse:
        """Return a :class:`MediaResponse` with the clip's media content.

        For binary media, set ``data`` to raw bytes with an appropriate
        ``mimetype``.  For structured data (e.g. text paragraphs), set
        ``data`` to a JSON-serialisable dict with ``mimetype="application/json"``.
        """


class Processor(ABC):
    """Abstract base class for all processors (detectors, extractors, etc.).

    A *Processor* takes a single media clip and produces an answer.  The
    exact type of the answer depends on the subclass:

    * A :class:`Detector` returns ``bool`` — "does this clip match?"
    * An :class:`Extractor` returns ``list[dict]`` — "what details are
      inside this clip?"

    Every processor knows its :attr:`name` (a unique human-readable
    identifier) and the :attr:`media_type` it operates on (e.g.
    ``"audio"``, ``"image"``).

    Subclasses must implement:

    * :attr:`name`
    * :attr:`media_type`
    * :meth:`process` — run the processor on a single clip dict.

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
        """Unique identifier for this processor, e.g. ``"dog_barks"``."""

    @property
    @abstractmethod
    def media_type(self) -> str:
        """The media ``type_id`` this processor operates on (e.g. ``"image"``)."""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Load any heavyweight resources (model weights, etc.).

        Called lazily before the first :meth:`process` call.  The default
        implementation is a no-op — override in subclasses that need
        one-time model loading.
        """

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    @abstractmethod
    def process(self, clip: dict[str, Any]) -> Any:
        """Run this processor on *clip* and return the result.

        The return type depends on the subclass:

        * :class:`Detector` → ``bool``
        * :class:`Extractor` → ``list[dict[str, Any]]``
        """

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary of this processor's metadata."""
        return {
            "name": self.name,
            "media_type": self.media_type,
        }


class Detector(Processor):
    """Abstract base class for detectors.

    A *Detector* answers "is this clip Good?" with a boolean.  Each
    concrete ``Detector`` operates on exactly **one** media type (declared
    via :attr:`media_type`).

    Subclasses must implement:

    * :attr:`name` — unique identifier for this detector.
    * :attr:`media_type` — which media type it works on.
    * :meth:`detect` — run detection on a single clip dict and return
      ``True`` if the clip matches, ``False`` otherwise.

    The generic :meth:`process` method delegates to :meth:`detect`.
    """

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    @abstractmethod
    def detect(self, clip: dict[str, Any]) -> bool:
        """Run detection on *clip* and return whether it matches.

        Returns ``True`` if the clip is a positive match for this detector,
        ``False`` otherwise.
        """

    def process(self, clip: dict[str, Any]) -> bool:
        """Run detection on *clip* (delegates to :meth:`detect`)."""
        return self.detect(clip)


class Extractor(Processor):
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

    The generic :meth:`process` method delegates to :meth:`extract`.
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

    def process(self, clip: dict[str, Any]) -> list[dict[str, Any]]:
        """Run extraction on *clip* (delegates to :meth:`extract`)."""
        return self.extract(clip)
