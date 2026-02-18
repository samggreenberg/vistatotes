"""Embedding generation — delegates to the media type registry.

The actual embedding logic now lives inside each
:class:`~vistatotes.media.base.MediaType` implementation.  This module keeps
its original public API as thin wrappers so that existing callers
(``datasets/loader.py``, ``routes/sorting.py``, etc.) continue to work
without modification.
"""

from pathlib import Path
from typing import Optional

import numpy as np


def embed_audio_file(audio_path: Path) -> Optional[np.ndarray]:
    """Generate a CLAP audio embedding for *audio_path*.

    Delegates to :class:`~vistatotes.media.audio.media_type.AudioMediaType`.
    """
    from vistatotes.media import get as media_get
    return media_get("audio").embed_media(audio_path)


def embed_video_file(video_path: Path) -> Optional[np.ndarray]:
    """Generate an X-CLIP video embedding for *video_path*.

    Delegates to :class:`~vistatotes.media.video.media_type.VideoMediaType`.
    """
    from vistatotes.media import get as media_get
    return media_get("video").embed_media(video_path)


def embed_image_file(image_path: Path) -> Optional[np.ndarray]:
    """Generate a CLIP image embedding for *image_path*.

    Delegates to :class:`~vistatotes.media.image.media_type.ImageMediaType`.
    """
    from vistatotes.media import get as media_get
    return media_get("image").embed_media(image_path)


def embed_paragraph_file(text_path: Path) -> Optional[np.ndarray]:
    """Generate an E5-base-v2 embedding for *text_path*.

    Delegates to :class:`~vistatotes.media.text.media_type.TextMediaType`.
    """
    from vistatotes.media import get as media_get
    return media_get("paragraph").embed_media(text_path)


def embed_text_query(text: str, media_type: str) -> Optional[np.ndarray]:
    """Embed *text* in the vector space of the given *media_type*.

    Delegates to the registered :class:`~vistatotes.media.base.MediaType`'s
    :meth:`~vistatotes.media.base.MediaType.embed_text` method, so the
    resulting vector can be compared against media embeddings via cosine
    similarity.

    Args:
        text: The natural-language search query to embed.
        media_type: Internal type identifier, e.g. ``"audio"``, ``"video"``,
            ``"image"``, or ``"paragraph"``.

    Returns:
        A 1-D ``numpy.ndarray`` embedding, or ``None`` if the media type is
        not registered or the model is not loaded.
    """
    from vistatotes.media import get as media_get
    try:
        return media_get(media_type).embed_text(text)
    except KeyError:
        return None
