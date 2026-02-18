"""Model loading and initialisation — delegates to the media type registry.

All embedding models are now owned by their respective
:class:`~vistatotes.media.base.MediaType` instances.  This module keeps its
original public API (``initialize_models``, ``get_clap_model``, etc.) as
thin wrappers so that existing callers continue to work unchanged.
"""

import gc

from config import MODELS_CACHE_DIR


def initialize_models() -> None:
    """Load all embedding models by iterating the media type registry.

    Each registered :class:`~vistatotes.media.base.MediaType` is responsible
    for loading its own model(s) via
    :meth:`~vistatotes.media.base.MediaType.load_models`.
    Calling this function a second time is a no-op because each implementation
    guards against re-loading with an early-return check.
    """
    print("DEBUG: initialize_models called", flush=True)

    MODELS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    import torch
    torch.set_num_threads(1)
    gc.collect()

    from vistatotes.media import all_types
    for media_type in all_types():
        media_type.load_models()

    print("DEBUG: All models loaded and ready", flush=True)


# ---------------------------------------------------------------------------
# Backward-compatible getter functions
#
# These return the model instances held by their respective MediaType objects.
# Existing callers that import these functions directly continue to work.
# ---------------------------------------------------------------------------

def get_clap_model():
    """Return ``(clap_model, clap_processor)`` from the audio media type."""
    from vistatotes.media import get as media_get
    return media_get("audio")._get_model_and_processor()


def get_xclip_model():
    """Return ``(xclip_model, xclip_processor)`` from the video media type."""
    from vistatotes.media import get as media_get
    return media_get("video")._get_model_and_processor()


def get_clip_model():
    """Return ``(clip_model, clip_processor)`` from the image media type."""
    from vistatotes.media import get as media_get
    return media_get("image")._get_model_and_processor()


def get_e5_model():
    """Return the E5 ``SentenceTransformer`` from the text media type."""
    from vistatotes.media import get as media_get
    return media_get("paragraph")._get_model()
