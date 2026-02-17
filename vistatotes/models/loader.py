"""Model loading and initialization for embeddings."""

import gc
from typing import Optional

import torch
from sentence_transformers import SentenceTransformer
from transformers import (
    ClapModel,
    ClapProcessor,
    CLIPModel,
    CLIPProcessor,
    XCLIPModel,
    XCLIPProcessor,
)

from config import (
    CLAP_MODEL_ID,
    CLIP_MODEL_ID,
    E5_MODEL_ID,
    MODELS_CACHE_DIR,
    XCLIP_MODEL_ID,
)

# Model instances
clap_model: Optional[ClapModel] = None
clap_processor: Optional[ClapProcessor] = None
xclip_model: Optional[XCLIPModel] = None
xclip_processor: Optional[XCLIPProcessor] = None
clip_model: Optional[CLIPModel] = None
clip_processor: Optional[CLIPProcessor] = None
e5_model: Optional[SentenceTransformer] = None


def initialize_models() -> None:
    """Download (if necessary) and load all four embedding models into memory.

    Models are loaded lazily: a model is only fetched and instantiated if its
    global variable is currently ``None``. Calling this function a second time
    is therefore a no-op if all models are already loaded. Models are cached to
    ``MODELS_CACHE_DIR`` so that subsequent startups do not require a network
    connection.

    Loaded models (one per media modality):

    - **CLAP** (``CLAP_MODEL_ID``) – audio embeddings.
    - **X-CLIP** (``XCLIP_MODEL_ID``) – video embeddings.
    - **CLIP** (``CLIP_MODEL_ID``) – image embeddings.
    - **E5-large-v2** (``E5_MODEL_ID``) – paragraph embeddings.

    Sets ``torch.set_num_threads(1)`` and calls ``gc.collect()`` before loading
    to reduce peak memory usage in constrained environments.
    """
    global clap_model, clap_processor, xclip_model, xclip_processor
    global clip_model, clip_processor, e5_model

    print("DEBUG: initialize_models called", flush=True)

    # Create models cache directory if it doesn't exist
    MODELS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_dir = str(MODELS_CACHE_DIR)

    # Optimize for low-memory environments
    torch.set_num_threads(1)
    gc.collect()

    # Load CLAP model for Sounds modality
    if clap_model is None:
        print("DEBUG: Loading CLAP model for Sounds (Hugging Face)...", flush=True)
        clap_model = ClapModel.from_pretrained(
            CLAP_MODEL_ID, low_cpu_mem_usage=True, cache_dir=cache_dir
        )
        clap_processor = ClapProcessor.from_pretrained(CLAP_MODEL_ID, cache_dir=cache_dir)
        print("DEBUG: CLAP model loaded.", flush=True)

    # Load X-CLIP model for Videos modality
    if xclip_model is None:
        print("DEBUG: Loading X-CLIP model for Videos (Hugging Face)...", flush=True)
        xclip_model = XCLIPModel.from_pretrained(
            XCLIP_MODEL_ID, low_cpu_mem_usage=True, cache_dir=cache_dir
        )
        xclip_processor = XCLIPProcessor.from_pretrained(XCLIP_MODEL_ID, cache_dir=cache_dir)
        print("DEBUG: X-CLIP model loaded.", flush=True)

    # Load CLIP model for Images modality
    if clip_model is None:
        print("DEBUG: Loading CLIP model for Images (Hugging Face)...", flush=True)
        clip_model = CLIPModel.from_pretrained(
            CLIP_MODEL_ID, low_cpu_mem_usage=True, cache_dir=cache_dir
        )
        clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID, cache_dir=cache_dir)
        print("DEBUG: CLIP model loaded.", flush=True)

    # Load E5-LARGE-V2 model for Paragraphs modality
    if e5_model is None:
        print(
            "DEBUG: Loading E5-LARGE-V2 model for Paragraphs (SentenceTransformers)...",
            flush=True,
        )
        e5_model = SentenceTransformer(E5_MODEL_ID, cache_folder=cache_dir)
        print("DEBUG: E5-LARGE-V2 model loaded.", flush=True)

    print("DEBUG: All models loaded and ready", flush=True)


def get_clap_model() -> tuple[Optional[ClapModel], Optional[ClapProcessor]]:
    """Return the global CLAP model and processor instances.

    Returns:
        A 2-tuple ``(clap_model, clap_processor)`` where each element is either
        the loaded instance or ``None`` if :func:`initialize_models` has not
        been called yet.
    """
    return clap_model, clap_processor


def get_xclip_model() -> tuple[Optional[XCLIPModel], Optional[XCLIPProcessor]]:
    """Return the global X-CLIP model and processor instances.

    Returns:
        A 2-tuple ``(xclip_model, xclip_processor)`` where each element is
        either the loaded instance or ``None`` if :func:`initialize_models` has
        not been called yet.
    """
    return xclip_model, xclip_processor


def get_clip_model() -> tuple[Optional[CLIPModel], Optional[CLIPProcessor]]:
    """Return the global CLIP model and processor instances.

    Returns:
        A 2-tuple ``(clip_model, clip_processor)`` where each element is either
        the loaded instance or ``None`` if :func:`initialize_models` has not
        been called yet.
    """
    return clip_model, clip_processor


def get_e5_model() -> Optional[SentenceTransformer]:
    """Return the global E5-large-v2 SentenceTransformer instance.

    Returns:
        The loaded ``SentenceTransformer`` model, or ``None`` if
        :func:`initialize_models` has not been called yet.
    """
    return e5_model
