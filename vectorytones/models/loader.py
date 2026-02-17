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

from config import CLAP_MODEL_ID, CLIP_MODEL_ID, E5_MODEL_ID, XCLIP_MODEL_ID

# Model instances
clap_model: Optional[ClapModel] = None
clap_processor: Optional[ClapProcessor] = None
xclip_model: Optional[XCLIPModel] = None
xclip_processor: Optional[XCLIPProcessor] = None
clip_model: Optional[CLIPModel] = None
clip_processor: Optional[CLIPProcessor] = None
e5_model: Optional[SentenceTransformer] = None


def initialize_models() -> None:
    """Initialize all embedding models."""
    global clap_model, clap_processor, xclip_model, xclip_processor
    global clip_model, clip_processor, e5_model

    print("DEBUG: initialize_models called", flush=True)

    # Optimize for low-memory environments
    torch.set_num_threads(1)
    gc.collect()

    # Load CLAP model for Sounds modality
    if clap_model is None:
        print("DEBUG: Loading CLAP model for Sounds (Hugging Face)...", flush=True)
        clap_model = ClapModel.from_pretrained(CLAP_MODEL_ID, low_cpu_mem_usage=True)
        clap_processor = ClapProcessor.from_pretrained(CLAP_MODEL_ID)
        print("DEBUG: CLAP model loaded.", flush=True)

    # Load X-CLIP model for Videos modality
    if xclip_model is None:
        print("DEBUG: Loading X-CLIP model for Videos (Hugging Face)...", flush=True)
        xclip_model = XCLIPModel.from_pretrained(XCLIP_MODEL_ID, low_cpu_mem_usage=True)
        xclip_processor = XCLIPProcessor.from_pretrained(XCLIP_MODEL_ID)
        print("DEBUG: X-CLIP model loaded.", flush=True)

    # Load CLIP model for Images modality
    if clip_model is None:
        print("DEBUG: Loading CLIP model for Images (Hugging Face)...", flush=True)
        clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID, low_cpu_mem_usage=True)
        clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
        print("DEBUG: CLIP model loaded.", flush=True)

    # Load E5-LARGE-V2 model for Paragraphs modality
    if e5_model is None:
        print(
            "DEBUG: Loading E5-LARGE-V2 model for Paragraphs (SentenceTransformers)...",
            flush=True,
        )
        e5_model = SentenceTransformer(E5_MODEL_ID)
        print("DEBUG: E5-LARGE-V2 model loaded.", flush=True)

    print("DEBUG: All models loaded and ready", flush=True)


def get_clap_model() -> tuple[Optional[ClapModel], Optional[ClapProcessor]]:
    """Get CLAP model and processor."""
    return clap_model, clap_processor


def get_xclip_model() -> tuple[Optional[XCLIPModel], Optional[XCLIPProcessor]]:
    """Get X-CLIP model and processor."""
    return xclip_model, xclip_processor


def get_clip_model() -> tuple[Optional[CLIPModel], Optional[CLIPProcessor]]:
    """Get CLIP model and processor."""
    return clip_model, clip_processor


def get_e5_model() -> Optional[SentenceTransformer]:
    """Get E5 model."""
    return e5_model
