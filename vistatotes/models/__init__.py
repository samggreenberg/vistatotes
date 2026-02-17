"""Model loading, embeddings, and training utilities."""

from vistatotes.models.embeddings import (
    embed_audio_file,
    embed_image_file,
    embed_paragraph_file,
    embed_text_query,
    embed_video_file,
)
from vistatotes.models.loader import (
    get_clap_model,
    get_clip_model,
    get_e5_model,
    get_xclip_model,
    initialize_models,
)
from vistatotes.models.progress import analyze_labeling_progress
from vistatotes.models.training import (
    calculate_cross_calibration_threshold,
    calculate_gmm_threshold,
    find_optimal_threshold,
    train_and_score,
    train_model,
)

__all__ = [
    # Embeddings
    "embed_audio_file",
    "embed_video_file",
    "embed_image_file",
    "embed_paragraph_file",
    "embed_text_query",
    # Loader
    "initialize_models",
    "get_clap_model",
    "get_xclip_model",
    "get_clip_model",
    "get_e5_model",
    # Training
    "train_model",
    "train_and_score",
    "calculate_gmm_threshold",
    "find_optimal_threshold",
    "calculate_cross_calibration_threshold",
    # Progress
    "analyze_labeling_progress",
]
