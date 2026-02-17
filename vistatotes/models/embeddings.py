"""Embedding generation for different media types."""

from pathlib import Path
from typing import Optional

import cv2
import librosa
import numpy as np
import torch
from PIL import Image

from config import SAMPLE_RATE
from vistatotes.models.loader import (
    get_clap_model,
    get_clip_model,
    get_e5_model,
    get_xclip_model,
)


def embed_audio_file(audio_path: Path) -> Optional[np.ndarray]:
    """Generate CLAP embedding for a single audio file."""
    clap_model, clap_processor = get_clap_model()
    if clap_model is None or clap_processor is None:
        return None

    try:
        # Load audio at 48kHz for CLAP
        audio_data, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

        # Get embedding from CLAP
        inputs = clap_processor(
            audio=audio_data,
            sampling_rate=SAMPLE_RATE,  # type: ignore
            return_tensors="pt",  # type: ignore
            padding="max_length",  # type: ignore
            max_length=480000,  # type: ignore
            truncation=True,  # type: ignore
        )  # type: ignore
        with torch.no_grad():
            outputs = clap_model.audio_model(**inputs)
            embedding = (
                clap_model.audio_projection(outputs.pooler_output)
                .detach()
                .cpu()
                .numpy()
            )

        return embedding[0]
    except Exception as e:
        print(f"Error embedding {audio_path}: {e}")
        return None


def embed_video_file(video_path: Path) -> Optional[np.ndarray]:
    """Generate X-CLIP embedding for a single video file."""
    xclip_model, xclip_processor = get_xclip_model()
    if xclip_model is None or xclip_processor is None:
        return None

    try:
        # Load video with OpenCV
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error opening video {video_path}")
            return None

        frames = []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Sample 8 frames evenly spaced throughout the video (X-CLIP default)
        num_frames = 8
        if frame_count < num_frames:
            num_frames = max(1, frame_count)

        indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB and then to PIL Image
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(frame)
                frames.append(pil_frame)

        cap.release()

        if len(frames) == 0:
            print(f"Error: could not extract frames from {video_path}")
            return None

        # Process frames with X-CLIP processor
        # X-CLIP expects a list of PIL Images
        inputs = xclip_processor(videos=list(frames), return_tensors="pt")  # type: ignore

        # Get embedding from X-CLIP
        with torch.no_grad():
            outputs = xclip_model.get_video_features(**inputs)
            embedding = outputs.detach().cpu().numpy()  # type: ignore

        return embedding[0]
    except Exception as e:
        print(f"Error embedding {video_path}: {e}")
        return None


def embed_image_file(image_path: Path) -> Optional[np.ndarray]:
    """Generate CLIP embedding for a single image file."""
    clip_model, clip_processor = get_clip_model()
    if clip_model is None or clip_processor is None:
        return None

    try:
        # Load image with PIL
        image = Image.open(image_path).convert("RGB")

        # Process image with CLIP processor
        inputs = clip_processor(images=image, return_tensors="pt")  # type: ignore

        # Get embedding from CLIP vision model
        with torch.no_grad():
            outputs = clip_model.get_image_features(**inputs)
            embedding = outputs.detach().cpu().numpy()  # type: ignore

        return embedding[0]
    except Exception as e:
        print(f"Error embedding {image_path}: {e}")
        return None


def embed_paragraph_file(text_path: Path) -> Optional[np.ndarray]:
    """Generate E5-LARGE-V2 embedding for a text/paragraph file."""
    e5_model = get_e5_model()
    if e5_model is None:
        return None

    try:
        # Read text file
        with open(text_path, "r", encoding="utf-8") as f:
            text_content = f.read().strip()

        if not text_content:
            print(f"Warning: empty text file {text_path}")
            return None

        # Embed with E5-LARGE-V2 using "passage:" prefix
        # This is important for E5 model to distinguish between passages and queries
        embedding = e5_model.encode(
            f"passage: {text_content}", normalize_embeddings=True
        )

        return embedding  # type: ignore
    except Exception as e:
        print(f"Error embedding {text_path}: {e}")
        return None


def embed_text_query(text: str, media_type: str) -> Optional[np.ndarray]:
    """Embed a text query for the given media type.

    Args:
        text: The text query to embed
        media_type: One of "audio", "video", "image", "paragraph"

    Returns:
        The text embedding as a numpy array, or None if failed
    """
    if media_type == "audio":
        clap_model, clap_processor = get_clap_model()
        if clap_model is None or clap_processor is None:
            return None
        inputs = clap_processor(text=[text], return_tensors="pt")  # type: ignore
        with torch.no_grad():
            outputs = clap_model.text_model(**inputs)
            text_vec = (
                clap_model.text_projection(outputs.pooler_output)
                .detach()
                .cpu()
                .numpy()[0]
            )
        return text_vec

    elif media_type == "video":
        xclip_model, xclip_processor = get_xclip_model()
        if xclip_model is None or xclip_processor is None:
            return None
        inputs = xclip_processor(text=[text], return_tensors="pt")  # type: ignore
        with torch.no_grad():
            text_vec = xclip_model.get_text_features(**inputs).detach().cpu().numpy()[0]  # type: ignore
        return text_vec

    elif media_type == "image":
        clip_model, clip_processor = get_clip_model()
        if clip_model is None or clip_processor is None:
            return None
        inputs = clip_processor(text=[text], return_tensors="pt")  # type: ignore
        with torch.no_grad():
            text_vec = clip_model.get_text_features(**inputs).detach().cpu().numpy()[0]  # type: ignore
        return text_vec

    elif media_type == "paragraph":
        e5_model = get_e5_model()
        if e5_model is None:
            return None
        text_vec = e5_model.encode(f"query: {text}", normalize_embeddings=True)
        return text_vec  # type: ignore

    return None
