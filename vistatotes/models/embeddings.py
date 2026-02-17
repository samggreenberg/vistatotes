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
    """Generate a CLAP audio embedding vector for a single audio file.

    Loads the audio at the configured ``SAMPLE_RATE`` (48 kHz for CLAP),
    runs it through the CLAP audio encoder, and projects it to the shared
    audio-text embedding space.

    Args:
        audio_path: Path to an audio file in any format supported by
            ``librosa.load`` (WAV, MP3, FLAC, OGG, etc.).

    Returns:
        A 1-D ``numpy.ndarray`` of shape ``(512,)`` containing the CLAP audio
        embedding, or ``None`` if the CLAP model is not loaded or if an
        exception occurs during loading or embedding.
    """
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
    """Generate an X-CLIP video embedding vector for a single video file.

    Samples up to 8 frames evenly spaced throughout the video (the default
    expected by X-CLIP), converts them from BGR to RGB PIL Images, and passes
    them through the X-CLIP video encoder.

    Args:
        video_path: Path to a video file in any format supported by OpenCV
            (MP4, AVI, MOV, WEBM, MKV, etc.).

    Returns:
        A 1-D ``numpy.ndarray`` containing the X-CLIP video embedding, or
        ``None`` if the X-CLIP model is not loaded, the video cannot be opened,
        no frames could be extracted, or an exception occurs during processing.
    """
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
    """Generate a CLIP image embedding vector for a single image file.

    Opens the image with PIL, converts it to RGB, and runs it through the
    CLIP vision encoder to obtain a feature vector in the shared image-text
    embedding space.

    Args:
        image_path: Path to an image file in any format supported by PIL
            (JPEG, PNG, GIF, BMP, WEBP, etc.).

    Returns:
        A 1-D ``numpy.ndarray`` containing the CLIP image embedding, or
        ``None`` if the CLIP model is not loaded or if an exception occurs
        during loading or embedding.
    """
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
    """Generate an E5-large-v2 embedding vector for a plain-text file.

    Reads the file as UTF-8, prepends the ``"passage: "`` prefix required by
    the E5 asymmetric retrieval model, and encodes it with L2 normalisation.

    Args:
        text_path: Path to a UTF-8 encoded plain-text file (e.g. ``.txt`` or
            ``.md``).

    Returns:
        A 1-D ``numpy.ndarray`` containing the L2-normalised E5 embedding, or
        ``None`` if the E5 model is not loaded, the file is empty, or an
        exception occurs during reading or encoding.
    """
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
    """Embed a free-text search query into the embedding space for a given media type.

    Selects the appropriate text encoder based on ``media_type`` and embeds
    ``text`` so that it can be compared (via cosine similarity) against media
    embeddings produced by the corresponding media encoder.

    Model–media-type mapping:

    - ``"audio"``     → CLAP text encoder + projection head.
    - ``"video"``     → X-CLIP text encoder.
    - ``"image"``     → CLIP text encoder.
    - ``"paragraph"`` → E5-large-v2 with ``"query: "`` prefix.

    Args:
        text: The natural-language search query string to embed.
        media_type: The media modality the query should be matched against.
            Must be one of ``"audio"``, ``"video"``, ``"image"``, or
            ``"paragraph"``.

    Returns:
        A 1-D ``numpy.ndarray`` containing the text embedding in the shared
        embedding space for the requested media type, or ``None`` if:

        - The required model is not loaded.
        - ``media_type`` is not one of the four recognised values.
        - An exception occurs during encoding.
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
