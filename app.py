import os

# Limit threads to reduce memory overhead in constrained environments
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Visual feedback for startup
print("⏳ Initializing VectoryTones...", flush=True)

import csv
import gc
import hashlib
import io
import math
import pickle
import struct
import threading
import wave
import zipfile
from pathlib import Path
from typing import Optional

print("⏳ Importing ML libraries (this may take a few seconds)...", flush=True)

import cv2
import librosa
import numpy as np
import requests
import torch
import torch.nn as nn
from flask import Flask, jsonify, request, send_file
from PIL import Image
from sentence_transformers import SentenceTransformer
from sklearn.datasets import fetch_20newsgroups
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
from transformers import (
    ClapModel,
    ClapProcessor,
    CLIPModel,
    CLIPProcessor,
    XCLIPModel,
    XCLIPProcessor,
)

# Import refactored modules
from config import (
    AUDIO_DIR,
    CIFAR10_DOWNLOAD_SIZE_MB,
    CIFAR10_URL,
    CLIPS_PER_CATEGORY,
    CLIPS_PER_VIDEO_CATEGORY,
    DATA_DIR,
    EMBEDDINGS_DIR,
    ESC50_DOWNLOAD_SIZE_MB,
    ESC50_URL,
    IMAGE_DIR,
    IMAGES_PER_CIFAR10_CATEGORY,
    NUM_CLIPS,
    PARAGRAPH_DIR,
    SAMPLE_RATE,
    SAMPLE_VIDEOS_DOWNLOAD_SIZE_MB,
    SAMPLE_VIDEOS_URL,
    VIDEO_DIR,
)
from vectorytones.audio import generate_wav, wav_bytes_to_float
from vectorytones.datasets import DEMO_DATASETS
from vectorytones.models import (
    calculate_gmm_threshold,
    embed_audio_file,
    embed_image_file,
    embed_paragraph_file,
    embed_text_query,
    embed_video_file,
    get_clap_model,
    get_clip_model,
    get_e5_model,
    get_xclip_model,
    initialize_models,
    train_and_score,
)
from vectorytones.utils import (
    bad_votes,
    clips,
    get_inclusion,
    get_progress,
    good_votes,
    set_inclusion,
    update_progress,
)

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Clip generation
# ---------------------------------------------------------------------------


def init_clips():
    print("DEBUG: Generating synthetic waveforms...", flush=True)
    for i in range(1, NUM_CLIPS + 1):
        freq = 200 + (i - 1) * 50  # 200 Hz .. 1150 Hz
        duration = round(1.0 + (i % 5) * 0.5, 1)  # 1.0 – 3.0 s
        wav_bytes = generate_wav(freq, duration)
        clips[i] = {
            "id": i,
            "type": "audio",
            "frequency": freq,
            "duration": duration,
            "file_size": len(wav_bytes),
            "md5": hashlib.md5(wav_bytes).hexdigest(),
            "embedding": None,
            "wav_bytes": wav_bytes,
            "video_bytes": None,
            "filename": f"synthetic_{i}.wav",
            "category": "synthetic",
        }

    # Compute CLAP audio embeddings for all clips in one batch
    print("DEBUG: Converting audio to float arrays...", flush=True)
    audio_arrays = []
    max_len = 0
    for i in range(1, NUM_CLIPS + 1):
        arr = wav_bytes_to_float(clips[i]["wav_bytes"])
        audio_arrays.append(arr)
        max_len = max(max_len, len(arr))

    if clap_model is None or clap_processor is None:
        raise RuntimeError("CLAP model not loaded")
    else:
        print(
            f"DEBUG: Running CLAP inference on {len(audio_arrays)} items...", flush=True
        )
        inputs = clap_processor(
            audio=audio_arrays,
            return_tensors="pt",  # type: ignore
            padding="max_length",  # Pad to 10s to match HTSAT training # type: ignore
            max_length=480000,  # 48kHz * 10s # type: ignore
            truncation=True,  # type: ignore
            sampling_rate=SAMPLE_RATE,  # type: ignore
        )
        with torch.no_grad():
            outputs = clap_model.audio_model(**inputs)
            embeddings = (
                clap_model.audio_projection(outputs.pooler_output)
                .detach()
                .cpu()
                .numpy()
            )
        print("DEBUG: Inference complete.", flush=True)

    for i in range(1, NUM_CLIPS + 1):
        clips[i]["embedding"] = embeddings[i - 1]


# Model initialization is now handled by vectorytones.models.initialize_models()


# ---------------------------------------------------------------------------
# Dataset management functions
# ---------------------------------------------------------------------------


def clear_dataset():
    """Clear the current dataset."""
    global clips, good_votes, bad_votes
    clips.clear()
    good_votes.clear()
    bad_votes.clear()


def load_dataset_from_pickle(file_path: Path):
    """Load a dataset from a pickle file."""
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    global clips
    clips.clear()

    # Handle both old format (just clips dict) and new format (with metadata)
    if isinstance(data, dict) and "clips" in data:
        clips_data = data["clips"]
    else:
        clips_data = data

    # Convert to the app's clip format
    missing_media = 0
    for clip_id, clip_info in clips_data.items():
        # Determine media type
        media_type = clip_info.get("type", "audio")

        # Load the actual media file
        wav_bytes = None
        video_bytes = None
        image_bytes = None
        text_content = None
        media_bytes = None

        if media_type == "audio":
            if "wav_bytes" in clip_info:
                wav_bytes = clip_info["wav_bytes"]
                media_bytes = wav_bytes
            elif "filename" in clip_info and "audio_dir" in data:
                audio_path = Path(data["audio_dir"]) / clip_info["filename"]
                if audio_path.exists():
                    with open(audio_path, "rb") as f:
                        wav_bytes = f.read()
                        media_bytes = wav_bytes
                else:
                    missing_media += 1

        elif media_type == "video":
            if "video_bytes" in clip_info:
                video_bytes = clip_info["video_bytes"]
                media_bytes = video_bytes
            elif "filename" in clip_info and "video_dir" in data:
                video_path = Path(data["video_dir"]) / clip_info["filename"]
                if video_path.exists():
                    with open(video_path, "rb") as f:
                        video_bytes = f.read()
                        media_bytes = video_bytes
                else:
                    missing_media += 1

        elif media_type == "image":
            if "image_bytes" in clip_info:
                image_bytes = clip_info["image_bytes"]
                media_bytes = image_bytes
            elif "filename" in clip_info and "image_dir" in data:
                image_path = Path(data["image_dir"]) / clip_info["filename"]
                if image_path.exists():
                    with open(image_path, "rb") as f:
                        image_bytes = f.read()
                        media_bytes = image_bytes
                else:
                    missing_media += 1

        elif media_type == "paragraph":
            if "text_content" in clip_info:
                text_content = clip_info["text_content"]
                media_bytes = text_content.encode("utf-8")  # For MD5 hash
            elif "filename" in clip_info and "text_dir" in data:
                text_path = Path(data["text_dir"]) / clip_info["filename"]
                if text_path.exists():
                    with open(text_path, "r", encoding="utf-8") as f:
                        text_content = f.read()
                        media_bytes = text_content.encode("utf-8")
                else:
                    missing_media += 1

        if media_bytes:
            clip_data = {
                "id": clip_id,
                "type": media_type,
                "duration": clip_info.get("duration", 0),
                "file_size": clip_info.get("file_size", len(media_bytes)),
                "md5": hashlib.md5(media_bytes).hexdigest(),
                "embedding": np.array(clip_info["embedding"]),
                "wav_bytes": wav_bytes,
                "video_bytes": video_bytes,
                "image_bytes": image_bytes,
                "text_content": text_content,
                "filename": clip_info.get("filename", f"clip_{clip_id}.{media_type}"),
                "category": clip_info.get("category", "unknown"),
            }
            # Add media-specific metadata
            if media_type == "image":
                clip_data["width"] = clip_info.get("width")
                clip_data["height"] = clip_info.get("height")
            elif media_type == "paragraph":
                clip_data["word_count"] = clip_info.get("word_count")
                clip_data["character_count"] = clip_info.get("character_count")

            clips[clip_id] = clip_data

    if missing_media > 0:
        print(
            f"WARNING: {missing_media} media files missing from {file_path}", flush=True
        )


# Embedding functions are now in vectorytones.models.embeddings

def embed_image_file_from_pil(image: Image.Image) -> Optional[np.ndarray]:
    """Generate CLIP embedding for a PIL Image."""
    clip_model, clip_processor = get_clip_model()
    if clip_model is None or clip_processor is None:
        return None

    try:
        # Ensure RGB mode
        image = image.convert("RGB")

        # Process image with CLIP processor
        inputs = clip_processor(images=image, return_tensors="pt")

        # Get embedding from CLIP vision model
        with torch.no_grad():
            outputs = clip_model.get_image_features(**inputs)
            embedding = outputs.numpy()

        return embedding[0]
    except Exception as e:
        print(f"Error embedding image: {e}")
        return None


def load_dataset_from_folder(folder_path: Path, media_type: str = "sounds"):
    """Generate dataset from a folder of media files.

    Args:
        folder_path: Path to folder containing media files
        media_type: One of "sounds", "videos", "images", "paragraphs"
    """
    global clips

    update_progress("embedding", "Scanning media files...", 0, 0)

    # Define file extensions for each media type
    media_extensions = {
        "sounds": ["*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a"],
        "videos": ["*.mp4", "*.avi", "*.mov", "*.webm", "*.mkv"],
        "images": ["*.jpg", "*.jpeg", "*.png", "*.gif", "*.bmp", "*.webp"],
        "paragraphs": ["*.txt", "*.md"],
    }

    # Define embedding functions for each media type
    embed_functions = {
        "sounds": embed_audio_file,
        "videos": embed_video_file,
        "images": embed_image_file,
        "paragraphs": embed_paragraph_file,
    }

    # Map modality names to internal type strings
    type_mapping = {
        "sounds": "audio",
        "videos": "video",
        "images": "image",
        "paragraphs": "paragraph",
    }

    if media_type not in media_extensions:
        raise ValueError(f"Invalid media type: {media_type}")

    # Find all files of the specified media type
    media_files = []
    for ext in media_extensions[media_type]:
        media_files.extend(folder_path.glob(ext))

    if not media_files:
        raise ValueError(f"No {media_type} files found in folder")

    clips.clear()
    clip_id = 1

    # Process all media files
    total_files = len(media_files)
    embed_func = embed_functions[media_type]

    for i, file_path in enumerate(media_files):
        update_progress(
            "embedding",
            f"Embedding {media_type} {file_path.name}...",
            i + 1,
            total_files,
        )

        embedding = embed_func(file_path)
        if embedding is None:
            continue

        # Load file to get bytes
        with open(file_path, "rb") as f:
            file_bytes = f.read()

        # Initialize clip data
        clip_data = {
            "id": clip_id,
            "type": type_mapping[media_type],
            "file_size": len(file_bytes),
            "md5": hashlib.md5(file_bytes).hexdigest(),
            "embedding": embedding,
            "filename": file_path.name,
            "category": "custom",
            "wav_bytes": None,
            "video_bytes": None,
            "image_bytes": None,
            "text_content": None,
            "duration": 0,
        }

        # Set media-specific fields
        if media_type == "sounds":
            clip_data["wav_bytes"] = file_bytes
            # Get duration
            try:
                audio_data, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
                clip_data["duration"] = len(audio_data) / sr
            except Exception:
                pass

        elif media_type == "videos":
            clip_data["video_bytes"] = file_bytes
            # Get duration using OpenCV
            try:
                cap = cv2.VideoCapture(str(file_path))
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                clip_data["duration"] = frame_count / fps if fps > 0 else 0
                cap.release()
            except Exception:
                pass

        elif media_type == "images":
            clip_data["image_bytes"] = file_bytes
            # Get image dimensions
            try:
                img = Image.open(file_path)
                clip_data["width"] = img.width
                clip_data["height"] = img.height
            except Exception:
                pass

        elif media_type == "paragraphs":
            # Store text content directly (not as bytes)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text_content = f.read().strip()
                clip_data["text_content"] = text_content
                clip_data["word_count"] = len(text_content.split())
                clip_data["character_count"] = len(text_content)
            except Exception:
                pass

        clips[clip_id] = clip_data
        clip_id += 1

    update_progress("idle", f"Loaded {len(clips)} {media_type} clips from folder")


def download_file_with_progress(url: str, dest_path: Path, expected_size: int = 0):
    """Download a file with progress tracking."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))
    if total_size == 0:
        total_size = expected_size

    downloaded = 0
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            size = f.write(chunk)
            downloaded += size
            update_progress(
                "downloading", "Downloading ESC-50...", downloaded, total_size
            )


def download_esc50() -> Path:
    """Download and extract ESC-50 dataset."""
    zip_path = DATA_DIR / "esc50.zip"
    DATA_DIR.mkdir(exist_ok=True)

    if not zip_path.exists():
        update_progress("downloading", "Starting download...", 0, 0)
        download_file_with_progress(
            ESC50_URL, zip_path, ESC50_DOWNLOAD_SIZE_MB * 1024 * 1024
        )

    extract_dir = DATA_DIR / "ESC-50-master"
    if not extract_dir.exists():
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            members = zip_ref.namelist()
            total = len(members)
            for i, member in enumerate(members, 1):
                update_progress(
                    "downloading",
                    f"Extracting {member.split('/')[-1]}...",
                    i,
                    total,
                )
                zip_ref.extract(member, DATA_DIR)

    return extract_dir / "audio"


def load_esc50_metadata(esc50_dir: Path) -> dict:
    """Load ESC-50 metadata CSV."""
    meta_file = esc50_dir / "meta" / "esc50.csv"

    metadata = {}
    with open(meta_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["filename"]
            metadata[filename] = {
                "category": row["category"],
                "esc10": row["esc10"] == "True",
                "target": int(row["target"]),
                "fold": int(row["fold"]),
            }
    return metadata


def download_cifar10() -> Path:
    """Download and extract CIFAR-10 dataset."""
    import tarfile

    tar_path = DATA_DIR / "cifar-10-python.tar.gz"
    DATA_DIR.mkdir(exist_ok=True)

    if not tar_path.exists():
        update_progress("downloading", "Starting CIFAR-10 download...", 0, 0)
        download_file_with_progress(
            CIFAR10_URL, tar_path, CIFAR10_DOWNLOAD_SIZE_MB * 1024 * 1024
        )

    extract_dir = DATA_DIR / "cifar-10-batches-py"
    if not extract_dir.exists():
        update_progress("downloading", "Extracting CIFAR-10...", 0, 0)
        with tarfile.open(tar_path, "r:gz") as tar_ref:
            tar_ref.extractall(DATA_DIR)

    return extract_dir


def load_cifar10_batch(file_path: Path) -> tuple:
    """Load a CIFAR-10 batch file and return images and labels."""
    with open(file_path, "rb") as f:
        batch = pickle.load(f, encoding="bytes")

    # CIFAR-10 label names
    label_names = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    images = batch[b"data"]
    labels = batch[b"labels"]

    # Reshape images from (10000, 3072) to (10000, 32, 32, 3)
    images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    return images, labels, label_names


def download_ucf101_subset() -> Path:
    """Download UCF-101 action recognition videos.

    Note: UCF-101 is distributed as a RAR file. For demo purposes, we'll
    try to download from a mirror or use a smaller subset if available.
    """
    # Try to download from a ZIP mirror or subset
    video_dir = VIDEO_DIR / "ucf101"
    VIDEO_DIR.mkdir(exist_ok=True, parents=True)

    # For now, check if videos already exist
    if video_dir.exists() and any(video_dir.glob("*/*.avi")):
        return video_dir

    # If not available, raise an error with instructions
    raise ValueError(
        "UCF-101 video dataset not found. To use video datasets:\n"
        "1. Download UCF-101 from https://www.crcv.ucf.edu/data/UCF101.php\n"
        "2. Extract to data/video/ucf101/ directory\n"
        "3. Or use 'Load from Folder' to import your own video files\n\n"
        "The UCF-101 dataset is ~6.5GB and distributed as a RAR file.\n"
        "For automatic download support, we recommend using smaller datasets\n"
        "or organizing your own video collection in folders by category."
    )


def load_video_metadata_from_folders(video_dir: Path, categories: list[str]) -> dict:
    """Load video file metadata from category folders."""
    metadata = {}

    for category_folder in video_dir.iterdir():
        if not category_folder.is_dir():
            continue

        category_name = category_folder.name
        if category_name not in categories:
            continue

        # Find all video files in this category
        for ext in ["*.mp4", "*.avi", "*.mov", "*.webm", "*.mkv"]:
            for video_path in category_folder.glob(ext):
                metadata[video_path.name] = {
                    "category": category_name,
                    "path": video_path,
                }

    return metadata


def load_image_metadata_from_folders(image_dir: Path, categories: list[str]) -> dict:
    """Load image file metadata from category folders."""
    metadata = {}

    for category_folder in image_dir.iterdir():
        if not category_folder.is_dir():
            continue

        category_name = category_folder.name
        if category_name not in categories:
            continue

        # Find all image files in this category
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp", "*.webp"]:
            for image_path in category_folder.glob(ext):
                metadata[image_path.name] = {
                    "category": category_name,
                    "path": image_path,
                }

    return metadata


def load_paragraph_metadata_from_folders(text_dir: Path, categories: list[str]) -> dict:
    """Load paragraph/text file metadata from category folders."""
    metadata = {}

    for category_folder in text_dir.iterdir():
        if not category_folder.is_dir():
            continue

        category_name = category_folder.name
        if category_name not in categories:
            continue

        # Find all text files in this category
        for ext in ["*.txt", "*.md"]:
            for text_path in category_folder.glob(ext):
                metadata[text_path.name] = {
                    "category": category_name,
                    "path": text_path,
                }

    return metadata


def download_20newsgroups(categories: list[str]) -> tuple:
    """Download and prepare 20 Newsgroups dataset.

    Returns:
        tuple: (texts, labels, category_names) where texts is list of strings,
               labels is list of category indices, and category_names is list of category names
    """
    update_progress("downloading", "Downloading 20 Newsgroups dataset...", 0, 0)

    # Map our category names to 20 newsgroups categories
    # We'll use a subset that maps well to common news categories
    category_mapping = {
        "world": "talk.politics.misc",
        "sports": "rec.sport.baseball",
        "business": "misc.forsale",
        "science": "sci.space",
    }

    # Get the actual newsgroup categories to download
    newsgroup_categories = [category_mapping.get(cat, cat) for cat in categories]

    # Download the dataset (sklearn handles caching automatically)
    newsgroups = fetch_20newsgroups(
        subset="train",
        categories=newsgroup_categories,
        remove=("headers", "footers", "quotes"),
        shuffle=True,
        random_state=42,
    )

    # Map back to our category names
    texts = newsgroups.data
    labels = newsgroups.target
    target_names = [
        list(category_mapping.keys())[
            list(category_mapping.values()).index(newsgroups.target_names[i])
        ]
        if newsgroups.target_names[i] in category_mapping.values()
        else newsgroups.target_names[i]
        for i in range(len(newsgroups.target_names))
    ]

    return texts, labels, target_names


def load_demo_dataset(dataset_name: str):
    """Load a demo dataset, downloading and embedding if necessary."""
    global clips

    if dataset_name not in DEMO_DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    dataset_info = DEMO_DATASETS[dataset_name]
    media_type = dataset_info.get("media_type", "audio")

    # Check if already embedded
    pkl_file = EMBEDDINGS_DIR / f"{dataset_name}.pkl"
    if pkl_file.exists():
        update_progress("loading", f"Loading {dataset_name} dataset...", 0, 0)
        load_dataset_from_pickle(pkl_file)

        # Check if any clips were actually loaded
        if len(clips) == 0:
            # Pickle file exists but media files are missing, delete and re-embed
            update_progress(
                "loading", f"Media files missing, re-embedding {dataset_name}...", 0, 0
            )
            pkl_file.unlink()
        else:
            update_progress("idle", f"Loaded {dataset_name} dataset")
            return

    # Process based on media type
    if media_type == "image":
        # Handle image datasets
        image_source = dataset_info.get("source", "cifar10_sample")

        if image_source == "cifar10_sample":
            # Download CIFAR-10 if needed
            cifar_dir = download_cifar10()

            # Load CIFAR-10 training batch
            batch_file = cifar_dir / "data_batch_1"
            images, labels, label_names = load_cifar10_batch(batch_file)

            # Filter to requested categories
            category_indices = {label_names[i]: i for i in range(len(label_names))}
            requested_categories = dataset_info["categories"]

            # Collect images for requested categories
            selected_images = []
            selected_labels = []

            for cat in requested_categories:
                if cat in category_indices:
                    cat_idx = category_indices[cat]
                    # Get first N images of this category
                    cat_mask = [i for i, lbl in enumerate(labels) if lbl == cat_idx]
                    for idx in cat_mask[:IMAGES_PER_CIFAR10_CATEGORY]:
                        selected_images.append(images[idx])
                        selected_labels.append(cat)

            # Generate embeddings for images
            clips.clear()
            clip_id = 1
            total = len(selected_images)
            update_progress(
                "embedding", f"Starting embedding for {total} images...", 0, total
            )

            for i, (image_array, category) in enumerate(
                zip(selected_images, selected_labels)
            ):
                update_progress(
                    "embedding",
                    f"Embedding {category}: image {i + 1}/{total}",
                    i + 1,
                    total,
                )

                # Convert numpy array to PIL Image
                img = Image.fromarray(image_array.astype("uint8"), "RGB")

                # Convert to bytes
                img_buffer = io.BytesIO()
                img.save(img_buffer, format="PNG")
                image_bytes = img_buffer.getvalue()

                # Get embedding
                embedding = embed_image_file_from_pil(img)
                if embedding is None:
                    continue

                clips[clip_id] = {
                    "id": clip_id,
                    "type": "image",
                    "duration": 0,  # Images don't have duration
                    "file_size": len(image_bytes),
                    "md5": hashlib.md5(image_bytes).hexdigest(),
                    "embedding": embedding,
                    "wav_bytes": None,
                    "video_bytes": None,
                    "image_bytes": image_bytes,
                    "filename": f"{category}_{clip_id}.png",
                    "category": category,
                    "width": img.width,
                    "height": img.height,
                }
                clip_id += 1

            # Save for future use
            EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
            with open(pkl_file, "wb") as f:
                pickle.dump(
                    {
                        "name": dataset_name,
                        "clips": {
                            cid: {
                                k: v.tolist() if isinstance(v, np.ndarray) else v
                                for k, v in clip.items()
                                if k not in ["wav_bytes", "video_bytes", "image_bytes"]
                            }
                            for cid, clip in clips.items()
                        },
                    },
                    f,
                )

            update_progress("idle", f"Loaded {dataset_name} dataset")
            return

    elif media_type == "paragraph":
        # Handle paragraph datasets
        paragraph_source = dataset_info.get("source", "ag_news_sample")

        if paragraph_source == "ag_news_sample":
            # Use 20 Newsgroups dataset from scikit-learn
            texts, labels, category_names = download_20newsgroups(
                dataset_info["categories"]
            )

            # Limit number of texts per category for demo
            max_per_category = 50
            selected_texts = []
            selected_categories = []

            for cat_name in dataset_info["categories"]:
                if cat_name in category_names:
                    cat_idx = category_names.index(cat_name)
                    cat_texts = [
                        texts[i] for i, lbl in enumerate(labels) if lbl == cat_idx
                    ]
                    # Limit to max_per_category
                    for text in cat_texts[:max_per_category]:
                        selected_texts.append(text)
                        selected_categories.append(cat_name)

            # Generate embeddings for paragraphs
            clips.clear()
            clip_id = 1
            total = len(selected_texts)
            update_progress(
                "embedding",
                f"Starting embedding for {total} paragraphs...",
                0,
                total,
            )

            for i, (text_content, category) in enumerate(
                zip(selected_texts, selected_categories)
            ):
                update_progress(
                    "embedding",
                    f"Embedding {category}: paragraph {i + 1}/{total}",
                    i + 1,
                    total,
                )

                # Truncate very long texts (keep first 1000 chars for demo)
                text_content = text_content[:1000].strip()
                if not text_content:
                    continue

                # Embed with E5-LARGE-V2 using "passage:" prefix
                if e5_model is None:
                    continue

                try:
                    embedding = e5_model.encode(
                        f"passage: {text_content}", normalize_embeddings=True
                    )
                except Exception as e:
                    print(f"Error embedding paragraph: {e}")
                    continue

                word_count = len(text_content.split())
                character_count = len(text_content)
                text_bytes = text_content.encode("utf-8")

                clips[clip_id] = {
                    "id": clip_id,
                    "type": "paragraph",
                    "duration": 0,  # Paragraphs don't have duration
                    "file_size": len(text_bytes),
                    "md5": hashlib.md5(text_bytes).hexdigest(),
                    "embedding": embedding,
                    "wav_bytes": None,
                    "video_bytes": None,
                    "image_bytes": None,
                    "text_content": text_content,
                    "filename": f"{category}_{clip_id}.txt",
                    "category": category,
                    "word_count": word_count,
                    "character_count": character_count,
                }
                clip_id += 1

            # Save for future use
            EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
            with open(pkl_file, "wb") as f:
                pickle.dump(
                    {
                        "name": dataset_name,
                        "clips": {
                            cid: {
                                k: v.tolist() if isinstance(v, np.ndarray) else v
                                for k, v in clip.items()
                                if k
                                not in [
                                    "wav_bytes",
                                    "video_bytes",
                                    "image_bytes",
                                    "text_content",
                                ]
                            }
                            for cid, clip in clips.items()
                        },
                    },
                    f,
                )

            update_progress("idle", f"Loaded {dataset_name} dataset")
            return

    elif media_type == "video":
        # Handle video datasets
        video_source = dataset_info.get("source", "ucf101")

        if video_source == "ucf101":
            try:
                video_dir = download_ucf101_subset()
            except ValueError as e:
                # If UCF-101 is not available, provide helpful error message
                update_progress("idle", "")
                raise e

            metadata = load_video_metadata_from_folders(
                video_dir, dataset_info["categories"]
            )
            video_files = [(meta["path"], meta) for meta in metadata.values()]

            # Generate embeddings for videos
            clips.clear()
            clip_id = 1
            total = len(video_files)
            update_progress(
                "embedding", f"Starting embedding for {total} video files...", 0, total
            )

            for i, (video_path, meta) in enumerate(video_files):
                update_progress(
                    "embedding",
                    f"Embedding {meta['category']}: {video_path.name} ({i + 1}/{total})",
                    i + 1,
                    total,
                )

                embedding = embed_video_file(video_path)
                if embedding is None:
                    continue

                with open(video_path, "rb") as f:
                    video_bytes = f.read()

                # Get duration using OpenCV
                try:
                    cap = cv2.VideoCapture(str(video_path))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    duration = frame_count / fps if fps > 0 else 0
                    cap.release()
                except Exception:
                    duration = 0

                clips[clip_id] = {
                    "id": clip_id,
                    "type": "video",
                    "duration": duration,
                    "file_size": len(video_bytes),
                    "md5": hashlib.md5(video_bytes).hexdigest(),
                    "embedding": embedding,
                    "wav_bytes": None,
                    "video_bytes": video_bytes,
                    "filename": video_path.name,
                    "category": meta["category"],
                }
                clip_id += 1

            # Save for future use
            EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
            with open(pkl_file, "wb") as f:
                pickle.dump(
                    {
                        "name": dataset_name,
                        "clips": {
                            cid: {
                                k: v.tolist() if isinstance(v, np.ndarray) else v
                                for k, v in clip.items()
                                if k not in ["wav_bytes", "video_bytes"]
                            }
                            for cid, clip in clips.items()
                        },
                        "video_dir": str(video_dir.absolute()),
                    },
                    f,
                )

            update_progress("idle", f"Loaded {dataset_name} dataset")
            return

    # Handle audio datasets (existing ESC-50 logic)
    audio_dir = download_esc50()
    metadata = load_esc50_metadata(audio_dir.parent)

    # Filter files for this dataset
    categories = DEMO_DATASETS[dataset_name]["categories"]
    audio_files = []
    for audio_path in sorted(audio_dir.glob("*.wav")):
        if audio_path.name in metadata:
            if metadata[audio_path.name]["category"] in categories:
                audio_files.append((audio_path, metadata[audio_path.name]))

    # Generate embeddings
    clips.clear()
    clip_id = 1
    total = len(audio_files)
    update_progress(
        "embedding", f"Starting embedding for {total} audio files...", 0, total
    )

    for i, (audio_path, meta) in enumerate(audio_files):
        update_progress(
            "embedding",
            f"Embedding {meta['category']}: {audio_path.name} ({i + 1}/{total})",
            i + 1,
            total,
        )

        embedding = embed_audio_file(audio_path)
        if embedding is None:
            continue

        with open(audio_path, "rb") as f:
            wav_bytes = f.read()

        try:
            audio_data, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
            duration = len(audio_data) / sr
        except Exception:
            duration = 0

        clips[clip_id] = {
            "id": clip_id,
            "type": "audio",
            "duration": duration,
            "file_size": len(wav_bytes),
            "md5": hashlib.md5(wav_bytes).hexdigest(),
            "embedding": embedding,
            "wav_bytes": wav_bytes,
            "video_bytes": None,
            "filename": audio_path.name,
            "category": meta["category"],
        }
        clip_id += 1

    # Save for future use
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(pkl_file, "wb") as f:
        pickle.dump(
            {
                "name": dataset_name,
                "clips": {
                    cid: {
                        k: v.tolist() if isinstance(v, np.ndarray) else v
                        for k, v in clip.items()
                        if k != "wav_bytes"
                    }
                    for cid, clip in clips.items()
                },
                "audio_dir": str(audio_dir.absolute()),
            },
            f,
        )

    update_progress("idle", f"Loaded {dataset_name} dataset")


def export_dataset_to_file() -> bytes:
    """Export the current dataset to a pickle file."""
    data = {
        "clips": {
            cid: {
                "id": clip["id"],
                "type": clip.get("type", "audio"),
                "duration": clip["duration"],
                "file_size": clip["file_size"],
                "md5": clip["md5"],
                "embedding": clip["embedding"].tolist()
                if isinstance(clip["embedding"], np.ndarray)
                else clip["embedding"],
                "filename": clip.get("filename", f"clip_{cid}.wav"),
                "category": clip.get("category", "unknown"),
                "wav_bytes": clip.get("wav_bytes"),
                "video_bytes": clip.get("video_bytes"),
            }
            for cid, clip in clips.items()
        }
    }

    buf = io.BytesIO()
    pickle.dump(data, buf)
    buf.seek(0)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.before_request
def log_request_info():
    """Log incoming requests to debug connection issues."""
    if "/api/dataset/progress" not in request.path:
        print(f"DEBUG: Incoming {request.method} {request.path}", flush=True)

@app.after_request
def log_response_info(response):
    if "/api/dataset/progress" not in request.path:
        print(f"DEBUG: Outgoing {response.status_code}", flush=True)
    return response


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/favicon.ico")
def favicon():
    # Return empty response if file doesn't exist to stop 404 logs
    if not (Path(app.root_path) / "static" / "favicon.ico").exists():
        return "", 204
    return app.send_static_file("favicon.ico")


@app.route("/api/clips")
def list_clips():
    result = []
    for c in clips.values():
        clip_data = {
            "id": c["id"],
            "type": c.get("type", "audio"),
            "duration": c["duration"],
            "file_size": c["file_size"],
            "filename": c.get("filename", f"clip_{c['id']}.wav"),
            "category": c.get("category", "unknown"),
            "md5": c["md5"],
        }
        # Only include frequency if it exists (for synthetic clips)
        if "frequency" in c:
            clip_data["frequency"] = c["frequency"]
        result.append(clip_data)
    return jsonify(result)


@app.route("/api/clips/<int:clip_id>/audio")
def clip_audio(clip_id):
    c = clips.get(clip_id)
    if not c:
        return jsonify({"error": "not found"}), 404
    return send_file(
        io.BytesIO(c["wav_bytes"]),
        mimetype="audio/wav",
        download_name=f"clip_{clip_id}.wav",
    )


@app.route("/api/clips/<int:clip_id>/video")
def clip_video(clip_id):
    c = clips.get(clip_id)
    if not c:
        return jsonify({"error": "not found"}), 404
    if c.get("type") != "video" or not c.get("video_bytes"):
        return jsonify({"error": "not a video clip"}), 400

    # Determine mimetype based on filename extension
    filename = c.get("filename", "")
    if filename.endswith(".webm"):
        mimetype = "video/webm"
    elif filename.endswith(".mov"):
        mimetype = "video/quicktime"
    elif filename.endswith(".avi"):
        mimetype = "video/x-msvideo"
    else:
        mimetype = "video/mp4"

    return send_file(
        io.BytesIO(c["video_bytes"]),
        mimetype=mimetype,
        download_name=f"clip_{clip_id}.mp4",
    )


@app.route("/api/clips/<int:clip_id>/image")
def clip_image(clip_id):
    c = clips.get(clip_id)
    if not c:
        return jsonify({"error": "not found"}), 404
    if c.get("type") != "image" or not c.get("image_bytes"):
        return jsonify({"error": "not an image clip"}), 400

    # Determine mimetype based on filename extension
    filename = c.get("filename", "")
    if filename.endswith(".png"):
        mimetype = "image/png"
    elif filename.endswith(".gif"):
        mimetype = "image/gif"
    elif filename.endswith(".webp"):
        mimetype = "image/webp"
    elif filename.endswith(".bmp"):
        mimetype = "image/bmp"
    else:
        mimetype = "image/jpeg"

    return send_file(
        io.BytesIO(c["image_bytes"]),
        mimetype=mimetype,
        download_name=f"clip_{clip_id}.jpg",
    )


@app.route("/api/clips/<int:clip_id>/paragraph")
def clip_paragraph(clip_id):
    c = clips.get(clip_id)
    if not c:
        return jsonify({"error": "not found"}), 404
    if c.get("type") != "paragraph" or not c.get("text_content"):
        return jsonify({"error": "not a paragraph clip"}), 400

    return jsonify(
        {
            "content": c.get("text_content", ""),
            "word_count": c.get("word_count", 0),
            "character_count": c.get("character_count", 0),
        }
    )


@app.route("/api/clips/<int:clip_id>/vote", methods=["POST"])
def vote_clip(clip_id):
    if clip_id not in clips:
        print(f"DEBUG: Vote failed - Clip {clip_id} not found", flush=True)
        return jsonify({"error": "not found"}), 404

    try:
        data = request.get_json(force=True)
    except Exception as e:
        print(f"DEBUG: Vote failed - JSON error: {e}", flush=True)
        return jsonify({"error": "Invalid request body"}), 400

    if data is None:
        return jsonify({"error": "Invalid request body"}), 400

    vote = data.get("vote")
    if vote not in ("good", "bad"):
        print(f"DEBUG: Vote failed - Invalid vote '{vote}'", flush=True)
        return jsonify({"error": "vote must be 'good' or 'bad'"}), 400

    if vote == "good":
        if clip_id in good_votes:
            good_votes.pop(clip_id, None)
        else:
            bad_votes.pop(clip_id, None)
            good_votes[clip_id] = None
    else:
        if clip_id in bad_votes:
            bad_votes.pop(clip_id, None)
        else:
            good_votes.pop(clip_id, None)
            bad_votes[clip_id] = None

    print(f"DEBUG: Vote '{vote}' recorded for clip {clip_id}", flush=True)
    return jsonify({"ok": True})


# ML training functions are now in vectorytones.models.training

@app.route("/api/sort", methods=["POST"])
def sort_clips():
    """Return clips sorted by cosine similarity to a text query."""
    try:
        data = request.get_json(force=True)
    except Exception as e:
        print(f"DEBUG: Sort failed - JSON error: {e}", flush=True)
        return jsonify({"error": "Invalid request body"}), 400

    if data is None:
        return jsonify({"error": "Invalid request body"}), 400

    text = data.get("text", "").strip()
    print(f"DEBUG: Sort request for '{text}'", flush=True)

    if not text:
        return jsonify({"error": "text is required"}), 400

    # Determine media type from current clips
    if not clips:
        print("DEBUG: Sort failed - No clips loaded", flush=True)
        return jsonify({"error": "No clips loaded"}), 400

    media_type = next(iter(clips.values())).get("type", "audio")

    # Embed text query using refactored module
    text_vec = embed_text_query(text, media_type)
    if text_vec is None:
        print(f"DEBUG: Sort failed - Could not embed text for media type {media_type}", flush=True)
        return jsonify({"error": f"Could not embed text for media type {media_type}"}), 500

    results = []
    scores = []
    for clip_id, clip in clips.items():
        media_vec = clip["embedding"]
        norm_product = np.linalg.norm(media_vec) * np.linalg.norm(text_vec)
        if norm_product == 0:
            similarity = 0.0
        else:
            similarity = float(np.dot(media_vec, text_vec) / norm_product)
        results.append({"id": clip_id, "similarity": round(similarity, 4)})
        scores.append(similarity)

    # Calculate GMM-based threshold
    threshold = calculate_gmm_threshold(scores)

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return jsonify({"results": results, "threshold": round(threshold, 4)})


@app.route("/api/learned-sort", methods=["POST"])
def learned_sort():
    """Train MLP on voted clips, return all clips sorted by predicted score."""
    if not good_votes or not bad_votes:
        return jsonify({"error": "need at least one good and one bad vote"}), 400
    results, threshold = train_and_score(clips, good_votes, bad_votes, get_inclusion())
    return jsonify({"results": results, "threshold": round(threshold, 4)})


@app.route("/api/votes")
def get_votes():
    return jsonify(
        {
            "good": list(good_votes),  # Maintains insertion order (dict keys)
            "bad": list(bad_votes),  # Maintains insertion order (dict keys)
        }
    )


@app.route("/api/labels/export")
def export_labels():
    """Export labels as JSON keyed by clip MD5 hash."""
    labels = []
    for cid in good_votes:  # Maintains insertion order
        clip = clips.get(cid)
        if clip:
            labels.append({"md5": clip["md5"], "label": "good"})
    for cid in bad_votes:  # Maintains insertion order
        clip = clips.get(cid)
        if clip:
            labels.append({"md5": clip["md5"], "label": "bad"})
    return jsonify({"labels": labels})


@app.route("/api/labels/import", methods=["POST"])
def import_labels():
    """Import labels from JSON, matching clips by MD5 hash."""
    data = request.get_json(force=True)
    if data is None:
        return jsonify({"error": "Invalid request body"}), 400

    labels = data.get("labels")
    if not isinstance(labels, list):
        return jsonify({"error": "labels must be a list"}), 400

    # Build MD5 -> clip ID lookup
    md5_to_id = {clip["md5"]: clip["id"] for clip in clips.values()}

    applied = 0
    skipped = 0
    for entry in labels:
        md5 = entry.get("md5")
        label = entry.get("label")
        if label not in ("good", "bad"):
            skipped += 1
            continue
        cid = md5_to_id.get(md5)
        if cid is None:
            skipped += 1
            continue

        # Apply the label, overriding any existing vote
        if label == "good":
            bad_votes.pop(cid, None)
            good_votes[cid] = None
        else:
            good_votes.pop(cid, None)
            bad_votes[cid] = None
        applied += 1

    return jsonify({"applied": applied, "skipped": skipped})


@app.route("/api/inclusion")
def get_inclusion():
    """Get the current Inclusion setting."""
    return jsonify({"inclusion": inclusion})


@app.route("/api/inclusion", methods=["POST"])
def set_inclusion():
    """Set the Inclusion setting."""
    global inclusion
    data = request.get_json(force=True)
    if data is None:
        return jsonify({"error": "Invalid request body"}), 400

    new_inclusion = data.get("inclusion")

    if not isinstance(new_inclusion, (int, float)):
        return jsonify({"error": "inclusion must be a number"}), 400

    # Clamp to -10 to +10 range
    new_inclusion = int(max(-10, min(10, new_inclusion)))
    inclusion = new_inclusion

    return jsonify({"inclusion": inclusion})


@app.route("/api/detector/export", methods=["POST"])
def export_detector():
    """Train MLP on current votes and export the model weights."""
    if not good_votes or not bad_votes:
        return jsonify({"error": "need at least one good and one bad vote"}), 400

    # Train the model
    X_list = []
    y_list = []
    for cid in good_votes:
        X_list.append(clips[cid]["embedding"])
        y_list.append(1.0)
    for cid in bad_votes:
        X_list.append(clips[cid]["embedding"])
        y_list.append(0.0)

    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    y = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1)

    input_dim = X.shape[1]

    # Calculate threshold using cross-calibration with inclusion
    threshold = calculate_cross_calibration_threshold(
        X_list, y_list, input_dim, inclusion
    )

    # Train final model on all data with inclusion
    model = train_model(X, y, input_dim, inclusion)

    # Extract model weights
    state_dict = model.state_dict()
    weights = {}
    for key, value in state_dict.items():
        weights[key] = value.tolist()

    return jsonify({"weights": weights, "threshold": round(threshold, 4)})


@app.route("/api/detector-sort", methods=["POST"])
def detector_sort():
    """Score all clips using a loaded detector model."""
    data = request.get_json(force=True)
    if data is None:
        return jsonify({"error": "Invalid request body"}), 400

    detector = data.get("detector")
    if not detector:
        return jsonify({"error": "detector is required"}), 400

    weights = detector.get("weights")
    threshold = detector.get("threshold", 0.5)

    if not weights:
        return jsonify({"error": "detector weights are required"}), 400

    # Reconstruct the model from weights
    # Determine input_dim from the first layer weights
    input_dim = len(weights["0.weight"][0])

    model = nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
        nn.Sigmoid(),
    )

    # Load weights
    state_dict = {}
    for key, value in weights.items():
        state_dict[key] = torch.tensor(value, dtype=torch.float32)
    model.load_state_dict(state_dict)
    model.eval()

    # Score every clip
    all_ids = sorted(clips.keys())
    all_embs = np.array([clips[cid]["embedding"] for cid in all_ids])
    X_all = torch.tensor(all_embs, dtype=torch.float32)
    with torch.no_grad():
        scores = model(X_all).squeeze(1).tolist()

    results = [{"id": cid, "score": round(s, 4)} for cid, s in zip(all_ids, scores)]
    results.sort(key=lambda x: x["score"], reverse=True)
    return jsonify({"results": results, "threshold": round(threshold, 4)})


@app.route("/api/example-sort", methods=["POST"])
def example_sort():
    """Sort clips by similarity to an uploaded example audio file."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    if clap_model is None or clap_processor is None:
        return jsonify({"error": "CLAP model not loaded"}), 500

    try:
        # Save uploaded file to temp location
        temp_path = DATA_DIR / "temp_example.wav"
        DATA_DIR.mkdir(exist_ok=True)
        file.save(temp_path)

        # Embed the example audio file
        example_embedding = embed_audio_file(temp_path)

        # Clean up temp file
        temp_path.unlink()

        if example_embedding is None:
            return jsonify({"error": "Failed to embed audio file"}), 500

        # Calculate cosine similarity with all clips
        results = []
        scores = []
        for clip_id, clip in clips.items():
            audio_vec = clip["embedding"]
            norm_product = np.linalg.norm(audio_vec) * np.linalg.norm(example_embedding)
            if norm_product == 0:
                similarity = 0.0
            else:
                similarity = float(np.dot(audio_vec, example_embedding) / norm_product)
            results.append({"id": clip_id, "similarity": round(similarity, 4)})
            scores.append(similarity)

        # Calculate GMM-based threshold
        threshold = calculate_gmm_threshold(scores)

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return jsonify({"results": results, "threshold": round(threshold, 4)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/label-file-sort", methods=["POST"])
def label_file_sort():
    """Train MLP on external audio files from a label file, then sort all clips."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    if clap_model is None or clap_processor is None:
        return jsonify({"error": "CLAP model not loaded"}), 500

    try:
        # Parse the label file
        text = file.read().decode("utf-8")
        label_data = None
        try:
            label_data = eval(text) if text.strip().startswith("{") else None
            if label_data is None:
                import json

                label_data = json.loads(text)
        except Exception:
            return jsonify({"error": "Invalid label file format"}), 400

        # Extract labels list
        labels = label_data.get("labels", [])
        if not labels:
            return jsonify({"error": "No labels found in file"}), 400

        # Load and embed each labeled audio file
        X_list = []
        y_list = []
        loaded_count = 0
        skipped_count = 0

        for entry in labels:
            label = entry.get("label")
            if label not in ("good", "bad"):
                skipped_count += 1
                continue

            # Try to get audio file path
            audio_path = entry.get("path") or entry.get("file") or entry.get("filename")
            if not audio_path:
                skipped_count += 1
                continue

            audio_path = Path(audio_path)
            if not audio_path.exists():
                skipped_count += 1
                continue

            # Embed the audio file
            embedding = embed_audio_file(audio_path)
            if embedding is None:
                skipped_count += 1
                continue

            X_list.append(embedding)
            y_list.append(1.0 if label == "good" else 0.0)
            loaded_count += 1

        if loaded_count < 2:
            return (
                jsonify(
                    {
                        "error": f"Need at least 2 valid labeled files (loaded {loaded_count}, skipped {skipped_count})"
                    }
                ),
                400,
            )

        # Check if we have both good and bad examples
        num_good = sum(1 for y in y_list if y == 1.0)
        num_bad = len(y_list) - num_good
        if num_good == 0 or num_bad == 0:
            return (
                jsonify(
                    {"error": "Need at least one good and one bad labeled example"}
                ),
                400,
            )

        # Train MLP using the same approach as learned sort
        X = torch.tensor(np.array(X_list), dtype=torch.float32)
        y = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1)

        input_dim = X.shape[1]

        # Calculate threshold using cross-calibration
        threshold = calculate_cross_calibration_threshold(
            X_list, y_list, input_dim, inclusion
        )

        # Train final model on all data
        model = train_model(X, y, input_dim, inclusion)

        # Score every clip in the dataset
        all_ids = sorted(clips.keys())
        all_embs = np.array([clips[cid]["embedding"] for cid in all_ids])
        X_all = torch.tensor(all_embs, dtype=torch.float32)
        with torch.no_grad():
            scores = model(X_all).squeeze(1).tolist()

        results = [{"id": cid, "score": round(s, 4)} for cid, s in zip(all_ids, scores)]
        results.sort(key=lambda x: x["score"], reverse=True)

        return jsonify(
            {
                "results": results,
                "threshold": round(threshold, 4),
                "loaded": loaded_count,
                "skipped": skipped_count,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Dataset management routes
# ---------------------------------------------------------------------------


@app.route("/api/dataset/status")
def dataset_status():
    """Return the current dataset status."""
    return jsonify(
        {
            "loaded": len(clips) > 0,
            "num_clips": len(clips),
            "has_votes": len(good_votes) + len(bad_votes) > 0,
        }
    )


@app.route("/api/dataset/progress")
def dataset_progress():
    """Return the current progress of long-running operations."""
    with progress_lock:
        return jsonify(progress_data.copy())


@app.route("/api/dataset/demo-list")
def demo_dataset_list():
    """List available demo datasets."""
    demos = []
    for name, dataset_info in DEMO_DATASETS.items():
        pkl_file = EMBEDDINGS_DIR / f"{name}.pkl"
        is_ready = pkl_file.exists()

        media_type = dataset_info.get("media_type", "audio")

        # Calculate number of files
        num_categories = len(dataset_info["categories"])
        if media_type == "video":
            # For video datasets, estimate based on available files or use default
            num_files = num_categories * CLIPS_PER_VIDEO_CATEGORY
        else:
            # For audio datasets (ESC-50 has 40 clips per category)
            num_files = num_categories * CLIPS_PER_CATEGORY

        # Calculate download size
        if is_ready:
            # If ready, show the actual .pkl file size
            download_size_mb = pkl_file.stat().st_size / (1024 * 1024)
        else:
            # If not ready, estimate download size
            if media_type == "video":
                # Check if video files exist
                video_source = dataset_info.get("source", "ucf101")
                if video_source == "ucf101":
                    video_dir = VIDEO_DIR / "ucf101"
                    if video_dir.exists():
                        # Videos are present, just need to embed
                        download_size_mb = 0
                        is_ready = False  # Not embedded yet, but videos available
                    else:
                        # Need to download/obtain videos (manual process for UCF-101)
                        download_size_mb = 0  # Manual download required
                else:
                    download_size_mb = SAMPLE_VIDEOS_DOWNLOAD_SIZE_MB
            elif media_type == "image":
                # CIFAR-10 dataset
                download_size_mb = CIFAR10_DOWNLOAD_SIZE_MB
            elif media_type == "paragraph":
                # 20 Newsgroups is small (scikit-learn downloads automatically)
                download_size_mb = 15  # Approximate size
            else:
                # Audio dataset - ESC-50 download
                download_size_mb = ESC50_DOWNLOAD_SIZE_MB

        demos.append(
            {
                "name": name,
                "ready": is_ready,
                "num_categories": num_categories,
                "num_files": num_files,
                "download_size_mb": round(download_size_mb, 1),
                "description": dataset_info.get("description", ""),
                "media_type": dataset_info.get("media_type", "audio"),
            }
        )
    return jsonify({"datasets": demos})


@app.route("/api/dataset/load-demo", methods=["POST"])
def load_demo_dataset_route():
    """Load a demo dataset in a background thread."""
    data = request.get_json(force=True)
    if data is None:
        return jsonify({"error": "Invalid request body"}), 400

    dataset_name = data.get("name")

    if not dataset_name or dataset_name not in DEMO_DATASETS:
        return jsonify({"error": "Invalid dataset name"}), 400

    def load_task():
        try:
            clear_dataset()
            load_demo_dataset(dataset_name)
        except Exception as e:
            update_progress("idle", "", 0, 0, str(e))

    thread = threading.Thread(target=load_task, daemon=True)
    thread.start()

    return jsonify({"ok": True, "message": "Loading started"})


@app.route("/api/dataset/load-file", methods=["POST"])
def load_dataset_file():
    """Load a dataset from an uploaded pickle file."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    def load_task():
        try:
            update_progress("loading", "Loading dataset from file...", 0, 0)
            # Save to temp file
            temp_path = DATA_DIR / "temp_upload.pkl"
            DATA_DIR.mkdir(exist_ok=True)
            file.save(temp_path)

            clear_dataset()
            load_dataset_from_pickle(temp_path)

            # Clean up
            temp_path.unlink()
            update_progress("idle", f"Loaded {len(clips)} clips from file")
        except Exception as e:
            update_progress("idle", "", 0, 0, str(e))

    thread = threading.Thread(target=load_task, daemon=True)
    thread.start()

    return jsonify({"ok": True, "message": "Loading started"})


@app.route("/api/dataset/load-folder", methods=["POST"])
def load_dataset_folder():
    """Generate dataset from a folder of media files."""
    data = request.get_json(force=True)
    if data is None:
        return jsonify({"error": "Invalid request body"}), 400

    folder_path = data.get("path")
    media_type = data.get(
        "media_type", "sounds"
    )  # Default to sounds for backward compatibility

    if not folder_path:
        return jsonify({"error": "No folder path provided"}), 400

    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        return jsonify({"error": "Invalid folder path"}), 400

    def load_task():
        try:
            clear_dataset()
            load_dataset_from_folder(folder, media_type=media_type)
        except Exception as e:
            update_progress("idle", "", 0, 0, str(e))

    thread = threading.Thread(target=load_task, daemon=True)
    thread.start()

    return jsonify({"ok": True, "message": "Loading started"})


@app.route("/api/dataset/export")
def export_dataset():
    """Export the current dataset to a pickle file."""
    if not clips:
        return jsonify({"error": "No dataset loaded"}), 400

    try:
        dataset_bytes = export_dataset_to_file()
        return send_file(
            io.BytesIO(dataset_bytes),
            mimetype="application/octet-stream",
            download_name="vectorytones_dataset.pkl",
            as_attachment=True,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/dataset/clear", methods=["POST"])
def clear_dataset_route():
    """Clear the current dataset."""
    clear_dataset()
    return jsonify({"ok": True})


if __name__ == "__main__":
    # Initialize before server starts to avoid lazy-loading crashes
    initialize_models()
    # Disable reloader and debug mode to save memory (prevents OOM in Codespaces)
    app.run(debug=False, use_reloader=False, port=5000)
