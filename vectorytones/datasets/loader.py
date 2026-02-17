"""Dataset loading and management utilities."""

import csv
import hashlib
import io
import pickle
from pathlib import Path
from typing import Optional

import cv2
import librosa
import numpy as np
import torch
from PIL import Image

from config import (
    CLIPS_PER_CATEGORY,
    CLIPS_PER_VIDEO_CATEGORY,
    EMBEDDINGS_DIR,
    IMAGES_PER_CIFAR10_CATEGORY,
    SAMPLE_RATE,
)
from vectorytones.datasets import DEMO_DATASETS
from vectorytones.datasets.downloader import (
    download_20newsgroups,
    download_cifar10,
    download_esc50,
    download_ucf101_subset,
)
from vectorytones.models import (
    embed_audio_file,
    embed_image_file,
    embed_paragraph_file,
    embed_video_file,
    get_clip_model,
    get_e5_model,
)
from vectorytones.utils import update_progress


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


def load_dataset_from_folder(folder_path: Path, media_type: str, clips: dict):
    """Generate dataset from a folder of media files.

    Args:
        folder_path: Path to folder containing media files
        media_type: One of "sounds", "videos", "images", "paragraphs"
        clips: Dictionary to populate with loaded clips
    """
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


def load_dataset_from_pickle(file_path: Path, clips: dict):
    """Load a dataset from a pickle file.

    Args:
        file_path: Path to the pickle file
        clips: Dictionary to populate with loaded clips
    """
    with open(file_path, "rb") as f:
        data = pickle.load(f)

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


def load_demo_dataset(dataset_name: str, clips: dict, e5_model):
    """Load a demo dataset, downloading and embedding if necessary.

    Args:
        dataset_name: Name of the demo dataset to load
        clips: Dictionary to populate with loaded clips
        e5_model: E5 model instance for paragraph embeddings
    """
    if dataset_name not in DEMO_DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    dataset_info = DEMO_DATASETS[dataset_name]
    media_type = dataset_info.get("media_type", "audio")

    # Check if already embedded
    pkl_file = EMBEDDINGS_DIR / f"{dataset_name}.pkl"
    if pkl_file.exists():
        update_progress("loading", f"Loading {dataset_name} dataset...", 0, 0)
        load_dataset_from_pickle(pkl_file, clips)

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


def export_dataset_to_file(clips: dict) -> bytes:
    """Export the current dataset to a pickle file.

    Args:
        clips: Dictionary of clips to export

    Returns:
        Pickle file contents as bytes
    """
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
