"""Dataset loading and management utilities.

All public functions that perform I/O accept an optional ``on_progress``
callback with the signature
``(status: str, message: str, current: int, total: int) -> None``.
When omitted the functions fall back to the application-wide
:func:`~vtsearch.utils.update_progress` reporter; pass an explicit callback
to use these functions outside the Flask app.
"""

import csv
import hashlib
import io
import pickle
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from PIL import Image

from config import EMBEDDINGS_DIR, IMAGES_PER_CALTECH101_CATEGORY, IMAGES_PER_CIFAR10_CATEGORY, TEXTS_PER_CATEGORY
from vtsearch.datasets.config import DEMO_DATASETS
from vtsearch.datasets.downloader import (
    download_20newsgroups,
    download_caltech101,
    download_cifar10,
    download_esc50,
    download_ucf101_subset,
)

ProgressCallback = Callable[[str, str, int, int], None]


def _default_progress() -> ProgressCallback:
    """Lazily resolve the application-wide progress callback."""
    from vtsearch.utils import update_progress

    return update_progress


def load_esc50_metadata(esc50_dir: Path) -> dict[str, dict[str, Any]]:
    """Load clip metadata from the ESC-50 ``esc50.csv`` metadata file.

    Reads ``<esc50_dir>/meta/esc50.csv`` and builds a mapping from audio
    filename to its associated metadata fields.

    Args:
        esc50_dir: Path to the root ESC-50 dataset directory (the directory that
            contains the ``meta/`` and ``audio/`` subdirectories).

    Returns:
        A dict mapping audio filename (e.g. ``"1-100032-A-0.wav"``) to a dict
        with the keys:

        - ``"category"`` (``str``): Human-readable sound category label.
        - ``"esc10"`` (``bool``): Whether the clip belongs to the ESC-10 subset.
        - ``"target"`` (``int``): Integer class index.
        - ``"fold"`` (``int``): Cross-validation fold number (1–5).
    """
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


def load_cifar10_batch(file_path: Path) -> tuple[np.ndarray, list[int], list[str]]:
    """Load a CIFAR-10 pickle batch file and return images, labels, and label names.

    Args:
        file_path: Path to a CIFAR-10 batch file (e.g. ``data_batch_1``) in the
            unpickled binary format used by the original dataset.

    Returns:
        A 3-tuple ``(images, labels, label_names)`` where:

        - ``images`` is a ``numpy.ndarray`` of shape ``(N, 32, 32, 3)`` with
          ``uint8`` pixel values in RGB order.
        - ``labels`` is a list of integer class indices (one per image), each in
          the range ``[0, 9]``.
        - ``label_names`` is a fixed list of 10 human-readable class name strings
          (e.g. ``"airplane"``, ``"automobile"``, …, ``"truck"``), ordered so that
          ``label_names[i]`` corresponds to label value ``i``.
    """
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


def load_video_metadata_from_folders(video_dir: Path, categories: list[str]) -> dict[str, dict[str, Any]]:
    """Scan category subdirectories and collect video file metadata.

    Iterates over immediate subdirectories of ``video_dir``, keeping only those
    whose name appears in ``categories``, and collects paths for all video files
    with common extensions (``mp4``, ``avi``, ``mov``, ``webm``, ``mkv``).

    Args:
        video_dir: Root directory whose immediate subdirectories represent
            category folders.
        categories: List of category folder names to include. Subdirectories
            not in this list are skipped.

    Returns:
        A dict mapping video filename (basename only) to a dict with the keys:

        - ``"category"`` (``str``): Name of the category folder.
        - ``"path"`` (``Path``): Full path to the video file.
    """
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


def load_image_metadata_from_folders(image_dir: Path, categories: list[str]) -> dict[str, dict[str, Any]]:
    """Scan category subdirectories and collect image file metadata.

    Iterates over immediate subdirectories of ``image_dir``, keeping only those
    whose name appears in ``categories``, and collects paths for all image files
    with common extensions (``png``, ``jpg``, ``jpeg``, ``gif``, ``bmp``, ``webp``).

    Args:
        image_dir: Root directory whose immediate subdirectories represent
            category folders.
        categories: List of category folder names to include. Subdirectories
            not in this list are skipped.

    Returns:
        A dict mapping image filename (basename only) to a dict with the keys:

        - ``"category"`` (``str``): Name of the category folder.
        - ``"path"`` (``Path``): Full path to the image file.
    """
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


def load_paragraph_metadata_from_folders(text_dir: Path, categories: list[str]) -> dict[str, dict[str, Any]]:
    """Scan category subdirectories and collect text file metadata.

    Iterates over immediate subdirectories of ``text_dir``, keeping only those
    whose name appears in ``categories``, and collects paths for all plain-text
    files with extensions ``txt`` or ``md``.

    Args:
        text_dir: Root directory whose immediate subdirectories represent
            category folders.
        categories: List of category folder names to include. Subdirectories
            not in this list are skipped.

    Returns:
        A dict mapping text filename (basename only) to a dict with the keys:

        - ``"category"`` (``str``): Name of the category folder.
        - ``"path"`` (``Path``): Full path to the text file.
    """
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


def load_dataset_from_folder(
    folder_path: Path,
    media_type: str,
    clips: dict[int, dict[str, Any]],
    content_vectors: dict[str, Any] | None = None,
    on_progress: Optional[ProgressCallback] = None,
    origin: dict[str, Any] | None = None,
) -> None:
    """Generate a dataset in-place from a flat folder of media files.

    Scans ``folder_path`` for all files matching the extensions for ``media_type``,
    embeds each file using the appropriate model, and populates ``clips`` with
    the resulting clip dicts. Progress is reported via :func:`update_progress`.

    Files whose basename appears in ``content_vectors`` will use the supplied
    embedding instead of running the embedding model.  This allows importers
    that already provide content vectors to avoid redundant computation.

    The ``clips`` dict is cleared before loading begins.

    ``media_type`` is looked up in the media type registry by
    :attr:`~vtsearch.media.base.MediaType.folder_import_name` (e.g.
    ``"sounds"``, ``"videos"``, ``"images"``, ``"paragraphs"``).  Adding a
    new media type to the registry automatically makes it available here
    without any changes to this function.

    Args:
        folder_path: Path to a flat directory containing media files.
        media_type: Folder-import alias for the media type (e.g. ``"sounds"``).
        clips: Dict to populate in-place. Existing entries are removed before
            loading. Keys are sequential integer clip IDs starting from 1.
        content_vectors: Optional mapping of filename (basename) to a
            pre-computed embedding ``numpy.ndarray``.  When a file's name is
            found in this dict the supplied vector is used directly and the
            embedding model is not invoked for that file.
        origin: Optional serialised
            :class:`~vtsearch.datasets.origin.Origin` dict to attach to each
            clip (as ``clip["origin"]``).  When ``None`` no origin is set
            and the caller is expected to set it afterwards.

    Raises:
        ValueError: If ``media_type`` is not recognised, or if no matching
            files are found in ``folder_path``.
    """
    from vtsearch.media import get_by_folder_name

    if on_progress is None:
        on_progress = _default_progress()

    on_progress("embedding", "Scanning media files...", 0, 0)

    try:
        mt = get_by_folder_name(media_type)
    except KeyError:
        raise ValueError(f"Invalid media type: {media_type}")

    # Find all files of the specified media type
    media_files = []
    for ext in mt.file_extensions:
        media_files.extend(folder_path.glob(ext))

    if not media_files:
        raise ValueError(f"No {media_type} files found in folder")

    clips.clear()
    clip_id = 1
    total_files = len(media_files)

    for i, file_path in enumerate(media_files):
        on_progress(
            "embedding",
            f"Embedding {media_type} {file_path.name}...",
            i + 1,
            total_files,
        )

        if content_vectors and file_path.name in content_vectors:
            embedding = content_vectors[file_path.name]
        else:
            embedding = mt.embed_media(file_path)
            if embedding is None:
                continue

        with open(file_path, "rb") as f:
            file_bytes = f.read()

        # Build the base clip dict
        clip_data: dict[str, Any] = {
            "id": clip_id,
            "type": mt.type_id,
            "file_size": len(file_bytes),
            "md5": hashlib.md5(file_bytes).hexdigest(),
            "embedding": embedding,
            "filename": file_path.name,
            "category": "custom",
            "origin": origin,
            "origin_name": file_path.name,
            # Null-out optional media fields so clips from different types
            # stored in the same dict have consistent keys.
            "clip_bytes": None,
            "clip_string": None,
            "duration": 0,
        }

        # Merge in media-specific fields from the media type
        clip_data.update(mt.load_clip_data(file_path))

        clips[clip_id] = clip_data
        clip_id += 1

    on_progress("idle", f"Loaded {len(clips)} {media_type} clips from folder")


def load_dataset_from_pickle(
    file_path: Path,
    clips: dict[int, dict[str, Any]],
) -> dict[str, Any] | None:
    """Load a dataset from a pickle file into the clips dict in-place.

    Supports two pickle formats:

    - **New format**: A dict with a ``"clips"`` key mapping to clip data dicts.
      May also include ``"audio_dir"``, ``"video_dir"``, ``"image_dir"``, or
      ``"text_dir"`` keys pointing to directories containing the raw media files
      when the bytes are not stored inline.
    - **Old format**: A plain dict mapping clip ID to clip data dict (no wrapping
      ``"clips"`` key).

    If media bytes are not stored inline in the pickle, the function attempts to
    load them from the companion directory entry in the pickle. Clips for which
    no media bytes can be resolved are silently skipped (a warning is printed to
    stdout after loading).

    The ``clips`` dict is cleared before loading begins.

    Args:
        file_path: Path to a ``.pkl`` file previously created by
            :func:`export_dataset_to_file` or :func:`load_demo_dataset`.
        clips: Dict to populate in-place. Existing entries are removed before
            loading. Keys are clip IDs (int); values are clip data dicts.

    Returns:
        ``None``.  (Formerly returned ``creation_info``; that field has been
        removed.)
    """
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    clips.clear()

    # Handle both old format (just clips dict) and new format (with metadata)
    if isinstance(data, dict) and "clips" in data:
        clips_data = data["clips"]
        # Old pickles may contain creation_info; extract a fallback origin
        # for clips that predate per-element origin tracking.
        creation_info = data.get("creation_info")
    else:
        clips_data = data
        creation_info = None

    fallback_origin = None
    if creation_info:
        fallback_origin = {
            "importer": creation_info.get("importer", "unknown"),
            "params": creation_info.get("field_values", {}),
        }

    # Convert to the app's clip format
    missing_media = 0
    for clip_id, clip_info in clips_data.items():
        # Determine media type
        media_type = clip_info.get("type", "audio")

        # Load the actual media content.
        # Support both new key names (clip_bytes/clip_string) and legacy
        # per-media-type key names (wav_bytes/video_bytes/image_bytes/
        # text_content) for backward compatibility with old pickles.
        clip_bytes = None
        clip_string = None
        media_bytes = None

        if media_type == "audio":
            clip_bytes = clip_info.get("clip_bytes") or clip_info.get("wav_bytes")
            if clip_bytes is not None:
                media_bytes = clip_bytes
            elif "filename" in clip_info and "audio_dir" in data:
                audio_path = Path(data["audio_dir"]) / clip_info["filename"]
                if audio_path.exists():
                    with open(audio_path, "rb") as f:
                        clip_bytes = f.read()
                        media_bytes = clip_bytes
                else:
                    missing_media += 1

        elif media_type == "video":
            clip_bytes = clip_info.get("clip_bytes") or clip_info.get("video_bytes")
            if clip_bytes is not None:
                media_bytes = clip_bytes
            elif "filename" in clip_info and "video_dir" in data:
                video_path = Path(data["video_dir"]) / clip_info["filename"]
                if video_path.exists():
                    with open(video_path, "rb") as f:
                        clip_bytes = f.read()
                        media_bytes = clip_bytes
                else:
                    missing_media += 1

        elif media_type == "image":
            clip_bytes = clip_info.get("clip_bytes") or clip_info.get("image_bytes")
            if clip_bytes is not None:
                media_bytes = clip_bytes
            elif "filename" in clip_info and "image_dir" in data:
                image_path = Path(data["image_dir"]) / clip_info["filename"]
                if image_path.exists():
                    with open(image_path, "rb") as f:
                        clip_bytes = f.read()
                        media_bytes = clip_bytes
                else:
                    missing_media += 1

        elif media_type == "paragraph":
            clip_string = clip_info.get("clip_string") or clip_info.get("text_content")
            if clip_string is not None:
                media_bytes = clip_string.encode("utf-8")  # For MD5 hash
            elif "filename" in clip_info and "text_dir" in data:
                text_path = Path(data["text_dir"]) / clip_info["filename"]
                if text_path.exists():
                    with open(text_path, "r", encoding="utf-8") as f:
                        clip_string = f.read()
                        media_bytes = clip_string.encode("utf-8")
                else:
                    missing_media += 1

        if media_bytes:
            fname = clip_info.get("filename", f"clip_{clip_id}.{media_type}")
            clip_data = {
                "id": clip_id,
                "type": media_type,
                "duration": clip_info.get("duration", 0),
                "file_size": clip_info.get("file_size", len(media_bytes)),
                "md5": hashlib.md5(media_bytes).hexdigest(),
                "embedding": np.array(clip_info["embedding"]),
                "clip_bytes": clip_bytes,
                "clip_string": clip_string,
                "filename": fname,
                "category": clip_info.get("category", "unknown"),
                "origin": clip_info.get("origin", fallback_origin),
                "origin_name": clip_info.get("origin_name", fname),
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
        print(f"WARNING: {missing_media} media files missing from {file_path}", flush=True)

    return None


def embed_image_file_from_pil(image: Image.Image) -> Optional[np.ndarray]:
    """Generate a CLIP embedding vector for a PIL Image object.

    A convenience wrapper for cases where the image is already in memory
    (e.g. reconstructed from a NumPy array during CIFAR-10 loading).

    Delegates to :meth:`~vtsearch.media.image.media_type.ImageMediaType.embed_pil_image`.

    Args:
        image: A PIL Image in any mode.

    Returns:
        A 1-D ``numpy.ndarray`` of shape ``(embedding_dim,)``, or ``None`` if
        the CLIP model is not loaded or an exception occurs.
    """
    from vtsearch.media import get as media_get

    return media_get("image").embed_pil_image(image)


def load_demo_dataset(
    dataset_name: str,
    clips: dict[int, dict[str, Any]],
    e5_model: Any = None,
    on_progress: Optional[ProgressCallback] = None,
) -> None:
    """Load a named demo dataset into the clips dict, downloading and embedding as needed.

    Checks for a cached ``.pkl`` file in ``EMBEDDINGS_DIR``; if found, loads
    from that file. If the cache is missing or the media bytes it references can
    no longer be found on disk, the raw data is re-downloaded and re-embedded.

    Supported datasets and their sources:

    - Audio datasets (ESC-50 subsets): downloaded from ``ESC50_URL``, embedded
      with CLAP.
    - Image datasets (CIFAR-10 subsets): downloaded from ``CIFAR10_URL``,
      embedded with CLIP.
    - Paragraph datasets (20 Newsgroups subsets): downloaded via
      ``sklearn.datasets.fetch_20newsgroups``, embedded with E5-base-v2.
    - Video datasets (UCF-101): must be manually placed at
      ``VIDEO_DIR/ucf101/``; embedded with X-CLIP.

    Progress throughout the operation is reported via :func:`update_progress`.

    Args:
        dataset_name: Key into ``DEMO_DATASETS`` identifying which demo dataset
            to load.  Raises ``ValueError`` if the key is not found.
        clips: Dict to populate in-place. Existing entries are removed before
            loading. Keys are integer clip IDs; values are clip data dicts.
        e5_model: Deprecated — kept for backward compatibility but no longer
            used.  The text embedding model is obtained from the media type
            registry.

    Raises:
        ValueError: If ``dataset_name`` is not in ``DEMO_DATASETS``, or if the
            UCF-101 dataset is requested but not yet downloaded.
    """
    if on_progress is None:
        on_progress = _default_progress()

    if dataset_name not in DEMO_DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    dataset_info = DEMO_DATASETS[dataset_name]
    media_type = dataset_info.get("media_type", "audio")
    demo_origin: dict[str, Any] = {"importer": "demo", "params": {"name": dataset_name}}

    # Check if already embedded
    pkl_file = EMBEDDINGS_DIR / f"{dataset_name}.pkl"
    if pkl_file.exists():
        on_progress("loading", f"Loading {dataset_name} dataset...", 0, 0)
        load_dataset_from_pickle(pkl_file, clips)

        # Check if any clips were actually loaded
        if len(clips) == 0:
            # Pickle file exists but media files are missing, delete and re-embed
            on_progress("loading", f"Media files missing, re-embedding {dataset_name}...", 0, 0)
            pkl_file.unlink()
        else:
            on_progress("idle", f"Loaded {dataset_name} dataset")
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
            on_progress("embedding", f"Starting embedding for {total} images...", 0, total)

            for i, (image_array, category) in enumerate(zip(selected_images, selected_labels)):
                on_progress(
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

                # Get embedding via registry
                embedding = embed_image_file_from_pil(img)
                if embedding is None:
                    continue

                fname = f"{category}_{clip_id}.png"
                clips[clip_id] = {
                    "id": clip_id,
                    "type": "image",
                    "duration": 0,  # Images don't have duration
                    "file_size": len(image_bytes),
                    "md5": hashlib.md5(image_bytes).hexdigest(),
                    "embedding": embedding,
                    "clip_bytes": image_bytes,
                    "clip_string": None,
                    "filename": fname,
                    "category": category,
                    "width": img.width,
                    "height": img.height,
                    "origin": demo_origin,
                    "origin_name": fname,
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
                            }
                            for cid, clip in clips.items()
                        },
                    },
                    f,
                )

            on_progress("idle", f"Loaded {dataset_name} dataset")
            return

        elif image_source == "caltech101":
            # Download Caltech-101 if needed
            caltech_dir = download_caltech101()

            # Scan category folders for images
            metadata = load_image_metadata_from_folders(caltech_dir, dataset_info["categories"])

            # Limit per category
            cat_counts: dict[str, int] = {}
            selected: list[tuple[Path, str]] = []
            for fname, meta in sorted(metadata.items()):
                cat = meta["category"]
                cat_counts.setdefault(cat, 0)
                if cat_counts[cat] < IMAGES_PER_CALTECH101_CATEGORY:
                    selected.append((meta["path"], cat))
                    cat_counts[cat] += 1

            from vtsearch.media import get as media_get

            image_mt = media_get("image")

            clips.clear()
            clip_id = 1
            total = len(selected)
            on_progress("embedding", f"Starting embedding for {total} images...", 0, total)

            for i, (img_path, category) in enumerate(selected):
                on_progress(
                    "embedding",
                    f"Embedding {category}: {img_path.name} ({i + 1}/{total})",
                    i + 1,
                    total,
                )

                embedding = image_mt.embed_media(img_path)
                if embedding is None:
                    continue

                with open(img_path, "rb") as f:
                    image_bytes = f.read()

                try:
                    img = Image.open(img_path)
                    width, height = img.width, img.height
                except Exception:
                    width, height = None, None

                clips[clip_id] = {
                    "id": clip_id,
                    "type": "image",
                    "duration": 0,
                    "file_size": len(image_bytes),
                    "md5": hashlib.md5(image_bytes).hexdigest(),
                    "embedding": embedding,
                    "clip_bytes": image_bytes,
                    "clip_string": None,
                    "filename": img_path.name,
                    "category": category,
                    "width": width,
                    "height": height,
                    "origin": demo_origin,
                    "origin_name": img_path.name,
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
                            }
                            for cid, clip in clips.items()
                        },
                    },
                    f,
                )

            on_progress("idle", f"Loaded {dataset_name} dataset")
            return

    elif media_type == "paragraph":
        # Handle paragraph datasets — use text media type from registry
        from vtsearch.media import get as media_get

        text_mt = media_get("paragraph")

        paragraph_source = dataset_info.get("source", "ag_news_sample")

        if paragraph_source == "ag_news_sample":
            # Use 20 Newsgroups dataset from scikit-learn
            texts, labels, category_names = download_20newsgroups(dataset_info["categories"])

            # Limit number of texts per category for demo
            max_per_category = TEXTS_PER_CATEGORY
            selected_texts = []
            selected_categories = []

            for cat_name in dataset_info["categories"]:
                if cat_name in category_names:
                    cat_idx = category_names.index(cat_name)
                    cat_texts = [texts[i] for i, lbl in enumerate(labels) if lbl == cat_idx]
                    # Limit to max_per_category
                    for text in cat_texts[:max_per_category]:
                        selected_texts.append(text)
                        selected_categories.append(cat_name)

            # Generate embeddings for paragraphs
            clips.clear()
            clip_id = 1
            total = len(selected_texts)
            on_progress(
                "embedding",
                f"Starting embedding for {total} paragraphs...",
                0,
                total,
            )

            for i, (text_content, category) in enumerate(zip(selected_texts, selected_categories)):
                on_progress(
                    "embedding",
                    f"Embedding {category}: paragraph {i + 1}/{total}",
                    i + 1,
                    total,
                )

                # Truncate very long texts (keep first 1000 chars for demo)
                text_content = text_content[:1000].strip()
                if not text_content:
                    continue

                # Embed via text media type
                try:
                    embedding = text_mt.embed_text_passage(text_content)
                except Exception as e:
                    print(f"Error embedding paragraph: {e}")
                    continue

                if embedding is None:
                    continue

                word_count = len(text_content.split())
                character_count = len(text_content)
                text_bytes = text_content.encode("utf-8")

                fname = f"{category}_{clip_id}.txt"
                clips[clip_id] = {
                    "id": clip_id,
                    "type": "paragraph",
                    "duration": 0,  # Paragraphs don't have duration
                    "file_size": len(text_bytes),
                    "md5": hashlib.md5(text_bytes).hexdigest(),
                    "embedding": embedding,
                    "clip_bytes": None,
                    "clip_string": text_content,
                    "filename": fname,
                    "category": category,
                    "word_count": word_count,
                    "character_count": character_count,
                    "origin": demo_origin,
                    "origin_name": fname,
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
                            }
                            for cid, clip in clips.items()
                        },
                    },
                    f,
                )

            on_progress("idle", f"Loaded {dataset_name} dataset")
            return

    elif media_type == "video":
        # Handle video datasets
        video_source = dataset_info.get("source", "ucf101")

        if video_source == "ucf101":
            from vtsearch.media import get as media_get

            video_mt = media_get("video")

            try:
                video_dir = download_ucf101_subset()
            except ValueError as e:
                # If UCF-101 is not available, provide helpful error message
                on_progress("idle", "")
                raise e

            metadata = load_video_metadata_from_folders(video_dir, dataset_info["categories"])
            video_files = [(meta["path"], meta) for meta in metadata.values()]

            # Generate embeddings for videos
            clips.clear()
            clip_id = 1
            total = len(video_files)
            on_progress("embedding", f"Starting embedding for {total} video files...", 0, total)

            for i, (video_path, meta) in enumerate(video_files):
                on_progress(
                    "embedding",
                    f"Embedding {meta['category']}: {video_path.name} ({i + 1}/{total})",
                    i + 1,
                    total,
                )

                embedding = video_mt.embed_media(video_path)
                if embedding is None:
                    continue

                with open(video_path, "rb") as f:
                    video_bytes = f.read()

                media_fields = video_mt.load_clip_data(video_path)

                clips[clip_id] = {
                    "id": clip_id,
                    "type": "video",
                    "duration": media_fields["duration"],
                    "file_size": len(video_bytes),
                    "md5": hashlib.md5(video_bytes).hexdigest(),
                    "embedding": embedding,
                    "clip_bytes": video_bytes,
                    "filename": video_path.name,
                    "category": meta["category"],
                    "origin": demo_origin,
                    "origin_name": video_path.name,
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
                                if k != "clip_bytes"
                            }
                            for cid, clip in clips.items()
                        },
                        "video_dir": str(video_dir.absolute()),
                    },
                    f,
                )

            on_progress("idle", f"Loaded {dataset_name} dataset")
            return

    # Handle audio datasets (ESC-50 logic)
    from vtsearch.media import get as media_get

    audio_mt = media_get("audio")

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
    on_progress("embedding", f"Starting embedding for {total} audio files...", 0, total)

    for i, (audio_path, meta) in enumerate(audio_files):
        on_progress(
            "embedding",
            f"Embedding {meta['category']}: {audio_path.name} ({i + 1}/{total})",
            i + 1,
            total,
        )

        embedding = audio_mt.embed_media(audio_path)
        if embedding is None:
            continue

        with open(audio_path, "rb") as f:
            wav_bytes = f.read()

        media_fields = audio_mt.load_clip_data(audio_path)

        clips[clip_id] = {
            "id": clip_id,
            "type": "audio",
            "duration": media_fields["duration"],
            "file_size": len(wav_bytes),
            "md5": hashlib.md5(wav_bytes).hexdigest(),
            "embedding": embedding,
            "clip_bytes": wav_bytes,
            "filename": audio_path.name,
            "category": meta["category"],
            "origin": demo_origin,
            "origin_name": audio_path.name,
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
                        if k != "clip_bytes"
                    }
                    for cid, clip in clips.items()
                },
                "audio_dir": str(audio_dir.absolute()),
            },
            f,
        )

    on_progress("idle", f"Loaded {dataset_name} dataset")


def export_dataset_to_file(
    clips: dict[int, dict[str, Any]],
) -> bytes:
    """Serialise the current clip dataset to a pickle-formatted byte string.

    Converts the in-memory ``clips`` dict to a portable format (converting any
    ``numpy.ndarray`` embeddings to plain Python lists) and returns it as bytes
    suitable for writing to a ``.pkl`` file or sending as an HTTP response.

    The resulting bytes can be reloaded with :func:`load_dataset_from_pickle`.

    Args:
        clips: Mapping of clip ID to clip data dict.

    Returns:
        Raw bytes of the pickled dataset dict.
    """
    data: dict[str, Any] = {
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
                "origin": clip.get("origin"),
                "origin_name": clip.get("origin_name", clip.get("filename", "")),
                "clip_bytes": clip.get("clip_bytes"),
                "clip_string": clip.get("clip_string"),
                "word_count": clip.get("word_count"),
                "character_count": clip.get("character_count"),
                "width": clip.get("width"),
                "height": clip.get("height"),
            }
            for cid, clip in clips.items()
        }
    }

    buf = io.BytesIO()
    pickle.dump(data, buf)
    buf.seek(0)
    return buf.getvalue()
