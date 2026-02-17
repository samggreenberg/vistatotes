"""Dataset downloading utilities."""

import zipfile
from pathlib import Path

import requests

from config import (
    CIFAR10_DOWNLOAD_SIZE_MB,
    CIFAR10_URL,
    DATA_DIR,
    ESC50_DOWNLOAD_SIZE_MB,
    ESC50_URL,
    VIDEO_DIR,
)
from vectorytones.utils import update_progress


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


def download_20newsgroups(categories: list[str]) -> tuple:
    """Download and prepare 20 Newsgroups dataset.

    Returns:
        tuple: (texts, labels, category_names) where texts is list of strings,
               labels is list of category indices, and category_names is list of category names
    """
    from sklearn.datasets import fetch_20newsgroups

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
