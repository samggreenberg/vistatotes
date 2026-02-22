"""Dataset downloading utilities.

All public functions accept an optional ``on_progress`` callback with the
signature ``(status: str, message: str, current: int, total: int) -> None``.
When omitted the functions fall back to the application-wide
:func:`~vtsearch.utils.update_progress` reporter; pass an explicit callback
to use these functions outside the Flask app (scripts, notebooks, tests).
"""

import zipfile
from pathlib import Path
from typing import Callable, Optional

import requests

from vtsearch.config import (
    CALTECH101_DOWNLOAD_SIZE_MB,
    CALTECH101_URL,
    CIFAR10_DOWNLOAD_SIZE_MB,
    CIFAR10_URL,
    DATA_DIR,
    ESC50_DOWNLOAD_SIZE_MB,
    ESC50_URL,
    IMAGE_DIR,
    VIDEO_DIR,
)

ProgressCallback = Callable[[str, str, int, int], None]


def _default_progress() -> ProgressCallback:
    """Lazily resolve the application-wide progress callback."""
    from vtsearch.utils import update_progress

    return update_progress


def download_file_with_progress(
    url: str,
    dest_path: Path,
    expected_size: int = 0,
    on_progress: Optional[ProgressCallback] = None,
) -> None:
    """Download a file from a URL to a local path, reporting byte-level progress.

    Streams the HTTP response in 8 KB chunks and calls *on_progress*
    after each chunk so that a polling client can track download progress.

    Args:
        url: The HTTP/HTTPS URL to download from.
        dest_path: Local filesystem path where the downloaded file will be written.
        expected_size: Expected file size in bytes, used as a fallback when the
            server does not supply a ``Content-Length`` header. Pass 0 (default)
            if the size is unknown.
        on_progress: Optional progress callback. Falls back to the
            application-wide ``update_progress`` when ``None``.

    Raises:
        requests.HTTPError: If the server returns a non-2xx status code.
    """
    if on_progress is None:
        on_progress = _default_progress()

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
            on_progress("downloading", "Downloading ESC-50...", downloaded, total_size)


def download_esc50(on_progress: Optional[ProgressCallback] = None) -> Path:
    """Download and extract the ESC-50 environmental sounds dataset.

    Downloads ``esc50.zip`` from the configured ``ESC50_URL`` into ``DATA_DIR``
    if it is not already present, then extracts it and deletes the zip to
    reclaim disk space. Both steps report progress via *on_progress*.

    Args:
        on_progress: Optional progress callback. Falls back to the
            application-wide ``update_progress`` when ``None``.

    Returns:
        Path to the ``audio/`` subdirectory inside the extracted ``ESC-50-master``
        directory (e.g. ``data/ESC-50-master/audio``).
    """
    if on_progress is None:
        on_progress = _default_progress()

    zip_path = DATA_DIR / "esc50.zip"
    extract_dir = DATA_DIR / "ESC-50-master"
    DATA_DIR.mkdir(exist_ok=True)

    if not extract_dir.exists():
        if not zip_path.exists():
            on_progress("downloading", "Starting download...", 0, 0)
            download_file_with_progress(ESC50_URL, zip_path, ESC50_DOWNLOAD_SIZE_MB * 1024 * 1024, on_progress)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            members = zip_ref.namelist()
            total = len(members)
            for i, member in enumerate(members, 1):
                on_progress(
                    "downloading",
                    f"Extracting {member.split('/')[-1]}...",
                    i,
                    total,
                )
                zip_ref.extract(member, DATA_DIR)

        zip_path.unlink(missing_ok=True)

    return extract_dir / "audio"


def download_cifar10(on_progress: Optional[ProgressCallback] = None) -> Path:
    """Download and extract the CIFAR-10 image classification dataset.

    Downloads ``cifar-10-python.tar.gz`` from the configured ``CIFAR10_URL``
    into ``DATA_DIR`` if it is not already present, then extracts it and deletes
    the archive to reclaim disk space. Both steps report progress via
    *on_progress*.

    Args:
        on_progress: Optional progress callback. Falls back to the
            application-wide ``update_progress`` when ``None``.

    Returns:
        Path to the ``cifar-10-batches-py/`` directory containing the raw pickle
        batch files (e.g. ``data/cifar-10-batches-py``).
    """
    if on_progress is None:
        on_progress = _default_progress()

    import tarfile

    tar_path = DATA_DIR / "cifar-10-python.tar.gz"
    extract_dir = DATA_DIR / "cifar-10-batches-py"
    DATA_DIR.mkdir(exist_ok=True)

    if not extract_dir.exists():
        if not tar_path.exists():
            on_progress("downloading", "Starting CIFAR-10 download...", 0, 0)
            download_file_with_progress(CIFAR10_URL, tar_path, CIFAR10_DOWNLOAD_SIZE_MB * 1024 * 1024, on_progress)

        on_progress("downloading", "Extracting CIFAR-10...", 0, 0)
        with tarfile.open(tar_path, "r:gz") as tar_ref:
            tar_ref.extractall(DATA_DIR)

        tar_path.unlink(missing_ok=True)

    return extract_dir


def download_caltech101(on_progress: Optional[ProgressCallback] = None) -> Path:
    """Download and extract the Caltech-101 image classification dataset.

    Downloads ``caltech-101.zip`` from the configured ``CALTECH101_URL``
    into ``DATA_DIR`` if it is not already present, then extracts it and
    deletes the archive to reclaim disk space.

    Args:
        on_progress: Optional progress callback. Falls back to the
            application-wide ``update_progress`` when ``None``.

    Returns:
        Path to the ``101_ObjectCategories/`` directory containing category
        subfolders of JPEG images (e.g.
        ``data/caltech-101/101_ObjectCategories``).
    """
    if on_progress is None:
        on_progress = _default_progress()

    zip_path = DATA_DIR / "caltech-101.zip"
    extract_dir = DATA_DIR / "caltech-101"
    DATA_DIR.mkdir(exist_ok=True)
    IMAGE_DIR.mkdir(exist_ok=True, parents=True)

    categories_dir = extract_dir / "101_ObjectCategories"
    if not categories_dir.exists():
        if not zip_path.exists():
            on_progress("downloading", "Starting Caltech-101 download...", 0, 0)
            download_file_with_progress(
                CALTECH101_URL, zip_path, CALTECH101_DOWNLOAD_SIZE_MB * 1024 * 1024, on_progress
            )

        on_progress("downloading", "Extracting Caltech-101...", 0, 0)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            members = zip_ref.namelist()
            total = len(members)
            for i, member in enumerate(members, 1):
                if i % 100 == 0 or i == total:
                    on_progress(
                        "downloading",
                        f"Extracting Caltech-101 ({i}/{total})...",
                        i,
                        total,
                    )
                zip_ref.extract(member, extract_dir)

        zip_path.unlink(missing_ok=True)

    return categories_dir


def download_ucf101_subset() -> Path:
    """Return the path to the UCF-101 video dataset, raising if it is not present.

    UCF-101 is distributed as a ~6.5 GB RAR file and cannot be downloaded
    automatically. This function checks whether the dataset has already been
    manually placed in the expected directory and returns its path, or raises a
    descriptive error with setup instructions.

    Returns:
        Path to the ``ucf101/`` directory inside ``VIDEO_DIR`` (e.g.
        ``data/video/ucf101``), guaranteed to contain at least one ``*.avi`` file
        in a subdirectory.

    Raises:
        ValueError: If the UCF-101 directory does not exist or contains no
            ``*.avi`` files. The error message includes manual download
            instructions.
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


def download_20newsgroups(
    categories: list[str],
    on_progress: Optional[ProgressCallback] = None,
) -> tuple[list[str], list[int], list[str]]:
    """Download and prepare a subset of the 20 Newsgroups text dataset.

    Uses scikit-learn's :func:`sklearn.datasets.fetch_20newsgroups` (which
    handles caching automatically) to fetch training articles for the requested
    category names. Category names are mapped from simplified labels (e.g.
    ``"science"``) to the full newsgroup names (e.g. ``"sci.space"``) before
    downloading, then mapped back for the returned ``target_names``.

    Args:
        categories: List of simplified category names to include. Recognised
            values and their newsgroup mappings are:

            - ``"world"``    → ``"talk.politics.misc"``
            - ``"sports"``   → ``"rec.sport.baseball"``
            - ``"business"`` → ``"misc.forsale"``
            - ``"science"``  → ``"sci.space"``

            Any category not in the mapping is passed through unchanged as the
            full newsgroup name.
        on_progress: Optional progress callback. Falls back to the
            application-wide ``update_progress`` when ``None``.

    Returns:
        A 3-tuple ``(texts, labels, category_names)`` where:

        - ``texts`` is a list of article strings (headers, footers, and quoted
          text removed).
        - ``labels`` is a list of integer category indices, aligned with
          ``texts``, referencing ``category_names``.
        - ``category_names`` is a list of simplified category name strings,
          ordered to correspond with label index values.
    """
    if on_progress is None:
        on_progress = _default_progress()

    from sklearn.datasets import fetch_20newsgroups

    on_progress("downloading", "Downloading 20 Newsgroups dataset...", 0, 0)

    # Map our category names to 20 newsgroups categories.
    # Covers all 20 newsgroups under shorter, friendlier aliases.
    category_mapping = {
        "world": "talk.politics.misc",
        "sports": "rec.sport.baseball",
        "business": "misc.forsale",
        "science": "sci.space",
        "technology": "comp.graphics",
        "medicine": "sci.med",
        "cars": "rec.autos",
        "hockey": "rec.sport.hockey",
        "electronics": "sci.electronics",
        "crypto": "sci.crypt",
        "religion": "soc.religion.christian",
        "guns": "talk.politics.guns",
        "atheism": "alt.atheism",
        "mac": "comp.sys.mac.hardware",
        "pc_hardware": "comp.sys.ibm.pc.hardware",
        "windows": "comp.os.ms-windows.misc",
        "x_windows": "comp.windows.x",
        "motorcycles": "rec.motorcycles",
        "mideast": "talk.politics.mideast",
        "religion_misc": "talk.religion.misc",
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
        list(category_mapping.keys())[list(category_mapping.values()).index(newsgroups.target_names[i])]
        if newsgroups.target_names[i] in category_mapping.values()
        else newsgroups.target_names[i]
        for i in range(len(newsgroups.target_names))
    ]

    return texts, labels, target_names
