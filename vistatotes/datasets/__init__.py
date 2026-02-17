"""Dataset configuration and management."""

from vistatotes.datasets.config import DEMO_DATASETS
from vistatotes.datasets.downloader import (
    download_20newsgroups,
    download_cifar10,
    download_esc50,
    download_file_with_progress,
    download_ucf101_subset,
)
from vistatotes.datasets.loader import (
    export_dataset_to_file,
    load_cifar10_batch,
    load_dataset_from_folder,
    load_dataset_from_pickle,
    load_demo_dataset,
    load_esc50_metadata,
    load_image_metadata_from_folders,
    load_paragraph_metadata_from_folders,
    load_video_metadata_from_folders,
)

__all__ = [
    "DEMO_DATASETS",
    # Downloader functions
    "download_file_with_progress",
    "download_esc50",
    "download_cifar10",
    "download_ucf101_subset",
    "download_20newsgroups",
    # Loader functions
    "load_esc50_metadata",
    "load_cifar10_batch",
    "load_video_metadata_from_folders",
    "load_image_metadata_from_folders",
    "load_paragraph_metadata_from_folders",
    "load_dataset_from_folder",
    "load_dataset_from_pickle",
    "load_demo_dataset",
    "export_dataset_to_file",
]
