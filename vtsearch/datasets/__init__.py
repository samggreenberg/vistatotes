"""Dataset configuration and management."""

from vtsearch.datasets.config import DEMO_DATASETS
from vtsearch.datasets.downloader import (
    download_20newsgroups,
    download_cifar10,
    download_esc50,
    download_file_with_progress,
    download_ucf101_subset,
)
from vtsearch.datasets.importers import get_importer, list_importers
from vtsearch.datasets.importers.base import DatasetImporter, ImporterField
from vtsearch.datasets.loader import (
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
from vtsearch.datasets.split import split_dataset

__all__ = [
    "DEMO_DATASETS",
    # Importer registry
    "DatasetImporter",
    "ImporterField",
    "get_importer",
    "list_importers",
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
    # Split utilities
    "split_dataset",
]
