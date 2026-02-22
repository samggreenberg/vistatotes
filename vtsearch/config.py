"""Configuration and constants for VTSearch."""

from pathlib import Path

# Audio settings
SAMPLE_RATE = 48000
NUM_CLIPS = 20

# Dataset paths
DATA_DIR = Path("data")
AUDIO_DIR = DATA_DIR / "audio"
VIDEO_DIR = DATA_DIR / "video"
IMAGE_DIR = DATA_DIR / "images"
PARAGRAPH_DIR = DATA_DIR / "paragraphs"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
MODELS_CACHE_DIR = DATA_DIR / "models"

# Dataset URLs
ESC50_URL = "https://github.com/karolpiczak/ESC-50/archive/master.zip"
SAMPLE_VIDEOS_URL = "https://github.com/sample-datasets/video-clips/archive/refs/heads/main.zip"
CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CALTECH101_URL = "https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip"

# Dataset size estimates
ESC50_DOWNLOAD_SIZE_MB = 600
SAMPLE_VIDEOS_DOWNLOAD_SIZE_MB = 150
CIFAR10_DOWNLOAD_SIZE_MB = 170
CALTECH101_DOWNLOAD_SIZE_MB = 131
CLIPS_PER_CATEGORY = 40
CLIPS_PER_VIDEO_CATEGORY = 10
IMAGES_PER_CIFAR10_CATEGORY = 100
IMAGES_PER_CALTECH101_CATEGORY = 80
TEXTS_PER_CATEGORY = 50

# Training
TRAIN_EPOCHS = 200

# Model IDs
CLAP_MODEL_ID = "laion/clap-htsat-unfused"
XCLIP_MODEL_ID = "microsoft/xclip-base-patch32"
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
E5_MODEL_ID = "intfloat/e5-base-v2"
