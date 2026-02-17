#!/usr/bin/env python3
"""
Setup script to download audio datasets and generate LAION-CLAP embeddings.
Run this once to prepare demo datasets for the VectoryTones application.
"""

import json
import pickle
import zipfile
from pathlib import Path

import librosa
import requests
import torch
from tqdm import tqdm
from transformers import ClapModel, ClapProcessor

# Configuration
DATA_DIR = Path("data")
AUDIO_DIR = DATA_DIR / "audio"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
ESC50_URL = "https://github.com/karolpiczak/ESC-50/archive/master.zip"
SAMPLE_RATE = 48000  # LAION-CLAP expects 48kHz

# ESC-50 category organization
DATASET_CATEGORIES = {
    "animals": [
        "dog",
        "rooster",
        "pig",
        "cow",
        "frog",
        "cat",
        "hen",
        "insects",
        "sheep",
        "crow",
        "rain",
        "sea_waves",
        "crackling_fire",
        "crickets",
        "chirping_birds",
        "water_drops",
        "wind",
        "pouring_water",
        "toilet_flush",
        "thunderstorm",
    ],
    "natural": [
        "rain",
        "sea_waves",
        "crackling_fire",
        "crickets",
        "chirping_birds",
        "water_drops",
        "wind",
        "pouring_water",
        "thunderstorm",
        "frog",
    ],
    "urban": [
        "clock_alarm",
        "clock_tick",
        "door_wood_knock",
        "mouse_click",
        "keyboard_typing",
        "door_wood_creaks",
        "can_opening",
        "washing_machine",
        "vacuum_cleaner",
        "helicopter",
        "chainsaw",
        "siren",
        "car_horn",
        "engine",
        "train",
        "church_bells",
        "airplane",
        "fireworks",
        "hand_saw",
    ],
    "household": [
        "clock_alarm",
        "clock_tick",
        "door_wood_knock",
        "mouse_click",
        "keyboard_typing",
        "door_wood_creaks",
        "can_opening",
        "washing_machine",
        "vacuum_cleaner",
        "sneezing",
        "coughing",
        "breathing",
        "laughing",
        "brushing_teeth",
        "snoring",
        "drinking_sipping",
        "footsteps",
    ],
}


def download_file(url: str, dest_path: Path) -> None:
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))

    with (
        open(dest_path, "wb") as f,
        tqdm(
            desc=dest_path.name,
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar,
    ):
        for chunk in response.iter_content(chunk_size=8192):
            size = f.write(chunk)
            pbar.update(size)


def download_esc50() -> Path:
    """Download and extract ESC-50 dataset."""
    print("📦 Downloading ESC-50 dataset...")

    zip_path = DATA_DIR / "esc50.zip"
    DATA_DIR.mkdir(exist_ok=True)

    if not zip_path.exists():
        download_file(ESC50_URL, zip_path)
    else:
        print(f"  ✓ Already downloaded: {zip_path}")

    print("📂 Extracting ESC-50...")
    extract_dir = DATA_DIR / "ESC-50-master"
    if not extract_dir.exists():
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(DATA_DIR)
        print(f"  ✓ Extracted to: {extract_dir}")
    else:
        print(f"  ✓ Already extracted: {extract_dir}")

    return extract_dir / "audio"


def load_esc50_metadata(esc50_dir: Path) -> dict:
    """Load ESC-50 metadata CSV."""
    import csv

    meta_file = esc50_dir.parent / "meta" / "esc50.csv"

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


def load_clap_model():
    """Load pretrained LAION-CLAP model."""
    print("🤖 Loading CLAP model (Hugging Face)...")
    model_id = "laion/clap-htsat-unfused"
    model = ClapModel.from_pretrained(model_id)
    processor = ClapProcessor.from_pretrained(model_id, use_fast=False)
    print("  ✓ Model loaded")
    return model, processor


def generate_embeddings(audio_dir: Path, model, processor, metadata: dict) -> dict:
    """Generate CLAP embeddings for all audio files."""
    print("🎵 Generating CLAP embeddings...")

    audio_files = sorted(audio_dir.glob("*.wav"))
    embeddings_data = {}

    for audio_file in tqdm(audio_files, desc="Processing audio"):
        filename = audio_file.name

        # Load audio at 48kHz for CLAP
        audio_data, sr = librosa.load(audio_file, sr=SAMPLE_RATE, mono=True)

        # Get embedding from CLAP
        inputs = processor(
            audio=audio_data,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding="max_length",
            max_length=480000,
            truncation=True,
        )
        with torch.no_grad():
            outputs = model.audio_model(**inputs)
            embedding = model.audio_projection(outputs.pooler_output).numpy()

        # Store embedding and metadata
        embeddings_data[filename] = {
            "filename": filename,
            "category": metadata[filename]["category"],
            "embedding": embedding[0].tolist(),  # Convert numpy to list
            "duration": len(audio_data) / sr,
            "file_size": audio_file.stat().st_size,
        }

    return embeddings_data


def organize_datasets(embeddings_data: dict) -> dict:
    """Organize embeddings into themed datasets."""
    print("📚 Organizing into themed datasets...")

    datasets = {}
    for dataset_name, categories in DATASET_CATEGORIES.items():
        dataset_clips = {}
        clip_id = 1

        for filename, data in embeddings_data.items():
            if data["category"] in categories:
                dataset_clips[clip_id] = {
                    "id": clip_id,
                    "filename": filename,
                    "category": data["category"],
                    "duration": data["duration"],
                    "file_size": data["file_size"],
                    "embedding": data["embedding"],
                }
                clip_id += 1

        datasets[dataset_name] = dataset_clips
        print(f"  ✓ {dataset_name}: {len(dataset_clips)} clips")

    return datasets


def save_datasets(datasets: dict, audio_dir: Path) -> None:
    """Save datasets with embeddings to disk."""
    print("💾 Saving datasets...")

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    for dataset_name, clips in datasets.items():
        # Save embeddings and metadata
        output_file = EMBEDDINGS_DIR / f"{dataset_name}.pkl"
        with open(output_file, "wb") as f:
            pickle.dump(
                {
                    "name": dataset_name,
                    "clips": clips,
                    "audio_dir": str(audio_dir.absolute()),
                },
                f,
            )
        print(f"  ✓ Saved: {output_file}")

        # Also save a human-readable JSON (without embeddings for readability)
        json_file = EMBEDDINGS_DIR / f"{dataset_name}_info.json"
        json_data = {
            "name": dataset_name,
            "num_clips": len(clips),
            "clips": {
                clip_id: {k: v for k, v in clip_data.items() if k != "embedding"}
                for clip_id, clip_data in clips.items()
            },
        }
        with open(json_file, "w") as f:
            json.dump(json_data, f, indent=2)
        print(f"  ✓ Saved info: {json_file}")


def main():
    print("=" * 60)
    print("VectoryTones Dataset Setup")
    print("=" * 60)
    print()

    # Step 1: Download ESC-50
    audio_dir = download_esc50()
    print()

    # Step 2: Load metadata
    print("📋 Loading ESC-50 metadata...")
    metadata = load_esc50_metadata(audio_dir.parent)
    print(f"  ✓ Loaded metadata for {len(metadata)} files")
    print()

    # Step 3: Load CLAP model
    model, processor = load_clap_model()
    print()

    # Step 4: Generate embeddings
    embeddings_data = generate_embeddings(audio_dir, model, processor, metadata)
    print(f"  ✓ Generated {len(embeddings_data)} embeddings")
    print()

    # Step 5: Organize into datasets
    datasets = organize_datasets(embeddings_data)
    print()

    # Step 6: Save datasets
    save_datasets(datasets, audio_dir)
    print()

    print("=" * 60)
    print("✅ Setup complete!")
    print("=" * 60)
    print(f"\nDatasets saved to: {EMBEDDINGS_DIR.absolute()}")
    print(f"Audio files at: {audio_dir.absolute()}")
    print("\nAvailable datasets:")
    for name, clips in datasets.items():
        print(f"  - {name}: {len(clips)} clips")
    print("\nYou can now run the Flask app with these preset datasets!")


if __name__ == "__main__":
    main()
