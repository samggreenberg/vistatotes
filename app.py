import os

# Limit threads to reduce memory overhead in constrained environments
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

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

import librosa
import numpy as np
import requests
import torch
import torch.nn as nn
from flask import Flask, jsonify, request, send_file
from sklearn.mixture import GaussianMixture
from transformers import ClapModel, ClapProcessor

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Clip generation
# ---------------------------------------------------------------------------

SAMPLE_RATE = 48000
NUM_CLIPS = 20

clips = {}  # id -> {id, duration, file_size, embedding, wav_bytes}
good_votes: dict[int, None] = {}  # OrderedDict behavior via dict (Python 3.7+)
bad_votes: dict[int, None] = {}  # OrderedDict behavior via dict (Python 3.7+)
inclusion: int = 0  # Inclusion setting: -10 to +10, default 0

# Load CLAP model for audio/text embeddings
clap_model: Optional[ClapModel] = None
clap_processor: Optional[ClapProcessor] = None

# Dataset management
DATA_DIR = Path("data")
AUDIO_DIR = DATA_DIR / "audio"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
ESC50_URL = "https://github.com/karolpiczak/ESC-50/archive/master.zip"

# Progress tracking for long-running operations
progress_lock = threading.Lock()
progress_data = {
    "status": "idle",  # idle, loading, downloading, embedding
    "message": "",
    "current": 0,
    "total": 0,
    "error": None,
}

# Demo dataset definitions (from setup_datasets.py)
# ESC-50 has 40 clips per category
CLIPS_PER_CATEGORY = 40
ESC50_DOWNLOAD_SIZE_MB = 600  # Approximate download size of full ESC-50 dataset

DEMO_DATASETS = {
    "animals": {
        "categories": [
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
        "description": "Animal and nature sounds",
    },
    "natural": {
        "categories": [
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
        "description": "Natural environmental sounds",
    },
    "urban": {
        "categories": [
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
        "description": "Urban and mechanical sounds",
    },
    "household": {
        "categories": [
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
        "description": "Household and human sounds",
    },
}


def generate_wav(frequency: float, duration: float) -> bytes:
    """Return raw WAV bytes for a sine-wave tone."""
    num_samples = int(SAMPLE_RATE * duration)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        samples = []
        for i in range(num_samples):
            t = i / SAMPLE_RATE
            value = int(32767 * 0.5 * math.sin(2 * math.pi * frequency * t))
            samples.append(struct.pack("<h", value))
        wf.writeframes(b"".join(samples))
    return buf.getvalue()


def wav_bytes_to_float(wav_bytes: bytes) -> np.ndarray:
    """Convert WAV bytes to a float32 numpy array normalised to [-1, 1]."""
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        frames = wf.readframes(wf.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


def init_clips():
    print("DEBUG: Generating synthetic waveforms...", flush=True)
    for i in range(1, NUM_CLIPS + 1):
        freq = 200 + (i - 1) * 50  # 200 Hz .. 1150 Hz
        duration = round(1.0 + (i % 5) * 0.5, 1)  # 1.0 – 3.0 s
        wav_bytes = generate_wav(freq, duration)
        clips[i] = {
            "id": i,
            "frequency": freq,
            "duration": duration,
            "file_size": len(wav_bytes),
            "md5": hashlib.md5(wav_bytes).hexdigest(),
            "embedding": None,
            "wav_bytes": wav_bytes,
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
            embeddings = clap_model.audio_projection(outputs.pooler_output).numpy()
        print("DEBUG: Inference complete.", flush=True)

    for i in range(1, NUM_CLIPS + 1):
        clips[i]["embedding"] = embeddings[i - 1]


def initialize_app():
    global clap_model, clap_processor
    print("DEBUG: initialize_app called", flush=True)

    # Optimize for low-memory environments
    torch.set_num_threads(1)
    gc.collect()

    if clap_model is None:
        print("DEBUG: Loading CLAP model (Hugging Face)...", flush=True)
        # Use the unfused model (~600MB) and low_cpu_mem_usage to avoid RAM spikes
        model_id = "laion/clap-htsat-unfused"
        clap_model = ClapModel.from_pretrained(model_id, low_cpu_mem_usage=True)
        clap_processor = ClapProcessor.from_pretrained(model_id)
        print("DEBUG: CLAP model loaded.", flush=True)

    # Don't automatically load clips - user will load dataset via UI
    print("DEBUG: Ready to load dataset via UI", flush=True)


# ---------------------------------------------------------------------------
# Dataset management functions
# ---------------------------------------------------------------------------


def update_progress(status, message="", current=0, total=0, error=None):
    """Update the global progress tracker."""
    with progress_lock:
        progress_data["status"] = status
        progress_data["message"] = message
        progress_data["current"] = current
        progress_data["total"] = total
        progress_data["error"] = error


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
    missing_audio = 0
    for clip_id, clip_info in clips_data.items():
        # Load the actual audio file if we have a filename and audio_dir
        wav_bytes = None
        if "wav_bytes" in clip_info:
            wav_bytes = clip_info["wav_bytes"]
        elif "filename" in clip_info and "audio_dir" in data:
            audio_path = Path(data["audio_dir"]) / clip_info["filename"]
            if audio_path.exists():
                with open(audio_path, "rb") as f:
                    wav_bytes = f.read()
            else:
                missing_audio += 1

        if wav_bytes:
            clips[clip_id] = {
                "id": clip_id,
                "duration": clip_info.get("duration", 0),
                "file_size": clip_info.get("file_size", len(wav_bytes)),
                "md5": hashlib.md5(wav_bytes).hexdigest(),
                "embedding": np.array(clip_info["embedding"]),
                "wav_bytes": wav_bytes,
                "filename": clip_info.get("filename", f"clip_{clip_id}.wav"),
                "category": clip_info.get("category", "unknown"),
            }

    if missing_audio > 0:
        print(
            f"WARNING: {missing_audio} audio files missing from {file_path}", flush=True
        )


def embed_audio_file(audio_path: Path) -> Optional[np.ndarray]:
    """Generate CLAP embedding for a single audio file."""
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
            embedding = clap_model.audio_projection(outputs.pooler_output).numpy()

        return embedding[0]
    except Exception as e:
        print(f"Error embedding {audio_path}: {e}")
        return None


def load_dataset_from_folder(folder_path: Path):
    """Generate dataset from a folder of audio files."""
    global clips

    update_progress("embedding", "Scanning audio files...", 0, 0)

    # Find all audio files
    audio_files = []
    for ext in ["*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a"]:
        audio_files.extend(folder_path.glob(ext))

    if not audio_files:
        raise ValueError("No audio files found in folder")

    clips.clear()
    clip_id = 1

    total = len(audio_files)
    for i, audio_path in enumerate(audio_files):
        update_progress("embedding", f"Embedding {audio_path.name}...", i + 1, total)

        embedding = embed_audio_file(audio_path)
        if embedding is None:
            continue

        # Load file to get wav_bytes
        with open(audio_path, "rb") as f:
            file_bytes = f.read()

        # Get duration
        try:
            audio_data, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
            duration = len(audio_data) / sr
        except Exception:
            duration = 0

        clips[clip_id] = {
            "id": clip_id,
            "duration": duration,
            "file_size": len(file_bytes),
            "md5": hashlib.md5(file_bytes).hexdigest(),
            "embedding": embedding,
            "wav_bytes": file_bytes,
            "filename": audio_path.name,
            "category": "custom",
        }
        clip_id += 1

    update_progress("idle", f"Loaded {len(clips)} clips from folder")


def download_file_with_progress(url: str, dest_path: Path):
    """Download a file with progress tracking."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))

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
        download_file_with_progress(ESC50_URL, zip_path)

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


def load_demo_dataset(dataset_name: str):
    """Load a demo dataset, downloading and embedding if necessary."""
    global clips

    if dataset_name not in DEMO_DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Check if already embedded
    pkl_file = EMBEDDINGS_DIR / f"{dataset_name}.pkl"
    if pkl_file.exists():
        update_progress("loading", f"Loading {dataset_name} dataset...", 0, 0)
        load_dataset_from_pickle(pkl_file)

        # Check if any clips were actually loaded
        if len(clips) == 0:
            # Pickle file exists but audio files are missing, delete and re-embed
            update_progress(
                "loading", f"Audio files missing, re-embedding {dataset_name}...", 0, 0
            )
            pkl_file.unlink()
        else:
            update_progress("idle", f"Loaded {dataset_name} dataset")
            return

    # Need to download and embed
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
            "duration": duration,
            "file_size": len(wav_bytes),
            "md5": hashlib.md5(wav_bytes).hexdigest(),
            "embedding": embedding,
            "wav_bytes": wav_bytes,
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
                "duration": clip["duration"],
                "file_size": clip["file_size"],
                "md5": clip["md5"],
                "embedding": clip["embedding"].tolist()
                if isinstance(clip["embedding"], np.ndarray)
                else clip["embedding"],
                "filename": clip.get("filename", f"clip_{cid}.wav"),
                "category": clip.get("category", "unknown"),
                "wav_bytes": clip["wav_bytes"],
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


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/api/clips")
def list_clips():
    result = []
    for c in clips.values():
        clip_data = {
            "id": c["id"],
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


@app.route("/api/clips/<int:clip_id>/vote", methods=["POST"])
def vote_clip(clip_id):
    if clip_id not in clips:
        return jsonify({"error": "not found"}), 404
    data = request.get_json(force=True)
    if data is None:
        return jsonify({"error": "Invalid request body"}), 400

    vote = data.get("vote")
    if vote not in ("good", "bad"):
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

    return jsonify({"ok": True})


def calculate_gmm_threshold(scores):
    """
    Use Gaussian Mixture Model to find threshold between two distributions.
    Assumes scores come from a bimodal distribution (Bad + Good).
    """
    if len(scores) < 2:
        return 0.5

    # Reshape for sklearn
    X = np.array(scores).reshape(-1, 1)

    try:
        # Fit a 2-component GMM
        gmm: GaussianMixture = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(X)

        # Get the means of the two components
        means = np.ravel(gmm.means_)
        # stds = np.sqrt(gmm.covariances_.flatten())

        # Identify which component is "low" (Bad) and which is "high" (Good)
        low_idx = 0 if means[0] < means[1] else 1
        high_idx = 1 - low_idx

        # Threshold is at the intersection of the two Gaussians
        # For simplicity, use the midpoint between means
        threshold = (means[low_idx] + means[high_idx]) / 2.0

        return float(threshold)
    except Exception:
        # If GMM fails, return median
        return float(np.median(scores))


@app.route("/api/sort", methods=["POST"])
def sort_clips():
    """Return clips sorted by cosine similarity to a text query."""
    data = request.get_json(force=True)
    if data is None:
        return jsonify({"error": "Invalid request body"}), 400

    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "text is required"}), 400

    if clap_model is None or clap_processor is None:
        return jsonify({"error": "CLAP model not loaded"}), 500

    inputs = clap_processor(text=[text], return_tensors="pt")  # type: ignore
    with torch.no_grad():
        outputs = clap_model.text_model(**inputs)
        text_vec = clap_model.text_projection(outputs.pooler_output).numpy()[0]

    results = []
    scores = []
    for clip_id, clip in clips.items():
        audio_vec = clip["embedding"]
        norm_product = np.linalg.norm(audio_vec) * np.linalg.norm(text_vec)
        if norm_product == 0:
            similarity = 0.0
        else:
            similarity = float(np.dot(audio_vec, text_vec) / norm_product)
        results.append({"id": clip_id, "similarity": round(similarity, 4)})
        scores.append(similarity)

    # Calculate GMM-based threshold
    threshold = calculate_gmm_threshold(scores)

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return jsonify({"results": results, "threshold": round(threshold, 4)})


def train_model(X_train, y_train, input_dim, inclusion_value=0):
    """
    Train a small MLP and return the trained model.

    Args:
        X_train: Training data
        y_train: Training labels (1 for good, 0 for bad)
        input_dim: Input dimension
        inclusion_value: Inclusion setting (-10 to +10)
            - If 0: balance classes equally
            - If positive: weight True samples more (effectively more Trues)
            - If negative: weight False samples more (effectively more Falses)
    """
    model = nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
        nn.Sigmoid(),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Calculate class weights based on inclusion
    num_true = y_train.sum().item()
    num_false = len(y_train) - num_true

    # Base weights for balanced classes
    if num_true > 0 and num_false > 0:
        weight_true = num_false / num_true
        weight_false = 1.0
    else:
        weight_true = 1.0
        weight_false = 1.0

    # Adjust weights based on inclusion
    if inclusion_value >= 0:
        # Increase weight for True samples
        weight_true *= 2.0**inclusion_value
    else:
        # Increase weight for False samples
        weight_false *= 2.0 ** (-inclusion_value)

    # Create sample weights
    weights = torch.where(y_train == 1, weight_true, weight_false).squeeze()
    loss_fn = nn.BCELoss(reduction="none")

    model.train()
    for _ in range(200):
        optimizer.zero_grad()
        predictions = model(X_train)
        losses = loss_fn(predictions, y_train)
        weighted_loss = (losses.squeeze() * weights).mean()
        weighted_loss.backward()
        optimizer.step()

    model.eval()
    return model


def find_optimal_threshold(scores, labels, inclusion_value=0):
    """
    Find the best threshold that separates good (1) from bad (0).

    Args:
        scores: List of model scores
        labels: List of true labels (1 for good, 0 for bad)
        inclusion_value: Inclusion setting (-10 to +10)
            - If 0: minimize fpr + fnr
            - If positive: minimize fpr + 2^inclusion * fnr (include more)
            - If negative: minimize 2^inclusion * fpr + fnr (exclude more)
    """
    sorted_pairs = sorted(zip(scores, labels), reverse=True)
    best_threshold = 0.5
    best_cost = float("inf")

    # Calculate weights based on inclusion
    if inclusion_value >= 0:
        fpr_weight = 1.0
        fnr_weight = 2.0**inclusion_value
    else:
        fpr_weight = 2.0**inclusion_value
        fnr_weight = 1.0

    for i in range(len(sorted_pairs)):
        threshold = sorted_pairs[i][0]

        # Calculate FPR and FNR at this threshold
        fp = 0  # false positives
        fn = 0  # false negatives
        tp = 0  # true positives
        tn = 0  # true negatives

        for score, label in sorted_pairs:
            predicted = 1 if score >= threshold else 0
            if predicted == 1 and label == 0:
                fp += 1
            elif predicted == 0 and label == 1:
                fn += 1
            elif predicted == 1 and label == 1:
                tp += 1
            else:  # predicted == 0 and label == 0
                tn += 1

        # Calculate rates
        total_positives = sum(1 for _, label in sorted_pairs if label == 1)
        total_negatives = len(sorted_pairs) - total_positives

        fpr = fp / total_negatives if total_negatives > 0 else 0
        fnr = fn / total_positives if total_positives > 0 else 0

        # Calculate weighted cost
        cost = fpr_weight * fpr + fnr_weight * fnr

        if cost < best_cost:
            best_cost = cost
            best_threshold = threshold

    return best_threshold


def calculate_cross_calibration_threshold(X_list, y_list, input_dim, inclusion_value=0):
    """
    Calculate threshold using cross-calibration:
    - Split data into two halves D1 and D2
    - Train M1 on D1, find threshold t1 using M1 on D2
    - Train M2 on D2, find threshold t2 using M2 on D1
    - Return mean(t1, t2)
    """
    n = len(X_list)
    if n < 4:
        # Not enough data for cross-calibration
        return 0.5

    # Split data in half
    mid = n // 2
    indices = np.random.permutation(n)
    idx1 = indices[:mid]
    idx2 = indices[mid:]

    X_np = np.array(X_list)
    y_np = np.array(y_list)

    # Train M1 on D1
    X1 = torch.tensor(X_np[idx1], dtype=torch.float32)
    y1 = torch.tensor(y_np[idx1], dtype=torch.float32).unsqueeze(1)
    M1 = train_model(X1, y1, input_dim, inclusion_value)

    # Train M2 on D2
    X2 = torch.tensor(X_np[idx2], dtype=torch.float32)
    y2 = torch.tensor(y_np[idx2], dtype=torch.float32).unsqueeze(1)
    M2 = train_model(X2, y2, input_dim, inclusion_value)

    # Find t1: use M1 on D2
    with torch.no_grad():
        scores1_on_2 = M1(X2).squeeze(1).tolist()
    t1 = find_optimal_threshold(scores1_on_2, y_np[idx2].tolist(), inclusion_value)

    # Find t2: use M2 on D1
    with torch.no_grad():
        scores2_on_1 = M2(X1).squeeze(1).tolist()
    t2 = find_optimal_threshold(scores2_on_1, y_np[idx1].tolist(), inclusion_value)

    # Return mean
    return (t1 + t2) / 2.0


def train_and_score(inclusion_value=0):
    """Train a small MLP on voted clip embeddings and score every clip."""
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

    # Calculate threshold using cross-calibration
    threshold = calculate_cross_calibration_threshold(
        X_list, y_list, input_dim, inclusion_value
    )

    # Train final model on all data
    model = train_model(X, y, input_dim, inclusion_value)

    # Score every clip
    all_ids = sorted(clips.keys())
    all_embs = np.array([clips[cid]["embedding"] for cid in all_ids])
    X_all = torch.tensor(all_embs, dtype=torch.float32)
    with torch.no_grad():
        scores = model(X_all).squeeze(1).tolist()

    results = [{"id": cid, "score": round(s, 4)} for cid, s in zip(all_ids, scores)]
    results.sort(key=lambda x: x["score"], reverse=True)
    return results, threshold


@app.route("/api/learned-sort", methods=["POST"])
def learned_sort():
    """Train MLP on voted clips, return all clips sorted by predicted score."""
    if not good_votes or not bad_votes:
        return jsonify({"error": "need at least one good and one bad vote"}), 400
    results, threshold = train_and_score(inclusion)
    return jsonify({"results": results, "threshold": round(threshold, 4)})


@app.route("/api/votes")
def get_votes():
    return jsonify(
        {
            "good": list(good_votes),  # Maintains insertion order (dict keys)
            "bad": list(bad_votes),    # Maintains insertion order (dict keys)
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

        # Calculate number of files (40 clips per ESC-50 category)
        num_categories = len(dataset_info["categories"])
        num_files = num_categories * CLIPS_PER_CATEGORY

        # Calculate download size
        if is_ready:
            # If ready, show the actual .pkl file size
            download_size_mb = pkl_file.stat().st_size / (1024 * 1024)
        else:
            # If not ready, user needs to download full ESC-50
            download_size_mb = ESC50_DOWNLOAD_SIZE_MB

        demos.append(
            {
                "name": name,
                "ready": is_ready,
                "num_categories": num_categories,
                "num_files": num_files,
                "download_size_mb": round(download_size_mb, 1),
                "description": dataset_info.get("description", ""),
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
    """Generate dataset from a folder of audio files."""
    data = request.get_json(force=True)
    if data is None:
        return jsonify({"error": "Invalid request body"}), 400

    folder_path = data.get("path")

    if not folder_path:
        return jsonify({"error": "No folder path provided"}), 400

    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        return jsonify({"error": "Invalid folder path"}), 400

    def load_task():
        try:
            clear_dataset()
            load_dataset_from_folder(folder)
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
    initialize_app()
    # Disable reloader and debug mode to save memory (prevents OOM in Codespaces)
    app.run(debug=False, use_reloader=False, port=5000)
