import hashlib
import io
import math
import struct
import wave

import numpy as np
import torch
import torch.nn as nn
import laion_clap
from sklearn.mixture import GaussianMixture

from flask import Flask, jsonify, request, send_file

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Clip generation
# ---------------------------------------------------------------------------

SAMPLE_RATE = 44100
NUM_CLIPS = 20

clips = {}  # id -> {id, duration, file_size, embedding, wav_bytes}
good_votes: set[int] = set()
bad_votes: set[int] = set()
inclusion = 0  # Inclusion setting: -10 to +10, default 0

# Load CLAP model for audio/text embeddings
clap_model = laion_clap.CLAP_Module(enable_fusion=False)
clap_model.load_ckpt()  # downloads default pretrained checkpoint


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
        }

    # Compute CLAP audio embeddings for all clips in one batch
    audio_arrays = []
    max_len = 0
    for i in range(1, NUM_CLIPS + 1):
        arr = wav_bytes_to_float(clips[i]["wav_bytes"])
        audio_arrays.append(arr)
        max_len = max(max_len, len(arr))

    # Pad all arrays to the same length for batched inference
    padded = np.zeros((NUM_CLIPS, max_len), dtype=np.float32)
    for idx, arr in enumerate(audio_arrays):
        padded[idx, : len(arr)] = arr

    embeddings = clap_model.get_audio_embedding_from_data(padded)

    for i in range(1, NUM_CLIPS + 1):
        clips[i]["embedding"] = embeddings[i - 1]


init_clips()

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
        result.append(
            {
                "id": c["id"],
                "frequency": c["frequency"],
                "duration": c["duration"],
                "file_size": c["file_size"],
            }
        )
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
    vote = data.get("vote")
    if vote not in ("good", "bad"):
        return jsonify({"error": "vote must be 'good' or 'bad'"}), 400

    if vote == "good":
        if clip_id in good_votes:
            good_votes.discard(clip_id)
        else:
            bad_votes.discard(clip_id)
            good_votes.add(clip_id)
    else:
        if clip_id in bad_votes:
            bad_votes.discard(clip_id)
        else:
            good_votes.discard(clip_id)
            bad_votes.add(clip_id)

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
        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(X)

        # Get the means of the two components
        means = gmm.means_.flatten()
        stds = np.sqrt(gmm.covariances_.flatten())

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
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "text is required"}), 400

    text_embedding = clap_model.get_text_embedding([text])
    text_vec = text_embedding[0]

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
        weight_true *= (2.0 ** inclusion_value)
    else:
        # Increase weight for False samples
        weight_false *= (2.0 ** (-inclusion_value))

    # Create sample weights
    weights = torch.where(y_train == 1, weight_true, weight_false).squeeze()
    loss_fn = nn.BCELoss(reduction='none')

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
    best_cost = float('inf')

    # Calculate weights based on inclusion
    if inclusion_value >= 0:
        fpr_weight = 1.0
        fnr_weight = 2.0 ** inclusion_value
    else:
        fpr_weight = 2.0 ** inclusion_value
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
    threshold = calculate_cross_calibration_threshold(X_list, y_list, input_dim, inclusion_value)

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
            "good": sorted(good_votes),
            "bad": sorted(bad_votes),
        }
    )


@app.route("/api/labels/export")
def export_labels():
    """Export labels as JSON keyed by clip MD5 hash."""
    labels = []
    for cid in sorted(good_votes):
        clip = clips.get(cid)
        if clip:
            labels.append({"md5": clip["md5"], "label": "good"})
    for cid in sorted(bad_votes):
        clip = clips.get(cid)
        if clip:
            labels.append({"md5": clip["md5"], "label": "bad"})
    return jsonify({"labels": labels})


@app.route("/api/labels/import", methods=["POST"])
def import_labels():
    """Import labels from JSON, matching clips by MD5 hash."""
    data = request.get_json(force=True)
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
            bad_votes.discard(cid)
            good_votes.add(cid)
        else:
            good_votes.discard(cid)
            bad_votes.add(cid)
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
    new_inclusion = data.get("inclusion")

    if not isinstance(new_inclusion, (int, float)):
        return jsonify({"error": "inclusion must be a number"}), 400

    # Clamp to -10 to +10 range
    new_inclusion = max(-10, min(10, new_inclusion))
    inclusion = new_inclusion

    return jsonify({"inclusion": inclusion})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
