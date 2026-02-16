import io
import math
import struct
import wave

import numpy as np
import torch
import torch.nn as nn
import laion_clap

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
    for clip_id, clip in clips.items():
        audio_vec = clip["embedding"]
        norm_product = np.linalg.norm(audio_vec) * np.linalg.norm(text_vec)
        if norm_product == 0:
            similarity = 0.0
        else:
            similarity = float(np.dot(audio_vec, text_vec) / norm_product)
        results.append({"id": clip_id, "similarity": round(similarity, 4)})

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return jsonify(results)


def train_and_score():
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
    model = nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
        nn.Sigmoid(),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.BCELoss()

    model.train()
    for _ in range(200):
        optimizer.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        optimizer.step()

    # Score every clip
    model.eval()
    all_ids = sorted(clips.keys())
    all_embs = np.array([clips[cid]["embedding"] for cid in all_ids])
    X_all = torch.tensor(all_embs, dtype=torch.float32)
    with torch.no_grad():
        scores = model(X_all).squeeze(1).tolist()

    results = [{"id": cid, "score": round(s, 4)} for cid, s in zip(all_ids, scores)]
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


@app.route("/api/learned-sort", methods=["POST"])
def learned_sort():
    """Train MLP on voted clips, return all clips sorted by predicted score."""
    if not good_votes or not bad_votes:
        return jsonify({"error": "need at least one good and one bad vote"}), 400
    results = train_and_score()
    return jsonify(results)


@app.route("/api/votes")
def get_votes():
    return jsonify(
        {
            "good": sorted(good_votes),
            "bad": sorted(bad_votes),
        }
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
