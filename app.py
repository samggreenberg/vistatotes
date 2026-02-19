import os

# Limit threads to reduce memory overhead in constrained environments
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Visual feedback for startup
print("⏳ Initializing VistaTotes...", flush=True)

import hashlib

print("⏳ Importing ML libraries (this may take a few seconds)...", flush=True)

from flask import Flask
from tqdm import tqdm

# Import refactored modules
from config import DATA_DIR, NUM_CLIPS
from vistatotes.audio import generate_wav
from vistatotes.models import embed_audio_file, initialize_models
from vistatotes.routes import clips_bp, datasets_bp, main_bp, sorting_bp
from vistatotes.utils import clips

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Clip generation
# ---------------------------------------------------------------------------


def init_clips():
    print("DEBUG: Generating synthetic waveforms...", flush=True)
    DATA_DIR.mkdir(exist_ok=True)
    temp_path = DATA_DIR / "temp_embed.wav"

    for i in range(1, NUM_CLIPS + 1):
        freq = 200 + (i - 1) * 50  # 200 Hz .. 1150 Hz
        duration = round(1.0 + (i % 5) * 0.5, 1)  # 1.0 – 3.0 s
        wav_bytes = generate_wav(freq, duration)

        # Generate embedding by saving to temp file
        temp_path.write_bytes(wav_bytes)
        embedding = embed_audio_file(temp_path)

        clips[i] = {
            "id": i,
            "type": "audio",
            "frequency": freq,
            "duration": duration,
            "file_size": len(wav_bytes),
            "md5": hashlib.md5(wav_bytes).hexdigest(),
            "embedding": embedding,
            "wav_bytes": wav_bytes,
        }

    # Clean up temp file
    if temp_path.exists():
        temp_path.unlink()


# Model initialization is now handled by vectorytones.models.initialize_models()


# ---------------------------------------------------------------------------
# Register Blueprints
# ---------------------------------------------------------------------------

app.register_blueprint(main_bp)
app.register_blueprint(clips_bp)
app.register_blueprint(sorting_bp)
app.register_blueprint(datasets_bp)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # Check if we're running in local mode or deployment mode
    if len(sys.argv) > 1 and sys.argv[1] == "--local":
        # Local development mode
        print("🚀 Running in LOCAL mode (accessible from other devices)", flush=True)
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
    else:
        # Production mode - load models on startup; dataset is chosen by the user
        print("🚀 Running in PRODUCTION mode", flush=True)

        # Use tqdm for loading feedback
        with tqdm(total=1, desc="Initializing", unit="step") as pbar:
            pbar.set_description("Loading models")
            initialize_models()
            pbar.update(1)

            pbar.set_description("Ready")

        print("✅ VistaTotes is ready!", flush=True)
        print("🌐 Open http://localhost:5000 in your browser", flush=True)

        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
