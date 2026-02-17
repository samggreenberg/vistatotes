import os

# Limit threads to reduce memory overhead in constrained environments
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Visual feedback for startup
print("⏳ Initializing VectoryTones...", flush=True)

import hashlib
from pathlib import Path

print("⏳ Importing ML libraries (this may take a few seconds)...", flush=True)

from flask import Flask
from tqdm import tqdm

# Import refactored modules
from config import NUM_CLIPS
from vectorytones.audio import generate_wav
from vectorytones.models import initialize_models
from vectorytones.routes import clips_bp, datasets_bp, main_bp, sorting_bp
from vectorytones.utils import clips

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Clip generation
# ---------------------------------------------------------------------------


def init_clips():
    print("DEBUG: Generating synthetic waveforms...", flush=True)
    for i in range(1, NUM_CLIPS + 1):
        freq = 200 + (i - 1) * 50  # 200 Hz .. 1150 Hz
        duration = round(1.0 + (i % 5) * 0.5, 1)  # 1.0 – 3.0 s
        wav_bytes = generate_wav(freq, duration)
        clips[i] = {
            "id": i,
            "type": "audio",
            "frequency": freq,
            "duration": duration,
            "file_size": len(wav_bytes),
            "md5": hashlib.md5(wav_bytes).hexdigest(),
            "embedding": None,
            "wav_bytes": wav_bytes,
        }


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
        # Production mode - load models and synthetic clips on startup
        print("🚀 Running in PRODUCTION mode", flush=True)

        # Use tqdm for loading feedback
        with tqdm(total=2, desc="Initializing", unit="step") as pbar:
            pbar.set_description("Initializing clips")
            init_clips()
            pbar.update(1)

            pbar.set_description("Loading models")
            initialize_models()
            pbar.update(1)

            pbar.set_description("Ready")

        print("✅ VectoryTones is ready!", flush=True)
        print("🌐 Open http://localhost:5000 in your browser", flush=True)

        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
