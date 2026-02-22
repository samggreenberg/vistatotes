"""Test clip generation and embedding cache management."""

import hashlib

import numpy as np

from vtsearch.audio import generate_wav
from vtsearch.config import DATA_DIR, NUM_CLIPS
from vtsearch.models import embed_audio_file
from vtsearch.utils import clips


def _embedding_cache_path():
    """Path to the cached test-clip embeddings file."""
    return DATA_DIR / "clip_embedding_cache.npz"


def _embedding_cache_key():
    """Deterministic key derived from clip-generation parameters so the cache
    auto-invalidates when NUM_CLIPS, frequencies, or durations change."""
    parts = []
    for i in range(1, NUM_CLIPS + 1):
        freq = 200 + (i - 1) * 50
        dur = round(1.0 + (i % 5) * 0.5, 1)
        parts.append(f"{i}:{freq}:{dur}")
    return hashlib.md5("|".join(parts).encode()).hexdigest()


def init_clips():
    """Generate test clips with embeddings, using a cache for speed."""
    DATA_DIR.mkdir(exist_ok=True)
    temp_path = DATA_DIR / "temp_embed.wav"
    cache_path = _embedding_cache_path()
    cache_key = _embedding_cache_key()

    # Try to load cached embeddings
    cached_embeddings = {}
    if cache_path.exists():
        try:
            data = np.load(cache_path, allow_pickle=False)
            if "cache_key" in data and str(data["cache_key"]) == cache_key:
                for i in range(1, NUM_CLIPS + 1):
                    k = f"emb_{i}"
                    if k in data:
                        cached_embeddings[i] = data[k]
        except Exception:
            pass

    new_embeddings = {}
    for i in range(1, NUM_CLIPS + 1):
        freq = 200 + (i - 1) * 50  # 200 Hz .. 1150 Hz
        duration = round(1.0 + (i % 5) * 0.5, 1)  # 1.0 – 3.0 s
        wav_bytes = generate_wav(freq, duration)

        if i in cached_embeddings:
            embedding = cached_embeddings[i]
        else:
            # Generate embedding by saving to temp file
            temp_path.write_bytes(wav_bytes)
            embedding = embed_audio_file(temp_path)
            new_embeddings[i] = embedding

        fname = f"test_clip_{i}.wav"
        clips[i] = {
            "id": i,
            "type": "audio",
            "frequency": freq,
            "duration": duration,
            "file_size": len(wav_bytes),
            "md5": hashlib.md5(wav_bytes).hexdigest(),
            "embedding": embedding,
            "clip_bytes": wav_bytes,
            "filename": fname,
            "category": "test",
            "origin": {"importer": "test", "params": {}},
            "origin_name": fname,
        }

    # Clean up temp file
    if temp_path.exists():
        temp_path.unlink()

    # Save cache if we computed any new embeddings (or cache didn't exist)
    if new_embeddings or not cached_embeddings:
        save_data = {"cache_key": np.array(cache_key)}
        for i in range(1, NUM_CLIPS + 1):
            save_data[f"emb_{i}"] = clips[i]["embedding"]
        np.savez(cache_path, **save_data)
