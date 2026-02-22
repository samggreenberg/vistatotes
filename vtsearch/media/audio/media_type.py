"""Audio media type — CLAP embeddings, WAV/MP3/FLAC/OGG/M4A files."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import torch
from transformers import ClapModel, ClapProcessor

from config import CLAP_MODEL_ID, DATA_DIR, MODELS_CACHE_DIR, SAMPLE_RATE
from vtsearch.media.base import DemoDataset, MediaResponse, MediaType, ProgressCallback, _noop_progress


class AudioMediaType(MediaType):
    """Handles audio clips using the CLAP model (laion/clap-htsat-unfused).

    * Embeds audio files via CLAP's audio encoder + projection head.
    * Embeds text queries via CLAP's text encoder + projection head, so
      queries land in the same 512-dimensional space as audio embeddings.
    * Serves clips as ``audio/wav`` streams.
    """

    def __init__(self) -> None:
        self._model: Optional[ClapModel] = None
        self._processor: Optional[ClapProcessor] = None
        self._on_progress: ProgressCallback = _noop_progress

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def type_id(self) -> str:
        return "audio"

    @property
    def name(self) -> str:
        return "Audio"

    @property
    def icon(self) -> str:
        return "🔊"

    # ------------------------------------------------------------------
    # File import
    # ------------------------------------------------------------------

    @property
    def file_extensions(self) -> list:
        return ["*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a"]

    @property
    def folder_import_name(self) -> str:
        return "sounds"

    # ------------------------------------------------------------------
    # Viewer
    # ------------------------------------------------------------------

    @property
    def loops(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Demo datasets
    # ------------------------------------------------------------------

    @property
    def demo_datasets(self) -> list:
        return [
            DemoDataset(
                id="sounds_s",
                label="ESC-50 Animals & Fireside (S)",
                description=(
                    "200 clips of dogs, cats, roosters, church bells, and crackling"
                    " fire from the ESC-50 collection."
                ),
                categories=[
                    "dog",
                    "cat",
                    "rooster",
                    "church_bells",
                    "crackling_fire",
                ],
                required_folder=DATA_DIR / "ESC-50-master" / "audio",
            ),
            DemoDataset(
                id="sounds_m",
                label="ESC-50 Everyday Sounds (M)",
                description=(
                    "400 clips of babies, laughter, clapping, footsteps, chainsaws,"
                    " airplanes, and more from the ESC-50 collection."
                ),
                categories=[
                    "crying_baby",
                    "laughing",
                    "clapping",
                    "footsteps",
                    "sneezing",
                    "chainsaw",
                    "airplane",
                    "fireworks",
                    "pig",
                    "cow",
                ],
                required_folder=DATA_DIR / "ESC-50-master" / "audio",
            ),
            DemoDataset(
                id="sounds_l",
                label="ESC-50 Environmental Mix (L)",
                description=(
                    "800 clips spanning nature, animals, weather, traffic, and"
                    " household sounds from the ESC-50 collection."
                ),
                categories=[
                    "chirping_birds",
                    "crow",
                    "frog",
                    "insects",
                    "rain",
                    "sea_waves",
                    "thunderstorm",
                    "wind",
                    "water_drops",
                    "crickets",
                    "car_horn",
                    "siren",
                    "engine",
                    "train",
                    "helicopter",
                    "vacuum_cleaner",
                    "washing_machine",
                    "clock_alarm",
                    "keyboard_typing",
                    "door_wood_knock",
                ],
                required_folder=DATA_DIR / "ESC-50-master" / "audio",
            ),
        ]

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    @property
    def description_wrappers(self) -> list[str]:
        return [
            "the sound of {text}",
            "a recording of {text}",
            "{text}",
            "audio of {text}",
            "the noise of {text}",
        ]

    def load_models(self) -> None:
        if self._model is not None:
            return
        import gc

        gc.collect()
        cache_dir = str(MODELS_CACHE_DIR)
        self._on_progress("loading", "Loading audio embedder (CLAP model)...", 0, 0)
        self._model = ClapModel.from_pretrained(CLAP_MODEL_ID, low_cpu_mem_usage=True, cache_dir=cache_dir)
        self._processor = ClapProcessor.from_pretrained(CLAP_MODEL_ID, cache_dir=cache_dir)

    def embed_media(self, file_path: Path) -> Optional[np.ndarray]:
        if self._model is None:
            self.load_models()
        if self._model is None or self._processor is None:
            return None
        try:
            audio_data, _sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
            inputs = self._processor(
                audio=audio_data,
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
                padding="max_length",
                max_length=480000,
                truncation=True,
            )
            with torch.no_grad():
                outputs = self._model.audio_model(**inputs)
                embedding = self._model.audio_projection(outputs.pooler_output).detach().cpu().numpy()
            return embedding[0]
        except Exception as e:
            print(f"Error embedding {file_path}: {e}")
            return None

    def embed_text(self, text: str) -> Optional[np.ndarray]:
        if self._model is None:
            self.load_models()
        if self._model is None or self._processor is None:
            return None
        try:
            inputs = self._processor(text=[text], return_tensors="pt")
            with torch.no_grad():
                outputs = self._model.text_model(**inputs)
                text_vec = self._model.text_projection(outputs.pooler_output).detach().cpu().numpy()[0]
            return text_vec
        except Exception as e:
            print(f"Error embedding text query for audio: {e}")
            return None

    # internal helpers used by loader.py's get_clap_model() bridge
    def _get_model_and_processor(self):
        if self._model is None:
            self.load_models()
        return self._model, self._processor

    # ------------------------------------------------------------------
    # Clip data
    # ------------------------------------------------------------------

    def load_clip_data(self, file_path: Path) -> dict:
        with open(file_path, "rb") as f:
            clip_bytes = f.read()
        try:
            audio_data, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
            duration = len(audio_data) / sr
        except Exception:
            duration = 0.0
        return {"clip_bytes": clip_bytes, "duration": duration}

    # ------------------------------------------------------------------
    # HTTP serving
    # ------------------------------------------------------------------

    def clip_response(self, clip: dict) -> MediaResponse:
        return MediaResponse(
            data=clip["clip_bytes"],
            mimetype="audio/wav",
            download_name=f"clip_{clip['id']}.wav",
        )
