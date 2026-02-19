"""Image media type — CLIP embeddings, JPEG/PNG/GIF/BMP/WEBP files."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from flask import Response, send_file
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from config import CLIP_MODEL_ID, MODELS_CACHE_DIR
from vistatotes.media.base import DemoDataset, MediaType

_IMAGE_MIME_TYPES: dict[str, str] = {
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


class ImageMediaType(MediaType):
    """Handles image clips using the CLIP model (openai/clip-vit-base-patch32).

    * Embeds images via CLIP's vision encoder (768-dim vectors).
    * Embeds text queries via CLIP's text encoder (same 768-dim space).
    * Serves clips as image files with MIME types inferred from extension.
    * Also exposes :meth:`embed_pil_image` for in-memory PIL Image objects
      (used when generating CIFAR-10 demo datasets).
    """

    def __init__(self) -> None:
        self._model: Optional[CLIPModel] = None
        self._processor: Optional[CLIPProcessor] = None

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def type_id(self) -> str:
        return "image"

    @property
    def name(self) -> str:
        return "Image"

    @property
    def icon(self) -> str:
        return "🖼️"

    # ------------------------------------------------------------------
    # File import
    # ------------------------------------------------------------------

    @property
    def file_extensions(self) -> list:
        return ["*.jpg", "*.jpeg", "*.png", "*.gif", "*.bmp", "*.webp"]

    @property
    def folder_import_name(self) -> str:
        return "images"

    # ------------------------------------------------------------------
    # Viewer
    # ------------------------------------------------------------------

    @property
    def loops(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # Demo datasets
    # ------------------------------------------------------------------

    @property
    def demo_datasets(self) -> list:
        return [
            DemoDataset(
                id="animals_images",
                label="Animals & Wildlife",
                description=(
                    "400 photographs of birds, cats, dogs, horses, deer, and frogs"
                    " sourced from the CIFAR-10 dataset."
                ),
                categories=["bird", "cat", "deer", "dog", "frog", "horse"],
                source="cifar10_sample",
            ),
            DemoDataset(
                id="vehicles_images",
                label="Vehicles & Transport",
                description=(
                    "400 photographs of airplanes, cars, ships, and trucks sourced"
                    " from the CIFAR-10 dataset."
                ),
                categories=["airplane", "automobile", "ship", "truck"],
                source="cifar10_sample",
            ),
        ]

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def load_models(self) -> None:
        if self._model is not None:
            return
        import gc

        gc.collect()
        cache_dir = str(MODELS_CACHE_DIR)
        print("DEBUG: Loading CLIP model for Image...", flush=True)
        self._model = CLIPModel.from_pretrained(
            CLIP_MODEL_ID, low_cpu_mem_usage=True, cache_dir=cache_dir
        )
        self._processor = CLIPProcessor.from_pretrained(
            CLIP_MODEL_ID, cache_dir=cache_dir
        )
        print("DEBUG: CLIP model loaded.", flush=True)

    def embed_media(self, file_path: Path) -> Optional[np.ndarray]:
        if self._model is None:
            self.load_models()
        if self._model is None or self._processor is None:
            return None
        try:
            image = Image.open(file_path).convert("RGB")
            return self.embed_pil_image(image)
        except Exception as e:
            print(f"Error embedding {file_path}: {e}")
            return None

    def embed_pil_image(self, image: Image.Image) -> Optional[np.ndarray]:
        """Embed a PIL Image that is already in memory (e.g. from CIFAR-10)."""
        if self._model is None:
            self.load_models()
        if self._model is None or self._processor is None:
            return None
        try:
            image = image.convert("RGB")
            inputs = self._processor(images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = self._model.get_image_features(**inputs)
                embedding = outputs.detach().cpu().numpy()
            return embedding[0]
        except Exception as e:
            print(f"Error embedding PIL image: {e}")
            return None

    def embed_text(self, text: str) -> Optional[np.ndarray]:
        if self._model is None:
            self.load_models()
        if self._model is None or self._processor is None:
            return None
        try:
            inputs = self._processor(text=[text], return_tensors="pt")
            with torch.no_grad():
                text_vec = (
                    self._model.get_text_features(**inputs).detach().cpu().numpy()[0]
                )
            return text_vec
        except Exception as e:
            print(f"Error embedding text query for image: {e}")
            return None

    # internal helper used by loader.py's get_clip_model() bridge
    def _get_model_and_processor(self):
        return self._model, self._processor

    # ------------------------------------------------------------------
    # Clip data
    # ------------------------------------------------------------------

    def load_clip_data(self, file_path: Path) -> dict:
        with open(file_path, "rb") as f:
            image_bytes = f.read()
        try:
            img = Image.open(file_path)
            width, height = img.width, img.height
        except Exception:
            width, height = None, None
        return {
            "image_bytes": image_bytes,
            "duration": 0,
            "width": width,
            "height": height,
        }

    # ------------------------------------------------------------------
    # HTTP serving
    # ------------------------------------------------------------------

    def clip_response(self, clip: dict) -> Response:
        filename = clip.get("filename", "")
        ext = Path(filename).suffix.lower() if filename else ".jpg"
        mimetype = _IMAGE_MIME_TYPES.get(ext, "image/jpeg")
        return send_file(
            io.BytesIO(clip["image_bytes"]),
            mimetype=mimetype,
            download_name=f"clip_{clip['id']}{ext}",
        )
