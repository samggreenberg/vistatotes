"""Image media type — CLIP embeddings, JPEG/PNG/GIF/BMP/WEBP files."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from config import CLIP_MODEL_ID, DATA_DIR, MODELS_CACHE_DIR
from vtsearch.media.base import DemoDataset, MediaResponse, MediaType, ProgressCallback, _noop_progress


def _extract_tensor(output: object) -> torch.Tensor:
    """Extract a plain tensor from model output.

    Depending on the transformers version, get_image_features() / get_text_features()
    may return either a raw tensor or a BaseModelOutputWithPooling dataclass.
    This helper handles both cases.
    """
    if isinstance(output, torch.Tensor):
        return output
    for attr in ("image_embeds", "text_embeds", "pooler_output"):
        val = getattr(output, attr, None)
        if isinstance(val, torch.Tensor):
            return val
    # Final fallback: treat as tuple-like and return first element
    return output[0]  # type: ignore[index]


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
        self._on_progress: ProgressCallback = _noop_progress

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
                id="images_s",
                label="Caltech-101 Nature & Flight (S)",
                description=(
                    "Photographs of butterflies, sunflowers, starfish, and"
                    " helicopters from the Caltech-101 dataset."
                ),
                categories=["butterfly", "sunflower", "starfish", "helicopter"],
                source="caltech101",
                required_folder=DATA_DIR / "caltech-101" / "101_ObjectCategories",
            ),
            DemoDataset(
                id="images_m",
                label="Caltech-101 Animals & Objects (M)",
                description=(
                    "Photographs of dolphins, grand pianos, elephants, kangaroos,"
                    " laptops, lobsters, watches, and flamingos from Caltech-101."
                ),
                categories=[
                    "dolphin",
                    "grand_piano",
                    "elephant",
                    "kangaroo",
                    "laptop",
                    "lobster",
                    "watch",
                    "flamingo",
                ],
                source="caltech101",
                required_folder=DATA_DIR / "caltech-101" / "101_ObjectCategories",
            ),
            DemoDataset(
                id="images_l",
                label="Caltech-101 Diverse Objects (L)",
                description=(
                    "Photographs across 15 diverse categories including animals,"
                    " instruments, vehicles, and objects from Caltech-101."
                ),
                categories=[
                    "scorpion",
                    "stop_sign",
                    "chandelier",
                    "rhino",
                    "rooster",
                    "soccer_ball",
                    "yin_yang",
                    "leopards",
                    "hawksbill",
                    "revolver",
                    "schooner",
                    "ibis",
                    "trilobite",
                    "ceiling_fan",
                    "dalmatian",
                ],
                source="caltech101",
                required_folder=DATA_DIR / "caltech-101" / "101_ObjectCategories",
            ),
        ]

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    @property
    def description_wrappers(self) -> list[str]:
        return [
            "a photo of {text}",
            "a photograph of {text}",
            "an image of {text}",
            "{text}",
            "a picture of {text}",
        ]

    def load_models(self) -> None:
        if self._model is not None:
            return
        import gc

        gc.collect()
        cache_dir = str(MODELS_CACHE_DIR)
        self._on_progress("loading", "Loading image embedder (CLIP model)...", 0, 0)
        # Older CLIP checkpoints include position_ids buffers that newer transformers
        # versions compute on-the-fly.  Tell the loader to silently ignore them.
        CLIPModel._keys_to_ignore_on_load_unexpected = [r".*position_ids.*"]
        self._model = CLIPModel.from_pretrained(CLIP_MODEL_ID, low_cpu_mem_usage=True, cache_dir=cache_dir)
        self._processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID, cache_dir=cache_dir, use_fast=True)

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
                embedding = _extract_tensor(outputs).detach().cpu().numpy()
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
                text_vec = _extract_tensor(self._model.get_text_features(**inputs)).detach().cpu().numpy()[0]
            return text_vec
        except Exception as e:
            print(f"Error embedding text query for image: {e}")
            return None

    # internal helper used by loader.py's get_clip_model() bridge
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
            img = Image.open(file_path)
            width, height = img.width, img.height
        except Exception:
            width, height = None, None
        return {
            "clip_bytes": clip_bytes,
            "duration": 0,
            "width": width,
            "height": height,
        }

    # ------------------------------------------------------------------
    # HTTP serving
    # ------------------------------------------------------------------

    def clip_response(self, clip: dict) -> MediaResponse:
        filename = clip.get("filename", "")
        ext = Path(filename).suffix.lower() if filename else ".jpg"
        mimetype = _IMAGE_MIME_TYPES.get(ext, "image/jpeg")
        return MediaResponse(
            data=clip["clip_bytes"],
            mimetype=mimetype,
            download_name=f"clip_{clip['id']}{ext}",
        )
