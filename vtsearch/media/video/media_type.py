"""Video media type — X-CLIP embeddings, MP4/AVI/MOV/WEBM/MKV files."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from transformers import XCLIPModel, XCLIPProcessor

from config import MODELS_CACHE_DIR, VIDEO_DIR, XCLIP_MODEL_ID
from vtsearch.media.base import DemoDataset, MediaResponse, MediaType, ProgressCallback, _noop_progress


def _extract_tensor(output: object) -> torch.Tensor:
    """Extract a plain tensor from model output.

    Depending on the transformers version, get_video_features() / get_text_features()
    may return either a raw tensor or a BaseModelOutputWithPooling dataclass.
    This helper handles both cases.
    """
    if isinstance(output, torch.Tensor):
        return output
    for attr in ("video_embeds", "text_embeds", "pooler_output"):
        val = getattr(output, attr, None)
        if isinstance(val, torch.Tensor):
            return val
    # Final fallback: treat as tuple-like and return first element
    return output[0]  # type: ignore[index]


_VIDEO_MIME_TYPES: dict[str, str] = {
    ".webm": "video/webm",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
}


class VideoMediaType(MediaType):
    """Handles video clips using the X-CLIP model (microsoft/xclip-base-patch32).

    * Embeds videos by sampling 8 evenly-spaced frames and running them through
      X-CLIP's video encoder.
    * Embeds text queries via X-CLIP's text encoder (same 768-dim space).
    * Serves clips with MIME types inferred from the file extension.
    """

    def __init__(self) -> None:
        self._model: Optional[XCLIPModel] = None
        self._processor: Optional[XCLIPProcessor] = None
        self._on_progress: ProgressCallback = _noop_progress

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def type_id(self) -> str:
        return "video"

    @property
    def name(self) -> str:
        return "Video"

    @property
    def icon(self) -> str:
        return "🎬"

    # ------------------------------------------------------------------
    # File import
    # ------------------------------------------------------------------

    @property
    def file_extensions(self) -> list:
        return ["*.mp4", "*.avi", "*.mov", "*.webm", "*.mkv"]

    @property
    def folder_import_name(self) -> str:
        return "videos"

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
                id="activities_video",
                label="Personal Activities",
                description=(
                    "Short clips of everyday personal activities like grooming,"
                    " playing instruments, and yo-yo from UCF-101."
                ),
                categories=[
                    "ApplyEyeMakeup",
                    "ApplyLipstick",
                    "BrushingTeeth",
                    "Drumming",
                    "YoYo",
                ],
                source="ucf101",
                required_folder=VIDEO_DIR / "ucf101",
            ),
            DemoDataset(
                id="sports_video",
                label="Sports & Exercise",
                description=(
                    "Short clips of physical activities including cliff diving,"
                    " jump rope, push-ups, and tai chi from UCF-101."
                ),
                categories=[
                    "CliffDiving",
                    "HandstandWalking",
                    "JumpRope",
                    "PushUps",
                    "TaiChi",
                ],
                source="ucf101",
                required_folder=VIDEO_DIR / "ucf101",
            ),
        ]

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    @property
    def description_wrappers(self) -> list[str]:
        return [
            "a video of {text}",
            "a clip showing {text}",
            "{text}",
            "footage of {text}",
            "a video clip of {text}",
        ]

    def load_models(self) -> None:
        if self._model is not None:
            return
        import gc

        gc.collect()
        cache_dir = str(MODELS_CACHE_DIR)
        self._on_progress("loading", "Loading video embedder (X-CLIP model)...", 0, 0)
        self._model = XCLIPModel.from_pretrained(XCLIP_MODEL_ID, low_cpu_mem_usage=True, cache_dir=cache_dir)
        self._processor = XCLIPProcessor.from_pretrained(XCLIP_MODEL_ID, cache_dir=cache_dir, use_fast=False)

    def embed_media(self, file_path: Path) -> Optional[np.ndarray]:
        if self._model is None:
            self.load_models()
        if self._model is None or self._processor is None:
            return None
        try:
            import cv2  # noqa: PLC0415  (lazy import — cv2 is optional)

            cap = cv2.VideoCapture(str(file_path))
            if not cap.isOpened():
                print(f"Error opening video {file_path}")
                return None

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            num_frames = min(8, max(1, frame_count))
            indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)

            frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(frame))
            cap.release()

            if not frames:
                print(f"Error: could not extract frames from {file_path}")
                return None

            inputs = self._processor(videos=list(frames), return_tensors="pt")
            with torch.no_grad():
                outputs = self._model.get_video_features(**inputs)
                embedding = _extract_tensor(outputs).detach().cpu().numpy()
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
                text_vec = _extract_tensor(self._model.get_text_features(**inputs)).detach().cpu().numpy()[0]
            return text_vec
        except Exception as e:
            print(f"Error embedding text query for video: {e}")
            return None

    # internal helper used by loader.py's get_xclip_model() bridge
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
            import cv2  # noqa: PLC0415

            cap = cv2.VideoCapture(str(file_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = frame_count / fps if fps > 0 else 0.0
            cap.release()
        except Exception:
            duration = 0.0
        return {"clip_bytes": clip_bytes, "duration": duration}

    # ------------------------------------------------------------------
    # HTTP serving
    # ------------------------------------------------------------------

    def clip_response(self, clip: dict) -> MediaResponse:
        filename = clip.get("filename", "")
        ext = Path(filename).suffix.lower() if filename else ".mp4"
        mimetype = _VIDEO_MIME_TYPES.get(ext, "video/mp4")
        return MediaResponse(
            data=clip["clip_bytes"],
            mimetype=mimetype,
            download_name=f"clip_{clip['id']}{ext}",
        )
