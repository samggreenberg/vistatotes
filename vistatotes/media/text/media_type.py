"""Text (paragraph) media type — E5-base-v2 embeddings, TXT/MD files."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from flask import Response, jsonify
from sentence_transformers import SentenceTransformer

from config import E5_MODEL_ID, MODELS_CACHE_DIR
from vistatotes.media.base import DemoDataset, MediaType


class TextMediaType(MediaType):
    """Handles plain-text paragraphs using the E5-base-v2 model.

    * Embeds text files with the ``"passage: "`` prefix required by E5's
      asymmetric retrieval design (768-dim, L2-normalised).
    * Embeds text queries with the ``"query: "`` prefix so they land in the
      same space.
    * Serves clips as JSON objects containing the text content and word/
      character statistics (no binary bytes).
    """

    def __init__(self) -> None:
        self._model: Optional[SentenceTransformer] = None

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def type_id(self) -> str:
        return "paragraph"

    @property
    def name(self) -> str:
        return "Text"

    @property
    def icon(self) -> str:
        return "📄"

    # ------------------------------------------------------------------
    # File import
    # ------------------------------------------------------------------

    @property
    def file_extensions(self) -> list:
        return ["*.txt", "*.md"]

    @property
    def folder_import_name(self) -> str:
        return "paragraphs"

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
                id="world_news",
                label="World & Business News",
                description=(
                    "Paragraphs drawn from international news and business articles"
                    " in the 20 Newsgroups collection."
                ),
                categories=["world", "business"],
                source="ag_news_sample",
            ),
            DemoDataset(
                id="sports_science_news",
                label="Sports & Science News",
                description=(
                    "Paragraphs drawn from sports coverage and science journalism"
                    " in the 20 Newsgroups collection."
                ),
                categories=["sports", "science"],
                source="ag_news_sample",
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
        print("DEBUG: Loading E5-base-v2 model for Text...", flush=True)
        self._model = SentenceTransformer(E5_MODEL_ID, cache_folder=cache_dir)
        print("DEBUG: E5-base-v2 model loaded.", flush=True)

    def embed_media(self, file_path: Path) -> Optional[np.ndarray]:
        if self._model is None:
            self.load_models()
        if self._model is None:
            return None
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text_content = f.read().strip()
            if not text_content:
                print(f"Warning: empty text file {file_path}")
                return None
            return self._model.encode(
                f"passage: {text_content}", normalize_embeddings=True
            )
        except Exception as e:
            print(f"Error embedding {file_path}: {e}")
            return None

    def embed_text_passage(self, text: str) -> Optional[np.ndarray]:
        """Embed *text* as a passage (used when loading demo datasets in-memory)."""
        if self._model is None:
            self.load_models()
        if self._model is None:
            return None
        try:
            return self._model.encode(f"passage: {text}", normalize_embeddings=True)
        except Exception as e:
            print(f"Error embedding passage: {e}")
            return None

    def embed_text(self, text: str) -> Optional[np.ndarray]:
        if self._model is None:
            self.load_models()
        if self._model is None:
            return None
        try:
            return self._model.encode(f"query: {text}", normalize_embeddings=True)
        except Exception as e:
            print(f"Error embedding text query for text: {e}")
            return None

    # internal helper used by loader.py's get_e5_model() bridge
    def _get_model(self) -> Optional[SentenceTransformer]:
        return self._model

    # ------------------------------------------------------------------
    # Clip data
    # ------------------------------------------------------------------

    def load_clip_data(self, file_path: Path) -> dict:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text_content = f.read().strip()
        except Exception:
            text_content = ""
        return {
            "text_content": text_content,
            "duration": 0,
            "word_count": len(text_content.split()),
            "character_count": len(text_content),
        }

    # ------------------------------------------------------------------
    # HTTP serving
    # ------------------------------------------------------------------

    def clip_response(self, clip: dict) -> Response:
        return jsonify(
            {
                "content": clip.get("text_content", ""),
                "word_count": clip.get("word_count", 0),
                "character_count": clip.get("character_count", 0),
            }
        )
