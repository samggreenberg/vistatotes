"""Tests for the Enrich Sort Descriptions feature.

Covers:
- description_wrappers property on each media type
- embed_text_enriched method (base class logic)
- enrich_descriptions setting (get/set/persist)
- embed_text_query with enrich=True
- eval runner with enrich=True
- settings API route for enrich_descriptions
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from vtsearch.eval.config import EvalQuery
from vtsearch.eval.runner import eval_text_sort
from vtsearch.media.audio.media_type import AudioMediaType
from vtsearch.media.base import MediaType
from vtsearch.media.image.media_type import ImageMediaType
from vtsearch.media.text.media_type import TextMediaType
from vtsearch.media.video.media_type import VideoMediaType
from vtsearch.models.embeddings import embed_text_query


# =====================================================================
# description_wrappers property
# =====================================================================


class TestDescriptionWrappers:
    def test_audio_has_wrappers(self):
        mt = AudioMediaType()
        wrappers = mt.description_wrappers
        assert len(wrappers) >= 3
        for w in wrappers:
            assert "{text}" in w

    def test_image_has_wrappers(self):
        mt = ImageMediaType()
        wrappers = mt.description_wrappers
        assert len(wrappers) >= 3
        for w in wrappers:
            assert "{text}" in w

    def test_text_has_wrappers(self):
        mt = TextMediaType()
        wrappers = mt.description_wrappers
        assert len(wrappers) >= 3
        for w in wrappers:
            assert "{text}" in w

    def test_video_has_wrappers(self):
        mt = VideoMediaType()
        wrappers = mt.description_wrappers
        assert len(wrappers) >= 3
        for w in wrappers:
            assert "{text}" in w

    def test_wrappers_include_bare_text(self):
        """Each media type should include a plain '{text}' wrapper."""
        for mt_cls in (AudioMediaType, ImageMediaType, TextMediaType, VideoMediaType):
            mt = mt_cls()
            assert "{text}" in mt.description_wrappers, f"{mt_cls.__name__} missing bare '{{text}}' wrapper"

    def test_wrappers_format_correctly(self):
        """All wrappers should format without errors."""
        for mt_cls in (AudioMediaType, ImageMediaType, TextMediaType, VideoMediaType):
            mt = mt_cls()
            for wrapper in mt.description_wrappers:
                result = wrapper.format(text="test query")
                assert "test query" in result


# =====================================================================
# embed_text_enriched (base class logic)
# =====================================================================


class TestEmbedTextEnriched:
    def _make_mock_media_type(self, wrappers, embed_fn):
        """Create a minimal concrete MediaType subclass for testing."""

        class MockMediaType(MediaType):
            @property
            def type_id(self):
                return "mock"

            @property
            def name(self):
                return "Mock"

            @property
            def icon(self):
                return "?"

            @property
            def file_extensions(self):
                return []

            @property
            def loops(self):
                return False

            @property
            def demo_datasets(self):
                return []

            @property
            def description_wrappers(self):
                return wrappers

            def load_models(self):
                pass

            def embed_media(self, file_path):
                return None

            def embed_text(self, text):
                return embed_fn(text)

            def load_clip_data(self, file_path):
                return {"duration": 0}

            def clip_response(self, clip):
                from vtsearch.media.base import MediaResponse

                return MediaResponse(data=b"", mimetype="text/plain")

        return MockMediaType()

    def test_enriched_averages_wrapper_embeddings(self):
        """embed_text_enriched should average embeddings across wrappers."""
        call_log = []

        def mock_embed(text):
            call_log.append(text)
            # Return different vectors for different wrapped texts
            if text.startswith("a photo"):
                return np.array([1.0, 0.0, 0.0])
            elif text.startswith("an image"):
                return np.array([0.0, 1.0, 0.0])
            else:
                return np.array([0.0, 0.0, 1.0])

        mt = self._make_mock_media_type(
            wrappers=["a photo of {text}", "an image of {text}", "{text}"],
            embed_fn=mock_embed,
        )

        result = mt.embed_text_enriched("cats")
        assert result is not None
        assert len(call_log) == 3
        assert "a photo of cats" in call_log
        assert "an image of cats" in call_log
        assert "cats" in call_log

        # Result should be the L2-normalised mean of the three vectors
        expected = np.mean([[1, 0, 0], [0, 1, 0], [0, 0, 1]], axis=0)
        expected = expected / np.linalg.norm(expected)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_enriched_falls_back_when_no_wrappers(self):
        """If no wrappers, embed_text_enriched should fall back to embed_text."""

        def mock_embed(text):
            return np.array([1.0, 2.0, 3.0])

        mt = self._make_mock_media_type(wrappers=[], embed_fn=mock_embed)
        result = mt.embed_text_enriched("test")
        np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0]))

    def test_enriched_falls_back_when_all_fail(self):
        """If all wrapper embeddings fail, fall back to plain embed_text."""
        calls = {"count": 0}

        def mock_embed(text):
            calls["count"] += 1
            if calls["count"] <= 2:
                return None  # Fail for wrapped texts
            return np.array([1.0, 0.0])  # Succeed for plain fallback

        mt = self._make_mock_media_type(
            wrappers=["wrapper1 {text}", "wrapper2 {text}"],
            embed_fn=mock_embed,
        )
        result = mt.embed_text_enriched("test")
        assert result is not None
        np.testing.assert_array_equal(result, np.array([1.0, 0.0]))

    def test_enriched_skips_failed_wrappers(self):
        """Wrappers that fail to embed should be skipped, not crash."""

        def mock_embed(text):
            if "bad" in text:
                return None
            return np.array([1.0, 0.0])

        mt = self._make_mock_media_type(
            wrappers=["good {text}", "bad {text}"],
            embed_fn=mock_embed,
        )
        result = mt.embed_text_enriched("query")
        assert result is not None
        # Only one embedding succeeded, so result is just that one (normalised)
        expected = np.array([1.0, 0.0])
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_enriched_result_is_normalised(self):
        """The enriched embedding should be L2-normalised."""

        def mock_embed(text):
            return np.array([3.0, 4.0])

        mt = self._make_mock_media_type(
            wrappers=["w1 {text}", "w2 {text}"],
            embed_fn=mock_embed,
        )
        result = mt.embed_text_enriched("test")
        assert abs(np.linalg.norm(result) - 1.0) < 1e-6


# =====================================================================
# embed_text_query with enrich flag
# =====================================================================


class TestEmbedTextQueryEnrich:
    def test_enrich_false_calls_embed_text(self):
        """enrich=False should call embed_text, not embed_text_enriched."""
        mock_vec = np.array([1.0, 0.0])

        class FakeMT:
            def embed_text(self, text):
                return mock_vec

            def embed_text_enriched(self, text):
                raise AssertionError("Should not be called")

        with patch("vtsearch.media.get", return_value=FakeMT()):
            result = embed_text_query("test", "audio", enrich=False)
        np.testing.assert_array_equal(result, mock_vec)

    def test_enrich_true_calls_embed_text_enriched(self):
        """enrich=True should call embed_text_enriched."""
        mock_vec = np.array([0.0, 1.0])

        class FakeMT:
            def embed_text(self, text):
                raise AssertionError("Should not be called")

            def embed_text_enriched(self, text):
                return mock_vec

        with patch("vtsearch.media.get", return_value=FakeMT()):
            result = embed_text_query("test", "audio", enrich=True)
        np.testing.assert_array_equal(result, mock_vec)


# =====================================================================
# enrich_descriptions setting
# =====================================================================


class TestEnrichDescriptionsSetting:
    @pytest.fixture(autouse=True)
    def isolated_settings(self, tmp_path, monkeypatch):
        from vtsearch import settings as settings_mod

        test_settings_path = tmp_path / "settings.json"
        monkeypatch.setattr(settings_mod, "SETTINGS_PATH", test_settings_path)
        settings_mod.reset()
        yield test_settings_path
        settings_mod.reset()

    def test_default_is_false(self):
        from vtsearch import settings as settings_mod

        assert settings_mod.get_enrich_descriptions() is False

    def test_set_and_get(self):
        from vtsearch import settings as settings_mod

        settings_mod.set_enrich_descriptions(True)
        assert settings_mod.get_enrich_descriptions() is True

        settings_mod.set_enrich_descriptions(False)
        assert settings_mod.get_enrich_descriptions() is False

    def test_persists_to_disk(self, isolated_settings):
        import json

        from vtsearch import settings as settings_mod

        settings_mod.set_enrich_descriptions(True)
        raw = json.loads(isolated_settings.read_text())
        assert raw["enrich_descriptions"] is True

    def test_in_get_all(self):
        from vtsearch import settings as settings_mod

        data = settings_mod.get_all()
        assert "enrich_descriptions" in data
        assert data["enrich_descriptions"] is False


# =====================================================================
# Settings API route
# =====================================================================


class TestEnrichDescriptionsAPI:
    @pytest.fixture(autouse=True)
    def isolated_settings(self, tmp_path, monkeypatch):
        from vtsearch import settings as settings_mod

        test_settings_path = tmp_path / "settings.json"
        monkeypatch.setattr(settings_mod, "SETTINGS_PATH", test_settings_path)
        settings_mod.reset()
        yield
        settings_mod.reset()

    @pytest.fixture
    def client(self):
        import app as app_module

        app_module.app.config["TESTING"] = True
        with app_module.app.test_client() as c:
            yield c

    def test_get_includes_enrich_descriptions(self, client):
        res = client.get("/api/settings")
        assert res.status_code == 200
        data = res.get_json()
        assert "enrich_descriptions" in data
        assert data["enrich_descriptions"] is False

    def test_put_enrich_descriptions(self, client):
        res = client.put("/api/settings", json={"enrich_descriptions": True})
        assert res.status_code == 200
        data = res.get_json()
        assert data["enrich_descriptions"] is True

        # Verify it persisted
        res2 = client.get("/api/settings")
        assert res2.get_json()["enrich_descriptions"] is True


# =====================================================================
# Eval runner with enrich flag
# =====================================================================


class TestEvalTextSortEnrich:
    def _make_synthetic_clips(self):
        rng = np.random.RandomState(0)
        clips = {}
        clip_id = 1
        cat_dir = np.zeros(16)
        cat_dir[0] = 1.0
        for _ in range(10):
            emb = cat_dir + rng.normal(0, 0.05, 16)
            emb /= np.linalg.norm(emb)
            clips[clip_id] = {"id": clip_id, "embedding": emb, "category": "cat", "type": "image"}
            clip_id += 1
        dog_dir = np.zeros(16)
        dog_dir[1] = 1.0
        for _ in range(10):
            emb = dog_dir + rng.normal(0, 0.05, 16)
            emb /= np.linalg.norm(emb)
            clips[clip_id] = {"id": clip_id, "embedding": emb, "category": "dog", "type": "image"}
            clip_id += 1
        return clips, cat_dir, dog_dir

    def test_eval_text_sort_with_enrich(self):
        """eval_text_sort should pass enrich to embed_text_query."""
        clips, cat_dir, dog_dir = self._make_synthetic_clips()
        queries = [EvalQuery("a cat", "cat")]

        call_kwargs = []

        def mock_embed(text, media_type, enrich=False):
            call_kwargs.append({"enrich": enrich})
            if "cat" in text:
                return cat_dir.copy()
            return dog_dir.copy()

        with patch("vtsearch.models.embeddings.embed_text_query", side_effect=mock_embed):
            results = eval_text_sort(clips, queries, "image", k_values=[5], enrich=True)

        assert len(results) == 1
        assert all(kw["enrich"] is True for kw in call_kwargs)

    def test_eval_text_sort_without_enrich(self):
        """eval_text_sort with enrich=False should pass enrich=False."""
        clips, cat_dir, dog_dir = self._make_synthetic_clips()
        queries = [EvalQuery("a cat", "cat")]

        call_kwargs = []

        def mock_embed(text, media_type, enrich=False):
            call_kwargs.append({"enrich": enrich})
            return cat_dir.copy()

        with patch("vtsearch.models.embeddings.embed_text_query", side_effect=mock_embed):
            eval_text_sort(clips, queries, "image", k_values=[5], enrich=False)

        assert all(kw["enrich"] is False for kw in call_kwargs)
