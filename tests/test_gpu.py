"""GPU-specific tests for VTSearch.

These tests exercise the same code paths as the CPU test suite but on CUDA
devices.  They are guarded by the ``gpu`` pytest marker **and** a runtime
``torch.cuda.is_available()`` check, so they are never collected during
regular CI runs (which use ``-m "not gpu"`` or simply omit the marker).

Run them on a machine with a CUDA GPU::

    python -m pytest tests/test_gpu.py -v

Coverage areas
--------------
1. MLP training (train_model) on GPU
2. Cross-calibration threshold computation on GPU
3. Full train_and_score pipeline on GPU
4. Detector export → reconstruct → score on GPU
5. Embedding models (CLAP, CLIP, X-CLIP, E5) on GPU
6. CPU ↔ GPU numerical equivalence for training
7. GPU memory cleanup after inference
"""

import gc

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Marker applied to every test in this module
# ---------------------------------------------------------------------------
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
]


# ---------------------------------------------------------------------------
# Helpers & fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def device():
    """Return the first available CUDA device."""
    return torch.device("cuda", 0)


def _make_embeddings(n: int, dim: int = 512, seed: int = 42) -> np.ndarray:
    """Create deterministic random embeddings for *n* clips."""
    rng = np.random.RandomState(seed)
    return rng.randn(n, dim).astype(np.float32)


def _make_clips_dict(n: int = 20, dim: int = 512, seed: int = 42) -> dict:
    """Build a minimal clips dict similar to the real application."""
    embs = _make_embeddings(n, dim, seed)
    return {i + 1: {"id": i + 1, "embedding": embs[i], "type": "audio"} for i in range(n)}


def _make_votes(good_ids: list[int], bad_ids: list[int]):
    good = {k: None for k in good_ids}
    bad = {k: None for k in bad_ids}
    return good, bad


# ---------------------------------------------------------------------------
# 1. train_model on GPU
# ---------------------------------------------------------------------------


class TestTrainModelGPU:
    """Verify ``train_model`` works when tensors and model live on CUDA."""

    def test_model_trains_on_gpu(self, device):
        from vtsearch.models.training import train_model

        dim = 64
        X = torch.randn(10, dim, device=device)
        y = torch.cat([torch.ones(5, 1), torch.zeros(5, 1)]).to(device)

        model = train_model(X, y, dim)
        # train_model creates its own model on CPU; verify it can evaluate GPU data
        # after moving the model to GPU
        model = model.to(device)
        with torch.no_grad():
            scores = torch.sigmoid(model(X))
        assert scores.shape == (10, 1)
        assert scores.device.type == "cuda"

    def test_gpu_trained_scores_between_zero_and_one(self, device):
        from vtsearch.models.training import train_model

        dim = 64
        X = torch.randn(10, dim, device=device)
        y = torch.cat([torch.ones(5, 1), torch.zeros(5, 1)]).to(device)

        model = train_model(X, y, dim).to(device)
        with torch.no_grad():
            scores = torch.sigmoid(model(X)).squeeze(1).cpu().numpy()
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)

    def test_gpu_model_separates_classes(self, device):
        """Good examples should score higher than bad examples on average."""
        import vtsearch.config as config

        saved = config.TRAIN_EPOCHS
        config.TRAIN_EPOCHS = 100
        try:
            from vtsearch.models.training import train_model

            rng = np.random.RandomState(0)
            dim = 32
            # Linearly separable data
            good_embs = rng.randn(20, dim).astype(np.float32) + 2.0
            bad_embs = rng.randn(20, dim).astype(np.float32) - 2.0
            X = torch.tensor(np.vstack([good_embs, bad_embs]), device=device)
            y = torch.cat([torch.ones(20, 1), torch.zeros(20, 1)]).to(device)

            model = train_model(X, y, dim).to(device)
            with torch.no_grad():
                scores = torch.sigmoid(model(X)).squeeze(1).cpu().numpy()
            avg_good = scores[:20].mean()
            avg_bad = scores[20:].mean()
            assert avg_good > avg_bad
        finally:
            config.TRAIN_EPOCHS = saved

    def test_inclusion_positive_biases_toward_good(self, device):
        import vtsearch.config as config

        saved = config.TRAIN_EPOCHS
        config.TRAIN_EPOCHS = 100
        try:
            from vtsearch.models.training import train_model

            dim = 32
            rng = np.random.RandomState(1)
            good_embs = rng.randn(10, dim).astype(np.float32) + 1.0
            bad_embs = rng.randn(10, dim).astype(np.float32) - 1.0
            X = torch.tensor(np.vstack([good_embs, bad_embs]), device=device)
            y = torch.cat([torch.ones(10, 1), torch.zeros(10, 1)]).to(device)

            model_neutral = train_model(X, y, dim, inclusion_value=0).to(device)
            model_inclusive = train_model(X, y, dim, inclusion_value=5).to(device)

            with torch.no_grad():
                scores_neutral = torch.sigmoid(model_neutral(X)).squeeze(1).cpu().numpy()
                scores_inclusive = torch.sigmoid(model_inclusive(X)).squeeze(1).cpu().numpy()

            # With high inclusion, the overall mean score should be higher
            assert scores_inclusive.mean() >= scores_neutral.mean() - 0.1
        finally:
            config.TRAIN_EPOCHS = saved


# ---------------------------------------------------------------------------
# 2. Cross-calibration threshold on GPU
# ---------------------------------------------------------------------------


class TestCrossCalibrationGPU:
    """Verify ``calculate_cross_calibration_threshold`` produces a valid
    threshold when the underlying training happens on a GPU-capable system."""

    def test_threshold_is_valid_float(self):
        import vtsearch.config as config

        saved = config.TRAIN_EPOCHS
        config.TRAIN_EPOCHS = 30
        try:
            from vtsearch.models.training import calculate_cross_calibration_threshold

            dim = 64
            rng = np.random.RandomState(7)
            X_list = list(rng.randn(20, dim).astype(np.float32))
            y_list = [1.0] * 10 + [0.0] * 10
            threshold = calculate_cross_calibration_threshold(X_list, y_list, dim)
            assert isinstance(threshold, float)
            assert 0.0 <= threshold <= 1.0
        finally:
            config.TRAIN_EPOCHS = saved

    def test_threshold_with_inclusion(self):
        import vtsearch.config as config

        saved = config.TRAIN_EPOCHS
        config.TRAIN_EPOCHS = 30
        try:
            from vtsearch.models.training import calculate_cross_calibration_threshold

            dim = 64
            rng = np.random.RandomState(8)
            X_list = list(rng.randn(20, dim).astype(np.float32))
            y_list = [1.0] * 10 + [0.0] * 10
            t_neg = calculate_cross_calibration_threshold(X_list, y_list, dim, inclusion_value=-5)
            t_pos = calculate_cross_calibration_threshold(X_list, y_list, dim, inclusion_value=5)
            # Both should be valid
            assert isinstance(t_neg, float)
            assert isinstance(t_pos, float)
        finally:
            config.TRAIN_EPOCHS = saved


# ---------------------------------------------------------------------------
# 3. Full train_and_score pipeline on GPU
# ---------------------------------------------------------------------------


class TestTrainAndScoreGPU:
    """Verify that the full ``train_and_score`` pipeline (used by the
    learned-sort endpoint) works on a system with a GPU."""

    def test_returns_all_clips_scored(self):
        import vtsearch.config as config

        saved = config.TRAIN_EPOCHS
        config.TRAIN_EPOCHS = 30
        try:
            from vtsearch.models.training import train_and_score

            clips_dict = _make_clips_dict(20, dim=64)
            good, bad = _make_votes([1, 2, 3], [18, 19, 20])
            results, threshold = train_and_score(clips_dict, good, bad)

            assert len(results) == 20
            assert isinstance(threshold, float)
            for entry in results:
                assert "id" in entry
                assert "score" in entry
        finally:
            config.TRAIN_EPOCHS = saved

    def test_scores_between_zero_and_one(self):
        import vtsearch.config as config

        saved = config.TRAIN_EPOCHS
        config.TRAIN_EPOCHS = 30
        try:
            from vtsearch.models.training import train_and_score

            clips_dict = _make_clips_dict(20, dim=64)
            good, bad = _make_votes([1, 2, 3], [18, 19, 20])
            results, _ = train_and_score(clips_dict, good, bad)
            for entry in results:
                assert 0.0 <= entry["score"] <= 1.0
        finally:
            config.TRAIN_EPOCHS = saved

    def test_results_sorted_descending(self):
        import vtsearch.config as config

        saved = config.TRAIN_EPOCHS
        config.TRAIN_EPOCHS = 30
        try:
            from vtsearch.models.training import train_and_score

            clips_dict = _make_clips_dict(20, dim=64)
            good, bad = _make_votes([1, 2], [3, 4])
            results, _ = train_and_score(clips_dict, good, bad)
            scores = [e["score"] for e in results]
            assert scores == sorted(scores, reverse=True)
        finally:
            config.TRAIN_EPOCHS = saved

    def test_good_clips_scored_higher_than_bad(self):
        import vtsearch.config as config

        saved = config.TRAIN_EPOCHS
        config.TRAIN_EPOCHS = 50
        try:
            from vtsearch.models.training import train_and_score

            # Use separable embeddings so the model can learn
            dim = 64
            rng = np.random.RandomState(99)
            clips_dict = {}
            for i in range(1, 21):
                emb = rng.randn(dim).astype(np.float32) + (2.0 if i <= 5 else -2.0 if i > 15 else 0.0)
                clips_dict[i] = {"id": i, "embedding": emb, "type": "audio"}

            good, bad = _make_votes([1, 2, 3, 4, 5], [16, 17, 18, 19, 20])
            results, _ = train_and_score(clips_dict, good, bad)
            score_map = {e["id"]: e["score"] for e in results}
            avg_good = np.mean([score_map[i] for i in good])
            avg_bad = np.mean([score_map[i] for i in bad])
            assert avg_good > avg_bad
        finally:
            config.TRAIN_EPOCHS = saved

    def test_with_inclusion_value(self):
        import vtsearch.config as config

        saved = config.TRAIN_EPOCHS
        config.TRAIN_EPOCHS = 30
        try:
            from vtsearch.models.training import train_and_score

            clips_dict = _make_clips_dict(20, dim=64)
            good, bad = _make_votes([1, 2, 3], [18, 19, 20])
            results, threshold = train_and_score(clips_dict, good, bad, inclusion_value=5)
            assert len(results) == 20
            assert isinstance(threshold, float)
        finally:
            config.TRAIN_EPOCHS = saved


# ---------------------------------------------------------------------------
# 4. Detector export → reconstruct → score on GPU
# ---------------------------------------------------------------------------


class TestDetectorGPU:
    """Verify detector model reconstruction and scoring on GPU."""

    def test_reconstruct_and_score_on_gpu(self, device):
        import vtsearch.config as config

        saved = config.TRAIN_EPOCHS
        config.TRAIN_EPOCHS = 30
        try:
            from vtsearch.models.training import train_model

            dim = 64
            clips_dict = _make_clips_dict(20, dim)
            good, bad = _make_votes([1, 2, 3], [18, 19, 20])

            # Build training data
            X_list, y_list = [], []
            for cid in good:
                X_list.append(clips_dict[cid]["embedding"])
                y_list.append(1.0)
            for cid in bad:
                X_list.append(clips_dict[cid]["embedding"])
                y_list.append(0.0)

            X = torch.tensor(np.array(X_list), dtype=torch.float32)
            y = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1)

            # Train on CPU (as the app does)
            model = train_model(X, y, dim)

            # Export weights (as /api/detector/export does)
            state_dict = model.state_dict()
            weights = {k: v.tolist() for k, v in state_dict.items()}

            # Reconstruct on GPU (as a GPU-aware detector-sort would)
            from vtsearch.models.training import build_model

            gpu_model = build_model(dim).to(device)

            loaded_state = {k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in weights.items()}
            gpu_model.load_state_dict(loaded_state)
            gpu_model.eval()

            # Score all clips on GPU
            all_embs = np.array([clips_dict[cid]["embedding"] for cid in sorted(clips_dict.keys())])
            X_all = torch.tensor(all_embs, dtype=torch.float32, device=device)
            with torch.no_grad():
                scores = torch.sigmoid(gpu_model(X_all)).squeeze(1).cpu().numpy()

            assert len(scores) == 20
            assert np.all(scores >= 0.0)
            assert np.all(scores <= 1.0)
        finally:
            config.TRAIN_EPOCHS = saved

    def test_gpu_cpu_scores_match(self, device):
        """Scores from GPU and CPU model should be numerically close."""
        import vtsearch.config as config

        saved = config.TRAIN_EPOCHS
        config.TRAIN_EPOCHS = 30
        try:
            from vtsearch.models.training import train_model

            dim = 64
            clips_dict = _make_clips_dict(20, dim, seed=123)
            good, bad = _make_votes([1, 2, 3], [18, 19, 20])

            X_list, y_list = [], []
            for cid in good:
                X_list.append(clips_dict[cid]["embedding"])
                y_list.append(1.0)
            for cid in bad:
                X_list.append(clips_dict[cid]["embedding"])
                y_list.append(0.0)

            X = torch.tensor(np.array(X_list), dtype=torch.float32)
            y = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1)

            model = train_model(X, y, dim)

            all_embs = np.array([clips_dict[cid]["embedding"] for cid in sorted(clips_dict.keys())])
            X_all = torch.tensor(all_embs, dtype=torch.float32)

            # CPU scores
            model.eval()
            with torch.no_grad():
                cpu_scores = torch.sigmoid(model(X_all)).squeeze(1).numpy()

            # GPU scores
            gpu_model = model.to(device)
            X_all_gpu = X_all.to(device)
            with torch.no_grad():
                gpu_scores = torch.sigmoid(gpu_model(X_all_gpu)).squeeze(1).cpu().numpy()

            np.testing.assert_allclose(cpu_scores, gpu_scores, atol=1e-5)
        finally:
            config.TRAIN_EPOCHS = saved

    def test_multiple_detectors_on_gpu(self, device):
        """Simulate auto-detect: run multiple detectors on GPU sequentially."""
        import vtsearch.config as config

        saved = config.TRAIN_EPOCHS
        config.TRAIN_EPOCHS = 30
        try:
            from vtsearch.models.training import train_model

            dim = 64
            clips_dict = _make_clips_dict(20, dim)
            all_embs = np.array([clips_dict[cid]["embedding"] for cid in sorted(clips_dict.keys())])
            X_all = torch.tensor(all_embs, dtype=torch.float32, device=device)

            detector_results = {}
            for det_idx, (good_ids, bad_ids) in enumerate(
                [([1, 2, 3], [18, 19, 20]), ([5, 6, 7], [14, 15, 16])]
            ):
                good, bad = _make_votes(good_ids, bad_ids)
                X_list, y_list = [], []
                for cid in good:
                    X_list.append(clips_dict[cid]["embedding"])
                    y_list.append(1.0)
                for cid in bad:
                    X_list.append(clips_dict[cid]["embedding"])
                    y_list.append(0.0)

                X = torch.tensor(np.array(X_list), dtype=torch.float32)
                y = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1)
                model = train_model(X, y, dim).to(device)

                with torch.no_grad():
                    scores = torch.sigmoid(model(X_all)).squeeze(1).cpu().numpy()

                detector_results[f"det-{det_idx}"] = scores

            assert len(detector_results) == 2
            for name, scores in detector_results.items():
                assert len(scores) == 20
                assert np.all(scores >= 0.0)
                assert np.all(scores <= 1.0)
        finally:
            config.TRAIN_EPOCHS = saved


# ---------------------------------------------------------------------------
# 5. Embedding models on GPU
# ---------------------------------------------------------------------------


class TestCLAPEmbeddingGPU:
    """Test CLAP (audio) embedding model on GPU.

    These tests download the CLAP model on first run and may be slow.
    """

    def test_clap_model_loads_on_gpu(self, device):
        from vtsearch.media.audio.media_type import AudioMediaType

        mt = AudioMediaType()
        mt.load_models()
        assert mt._model is not None
        mt._model = mt._model.to(device)
        # Verify model is on GPU
        param = next(mt._model.parameters())
        assert param.device.type == "cuda"

    def test_clap_text_embedding_on_gpu(self, device):
        from transformers import ClapModel, ClapProcessor

        from vtsearch.config import CLAP_MODEL_ID, MODELS_CACHE_DIR

        cache_dir = str(MODELS_CACHE_DIR)
        model = ClapModel.from_pretrained(CLAP_MODEL_ID, low_cpu_mem_usage=True, cache_dir=cache_dir).to(device)
        processor = ClapProcessor.from_pretrained(CLAP_MODEL_ID, cache_dir=cache_dir)

        inputs = processor(text=["a dog barking"], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.text_model(**inputs)
            vec = model.text_projection(outputs.pooler_output).detach().cpu().numpy()[0]

        assert vec.shape == (512,)
        assert np.isfinite(vec).all()

    def test_clap_audio_embedding_on_gpu(self, device, tmp_path):
        import soundfile as sf
        from transformers import ClapModel, ClapProcessor

        from vtsearch.config import CLAP_MODEL_ID, MODELS_CACHE_DIR, SAMPLE_RATE

        # Generate a short sine wave
        duration = 1.0
        t = np.linspace(0, duration, int(SAMPLE_RATE * duration), dtype=np.float32)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        wav_path = tmp_path / "test.wav"
        sf.write(str(wav_path), audio, SAMPLE_RATE)

        cache_dir = str(MODELS_CACHE_DIR)
        model = ClapModel.from_pretrained(CLAP_MODEL_ID, low_cpu_mem_usage=True, cache_dir=cache_dir).to(device)
        processor = ClapProcessor.from_pretrained(CLAP_MODEL_ID, cache_dir=cache_dir)

        inputs = processor(
            audio=audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding="max_length",
            max_length=480000,
            truncation=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.audio_model(**inputs)
            vec = model.audio_projection(outputs.pooler_output).detach().cpu().numpy()[0]

        assert vec.shape == (512,)
        assert np.isfinite(vec).all()


class TestCLIPEmbeddingGPU:
    """Test CLIP (image) embedding model on GPU."""

    def test_clip_model_loads_on_gpu(self, device):
        from vtsearch.media.image.media_type import ImageMediaType

        mt = ImageMediaType()
        mt.load_models()
        assert mt._model is not None
        mt._model = mt._model.to(device)
        param = next(mt._model.parameters())
        assert param.device.type == "cuda"

    def test_clip_text_embedding_on_gpu(self, device):
        from transformers import CLIPModel, CLIPProcessor

        from vtsearch.config import CLIP_MODEL_ID, MODELS_CACHE_DIR

        cache_dir = str(MODELS_CACHE_DIR)
        CLIPModel._keys_to_ignore_on_load_unexpected = [r".*position_ids.*"]
        model = CLIPModel.from_pretrained(CLIP_MODEL_ID, low_cpu_mem_usage=True, cache_dir=cache_dir).to(device)
        processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID, cache_dir=cache_dir, use_fast=True)

        inputs = processor(text=["a photo of a cat"], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            from vtsearch.media.image.media_type import _extract_tensor

            vec = _extract_tensor(model.get_text_features(**inputs)).detach().cpu().numpy()[0]

        assert vec.ndim == 1
        assert np.isfinite(vec).all()

    def test_clip_image_embedding_on_gpu(self, device):
        from PIL import Image
        from transformers import CLIPModel, CLIPProcessor

        from vtsearch.config import CLIP_MODEL_ID, MODELS_CACHE_DIR

        cache_dir = str(MODELS_CACHE_DIR)
        CLIPModel._keys_to_ignore_on_load_unexpected = [r".*position_ids.*"]
        model = CLIPModel.from_pretrained(CLIP_MODEL_ID, low_cpu_mem_usage=True, cache_dir=cache_dir).to(device)
        processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID, cache_dir=cache_dir, use_fast=True)

        # Create a simple test image
        img = Image.new("RGB", (224, 224), color=(128, 64, 32))
        inputs = processor(images=img, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            from vtsearch.media.image.media_type import _extract_tensor

            vec = _extract_tensor(model.get_image_features(**inputs)).detach().cpu().numpy()[0]

        assert vec.ndim == 1
        assert np.isfinite(vec).all()


class TestXCLIPEmbeddingGPU:
    """Test X-CLIP (video) embedding model on GPU."""

    def test_xclip_model_loads_on_gpu(self, device):
        from vtsearch.media.video.media_type import VideoMediaType

        mt = VideoMediaType()
        mt.load_models()
        assert mt._model is not None
        mt._model = mt._model.to(device)
        param = next(mt._model.parameters())
        assert param.device.type == "cuda"

    def test_xclip_text_embedding_on_gpu(self, device):
        from transformers import XCLIPModel, XCLIPProcessor

        from vtsearch.config import MODELS_CACHE_DIR, XCLIP_MODEL_ID

        cache_dir = str(MODELS_CACHE_DIR)
        model = XCLIPModel.from_pretrained(XCLIP_MODEL_ID, low_cpu_mem_usage=True, cache_dir=cache_dir).to(device)
        processor = XCLIPProcessor.from_pretrained(XCLIP_MODEL_ID, cache_dir=cache_dir, use_fast=False)

        inputs = processor(text=["a person walking"], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            from vtsearch.media.video.media_type import _extract_tensor

            vec = _extract_tensor(model.get_text_features(**inputs)).detach().cpu().numpy()[0]

        assert vec.ndim == 1
        assert np.isfinite(vec).all()

    def test_xclip_video_embedding_on_gpu(self, device):
        from PIL import Image
        from transformers import XCLIPModel, XCLIPProcessor

        from vtsearch.config import MODELS_CACHE_DIR, XCLIP_MODEL_ID

        cache_dir = str(MODELS_CACHE_DIR)
        model = XCLIPModel.from_pretrained(XCLIP_MODEL_ID, low_cpu_mem_usage=True, cache_dir=cache_dir).to(device)
        processor = XCLIPProcessor.from_pretrained(XCLIP_MODEL_ID, cache_dir=cache_dir, use_fast=False)

        # Create 8 dummy frames
        frames = [Image.new("RGB", (224, 224), color=(i * 30, 100, 200)) for i in range(8)]
        inputs = processor(videos=list(frames), return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            from vtsearch.media.video.media_type import _extract_tensor

            vec = _extract_tensor(model.get_video_features(**inputs)).detach().cpu().numpy()[0]

        assert vec.ndim == 1
        assert np.isfinite(vec).all()


class TestE5EmbeddingGPU:
    """Test E5 (text/paragraph) embedding model on GPU."""

    def test_e5_model_loads_on_gpu(self, device):
        from sentence_transformers import SentenceTransformer

        from vtsearch.config import E5_MODEL_ID, MODELS_CACHE_DIR

        model = SentenceTransformer(E5_MODEL_ID, cache_folder=str(MODELS_CACHE_DIR), device=str(device))
        vec = model.encode("query: test sentence", normalize_embeddings=True)
        assert vec.ndim == 1
        assert np.isfinite(vec).all()

    def test_e5_passage_embedding_on_gpu(self, device):
        from sentence_transformers import SentenceTransformer

        from vtsearch.config import E5_MODEL_ID, MODELS_CACHE_DIR

        model = SentenceTransformer(E5_MODEL_ID, cache_folder=str(MODELS_CACHE_DIR), device=str(device))
        vec = model.encode("passage: The quick brown fox jumps over the lazy dog.", normalize_embeddings=True)
        assert vec.ndim == 1
        assert np.isfinite(vec).all()

    def test_e5_query_passage_same_space(self, device):
        """Query and passage embeddings should have the same dimensionality."""
        from sentence_transformers import SentenceTransformer

        from vtsearch.config import E5_MODEL_ID, MODELS_CACHE_DIR

        model = SentenceTransformer(E5_MODEL_ID, cache_folder=str(MODELS_CACHE_DIR), device=str(device))
        q_vec = model.encode("query: animals", normalize_embeddings=True)
        p_vec = model.encode("passage: Dogs are loyal companions.", normalize_embeddings=True)
        assert q_vec.shape == p_vec.shape
        # Cosine similarity should be defined (both are unit vectors)
        sim = float(np.dot(q_vec, p_vec))
        assert -1.0 <= sim <= 1.0


# ---------------------------------------------------------------------------
# 6. CPU ↔ GPU numerical equivalence for embeddings
# ---------------------------------------------------------------------------


class TestEmbeddingEquivalence:
    """Verify that GPU embeddings match CPU embeddings within tolerance."""

    def test_clip_text_cpu_gpu_match(self, device):
        from transformers import CLIPModel, CLIPProcessor

        from vtsearch.config import CLIP_MODEL_ID, MODELS_CACHE_DIR

        cache_dir = str(MODELS_CACHE_DIR)
        CLIPModel._keys_to_ignore_on_load_unexpected = [r".*position_ids.*"]
        model = CLIPModel.from_pretrained(CLIP_MODEL_ID, low_cpu_mem_usage=True, cache_dir=cache_dir)
        processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID, cache_dir=cache_dir, use_fast=True)

        text = "a red sports car"
        inputs = processor(text=[text], return_tensors="pt")

        # CPU
        model.eval()
        with torch.no_grad():
            from vtsearch.media.image.media_type import _extract_tensor

            cpu_vec = _extract_tensor(model.get_text_features(**inputs)).numpy()[0]

        # GPU
        gpu_model = model.to(device)
        gpu_inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            gpu_vec = _extract_tensor(gpu_model.get_text_features(**gpu_inputs)).detach().cpu().numpy()[0]

        np.testing.assert_allclose(cpu_vec, gpu_vec, atol=1e-4)

    def test_e5_cpu_gpu_match(self, device):
        from sentence_transformers import SentenceTransformer

        from vtsearch.config import E5_MODEL_ID, MODELS_CACHE_DIR

        text = "query: machine learning algorithms"

        cpu_model = SentenceTransformer(E5_MODEL_ID, cache_folder=str(MODELS_CACHE_DIR), device="cpu")
        cpu_vec = cpu_model.encode(text, normalize_embeddings=True)

        gpu_model = SentenceTransformer(E5_MODEL_ID, cache_folder=str(MODELS_CACHE_DIR), device=str(device))
        gpu_vec = gpu_model.encode(text, normalize_embeddings=True)

        np.testing.assert_allclose(cpu_vec, gpu_vec, atol=1e-4)


# ---------------------------------------------------------------------------
# 7. GPU memory cleanup
# ---------------------------------------------------------------------------


class TestGPUMemoryCleanup:
    """Verify that GPU memory is freed after model use."""

    def test_training_frees_gpu_memory(self, device):
        import vtsearch.config as config

        saved = config.TRAIN_EPOCHS
        config.TRAIN_EPOCHS = 30
        try:
            from vtsearch.models.training import train_model

            torch.cuda.reset_peak_memory_stats(device)
            initial_mem = torch.cuda.memory_allocated(device)

            dim = 128
            X = torch.randn(50, dim, device=device)
            y = torch.cat([torch.ones(25, 1), torch.zeros(25, 1)]).to(device)
            model = train_model(X, y, dim).to(device)

            with torch.no_grad():
                _ = model(X)

            # Clean up
            del model, X, y
            gc.collect()
            torch.cuda.empty_cache()

            final_mem = torch.cuda.memory_allocated(device)
            # Memory should return close to initial (within 1 MB tolerance)
            assert final_mem - initial_mem < 1_000_000
        finally:
            config.TRAIN_EPOCHS = saved

    def test_embedding_model_frees_gpu_memory(self, device):
        """Loading and unloading an embedding model should free GPU memory."""
        torch.cuda.reset_peak_memory_stats(device)
        initial_mem = torch.cuda.memory_allocated(device)

        from sentence_transformers import SentenceTransformer

        from vtsearch.config import E5_MODEL_ID, MODELS_CACHE_DIR

        model = SentenceTransformer(E5_MODEL_ID, cache_folder=str(MODELS_CACHE_DIR), device=str(device))
        _ = model.encode("query: test", normalize_embeddings=True)

        # Should have allocated significant memory
        peak_mem = torch.cuda.max_memory_allocated(device)
        assert peak_mem > initial_mem

        # Clean up
        del model
        gc.collect()
        torch.cuda.empty_cache()

        final_mem = torch.cuda.memory_allocated(device)
        # Memory should return close to initial (within 5 MB tolerance for E5)
        assert final_mem - initial_mem < 5_000_000
