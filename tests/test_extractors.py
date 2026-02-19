import io
from unittest.mock import MagicMock

import pytest
from PIL import Image

from vistatotes.media.base import Extractor


# ---------------------------------------------------------------------------
# A minimal concrete Extractor for unit-testing the ABC
# ---------------------------------------------------------------------------


class StubExtractor(Extractor):
    """Trivial extractor that always returns a fixed list of results."""

    def __init__(self, name="stub", media_type="audio", results=None):
        self._name = name
        self._media_type = media_type
        self._results = results if results is not None else []

    @property
    def name(self):
        return self._name

    @property
    def media_type(self):
        return self._media_type

    def extract(self, clip):
        return self._results


class TestExtractorABC:
    """Verify the Extractor base class contract."""

    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            Extractor()

    def test_stub_extractor_name(self):
        ext = StubExtractor(name="my-ext")
        assert ext.name == "my-ext"

    def test_stub_extractor_media_type(self):
        ext = StubExtractor(media_type="image")
        assert ext.media_type == "image"

    def test_stub_extractor_empty_extract(self):
        ext = StubExtractor()
        assert ext.extract({}) == []

    def test_stub_extractor_returns_results(self):
        hits = [{"confidence": 0.9, "label": "cat"}]
        ext = StubExtractor(results=hits)
        assert ext.extract({}) == hits

    def test_to_dict(self):
        ext = StubExtractor(name="test", media_type="video")
        d = ext.to_dict()
        assert d["name"] == "test"
        assert d["media_type"] == "video"

    def test_load_model_is_noop_by_default(self):
        ext = StubExtractor()
        ext.load_model()  # should not raise


# ---------------------------------------------------------------------------
# ImageClassExtractor unit tests (mock YOLO)
# ---------------------------------------------------------------------------


class TestImageClassExtractor:
    def _make_image_bytes(self, width=64, height=64):
        img = Image.new("RGB", (width, height), color=(128, 128, 128))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def test_from_config(self):
        from vistatotes.media.image.extractor import ImageClassExtractor

        ext = ImageClassExtractor.from_config("my-ext", {"target_class": "car", "threshold": 0.5})
        assert ext.name == "my-ext"
        assert ext.target_class == "car"
        assert ext.threshold == 0.5
        assert ext.media_type == "image"

    def test_to_dict(self):
        from vistatotes.media.image.extractor import ImageClassExtractor

        ext = ImageClassExtractor("test", "person", threshold=0.3, model_id="yolo11n.pt")
        d = ext.to_dict()
        assert d["name"] == "test"
        assert d["media_type"] == "image"
        assert d["extractor_type"] == "image_class"
        assert d["config"]["target_class"] == "person"
        assert d["config"]["threshold"] == 0.3

    def _make_mock_yolo_model(self, detections):
        """Build a mock YOLO model that returns *detections*.

        *detections* is a list of ``(class_id, confidence, bbox)`` tuples.
        """
        import torch

        mock_model = MagicMock()
        mock_boxes = MagicMock()
        mock_boxes.conf = torch.tensor([d[1] for d in detections])
        mock_boxes.cls = torch.tensor([d[0] for d in detections])
        mock_boxes.xyxy = torch.tensor([d[2] for d in detections])
        mock_boxes.__len__ = lambda self: len(detections)

        mock_result = MagicMock()
        mock_result.boxes = mock_boxes
        return mock_model, mock_result

    def test_extract_returns_matching_boxes(self):
        from vistatotes.media.image.extractor import ImageClassExtractor

        detections = [
            (0, 0.9, [10.0, 20.0, 100.0, 200.0]),
            (1, 0.4, [5.0, 5.0, 50.0, 50.0]),
            (0, 0.8, [30.0, 40.0, 80.0, 90.0]),
        ]
        mock_model, mock_result = self._make_mock_yolo_model(detections)
        mock_result.names = {0: "person", 1: "car"}
        mock_model.return_value = [mock_result]

        ext = ImageClassExtractor("test", "person", threshold=0.5)
        ext._model = mock_model  # inject mock directly, skip load_model
        clip = {"image_bytes": self._make_image_bytes()}
        hits = ext.extract(clip)

        assert len(hits) == 2
        assert all(h["label"] == "person" for h in hits)
        assert hits[0]["confidence"] >= hits[1]["confidence"]
        assert len(hits[0]["bbox"]) == 4

    def test_extract_filters_below_threshold(self):
        from vistatotes.media.image.extractor import ImageClassExtractor

        detections = [(0, 0.2, [10.0, 20.0, 100.0, 200.0])]
        mock_model, mock_result = self._make_mock_yolo_model(detections)
        mock_result.names = {0: "person"}
        mock_model.return_value = [mock_result]

        ext = ImageClassExtractor("test", "person", threshold=0.5)
        ext._model = mock_model
        clip = {"image_bytes": self._make_image_bytes()}
        hits = ext.extract(clip)

        assert len(hits) == 0

    def test_extract_filters_wrong_class(self):
        from vistatotes.media.image.extractor import ImageClassExtractor

        detections = [(1, 0.9, [10.0, 20.0, 100.0, 200.0])]
        mock_model, mock_result = self._make_mock_yolo_model(detections)
        mock_result.names = {0: "person", 1: "car"}
        mock_model.return_value = [mock_result]

        ext = ImageClassExtractor("test", "person", threshold=0.25)
        ext._model = mock_model
        clip = {"image_bytes": self._make_image_bytes()}
        hits = ext.extract(clip)

        assert len(hits) == 0

    def test_extract_missing_image_bytes(self):
        from vistatotes.media.image.extractor import ImageClassExtractor

        ext = ImageClassExtractor("test", "person")
        # Provide a real no-op model to avoid import
        ext._model = MagicMock()
        assert ext.extract({}) == []


# ---------------------------------------------------------------------------
# Favorite Extractors API tests
# ---------------------------------------------------------------------------


class TestFavoriteExtractors:
    @pytest.fixture(autouse=True)
    def clear_favorites(self):
        from vistatotes.utils.state import favorite_extractors

        favorite_extractors.clear()
        yield
        favorite_extractors.clear()

    def _post_extractor(self, client, name="test-ext"):
        return client.post(
            "/api/favorite-extractors",
            json={
                "name": name,
                "extractor_type": "image_class",
                "media_type": "image",
                "config": {"target_class": "person", "threshold": 0.5},
            },
        )

    # -- GET list --

    def test_get_empty_list(self, client):
        resp = client.get("/api/favorite-extractors")
        assert resp.status_code == 200
        assert resp.get_json()["extractors"] == []

    def test_get_list_after_add(self, client):
        self._post_extractor(client, "my-extractor")
        resp = client.get("/api/favorite-extractors")
        data = resp.get_json()
        assert len(data["extractors"]) == 1
        ext = data["extractors"][0]
        assert ext["name"] == "my-extractor"
        assert ext["media_type"] == "image"
        assert ext["extractor_type"] == "image_class"

    # -- POST add --

    def test_add_returns_success(self, client):
        resp = self._post_extractor(client)
        assert resp.status_code == 200
        assert resp.get_json()["success"] is True

    def test_add_missing_name_returns_400(self, client):
        resp = client.post(
            "/api/favorite-extractors",
            json={
                "extractor_type": "image_class",
                "media_type": "image",
                "config": {"target_class": "person"},
            },
        )
        assert resp.status_code == 400

    def test_add_missing_extractor_type_returns_400(self, client):
        resp = client.post(
            "/api/favorite-extractors",
            json={
                "name": "test",
                "media_type": "image",
                "config": {"target_class": "person"},
            },
        )
        assert resp.status_code == 400

    def test_add_missing_media_type_returns_400(self, client):
        resp = client.post(
            "/api/favorite-extractors",
            json={
                "name": "test",
                "extractor_type": "image_class",
                "config": {"target_class": "person"},
            },
        )
        assert resp.status_code == 400

    def test_add_missing_config_returns_400(self, client):
        resp = client.post(
            "/api/favorite-extractors",
            json={
                "name": "test",
                "extractor_type": "image_class",
                "media_type": "image",
            },
        )
        assert resp.status_code == 400

    def test_add_multiple(self, client):
        self._post_extractor(client, "ext-a")
        self._post_extractor(client, "ext-b")
        resp = client.get("/api/favorite-extractors")
        names = {e["name"] for e in resp.get_json()["extractors"]}
        assert names == {"ext-a", "ext-b"}

    def test_add_overwrites_existing(self, client):
        self._post_extractor(client, "dup")
        self._post_extractor(client, "dup")
        resp = client.get("/api/favorite-extractors")
        assert len(resp.get_json()["extractors"]) == 1

    # -- DELETE --

    def test_delete_extractor(self, client):
        self._post_extractor(client, "to-delete")
        resp = client.delete("/api/favorite-extractors/to-delete")
        assert resp.status_code == 200
        assert resp.get_json()["success"] is True
        resp = client.get("/api/favorite-extractors")
        assert resp.get_json()["extractors"] == []

    def test_delete_nonexistent_returns_404(self, client):
        resp = client.delete("/api/favorite-extractors/does-not-exist")
        assert resp.status_code == 404

    # -- RENAME --

    def test_rename_extractor(self, client):
        self._post_extractor(client, "old-name")
        resp = client.put(
            "/api/favorite-extractors/old-name/rename",
            json={"new_name": "new-name"},
        )
        assert resp.status_code == 200
        assert resp.get_json()["new_name"] == "new-name"
        names = [e["name"] for e in client.get("/api/favorite-extractors").get_json()["extractors"]]
        assert "new-name" in names
        assert "old-name" not in names

    def test_rename_nonexistent_returns_400(self, client):
        resp = client.put(
            "/api/favorite-extractors/ghost/rename",
            json={"new_name": "anything"},
        )
        assert resp.status_code == 400

    def test_rename_to_existing_name_returns_400(self, client):
        self._post_extractor(client, "ext-a")
        self._post_extractor(client, "ext-b")
        resp = client.put(
            "/api/favorite-extractors/ext-a/rename",
            json={"new_name": "ext-b"},
        )
        assert resp.status_code == 400

    def test_rename_missing_new_name_returns_400(self, client):
        self._post_extractor(client, "some-ext")
        resp = client.put(
            "/api/favorite-extractors/some-ext/rename",
            json={},
        )
        assert resp.status_code == 400

    # -- stored data --

    def test_stored_extractor_has_correct_fields(self, client):
        self._post_extractor(client, "field-check")
        from vistatotes.utils.state import favorite_extractors

        assert "field-check" in favorite_extractors
        stored = favorite_extractors["field-check"]
        assert stored["name"] == "field-check"
        assert stored["extractor_type"] == "image_class"
        assert stored["media_type"] == "image"
        assert "config" in stored
        assert "created_at" in stored


# ---------------------------------------------------------------------------
# Extract endpoint tests (with mocked YOLO)
# ---------------------------------------------------------------------------


class TestExtractEndpoint:
    def test_extract_requires_clips(self, client):
        from vistatotes.utils.state import clips

        saved = dict(clips)
        clips.clear()
        try:
            resp = client.post(
                "/api/extract",
                json={"extractor_type": "image_class", "config": {"target_class": "person"}},
            )
            assert resp.status_code == 400
            assert "No clips loaded" in resp.get_json()["error"]
        finally:
            clips.update(saved)

    def test_extract_media_type_mismatch(self, client):
        # Clips are audio, but image_class extractor expects image
        resp = client.post(
            "/api/extract",
            json={"extractor_type": "image_class", "config": {"target_class": "person"}},
        )
        assert resp.status_code == 400
        assert "does not match" in resp.get_json()["error"]

    def test_extract_missing_extractor_type(self, client):
        resp = client.post("/api/extract", json={"config": {"target_class": "person"}})
        assert resp.status_code == 400

    def test_extract_missing_config(self, client):
        resp = client.post("/api/extract", json={"extractor_type": "image_class"})
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Auto-extract endpoint tests
# ---------------------------------------------------------------------------


class TestAutoExtract:
    @pytest.fixture(autouse=True)
    def clear_favorites(self):
        from vistatotes.utils.state import favorite_extractors

        favorite_extractors.clear()
        yield
        favorite_extractors.clear()

    def test_no_clips_returns_400(self, client):
        from vistatotes.utils.state import clips

        saved = dict(clips)
        clips.clear()
        try:
            resp = client.post("/api/auto-extract")
            assert resp.status_code == 400
        finally:
            clips.update(saved)

    def test_no_extractors_returns_400(self, client):
        resp = client.post("/api/auto-extract")
        assert resp.status_code == 400
        assert "No favorite extractors" in resp.get_json()["error"]

    def test_no_matching_media_type_returns_400(self, client):
        # Clips are audio; add an image extractor
        from vistatotes.utils.state import favorite_extractors

        favorite_extractors["img-ext"] = {
            "name": "img-ext",
            "extractor_type": "image_class",
            "media_type": "image",
            "config": {"target_class": "person"},
            "created_at": 0,
        }
        resp = client.post("/api/auto-extract")
        assert resp.status_code == 400
