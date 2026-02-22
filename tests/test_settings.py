"""Tests for the persistent settings module.

Covers:
- Settings file read/write (vtsearch.settings)
- Volume persistence
- Favorite processor recipes: add, remove, list, CLI command generation
- ensure_favorite_processors_imported (lazy import on autodetect)
- Flask API routes: GET/PUT /api/settings,
  GET/POST/DELETE /api/settings/favorite-processors
"""

from __future__ import annotations

import json

import pytest

import app as app_module  # noqa: F401 — triggers conftest clip init
from vtsearch import settings as settings_mod


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    """Use a temp file for each test so tests don't interfere."""
    test_settings_path = tmp_path / "settings.json"
    monkeypatch.setattr(settings_mod, "SETTINGS_PATH", test_settings_path)
    settings_mod.reset()
    yield test_settings_path
    settings_mod.reset()


@pytest.fixture
def client():
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        yield c


# ---------------------------------------------------------------------------
# Settings module unit tests
# ---------------------------------------------------------------------------


class TestSettingsModule:
    def test_defaults_when_no_file(self):
        data = settings_mod.get_all()
        assert data["volume"] == 1.0
        assert data["favorite_processors"] == []

    def test_get_set_volume(self, isolated_settings):
        settings_mod.set_volume(0.42)
        assert settings_mod.get_volume() == pytest.approx(0.42)

        # Persisted to disk
        raw = json.loads(isolated_settings.read_text())
        assert raw["volume"] == pytest.approx(0.42)

    def test_volume_clamped(self):
        settings_mod.set_volume(5.0)
        assert settings_mod.get_volume() == 1.0

        settings_mod.set_volume(-3.0)
        assert settings_mod.get_volume() == 0.0

    def test_get_set_inclusion(self, isolated_settings):
        settings_mod.set_inclusion(5)
        assert settings_mod.get_inclusion() == 5

        # Persisted to disk
        raw = json.loads(isolated_settings.read_text())
        assert raw["inclusion"] == 5

    def test_inclusion_clamped(self):
        settings_mod.set_inclusion(100)
        assert settings_mod.get_inclusion() == 10

        settings_mod.set_inclusion(-100)
        assert settings_mod.get_inclusion() == -10

    def test_inclusion_default(self):
        assert settings_mod.get_inclusion() == 0

    def test_inclusion_persists_across_reset(self, isolated_settings):
        settings_mod.set_inclusion(7)
        settings_mod.reset()
        assert settings_mod.get_inclusion() == 7

    def test_add_favorite_processor(self, isolated_settings):
        settings_mod.add_favorite_processor(
            "my det", "detector_file", {"file": "/tmp/det.json"}
        )
        procs = settings_mod.get_favorite_processors()
        assert len(procs) == 1
        assert procs[0]["processor_name"] == "my det"
        assert procs[0]["processor_importer"] == "detector_file"
        assert procs[0]["field_values"]["file"] == "/tmp/det.json"

    def test_add_overwrites_same_name(self):
        settings_mod.add_favorite_processor("a", "detector_file", {"file": "1.json"})
        settings_mod.add_favorite_processor("a", "detector_file", {"file": "2.json"})
        procs = settings_mod.get_favorite_processors()
        assert len(procs) == 1
        assert procs[0]["field_values"]["file"] == "2.json"

    def test_remove_favorite_processor(self):
        settings_mod.add_favorite_processor("x", "detector_file", {"file": "x.json"})
        assert settings_mod.remove_favorite_processor("x") is True
        assert settings_mod.get_favorite_processors() == []

    def test_remove_nonexistent(self):
        assert settings_mod.remove_favorite_processor("nope") is False

    def test_to_settings_json(self):
        entry = {
            "processor_name": "my detector",
            "processor_importer": "detector_file",
            "field_values": {"file": "/path/to/det.json"},
        }
        snippet = settings_mod.to_settings_json(entry)
        import json

        parsed = json.loads(snippet)
        assert parsed["processor_name"] == "my detector"
        assert parsed["processor_importer"] == "detector_file"
        assert parsed["field_values"]["file"] == "/path/to/det.json"

    def test_to_settings_json_with_spaces(self):
        entry = {
            "processor_name": "det",
            "processor_importer": "detector_file",
            "field_values": {"file": "/my path/det.json"},
        }
        snippet = settings_mod.to_settings_json(entry)
        import json

        parsed = json.loads(snippet)
        assert parsed["field_values"]["file"] == "/my path/det.json"

    def test_persistence_survives_reset(self, isolated_settings):
        settings_mod.set_volume(0.7)
        settings_mod.add_favorite_processor("p", "detector_file", {"file": "p.json"})

        # Simulate restart
        settings_mod.reset()

        assert settings_mod.get_volume() == pytest.approx(0.7)
        procs = settings_mod.get_favorite_processors()
        assert len(procs) == 1
        assert procs[0]["processor_name"] == "p"

    def test_corrupt_settings_file(self, isolated_settings):
        isolated_settings.write_text("not json!!!")
        settings_mod.reset()
        # Should fall back to defaults
        assert settings_mod.get_volume() == 1.0
        assert settings_mod.get_favorite_processors() == []


# ---------------------------------------------------------------------------
# ensure_favorite_processors_imported
# ---------------------------------------------------------------------------


class TestEnsureFavoriteProcessorsImported:
    def test_imports_detector_file_processor(self, tmp_path):
        """A favorite processor recipe with detector_file should be imported."""
        from vtsearch.utils import favorite_detectors

        # Create a fake detector JSON
        det_weights = {
            "0.weight": [[0.1] * 512],
            "0.bias": [0.0],
            "2.weight": [[0.2]],
            "2.bias": [0.1],
        }
        det_path = tmp_path / "test_det.json"
        det_path.write_text(json.dumps({
            "media_type": "audio",
            "weights": det_weights,
            "threshold": 0.5,
        }))

        # Clear any existing detectors with this name
        favorite_detectors.pop("settings_test_det", None)

        settings_mod.add_favorite_processor(
            "settings_test_det", "detector_file", {"file": str(det_path)}
        )

        imported = settings_mod.ensure_favorite_processors_imported()
        assert "settings_test_det" in imported
        assert "settings_test_det" in favorite_detectors

        # Clean up
        favorite_detectors.pop("settings_test_det", None)

    def test_skips_already_imported(self, tmp_path):
        """If a detector with the same name already exists, skip it."""
        from vtsearch.utils import add_favorite_detector, favorite_detectors

        add_favorite_detector("existing", "audio", {"0.weight": [[0.1]], "0.bias": [0.0]}, 0.5)

        settings_mod.add_favorite_processor(
            "existing", "detector_file", {"file": "/nonexistent.json"}
        )

        imported = settings_mod.ensure_favorite_processors_imported()
        assert imported == []

        # Clean up
        favorite_detectors.pop("existing", None)

    def test_handles_bad_importer_gracefully(self):
        """Unknown importer name should not crash."""
        settings_mod.add_favorite_processor(
            "bad_proc", "totally_fake_importer", {"file": "x.json"}
        )

        imported = settings_mod.ensure_favorite_processors_imported()
        assert imported == []


# ---------------------------------------------------------------------------
# Flask API routes
# ---------------------------------------------------------------------------


class TestSettingsAPI:
    def test_get_settings(self, client):
        res = client.get("/api/settings")
        assert res.status_code == 200
        data = res.get_json()
        assert "volume" in data
        assert "favorite_processors" in data

    def test_update_volume(self, client):
        res = client.put(
            "/api/settings",
            json={"volume": 0.65},
        )
        assert res.status_code == 200
        data = res.get_json()
        assert data["volume"] == pytest.approx(0.65)

        # Verify it persisted
        res2 = client.get("/api/settings")
        assert res2.get_json()["volume"] == pytest.approx(0.65)

    def test_update_inclusion(self, client):
        res = client.put(
            "/api/settings",
            json={"inclusion": 5},
        )
        assert res.status_code == 200
        data = res.get_json()
        assert data["inclusion"] == 5

        # Verify it persisted
        res2 = client.get("/api/settings")
        assert res2.get_json()["inclusion"] == 5

    def test_update_inclusion_clamped(self, client):
        res = client.put("/api/settings", json={"inclusion": 99})
        assert res.status_code == 200
        assert res.get_json()["inclusion"] == 10

    def test_update_inclusion_invalid(self, client):
        res = client.put(
            "/api/settings",
            json={"inclusion": "not a number"},
        )
        assert res.status_code == 400

    def test_update_volume_invalid(self, client):
        res = client.put(
            "/api/settings",
            json={"volume": "not a number"},
        )
        assert res.status_code == 400

    def test_update_empty_body(self, client):
        res = client.put(
            "/api/settings",
            data="",
            content_type="application/json",
        )
        assert res.status_code == 400

    def test_add_favorite_processor(self, client):
        res = client.post(
            "/api/settings/favorite-processors",
            json={
                "processor_name": "api_test",
                "processor_importer": "detector_file",
                "field_values": {"file": "/tmp/det.json"},
            },
        )
        assert res.status_code == 200
        data = res.get_json()
        assert data["success"] is True
        assert data["processor_name"] == "api_test"
        assert "settings_json" in data

    def test_add_favorite_processor_missing_name(self, client):
        res = client.post(
            "/api/settings/favorite-processors",
            json={"processor_importer": "detector_file", "field_values": {}},
        )
        assert res.status_code == 400

    def test_add_favorite_processor_missing_importer(self, client):
        res = client.post(
            "/api/settings/favorite-processors",
            json={"processor_name": "x", "field_values": {}},
        )
        assert res.status_code == 400

    def test_list_favorite_processors(self, client):
        # Add one first
        client.post(
            "/api/settings/favorite-processors",
            json={
                "processor_name": "list_test",
                "processor_importer": "detector_file",
                "field_values": {"file": "x.json"},
            },
        )

        res = client.get("/api/settings/favorite-processors")
        assert res.status_code == 200
        data = res.get_json()
        assert any(p["processor_name"] == "list_test" for p in data["favorite_processors"])

    def test_delete_favorite_processor(self, client):
        client.post(
            "/api/settings/favorite-processors",
            json={
                "processor_name": "del_test",
                "processor_importer": "detector_file",
                "field_values": {"file": "x.json"},
            },
        )

        res = client.delete("/api/settings/favorite-processors/del_test")
        assert res.status_code == 200

        # Verify it's gone
        res2 = client.get("/api/settings/favorite-processors")
        data = res2.get_json()
        assert not any(p["processor_name"] == "del_test" for p in data["favorite_processors"])

    def test_delete_nonexistent(self, client):
        res = client.delete("/api/settings/favorite-processors/nope")
        assert res.status_code == 404

    def test_get_settings_includes_settings_json(self, client):
        client.post(
            "/api/settings/favorite-processors",
            json={
                "processor_name": "cmd_test",
                "processor_importer": "detector_file",
                "field_values": {"file": "det.json"},
            },
        )

        res = client.get("/api/settings")
        data = res.get_json()
        proc = next(p for p in data["favorite_processors"] if p["processor_name"] == "cmd_test")
        import json

        parsed = json.loads(proc["settings_json"])
        assert parsed["processor_name"] == "cmd_test"
        assert parsed["processor_importer"] == "detector_file"
