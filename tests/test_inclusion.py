import pytest

from vtsearch import settings as settings_mod


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    """Use a temp file so inclusion persistence doesn't affect other tests."""
    test_settings_path = tmp_path / "settings.json"
    monkeypatch.setattr(settings_mod, "SETTINGS_PATH", test_settings_path)
    settings_mod.reset()
    yield test_settings_path
    settings_mod.reset()


class TestInclusionEndpoints:
    def test_get_default_inclusion(self, client):
        resp = client.get("/api/inclusion")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "inclusion" in data
        assert isinstance(data["inclusion"], int)

    def test_set_inclusion_valid_value(self, client):
        resp = client.post("/api/inclusion", json={"inclusion": 5})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["inclusion"] == 5

        # Verify it persists
        resp = client.get("/api/inclusion")
        data = resp.get_json()
        assert data["inclusion"] == 5

    def test_set_inclusion_negative_value(self, client):
        resp = client.post("/api/inclusion", json={"inclusion": -5})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["inclusion"] == -5

    def test_set_inclusion_clamped_to_max(self, client):
        resp = client.post("/api/inclusion", json={"inclusion": 100})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["inclusion"] == 10  # Clamped to max

    def test_set_inclusion_clamped_to_min(self, client):
        resp = client.post("/api/inclusion", json={"inclusion": -100})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["inclusion"] == -10  # Clamped to min

    def test_set_inclusion_float_converted_to_int(self, client):
        resp = client.post("/api/inclusion", json={"inclusion": 3.7})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["inclusion"] == 3  # Converted to int

    def test_set_inclusion_invalid_type(self, client):
        resp = client.post("/api/inclusion", json={"inclusion": "not a number"})
        assert resp.status_code == 400

    def test_set_inclusion_missing_field(self, client):
        resp = client.post("/api/inclusion", json={"wrong": 5})
        assert resp.status_code == 400
