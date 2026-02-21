import app as app_module
from vtsearch.utils import get_dataset_creation_info, set_dataset_creation_info


class TestExportLabels:
    def test_empty_export(self, client):
        resp = client.get("/api/labels/export")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data == {"labels": []}

    def test_export_good_labels(self, client):
        app_module.good_votes.update({k: None for k in [1, 2]})
        resp = client.get("/api/labels/export")
        data = resp.get_json()
        assert len(data["labels"]) == 2
        assert all(e["label"] == "good" for e in data["labels"])

    def test_export_bad_labels(self, client):
        app_module.bad_votes.update({k: None for k in [3, 4]})
        resp = client.get("/api/labels/export")
        data = resp.get_json()
        assert len(data["labels"]) == 2
        assert all(e["label"] == "bad" for e in data["labels"])

    def test_export_mixed_labels(self, client):
        app_module.good_votes.update({k: None for k in [1, 2]})
        app_module.bad_votes.update({k: None for k in [3, 4]})
        resp = client.get("/api/labels/export")
        data = resp.get_json()
        assert len(data["labels"]) == 4

    def test_export_contains_md5_and_label(self, client):
        app_module.good_votes[1] = None
        resp = client.get("/api/labels/export")
        data = resp.get_json()
        entry = data["labels"][0]
        assert "md5" in entry
        assert "label" in entry
        assert entry["md5"] == app_module.clips[1]["md5"]
        assert entry["label"] == "good"

    def test_export_includes_dataset_creation_info(self, client):
        creation_info = {
            "importer": "folder",
            "display_name": "Generate from Folder",
            "field_values": {"path": "/data/my_data"},
            "cli_args": "--importer folder --path /data/my_data",
        }
        set_dataset_creation_info(creation_info)
        try:
            app_module.good_votes[1] = None
            resp = client.get("/api/labels/export")
            data = resp.get_json()
            assert "dataset_creation_info" in data
            assert data["dataset_creation_info"]["importer"] == "folder"
            assert data["dataset_creation_info"]["field_values"]["path"] == "/data/my_data"
        finally:
            set_dataset_creation_info(None)

    def test_export_omits_creation_info_when_none(self, client):
        set_dataset_creation_info(None)
        app_module.good_votes[1] = None
        resp = client.get("/api/labels/export")
        data = resp.get_json()
        assert "dataset_creation_info" not in data


class TestImportLabels:
    def test_import_good_label(self, client):
        labels = [{"md5": app_module.clips[1]["md5"], "label": "good"}]
        resp = client.post("/api/labels/import", json={"labels": labels})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["applied"] == 1
        assert data["skipped"] == 0
        assert 1 in app_module.good_votes

    def test_import_bad_label(self, client):
        labels = [{"md5": app_module.clips[1]["md5"], "label": "bad"}]
        resp = client.post("/api/labels/import", json={"labels": labels})
        assert resp.status_code == 200
        assert 1 in app_module.bad_votes

    def test_import_skips_unknown_md5(self, client):
        labels = [{"md5": "nonexistent_md5", "label": "good"}]
        resp = client.post("/api/labels/import", json={"labels": labels})
        data = resp.get_json()
        assert data["applied"] == 0
        assert data["skipped"] == 1

    def test_import_overrides_existing_label(self, client):
        app_module.good_votes[1] = None
        labels = [{"md5": app_module.clips[1]["md5"], "label": "bad"}]
        client.post("/api/labels/import", json={"labels": labels})
        assert 1 not in app_module.good_votes
        assert 1 in app_module.bad_votes

    def test_import_mixed_known_and_unknown(self, client):
        labels = [
            {"md5": app_module.clips[1]["md5"], "label": "good"},
            {"md5": "unknown_md5", "label": "good"},
        ]
        resp = client.post("/api/labels/import", json={"labels": labels})
        data = resp.get_json()
        assert data["applied"] == 1
        assert data["skipped"] == 1

    def test_import_invalid_label_value(self, client):
        labels = [{"md5": app_module.clips[1]["md5"], "label": "meh"}]
        resp = client.post("/api/labels/import", json={"labels": labels})
        data = resp.get_json()
        assert data["applied"] == 0
        assert data["skipped"] == 1

    def test_import_not_a_list(self, client):
        resp = client.post(
            "/api/labels/import",
            json={"labels": "not a list"},
        )
        assert resp.status_code == 400

    def test_import_multiple_labels(self, client):
        labels = []
        for cid in [1, 2, 3]:
            labels.append({"md5": app_module.clips[cid]["md5"], "label": "good"})
        for cid in [4, 5]:
            labels.append({"md5": app_module.clips[cid]["md5"], "label": "bad"})
        resp = client.post("/api/labels/import", json={"labels": labels})
        data = resp.get_json()
        assert data["applied"] == 5
        assert data["skipped"] == 0
        assert set(app_module.good_votes) == {1, 2, 3}
        assert set(app_module.bad_votes) == {4, 5}

    def test_roundtrip_export_import(self, client):
        """Export labels, clear votes, import, and verify same state."""
        app_module.good_votes.update({k: None for k in [1, 3, 5]})
        app_module.bad_votes.update({k: None for k in [2, 4]})
        resp = client.get("/api/labels/export")
        exported = resp.get_json()

        app_module.good_votes.clear()
        app_module.bad_votes.clear()

        resp = client.post("/api/labels/import", json=exported)
        data = resp.get_json()
        assert data["applied"] == 5
        assert set(app_module.good_votes) == {1, 3, 5}
        assert set(app_module.bad_votes) == {2, 4}

    def test_roundtrip_with_creation_info(self, client):
        """Export with dataset_creation_info, re-import via legacy route — info is ignored."""
        creation_info = {
            "importer": "folder",
            "display_name": "Generate from Folder",
            "field_values": {"path": "/data/test"},
            "cli_args": "--importer folder --path /data/test",
        }
        set_dataset_creation_info(creation_info)
        try:
            app_module.good_votes.update({k: None for k in [1, 2]})
            app_module.bad_votes.update({k: None for k in [3]})
            resp = client.get("/api/labels/export")
            exported = resp.get_json()
            assert "dataset_creation_info" in exported

            app_module.good_votes.clear()
            app_module.bad_votes.clear()

            resp = client.post("/api/labels/import", json=exported)
            data = resp.get_json()
            assert data["applied"] == 3
            assert set(app_module.good_votes) == {1, 2}
            assert set(app_module.bad_votes) == {3}
        finally:
            set_dataset_creation_info(None)
