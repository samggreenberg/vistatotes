import app as app_module


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

    def test_export_does_not_include_creation_info(self, client):
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

    def test_import_matches_by_origin(self, client):
        """Labels with origin+origin_name match the correct clip."""
        clip = app_module.clips[1]
        labels = [
            {
                "md5": "wrong_md5_on_purpose",
                "label": "good",
                "origin": clip["origin"],
                "origin_name": clip["origin_name"],
            }
        ]
        resp = client.post("/api/labels/import", json={"labels": labels})
        data = resp.get_json()
        assert data["applied"] == 1
        assert 1 in app_module.good_votes

    def test_import_duplicate_md5_labels_both_clips(self, client):
        """Two clips sharing the same MD5 should both receive the label."""
        # Temporarily give clip 2 the same MD5 as clip 1
        original_md5 = app_module.clips[2]["md5"]
        app_module.clips[2]["md5"] = app_module.clips[1]["md5"]
        try:
            shared_md5 = app_module.clips[1]["md5"]
            labels = [{"md5": shared_md5, "label": "good"}]
            resp = client.post("/api/labels/import", json={"labels": labels})
            data = resp.get_json()
            assert data["applied"] == 1
            # Both clips with the same MD5 should receive the label
            assert 1 in app_module.good_votes
            assert 2 in app_module.good_votes
        finally:
            app_module.clips[2]["md5"] = original_md5
