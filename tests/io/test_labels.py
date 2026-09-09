import json

import pytest

from tests import load_detector_and_wait as _load_detector_and_wait
from vtsearch.state import bad_votes, good_votes, medias


def _read_ndjson(resp):
    """Parse a streamed NDJSON export response into a list of label dicts."""
    text = resp.get_data(as_text=True)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


class TestExportLabels:
    def test_empty_export(self, client):
        resp = client.get("/api/labels/export")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data == {"labels": []}

    def test_export_good_labels(self, client):
        good_votes.update({k: None for k in [1, 2]})
        resp = client.get("/api/labels/export")
        data = resp.get_json()
        assert len(data["labels"]) == 2
        assert all(e["label"] == "good" for e in data["labels"])

    def test_export_bad_labels(self, client):
        bad_votes.update({k: None for k in [3, 4]})
        resp = client.get("/api/labels/export")
        data = resp.get_json()
        assert len(data["labels"]) == 2
        assert all(e["label"] == "bad" for e in data["labels"])

    def test_export_mixed_labels(self, client):
        good_votes.update({k: None for k in [1, 2]})
        bad_votes.update({k: None for k in [3, 4]})
        resp = client.get("/api/labels/export")
        data = resp.get_json()
        assert len(data["labels"]) == 4

    def test_export_contains_md5_and_label(self, client):
        good_votes[1] = None
        resp = client.get("/api/labels/export")
        data = resp.get_json()
        entry = data["labels"][0]
        assert "md5" in entry
        assert "label" in entry
        assert entry["md5"] == medias[1]["md5"]
        assert entry["label"] == "good"

    def test_export_does_not_include_creation_info(self, client):
        good_votes[1] = None
        resp = client.get("/api/labels/export")
        data = resp.get_json()
        assert "dataset_creation_info" not in data


class TestExportLabelsNdjson:
    """The streaming ``?format=ndjson`` variant of ``GET /api/labels/export`` (S13)."""

    def test_empty_export_streams_nothing(self, client):
        resp = client.get("/api/labels/export?format=ndjson")
        assert resp.status_code == 200
        assert resp.mimetype == "application/x-ndjson"
        assert _read_ndjson(resp) == []

    def test_streams_one_line_per_label(self, client):
        good_votes.update({k: None for k in [1, 2]})
        bad_votes.update({k: None for k in [3, 4]})
        resp = client.get("/api/labels/export?format=ndjson")
        assert resp.mimetype == "application/x-ndjson"
        rows = _read_ndjson(resp)
        assert len(rows) == 4
        assert all("md5" in r and "label" in r for r in rows)

    def test_ndjson_matches_buffered_labels(self, client):
        """The streamed rows equal the buffered ``labels`` list, entry for entry."""
        good_votes.update({k: None for k in [1, 3, 5]})
        bad_votes.update({k: None for k in [2, 4]})

        buffered = client.get("/api/labels/export").get_json()["labels"]
        streamed = _read_ndjson(client.get("/api/labels/export?format=ndjson"))
        assert streamed == buffered

    def test_goods_only_filter(self, client):
        good_votes.update({k: None for k in [1, 2]})
        bad_votes.update({k: None for k in [3, 4]})
        resp = client.get("/api/labels/export?format=ndjson&goods_only=true")
        rows = _read_ndjson(resp)
        assert len(rows) == 2
        assert all(r["label"] == "good" for r in rows)

    def test_corrections_filter_streams_only_changed(self, client):
        from vtsearch.state import set_find_initial_labels

        set_find_initial_labels({1: "good", 2: "bad", 3: "good"})
        good_votes.update({1: None, 2: None})  # 2 was bad -> correction
        bad_votes[3] = None  # 3 was good -> correction

        resp = client.get("/api/labels/export?format=ndjson&label_filter=corrections")
        rows = _read_ndjson(resp)
        assert len(rows) == 2
        assert all(r["is_correction"] is True for r in rows)
        md5s = {r["md5"] for r in rows}
        assert medias[2]["md5"] in md5s
        assert medias[3]["md5"] in md5s

    def test_corrections_filter_empty_without_find_labels(self, client):
        good_votes[1] = None
        bad_votes[2] = None
        resp = client.get("/api/labels/export?format=ndjson&label_filter=corrections")
        assert _read_ndjson(resp) == []

    def test_enrich_attaches_custom_metadata_but_no_available_columns(self, client):
        good_votes[1] = None
        resp = client.get("/api/labels/export?format=ndjson&enrich=true")
        rows = _read_ndjson(resp)
        assert len(rows) == 1
        # ``available_columns`` is a whole-set aggregate and is never a row.
        assert all("labels" not in r and "available_columns" not in r for r in rows)
        assert "custom_metadata" in rows[0]

    def test_invalid_format_rejected(self, client):
        resp = client.get("/api/labels/export?format=csv")
        assert resp.status_code == 422


class TestImportLabels:
    def test_import_good_label(self, client):
        labels = [{"md5": medias[1]["md5"], "label": "good"}]
        resp = client.post("/api/labels/import", json={"labels": labels})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["applied"] == 1
        assert data["skipped"] == 0
        assert 1 in good_votes

    def test_import_bad_label(self, client):
        labels = [{"md5": medias[1]["md5"], "label": "bad"}]
        resp = client.post("/api/labels/import", json={"labels": labels})
        assert resp.status_code == 200
        assert 1 in bad_votes

    def test_import_skips_unknown_md5(self, client):
        labels = [{"md5": "nonexistent_md5", "label": "good"}]
        resp = client.post("/api/labels/import", json={"labels": labels})
        data = resp.get_json()
        assert data["applied"] == 0
        assert data["skipped"] == 1

    def test_import_overrides_existing_label(self, client):
        good_votes[1] = None
        labels = [{"md5": medias[1]["md5"], "label": "bad"}]
        client.post("/api/labels/import", json={"labels": labels})
        assert 1 not in good_votes
        assert 1 in bad_votes

    def test_import_mixed_known_and_unknown(self, client):
        labels = [
            {"md5": medias[1]["md5"], "label": "good"},
            {"md5": "unknown_md5", "label": "good"},
        ]
        resp = client.post("/api/labels/import", json={"labels": labels})
        data = resp.get_json()
        assert data["applied"] == 1
        assert data["skipped"] == 1

    def test_import_invalid_label_value(self, client):
        labels = [{"md5": medias[1]["md5"], "label": "meh"}]
        resp = client.post("/api/labels/import", json={"labels": labels})
        data = resp.get_json()
        assert data["applied"] == 0
        assert data["skipped"] == 1

    def test_import_not_a_list(self, client):
        resp = client.post(
            "/api/labels/import",
            json={"labels": "not a list"},
        )
        # Marshmallow validates ``labels`` as a list → 422 with the
        # standard flask-smorest envelope.
        assert resp.status_code == 422

    def test_import_multiple_labels(self, client):
        labels = []
        for cid in [1, 2, 3]:
            labels.append({"md5": medias[cid]["md5"], "label": "good"})
        for cid in [4, 5]:
            labels.append({"md5": medias[cid]["md5"], "label": "bad"})
        resp = client.post("/api/labels/import", json={"labels": labels})
        data = resp.get_json()
        assert data["applied"] == 5
        assert data["skipped"] == 0
        assert set(good_votes) == {1, 2, 3}
        assert set(bad_votes) == {4, 5}

    def test_roundtrip_export_import(self, client):
        """Export labels, clear votes, import, and verify same state."""
        good_votes.update({k: None for k in [1, 3, 5]})
        bad_votes.update({k: None for k in [2, 4]})
        resp = client.get("/api/labels/export")
        exported = resp.get_json()

        good_votes.clear()
        bad_votes.clear()

        resp = client.post("/api/labels/import", json=exported)
        data = resp.get_json()
        assert data["applied"] == 5
        assert set(good_votes) == {1, 3, 5}
        assert set(bad_votes) == {2, 4}

    def test_import_matches_by_origin(self, client):
        """Labels with origin+origin_name match the correct media."""
        media = medias[1]
        labels = [
            {
                "md5": "wrong_md5_on_purpose",
                "label": "good",
                "origin": media["origin"],
                "origin_name": media["origin_name"],
            }
        ]
        resp = client.post("/api/labels/import", json={"labels": labels})
        data = resp.get_json()
        assert data["applied"] == 1
        assert 1 in good_votes

    def test_import_duplicate_md5_labels_both_clips(self, client):
        """Two medias sharing the same MD5 should both receive the label."""
        # Temporarily give media 2 the same MD5 as media 1
        original_md5 = medias[2]["md5"]
        medias[2]["md5"] = medias[1]["md5"]
        try:
            shared_md5 = medias[1]["md5"]
            labels = [{"md5": shared_md5, "label": "good"}]
            resp = client.post("/api/labels/import", json={"labels": labels})
            data = resp.get_json()
            assert data["applied"] == 1
            # Both medias with the same MD5 should receive the label
            assert 1 in good_votes
            assert 2 in good_votes
        finally:
            medias[2]["md5"] = original_md5


class TestExportOriginOnlyFallback:
    """Persisted labelset elements that don't resolve into the active dataset
    still export, marked ``origin_only`` (issue #2702).

    The export route composes from cid-keyed votes ∩ active medias; these
    tests plant elements in the detector's on-disk labelset whose origin/md5
    match nothing loaded, and verify they surface in the export instead of
    silently vanishing.
    """

    GHOST_GOOD = {
        "md5": "f" * 32,
        "label": "good",
        "origin": {"importer": "test", "params": {"shard": "elsewhere"}},
        "origin_name": "ghost_good.wav",
        "filename": "ghost_good.wav",
        "category": "elsewhere",
        "metadata": {"contentID": "c-123"},
    }
    GHOST_BAD = {
        "md5": "e" * 32,
        "label": "bad",
        "origin": {"importer": "test", "params": {"shard": "elsewhere"}},
        "origin_name": "ghost_bad.wav",
        "filename": "ghost_bad.wav",
        "category": "elsewhere",
    }

    def _setup_detector(self, client, with_votes=True):
        """Create + load a registry detector; optionally cast one good + one bad vote.

        Returns ``(detector_id, good_cid, bad_cid)``.
        """
        if len(medias) < 2:
            pytest.skip("Need at least 2 medias")
        ids = list(medias.keys())
        good_cid, bad_cid = ids[0], ids[1]

        res = client.post(
            "/api/detectors/registry",
            json={"name": "OriginOnlyExport", "media_type": "audio", "text_query": "test"},
        )
        detector_id = res.get_json()["detector"]["id"]
        _load_detector_and_wait(client, detector_id)

        if with_votes:
            client.post(f"/api/medias/{good_cid}/vote", json={"target": "good"})
            client.post(f"/api/medias/{bad_cid}/vote", json={"target": "bad"})
        return detector_id, good_cid, bad_cid

    def _inject_elements(self, detector_id, *elements):
        """Append raw label entries to the detector's on-disk labelset."""
        from vtscore.detectors.dataset_sync import reset_mtime_cache_for_tests
        from vtscore.detectors.registry import get_detector
        from vtscore.detectors.store import _detector_path, _read_detector, _write_detector

        entry = get_detector(detector_id)
        assert entry is not None
        path = _detector_path(entry["name"])
        data = _read_detector(path)
        assert data is not None
        labelset = data.get("labelset") or {"labels": []}
        labelset.setdefault("labels", []).extend(dict(e) for e in elements)
        data["labelset"] = labelset
        _write_detector(path, data)
        # Drop the TTL-cached mtime so the next request's rehydrate sees the
        # write immediately.
        reset_mtime_cache_for_tests()

    def test_unresolvable_element_exports_with_origin_only_flag(self, client):
        detector_id, good_cid, bad_cid = self._setup_detector(client)
        self._inject_elements(detector_id, self.GHOST_GOOD)

        data = client.get("/api/labels/export").get_json()
        by_name = {e.get("origin_name"): e for e in data["labels"]}
        ghost = by_name.get("ghost_good.wav")
        assert ghost is not None, f"origin-only element missing from export: {list(by_name)}"
        assert ghost["origin_only"] is True
        assert ghost["label"] == "good"
        # Vote-derived entries stay unmarked.
        vote_derived = [e for e in data["labels"] if e.get("origin_name") != "ghost_good.wav"]
        assert len(vote_derived) >= 2
        assert all("origin_only" not in e for e in vote_derived)

    def test_export_of_fully_nonoverlapping_labelset(self, client):
        """A detector whose labelset shares nothing with the active dataset
        exports the whole labelset instead of an empty set (the #2702 headline)."""
        detector_id, _, _ = self._setup_detector(client, with_votes=False)
        self._inject_elements(detector_id, self.GHOST_GOOD, self.GHOST_BAD)

        data = client.get("/api/labels/export").get_json()
        assert {e["origin_name"] for e in data["labels"]} == {"ghost_good.wav", "ghost_bad.wav"}
        assert all(e["origin_only"] is True for e in data["labels"])

    def test_ndjson_includes_origin_only_rows(self, client):
        detector_id, _, _ = self._setup_detector(client)
        self._inject_elements(detector_id, self.GHOST_GOOD, self.GHOST_BAD)

        buffered = client.get("/api/labels/export").get_json()["labels"]
        streamed = _read_ndjson(client.get("/api/labels/export?format=ndjson"))
        assert streamed == buffered
        assert sum(1 for r in streamed if r.get("origin_only")) == 2

    def test_good_bad_filters_apply_to_origin_only_entries(self, client):
        detector_id, _, _ = self._setup_detector(client)
        self._inject_elements(detector_id, self.GHOST_GOOD, self.GHOST_BAD)

        goods = client.get("/api/labels/export?goods_only=true").get_json()["labels"]
        assert "ghost_good.wav" in {e.get("origin_name") for e in goods}
        assert "ghost_bad.wav" not in {e.get("origin_name") for e in goods}

        bads = client.get("/api/labels/export?label_filter=bad").get_json()["labels"]
        assert {e.get("origin_name") for e in bads if e.get("origin_only")} == {"ghost_bad.wav"}
        assert all(e["label"] == "bad" for e in bads)

    def test_vote_scoped_filters_exclude_origin_only_entries(self, client):
        """corrections / unverified / verified partition session vote state;
        origin-only elements were never part of the session, so they opt out."""
        detector_id, _, _ = self._setup_detector(client)
        self._inject_elements(detector_id, self.GHOST_GOOD, self.GHOST_BAD)

        for label_filter in ("corrections", "unverified", "verified"):
            data = client.get(f"/api/labels/export?label_filter={label_filter}").get_json()
            assert all(not e.get("origin_only") for e in data["labels"]), label_filter

    def test_resolved_but_unvoted_element_is_not_resurrected(self, client):
        """An element that resolves into the active dataset but has no vote was
        unlabelled this session; the fallback must not re-export it from disk."""
        from vtscore.detectors.registry import get_detector
        from vtscore.detectors.store import _detector_path
        from vtscore.state.core import get_active_detector_context

        detector_id, good_cid, bad_cid = self._setup_detector(client)
        good_media = medias[good_cid]

        # Un-vote the good media (the vote sync also drops it from disk).
        client.post(f"/api/medias/{good_cid}/vote", json={"target": "none"})

        # Plant a stale on-disk entry for the unvoted media, then re-stamp the
        # context's cached mtime so ``ensure_votes_match_active_dataset``
        # doesn't treat the write as a labelset change and restore it as a
        # vote.  This reproduces the state where disk and session disagree
        # about a *resolvable* element - the session must win.
        self._inject_elements(
            detector_id,
            {
                "md5": good_media["md5"],
                "label": "good",
                "origin": good_media.get("origin"),
                "origin_name": good_media.get("origin_name", ""),
                "filename": good_media.get("filename", ""),
            },
        )
        det_ctx = get_active_detector_context()
        entry = get_detector(detector_id)
        assert entry is not None
        path = _detector_path(entry["name"])
        det_ctx.cached_labelset_mtime = path.stat().st_mtime

        data = client.get("/api/labels/export").get_json()
        assert all(e["md5"] != good_media["md5"] for e in data["labels"])

    def test_enrich_degrades_gracefully_for_origin_only_entries(self, client):
        """Origin-only rows can't be enriched from a media dict; their
        custom_metadata degrades to origin params + stored element metadata."""
        detector_id, _, _ = self._setup_detector(client)
        self._inject_elements(detector_id, self.GHOST_GOOD)

        data = client.get("/api/labels/export?enrich=true").get_json()
        ghost = next(e for e in data["labels"] if e.get("origin_only"))
        assert ghost["custom_metadata"]["contentID"] == "c-123"
        assert ghost["custom_metadata"]["shard"] == "elsewhere"
        assert "contentID" in data["available_columns"]
        assert "shard" in data["available_columns"]


class TestExportNamedDetectorLabelset:
    """``?detector_name=`` exports a *named* detector's persisted labelset.

    The Dashboard's row action points at a detector in a list, so its export
    must answer "what is detector X's labelset" rather than "what is labelled
    in whatever pair the top-bar pulldown is on" — the mismatch behind issue
    #3639, where the same row exported the right labels, then nothing, then
    the entire dataset depending only on where the ambient context had been.
    """

    GHOST = {
        "md5": "d" * 32,
        "label": "good",
        "origin": {"importer": "test", "params": {"shard": "elsewhere"}},
        "origin_name": "ghost_named.wav",
        "filename": "ghost_named.wav",
        "metadata": {"contentID": "c-999"},
    }

    def _detector_with_labels(self, client, name="NamedExport"):
        """Create + load a detector, vote one good and one bad, return its name."""
        if len(medias) < 2:
            pytest.skip("Need at least 2 medias")
        ids = list(medias.keys())
        good_cid, bad_cid = ids[0], ids[1]
        res = client.post(
            "/api/detectors/registry",
            json={"name": name, "media_type": "audio", "text_query": "test"},
        )
        detector_id = res.get_json()["detector"]["id"]
        _load_detector_and_wait(client, detector_id)
        client.post(f"/api/medias/{good_cid}/vote", json={"target": "good"})
        client.post(f"/api/medias/{bad_cid}/vote", json={"target": "bad"})
        return name, detector_id, good_cid, bad_cid

    def test_exports_the_named_detector_with_no_active_detector(self, client):
        """The headline: the row action names the detector, and the ambient
        pair is irrelevant. Symptom 2 of #3639 exported nothing here."""
        name, _, good_cid, bad_cid = self._detector_with_labels(client)

        # Fresh request with no X-Detector-Id at all, as a Dashboard visit
        # after a page refresh sends.
        from vtscore.state.core import get_active_detector_context

        det_ctx = get_active_detector_context()
        det_ctx.good_votes.clear()
        det_ctx.bad_votes.clear()

        data = client.get(f"/api/labels/export?detector_name={name}").get_json()
        assert {e["label"] for e in data["labels"]} == {"good", "bad"}
        assert {e["md5"] for e in data["labels"]} == {
            medias[good_cid]["md5"],
            medias[bad_cid]["md5"],
        }

    def test_live_find_session_does_not_leak_into_the_export(self, client):
        """Symptom 3: Find fills the detector's votes with its own call for
        every item in the dataset, flagged ``find_mode`` so they stay out of
        the labelset. A labelset export must not resurrect them as labels."""
        name, _, good_cid, _ = self._detector_with_labels(client)

        from vtscore.state.core import get_active_detector_context

        det_ctx = get_active_detector_context()
        det_ctx.find_mode = True
        # Find's presumptions: a call for *every* media in the dataset.
        det_ctx.good_votes.clear()
        det_ctx.bad_votes.clear()
        det_ctx.good_votes.update({cid: None for cid in medias})
        det_ctx.find_scores.update({cid: 0.9 for cid in medias})

        # The session-scoped export sees the whole collection...
        live = client.get("/api/labels/export").get_json()["labels"]
        assert len(live) == len(medias)

        # ...while the detector-scoped one stays the two real labels.
        scoped = client.get(f"/api/labels/export?detector_name={name}").get_json()["labels"]
        assert len(scoped) == 2
        assert {e["label"] for e in scoped} == {"good", "bad"}
        assert medias[good_cid]["md5"] in {e["md5"] for e in scoped}

    def test_other_detectors_labelset_is_not_exported(self, client):
        """Two detectors, one active: naming the other one exports the other
        one's labels, not the active pair's."""
        ids = list(medias.keys())
        first, first_id, first_good, _ = self._detector_with_labels(client, "FirstDetector")
        res = client.post(
            "/api/detectors/registry",
            json={"name": "SecondDetector", "media_type": "audio", "text_query": "test"},
        )
        second_id = res.get_json()["detector"]["id"]
        _load_detector_and_wait(client, second_id)
        second_good = ids[2] if len(ids) > 2 else ids[0]
        client.post(f"/api/medias/{second_good}/vote", json={"target": "good"})

        # SecondDetector is the active one now; ask for FirstDetector's.
        data = client.get(f"/api/labels/export?detector_name={first}").get_json()
        assert {e["md5"] for e in data["labels"]} == {
            medias[first_good]["md5"],
            medias[list(medias.keys())[1]]["md5"],
        }
        assert first_id != second_id

    def test_good_and_bad_filters_apply(self, client):
        name, _, good_cid, bad_cid = self._detector_with_labels(client)

        goods = client.get(f"/api/labels/export?detector_name={name}&goods_only=true").get_json()
        assert {e["md5"] for e in goods["labels"]} == {medias[good_cid]["md5"]}

        bads = client.get(f"/api/labels/export?detector_name={name}&label_filter=bad").get_json()
        assert {e["md5"] for e in bads["labels"]} == {medias[bad_cid]["md5"]}

    def test_unresolvable_element_is_marked_origin_only(self, client):
        from vtscore.detectors.dataset_sync import reset_mtime_cache_for_tests
        from vtscore.detectors.store import _detector_path, _read_detector, _write_detector

        name, _, _, _ = self._detector_with_labels(client)
        path = _detector_path(name)
        data = _read_detector(path)
        assert data is not None
        data.setdefault("labelset", {}).setdefault("labels", []).append(dict(self.GHOST))
        _write_detector(path, data)
        reset_mtime_cache_for_tests()

        exported = client.get(f"/api/labels/export?detector_name={name}&enrich=true").get_json()
        by_name = {e.get("origin_name"): e for e in exported["labels"]}
        ghost = by_name["ghost_named.wav"]
        assert ghost["origin_only"] is True
        assert ghost["custom_metadata"]["contentID"] == "c-999"
        # Elements the active dataset *can* see are ordinary labels.
        resolved = [e for e in exported["labels"] if e.get("origin_name") != "ghost_named.wav"]
        assert resolved and all("origin_only" not in e for e in resolved)

    def test_ndjson_matches_the_buffered_export(self, client):
        name, _, _, _ = self._detector_with_labels(client)
        buffered = client.get(f"/api/labels/export?detector_name={name}").get_json()["labels"]
        streamed = _read_ndjson(client.get(f"/api/labels/export?detector_name={name}&format=ndjson"))
        assert streamed == buffered

    def test_unknown_detector_is_404(self, client):
        resp = client.get("/api/labels/export?detector_name=NoSuchDetector")
        assert resp.status_code == 404

    def test_vote_scoped_filters_are_refused(self, client):
        """They partition the live session, which a persisted labelset has no
        part in; refusing beats silently exporting something else."""
        name, _, _, _ = self._detector_with_labels(client)
        for label_filter in ("corrections", "unverified", "verified"):
            resp = client.get(f"/api/labels/export?detector_name={name}&label_filter={label_filter}")
            assert resp.status_code == 400, label_filter
