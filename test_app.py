import hashlib
import io
import json
import struct
import tarfile
import tempfile
import wave
import zipfile
from pathlib import Path

import numpy as np
import pytest

import app as app_module

# Import refactored modules and make them accessible through app_module
from config import NUM_CLIPS, SAMPLE_RATE
from vistatotes.audio import generate_wav
from vistatotes.models import initialize_models, train_and_score
from vistatotes.utils import bad_votes, clips, good_votes

# Attach to app_module for backward compatibility with existing tests
app_module.NUM_CLIPS = NUM_CLIPS
app_module.SAMPLE_RATE = SAMPLE_RATE
app_module.generate_wav = generate_wav
app_module.train_and_score = train_and_score
app_module.clips = clips
app_module.good_votes = good_votes
app_module.bad_votes = bad_votes

# Initialize models and clips
initialize_models()
app_module.init_clips()


@pytest.fixture(autouse=True)
def reset_votes():
    """Reset vote state before each test."""
    good_votes.clear()
    bad_votes.clear()


@pytest.fixture
def client():
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        yield c


# ---------------------------------------------------------------------------
# generate_wav
# ---------------------------------------------------------------------------


class TestGenerateWav:
    def test_returns_valid_wav(self):
        data = app_module.generate_wav(440.0, 1.0)
        buf = io.BytesIO(data)
        with wave.open(buf, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == app_module.SAMPLE_RATE

    def test_duration_determines_frame_count(self):
        for dur in (0.5, 1.0, 2.0):
            data = app_module.generate_wav(440.0, dur)
            buf = io.BytesIO(data)
            with wave.open(buf, "rb") as wf:
                expected = int(app_module.SAMPLE_RATE * dur)
                assert wf.getnframes() == expected

    def test_different_frequencies_produce_different_output(self):
        wav_a = app_module.generate_wav(200.0, 0.5)
        wav_b = app_module.generate_wav(800.0, 0.5)
        assert wav_a != wav_b

    def test_zero_duration_produces_empty_frames(self):
        data = app_module.generate_wav(440.0, 0.0)
        buf = io.BytesIO(data)
        with wave.open(buf, "rb") as wf:
            assert wf.getnframes() == 0


# ---------------------------------------------------------------------------
# init_clips
# ---------------------------------------------------------------------------


class TestInitClips:
    def test_creates_correct_number_of_clips(self):
        assert len(app_module.clips) == app_module.NUM_CLIPS

    def test_clip_ids_are_1_to_num_clips(self):
        assert set(app_module.clips.keys()) == set(range(1, app_module.NUM_CLIPS + 1))

    def test_clip_frequencies(self):
        for i in range(1, app_module.NUM_CLIPS + 1):
            expected_freq = 200 + (i - 1) * 50
            assert app_module.clips[i]["frequency"] == expected_freq

    def test_clip_durations(self):
        for i in range(1, app_module.NUM_CLIPS + 1):
            expected_dur = round(1.0 + (i % 5) * 0.5, 1)
            assert app_module.clips[i]["duration"] == expected_dur

    def test_clip_has_embedding(self):
        for clip in app_module.clips.values():
            emb = clip["embedding"]
            assert isinstance(emb, np.ndarray)
            assert len(emb) > 0

    def test_clip_has_wav_bytes(self):
        for clip in app_module.clips.values():
            assert isinstance(clip["wav_bytes"], bytes)
            assert len(clip["wav_bytes"]) > 0

    def test_file_size_matches_wav_bytes_length(self):
        for clip in app_module.clips.values():
            assert clip["file_size"] == len(clip["wav_bytes"])

    def test_deterministic_embeddings(self):
        """CLAP embeddings should be deterministic for the same input audio."""
        emb_first = app_module.clips[1]["embedding"].copy()
        # Re-init and check the same values appear
        old_clips = dict(app_module.clips)
        app_module.clips.clear()
        app_module.init_clips()
        np.testing.assert_array_almost_equal(
            app_module.clips[1]["embedding"], emb_first
        )
        # Restore
        app_module.clips.clear()
        app_module.clips.update(old_clips)


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------


class TestIndex:
    def test_serves_index_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"VectoryTones" in resp.data


# ---------------------------------------------------------------------------
# GET /api/clips
# ---------------------------------------------------------------------------


class TestListClips:
    def test_returns_all_clips(self, client):
        resp = client.get("/api/clips")
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data) == app_module.NUM_CLIPS

    def test_clip_fields(self, client):
        resp = client.get("/api/clips")
        data = resp.get_json()
        for clip in data:
            assert "id" in clip
            assert "frequency" in clip
            assert "duration" in clip
            assert "file_size" in clip

    def test_does_not_expose_wav_bytes(self, client):
        resp = client.get("/api/clips")
        data = resp.get_json()
        for clip in data:
            assert "wav_bytes" not in clip

    def test_does_not_expose_embedding(self, client):
        resp = client.get("/api/clips")
        data = resp.get_json()
        for clip in data:
            assert "embedding" not in clip


# ---------------------------------------------------------------------------
# GET /api/clips/<id>/audio
# ---------------------------------------------------------------------------


class TestClipAudio:
    def test_returns_wav_for_valid_id(self, client):
        resp = client.get("/api/clips/1/audio")
        assert resp.status_code == 200
        assert resp.content_type == "audio/wav"

    def test_wav_is_valid(self, client):
        resp = client.get("/api/clips/1/audio")
        buf = io.BytesIO(resp.data)
        with wave.open(buf, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2

    def test_returns_404_for_invalid_id(self, client):
        resp = client.get("/api/clips/9999/audio")
        assert resp.status_code == 404
        data = resp.get_json()
        assert data["error"] == "not found"

    def test_returns_404_for_zero_id(self, client):
        resp = client.get("/api/clips/0/audio")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /api/clips/<id>/vote
# ---------------------------------------------------------------------------


class TestVoteClip:
    def test_vote_good(self, client):
        resp = client.post("/api/clips/1/vote", json={"vote": "good"})
        assert resp.status_code == 200
        assert resp.get_json()["ok"] is True
        assert 1 in app_module.good_votes

    def test_vote_bad(self, client):
        resp = client.post("/api/clips/1/vote", json={"vote": "bad"})
        assert resp.status_code == 200
        assert 1 in app_module.bad_votes

    def test_toggle_good_off(self, client):
        """Voting good twice should toggle it off."""
        client.post("/api/clips/1/vote", json={"vote": "good"})
        assert 1 in app_module.good_votes
        client.post("/api/clips/1/vote", json={"vote": "good"})
        assert 1 not in app_module.good_votes

    def test_toggle_bad_off(self, client):
        """Voting bad twice should toggle it off."""
        client.post("/api/clips/1/vote", json={"vote": "bad"})
        assert 1 in app_module.bad_votes
        client.post("/api/clips/1/vote", json={"vote": "bad"})
        assert 1 not in app_module.bad_votes

    def test_switch_from_good_to_bad(self, client):
        client.post("/api/clips/1/vote", json={"vote": "good"})
        client.post("/api/clips/1/vote", json={"vote": "bad"})
        assert 1 not in app_module.good_votes
        assert 1 in app_module.bad_votes

    def test_switch_from_bad_to_good(self, client):
        client.post("/api/clips/1/vote", json={"vote": "bad"})
        client.post("/api/clips/1/vote", json={"vote": "good"})
        assert 1 not in app_module.bad_votes
        assert 1 in app_module.good_votes

    def test_invalid_vote_value(self, client):
        resp = client.post("/api/clips/1/vote", json={"vote": "meh"})
        assert resp.status_code == 400
        assert "vote must be" in resp.get_json()["error"]

    def test_missing_vote_field(self, client):
        resp = client.post("/api/clips/1/vote", json={"wrong": "field"})
        assert resp.status_code == 400

    def test_vote_nonexistent_clip(self, client):
        resp = client.post("/api/clips/9999/vote", json={"vote": "good"})
        assert resp.status_code == 404

    def test_multiple_clips_independent_votes(self, client):
        client.post("/api/clips/1/vote", json={"vote": "good"})
        client.post("/api/clips/2/vote", json={"vote": "bad"})
        assert 1 in app_module.good_votes
        assert 2 in app_module.bad_votes
        assert 1 not in app_module.bad_votes
        assert 2 not in app_module.good_votes


# ---------------------------------------------------------------------------
# GET /api/votes
# ---------------------------------------------------------------------------


class TestGetVotes:
    def test_empty_votes(self, client):
        resp = client.get("/api/votes")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data == {"good": [], "bad": []}

    def test_returns_good_votes(self, client):
        app_module.good_votes.update({k: None for k in [1, 3, 5]})
        resp = client.get("/api/votes")
        data = resp.get_json()
        assert data["good"] == [1, 3, 5]  # sorted

    def test_returns_bad_votes(self, client):
        app_module.bad_votes.update({k: None for k in [2, 4]})
        resp = client.get("/api/votes")
        data = resp.get_json()
        assert data["bad"] == [2, 4]  # sorted

    def test_returns_both(self, client):
        app_module.good_votes[1] = None
        app_module.bad_votes[2] = None
        resp = client.get("/api/votes")
        data = resp.get_json()
        assert data["good"] == [1]
        assert data["bad"] == [2]

    def test_votes_after_voting_via_api(self, client):
        client.post("/api/clips/3/vote", json={"vote": "good"})
        client.post("/api/clips/5/vote", json={"vote": "bad"})
        resp = client.get("/api/votes")
        data = resp.get_json()
        assert 3 in data["good"]
        assert 5 in data["bad"]


# ---------------------------------------------------------------------------
# POST /api/sort
# ---------------------------------------------------------------------------


class TestSortClips:
    def test_returns_all_clips(self, client):
        resp = client.post("/api/sort", json={"text": "high pitched beep"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data["results"]) == app_module.NUM_CLIPS
        assert "threshold" in data

    def test_result_contains_id_and_similarity(self, client):
        resp = client.post("/api/sort", json={"text": "low tone"})
        data = resp.get_json()
        for entry in data["results"]:
            assert "id" in entry
            assert "similarity" in entry

    def test_sorted_by_descending_similarity(self, client):
        resp = client.post("/api/sort", json={"text": "a beeping sound"})
        data = resp.get_json()
        similarities = [e["similarity"] for e in data["results"]]
        assert similarities == sorted(similarities, reverse=True)

    def test_all_clip_ids_present(self, client):
        resp = client.post("/api/sort", json={"text": "sine wave"})
        data = resp.get_json()
        ids = {e["id"] for e in data["results"]}
        assert ids == set(range(1, app_module.NUM_CLIPS + 1))

    def test_similarity_values_in_range(self, client):
        resp = client.post("/api/sort", json={"text": "high pitch"})
        data = resp.get_json()
        for entry in data["results"]:
            assert -1.0 <= entry["similarity"] <= 1.0

    def test_empty_text_returns_400(self, client):
        resp = client.post("/api/sort", json={"text": ""})
        assert resp.status_code == 400

    def test_missing_text_returns_400(self, client):
        resp = client.post("/api/sort", json={"other": "field"})
        assert resp.status_code == 400

    def test_whitespace_only_returns_400(self, client):
        resp = client.post("/api/sort", json={"text": "   "})
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# train_and_score
# ---------------------------------------------------------------------------


class TestTrainAndScore:
    def test_returns_list_of_scored_clips(self):
        app_module.good_votes.update({k: None for k in [1, 2]})
        app_module.bad_votes.update({k: None for k in [3, 4]})
        results, threshold = app_module.train_and_score(
            app_module.clips, app_module.good_votes, app_module.bad_votes
        )
        assert len(results) == app_module.NUM_CLIPS
        assert isinstance(threshold, float)
        for entry in results:
            assert "id" in entry
            assert "score" in entry

    def test_scores_between_zero_and_one(self):
        app_module.good_votes.update({k: None for k in [1, 2]})
        app_module.bad_votes.update({k: None for k in [3, 4]})
        results, threshold = app_module.train_and_score(
            app_module.clips, app_module.good_votes, app_module.bad_votes
        )
        for entry in results:
            assert 0.0 <= entry["score"] <= 1.0

    def test_results_sorted_descending(self):
        app_module.good_votes.update({k: None for k in [1, 2]})
        app_module.bad_votes.update({k: None for k in [3, 4]})
        results, threshold = app_module.train_and_score(
            app_module.clips, app_module.good_votes, app_module.bad_votes
        )
        scores = [e["score"] for e in results]
        assert scores == sorted(scores, reverse=True)

    def test_good_clips_scored_higher_than_bad(self):
        app_module.good_votes.update({k: None for k in [1, 2, 3]})
        app_module.bad_votes.update({k: None for k in [18, 19, 20]})
        results, threshold = app_module.train_and_score(
            app_module.clips, app_module.good_votes, app_module.bad_votes
        )
        score_map = {e["id"]: e["score"] for e in results}
        avg_good = np.mean([score_map[i] for i in app_module.good_votes])
        avg_bad = np.mean([score_map[i] for i in app_module.bad_votes])
        assert avg_good > avg_bad


# ---------------------------------------------------------------------------
# POST /api/learned-sort
# ---------------------------------------------------------------------------


class TestLearnedSort:
    def test_returns_all_clips(self, client):
        app_module.good_votes.update({k: None for k in [1, 2]})
        app_module.bad_votes.update({k: None for k in [3, 4]})
        resp = client.post("/api/learned-sort")
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data["results"]) == app_module.NUM_CLIPS
        assert "threshold" in data

    def test_result_fields(self, client):
        app_module.good_votes.update({k: None for k in [1, 2]})
        app_module.bad_votes.update({k: None for k in [3, 4]})
        resp = client.post("/api/learned-sort")
        data = resp.get_json()
        for entry in data["results"]:
            assert "id" in entry
            assert "score" in entry

    def test_sorted_descending(self, client):
        app_module.good_votes.update({k: None for k in [1, 2]})
        app_module.bad_votes.update({k: None for k in [3, 4]})
        resp = client.post("/api/learned-sort")
        data = resp.get_json()
        scores = [e["score"] for e in data["results"]]
        assert scores == sorted(scores, reverse=True)

    def test_all_clip_ids_present(self, client):
        app_module.good_votes.update({k: None for k in [1, 2]})
        app_module.bad_votes.update({k: None for k in [3, 4]})
        resp = client.post("/api/learned-sort")
        data = resp.get_json()
        ids = {e["id"] for e in data["results"]}
        assert ids == set(range(1, app_module.NUM_CLIPS + 1))

    def test_only_good_votes_returns_400(self, client):
        app_module.good_votes.update({k: None for k in [1, 2]})
        resp = client.post("/api/learned-sort")
        assert resp.status_code == 400

    def test_only_bad_votes_returns_400(self, client):
        app_module.bad_votes.update({k: None for k in [3, 4]})
        resp = client.post("/api/learned-sort")
        assert resp.status_code == 400

    def test_scores_in_valid_range(self, client):
        app_module.good_votes.update({k: None for k in [1, 2]})
        app_module.bad_votes.update({k: None for k in [3, 4]})
        resp = client.post("/api/learned-sort")
        data = resp.get_json()
        for entry in data["results"]:
            assert 0.0 <= entry["score"] <= 1.0


# ---------------------------------------------------------------------------
# Clip MD5
# ---------------------------------------------------------------------------


class TestClipMD5:
    def test_clip_has_md5(self):
        for clip in app_module.clips.values():
            assert "md5" in clip
            assert isinstance(clip["md5"], str)
            assert len(clip["md5"]) == 32  # MD5 hex string

    def test_md5_matches_wav_bytes(self):
        for clip in app_module.clips.values():
            expected_md5 = hashlib.md5(clip["wav_bytes"]).hexdigest()
            assert clip["md5"] == expected_md5

    def test_different_clips_have_different_md5(self):
        md5_hashes = {clip["md5"] for clip in app_module.clips.values()}
        assert len(md5_hashes) == len(app_module.clips)

    def test_md5_deterministic(self):
        """MD5 should be the same for the same clip across re-init."""
        clip_1_md5 = app_module.clips[1]["md5"]
        old_clips = dict(app_module.clips)

        app_module.clips.clear()
        app_module.init_clips()
        assert app_module.clips[1]["md5"] == clip_1_md5

        # Restore
        app_module.clips.clear()
        app_module.clips.update(old_clips)


# ---------------------------------------------------------------------------
# GET /api/labels/export
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# POST /api/labels/import
# ---------------------------------------------------------------------------


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
        resp = client.post("/api/labels/import", json={"labels": labels})
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


# ---------------------------------------------------------------------------
# GET/POST /api/inclusion
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# POST /api/detector/export
# ---------------------------------------------------------------------------


class TestDetectorExport:
    def test_export_with_sufficient_votes(self, client):
        app_module.good_votes.update({k: None for k in [1, 2, 3]})
        app_module.bad_votes.update({k: None for k in [18, 19, 20]})
        resp = client.post("/api/detector/export")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "weights" in data
        assert "threshold" in data
        assert isinstance(data["weights"], dict)
        assert isinstance(data["threshold"], (int, float))

    def test_export_requires_good_votes(self, client):
        app_module.bad_votes.update({k: None for k in [1, 2]})
        resp = client.post("/api/detector/export")
        assert resp.status_code == 400
        data = resp.get_json()
        assert "need at least one good and one bad vote" in data["error"]

    def test_export_requires_bad_votes(self, client):
        app_module.good_votes.update({k: None for k in [1, 2]})
        resp = client.post("/api/detector/export")
        assert resp.status_code == 400

    def test_export_weights_structure(self, client):
        app_module.good_votes.update({k: None for k in [1, 2]})
        app_module.bad_votes.update({k: None for k in [3, 4]})
        resp = client.post("/api/detector/export")
        data = resp.get_json()
        weights = data["weights"]
        # MLP has 3 layers: Linear(input_dim, 64), ReLU, Linear(64, 1), Sigmoid
        # So we expect 4 keys: 0.weight, 0.bias, 2.weight, 2.bias
        assert "0.weight" in weights
        assert "0.bias" in weights
        assert "2.weight" in weights
        assert "2.bias" in weights


# ---------------------------------------------------------------------------
# POST /api/detector-sort
# ---------------------------------------------------------------------------


class TestDetectorSort:
    def test_sort_with_valid_detector(self, client):
        # First export a detector
        app_module.good_votes.update({k: None for k in [1, 2]})
        app_module.bad_votes.update({k: None for k in [3, 4]})
        export_resp = client.post("/api/detector/export")
        detector = export_resp.get_json()

        # Now use it to sort
        resp = client.post("/api/detector-sort", json={"detector": detector})
        assert resp.status_code == 200
        data = resp.get_json()
        assert "results" in data
        assert "threshold" in data
        assert len(data["results"]) == app_module.NUM_CLIPS

    def test_sort_results_sorted_descending(self, client):
        app_module.good_votes.update({k: None for k in [1, 2]})
        app_module.bad_votes.update({k: None for k in [3, 4]})
        export_resp = client.post("/api/detector/export")
        detector = export_resp.get_json()

        resp = client.post("/api/detector-sort", json={"detector": detector})
        data = resp.get_json()
        scores = [e["score"] for e in data["results"]]
        assert scores == sorted(scores, reverse=True)

    def test_sort_scores_in_valid_range(self, client):
        app_module.good_votes.update({k: None for k in [1, 2]})
        app_module.bad_votes.update({k: None for k in [3, 4]})
        export_resp = client.post("/api/detector/export")
        detector = export_resp.get_json()

        resp = client.post("/api/detector-sort", json={"detector": detector})
        data = resp.get_json()
        for entry in data["results"]:
            assert 0.0 <= entry["score"] <= 1.0

    def test_sort_missing_detector(self, client):
        resp = client.post("/api/detector-sort", json={})
        assert resp.status_code == 400

    def test_sort_missing_weights(self, client):
        resp = client.post("/api/detector-sort", json={"detector": {"threshold": 0.5}})
        assert resp.status_code == 400

    def test_detector_roundtrip(self, client):
        """Export a detector and verify it produces reasonable scores."""
        app_module.good_votes.update({k: None for k in [1, 2, 3]})
        app_module.bad_votes.update({k: None for k in [18, 19, 20]})

        # Export detector
        export_resp = client.post("/api/detector/export")
        detector = export_resp.get_json()

        # Use detector to sort
        resp = client.post("/api/detector-sort", json={"detector": detector})
        data = resp.get_json()
        score_map = {e["id"]: e["score"] for e in data["results"]}

        # Good clips should score higher than bad clips on average
        avg_good = np.mean([score_map[i] for i in app_module.good_votes])
        avg_bad = np.mean([score_map[i] for i in app_module.bad_votes])
        assert avg_good > avg_bad


# ---------------------------------------------------------------------------
# POST /api/example-sort
# ---------------------------------------------------------------------------


class TestExampleSort:
    def test_sort_with_audio_file(self, client):
        # Create a test WAV file in memory
        wav_bytes = app_module.generate_wav(440.0, 1.0)
        data = {"file": (io.BytesIO(wav_bytes), "test.wav")}

        resp = client.post(
            "/api/example-sort", data=data, content_type="multipart/form-data"
        )
        assert resp.status_code == 200
        result_data = resp.get_json()
        assert "results" in result_data
        assert "threshold" in result_data
        assert len(result_data["results"]) == app_module.NUM_CLIPS

    def test_sort_results_sorted_descending(self, client):
        wav_bytes = app_module.generate_wav(440.0, 1.0)
        data = {"file": (io.BytesIO(wav_bytes), "test.wav")}

        resp = client.post(
            "/api/example-sort", data=data, content_type="multipart/form-data"
        )
        result_data = resp.get_json()
        similarities = [e["similarity"] for e in result_data["results"]]
        assert similarities == sorted(similarities, reverse=True)

    def test_sort_similarity_in_valid_range(self, client):
        wav_bytes = app_module.generate_wav(440.0, 1.0)
        data = {"file": (io.BytesIO(wav_bytes), "test.wav")}

        resp = client.post(
            "/api/example-sort", data=data, content_type="multipart/form-data"
        )
        result_data = resp.get_json()
        for entry in result_data["results"]:
            assert -1.0 <= entry["similarity"] <= 1.0

    def test_sort_no_file(self, client):
        resp = client.post("/api/example-sort", data={})
        assert resp.status_code == 400

    def test_sort_empty_filename(self, client):
        data = {"file": (io.BytesIO(b""), "")}
        resp = client.post(
            "/api/example-sort", data=data, content_type="multipart/form-data"
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Dataset endpoints (basic smoke tests)
# ---------------------------------------------------------------------------


class TestDatasetEndpoints:
    def test_get_dataset_status(self, client):
        resp = client.get("/api/dataset/status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "num_clips" in data or "error" in data

    def test_get_dataset_demo_list(self, client):
        resp = client.get("/api/dataset/demo-list")
        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data, dict)
        # Should return available demo datasets
        assert "demos" in data or isinstance(data, dict)

    def test_clear_dataset(self, client):
        resp = client.post("/api/dataset/clear")
        assert resp.status_code == 200
        # After clearing, clips should be empty
        assert len(app_module.clips) == 0

        # Re-initialize for other tests
        app_module.init_clips()


# ---------------------------------------------------------------------------
# Favorite Detectors CRUD
# ---------------------------------------------------------------------------


class TestFavoriteDetectors:
    """Tests for the favorite-detectors management endpoints."""

    @pytest.fixture(autouse=True)
    def clear_favorites(self):
        from vistatotes.utils.state import favorite_detectors

        favorite_detectors.clear()
        yield
        favorite_detectors.clear()

    def _export_detector(self, client):
        """Helper: vote on some clips and export a valid detector payload."""
        app_module.good_votes.update({k: None for k in [1, 2, 3]})
        app_module.bad_votes.update({k: None for k in [18, 19, 20]})
        resp = client.post("/api/detector/export")
        assert resp.status_code == 200
        return resp.get_json()

    def _post_favorite(self, client, name, detector):
        return client.post(
            "/api/favorite-detectors",
            json={
                "name": name,
                "media_type": "audio",
                "weights": detector["weights"],
                "threshold": detector["threshold"],
            },
        )

    # -- GET list --

    def test_get_empty_list(self, client):
        resp = client.get("/api/favorite-detectors")
        assert resp.status_code == 200
        assert resp.get_json()["detectors"] == []

    def test_get_list_after_add(self, client):
        det = self._export_detector(client)
        self._post_favorite(client, "my-detector", det)

        resp = client.get("/api/favorite-detectors")
        data = resp.get_json()
        assert len(data["detectors"]) == 1
        d = data["detectors"][0]
        assert d["name"] == "my-detector"
        assert d["media_type"] == "audio"
        assert "threshold" in d

    # -- POST add --

    def test_add_detector_returns_success(self, client):
        det = self._export_detector(client)
        resp = self._post_favorite(client, "test-det", det)
        assert resp.status_code == 200
        assert resp.get_json()["success"] is True

    def test_add_missing_name_returns_400(self, client):
        det = self._export_detector(client)
        resp = client.post(
            "/api/favorite-detectors",
            json={"media_type": "audio", "weights": det["weights"]},
        )
        assert resp.status_code == 400

    def test_add_missing_media_type_returns_400(self, client):
        det = self._export_detector(client)
        resp = client.post(
            "/api/favorite-detectors",
            json={"name": "test", "weights": det["weights"]},
        )
        assert resp.status_code == 400

    def test_add_missing_weights_returns_400(self, client):
        resp = client.post(
            "/api/favorite-detectors",
            json={"name": "test", "media_type": "audio"},
        )
        assert resp.status_code == 400

    def test_add_multiple_detectors(self, client):
        det = self._export_detector(client)
        self._post_favorite(client, "det-a", det)
        app_module.good_votes.clear()
        app_module.bad_votes.clear()
        self._post_favorite(client, "det-b", det)

        resp = client.get("/api/favorite-detectors")
        names = {d["name"] for d in resp.get_json()["detectors"]}
        assert names == {"det-a", "det-b"}

    def test_add_overwrites_existing_name(self, client):
        det = self._export_detector(client)
        self._post_favorite(client, "dup", det)
        self._post_favorite(client, "dup", det)

        resp = client.get("/api/favorite-detectors")
        assert len(resp.get_json()["detectors"]) == 1

    # -- DELETE --

    def test_delete_detector(self, client):
        det = self._export_detector(client)
        self._post_favorite(client, "to-delete", det)

        resp = client.delete("/api/favorite-detectors/to-delete")
        assert resp.status_code == 200
        assert resp.get_json()["success"] is True

        resp = client.get("/api/favorite-detectors")
        assert resp.get_json()["detectors"] == []

    def test_delete_nonexistent_returns_404(self, client):
        resp = client.delete("/api/favorite-detectors/does-not-exist")
        assert resp.status_code == 404

    # -- RENAME --

    def test_rename_detector(self, client):
        det = self._export_detector(client)
        self._post_favorite(client, "old-name", det)

        resp = client.put(
            "/api/favorite-detectors/old-name/rename",
            json={"new_name": "new-name"},
        )
        assert resp.status_code == 200
        assert resp.get_json()["new_name"] == "new-name"

        names = [d["name"] for d in client.get("/api/favorite-detectors").get_json()["detectors"]]
        assert "new-name" in names
        assert "old-name" not in names

    def test_rename_nonexistent_returns_400(self, client):
        resp = client.put(
            "/api/favorite-detectors/ghost/rename",
            json={"new_name": "anything"},
        )
        assert resp.status_code == 400

    def test_rename_to_existing_name_returns_400(self, client):
        det = self._export_detector(client)
        self._post_favorite(client, "det-a", det)
        app_module.good_votes.clear()
        app_module.bad_votes.clear()
        self._post_favorite(client, "det-b", det)

        resp = client.put(
            "/api/favorite-detectors/det-a/rename",
            json={"new_name": "det-b"},
        )
        assert resp.status_code == 400

    def test_rename_missing_new_name_returns_400(self, client):
        det = self._export_detector(client)
        self._post_favorite(client, "some-det", det)

        resp = client.put(
            "/api/favorite-detectors/some-det/rename",
            json={},
        )
        assert resp.status_code == 400

    # -- import-pkl (detector JSON file) --

    def test_import_pkl_from_detector_json(self, client):
        det = self._export_detector(client)
        json_bytes = json.dumps(det).encode("utf-8")
        data = {
            "file": (io.BytesIO(json_bytes), "detector.json"),
            "name": "imported",
        }
        resp = client.post(
            "/api/favorite-detectors/import-pkl",
            data=data,
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200
        result = resp.get_json()
        assert result["success"] is True
        assert result["name"] == "imported"

    def test_import_pkl_uses_filename_stem_as_default_name(self, client):
        det = self._export_detector(client)
        json_bytes = json.dumps(det).encode("utf-8")
        data = {"file": (io.BytesIO(json_bytes), "my_detector.json")}
        resp = client.post(
            "/api/favorite-detectors/import-pkl",
            data=data,
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200
        assert resp.get_json()["name"] == "my_detector"

    def test_import_pkl_preserves_media_type_from_file(self, client):
        det = self._export_detector(client)
        # Embed explicit media_type in the "file" payload
        det["media_type"] = "image"
        json_bytes = json.dumps(det).encode("utf-8")
        data = {
            "file": (io.BytesIO(json_bytes), "image_detector.json"),
            "name": "img-det",
        }
        resp = client.post(
            "/api/favorite-detectors/import-pkl",
            data=data,
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200
        assert resp.get_json()["media_type"] == "image"

    def test_import_pkl_no_file_returns_400(self, client):
        resp = client.post("/api/favorite-detectors/import-pkl", data={})
        assert resp.status_code == 400

    def test_import_pkl_invalid_format_returns_400(self, client):
        data = {"file": (io.BytesIO(b'{"not_a_detector": true}'), "bad.json")}
        resp = client.post(
            "/api/favorite-detectors/import-pkl",
            data=data,
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400

    # -- Detector data is stored correctly --

    def test_stored_detector_has_correct_fields(self, client):
        det = self._export_detector(client)
        self._post_favorite(client, "field-check", det)

        from vistatotes.utils.state import favorite_detectors

        assert "field-check" in favorite_detectors
        stored = favorite_detectors["field-check"]
        assert stored["name"] == "field-check"
        assert stored["media_type"] == "audio"
        assert "weights" in stored
        assert "threshold" in stored
        assert "created_at" in stored


# ---------------------------------------------------------------------------
# Auto-Detect
# ---------------------------------------------------------------------------


class TestAutoDetect:
    """Tests for POST /api/auto-detect."""

    @pytest.fixture(autouse=True)
    def clear_favorites(self):
        from vistatotes.utils.state import favorite_detectors

        favorite_detectors.clear()
        yield
        favorite_detectors.clear()

    def _add_audio_detector(self, client, name="test-detector"):
        """Helper: create and save an audio detector."""
        app_module.good_votes.update({k: None for k in [1, 2, 3]})
        app_module.bad_votes.update({k: None for k in [18, 19, 20]})
        export_resp = client.post("/api/detector/export")
        assert export_resp.status_code == 200
        detector = export_resp.get_json()

        save_resp = client.post(
            "/api/favorite-detectors",
            json={
                "name": name,
                "media_type": "audio",
                "weights": detector["weights"],
                "threshold": detector["threshold"],
            },
        )
        assert save_resp.status_code == 200
        app_module.good_votes.clear()
        app_module.bad_votes.clear()

    # -- no matching detectors --

    def test_no_favorites_returns_400(self, client):
        resp = client.post("/api/auto-detect")
        assert resp.status_code == 400
        assert "No favorite detectors" in resp.get_json()["error"]

    def test_no_matching_media_type_returns_400(self, client):
        """A detector for a different media type should not match audio clips."""
        app_module.good_votes.update({k: None for k in [1, 2, 3]})
        app_module.bad_votes.update({k: None for k in [18, 19, 20]})
        export_resp = client.post("/api/detector/export")
        detector = export_resp.get_json()
        app_module.good_votes.clear()
        app_module.bad_votes.clear()

        # Save as "image" type — clips are audio, so it won't match
        client.post(
            "/api/favorite-detectors",
            json={
                "name": "image-detector",
                "media_type": "image",
                "weights": detector["weights"],
                "threshold": detector["threshold"],
            },
        )
        resp = client.post("/api/auto-detect")
        assert resp.status_code == 400

    # -- basic success --

    def test_returns_200_with_matching_detector(self, client):
        self._add_audio_detector(client)
        resp = client.post("/api/auto-detect")
        assert resp.status_code == 200

    def test_response_has_required_top_level_fields(self, client):
        self._add_audio_detector(client)
        data = client.post("/api/auto-detect").get_json()
        assert "media_type" in data
        assert "detectors_run" in data
        assert "results" in data

    def test_media_type_matches_clips(self, client):
        self._add_audio_detector(client)
        data = client.post("/api/auto-detect").get_json()
        assert data["media_type"] == "audio"

    def test_detectors_run_count(self, client):
        self._add_audio_detector(client, name="det-1")
        self._add_audio_detector(client, name="det-2")
        data = client.post("/api/auto-detect").get_json()
        assert data["detectors_run"] == 2

    # -- per-detector result structure --

    def test_each_result_has_required_fields(self, client):
        self._add_audio_detector(client, name="struct-check")
        data = client.post("/api/auto-detect").get_json()
        result = data["results"]["struct-check"]
        assert "detector_name" in result
        assert "threshold" in result
        assert "total_hits" in result
        assert "hits" in result

    def test_detector_name_matches_key(self, client):
        self._add_audio_detector(client, name="named-detector")
        data = client.post("/api/auto-detect").get_json()
        result = data["results"]["named-detector"]
        assert result["detector_name"] == "named-detector"

    def test_total_hits_matches_hits_length(self, client):
        self._add_audio_detector(client)
        data = client.post("/api/auto-detect").get_json()
        for result in data["results"].values():
            assert result["total_hits"] == len(result["hits"])

    # -- hit data safety --

    def test_hits_do_not_contain_embeddings(self, client):
        self._add_audio_detector(client)
        data = client.post("/api/auto-detect").get_json()
        for result in data["results"].values():
            for hit in result["hits"]:
                assert "embedding" not in hit

    def test_hits_do_not_contain_wav_bytes(self, client):
        self._add_audio_detector(client)
        data = client.post("/api/auto-detect").get_json()
        for result in data["results"].values():
            for hit in result["hits"]:
                assert "wav_bytes" not in hit

    def test_hits_contain_score(self, client):
        self._add_audio_detector(client)
        data = client.post("/api/auto-detect").get_json()
        for result in data["results"].values():
            for hit in result["hits"]:
                assert "score" in hit
                assert 0.0 <= hit["score"] <= 1.0

    def test_hits_sorted_descending_by_score(self, client):
        self._add_audio_detector(client)
        data = client.post("/api/auto-detect").get_json()
        for result in data["results"].values():
            scores = [h["score"] for h in result["hits"]]
            assert scores == sorted(scores, reverse=True)

    # -- threshold correctness --

    def test_all_hits_score_at_or_above_threshold(self, client):
        self._add_audio_detector(client)
        data = client.post("/api/auto-detect").get_json()
        for result in data["results"].values():
            threshold = result["threshold"]
            for hit in result["hits"]:
                assert hit["score"] >= threshold - 1e-6  # float tolerance


# ---------------------------------------------------------------------------
# Dataset selection at startup: no auto-generated clips
# ---------------------------------------------------------------------------


class TestStartupState:
    """App should start with an empty dataset so the selection screen shows."""

    def test_status_loaded_false_when_clips_empty(self, client):
        """GET /api/dataset/status returns loaded=False when clips is cleared."""
        saved = dict(app_module.clips)
        app_module.clips.clear()
        try:
            resp = client.get("/api/dataset/status")
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["loaded"] is False
            assert data["num_clips"] == 0
        finally:
            app_module.clips.update(saved)

    def test_init_clips_not_called_automatically(self):
        """init_clips() exists for testing but is not called in production startup.

        Verify that the production startup block in app.py does NOT call
        init_clips() – it should only load models and wait for user selection.
        """
        import ast
        import inspect

        source = inspect.getsource(app_module)
        tree = ast.parse(source)

        # Find the else-branch of the top-level if __name__ == '__main__' block
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                # Look for `if __name__ == "__main__"` or nested if/else
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        func = child.func
                        name = (
                            func.id
                            if isinstance(func, ast.Name)
                            else getattr(func, "attr", "")
                        )
                        # init_clips should not appear inside the else branch
                        # We verify it by checking the source of production block
                        # (the else branch starts after the --local check)
        # Simple source-level check: the else branch must not contain init_clips()
        # Extract the else branch text
        main_block_start = source.find('sys.argv[1] == "--local"')
        else_start = source.find("else:", main_block_start)
        assert else_start != -1, "Could not find else branch in __main__ block"
        else_body = source[else_start:]
        assert "init_clips()" not in else_body, (
            "init_clips() must not be called automatically in production startup"
        )


# ---------------------------------------------------------------------------
# Importer registry: icon field & folder/http_archive in extended list
# ---------------------------------------------------------------------------


class TestImporterMetadata:
    """Importer to_dict() must include the icon field."""

    def test_http_archive_display_name(self, client):
        resp = client.get("/api/dataset/importers")
        assert resp.status_code == 200
        data = resp.get_json()
        names = [imp["display_name"] for imp in data["importers"]]
        assert "Generate from HTTP Archive" in names

    def test_http_archive_icon_is_globe(self, client):
        resp = client.get("/api/dataset/importers")
        data = resp.get_json()
        http_imp = next(
            (i for i in data["importers"] if i["name"] == "http_archive"), None
        )
        assert http_imp is not None, "http_archive importer not found"
        assert http_imp["icon"] == "🌐"

    def test_http_archive_supports_tar_and_rar_in_description(self, client):
        resp = client.get("/api/dataset/importers")
        data = resp.get_json()
        http_imp = next(
            (i for i in data["importers"] if i["name"] == "http_archive"), None
        )
        assert http_imp is not None
        desc = http_imp["description"].lower()
        assert "tar" in desc
        assert "rar" in desc

    def test_folder_importer_in_extended_list(self, client):
        """Folder importer must appear in /api/dataset/importers (not a builtin)."""
        resp = client.get("/api/dataset/importers")
        data = resp.get_json()
        names = [imp["name"] for imp in data["importers"]]
        assert "folder" in names

    def test_folder_importer_icon(self, client):
        resp = client.get("/api/dataset/importers")
        data = resp.get_json()
        folder_imp = next(
            (i for i in data["importers"] if i["name"] == "folder"), None
        )
        assert folder_imp is not None
        assert folder_imp["icon"] == "📂"

    def test_folder_importer_description(self, client):
        resp = client.get("/api/dataset/importers")
        data = resp.get_json()
        folder_imp = next(
            (i for i in data["importers"] if i["name"] == "folder"), None
        )
        assert folder_imp is not None
        # Description must not mention specific media-type names
        desc = folder_imp["description"]
        assert "sounds/videos" not in desc
        assert "media files from a folder" in desc.lower()

    def test_all_importers_have_icon_field(self, client):
        resp = client.get("/api/dataset/importers")
        data = resp.get_json()
        for imp in data["importers"]:
            assert "icon" in imp, f"Importer '{imp['name']}' missing icon field"

    def test_pickle_not_in_extended_list(self, client):
        """Pickle importer keeps its dedicated UI and must not appear in the list."""
        resp = client.get("/api/dataset/importers")
        data = resp.get_json()
        names = [imp["name"] for imp in data["importers"]]
        assert "pickle" not in names

    def test_folder_media_type_field_is_first(self, client):
        """Media-type dropdown should come before the path field."""
        resp = client.get("/api/dataset/importers")
        data = resp.get_json()
        folder_imp = next(
            (i for i in data["importers"] if i["name"] == "folder"), None
        )
        assert folder_imp is not None
        keys = [f["key"] for f in folder_imp["fields"]]
        assert keys.index("media_type") < keys.index("path")


# ---------------------------------------------------------------------------
# HTTP Archive: _extract_archive helper
# ---------------------------------------------------------------------------


class TestExtractArchive:
    """Unit tests for the zip/tar extraction helper."""

    from vistatotes.datasets.importers.http_zip import _extract_archive

    def _make_wav_bytes(self) -> bytes:
        """Create a minimal valid WAV file in memory."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(44100)
            samples = struct.pack("<" + "h" * 100, *([0] * 100))
            wf.writeframes(samples)
        return buf.getvalue()

    def test_extract_zip(self, tmp_path):
        from vistatotes.datasets.importers.http_zip import _extract_archive

        wav_data = self._make_wav_bytes()
        zip_path = tmp_path / "test.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("sounds/tone.wav", wav_data)
        extract_dir = tmp_path / "out"
        extract_dir.mkdir()
        _extract_archive(zip_path, extract_dir)
        assert (extract_dir / "sounds" / "tone.wav").exists()

    def test_extract_tar_gz(self, tmp_path):
        from vistatotes.datasets.importers.http_zip import _extract_archive

        wav_data = self._make_wav_bytes()
        tar_path = tmp_path / "test.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tf:
            info = tarfile.TarInfo(name="sounds/tone.wav")
            info.size = len(wav_data)
            tf.addfile(info, io.BytesIO(wav_data))
        extract_dir = tmp_path / "out"
        extract_dir.mkdir()
        _extract_archive(tar_path, extract_dir)
        assert (extract_dir / "sounds" / "tone.wav").exists()

    def test_extract_tar_uncompressed(self, tmp_path):
        from vistatotes.datasets.importers.http_zip import _extract_archive

        wav_data = self._make_wav_bytes()
        tar_path = tmp_path / "test.tar"
        with tarfile.open(tar_path, "w") as tf:
            info = tarfile.TarInfo(name="tone.wav")
            info.size = len(wav_data)
            tf.addfile(info, io.BytesIO(wav_data))
        extract_dir = tmp_path / "out"
        extract_dir.mkdir()
        _extract_archive(tar_path, extract_dir)
        assert (extract_dir / "tone.wav").exists()

    def test_unsupported_format_raises(self, tmp_path):
        from vistatotes.datasets.importers.http_zip import _extract_archive

        bad_archive = tmp_path / "test.7z"
        bad_archive.write_bytes(b"not a real archive")
        extract_dir = tmp_path / "out"
        extract_dir.mkdir()
        with pytest.raises((ValueError, Exception)):
            _extract_archive(bad_archive, extract_dir)

    def test_rar_without_rarfile_raises_runtime_error(self, tmp_path):
        """Attempting RAR extraction without rarfile installed raises RuntimeError."""
        import sys
        import unittest.mock as mock

        from vistatotes.datasets.importers.http_zip import _extract_archive

        rar_path = tmp_path / "test.rar"
        rar_path.write_bytes(b"Rar!\x1a\x07\x00")  # RAR magic bytes (v4)
        extract_dir = tmp_path / "out"
        extract_dir.mkdir()

        with mock.patch.dict(sys.modules, {"rarfile": None}):
            with pytest.raises((RuntimeError, ImportError, Exception)):
                _extract_archive(rar_path, extract_dir)
