import hashlib
import io
import wave

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
