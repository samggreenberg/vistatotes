import io
import json
import struct
import wave

import pytest

import app as app_module


@pytest.fixture(autouse=True)
def reset_votes():
    """Reset vote state before each test."""
    app_module.good_votes.clear()
    app_module.bad_votes.clear()


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
            assert len(clip["embedding"]) == app_module.EMBEDDING_DIM

    def test_clip_has_wav_bytes(self):
        for clip in app_module.clips.values():
            assert isinstance(clip["wav_bytes"], bytes)
            assert len(clip["wav_bytes"]) > 0

    def test_file_size_matches_wav_bytes_length(self):
        for clip in app_module.clips.values():
            assert clip["file_size"] == len(clip["wav_bytes"])

    def test_deterministic_with_seed(self):
        """Embeddings should be deterministic because the RNG is seeded."""
        emb_first = app_module.clips[1]["embedding"][:]
        # Re-init and check the same values appear
        old_clips = dict(app_module.clips)
        app_module.clips.clear()
        app_module.init_clips()
        assert app_module.clips[1]["embedding"] == emb_first
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
        app_module.good_votes.update({3, 1, 5})
        resp = client.get("/api/votes")
        data = resp.get_json()
        assert data["good"] == [1, 3, 5]  # sorted

    def test_returns_bad_votes(self, client):
        app_module.bad_votes.update({4, 2})
        resp = client.get("/api/votes")
        data = resp.get_json()
        assert data["bad"] == [2, 4]  # sorted

    def test_returns_both(self, client):
        app_module.good_votes.add(1)
        app_module.bad_votes.add(2)
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
