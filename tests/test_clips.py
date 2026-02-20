import hashlib
import io
import wave

import numpy as np

import app as app_module


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
        np.testing.assert_array_almost_equal(app_module.clips[1]["embedding"], emb_first)
        # Restore
        app_module.clips.clear()
        app_module.clips.update(old_clips)


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
