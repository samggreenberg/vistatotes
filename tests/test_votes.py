import app as app_module
from vistatotes.models.progress import _cache_good_ids, _cache_bad_ids, _ensure_cache
from vistatotes.utils import clips, label_history


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


class TestLabelHistory:
    def test_vote_adds_history_entry(self, client):
        client.post("/api/clips/1/vote", json={"vote": "good"})
        assert len(label_history) == 1
        assert label_history[0][0] == 1
        assert label_history[0][1] == "good"

    def test_toggle_off_adds_unlabel_history(self, client):
        """Toggling off a vote should record an 'unlabel' event."""
        client.post("/api/clips/1/vote", json={"vote": "good"})
        client.post("/api/clips/1/vote", json={"vote": "good"})
        assert len(label_history) == 2
        assert label_history[1][1] == "unlabel"

    def test_toggle_off_bad_adds_unlabel_history(self, client):
        client.post("/api/clips/1/vote", json={"vote": "bad"})
        client.post("/api/clips/1/vote", json={"vote": "bad"})
        assert len(label_history) == 2
        assert label_history[1][1] == "unlabel"

    def test_switch_vote_adds_new_label_history(self, client):
        """Switching good->bad should add a 'bad' entry, not 'unlabel'."""
        client.post("/api/clips/1/vote", json={"vote": "good"})
        client.post("/api/clips/1/vote", json={"vote": "bad"})
        assert len(label_history) == 2
        assert label_history[0][1] == "good"
        assert label_history[1][1] == "bad"

    def test_toggle_off_then_revote(self, client):
        """Toggle off then revote should produce 3 history entries."""
        client.post("/api/clips/1/vote", json={"vote": "good"})
        client.post("/api/clips/1/vote", json={"vote": "good"})  # toggle off
        client.post("/api/clips/1/vote", json={"vote": "bad"})   # revote bad
        assert len(label_history) == 3
        assert label_history[0][1] == "good"
        assert label_history[1][1] == "unlabel"
        assert label_history[2][1] == "bad"
        assert 1 in app_module.bad_votes
        assert 1 not in app_module.good_votes


class TestProgressCacheWithLabelChanges:
    """Verify the progress cache stays consistent when labels are changed."""

    def test_cache_removes_clip_on_unlabel(self, client):
        """After toggling off, the progress cache should not include the clip."""
        client.post("/api/clips/1/vote", json={"vote": "good"})
        client.post("/api/clips/2/vote", json={"vote": "bad"})
        _ensure_cache(clips, label_history, 0)
        assert 1 in _cache_good_ids
        assert 2 in _cache_bad_ids

        # Toggle off clip 1
        client.post("/api/clips/1/vote", json={"vote": "good"})
        _ensure_cache(clips, label_history, 0)
        assert 1 not in _cache_good_ids
        assert 1 not in _cache_bad_ids
        assert 2 in _cache_bad_ids

    def test_cache_handles_switch_vote(self, client):
        """Switching good->bad should update cache running sets correctly."""
        client.post("/api/clips/1/vote", json={"vote": "good"})
        client.post("/api/clips/2/vote", json={"vote": "bad"})
        _ensure_cache(clips, label_history, 0)
        assert 1 in _cache_good_ids

        # Switch clip 1 from good to bad
        client.post("/api/clips/1/vote", json={"vote": "bad"})
        _ensure_cache(clips, label_history, 0)
        assert 1 not in _cache_good_ids
        assert 1 in _cache_bad_ids

    def test_cache_toggle_off_then_revote(self, client):
        """Toggle off then revote should leave cache in correct state."""
        client.post("/api/clips/1/vote", json={"vote": "good"})
        client.post("/api/clips/2/vote", json={"vote": "bad"})
        # Toggle off clip 1
        client.post("/api/clips/1/vote", json={"vote": "good"})
        # Revote as bad
        client.post("/api/clips/1/vote", json={"vote": "bad"})
        _ensure_cache(clips, label_history, 0)
        assert 1 in _cache_bad_ids
        assert 1 not in _cache_good_ids

    def test_learned_sort_after_toggle_off(self, client):
        """Learned sort should work after toggling off a vote."""
        client.post("/api/clips/1/vote", json={"vote": "good"})
        client.post("/api/clips/2/vote", json={"vote": "bad"})
        resp = client.post("/api/learned-sort")
        assert resp.status_code == 200

        # Toggle off good vote, add a different good vote
        client.post("/api/clips/1/vote", json={"vote": "good"})
        client.post("/api/clips/3/vote", json={"vote": "good"})
        resp = client.post("/api/learned-sort")
        assert resp.status_code == 200

    def test_learned_sort_returns_400_after_toggling_all_good(self, client):
        """If all good votes are toggled off, learned sort should return 400."""
        client.post("/api/clips/1/vote", json={"vote": "good"})
        client.post("/api/clips/2/vote", json={"vote": "bad"})
        # Toggle off the only good vote
        client.post("/api/clips/1/vote", json={"vote": "good"})
        resp = client.post("/api/learned-sort")
        assert resp.status_code == 400

    def test_labeling_status_after_label_change(self, client):
        """labeling-status endpoint should not crash after label changes."""
        # Vote enough clips to get past the minimum threshold
        for i in range(1, 6):
            client.post(f"/api/clips/{i}/vote", json={"vote": "good"})
        for i in range(6, 11):
            client.post(f"/api/clips/{i}/vote", json={"vote": "bad"})

        resp = client.get("/api/labeling-status")
        assert resp.status_code == 200

        # Now toggle off a good vote and switch a bad vote
        client.post("/api/clips/1/vote", json={"vote": "good"})   # toggle off
        client.post("/api/clips/6/vote", json={"vote": "good"})   # switch bad->good

        resp = client.get("/api/labeling-status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["good_count"] == 5  # lost 1, gained 1
        assert data["bad_count"] == 4   # lost 1
