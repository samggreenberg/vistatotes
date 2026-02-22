"""Tests for label sorting features: click-time tracking, learned-sort scores,
and the enriched /api/votes response."""

import app as app_module
import vtsearch.utils.state as _state
from vtsearch.utils import vote_click_times, last_learned_scores


class TestClickTimeTracking:
    """Verify that voting via the API assigns monotonically-increasing click times."""

    def test_vote_assigns_click_time(self, client):
        resp = client.post("/api/clips/1/vote", json={"vote": "good"})
        assert resp.status_code == 200
        assert 1 in vote_click_times
        assert vote_click_times[1] == 1

    def test_sequential_votes_increment(self, client):
        client.post("/api/clips/1/vote", json={"vote": "good"})
        client.post("/api/clips/2/vote", json={"vote": "bad"})
        client.post("/api/clips/3/vote", json={"vote": "good"})
        assert vote_click_times[1] == 1
        assert vote_click_times[2] == 2
        assert vote_click_times[3] == 3

    def test_unvote_removes_click_time(self, client):
        client.post("/api/clips/1/vote", json={"vote": "good"})
        assert 1 in vote_click_times
        # Toggle off
        client.post("/api/clips/1/vote", json={"vote": "good"})
        assert 1 not in vote_click_times

    def test_revote_gets_new_click_time(self, client):
        client.post("/api/clips/1/vote", json={"vote": "good"})
        assert vote_click_times[1] == 1
        # Toggle off (does not increment counter)
        client.post("/api/clips/1/vote", json={"vote": "good"})
        # Vote again — should get a new, higher click time
        client.post("/api/clips/1/vote", json={"vote": "good"})
        assert vote_click_times[1] == 2

    def test_switch_vote_gets_new_click_time(self, client):
        client.post("/api/clips/1/vote", json={"vote": "good"})
        assert vote_click_times[1] == 1
        # Switch from good to bad
        client.post("/api/clips/1/vote", json={"vote": "bad"})
        assert vote_click_times[1] == 2
        assert 1 in app_module.bad_votes
        assert 1 not in app_module.good_votes

    def test_imported_labels_have_no_click_time(self, client):
        """Labels added via import should not receive a click time."""
        labels = [{"md5": app_module.clips[1]["md5"], "label": "good"}]
        client.post("/api/labels/import", json={"labels": labels})
        assert 1 in app_module.good_votes
        assert 1 not in vote_click_times


class TestVotesEndpointEnriched:
    """The /api/votes endpoint should include click_times and learned_scores."""

    def test_votes_response_includes_click_times(self, client):
        client.post("/api/clips/1/vote", json={"vote": "good"})
        client.post("/api/clips/2/vote", json={"vote": "bad"})
        resp = client.get("/api/votes")
        data = resp.get_json()
        assert "click_times" in data
        assert data["click_times"]["1"] == 1
        assert data["click_times"]["2"] == 2

    def test_votes_response_includes_learned_scores(self, client):
        # Manually set some learned scores
        _state.last_learned_scores[1] = 0.95
        _state.last_learned_scores[2] = 0.1
        resp = client.get("/api/votes")
        data = resp.get_json()
        assert "learned_scores" in data
        assert data["learned_scores"]["1"] == 0.95
        assert data["learned_scores"]["2"] == 0.1

    def test_votes_response_empty_initially(self, client):
        resp = client.get("/api/votes")
        data = resp.get_json()
        assert data["click_times"] == {}
        assert data["learned_scores"] == {}

    def test_learned_scores_populated_after_learned_sort(self, client):
        """After a learned-sort, /api/votes should include scores for all clips."""
        # Set up votes (need at least one good and one bad)
        app_module.good_votes.update({1: None, 2: None})
        app_module.bad_votes.update({3: None, 4: None})
        # Trigger learned sort
        resp = client.post("/api/learned-sort")
        assert resp.status_code == 200
        sort_data = resp.get_json()
        assert "results" in sort_data

        # Now check /api/votes
        resp = client.get("/api/votes")
        data = resp.get_json()
        assert len(data["learned_scores"]) > 0
        # Every clip should have a score
        for cid in app_module.clips:
            assert str(cid) in data["learned_scores"]


class TestClearVotesResetsState:
    """Clearing votes should also clear click times and learned scores."""

    def test_clear_votes_clears_click_times(self, client):
        client.post("/api/clips/1/vote", json={"vote": "good"})
        assert len(vote_click_times) == 1
        _state.clear_votes()
        assert len(vote_click_times) == 0
        assert _state._click_counter == 0

    def test_clear_votes_clears_learned_scores(self, client):
        _state.last_learned_scores[1] = 0.9
        _state.clear_votes()
        assert len(last_learned_scores) == 0
