"""Tests for Origin and LabelSet classes.

Covers:
- Origin serialisation, display, equality, hashing
- LabeledElement serialisation
- LabelSet construction from clips+votes, from results, from dict
- LabelSet roundtrip serialisation
- DatasetImporter.build_origin()
- Clip matching helpers (build_clip_lookup, resolve_clip_ids)
- Integration with label export/import routes
"""

from __future__ import annotations

import app as app_module
from vtsearch.datasets.labelset import LabelSet, LabeledElement
from vtsearch.datasets.origin import Origin
from vtsearch.utils import build_clip_lookup, resolve_clip_ids


# ---------------------------------------------------------------------------
# Origin
# ---------------------------------------------------------------------------


class TestOrigin:
    def test_to_dict(self):
        o = Origin("folder", {"path": "/data", "media_type": "sounds"})
        d = o.to_dict()
        assert d == {"importer": "folder", "params": {"path": "/data", "media_type": "sounds"}}

    def test_from_dict(self):
        d = {"importer": "http_archive", "params": {"url": "https://example.com/data.zip"}}
        o = Origin.from_dict(d)
        assert o.importer == "http_archive"
        assert o.params == {"url": "https://example.com/data.zip"}

    def test_from_dict_missing_params(self):
        d = {"importer": "test"}
        o = Origin.from_dict(d)
        assert o.params == {}

    def test_display_with_params(self):
        o = Origin("folder", {"path": "/data/audio"})
        assert o.display() == "folder(/data/audio)"

    def test_display_without_params(self):
        o = Origin("unknown", {})
        assert o.display() == "unknown"

    def test_equality(self):
        a = Origin("folder", {"path": "/data"})
        b = Origin("folder", {"path": "/data"})
        assert a == b

    def test_inequality(self):
        a = Origin("folder", {"path": "/data"})
        b = Origin("folder", {"path": "/other"})
        assert a != b

    def test_hash_consistency(self):
        a = Origin("folder", {"path": "/data"})
        b = Origin("folder", {"path": "/data"})
        assert hash(a) == hash(b)

    def test_roundtrip(self):
        o = Origin("demo", {"name": "esc50_animals"})
        assert Origin.from_dict(o.to_dict()) == o

    def test_default_params(self):
        o = Origin("test")
        assert o.params == {}
        assert o.display() == "test"


# ---------------------------------------------------------------------------
# LabeledElement
# ---------------------------------------------------------------------------


class TestLabeledElement:
    def test_to_dict_minimal(self):
        e = LabeledElement(md5="abc", label="good")
        d = e.to_dict()
        assert d == {"md5": "abc", "label": "good"}
        # No optional keys when empty
        assert "origin" not in d
        assert "origin_name" not in d
        assert "filename" not in d
        assert "category" not in d

    def test_to_dict_full(self):
        origin = {"importer": "folder", "params": {"path": "/data"}}
        e = LabeledElement(
            md5="abc",
            label="bad",
            origin=origin,
            origin_name="clip.wav",
            filename="clip.wav",
            category="dogs",
        )
        d = e.to_dict()
        assert d["md5"] == "abc"
        assert d["label"] == "bad"
        assert d["origin"] == origin
        assert d["origin_name"] == "clip.wav"
        assert d["filename"] == "clip.wav"
        assert d["category"] == "dogs"

    def test_from_dict(self):
        d = {
            "md5": "xyz",
            "label": "good",
            "origin": {"importer": "demo", "params": {"name": "test"}},
            "origin_name": "test_1.wav",
            "filename": "test_1.wav",
            "category": "birds",
        }
        e = LabeledElement.from_dict(d)
        assert e.md5 == "xyz"
        assert e.label == "good"
        assert e.origin == d["origin"]
        assert e.origin_name == "test_1.wav"

    def test_from_dict_legacy(self):
        """Legacy label format with only md5 and label."""
        d = {"md5": "abc", "label": "bad"}
        e = LabeledElement.from_dict(d)
        assert e.md5 == "abc"
        assert e.label == "bad"
        assert e.origin is None
        assert e.origin_name == ""


# ---------------------------------------------------------------------------
# LabelSet
# ---------------------------------------------------------------------------


class TestLabelSet:
    def test_empty(self):
        ls = LabelSet()
        assert len(ls) == 0
        assert ls.to_dict() == {"labels": []}

    def test_from_elements(self):
        elements = [
            LabeledElement(md5="a", label="good"),
            LabeledElement(md5="b", label="bad"),
        ]
        ls = LabelSet(elements)
        assert len(ls) == 2

    def test_iter(self):
        elements = [LabeledElement(md5="a", label="good")]
        ls = LabelSet(elements)
        items = list(ls)
        assert len(items) == 1
        assert items[0].md5 == "a"

    def test_to_dict(self):
        origin = {"importer": "test", "params": {}}
        elements = [
            LabeledElement(md5="a", label="good", origin=origin, origin_name="a.wav"),
            LabeledElement(md5="b", label="bad"),
        ]
        ls = LabelSet(elements)
        d = ls.to_dict()
        assert len(d["labels"]) == 2
        assert d["labels"][0]["origin"] == origin
        assert d["labels"][0]["origin_name"] == "a.wav"
        # Second element has no origin
        assert "origin" not in d["labels"][1]

    def test_from_dict(self):
        d = {
            "labels": [
                {"md5": "a", "label": "good", "origin": {"importer": "test", "params": {}}},
                {"md5": "b", "label": "bad"},
            ]
        }
        ls = LabelSet.from_dict(d)
        assert len(ls) == 2
        assert ls.elements[0].origin == {"importer": "test", "params": {}}
        assert ls.elements[1].origin is None

    def test_from_dict_skips_non_dicts(self):
        d = {"labels": [{"md5": "a", "label": "good"}, "not a dict", 42]}
        ls = LabelSet.from_dict(d)
        assert len(ls) == 1

    def test_roundtrip(self):
        origin = {"importer": "folder", "params": {"path": "/data"}}
        elements = [
            LabeledElement(
                md5="abc",
                label="good",
                origin=origin,
                origin_name="clip.wav",
                filename="clip.wav",
                category="dogs",
            ),
        ]
        ls = LabelSet(elements)
        d = ls.to_dict()
        ls2 = LabelSet.from_dict(d)
        assert len(ls2) == 1
        assert ls2.elements[0].md5 == "abc"
        assert ls2.elements[0].origin == origin

    def test_from_clips_and_votes(self):
        clips = {
            1: {
                "md5": "hash1",
                "filename": "a.wav",
                "category": "dogs",
                "origin": {"importer": "folder", "params": {"path": "/data"}},
                "origin_name": "a.wav",
            },
            2: {
                "md5": "hash2",
                "filename": "b.wav",
                "category": "cats",
                "origin": {"importer": "folder", "params": {"path": "/data"}},
                "origin_name": "b.wav",
            },
            3: {"md5": "hash3", "filename": "c.wav", "category": "birds"},
        }
        good_votes = {1: None}
        bad_votes = {2: None, 3: None}

        ls = LabelSet.from_clips_and_votes(clips, good_votes, bad_votes)
        assert len(ls) == 3
        assert ls.elements[0].label == "good"
        assert ls.elements[0].origin == {"importer": "folder", "params": {"path": "/data"}}
        assert ls.elements[1].label == "bad"
        # Clip 3 has no origin set
        assert ls.elements[2].origin is None
        # But origin_name falls back to filename
        assert ls.elements[2].origin_name == "c.wav"

    def test_from_clips_and_votes_skips_missing_clips(self):
        clips = {1: {"md5": "hash1", "filename": "a.wav", "category": "x"}}
        good_votes = {1: None, 999: None}
        bad_votes = {}
        ls = LabelSet.from_clips_and_votes(clips, good_votes, bad_votes)
        assert len(ls) == 1

    def test_from_results(self):
        results = {
            "media_type": "audio",
            "detectors_run": 1,
            "results": {
                "detector1": {
                    "detector_name": "detector1",
                    "threshold": 0.5,
                    "total_hits": 2,
                    "hits": [
                        {
                            "id": 1,
                            "filename": "a.wav",
                            "category": "dogs",
                            "score": 0.9,
                            "md5": "hash1",
                            "origin": {"importer": "folder", "params": {"path": "/data"}},
                            "origin_name": "a.wav",
                        },
                        {
                            "id": 2,
                            "filename": "b.wav",
                            "category": "cats",
                            "score": 0.7,
                            "md5": "hash2",
                        },
                    ],
                }
            },
        }
        ls = LabelSet.from_results(results)
        assert len(ls) == 2
        assert ls.elements[0].md5 == "hash1"
        assert ls.elements[0].origin is not None
        assert ls.elements[1].md5 == "hash2"
        assert ls.elements[1].origin is None

    def test_from_results_with_clips_fallback(self):
        """When hits don't have origin, look it up from clips dict."""
        results = {
            "results": {
                "d1": {
                    "hits": [{"id": 1, "filename": "a.wav", "score": 0.9, "md5": "h1"}]
                }
            }
        }
        clips = {
            1: {
                "md5": "h1",
                "filename": "a.wav",
                "origin": {"importer": "folder", "params": {"path": "/tmp"}},
                "origin_name": "a.wav",
            }
        }
        ls = LabelSet.from_results(results, clips=clips)
        assert len(ls) == 1
        assert ls.elements[0].origin is not None
        assert ls.elements[0].origin["importer"] == "folder"


# ---------------------------------------------------------------------------
# DatasetImporter.build_origin()
# ---------------------------------------------------------------------------


class TestBuildOrigin:
    def test_build_origin_from_fields(self):
        from vtsearch.datasets.importers.base import DatasetImporter, ImporterField

        class TestImporter(DatasetImporter):
            name = "test"
            display_name = "Test"
            description = "Test importer."
            fields = [
                ImporterField("path", "Path", "folder"),
                ImporterField("media_type", "Media Type", "select", options=["sounds", "images"]),
                ImporterField("file", "File", "file"),  # Should be excluded
            ]

            def run(self, field_values, clips):
                pass

        imp = TestImporter()
        origin = imp.build_origin({"path": "/data/audio", "media_type": "sounds", "file": "ignored"})
        assert origin == {"importer": "test", "params": {"path": "/data/audio", "media_type": "sounds"}}

    def test_build_origin_excludes_empty_values(self):
        from vtsearch.datasets.importers.base import DatasetImporter, ImporterField

        class TestImporter(DatasetImporter):
            name = "t"
            display_name = "T"
            description = "T"
            fields = [
                ImporterField("path", "Path", "folder"),
                ImporterField("opt", "Opt", "text", required=False),
            ]

            def run(self, field_values, clips):
                pass

        imp = TestImporter()
        origin = imp.build_origin({"path": "/data", "opt": ""})
        assert origin == {"importer": "t", "params": {"path": "/data"}}


# ---------------------------------------------------------------------------
# Integration: clips have origin after init_clips
# ---------------------------------------------------------------------------


class TestClipOrigins:
    def test_init_clips_have_origin(self):
        """Every clip created by init_clips should have origin and origin_name."""
        for cid, clip in app_module.clips.items():
            assert "origin" in clip, f"Clip {cid} missing origin"
            assert "origin_name" in clip, f"Clip {cid} missing origin_name"
            assert clip["origin"]["importer"] == "test"
            assert clip["origin_name"] == clip["filename"]

    def test_init_clips_have_filename(self):
        for cid, clip in app_module.clips.items():
            assert "filename" in clip, f"Clip {cid} missing filename"
            assert clip["filename"].startswith("test_clip_")


# ---------------------------------------------------------------------------
# Integration: label export includes origin
# ---------------------------------------------------------------------------


class TestLabelExportOrigin:
    def test_export_includes_origin(self, client):
        app_module.good_votes[1] = None
        resp = client.get("/api/labels/export")
        data = resp.get_json()
        assert len(data["labels"]) == 1
        entry = data["labels"][0]
        assert "md5" in entry
        assert "label" in entry
        assert "origin" in entry
        assert "origin_name" in entry
        assert entry["origin"]["importer"] == "test"

    def test_export_roundtrip_with_origin(self, client):
        """Export labels with origin, re-import via legacy route, verify match."""
        app_module.good_votes.update({k: None for k in [1, 3]})
        app_module.bad_votes.update({k: None for k in [2]})
        resp = client.get("/api/labels/export")
        exported = resp.get_json()

        # Verify origin info is present
        for entry in exported["labels"]:
            assert "origin" in entry

        # Clear and re-import
        app_module.good_votes.clear()
        app_module.bad_votes.clear()
        resp = client.post("/api/labels/import", json=exported)
        data = resp.get_json()
        assert data["applied"] == 3
        assert set(app_module.good_votes) == {1, 3}
        assert set(app_module.bad_votes) == {2}

    def test_export_backwards_compatible(self, client):
        """Exported labels still have md5 and label keys for legacy consumers."""
        app_module.good_votes[1] = None
        resp = client.get("/api/labels/export")
        data = resp.get_json()
        entry = data["labels"][0]
        assert "md5" in entry
        assert "label" in entry
        assert entry["label"] == "good"

    def test_labelset_from_dict_legacy_format(self):
        """LabelSet.from_dict handles legacy label format (md5+label only)."""
        legacy_data = {
            "labels": [
                {"md5": "abc", "label": "good"},
                {"md5": "def", "label": "bad"},
            ]
        }
        ls = LabelSet.from_dict(legacy_data)
        assert len(ls) == 2
        assert ls.elements[0].origin is None
        assert ls.elements[0].md5 == "abc"


# ---------------------------------------------------------------------------
# Clip matching helpers
# ---------------------------------------------------------------------------


class TestBuildClipLookup:
    def test_origin_lookup_populated(self):
        clips = {
            1: {"id": 1, "md5": "h1", "origin": {"importer": "folder", "params": {"path": "/a"}}, "origin_name": "a.wav"},
            2: {"id": 2, "md5": "h2", "origin": {"importer": "folder", "params": {"path": "/a"}}, "origin_name": "b.wav"},
        }
        origin_lookup, md5_lookup = build_clip_lookup(clips)
        assert len(origin_lookup) == 2
        assert len(md5_lookup) == 2

    def test_md5_lookup_groups_duplicates(self):
        """Two clips with the same MD5 should both appear in the md5_lookup."""
        clips = {
            1: {"id": 1, "md5": "same_hash", "origin": {"importer": "folder", "params": {}}, "origin_name": "a.wav"},
            2: {"id": 2, "md5": "same_hash", "origin": {"importer": "folder", "params": {}}, "origin_name": "b.wav"},
        }
        _, md5_lookup = build_clip_lookup(clips)
        assert sorted(md5_lookup["same_hash"]) == [1, 2]

    def test_clips_without_origin_only_in_md5_lookup(self):
        clips = {
            1: {"id": 1, "md5": "h1"},
        }
        origin_lookup, md5_lookup = build_clip_lookup(clips)
        assert len(origin_lookup) == 0
        assert md5_lookup["h1"] == [1]


class TestResolveClipIds:
    def _make_lookups(self):
        clips = {
            1: {"id": 1, "md5": "h1", "origin": {"importer": "folder", "params": {"path": "/a"}}, "origin_name": "a.wav"},
            2: {"id": 2, "md5": "h2", "origin": {"importer": "folder", "params": {"path": "/a"}}, "origin_name": "b.wav"},
            3: {"id": 3, "md5": "h1", "origin": {"importer": "folder", "params": {"path": "/b"}}, "origin_name": "a.wav"},
        }
        return build_clip_lookup(clips)

    def test_match_by_origin(self):
        origin_lookup, md5_lookup = self._make_lookups()
        entry = {"md5": "wrong", "origin": {"importer": "folder", "params": {"path": "/a"}}, "origin_name": "a.wav"}
        ids = resolve_clip_ids(entry, origin_lookup, md5_lookup)
        assert ids == [1]

    def test_fallback_to_md5(self):
        origin_lookup, md5_lookup = self._make_lookups()
        entry = {"md5": "h2"}
        ids = resolve_clip_ids(entry, origin_lookup, md5_lookup)
        assert ids == [2]

    def test_md5_fallback_returns_all_duplicates(self):
        """When falling back to MD5, all clips with that hash are returned."""
        origin_lookup, md5_lookup = self._make_lookups()
        entry = {"md5": "h1"}
        ids = resolve_clip_ids(entry, origin_lookup, md5_lookup)
        assert sorted(ids) == [1, 3]

    def test_no_match_returns_empty(self):
        origin_lookup, md5_lookup = self._make_lookups()
        entry = {"md5": "nonexistent"}
        ids = resolve_clip_ids(entry, origin_lookup, md5_lookup)
        assert ids == []

    def test_union_of_origin_and_md5(self):
        """Origin and MD5 matches are unioned: all matching clips are returned."""
        origin_lookup, md5_lookup = self._make_lookups()
        # Origin matches clip 1, md5 "h1" matches clips 1 and 3 → union is [1, 3]
        entry = {"md5": "h1", "origin": {"importer": "folder", "params": {"path": "/a"}}, "origin_name": "a.wav"}
        ids = resolve_clip_ids(entry, origin_lookup, md5_lookup)
        assert sorted(ids) == [1, 3]
