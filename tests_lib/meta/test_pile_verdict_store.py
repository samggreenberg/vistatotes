"""The durable store for human answers, and the write that shares a file (#3729).

The pile is purgeable by design and every cell is rebuildable; a verdict is
neither. `verdict_store.py` is what keeps a copy of the un-rebuildable half in
the repository, and everything worth testing in it is a property that decides
whether a divergence is *noticed*:

* canonicalisation, because a check that reports a divergence every time a dict
  is reordered is a check people stop reading;
* the stored name, because two roots hold files with the same basename and a
  flat store keyed on one of them drops the other silently -- the failure this
  script exists to prevent, turned inward. The first export refused to run for
  exactly this reason;
* the row delta, because "how many judgements moved" is the question a person
  has when the check fires, and a hash cannot answer it.

`write_json_locked` gets a test for the half that is checkable in-process: the
file is replaced rather than truncated, so a reader never sees half of it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_PILE_DIR = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "pile"


@pytest.fixture(scope="module")
def store():
    """``verdict_store``, which does no env setup at import."""
    if str(_PILE_DIR) not in sys.path:
        sys.path.insert(0, str(_PILE_DIR))
    import verdict_store

    return verdict_store


@pytest.fixture(scope="module")
def corrections_mod():
    if str(_PILE_DIR) not in sys.path:
        sys.path.insert(0, str(_PILE_DIR))
    from pilebuild import corrections

    return corrections


@pytest.fixture(scope="module")
def pc():
    if str(_PILE_DIR) not in sys.path:
        sys.path.insert(0, str(_PILE_DIR))
    import pile_config

    return pile_config


def _row(iid: int, cls: str, present: bool = True) -> dict:
    return {"image_id": iid, "class": cls, "present": present, "source": "human_review"}


class TestCanonicalise:
    def test_row_order_and_key_order_do_not_change_the_bytes(self, store):
        """Two files holding the same answers must produce the same store entry."""
        a = json.dumps([_row(2, "bus"), _row(1, "car")])
        b = json.dumps([{"source": "human_review", "present": True, "class": "car", "image_id": 1}, _row(2, "bus")])

        assert store.canonicalise(a) == store.canonicalise(b)

    def test_a_changed_answer_does_change_the_bytes(self, store):
        a = json.dumps([_row(1, "car", present=True)])
        b = json.dumps([_row(1, "car", present=False)])

        assert store.canonicalise(a) != store.canonicalise(b)

    def test_a_non_json_artifact_passes_through_untouched(self, store):
        """The slate manifests are CSV; rewriting them would be an unverifiable change."""
        text = "image_id,class,cell\n7,bus,bus@small\n"

        assert store.canonicalise(text, ".csv") == text

    def test_a_file_of_unkeyed_rows_still_sorts_deterministically(self, store):
        """Canonicalising has to work on shapes this has never seen."""
        one = store.canonicalise(json.dumps([{"b": 2}, {"a": 1}]))
        two = store.canonicalise(json.dumps([{"a": 1}, {"b": 2}]))

        assert one == two


class TestStoredName:
    def test_the_root_and_the_relative_path_both_survive(self, store, pc):
        art = pc.HumanArtifact("{WORK}/verdicts_*.json", "human", "why")

        assert store.stored_name(art, Path("verdicts_20260820b.json")) == "WORK__verdicts_20260820b.json"

    def test_two_roots_holding_one_basename_do_not_collide(self, store, pc):
        """Measured: the same snapshot sits on scratch and on `/exp`."""
        work = pc.HumanArtifact("{WORK}/verdicts_*.json", "human", "why")
        exp = pc.HumanArtifact("{EXP}/verdicts_snapshot_*.json", "human", "why")
        rel = Path("verdicts_snapshot_20260820.json")

        assert store.stored_name(work, rel) != store.stored_name(exp, rel)

    def test_two_slate_dirs_holding_one_manifest_do_not_collide(self, store, pc):
        art = pc.HumanArtifact("{WORK}/slates*/*/manifest.csv", "support", "why")

        first = store.stored_name(art, Path("slates/bus/manifest.csv"))
        second = store.stored_name(art, Path("slates/car/manifest.csv"))

        assert first != second
        assert "/" not in first


class TestRowDelta:
    def test_it_counts_what_moved(self, store):
        committed = json.dumps([_row(1, "car"), _row(2, "bus")])
        working = json.dumps([_row(1, "car", present=False), _row(3, "dog")])

        assert store.row_delta(committed, working) == (1, 1, 1)

    def test_it_declines_a_shape_it_cannot_key(self, store):
        """Better to say 'bytes differ' than to invent a row count."""
        assert store.row_delta(json.dumps({"a": 1}), json.dumps({"a": 2})) is None


class TestHumanArtifact:
    def test_a_glob_finds_every_match_under_its_root(self, pc, tmp_path, monkeypatch):
        (tmp_path / "verdicts_a.json").write_text("[]")
        (tmp_path / "verdicts_b.json").write_text("[]")
        (tmp_path / "unrelated.txt").write_text("x")
        monkeypatch.setitem(pc.RECORD_ROOTS, "WORK", tmp_path)

        found = pc.HumanArtifact("{WORK}/verdicts_*.json", "human", "why").resolve()

        assert sorted(p.name for p in found) == ["verdicts_a.json", "verdicts_b.json"]

    def test_a_missing_root_is_empty_rather_than_an_error(self, pc, tmp_path, monkeypatch):
        """Every machine has a different subset; a laptop has none of it."""
        monkeypatch.setitem(pc.RECORD_ROOTS, "WORK", tmp_path / "nope")

        assert pc.HumanArtifact("{WORK}/verdicts_*.json", "human", "why").resolve() == []

    def test_every_declared_row_names_a_real_root(self, pc):
        """A typo in a root token would silently record nothing."""
        for art in pc.HUMAN_RECORD:
            assert art.root_token() in pc.RECORD_ROOTS, art.source

    def test_every_declared_row_says_why_it_is_kept(self, pc):
        """The reason has to travel with the row; that is the whole point of #3729."""
        for art in pc.HUMAN_RECORD:
            assert art.tier in ("human", "support", "derived"), art.source
            assert len(art.why) > 20, art.source


class TestDroppedRows:
    """The guard that stops a re-run deleting rows nobody can regenerate (#3727).

    Measured on the live file: `verdicts_to_corrections.py` run with its own
    committed defaults reproduces 488 of 640 rows, and no invocation anyone
    could reconstruct reproduces all of them. So a well-meant re-run -- exactly
    what someone does to pick up the new confirmation rows -- is a silent
    deletion of 384 human judgements unless something refuses.
    """

    def test_it_counts_the_lost_rows_by_source(self, corrections_mod):
        old = [_row(1, "car"), _row(2, "bus"), _row(3, "dog")]
        old[2]["source"] = "claude_triage"
        new = [_row(1, "car")]

        assert corrections_mod.dropped_rows(old, new) == {"human_review": 1, "claude_triage": 1}

    def test_a_superset_drops_nothing(self, corrections_mod):
        """Adding the confirmation rows is what a good re-run looks like."""
        old = [_row(1, "car")]
        new = [_row(1, "car"), _row(2, "bus")]

        assert corrections_mod.dropped_rows(old, new) == {}

    def test_a_changed_row_is_not_a_dropped_one(self, corrections_mod):
        """The pair still has an answer; only its content moved."""
        old = [_row(1, "car", present=True)]
        new = [_row(1, "car", present=False)]

        assert corrections_mod.dropped_rows(old, new) == {}

    def test_string_and_int_ids_are_the_same_pair(self, corrections_mod):
        """JSON has made these strings before; a false drop would block a good run."""
        old = [{"image_id": "1", "class": "car", "source": "human_review"}]

        assert corrections_mod.dropped_rows(old, [_row(1, "car")]) == {}


class TestWriteJsonLocked:
    def test_the_file_is_replaced_not_truncated(self, corrections_mod, tmp_path):
        """A reader arriving mid-write must never see half a verdict file."""
        path = tmp_path / "corrections.json"
        path.write_text(json.dumps([_row(1, "car")]))
        seen: list[int] = []

        real_replace = corrections_mod.os.replace

        def watching_replace(src, dst):
            # The instant before the swap, the old file must still be whole.
            seen.append(len(json.loads(Path(dst).read_text())))
            return real_replace(src, dst)

        corrections_mod.os.replace = watching_replace
        try:
            corrections_mod.write_json_locked(path, [_row(1, "car"), _row(2, "bus")])
        finally:
            corrections_mod.os.replace = real_replace

        assert seen == [1]
        assert len(json.loads(path.read_text())) == 2

    def test_it_creates_the_directory_and_leaves_no_temp_behind(self, corrections_mod, tmp_path):
        path = tmp_path / "new" / "corrections.json"

        corrections_mod.write_json_locked(path, [_row(1, "car")])

        assert json.loads(path.read_text()) == [_row(1, "car")]
        assert not list(path.parent.glob("*.tmp"))
