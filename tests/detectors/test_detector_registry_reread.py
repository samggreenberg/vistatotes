"""The detector registry must re-read when another process writes it (#3627).

`vtscore.datasets.registry` was fixed for this in #3167; its twin here was not.
The cache was filled once and refreshed only by *this* process's mutations, so
every read stayed blind to a sibling writer — a CLI run against the same data
dir, a second server, a hand-edited registry.

It surfaced while clearing six finished slates from a running VTSearch: the file
on disk held nine detectors, ``GET /api/detectors/registry`` went on listing
fifteen, and ``GET /api/detectors`` — which reads the detector files rather than
the registry — said nine. Two views of one dashboard, disagreeing, with the
stale one being the view a person was looking at.

Mutations were never at risk: :func:`_read_modify_write` re-reads under the lock
before mutating, so a stale cache could not clobber a sibling's write. This is
read-path staleness only, which is why nothing failed loudly.
"""

from __future__ import annotations

import importlib
import json

import pytest


@pytest.fixture
def registry(tmp_path, monkeypatch):
    """A detector registry module bound to an empty registry file in tmp_path."""
    monkeypatch.setenv("VTSEARCH_DATA_DIR", str(tmp_path))
    from vtscore.detectors import registry as reg

    importlib.reload(reg)
    reg.REGISTRY_PATH = tmp_path / "detector_registry.json"
    reg._entries = None
    reg._entries_stamp = None
    return reg


def _write(reg, names):
    """Write the registry the way another process would: straight to the file."""
    reg.REGISTRY_PATH.write_text(json.dumps([{"id": f"id{i}", "name": n} for i, n in enumerate(names)]))


def test_a_read_sees_a_write_by_another_process(registry):
    """The bug: the second read returned the first read's cache."""
    _write(registry, ["alpha", "beta"])
    assert {d["name"] for d in registry.list_detectors()} == {"alpha", "beta"}

    _write(registry, ["alpha"])  # a sibling process unregisters `beta`
    assert {d["name"] for d in registry.list_detectors()} == {"alpha"}, (
        "list_detectors served a stale cache after the registry file changed"
    )


def test_a_removal_by_another_process_is_visible(registry):
    """Clearing a finished slate from a CLI must not leave the app listing it."""
    _write(registry, ["done", "todo"])
    registry.list_detectors()
    _write(registry, ["todo"])
    assert registry.find_by_name("done") is None
    assert registry.find_by_name("todo") is not None


def test_an_unchanged_file_is_not_re_parsed(registry):
    """The stamp is what keeps a stat-per-read from becoming a parse-per-read."""
    _write(registry, ["alpha"])
    registry.list_detectors()
    first = registry._entries
    registry.list_detectors()
    assert registry._entries is first, "re-parsed a file that had not changed"


def test_a_missing_registry_reads_as_empty(registry):
    """No file is a legitimate state, not an error, and must not poison the stamp."""
    assert registry.list_detectors() == []
    _write(registry, ["alpha"])
    assert {d["name"] for d in registry.list_detectors()} == {"alpha"}


def test_this_process_own_mutation_stays_visible(registry):
    """`_read_modify_write` swaps the cache; the stamp must match what it wrote."""
    _write(registry, ["alpha"])
    registry.list_detectors()

    def mutate(entries):
        entries.append({"id": "idX", "name": "gamma"})
        return True

    registry._read_modify_write(mutate)
    assert {d["name"] for d in registry.list_detectors()} == {"alpha", "gamma"}


def test_recording_an_embedder_stamps_the_cache_it_swapped_in(registry):
    """The same courtesy `_read_modify_write` does, on the path that runs most.

    ``record_detector_embedder`` is an inline read-modify-write rather than a
    call to the shared helper, and it fires on *every* training cycle. It
    swapped the cache without stamping it, so the write it had just made left
    the stamp behind and the next read re-parsed a file this process already
    held in memory.
    """
    _write(registry, ["alpha"])
    registry.list_detectors()

    registry.record_detector_embedder("id0", "clip-vit-base")
    cached = registry._entries

    registry.list_detectors()
    assert registry._entries is cached, "re-parsed a registry this process had just written"
    assert registry.find_by_name("alpha")["embedder"] == "clip-vit-base"
