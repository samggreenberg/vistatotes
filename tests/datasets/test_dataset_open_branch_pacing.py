"""The dataset-open bar is paced for the branch its coverage step will take.

Opening a saved dataset is two tracker steps: read the pickle, then the coverage
atlas.  The second one **forks** — it restores the atlas cached in the pickle in
~10 ms, or rebuilds a hierarchical k-means at 0.0026 s/item — and #3521 measured
the two 110-700x apart on the same datasets.  A profile cell is
keyed ``(device, media_type, embedder)``, so before #3594 it could hold only one
answer for both, and whichever one it held made the other's bar up to 0.94 of a
bar wrong.

These tests cover the route's half of the fix: it must name the branch to the
lookup once it knows, and remember the answer on the registry entry so the
*first* call — made before the pickle has even been read — is right next time.
The profile-format half is in ``tests_lib/core/test_timing_branches.py``.
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np


def _image_media(media_id: int, rng: np.random.Generator) -> dict[str, Any]:
    """One embedded image media, with a vector distinct enough to cluster."""
    raw = b"\x89PNG" + bytes([media_id]) * 16
    return {
        "id": media_id,
        "media_type": "image",
        "duration": 0,
        "file_size": 2000,
        "md5": f"md5-{media_id}",
        "embeddings": {"siglip": rng.standard_normal(16).astype(np.float32)},
        "embedder": "siglip",
        "media_bytes": raw,
        "media_string": None,
        "media_path": None,
        "filename": f"img_{media_id}.png",
        "category": "test",
        "origin": None,
        "origin_name": f"img_{media_id}.png",
        "width": 32,
        "height": 32,
    }


def _write_dataset(path: Path, medias: dict[int, dict], coverage_atlas: object = None) -> None:
    """Write a dataset container, optionally carrying a cached coverage atlas."""
    from vtscore.datasets.container import write_container

    data: dict[str, Any] = {"medias": medias}
    if coverage_atlas is not None:
        data["coverage_atlas"] = coverage_atlas
    write_container(path, pickle.dumps(data), {"format_version": 1})


def _wait_for_task(task_id: str, timeout: float = 20.0) -> dict | None:
    """Poll the loading-task tracker until it reports ``idle``; return the task."""
    from vtscore.concurrency.progress import loading_tasks

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        task = next((t for t in loading_tasks.list_tasks() if t["task_id"] == task_id), None)
        if task is not None and task["status"] == "idle":
            return task
        time.sleep(0.05)
    return None


class _BranchSpy:
    """Records the ``branch`` every ``dataset_open`` weight lookup was given."""

    def __init__(self, monkeypatch):
        from vtscore import timing

        self.seen: list[str | None] = []
        real = timing.step_weights

        def spy(task: str, **kwargs):
            if task == "dataset_open":
                self.seen.append(kwargs.get("branch"))
            return real(task, **kwargs)

        monkeypatch.setattr(timing, "step_weights", spy)


def _load(client, dataset_id: str) -> dict:
    """Drive one full load through the route and return the finished task."""
    resp = client.post(f"/api/datasets/registry/{dataset_id}/load")
    assert resp.status_code == 200, resp.get_json()
    task = _wait_for_task(resp.get_json()["task_id"])
    assert task is not None, "load never reached idle"
    assert not task.get("error"), task.get("error")
    return task


class TestDatasetOpenBranchPacing:
    def test_a_rebuild_is_named_before_it_starts_and_remembered_after(self, client, monkeypatch):
        """A pickle with no cached atlas rebuilds, and the route says so twice.

        The second ``set_step_weights`` lands *before* ``build_coverage_atlas``
        runs, which is what makes it worth making: that call is the whole cost
        of the branch, and a bar that learned the branch afterwards would have
        already spent the entire rebuild paced for a restore.
        """
        from vtscore.datasets.registry import get_dataset, register_dataset, unregister_dataset
        from vtsearch.settings import get_saved_datasets_dir

        rng = np.random.default_rng(4212)
        medias = {i: _image_media(i, rng) for i in range(1, 13)}
        ds_dir = get_saved_datasets_dir()
        ds_dir.mkdir(parents=True, exist_ok=True)
        pkl_path = ds_dir / "test_open_branch_rebuild.pkl"
        _write_dataset(pkl_path, medias)

        entry = register_dataset(
            name="branch pacing (rebuild)",
            media_type="image",
            num_items=len(medias),
            pkl_path=str(pkl_path),
            embedder="siglip",
            created_by="default",
        )
        dataset_id = entry["id"]
        try:
            spy = _BranchSpy(monkeypatch)
            _load(client, dataset_id)
            assert spy.seen == [None, "rebuilt"], "paced once blind, then again for the branch"

            saved = get_dataset(dataset_id)
            assert saved is not None
            assert saved["coverage_branch"] == "rebuilt"

            # Whether a pickle carries a restorable atlas is a durable fact about
            # the file, so the memo is what makes the *first* call right — the
            # one the route has to make before it has read anything.
            client.post(f"/api/datasets/registry/{dataset_id}/unload")
            spy.seen.clear()
            _load(client, dataset_id)
            assert spy.seen == ["rebuilt", "rebuilt"]
        finally:
            client.post(f"/api/datasets/registry/{dataset_id}/unload")
            unregister_dataset(dataset_id)
            pkl_path.unlink(missing_ok=True)

    def test_a_cached_atlas_restores_and_the_bar_is_told(self, client, monkeypatch):
        """The cheap branch, which is the one a real deployment mostly takes.

        Paced as a rebuild it is the #3521 measurement that started this: the
        pickle read is the only real work, and it crawls across the 15 % of the
        bar the atlas is not using before the job jumps to done.
        """
        from vtscore.coverage.atlas import CoverageAtlas
        from vtscore.datasets.registry import get_dataset, register_dataset, unregister_dataset
        from vtsearch.settings import get_saved_datasets_dir
        from vtsearch.state import DatasetContext, build_coverage_atlas_for_context

        rng = np.random.default_rng(4213)
        medias = {i: _image_media(i, rng) for i in range(1, 13)}

        # Build the atlas the way an import would, then cache it in the pickle.
        ctx = DatasetContext("_atlas_seed")
        ctx.medias.update(medias)
        build_coverage_atlas_for_context(ctx)
        assert isinstance(ctx.coverage_atlas, CoverageAtlas)

        ds_dir = get_saved_datasets_dir()
        ds_dir.mkdir(parents=True, exist_ok=True)
        pkl_path = ds_dir / "test_open_branch_restore.pkl"
        _write_dataset(pkl_path, medias, coverage_atlas=ctx.coverage_atlas.to_serializable())

        entry = register_dataset(
            name="branch pacing (restore)",
            media_type="image",
            num_items=len(medias),
            pkl_path=str(pkl_path),
            embedder="siglip",
            created_by="default",
        )
        dataset_id = entry["id"]
        try:
            spy = _BranchSpy(monkeypatch)
            _load(client, dataset_id)
            assert spy.seen == [None, "restored"]

            saved = get_dataset(dataset_id)
            assert saved is not None
            assert saved["coverage_branch"] == "restored"
        finally:
            client.post(f"/api/datasets/registry/{dataset_id}/unload")
            unregister_dataset(dataset_id)
            pkl_path.unlink(missing_ok=True)
