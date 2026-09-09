"""A dataset import must feed the generic ``VTSEARCH_TIMING_RECORD`` sink.

``dataset_load`` is declared in the task registry like every other long-running
family, but for a while it was the one family with no ``record_task`` call site:
imports fed only the older, load-specific ``VTSEARCH_PROFILE_LOAD`` profiler, so
an admin who armed the documented recorder got rows for detector loads, text
sorts, and Finds while every import silently wrote nothing (#2845).

These tests drive a real load through ``_run_origin_load_in_background`` on the
calling thread and assert the rows land — and that they land in a shape the
fitter accepts, since rows the fitter rejects are as useless as rows never
written.

Staging imports (``_stage_importer_in_background``, the combine flow's half of
an import) are covered too, under their own ``dataset_stage`` family: they stop
before dedup, the coverage atlas, and the registry write, so recording them as
``dataset_load`` would teach the fit that finalize is free.
"""

from __future__ import annotations

from unittest import mock

import numpy as np

from vtscore.timing.fit import load_rows, normalize_row
from vtscore.timing.recorder import RECORD_ENV_VAR


def _sync_thread_factory():
    """Run the load body inline so the assertions see a finished load."""

    def fake_thread(target, daemon=True, name=None):
        t = mock.MagicMock()
        t.start = lambda: target()
        return t

    return fake_thread


def _fake_load(target_medias, embedder: str = ""):
    """Minimal importer: one already-embedded media, so no model is needed."""
    target_medias[1] = {
        "id": 1,
        "media_type": "audio",
        "duration": 1.0,
        "file_size": 100,
        "md5": "timing-row-md5",
        "embedder": embedder,
        "embedding": np.zeros(8, dtype=np.float32),
        "filename": "fake.wav",
        "category": "unknown",
        "origin": None,
        "origin_name": "fake.wav",
        "media_bytes": None,
        "media_string": None,
        "media_path": None,
    }


def _run_load(tmp_path, **kwargs) -> str:
    """Run one full load synchronously, returning nothing but the task's rows."""
    from vtsearch import settings as settings_mod
    from vtscore.datasets.load_pipeline import _run_origin_load_in_background
    from vtscore.datasets.registry import list_datasets, unregister_dataset

    settings_mod.set_saved_datasets_dir(str(tmp_path / "saved"))
    with mock.patch(
        "vtscore.datasets.load_pipeline.threading.Thread",
        side_effect=_sync_thread_factory(),
    ):
        task_id = _run_origin_load_in_background(
            _fake_load,
            {"importer": "test_timing", "params": {}},
            media_type="audio",
            embedder="clap",
            **kwargs,
        )
    for entry in list_datasets():
        unregister_dataset(entry["id"])
    return task_id


class TestDatasetLoadRecordsTimingRows:
    def test_import_appends_rows_to_the_timing_sink(self, isolated_settings, tmp_path, monkeypatch):
        """The regression itself: an import with the recorder armed writes rows."""
        sink = tmp_path / "timings.jsonl"
        monkeypatch.setenv(RECORD_ENV_VAR, str(sink))

        _run_load(tmp_path)

        assert sink.exists(), "an import with VTSEARCH_TIMING_RECORD armed must write rows (#2845)"
        rows = load_rows([str(sink)])
        assert rows, "the sink was created but empty"
        assert {r["task"] for r in rows} == {"dataset_load"}

    def test_rows_cover_the_declared_steps_and_survive_the_fitter(self, isolated_settings, tmp_path, monkeypatch):
        """Rows the fitter rejects are no better than rows never written, so the
        emitted shape is checked through ``normalize_row`` rather than by eye."""
        sink = tmp_path / "timings.jsonl"
        monkeypatch.setenv(RECORD_ENV_VAR, str(sink))

        _run_load(tmp_path)

        rows = load_rows([str(sink)])
        # A completed load emits every declared step, including the ones it
        # skipped (recorded as a real zero, not dropped).
        assert {r["step"] for r in rows} == {"download", "extract", "load", "embed", "finalize"}
        assert all(r["ok"] for r in rows)
        assert all(r["complete"] for r in rows), "a load that reached finalize must be marked complete"
        assert all(r["media_type"] == "audio" and r["embedder"] == "clap" for r in rows)
        assert all(normalize_row(r) is not None for r in rows), "the fitter must accept every emitted row"

    def test_an_import_that_named_no_embedder_records_the_one_it_used(self, isolated_settings, tmp_path, monkeypatch):
        """#3345: the resolved encoder must reach the row, not the caller's blank.

        The profile is keyed on ``(device, media_type, embedder)``. An import
        that let the media-type default stand recorded ``embedder: ""``, so it
        could only ever populate the media rollup — and the tuning script's own
        ``--drive`` flow names no embedder, which meant the documented way to
        build a profile could never fill an exact ``dataset_load`` cell.
        """
        sink = tmp_path / "timings.jsonl"
        monkeypatch.setenv(RECORD_ENV_VAR, str(sink))

        from vtsearch import settings as settings_mod
        from vtscore.datasets.load_pipeline import _run_origin_load_in_background
        from vtscore.datasets.registry import list_datasets, unregister_dataset

        settings_mod.set_saved_datasets_dir(str(tmp_path / "saved"))
        with mock.patch(
            "vtscore.datasets.load_pipeline.threading.Thread",
            side_effect=_sync_thread_factory(),
        ):
            _run_origin_load_in_background(
                lambda medias: _fake_load(medias, embedder="clap_general"),
                {"importer": "test_timing", "params": {}},
                media_type="audio",
                embedder="",  # the caller names none; the default is resolved later
            )
        for entry in list_datasets():
            unregister_dataset(entry["id"])

        rows = load_rows([str(sink)])
        assert rows
        assert all(r["embedder"] == "clap_general" for r in rows), (
            "the row must name the encoder the load actually used (#3345)"
        )

    def test_download_size_hint_rides_along_for_byte_scaled_steps(self, isolated_settings, tmp_path, monkeypatch):
        """``download``/``extract`` are fit as a per-MB rate, so a load that knows
        its archive size must record it or those two steps go unfittable."""
        sink = tmp_path / "timings.jsonl"
        monkeypatch.setenv(RECORD_ENV_VAR, str(sink))

        _run_load(tmp_path, download_size_mb_hint=42.5)

        rows = load_rows([str(sink)])
        assert rows and all(r["size_mb"] == 42.5 for r in rows)

    def test_failed_import_is_marked_not_ok(self, isolated_settings, tmp_path, monkeypatch):
        """A load that died partway measured an abort, not a cost; the fitter has
        to be able to drop it."""
        sink = tmp_path / "timings.jsonl"
        monkeypatch.setenv(RECORD_ENV_VAR, str(sink))

        from vtsearch import settings as settings_mod
        from vtscore.datasets.load_pipeline import _run_origin_load_in_background

        settings_mod.set_saved_datasets_dir(str(tmp_path / "saved"))
        with mock.patch(
            "vtscore.datasets.load_pipeline.threading.Thread",
            side_effect=_sync_thread_factory(),
        ):
            _run_origin_load_in_background(
                lambda target_medias: None,  # produces nothing -> load error
                {"importer": "test_timing_fail", "params": {}},
                media_type="audio",
            )

        rows = load_rows([str(sink)])
        assert rows, "even a failed load should record what it measured"
        assert all(not r["ok"] for r in rows)
        assert all(normalize_row(r) is None for r in rows), "the fitter must reject a failed run's rows"

    def test_disarmed_recorder_writes_nothing(self, isolated_settings, tmp_path, monkeypatch):
        """The off path stays free: no file, no subscription, no behaviour change."""
        sink = tmp_path / "timings.jsonl"
        monkeypatch.delenv(RECORD_ENV_VAR, raising=False)

        _run_load(tmp_path)

        assert not sink.exists()


class _StubImporter:
    """The smallest thing ``_stage_importer_in_background`` will accept."""

    fields: list = []

    def __init__(self, run=None):
        self._run = run or _fake_load

    def run(self, field_values, target_medias, **kwargs):
        self._run(target_medias)

    def resolve_display_name(self, field_values):
        return "stub staging import"


def _run_staging(tmp_path, monkeypatch, run=None) -> str:
    """Stage one import synchronously, leaving its pkl under *tmp_path*."""
    from vtscore.datasets import load_pipeline
    from vtscore.datasets.load_pipeline import _stage_importer_in_background

    monkeypatch.setattr(load_pipeline, "STAGING_DIR", tmp_path / "staging")
    with mock.patch(
        "vtscore.datasets.load_pipeline.threading.Thread",
        side_effect=_sync_thread_factory(),
    ):
        return _stage_importer_in_background(
            _StubImporter(run),
            {"media_type": "audio", "embedder": "clap"},
        )


class TestStagingImportRecordsTimingRows:
    """The staging half of an import records under its own task family."""

    def test_staging_import_appends_rows(self, isolated_settings, tmp_path, monkeypatch):
        sink = tmp_path / "timings.jsonl"
        monkeypatch.setenv(RECORD_ENV_VAR, str(sink))

        _run_staging(tmp_path, monkeypatch)

        assert sink.exists(), "a staging import with the recorder armed must write rows"
        rows = load_rows([str(sink)])
        assert rows
        assert {r["task"] for r in rows} == {"dataset_stage"}

    def test_rows_cover_the_staging_steps_and_survive_the_fitter(self, isolated_settings, tmp_path, monkeypatch):
        sink = tmp_path / "timings.jsonl"
        monkeypatch.setenv(RECORD_ENV_VAR, str(sink))

        _run_staging(tmp_path, monkeypatch)

        rows = load_rows([str(sink)])
        assert {r["step"] for r in rows} == {"acquire", "embed", "serialize"}
        assert all(r["ok"] and r["complete"] for r in rows)
        assert all(r["n"] == 1 for r in rows), "the staged item count must ride along as the scale variable"
        assert all(r["media_type"] == "audio" and r["embedder"] == "clap" for r in rows)
        assert all(normalize_row(r) is not None for r in rows)

    def test_staging_is_not_recorded_as_a_dataset_load(self, isolated_settings, tmp_path, monkeypatch):
        """Staging skips dedup, the atlas, and the registry write, so folding it
        into ``dataset_load`` would fit a finalize phase that never ran."""
        sink = tmp_path / "timings.jsonl"
        monkeypatch.setenv(RECORD_ENV_VAR, str(sink))

        _run_staging(tmp_path, monkeypatch)

        assert "dataset_load" not in {r["task"] for r in load_rows([str(sink)])}

    def test_failed_staging_is_marked_not_ok(self, isolated_settings, tmp_path, monkeypatch):
        sink = tmp_path / "timings.jsonl"
        monkeypatch.setenv(RECORD_ENV_VAR, str(sink))

        def _boom(target_medias):
            raise RuntimeError("importer exploded")

        _run_staging(tmp_path, monkeypatch, run=_boom)

        rows = load_rows([str(sink)])
        assert rows, "even a failed staging run should record what it measured"
        assert all(not r["ok"] for r in rows)
        assert all(normalize_row(r) is None for r in rows)

    def test_disarmed_recorder_writes_nothing(self, isolated_settings, tmp_path, monkeypatch):
        sink = tmp_path / "timings.jsonl"
        monkeypatch.delenv(RECORD_ENV_VAR, raising=False)

        _run_staging(tmp_path, monkeypatch)

        assert not sink.exists()


class _FakeClock:
    """Deterministic stand-in for the recorder's ``time`` module.

    Patched into :mod:`vtscore.timing.recorder` alone, so the phase boundaries
    under test are the only thing the fake clock decides.
    """

    def __init__(self) -> None:
        self.t = 0.0

    def monotonic(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


class _EmbedsInsideRunImporter:
    """An importer that embeds inside ``run()`` — what every demo source does.

    ``load_demo_dataset`` embeds inside ``load_demo_source``, which the staging
    flow calls from ``importer.run()``; it also signs off with a terminal
    ``"idle"`` that is the importer finishing, not the staging job.
    """

    fields: list = []

    def __init__(self, clock: _FakeClock, acquire: float, embed: float) -> None:
        self._clock, self._acquire, self._embed = clock, acquire, embed

    def run(self, field_values, target_medias, **kwargs):
        from vtscore.concurrency.progress import update_progress

        update_progress("loading", "Reading source…", 0, 1)
        self._clock.advance(self._acquire)
        update_progress("embedding", "Embedding…", 0, 1)
        self._clock.advance(self._embed)
        _fake_load(target_medias)
        update_progress("idle", "Loaded demo dataset", 0, 0)

    def resolve_display_name(self, field_values):
        return "embeds inside run"


class TestStagingRecordsTheImportersEmbeddingAsAnEmbed:
    """#3593: ``dataset_stage``·``embed`` must measure embedding.

    The staging flow bound the importer's progress sink straight to
    ``tracker.update``, so its calls arrived stepless, the tracker kept step 1
    through the whole of ``run()``, and a demo import's embedding was recorded
    as ``acquire``. #3521 §5 cleared the embeddings cache before every rep,
    moved 11-40 s of real embedding into the run, and watched ``embed`` not
    move: the boundary put it there, not the cache.
    """

    def _rows_by_step(self, tmp_path, monkeypatch, *, acquire: float, embed: float) -> dict:
        from vtscore.timing import recorder as recorder_mod

        sink = tmp_path / "timings.jsonl"
        monkeypatch.setenv(RECORD_ENV_VAR, str(sink))
        clock = _FakeClock()
        monkeypatch.setattr(recorder_mod, "time", clock)

        from vtscore.datasets import load_pipeline
        from vtscore.datasets.load_pipeline import _stage_importer_in_background

        monkeypatch.setattr(load_pipeline, "STAGING_DIR", tmp_path / "staging")
        with mock.patch(
            "vtscore.datasets.load_pipeline.threading.Thread",
            side_effect=_sync_thread_factory(),
        ):
            _stage_importer_in_background(
                _EmbedsInsideRunImporter(clock, acquire, embed),
                {"media_type": "audio", "embedder": "clap"},
            )
        return {r["step"]: r for r in load_rows([str(sink)])}

    def test_the_embed_step_carries_the_importers_embedding(self, isolated_settings, tmp_path, monkeypatch):
        rows = self._rows_by_step(tmp_path, monkeypatch, acquire=1.0, embed=9.0)

        assert rows["embed"]["seconds"] >= 9.0, "the embedding inside run() belongs to the embed step"
        assert rows["acquire"]["seconds"] == 1.0, "acquire must measure acquisition and nothing else"

    def test_acquire_no_longer_carries_the_embed_curve(self, isolated_settings, tmp_path, monkeypatch):
        """The shape of the defect, stated as the fit saw it: acquire scaling
        with the work and embed pinned near zero."""
        small = self._rows_by_step(tmp_path, monkeypatch, acquire=1.0, embed=2.0)
        large = self._rows_by_step(tmp_path, monkeypatch, acquire=1.0, embed=20.0)

        assert small["acquire"]["seconds"] == large["acquire"]["seconds"], "acquire must not scale with the embed"
        assert large["embed"]["seconds"] - small["embed"]["seconds"] >= 18.0

    def test_the_run_is_still_complete_and_fittable(self, isolated_settings, tmp_path, monkeypatch):
        """The importer's terminal ``"idle"`` must not park the staging task
        before it serializes, or the run stops reaching its last step and the
        fitter drops every row."""
        rows = self._rows_by_step(tmp_path, monkeypatch, acquire=1.0, embed=9.0)

        assert set(rows) == {"acquire", "embed", "serialize"}
        assert all(r["ok"] and r["complete"] for r in rows.values())
        assert all(normalize_row(r) is not None for r in rows.values())
