"""Tests for the timing recorder and the fit that turns its rows into a profile.

The recorder and the fitter are the two halves of the tuning loop, so they are
tested together and end-to-end: rows recorded from a simulated task must fit into
a profile that the reader then loads and paces with. If that round trip works,
an admin's sweep works.
"""

import json

import math

import pytest

from vtscore import timing
from vtscore.concurrency.progress import PROGRESS_COMMON_EXTRAS, ProgressTracker
from vtscore.timing import profile as timing_profile
from vtscore.timing import recorder as timing_recorder
from vtscore.timing.fit import (
    affine_fit,
    coverage_report,
    device_key,
    fit_profile,
    fit_step,
    load_rows,
    normalize_row,
)
from vtscore.timing.profile import StepCoeffs
from vtscore.timing.recorder import RECORD_ENV_VAR, record_task, recording_enabled


def _row(raw: dict) -> dict:
    """``normalize_row`` narrowed to non-``None`` (its reject return)."""
    row = normalize_row(raw)
    assert row is not None, f"expected {raw} to normalize"
    return row


def _fit(samples: list[dict], *, byte_scaled: bool) -> StepCoeffs:
    """``fit_step`` narrowed to non-``None`` (its no-samples return)."""
    coeffs = fit_step(samples, byte_scaled)
    assert coeffs is not None
    return coeffs


def _weights(task: str, **kwargs) -> list[float]:
    """``step_weights`` narrowed to non-``None`` (its no-coverage return)."""
    weights = timing.step_weights(task, **kwargs)
    assert weights is not None, f"expected weights for {task}"
    return weights


def _tracker() -> ProgressTracker:
    return ProgressTracker(extra_fields=dict(PROGRESS_COMMON_EXTRAS))


@pytest.fixture
def sink(tmp_path, monkeypatch):
    """Arm the recorder at a temp JSONL and yield a reader for what it wrote.

    Reading goes through :func:`load_rows`, so every test that inspects recorded
    output also exercises the path the tuning script takes.
    """
    path = tmp_path / "timings.jsonl"
    monkeypatch.setenv(RECORD_ENV_VAR, str(path))

    def read():
        return load_rows([str(path)]) if path.exists() else []

    read.path = str(path)
    return read


class TestRecorderArming:
    def test_disarmed_by_default(self, monkeypatch):
        monkeypatch.delenv(RECORD_ENV_VAR, raising=False)
        assert not recording_enabled()
        # The no-op stand-in must accept the whole surface without a tracker
        # ever being subscribed, so call sites need no branching.
        tracker = _tracker()
        with record_task(tracker, "text_sort") as rec:
            rec.set_scale(n=10)
        rec.finish(ok=True)
        assert tracker._subscribers == []

    def test_armed_recorder_unsubscribes_on_finish(self, sink, monkeypatch):
        tracker = _tracker()
        rec = record_task(tracker, "text_sort")
        rec.start()
        assert len(tracker._subscribers) == 1
        rec.finish()
        assert tracker._subscribers == []


class TestRecordedRows:
    def _run_text_sort(self, tracker, rec, *, skip_model_load=False):
        if not skip_model_load:
            tracker.update("sorting", "loading", 0, 0, step=1, total_steps=3)
        tracker.update("sorting", "embedding", 0, 0, step=2, total_steps=3)
        tracker.update("sorting", "scoring", 0, 0, step=3, total_steps=3)
        rec.set_scale(n=1234)

    def test_one_row_per_declared_step(self, sink):
        tracker = _tracker()
        with record_task(tracker, "text_sort", media_type="image", embedder="siglip") as rec:
            self._run_text_sort(tracker, rec)
        rows = sink()
        assert [r["step"] for r in rows] == ["load_model", "embed_query", "score"]
        assert all(r["task"] == "text_sort" and r["n"] == 1234 for r in rows)
        assert all(r["ok"] and r["complete"] for r in rows)

    def test_a_skipped_step_is_recorded_as_zero_not_dropped(self, sink):
        # A warm text sort never enters load_model because the encoder is
        # already resident. That is a real measurement of zero, and dropping it
        # would leave the fit permanently over-budgeting the phase.
        tracker = _tracker()
        with record_task(tracker, "text_sort") as rec:
            self._run_text_sort(tracker, rec, skip_model_load=True)
        rows = {r["step"]: r for r in sink()}
        assert set(rows) == {"load_model", "embed_query", "score"}
        assert rows["load_model"]["seconds"] == 0.0
        assert rows["load_model"]["complete"] is True

    def test_a_run_that_stops_early_is_marked_incomplete(self, sink):
        tracker = _tracker()
        rec = record_task(tracker, "text_sort")
        rec.start()
        tracker.update("sorting", "loading", 0, 0, step=1, total_steps=3)
        rec.finish(ok=False)
        rows = sink()
        assert rows and all(not r["complete"] and not r["ok"] for r in rows)
        assert [r["step"] for r in rows] == ["load_model"]

    def test_exception_inside_the_context_marks_the_run_bad(self, sink):
        tracker = _tracker()
        with pytest.raises(RuntimeError):
            with record_task(tracker, "find") as rec:
                tracker.update("running", "prepare", 0, 0, step=1, total_steps=3)
                rec.set_scale(n=5)
                raise RuntimeError("boom")
        assert all(not r["ok"] for r in sink())

    def test_auto_finish_closes_on_a_terminal_status(self, sink):
        # Singleton trackers (sort_progress, find_progress) end by parking at
        # "idle" on every exit path, which is a far more reliable end-of-task
        # signal than a finally on a route handler full of aborts.
        tracker = _tracker()
        rec = record_task(tracker, "find", auto_finish=True)
        rec.start()
        for step in (1, 2, 3):
            tracker.update("running", "x", 0, 0, step=step, total_steps=3)
        rec.set_scale(n=99)
        tracker.update("idle", "", 0, 0, step=None, total_steps=None)
        assert tracker._subscribers == []  # closed itself
        rows = sink()
        assert [r["step"] for r in rows] == ["prepare", "load", "score"]
        assert all(r["ok"] and r["complete"] for r in rows)

    def test_auto_finish_marks_an_errored_end_bad(self, sink):
        tracker = _tracker()
        rec = record_task(tracker, "find", auto_finish=True)
        rec.start()
        tracker.update("running", "x", 0, 0, step=1, total_steps=3)
        tracker.update("idle", "", 0, 0, step=None, total_steps=None, error="Cancelled")
        assert all(not r["ok"] for r in sink())

    def test_status_phases_disambiguate_a_shared_step(self, sink):
        # dataset_load's step 1 is both the transfer and the unpack; only the
        # status tells them apart.
        tracker = _tracker()
        with record_task(tracker, "dataset_load", status_phases={"extracting": "extract"}) as rec:
            tracker.update("downloading", "", 0, 0, step=1, total_steps=4)
            tracker.update("extracting", "", 0, 0, step=1, total_steps=4)
            tracker.update("loading", "", 0, 0, step=2, total_steps=4)
            tracker.update("embedding", "", 0, 0, step=3, total_steps=4)
            tracker.update("finalizing", "", 0, 0, step=4, total_steps=4)
            rec.set_scale(n=500, size_mb=120.0)
        rows = sink()
        assert [r["step"] for r in rows] == ["download", "extract", "load", "embed", "finalize"]
        assert all(r["size_mb"] == 120.0 for r in rows)

    def test_a_run_with_no_steps_writes_nothing(self, sink):
        tracker = _tracker()
        with record_task(tracker, "text_sort"):
            tracker.update("idle", "", 0, 0)
        assert sink() == []


class TestNormalizeRow:
    def test_generic_row_round_trips(self):
        row = _row(
            {
                "task": "find",
                "device": "cuda:0",
                "cuml": True,
                "media_type": "image",
                "embedder": "siglip",
                "n": 10,
                "size_mb": 0,
                "step": "score",
                "seconds": 1.5,
                "ok": True,
                "complete": True,
            }
        )
        assert row["device"] == "cuda+cuml"
        assert row["step"] == "score"
        assert row["seconds"] == 1.5

    def test_legacy_load_profiler_row_is_understood(self):
        # An existing dataset-load calibration sweep folds into a new profile
        # rather than having to be re-measured.
        row = _row(
            {
                "device": "cuda",
                "media_type": "image",
                "embedder": "siglip",
                "n": 500,
                "download_size_mb": 120.0,
                "phase": "model_load",
                "seconds": 0.5,
                "cuml": False,
            }
        )
        assert row["task"] == "dataset_load"
        assert row["step"] == "load"  # phase name mapped onto the registry's
        assert row["device"] == "cuda"
        assert row["size_mb"] == 120.0

    def test_legacy_sub_slot_row_becomes_a_slot_sample(self):
        row = _row(
            {
                "device": "cpu",
                "media_type": "image",
                "embedder": "",
                "n": 5,
                "phase": "finalize:coverage",
                "seconds": 2.0,
            }
        )
        assert (row["step"], row["slot"]) == ("finalize", "coverage")

    @pytest.mark.parametrize(
        "raw",
        [
            {"task": "text_sort", "step": "score", "seconds": 1.0, "ok": False, "complete": True},
            {"task": "text_sort", "step": "score", "seconds": 1.0, "ok": True, "complete": False},
            {"task": "text_sort", "step": "not_a_step", "seconds": 1.0},
            {"task": "who_knows", "step": "score", "seconds": 1.0},
            {"phase": "embed", "seconds": 1.0, "n": 0},  # legacy failed load
            {"task": "text_sort", "step": "score"},  # no duration
            {"seconds": 1.0},  # neither shape
        ],
    )
    def test_unusable_rows_are_rejected(self, raw):
        assert normalize_row(raw) is None

    def test_device_key_splits_cuda_on_cuml(self):
        assert device_key("cuda:0", True) == "cuda+cuml"
        assert device_key("cuda", False) == "cuda"
        assert device_key("cpu", True) == "cpu"


class TestFitting:
    def test_affine_fit_recovers_a_known_line(self):
        xs = [0.0, 10.0, 20.0, 30.0]
        ys = [2.0 + 0.5 * x for x in xs]
        a, b, r2 = affine_fit(xs, ys)
        assert a == pytest.approx(2.0)
        assert b == pytest.approx(0.5)
        assert r2 == pytest.approx(1.0)

    def test_no_spread_in_n_yields_no_slope(self):
        a, b, _ = affine_fit([100.0, 100.0, 100.0], [3.0, 4.0, 5.0])
        assert b == 0.0
        assert a == pytest.approx(4.0)

    def test_a_negative_slope_collapses_to_the_median(self):
        # Noise beating signal on a short step must not ship a coefficient that
        # extrapolates to "this gets faster the bigger it gets".
        samples = [
            {"n": 100.0, "size_mb": 0.0, "seconds": 5.0},
            {"n": 200.0, "size_mb": 0.0, "seconds": 3.0},
            {"n": 300.0, "size_mb": 0.0, "seconds": 1.0},
        ]
        coeffs = _fit(samples, byte_scaled=False)
        assert coeffs.b == 0.0
        assert coeffs.a == pytest.approx(3.0)

    def test_the_fits_r2_is_kept_not_discarded(self):
        # `affine_fit` has always computed an r2 and `fit_step` threw it away at
        # the call site, which made this the one place in the tree that measured
        # a fit's quality and discarded it (#3329). A clean line must arrive
        # with r2 ~ 1, and a noisy one materially below it, or the number is
        # being carried without meaning anything.
        clean = _fit(
            [{"n": float(x), "size_mb": 0.0, "seconds": 1.0 + 0.01 * x} for x in (100, 200, 300, 400)],
            byte_scaled=False,
        )
        assert clean.b > 0
        assert clean.r2 == pytest.approx(1.0)

        noisy = _fit(
            [
                {"n": 100.0, "size_mb": 0.0, "seconds": 2.0},
                {"n": 200.0, "size_mb": 0.0, "seconds": 9.0},
                {"n": 300.0, "size_mb": 0.0, "seconds": 4.0},
                {"n": 400.0, "size_mb": 0.0, "seconds": 12.0},
            ],
            byte_scaled=False,
        )
        assert noisy.b > 0
        assert noisy.r2 < 0.9

    def test_a_two_point_line_reports_no_r2(self):
        # Two points define a line exactly, so ss_res is 0 and r2 is 1.0
        # whatever the points are -- a goodness score that is arithmetic rather
        # than evidence (#3345). The slope is still the best line available and
        # is kept; only the claim about it is withheld.
        two = _fit(
            [
                {"n": 100.0, "size_mb": 0.0, "seconds": 3.0},
                {"n": 200.0, "size_mb": 0.0, "seconds": 5.0},
            ],
            byte_scaled=False,
        )
        assert two.a > 0, "this fixture is chosen NOT to clamp"
        assert two.b > 0, "the coefficients are still the best line available"
        assert math.isnan(two.r2)
        assert "r2" not in two.to_json()

    def test_a_clamped_two_point_line_still_reports_its_real_r2(self):
        # The two guards meet here, and the clamp wins on purpose. Once the
        # intercept is clamped the stored model is no longer the interpolant, so
        # its residuals are real and worth reporting however few points there
        # were. This is the `cuda+cuml|*|*` model-load cell from #3345: it used
        # to advertise r2 1.000 while mispredicting its own two samples by
        # 4290%, and the honest score is heavily negative.
        clamped = _fit(
            [
                {"n": 245.0, "size_mb": 0.0, "seconds": 1.8},
                {"n": 412.0, "size_mb": 0.0, "seconds": 54.9},
            ],
            byte_scaled=False,
        )
        assert clamped.a == 0.0
        assert clamped.r2 < 0, "worse than predicting the mean, and it should say so"

        # A third distinct size is enough to say something, so the r2 returns.
        three = _fit(
            [{"n": float(x), "size_mb": 0.0, "seconds": 1.0 + 0.01 * x} for x in (100, 200, 300)],
            byte_scaled=False,
        )
        assert three.r2 == pytest.approx(1.0)

    def test_repeats_at_two_sizes_still_report_no_r2(self):
        # Four samples, two sizes: the count looks comfortable and the fit is
        # still a two-point line with repeats. It is the number of *distinct*
        # sizes that decides whether an r2 means anything.
        coeffs = _fit(
            [
                {"n": n, "size_mb": 0.0, "seconds": secs}
                for n, secs in ((100.0, 2.0), (100.0, 2.4), (200.0, 4.1), (200.0, 3.7))
            ],
            byte_scaled=False,
        )
        assert coeffs.b > 0
        assert math.isnan(coeffs.r2)

    def test_a_clamped_intercept_is_rescored_against_what_is_stored(self):
        # `seconds()` must never hand the bar a negative slice, so a negative
        # OLS intercept is clamped to zero. That makes the stored model a
        # DIFFERENT line from the one `affine_fit` scored, and #3345 measured 58
        # of 195 affine cells in exactly that state -- one carrying r2 0.98 on
        # coefficients 52% out. The r2 has to describe the coefficients that
        # ship, not the line they were derived from.
        samples = [
            {"n": 245.0, "size_mb": 0.0, "seconds": 1.8},
            {"n": 412.0, "size_mb": 0.0, "seconds": 54.9},
            {"n": 600.0, "size_mb": 0.0, "seconds": 120.0},
        ]
        coeffs = _fit(samples, byte_scaled=False)
        assert coeffs.a == 0.0, "this fixture is chosen to land a negative intercept"
        assert coeffs.b > 0

        raw_r2 = affine_fit([s["n"] for s in samples], [s["seconds"] for s in samples])[2]
        # The unclamped line fits well; the clamped one does not, and the
        # reported number must be the second.
        assert raw_r2 > 0.95
        assert coeffs.r2 < raw_r2

        # And it must equal the goodness of the shipped model, computed the same
        # way `seconds()` evaluates it.
        expected = 1.0 - (
            sum((s["seconds"] - coeffs.seconds(n=s["n"])) ** 2 for s in samples)
            / sum((s["seconds"] - sum(x["seconds"] for x in samples) / 3) ** 2 for s in samples)
        )
        assert coeffs.r2 == pytest.approx(expected)

    def test_an_unclamped_fit_keeps_the_ols_r2(self):
        # The rescoring must not disturb the ordinary case: a non-negative
        # intercept means the stored line IS the fitted line.
        coeffs = _fit(
            [{"n": float(x), "size_mb": 0.0, "seconds": 5.0 + 0.01 * x} for x in (100, 200, 300, 400)],
            byte_scaled=False,
        )
        assert coeffs.a > 0
        assert coeffs.r2 == pytest.approx(1.0)

    def test_a_step_that_was_not_fitted_as_a_line_has_no_r2(self):
        # NaN here means "not fitted this way", which is a different statement
        # from a bad fit: the median fallback and the byte-scaled path never
        # drew a line, so attaching a goodness score to them would be a lie.
        median_fallback = _fit(
            [
                {"n": 100.0, "size_mb": 0.0, "seconds": 5.0},
                {"n": 200.0, "size_mb": 0.0, "seconds": 3.0},
                {"n": 300.0, "size_mb": 0.0, "seconds": 1.0},
            ],
            byte_scaled=False,
        )
        assert math.isnan(median_fallback.r2)
        byte_scaled = _fit([{"n": 500.0, "size_mb": 100.0, "seconds": 10.0}], byte_scaled=True)
        assert math.isnan(byte_scaled.r2)
        assert "r2" not in median_fallback.to_json()

    def test_r2_survives_a_json_round_trip(self):
        coeffs = StepCoeffs(a=1.0, b=2.0, r2=0.9876)
        assert coeffs.to_json()["r2"] == pytest.approx(0.9876)
        # `from_json` is Optional-returning (it parses untrusted profile JSON),
        # so narrow before reading the field rather than chaining through it.
        parsed = StepCoeffs.from_json(coeffs.to_json())
        assert parsed is not None
        assert parsed.r2 == pytest.approx(0.9876)
        bare = StepCoeffs.from_json({"a": 1.0})
        assert bare is not None
        assert math.isnan(bare.r2)

    def test_byte_scaled_step_fits_a_per_mb_rate(self):
        samples = [
            {"n": 500.0, "size_mb": 100.0, "seconds": 10.0},
            {"n": 5000.0, "size_mb": 400.0, "seconds": 40.0},
        ]
        coeffs = _fit(samples, byte_scaled=True)
        assert coeffs.per_mb == pytest.approx(0.1)
        assert (coeffs.a, coeffs.b) == (0.0, 0.0)

    def test_byte_scaled_step_with_no_archive_is_a_real_zero(self):
        # Every measured load hit a warm cache: this deployment genuinely pays
        # nothing to acquire, which is a measurement, not a missing one.
        coeffs = _fit([{"n": 5.0, "size_mb": 0.0, "seconds": 0.0}], byte_scaled=True)
        assert coeffs.per_mb == 0.0

    def test_thin_cells_are_dropped(self):
        rows = [
            {
                "task": "find",
                "device": "cpu",
                "media_type": "image",
                "embedder": "siglip",
                "n": 10,
                "size_mb": 0,
                "step": "score",
                "seconds": 1.0,
                "ok": True,
                "complete": True,
            }
        ]
        assert fit_profile(rows, min_samples=2)["tasks"] == {}
        assert fit_profile(rows, min_samples=1)["tasks"]["find"]["cells"]

    def test_rollup_cells_are_emitted_alongside_the_exact_one(self):
        rows = []
        for n in (100, 200):
            rows.append(
                {
                    "task": "find",
                    "device": "cpu",
                    "media_type": "image",
                    "embedder": "siglip",
                    "n": n,
                    "size_mb": 0,
                    "step": "score",
                    "seconds": 0.01 * n,
                    "ok": True,
                    "complete": True,
                }
            )
        cells = fit_profile(rows, min_samples=2)["tasks"]["find"]["cells"]
        assert set(cells) == {"cpu|image|siglip", "cpu|image|", "cpu||"}


class TestRoundTrip:
    def test_recorded_rows_fit_into_a_profile_that_paces_the_task(self, sink, tmp_path, monkeypatch):
        # The whole tuning loop in one test: record a task at two sizes, fit,
        # load the result, and check the pacing now reflects what was measured.
        monkeypatch.setattr(timing_profile, "resolve_device_name", lambda: "cpu")
        monkeypatch.setattr(timing_recorder, "resolve_device_name", lambda: "cpu")
        monkeypatch.setattr(timing_profile, "cuml_active", lambda: False)
        monkeypatch.setattr(timing_recorder, "cuml_active", lambda: False)

        clock = {"t": 0.0}
        monkeypatch.setattr(timing_recorder.time, "monotonic", lambda: clock["t"])

        for n, score_secs in ((1_000, 1.0), (100_000, 100.0)):
            tracker = _tracker()
            with record_task(tracker, "text_sort", media_type="image", embedder="siglip") as rec:
                tracker.update("sorting", "", 0, 0, step=1, total_steps=3)
                clock["t"] += 8.0  # a fixed 8s encoder load, whatever the size
                tracker.update("sorting", "", 0, 0, step=2, total_steps=3)
                clock["t"] += 0.05
                tracker.update("sorting", "", 0, 0, step=3, total_steps=3)
                clock["t"] += score_secs  # scoring scales with n
                rec.set_scale(n=n)

        profile = fit_profile(sink(), min_samples=2)
        path = tmp_path / "fitted.json"
        path.write_text(json.dumps(profile), encoding="utf-8")

        try:
            timing.reload_profile(str(path))
            small = _weights("text_sort", device="cpu", media_type="image", embedder="siglip", n=1_000)
            large = _weights("text_sort", device="cpu", media_type="image", embedder="siglip", n=100_000)
            # 8s load vs 1s score at n=1k; 8s load vs 100s score at n=100k.
            assert small[0] > small[2]
            assert large[2] > large[0]
        finally:
            timing.reload_profile("")

    def test_coverage_report_names_the_unmeasured_families(self):
        rows = [
            {
                "task": "find",
                "device": "cpu",
                "media_type": "",
                "embedder": "",
                "n": n,
                "size_mb": 0,
                "step": "score",
                "seconds": 1.0,
                "ok": True,
                "complete": True,
            }
            for n in (10, 20)
        ]
        lines = "\n".join(coverage_report(rows, fit_profile(rows, min_samples=2)))
        assert "find" in lines
        assert "NOT MEASURED" in lines
        assert "text_sort" in lines

    def test_coverage_report_reads_the_r2_it_fitted(self):
        # #3345: the r2 was persisted by #3334 and read by nothing, including
        # the one report an admin sees after a sweep. A cell count cannot say
        # whether the coefficients describe the data, so the report has to.
        rows = [
            {
                "task": "find",
                "device": "cpu",
                "media_type": "image",
                "embedder": "siglip",
                "n": n,
                "size_mb": 0,
                "step": "score",
                "seconds": 2.0 * n,  # exactly linear -> r2 == 1
                "ok": True,
                "complete": True,
            }
            for n in (10, 20, 30)
        ]
        lines = "\n".join(coverage_report(rows, fit_profile(rows, min_samples=2)))
        assert "affine" in lines
        assert "median r² 1.00" in lines

    def test_coverage_report_counts_the_fits_below_the_bar(self):
        # A median hides the tail, and the tail is where a bar drifts. A clean
        # cell and a noisy one must report the noisy one, not average it away.
        #
        # The noise is deliberately *inside* one encoder rather than across the
        # two. #3522 withholds a rollup step whose groups contradict each other,
        # so a fixture whose only bad fit was the pooled cell would now measure
        # that suppression instead of the r² count this test is about. These two
        # encoders average ~50 s and ~55 s, well inside the spread a rollup
        # keeps, so every cell survives and the poor fit is a real one.
        rows = []
        for embedder, secs in (
            ("clean", lambda n: 2.0 * n),
            ("noisy", lambda n: {10: 20.0, 20: 80.0, 30: 30.0}.get(n, 90.0)),
        ):
            rows += [
                {
                    "task": "find",
                    "device": "cpu",
                    "media_type": "image",
                    "embedder": embedder,
                    "n": n,
                    "size_mb": 0,
                    "step": "score",
                    "seconds": secs(n),
                    "ok": True,
                    "complete": True,
                }
                for n in (10, 20, 30, 40)
            ]
        lines = "\n".join(coverage_report(rows, fit_profile(rows, min_samples=2)))
        assert "below 0.90" in lines

    def test_coverage_report_says_when_a_task_is_too_short_to_pace(self):
        # #3596: text_sort's 288 step-samples were the largest count in #3521's
        # report and described a 0.9 s job that every arm of that study paced
        # 0.80-0.85 wrong. A sample count reads that as excellent coverage, so
        # the report has to say the other thing out loud.
        rows = [
            {
                "task": "text_sort",
                "device": "cpu",
                "media_type": "image",
                "embedder": "siglip",
                "n": n,
                "size_mb": 0,
                "step": step,
                "seconds": secs,
                "ok": True,
                "complete": True,
                "cold_model": False,
            }
            for n in (1000, 2000, 3000)
            for step, secs in (("load_model", 0.0), ("embed_query", 0.05), ("score", 0.0003 * n))
        ]
        lines = "\n".join(coverage_report(rows, fit_profile(rows, min_samples=2)))
        assert "TOO SHORT TO PACE" in lines
        assert "load_model 0.00" in lines and "embed_query 0.05" in lines

    def test_coverage_report_names_the_deferred_floor_beside_the_measurement(self):
        # The floor is right (a cold run really does pay it) and invisible: on a
        # sub-second task it is most of the predicted total, so it redistributes
        # the bar for every warm run while the rows say the step was free.
        rows = [
            {
                "task": "text_sort",
                "device": "cpu",
                "media_type": "image",
                "embedder": "siglip",
                "n": n,
                "size_mb": 0,
                "step": step,
                "seconds": secs,
                "ok": True,
                "complete": True,
                "cold_model": cold,
            }
            for cold, n in ((True, 1000), (False, 2000), (False, 3000))
            for step, secs in (
                ("load_model", 15.4 if cold else 0.0),
                ("embed_query", 0.05),
                ("score", 0.0003 * n),
            )
        ]
        fitted = fit_profile(rows, min_samples=2)
        # The fit really does store the floor, which is what the line reports.
        assert fitted["tasks"]["text_sort"]["cells"]["cpu|image|siglip"]["steps"]["load_model"]["a"] == 0.5
        lines = "\n".join(coverage_report(rows, fitted))
        assert "load_model: measured 0.00 s on 2 of 3 runs and real on 1" in lines
        assert "0.50 s floor" in lines

    def test_coverage_report_leaves_a_paceable_task_alone(self):
        # The counterpart the two tests above need: a task whose steps are long
        # enough to pace gets no pacing complaint, so the line means something.
        rows = [
            {
                "task": "text_sort",
                "device": "cpu",
                "media_type": "image",
                "embedder": "siglip",
                "n": n,
                "size_mb": 0,
                "step": step,
                "seconds": secs,
                "ok": True,
                "complete": True,
                "cold_model": False,
            }
            for n in (1000, 2000, 3000)
            for step, secs in (("load_model", 8.0), ("embed_query", 0.05), ("score", 0.003 * n))
        ]
        lines = "\n".join(coverage_report(rows, fit_profile(rows, min_samples=2)))
        assert "TOO SHORT TO PACE" not in lines
        assert "floor" not in lines

    def test_coverage_report_does_not_score_a_step_it_did_not_fit_as_a_line(self):
        # A byte-scaled step and a median fallback carry no r2 at all. Counting
        # either as a bad fit would be the exact confusion #3345 set out to
        # separate, so they are named as their own outcomes.
        rows = [
            {
                "device": "cpu",
                "media_type": "image",
                "embedder": "siglip",
                "dataset_id": "caltech101_s",
                "n": 400,  # one size only: no spread for OLS to use
                "download_size_mb": 131.0,
                "phase": phase,
                "seconds": 3.0,
            }
            for phase in ("download", "embed")
            for _ in range(2)
        ]
        lines = "\n".join(coverage_report(rows, fit_profile(rows, min_samples=2)))
        assert "byte-rate" in lines
        assert "median-fallback" in lines
        assert "affine" not in lines


def _load_row(media: str, embedder: str, step: str, n: float, seconds: float, device: str = "cuda") -> dict:
    """One recorded `dataset_load` step, for the rollup-suppression tests."""
    return {
        "task": "dataset_load",
        "device": device,
        "cuml": True,
        "media_type": media,
        "embedder": embedder,
        "n": n,
        "size_mb": 0,
        "step": step,
        "seconds": seconds,
        "ok": True,
        "complete": True,
    }


class TestContradictedRollups:
    """#3522: a rollup must not fit one line through groups it measured as unlike.

    A rollup cell is only ever *reached* for a combination the sweep never
    measured — `cell_keys` tries every more specific key first and the fitter
    emits one for everything it saw — so the rollup's whole job is
    extrapolation. #3345 measured what averaging unlike groups costs there:
    exact cells at median r² 1.00 / 3 % error against `(device, *, *)` at
    0.29 / 50 %, and 162 % on one arm.
    """

    #: #3345's measured mechanism, to the digit: `(cuda+cuml, *, *)` pooling an
    #: image import at 0.014 s/item with an audio one at 0.102 (7.3x apart).
    DIVERGENT = (("image", "siglip", 0.014), ("audio", "clap_general", 0.102))

    def _rows(self, groups, step="embed"):
        return [
            _load_row(media, embedder, step, n, rate * n)
            for media, embedder, rate in groups
            for n in (400, 800, 1600, 2400)
        ]

    def test_the_device_rollup_withholds_a_step_its_own_groups_contradict(self):
        cells = fit_profile(self._rows(self.DIVERGENT), min_samples=2)["tasks"]["dataset_load"]["cells"]
        # Both exact cells keep the slope they measured...
        assert cells["cuda+cuml|image|siglip"]["steps"]["embed"]["b"] == pytest.approx(0.014)
        assert cells["cuda+cuml|audio|clap_general"]["steps"]["embed"]["b"] == pytest.approx(0.102)
        # ...and the cell that pools them declines to claim a third number.
        assert "cuda+cuml||" not in cells

    def test_a_coherent_rollup_is_still_emitted(self):
        # The rollups are the reason a small sweep is worth running, so a merely
        # imprecise one must survive: only *contradiction* withholds a step.
        coherent = (("image", "siglip", 0.014), ("audio", "clap_general", 0.020))
        cells = fit_profile(self._rows(coherent), min_samples=2)["tasks"]["dataset_load"]["cells"]
        assert cells["cuda+cuml||"]["steps"]["embed"]["b"] == pytest.approx(0.017)

    def test_only_the_contradicted_step_is_withheld(self):
        # Suppression is per-step, not per-cell: `step_terms` falls one absent
        # step through to its shipped default while the rest of the cell applies.
        rows = self._rows(self.DIVERGENT, step="embed")
        rows += self._rows((("image", "siglip", 0.004), ("audio", "clap_general", 0.005)), step="finalize")
        steps = fit_profile(rows, min_samples=2)["tasks"]["dataset_load"]["cells"]["cuda+cuml||"]["steps"]
        assert "embed" not in steps
        assert steps["finalize"]["b"] == pytest.approx(0.0045)

    def test_the_media_rollup_withholds_contradicted_encoders(self):
        # Same mechanism one level in: `(device, media, *)` is reached only for
        # an encoder that media type was never measured with, and #3062 measured
        # a 4.75x spread in finalize's slope across embedding dimension.
        rows = self._rows((("image", "siglip", 0.014), ("image", "clip_b32", 0.070)))
        cells = fit_profile(rows, min_samples=2)["tasks"]["dataset_load"]["cells"]
        assert cells["cuda+cuml|image|siglip"]["steps"]["embed"]["b"] == pytest.approx(0.014)
        assert "cuda+cuml|image|" not in cells

    def test_a_single_group_rollup_is_never_contradicted(self):
        # One media type behind the rollup: it repeats the exact cell's claim
        # rather than blending anything, and is the coverage rollups exist for.
        rows = self._rows((("image", "siglip", 0.014),))
        cells = fit_profile(rows, min_samples=2)["tasks"]["dataset_load"]["cells"]
        assert cells["cuda+cuml||"]["steps"]["embed"]["b"] == pytest.approx(0.014)

    def test_a_step_every_group_calls_free_is_not_contradicted(self):
        # Sub-10ms against sub-10ms is arithmetic noise, not disagreement: a
        # ratio between two negligible numbers must not withhold a real cell.
        rows = [
            _load_row(media, embedder, "finalize", n, secs)
            for media, embedder, secs in (("image", "siglip", 0.001), ("audio", "clap_general", 0.006))
            for n in (400, 800, 1600, 2400)
        ]
        cells = fit_profile(rows, min_samples=2)["tasks"]["dataset_load"]["cells"]
        assert "finalize" in cells["cuda+cuml||"]["steps"]

    def test_one_free_group_beside_a_costly_one_is_contradicted(self):
        # The shape #3520/#3521 measured: a warm zero pooled with a cold cost.
        # Whichever way the lookup lands, the pooled number is wrong.
        rows = [
            _load_row(media, embedder, "finalize", n, secs)
            for media, embedder, secs in (("image", "siglip", 0.0), ("audio", "clap_general", 6.0))
            for n in (400, 800, 1600, 2400)
        ]
        cells = fit_profile(rows, min_samples=2)["tasks"]["dataset_load"]["cells"]
        assert "cuda+cuml||" not in cells

    def test_the_exact_cell_is_never_suppressed(self):
        # It wildcards nothing, so it pools nothing, however wild its samples.
        rows = [_load_row("image", "siglip", "embed", n, s) for n, s in ((400, 0.5), (800, 40.0), (1600, 2.0))]
        cells = fit_profile(rows, min_samples=2)["tasks"]["dataset_load"]["cells"]
        assert "embed" in cells["cuda+cuml|image|siglip"]["steps"]


class TestCoverageReportSpecificity:
    """#3522: the coverage report must not pool the specificity levels.

    "5 cells" reads like five measurements. #3345 measured that the level
    guaranteed to match is the weakest one, so the split is what tells an admin
    whether their sweep bought exact cells or fallbacks.
    """

    def _rows(self):
        return [
            _load_row(media, embedder, "embed", n, rate * n)
            for media, embedder, rate in (("image", "siglip", 0.014), ("audio", "clap_general", 0.102))
            for n in (400, 800, 1600, 2400)
        ]

    def test_the_quality_line_is_broken_down_by_specificity(self):
        rows = self._rows()
        lines = "\n".join(coverage_report(rows, fit_profile(rows, min_samples=2)))
        assert "exact  (device|media|embedder)  2 cells" in lines
        assert "rollup (device|media|*)         2 cells" in lines

    def test_a_withheld_rollup_is_named_rather_than_vanishing(self):
        # The device rollup has no surviving cell here, and silently omitting
        # its line would read as "the sweep never got that far" — the opposite
        # of what happened.
        rows = self._rows()
        lines = "\n".join(coverage_report(rows, fit_profile(rows, min_samples=2)))
        assert "rollup (device|*|*)" in lines
        assert "0 cells, 1 step withheld (pooled groups disagree)" in lines

    def test_the_split_reports_each_level_own_r2(self):
        # The point of the split: a weak rollup beside strong exact cells must
        # not be averaged into one reassuring median.
        rows = [
            _load_row(media, embedder, "embed", n, rate * n + noise)
            for media, embedder, rate, noise in (("image", "siglip", 0.05, 0.0), ("audio", "clap_general", 0.07, 0.0))
            for n in (400, 800, 1600, 2400)
        ]
        report = coverage_report(rows, fit_profile(rows, min_samples=2))
        exact = next(line for line in report if "exact" in line)
        rollup = next(line for line in report if "(device|*|*)" in line)
        assert "median r² 1.00" in exact  # each media type is exactly linear
        assert "median r² 1.00" not in rollup  # one slope through both is not

    def test_the_cell_counts_still_sum_to_the_header(self):
        rows = self._rows()
        profile = fit_profile(rows, min_samples=2)
        report = coverage_report(rows, profile)
        header = next(line for line in report if "dataset_load" in line)
        split = sum(int(line.split("  ")[-1].split(" cell")[0]) for line in report if "device|" in line)
        assert f"{split} cells" in header
