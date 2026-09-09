"""Cache branches: which path a step took, and what the fitter may do with it.

A step's cost often forks on a cache that its duration cannot reveal. #3345's
sweep opened 16 datasets and every one of them **restored** the coverage atlas
cached in its pickle, recording 0.008-0.016 s at every ``n`` from 245 to 2954;
the same sweep's ``dataset_stage`` leg read the embeddings pkl the
``dataset_load`` leg had just written and recorded 0.000-0.002 s of embedding
across all four image tiers, in a separate interpreter. Every one of those
numbers is correct. None of them is a cost model, because the branch a user
waits on — the hierarchical-k-means rebuild (0.0026 s/item, ~700× the restore
at n = 2954), the real embed — was never run (#3521, measured in
``docs/experiments/2026-09-02-timing-r2-3345/REPORT.md``).

These tests pin the four halves of the answer: the recorder carries the branch
name, the fitter prices a forked step only from the runs that did the work, a
step that only ever read a cache withholds its whole cell rather than shipping a
confident millisecond, and — once both paths *have* been measured — the profile
carries coefficients for each and a caller that names its branch is paced from
that branch alone (#3594). The last of those is what stops an admin's choice of
profile from being a choice of which branch to be wrong about.
"""

import json

import pytest

from vtscore import timing
from vtscore.concurrency.progress import PROGRESS_COMMON_EXTRAS, ProgressTracker
from vtscore.timing import recorder as timing_recorder
from vtscore.timing.fit import cheap_branch_only, coverage_report, fit_branches, fit_profile, fit_step, load_rows
from vtscore.timing.profile import StepCoeffs
from vtscore.timing.recorder import RECORD_ENV_VAR, note_branch, note_no_encoder_load, record_task


def _tracker() -> ProgressTracker:
    return ProgressTracker(extra_fields=dict(PROGRESS_COMMON_EXTRAS))


@pytest.fixture
def sink(tmp_path, monkeypatch):
    """Arm the recorder at a temp JSONL and yield a reader for what it wrote."""
    path = tmp_path / "timings.jsonl"
    monkeypatch.setenv(RECORD_ENV_VAR, str(path))
    timing_recorder.reset_seen_models_for_tests()

    def read():
        return load_rows([str(path)]) if path.exists() else []

    return read


def _open_run(tracker) -> None:
    """The two tracker steps a dataset open reports."""
    tracker.update("loading", "items", 0, 0, step=1, total_steps=2)
    tracker.update("loading", "coverage", 0, 0, step=2, total_steps=2)


def _sample(n: float, seconds: float, branch: str = "") -> dict:
    return {"n": n, "size_mb": 0.0, "seconds": seconds, "cold": False, "branch": branch}


class TestRecorderCarriesTheBranch:
    def test_marked_step_carries_its_branch(self, sink):
        tracker = _tracker()
        with record_task(tracker, "dataset_open", media_type="image") as rec:
            rec.mark_branch("coverage", "restored")
            _open_run(tracker)
            rec.set_scale(n=980)
        rows = {r["step"]: r for r in sink()}
        assert rows["coverage"]["branch"] == "restored"
        assert "branch" not in rows["items"], "an unforked step must not claim one"

    def test_note_branch_reaches_the_thread_bound_recorder(self, sink):
        """The deep call sites name the branch without holding the recorder."""
        tracker = _tracker()
        with record_task(tracker, "dataset_open", media_type="image") as rec:
            rec.bind_thread()
            note_branch("coverage", "rebuilt")  # as the route's atlas branch does
            _open_run(tracker)
        assert {r["step"]: r.get("branch") for r in sink()}["coverage"] == "rebuilt"

    def test_note_branch_is_a_no_op_with_nothing_recording(self):
        """Product code calls this unconditionally; it must never raise."""
        timing_recorder._active.recorder = None
        note_branch("embed", "cached")
        note_no_encoder_load()

    def test_a_cached_run_does_not_claim_the_encoder_key(self, sink):
        """The run that reads a pkl must leave the residency key for the one
        that really loads the model — otherwise the genuinely cold load that
        follows is written ``cold_model: false`` (the #3345 mislabel, reached
        by a different route)."""
        cached_tracker = _tracker()
        with record_task(cached_tracker, "dataset_stage", media_type="image", embedder="siglip") as rec:
            rec.bind_thread()
            note_branch("embed", "cached")
            note_no_encoder_load()
            for step in (1, 2, 3):
                cached_tracker.update("loading", "x", 0, 0, step=step, total_steps=3)
        fresh_tracker = _tracker()
        with record_task(fresh_tracker, "dataset_stage", media_type="image", embedder="siglip") as rec:
            rec.bind_thread()
            note_branch("embed", "fresh")
            for step in (1, 2, 3):
                fresh_tracker.update("loading", "x", 0, 0, step=step, total_steps=3)
        embeds = [r for r in sink() if r["step"] == "embed"]
        assert [r["branch"] for r in embeds] == ["cached", "fresh"]
        assert "cold_model" not in embeds[0], "a run that loaded no encoder claims nothing"
        assert embeds[1]["cold_model"] is True, "the real load is the cold one"

    def test_only_phases_records_a_partial_run(self, sink):
        """The on-demand atlas rebuild is a dataset_open's second step alone.

        Without the narrowing it would write ``items: 0.0`` — indistinguishable
        from a measurement that opening a dataset's pickle is free.
        """
        tracker = _tracker()
        rec = record_task(
            tracker,
            "dataset_open",
            media_type="image",
            status_phases={"loading": "coverage"},
            only_phases=("coverage",),
        )
        rec.start()
        rec.mark_branch("coverage", "rebuilt")
        tracker.update("loading", "Building coverage atlas…", 0, 0, step=1, total_steps=1)
        rec.finish(n=980)
        assert [r["step"] for r in sink()] == ["coverage"]


class TestFittingAForkedStep:
    def test_priced_from_the_runs_that_did_the_work(self):
        """A restore is not a cheap sample of a rebuild; it is a different path."""
        cheap = [_sample(n, 0.01, "restored") for n in (245, 980, 2954)]
        dear = [_sample(245, 61.0, "rebuilt"), _sample(980, 240.0, "rebuilt")]
        coeffs = fit_step(cheap + dear, byte_scaled=False)
        assert coeffs is not None
        # The two rebuilds alone: 61 s at 245 items, 240 s at 980.
        assert coeffs.b == pytest.approx(0.2435, abs=1e-3)
        assert coeffs.seconds(n=980) == pytest.approx(240.0, rel=0.02)

    def test_unmarked_rows_fit_exactly_as_they_did(self):
        """Absent markers are not evidence of a fork, and must change nothing."""
        plain = [_sample(100, 0.1), _sample(200, 0.2), _sample(300, 0.3)]
        assert fit_step(plain, byte_scaled=False) == fit_step(
            [{k: v for k, v in s.items() if k != "branch"} for s in plain], byte_scaled=False
        )

    def test_cheap_only_is_not_the_same_as_cheap(self):
        assert cheap_branch_only([_sample(1, 0.01, "restored")]) is True
        assert cheap_branch_only([_sample(1, 0.01, "restored"), _sample(1, 60.0, "rebuilt")]) is False
        assert cheap_branch_only([_sample(1, 0.01)]) is False, "unmarked is not cheap-only"
        assert cheap_branch_only([]) is False


def _row(step: str, seconds: float, n: float, branch: str = "") -> dict:
    row = {
        "task": "dataset_open",
        "device": "cuda",
        "cuml": False,
        "media_type": "image",
        "embedder": "siglip",
        "n": n,
        "size_mb": 0.0,
        "ok": True,
        "complete": True,
        "step": step,
        "seconds": seconds,
    }
    if branch:
        row["branch"] = branch
    return row


#: The three opens #3345 measured, at the sizes it measured them.
_RESTORED_ONLY = [
    row
    for n, items, cov in ((245, 0.4, 0.010), (980, 1.2, 0.012), (2954, 3.0, 0.016))
    for row in (_row("items", items, n), _row("coverage", cov, n, "restored"))
]


class TestACheapOnlyStepWithholdsItsCell:
    def test_no_cell_is_emitted(self):
        """Not just the coverage step: the whole cell.

        ``step_terms`` fills a missing step from ``TaskSpec.default_terms``,
        which are *pseudo-seconds*. Emitting a measured ``items`` in real
        seconds beside a 0.85 pseudo-second ``coverage`` would produce a weight
        vector in no units at all.
        """
        profile = fit_profile(_RESTORED_ONLY, min_samples=2)
        assert profile["tasks"].get("dataset_open", {}).get("cells", {}) == {}

    def test_one_real_rebuild_unlocks_the_cell(self):
        rows = _RESTORED_ONLY + [
            _row("coverage", 61.0, 245, "rebuilt"),
            _row("coverage", 240.0, 980, "rebuilt"),
            _row("items", 0.4, 245),
            _row("items", 1.2, 980),
        ]
        steps = fit_profile(rows, min_samples=2)["tasks"]["dataset_open"]["cells"]["cuda|image|siglip"]["steps"]
        assert set(steps) == {"items", "coverage"}
        assert steps["coverage"]["b"] > 0.2, "priced from the rebuilds, not the restores"

    def test_the_report_says_which_branch_it_saw(self):
        """The line a sample count cannot carry — the ask in #3521."""
        profile = fit_profile(_RESTORED_ONLY, min_samples=2)
        report = "\n".join(coverage_report(_RESTORED_ONLY, profile))
        assert "coverage: 3 runs, all 'restored'" in report
        assert "cached path only" in report

    def test_the_report_counts_runs_not_cell_buckets(self):
        """Every row lands in three cells at three specificities; counting the
        buckets would report each of the three opens nine times."""
        report = "\n".join(coverage_report(_RESTORED_ONLY, fit_profile(_RESTORED_ONLY, min_samples=2)))
        assert "9 runs" not in report


#: The same opens, with the rebuild branch driven too (``--cold-atlas``). The
#: numbers are #3521's, measured on ``rack7n06``: a restore is ~0.01 s at every
#: size, a rebuild is 0.98 s at 412 items and 7.7 s at 2954.
_BOTH_BRANCHES = [
    row
    for n, items, restored, rebuilt in ((412, 0.5, 0.0090, 0.98), (2954, 3.0, 0.011, 7.7))
    for row in (
        _row("items", items, n),
        _row("coverage", restored, n, "restored"),
        _row("coverage", rebuilt, n, "rebuilt"),
    )
]


def _fitted_open_cell() -> dict:
    """The ``dataset_open`` cell fitted from both branches of #3521's opens."""
    return fit_profile(_BOTH_BRANCHES, min_samples=2)["tasks"]["dataset_open"]["cells"]["cuda|image|siglip"]


class TestPricingBothBranches:
    """One cell, two branches — the ask in #3594.

    Before this, a cell held exactly one set of coefficients for ``coverage``,
    so an admin choosing a profile was choosing *which branch to be wrong
    about*: #3521's held-out error was 0.94 of the bar for a profile fitted from
    restores alone facing a rebuild, and 0.49 the other way.
    """

    def test_a_forked_step_carries_coefficients_for_each_branch(self):
        coverage = _fitted_open_cell()["steps"]["coverage"]
        assert set(coverage["branches"]) == {"restored", "rebuilt"}
        # 0.98 s → 7.7 s across 412 → 2954 items.
        assert coverage["branches"]["rebuilt"]["b"] == pytest.approx(0.00264, abs=1e-4)
        # A restore is near-free at both sizes — three orders of magnitude
        # below the rebuild it is stored beside, which is the whole point.
        restored = StepCoeffs.from_json(coverage["branches"]["restored"])
        assert restored is not None
        assert restored.seconds(n=2954) == pytest.approx(0.011, abs=0.002)

    def test_the_top_level_coefficients_still_describe_the_dear_branch(self):
        """What a reader that knows nothing of branches sees, unchanged.

        The branch axis is additive: an older build, or a caller that cannot say
        which path it is on, keeps being paced for the branch someone waits on.
        """
        coverage = _fitted_open_cell()["steps"]["coverage"]
        assert coverage["b"] == coverage["branches"]["rebuilt"]["b"]
        assert coverage["a"] == coverage["branches"]["rebuilt"]["a"]

    def test_an_unforked_step_gets_no_branch_split(self):
        """``items`` never forks, and a step measured on one branch only is not
        a split — writing one would claim knowledge of the branch nobody ran."""
        assert "branches" not in _fitted_open_cell()["steps"]["items"]
        single = [r for r in _BOTH_BRANCHES if r.get("branch") != "restored"]
        cell = fit_profile(single, min_samples=2)["tasks"]["dataset_open"]["cells"]["cuda|image|siglip"]
        assert "branches" not in cell["steps"]["coverage"]

    def test_byte_scaled_steps_are_never_split(self):
        assert fit_branches([_sample(1, 2.0, "fresh"), _sample(1, 0.0, "cached")], byte_scaled=True) == {}

    def test_the_report_names_the_step_it_priced_per_branch(self):
        report = "\n".join(coverage_report(_BOTH_BRANCHES, fit_profile(_BOTH_BRANCHES, min_samples=2)))
        assert "coverage: priced per branch" in report
        assert "rebuilt, restored" in report


class TestPacingFromTheBranch:
    """The lookup half: one profile, two branches, two different bars.

    Fitting per branch fixes nothing on its own — a profile cell is keyed
    ``(device, media_type, embedder)`` and the pacing call site had no way to say
    which path it was on, so whichever branch the fit chose was the one every
    open got paced for. These tests use the *same* profile for both and check the
    vectors come out different, which is the only assertion that can fail if the
    branch argument is ignored.
    """

    @pytest.fixture(autouse=True)
    def _profile(self, tmp_path):
        """Load the profile fitted from #3521's two-branch opens, then unload it."""
        path = tmp_path / "profile.json"
        path.write_text(json.dumps(fit_profile(_BOTH_BRANCHES, min_samples=2)), encoding="utf-8")
        timing.reload_profile(str(path))
        yield
        timing.reload_profile("")

    def _weights(self, branch=None, n: float = 2954) -> list[float]:
        weights = timing.step_weights(
            "dataset_open", device="cuda", media_type="image", embedder="siglip", n=n, branch=branch
        )
        assert weights is not None, "the fitted cell must cover this lookup"
        return weights

    def test_the_two_branches_get_different_bars(self):
        restored, rebuilt = self._weights("restored"), self._weights("rebuilt")
        assert restored != rebuilt
        # A restore leaves the pickle read as essentially the whole job; a
        # rebuild is 7.7 s against the read's 3.0 s and takes most of the bar.
        assert restored[1] < 0.01, "a restored atlas must not hold a slice of the bar"
        assert rebuilt[1] > 0.7, "a rebuilt one must hold most of it"

    def test_naming_no_branch_paces_for_the_dear_one(self):
        """Unchanged behaviour for every caller that cannot say, which is the
        safe direction: over-reserving the bar for a rebuild that turns out to
        be a restore ends in a jump forward, not a bar frozen at 15 %."""
        assert self._weights() == self._weights("rebuilt")

    def test_an_unmeasured_branch_name_falls_back_rather_than_blanking(self):
        """``deferred`` is a real branch this sweep never produced (no dataset
        was past the auto-build threshold). It must resolve to the step's own
        coefficients, not to zero and not to None."""
        assert self._weights("deferred") == self._weights()

    def test_a_mapping_names_the_branch_per_step(self):
        terms = timing.step_terms(
            "dataset_open",
            device="cuda",
            media_type="image",
            embedder="siglip",
            n=2954,
            branch={"coverage": "restored"},
        )
        assert terms is not None
        assert terms["coverage"] == pytest.approx(0.011, abs=0.002)

    def test_a_profile_without_the_branch_axis_ignores_the_argument(self, tmp_path):
        """Every profile an admin has already generated. Asking for a branch
        must return exactly what asking for none does — not None, not a blank."""
        doc = fit_profile(_BOTH_BRANCHES, min_samples=2)
        for cell in doc["tasks"]["dataset_open"]["cells"].values():
            for coeffs in cell["steps"].values():
                coeffs.pop("branches", None)
        path = tmp_path / "legacy.json"
        path.write_text(json.dumps(doc), encoding="utf-8")
        timing.reload_profile(str(path))
        assert self._weights("restored") == self._weights("rebuilt") == self._weights()

    def test_the_branch_split_survives_a_round_trip_through_the_reader(self):
        parsed = timing.active_profile()
        forks = parsed.branches["dataset_open"]["cuda|image|siglip"]["coverage"]
        assert set(forks) == {"restored", "rebuilt"}
        assert forks["rebuilt"].seconds(n=2954) > 100 * forks["restored"].seconds(n=2954)
