"""Tests for the per-environment timing profile (:mod:`vtscore.timing`).

Three things have to hold for the profile to be worth having:

1. **An instance with no profile behaves exactly as it always did.** The shipped
   defaults reproduce the hand-tuned vectors these tasks carried before, so
   adopting the mechanism is not itself a behaviour change.
2. **A profile actually changes the pacing, in the direction the measurement
   says.** A cheap step should get a small slice, and the same task at two very
   different sizes should get two different splits.
3. **A broken profile costs nothing but accuracy.** Bad JSON, an unknown task, a
   future schema version — none of them may raise into a progress update.
"""

import json

import pytest

from vtscore import timing
from vtscore.timing import profile as timing_profile
from vtscore.timing.profile import EMPTY_PROFILE, StepCoeffs, parse_profile
from vtscore.timing.tasks import TASKS


@pytest.fixture(autouse=True)
def _clean_profile():
    """Every test starts and ends with the built-in defaults active."""
    timing.reload_profile("")
    yield
    timing.reload_profile("")


def _write(tmp_path, doc, name="profile.json"):
    path = tmp_path / name
    path.write_text(json.dumps(doc), encoding="utf-8")
    return str(path)


def _profile(tasks):
    return {"schema": "vtsearch-timing-profile", "version": 1, "tasks": tasks}


def _weights(task: str, **kwargs) -> list[float]:
    """``step_weights`` narrowed to non-``None`` (its no-coverage return)."""
    weights = timing.step_weights(task, **kwargs)
    assert weights is not None, f"expected weights for {task}"
    return weights


def _terms(task: str, **kwargs) -> dict[str, float]:
    """``step_terms`` narrowed to non-``None`` (its no-coverage return)."""
    terms = timing.step_terms(task, **kwargs)
    assert terms is not None, f"expected terms for {task}"
    return terms


def _shares(task: str, step: str, **kwargs) -> dict[str, float]:
    """``slot_shares`` narrowed to non-``None``."""
    shares = timing.slot_shares(task, step, **kwargs)
    assert shares is not None, f"expected slot shares for {task}.{step}"
    return shares


class TestShippedDefaults:
    def test_every_task_declares_a_consistent_spec(self):
        for name, spec in TASKS.items():
            assert spec.name == name
            assert len(spec.step_index) == len(spec.steps)
            assert max(spec.step_index) == spec.tracker_steps
            assert set(spec.byte_scaled) <= set(spec.steps)

    def test_defaults_reproduce_the_pre_profile_vectors(self):
        # These are the literal vectors the tasks shipped with before the
        # profile existed. An instance with no VTSEARCH_TIMING_PROFILE must
        # pace identically to how it did then.
        assert _weights("text_sort", device="cpu") == pytest.approx([0.75, 0.05, 0.20])
        assert _weights("find", device="cpu") == pytest.approx([0.10, 0.30, 0.60])
        assert _weights("train_and_score", device="cpu") == pytest.approx([0.10, 0.45, 0.40, 0.05])
        assert _weights("detector_load", device="cpu") == pytest.approx([0.15, 0.15, 0.70])
        assert _weights("dataset_open", device="cpu") == pytest.approx([0.15, 0.85])
        assert _weights("dataset_promote", device="cpu") == pytest.approx([0.6, 0.35, 0.05])

    def test_unknown_task_returns_the_callers_fallback(self):
        assert timing.step_weights("not_a_task", fallback=[1.0, 2.0]) == [1.0, 2.0]
        assert timing.step_weights("not_a_task") is None

    def test_dataset_load_defers_to_its_own_cost_model(self):
        # dataset_load deliberately ships no flat default terms: its default is
        # the measured affine table in _load_cost_model, which is n-aware.
        assert TASKS["dataset_load"].default_terms == ()
        assert timing.step_terms("dataset_load", device="cpu", n=100) is None


class TestCellLookup:
    def test_keys_run_specific_to_general(self):
        keys = timing.cell_keys("cpu", "image", "siglip")
        assert keys == ("cpu|image|siglip", "cpu|image|", "cpu||")

    def test_cuda_tries_both_cuml_variants_before_generalizing(self, monkeypatch):
        monkeypatch.setattr(timing_profile, "cuml_active", lambda: True)
        keys = timing.cell_keys("cuda:0", "image", "siglip")
        # cuML-on first (it matches this host), but the cuML-off row for the
        # exact media+embedder still beats a media-agnostic row.
        assert keys[:2] == ("cuda+cuml|image|siglip", "cuda|image|siglip")
        assert keys[-1] == "cuda||"

    def test_exact_cell_wins_over_rollup(self, tmp_path):
        path = _write(
            tmp_path,
            _profile(
                {
                    "text_sort": {
                        "cells": {
                            "cpu|image|siglip": {"steps": {"load_model": 90.0, "score": 10.0}},
                            "cpu||": {"steps": {"load_model": 1.0, "score": 99.0}},
                        }
                    }
                }
            ),
        )
        timing.reload_profile(path)
        exact = _weights("text_sort", device="cpu", media_type="image", embedder="siglip")
        rollup = _weights("text_sort", device="cpu", media_type="audio", embedder="clap")
        assert exact[0] > exact[2]  # the exact cell says the model load dominates
        assert rollup[0] < rollup[2]  # the rollup says scoring does

    def test_wildcard_spellings_are_equivalent(self, tmp_path):
        path = _write(tmp_path, _profile({"find": {"cells": {"cpu|*|*": {"steps": {"score": 100.0}}}}}))
        timing.reload_profile(path)
        weights = _weights("find", device="cpu", media_type="video", embedder="xclip")
        assert weights[2] > 0.9

    def test_unnamed_steps_keep_their_shipped_default(self, tmp_path):
        # A partial measurement improves the steps it covers without blanking
        # the ones it doesn't.
        path = _write(tmp_path, _profile({"text_sort": {"cells": {"cpu||": {"steps": {"load_model": 3.0}}}}}))
        timing.reload_profile(path)
        terms = _terms("text_sort", device="cpu")
        assert terms["load_model"] == pytest.approx(3.0)
        assert terms["embed_query"] == pytest.approx(0.05)  # shipped default
        assert terms["score"] == pytest.approx(0.20)  # shipped default


class TestAffineScaling:
    def test_same_task_paces_differently_at_different_sizes(self, tmp_path):
        # The whole point: an 8s encoder load is most of a 1000-item sort and a
        # rounding error on a 500k-item one. One static vector cannot say both.
        path = _write(
            tmp_path,
            _profile(
                {
                    "text_sort": {
                        "cells": {
                            "cpu||": {
                                "steps": {
                                    "load_model": {"a": 8.0},
                                    "embed_query": {"a": 0.04},
                                    "score": {"a": 0.1, "b": 0.0002},
                                }
                            }
                        }
                    }
                }
            ),
        )
        timing.reload_profile(path)
        small = _weights("text_sort", device="cpu", n=1_000)
        large = _weights("text_sort", device="cpu", n=500_000)
        assert small[0] > 0.9  # tiny sort: the model load is the whole job
        assert large[2] > 0.9  # huge sort: scoring is
        assert sum(small) == pytest.approx(1.0)
        assert sum(large) == pytest.approx(1.0)

    def test_byte_scaled_terms_track_archive_size(self):
        coeffs = StepCoeffs(per_mb=0.1)
        assert coeffs.seconds(n=999_999, size_mb=100.0) == pytest.approx(10.0)
        assert coeffs.seconds(n=999_999, size_mb=0.0) == pytest.approx(0.0)

    def test_negative_intercept_is_clamped(self):
        # A steep least-squares slope can overshoot into a negative intercept;
        # a step must never be handed a negative slice of the bar.
        assert StepCoeffs(a=-5.0, b=0.001).seconds(n=100) == 0.0

    def test_phases_sharing_a_tracker_step_are_summed(self, tmp_path):
        # dataset_load's download and extract both report against step 1, so the
        # weight vector is 4 long even though the cost model has 5 phases.
        path = _write(
            tmp_path,
            _profile(
                {
                    "dataset_load": {
                        "cells": {
                            "cpu||": {
                                "steps": {
                                    "download": {"a": 30.0},
                                    "extract": {"a": 10.0},
                                    "load": {"a": 10.0},
                                    "embed": {"a": 40.0},
                                    "finalize": {"a": 10.0},
                                }
                            }
                        }
                    }
                }
            ),
        )
        timing.reload_profile(path)
        weights = _weights("dataset_load", device="cpu", n=100)
        assert len(weights) == 4
        assert weights[0] == pytest.approx(0.4)  # download + extract share step 1


class TestSkippedSteps:
    """A step this run will not enter is priced at zero, not merely cheap.

    #3596: `text_sort`'s `load_model` is seconds on a process's first sort and
    exactly zero on the next 47, so no single coefficient paces both branches —
    every profile #3521 fitted, and the shipped defaults, sat at 0.80-0.85 bar
    error on the task. The caller can tell the branches apart before it starts;
    `skip_steps` is how it says so.
    """

    def test_a_skipped_step_gives_its_whole_slice_to_the_others(self):
        # Shipped defaults: (0.75, 0.05, 0.20). Warm, the load never happens.
        warm = _weights("text_sort", device="cpu", skip_steps=("load_model",))
        assert warm[0] == 0.0
        assert warm[1] == pytest.approx(0.2)
        assert warm[2] == pytest.approx(0.8)
        assert sum(warm) == pytest.approx(1.0)

    def test_a_measured_cell_is_skipped_the_same_way(self, tmp_path):
        path = _write(
            tmp_path,
            _profile(
                {
                    "text_sort": {
                        "cells": {
                            "cpu||": {
                                "steps": {
                                    # What a sweep of 47 warm sorts and one cold
                                    # one actually fits: a floored model load
                                    # that is most of the predicted total.
                                    "load_model": {"a": 0.5},
                                    "embed_query": {"a": 0.05},
                                    "score": {"a": 0.85},
                                }
                            }
                        }
                    }
                }
            ),
        )
        timing.reload_profile(path)
        cold = _terms("text_sort", device="cpu")
        warm = _terms("text_sort", device="cpu", skip_steps=("load_model",))
        assert cold["load_model"] == pytest.approx(0.5)
        assert warm["load_model"] == 0.0
        assert warm["score"] == cold["score"]  # the others are untouched
        # The floor was 36% of the predicted bar and 0% of the measured one.
        assert _weights("text_sort", device="cpu")[0] == pytest.approx(0.5 / 1.4)
        assert _weights("text_sort", device="cpu", skip_steps=("load_model",))[0] == 0.0

    def test_skipping_everything_falls_back_rather_than_dividing_by_zero(self):
        assert timing.step_weights("text_sort", device="cpu", skip_steps=TASKS["text_sort"].steps) is None
        assert timing.step_weights(
            "text_sort", device="cpu", skip_steps=TASKS["text_sort"].steps, fallback=[1.0, 0.0, 0.0]
        ) == [1.0, 0.0, 0.0]

    def test_an_unknown_step_name_is_inert(self):
        # Skipping is a claim about this run, not about the registry, so a name
        # that no longer exists must not blank a task's pacing.
        assert _weights("text_sort", device="cpu", skip_steps=("no_such_step",)) == pytest.approx(
            _weights("text_sort", device="cpu")
        )


class TestSlotShares:
    def test_slot_shares_resolve_through_the_same_cell_chain(self, tmp_path):
        path = _write(
            tmp_path,
            _profile(
                {
                    "dataset_load": {
                        "cells": {
                            "cpu|image|": {
                                "steps": {"finalize": {"a": 1.0}},
                                "slots": {"finalize": {"coverage": 0.8, "registry": 0.2}},
                            }
                        }
                    }
                }
            ),
        )
        timing.reload_profile(path)
        shares = _shares("dataset_load", "finalize", device="cpu", media_type="image")
        assert shares == {"coverage": 0.8, "registry": 0.2}
        assert timing.slot_shares("dataset_load", "finalize", device="cpu", media_type="audio") is None
        assert timing.slot_shares("dataset_load", "embed", device="cpu", media_type="image") is None


class TestMalformedProfilesAreHarmless:
    @pytest.mark.parametrize(
        "doc",
        [
            "not an object",
            {"schema": "something-else", "version": 1, "tasks": {}},
            {"schema": "vtsearch-timing-profile", "version": 99, "tasks": {}},
            {"schema": "vtsearch-timing-profile", "version": 0, "tasks": {}},
            {"schema": "vtsearch-timing-profile", "version": 1},
        ],
    )
    def test_unusable_documents_yield_an_empty_profile(self, doc):
        assert parse_profile(doc, source="test") is EMPTY_PROFILE

    def test_invalid_json_falls_back_to_defaults(self, tmp_path):
        path = tmp_path / "broken.json"
        path.write_text("{ this is not json", encoding="utf-8")
        timing.reload_profile(str(path))
        assert not timing.active_profile()
        assert _weights("text_sort", device="cpu") == pytest.approx([0.75, 0.05, 0.20])

    def test_missing_file_falls_back_to_defaults(self, tmp_path):
        timing.reload_profile(str(tmp_path / "nope.json"))
        assert not timing.active_profile()
        assert _weights("find", device="cpu") == pytest.approx([0.10, 0.30, 0.60])

    def test_unknown_task_and_step_are_dropped_not_fatal(self, tmp_path):
        path = _write(
            tmp_path,
            _profile(
                {
                    "task_from_the_future": {"cells": {"cpu||": {"steps": {"whatever": 1.0}}}},
                    "text_sort": {"cells": {"cpu||": {"steps": {"score": 5.0, "not_a_step": 99.0}}}},
                }
            ),
        )
        timing.reload_profile(path)
        terms = _terms("text_sort", device="cpu")
        assert terms["score"] == pytest.approx(5.0)
        assert set(terms) == set(TASKS["text_sort"].steps)

    def test_all_zero_cell_falls_back_rather_than_dividing_by_zero(self, tmp_path):
        path = _write(
            tmp_path,
            _profile({"find": {"cells": {"cpu||": {"steps": {"prepare": 0.0, "load": 0.0, "score": 0.0}}}}}),
        )
        timing.reload_profile(path)
        assert timing.step_terms("find", device="cpu") is None
        assert timing.step_weights("find", device="cpu", fallback=[1.0, 1.0, 1.0]) == [1.0, 1.0, 1.0]

    def test_bare_number_is_shorthand_for_a_fixed_cost(self):
        assert StepCoeffs.from_json(4) == StepCoeffs(a=4.0)
        assert StepCoeffs.from_json("nope") is None
        assert StepCoeffs.from_json({"a": "nope"}) is None
        assert StepCoeffs.from_json(True) is None
