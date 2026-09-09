"""The review set for the exhaustive pass: controls, and the name a reviewer sees (#3720).

The pass runs as 25 per-class passes over one dataset, on the owner's ruling
that holding one class in mind beats recognising twenty-five at once. Two things
in that build decide whether the result can be trusted afterwards:

* **the controls**, which are COCO-answered images mixed in so every pass scores
  itself. They have to be drawn from the anchored half (COCO answered for all
  eighty classes at once, so one control scores every pass it appears in), be
  disjoint from the queue, and be reproducible;
* **the detector name**, because it is the only string the app shows while
  voting, so it is the only place a class's rule can live where the reviewer
  meets it (#3612 -- `book` split over magazines because the rule lived in a
  manifest instead).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PILE_DIR = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "pile"


@pytest.fixture(scope="module")
def pass25():
    if str(_PILE_DIR) not in sys.path:
        sys.path.insert(0, str(_PILE_DIR))
    import make_pass25

    return make_pass25


@pytest.fixture(scope="module")
def pc():
    if str(_PILE_DIR) not in sys.path:
        sys.path.insert(0, str(_PILE_DIR))
    import pile_config

    return pile_config


def _anchored(cls: str, iid: int, scored: bool = True) -> dict:
    return {"id": iid, "categories": [f"{cls}@small"], "coco_scored": scored, "origin_name": f"/vg/{iid}.jpg"}


class TestDrawControls:
    def test_it_draws_the_asked_for_number_per_class(self, pass25, pc):
        cls = pc.SCALE_CLASSES[0]
        medias = {i: _anchored(cls, i) for i in range(100)}
        held = {i: [cls] for i in range(100)}

        chosen = pass25.draw_controls(medias, held, per_class=12)

        assert len(chosen) == 12
        assert all(v == [cls] for v in chosen.values())

    def test_only_coco_scored_images_can_be_controls(self, pass25, pc):
        """A control is a key, and only COCO answers for all 25 classes at once."""
        cls = pc.SCALE_CLASSES[0]
        medias = {i: _anchored(cls, i, scored=False) for i in range(50)}
        held = {i: [cls] for i in range(50)}

        assert pass25.draw_controls(medias, held, per_class=12) == {}

    def test_an_image_coco_cannot_answer_for_is_not_drawn(self, pass25, pc):
        """Anchored by the stamp but absent from the join is no answer at all."""
        cls = pc.SCALE_CLASSES[0]
        medias = {i: _anchored(cls, i) for i in range(50)}

        assert pass25.draw_controls(medias, {}, per_class=12) == {}

    def test_the_draw_is_a_function_of_the_seed(self, pass25, pc):
        cls = pc.SCALE_CLASSES[0]
        medias = {i: _anchored(cls, i) for i in range(200)}
        held = {i: [cls] for i in range(200)}

        first = pass25.draw_controls(medias, held, per_class=12, seed=7)
        again = pass25.draw_controls(medias, held, per_class=12, seed=7)
        other = pass25.draw_controls(medias, held, per_class=12, seed=8)

        assert first == again
        assert set(first) != set(other)

    def test_one_image_is_never_drawn_twice(self, pass25, pc):
        """An image positive for two classes must not be counted as two controls."""
        a, b = pc.SCALE_CLASSES[0], pc.SCALE_CLASSES[1]
        medias = {i: {"id": i, "categories": [f"{a}@small", f"{b}@large"], "coco_scored": True} for i in range(30)}
        held = {i: [a, b] for i in range(30)}

        chosen = pass25.draw_controls(medias, held, per_class=12)

        assert len(chosen) == len(set(chosen))

    def test_it_carries_every_class_coco_says_is_present(self, pass25, pc):
        """The key is the whole answer, not just the class the image was drawn for."""
        a, b = pc.SCALE_CLASSES[0], pc.SCALE_CLASSES[1]
        medias = {1: _anchored(a, 1)}

        chosen = pass25.draw_controls(medias, {1: [a, b]}, per_class=1)

        assert chosen[1] == [a, b]


class TestDetectorName:
    def test_a_ruled_class_shows_its_rule(self, pass25, pc):
        """The rule is what the reviewer reads while voting, so it is the name."""
        ruled = next(c for c in pc.SCALE_CLASSES if pc.SCALE_CLASS_RULES.get(c))

        assert pass25.detector_name(ruled) == pc.SCALE_CLASS_RULES[ruled].name

    def test_an_unruled_class_falls_back_to_its_bare_name(self, pass25):
        """An unruled class falls back rather than crashing (#3673).

        Every class of the twenty-five carries a rule now that #3771 ruled
        `dog`'s wolf boundary, so the fallback is exercised with a name that
        is not one of them.
        """
        assert pass25.detector_name("nonesuch") == "nonesuch"

    def test_every_class_yields_a_non_empty_name(self, pass25, pc):
        for cls in pc.SCALE_CLASSES:
            assert pass25.detector_name(cls).strip()
