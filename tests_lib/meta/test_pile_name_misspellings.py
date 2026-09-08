"""The spellings a head-noun family cannot reach, and the guards on that search (#3663).

`vg_name_families.py` finds candidates by head noun, which cannot reach a
misspelling: `umberella` shares no token with `umbrella`. `name_misspellings.py`
closes that hole with an edit-distance search, and the whole risk in a search
like this is that it returns the dictionary. Two guards keep it honest, and both
are here because the measurement showed what happens without them:

* a **length floor**, because one edit from a four-letter class name is most of
  English -- `cup` would reach `cap`, `cut`, `can`, `cop`;
* **the shipped tables are excluded**, so the output is only what the curation
  actually dropped, which is the number #3663 asks for.

The result the search produced is that no candidate clears the evidence cuts:
the real misspellings carry 3-5 images each, and everything else is a different
common word (`pole` at 14,372 images is one edit from `phone`).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PILE_DIR = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "pile"


@pytest.fixture(scope="module")
def misspell():
    """``name_misspellings``, imported for its pure halves.

    It calls ``pc.setup_env()`` at import like its neighbours, which is fine
    here: the functions under test read nothing but their arguments.
    """
    if str(_PILE_DIR) not in sys.path:
        sys.path.insert(0, str(_PILE_DIR))
    import name_misspellings

    return name_misspellings


class TestEditDistance:
    def test_it_counts_the_edits(self, misspell):
        # Every misspelling this search was built for is ONE edit from the class
        # name -- umb-e-rella, umbre-a-lla, bicy[c]le -- which is why a cap of 2
        # is generous rather than tight, and why the noise it lets in (`pole` is
        # two from `phone`, on 14,372 images) is what the evidence cuts must
        # reject. I first asserted 2 for `umberella` here; the function was right
        # and the test was wrong, which is the reason to spell the edits out.
        assert misspell.edit_distance("umbrella", "umberella") == 1
        assert misspell.edit_distance("umbrella", "umbrealla") == 1
        assert misspell.edit_distance("bicycle", "bicyle") == 1
        assert misspell.edit_distance("phone", "pole") == 2

    def test_identical_strings_are_zero(self, misspell):
        assert misspell.edit_distance("clock", "clock") == 0

    def test_it_stops_once_the_cap_is_passed(self, misspell):
        """The cap is what makes a scan over 100k names cheap; it must not lie below it."""
        assert misspell.edit_distance("umbrella", "fire hydrant", cap=2) > 2
        assert misspell.edit_distance("cup", "cap", cap=2) == 1


class TestNearMisses:
    VOCAB = {
        "umbrella": 900,
        "umberella": 4,
        "unbrella": 3,
        "parasol": 60,
        "umbrellas": 120,
        "elephant": 500,
        "rare typo": 1,
    }

    def test_it_finds_the_misspellings_a_family_cannot(self, misspell):
        out = misspell.near_misses(
            self.VOCAB, {"umbrella": {"umbrella"}}, shipped=set(), max_edits=2, min_len=5, min_images=3
        )

        assert "umberella" in out["umbrella"]
        assert "unbrella" in out["umbrella"]

    def test_the_class_name_is_not_its_own_candidate(self, misspell):
        out = misspell.near_misses(
            self.VOCAB, {"umbrella": {"umbrella"}}, shipped=set(), max_edits=2, min_len=5, min_images=3
        )

        assert "umbrella" not in out["umbrella"]

    def test_a_shipped_name_is_not_a_finding(self, misspell):
        """The output has to be what curation dropped, not what it kept."""
        out = misspell.near_misses(
            self.VOCAB, {"umbrella": {"umbrella"}}, shipped={"umbrellas"}, max_edits=2, min_len=5, min_images=3
        )

        assert "umbrellas" not in out["umbrella"]

    def test_a_name_below_the_image_floor_is_dropped(self, misspell):
        """A one-image name cannot be evidence of anything, so it is not a candidate."""
        out = misspell.near_misses(
            self.VOCAB, {"umbrella": {"umbrella"}}, shipped=set(), max_edits=2, min_len=5, min_images=3
        )

        assert "rare typo" not in out["umbrella"]

    def test_a_short_target_contributes_nothing(self, misspell):
        """At four characters an edit is too cheap: `cup` would reach half of English."""
        vocab = {"cap": 100, "cut": 100, "cop": 100}

        out = misspell.near_misses(vocab, {"cup": {"cup"}}, shipped=set(), max_edits=2, min_len=5, min_images=3)

        assert out["cup"] == []

    def test_it_searches_from_the_aliases_too(self, misspell):
        """`hyrdant` is close to the alias `hydrant`, not to `fire hydrant`."""
        vocab = {"hyrdant": 3, "hydrant": 600}

        out = misspell.near_misses(
            vocab,
            {"fire hydrant": {"fire hydrant", "hydrant"}},
            shipped={"hydrant"},
            max_edits=2,
            min_len=5,
            min_images=3,
        )

        assert out["fire hydrant"] == ["hyrdant"]
