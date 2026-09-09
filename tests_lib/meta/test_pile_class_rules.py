"""A class's review rule reaches the reviewer, and lives in one place (#3612).

`vg_scale`'s slates are voted on bare images: files are named by image id, and
the reviewer never sees a manifest. So for a class whose plain English name is
not the whole question, the **dataset name is the entire brief**, and until
`pile_config.SCALE_CLASS_RULES` existed that name was typed by hand at slate
time and recorded nowhere.

Two failures follow from that, and both have happened:

* `book` split -- COCO annotates magazines as `book`, the human pass applied the
  narrower reading, and 21 verdicts landed on one definition against 49 on
  another.
* `cell phone`'s first slate was voted under a test that read "anything with a
  cord or a base station is Bad". That discriminates on a base being *present*
  when what it means is that the handset is not itself the whole device, so it
  rejected image 2387021 -- a mobile phone in a charging dock.

These pin the properties that stop the second one recurring: the wording is
recorded, it is what a re-review is issued under, and every slate maker builds
its detector name from it rather than from a format string of its own.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

_PILE_DIR = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "pile"

#: Every script that names a dataset a human votes in.
_SLATE_MAKERS = (
    "make_audit_slate.py",
    "make_positive_slate.py",
    "make_audit_pass.py",
    "make_definition_reslate.py",
)


@pytest.fixture(scope="module")
def pc():
    """``pile_config``, which is constants only -- ``setup_env()`` is explicit."""
    if str(_PILE_DIR) not in sys.path:
        sys.path.insert(0, str(_PILE_DIR))
    import pile_config

    return pile_config


def test_review_name_falls_back_to_the_bare_class(pc):
    """A class without a rule is its own definition, and keeps the old name.

    `dog` used to be the example, as the last of the twenty-five with no
    written rule; #3771 ruled its wolf boundary and gave it one, so every
    class *in the table* now has a name of its own. The fallback branch is
    still live -- `review_name` is called for any class, and a COCO class
    outside *C* has no entry -- so the example moved rather than the test
    going away.
    """
    assert "cat" not in pc.SCALE_CLASS_RULES
    assert pc.review_name("cat") == "cat"
    assert pc.review_name("cat", "positives") == "cat positives"


def test_review_name_carries_the_rule_through_every_pass(pc):
    """The suffix names the pass; the rule survives it."""
    assert pc.review_name("cell phone") == "cell phone not landlines"
    assert pc.review_name("cell phone", "audit") == "cell phone not landlines audit"


def test_rule_names_are_usable_as_a_detector_and_a_folder(pc):
    """The name becomes a directory (``name.replace(" ", "_")``) and an API path."""
    for cls, rule in pc.SCALE_CLASS_RULES.items():
        assert rule.name == rule.name.strip() and rule.name
        assert not set(rule.name) & set("/\\"), f"{cls}: path separator in {rule.name!r}"
        # Letters, digits and spaces only -- no punctuation to mangle a folder
        # name or an API path. Case is *not* constrained: `truck incl vans not
        # SUVs` and `car incl SUVs and minivans` carry an acronym that reads
        # wrong lowercased, and neither a directory nor a path minds it.
        assert re.fullmatch(r"[A-Za-z0-9 ]+", rule.name), f"{cls}: {rule.name!r} is not plain alphanumeric"
        # `ingest_slate.py` attributes an export by matching the slate folder in
        # the origin path, and a human has to recognise the class at a glance.
        assert rule.name.startswith(cls), f"{cls}: {rule.name!r} does not name its class"


def test_no_two_passes_of_one_class_share_a_detector(pc):
    """``ingest_slate.py`` keys a manifest row by (image, class, detector).

    Two slates of one class under one name would overwrite each other's rows,
    silently dropping a whole review pass. The suffixes are what separate them,
    so a rule name must not collide with another class's bare name either.
    """
    seen: dict[str, str] = {}
    classes = set(pc.SCALE_CLASSES) | set(pc.SCALE_CLASS_RULES)
    for cls in sorted(classes):
        for suffix in ("", "positives", "audit", "reviewed"):
            name = pc.review_name(cls, suffix)
            assert name not in seen, f"{cls}/{suffix!r} collides with {seen[name]}"
            seen[name] = f"{cls}/{suffix!r}"


def test_the_cell_phone_rule_admits_a_docked_mobile(pc):
    """The clause that rejected 2387021, stated as the dependency it means.

    "A base is present" and "the handset needs the base" are the same sentence
    about a desk phone and opposite sentences about a phone on a charger, which
    is the whole of #3612.
    """
    test = pc.SCALE_CLASS_RULES["cell phone"].test.lower()
    assert "needs the base" in test, "the Bad clause must turn on dependency, not presence"
    assert "dock" in test and "cradle" in test, "the near-miss must be settled explicitly"
    assert "landline" in test, "the clause still has to exclude landlines"


def test_the_dog_rule_turns_on_wild_versus_domestic(pc):
    """#3771's ruling, pinned as the discrimination it actually is.

    A husky and a wolf are the same silhouette, so a rule that read `not a
    wolf` and stopped would reject the pet husky it was never meant to. The
    ruling is *obviously wild*, on the same "obvious is the operative word"
    footing as the protocol's toy rule -- and the figurine and depiction cases
    the issue raised are deliberately NOT restated here, because they are
    already binding on every class.
    """
    rule = pc.SCALE_CLASS_RULES["dog"]
    test = rule.test.lower()
    assert "obviously wild" in test, "the Bad clause must turn on wildness, not on looks"
    assert "husky" in test, "the wolf-like breed the ruling protects has to be named"
    assert "domestic" in test
    # The generic rules stay generic: the entry may point at them, but it must
    # not re-rule them, which is how one class's wording starts to drift from
    # the protocol every other class is voted under.
    assert "depiction is not the object" in test and "obvious toy is not the object" in test


def test_every_slate_maker_names_its_detector_from_the_table():
    """No script may format a detector name of its own.

    A pass that builds ``f"{cls} positives"`` inline is a pass the rule never
    reaches -- which is exactly how the first `cell phone` slate was voted.
    """
    for script in _SLATE_MAKERS:
        src = (_PILE_DIR / script).read_text()
        assert "pc.review_name(" in src, f"{script} does not use pile_config.review_name"
        stray = re.findall(r'f"\{cls\}[^"]*"|f"\{c\}[^"]*"', src)
        assert not stray, f"{script} formats a detector name inline: {stray}"
