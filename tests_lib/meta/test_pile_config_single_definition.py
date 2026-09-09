"""``pile_config`` must define each module-level table exactly once (#3625).

Two pull requests each added a table called ``SCALE_CLASS_RULES`` to this file --
one a ``dict[str, str]`` of names, the other a ``dict[str, ClassRule]`` carrying
the name *and* the test. They touched different regions, so git merged them
without a conflict and both landed. Python bound the later one, which meant:

* every entry of the earlier table silently vanished;
* ``review_name`` did ``rule.name`` on a ``str`` and raised ``AttributeError``
  for every class that had a rule -- the thirteen the slate builders use;
* ``review_name("book")`` returned ``"book"``, quietly dropping the rule whose
  loss is the whole reason the table exists.

Nothing in review catches this: each diff is correct in isolation, the merge is
clean, and the file imports fine. A source-level check does catch it, and costs
nothing, so it guards every table here rather than just the one that broke.
"""

from __future__ import annotations

import ast
import collections
from pathlib import Path

import pytest

_PILE_CONFIG = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "pile" / "pile_config.py"


def _module_level_assignments() -> collections.Counter:
    """Every name bound by a top-level assignment, counted."""
    tree = ast.parse(_PILE_CONFIG.read_text())
    names: collections.Counter = collections.Counter()
    for node in tree.body:
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        for t in targets:
            if isinstance(t, ast.Name):
                names[t.id] += 1
    return names


def test_no_module_level_name_is_assigned_twice():
    """A second binding silently discards the first, table and all."""
    dupes = {n: c for n, c in _module_level_assignments().items() if c > 1}
    assert not dupes, (
        "pile_config.py binds these module-level names more than once, so only the "
        f"last one survives: {sorted(dupes)}. Merge them into a single definition "
        "rather than letting one shadow the other."
    )


def test_scale_class_rules_is_defined_exactly_once():
    """The specific table that broke, named so a failure says what to look at."""
    assert _module_level_assignments()["SCALE_CLASS_RULES"] == 1


def _duplicate_dict_keys() -> dict[str, list[str]]:
    """``{table: [key repeated, ...]}`` for every module-level dict literal here."""
    tree = ast.parse(_PILE_CONFIG.read_text())
    out: dict[str, list[str]] = {}
    for node in tree.body:
        target, value = None, None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target, value = node.target.id, node.value
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target, value = node.targets[0].id, node.value
        if target is None or not isinstance(value, ast.Dict):
            continue
        seen: set = set()
        dupes: list[str] = []
        for k in value.keys:
            if isinstance(k, ast.Constant):
                if k.value in seen:
                    dupes.append(str(k.value))
                seen.add(k.value)
        if dupes:
            out[target] = dupes
    return out


def test_no_dict_literal_repeats_a_key():
    """A repeated key keeps the LAST value, which is how a merge eats a rule.

    The sibling test above catches a whole table being declared twice. This
    catches the same shadowing one level down, and it is the form that actually
    bit: #3588 filled every candidate's ``test`` while another branch added the
    same twelve classes as name-only stubs, and git merged both sets into one
    literal. The file parsed, the dict had the right number of *keys*, and every
    rule written during the review had been silently replaced by an empty one.
    """
    dupes = _duplicate_dict_keys()
    assert not dupes, (
        "these module-level dicts repeat a key, so only the last value survives: "
        f"{dupes}. Two branches adding the same entry is the usual cause, and the "
        "merge is clean because the keys are on different lines."
    )


@pytest.fixture(scope="module")
def pc():
    import sys

    sys.path.insert(0, str(_PILE_CONFIG.parent))
    import pile_config

    return pile_config


def test_every_rule_is_a_ClassRule_not_a_bare_name(pc):
    """The shadowing bug presented as ``str`` where a ``ClassRule`` was expected."""
    wrong = sorted(k for k, v in pc.SCALE_CLASS_RULES.items() if isinstance(v, str))
    assert not wrong, f"these rules are bare strings, not ClassRule: {wrong}"


def test_review_name_resolves_for_every_class_that_has_a_rule(pc):
    """`review_name` raised AttributeError on all thirteen while shadowed."""
    for cls, rule in pc.SCALE_CLASS_RULES.items():
        assert pc.review_name(cls) == rule.name
        assert pc.scale_class_dataset_name(cls) == rule.name


def test_both_accessors_agree(pc):
    """Two spellings, one table -- that is what the collision broke."""
    for cls in (*pc.SCALE_CLASS_RULES, "dog", "kite"):
        assert pc.review_name(cls) == pc.scale_class_dataset_name(cls)
