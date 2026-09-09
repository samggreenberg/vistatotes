"""The deep-spike rule must mean the same thing in every script that reports it.

`analyze_spikes.py` owns the guardrail: a cell has a deep spike when, after the
warm-up, its cost clears ``DEEP_COST`` *and* exceeds the oracle's cost on the
same ranking by ``DEEP_EXCESS``. #3547's analyzers restate those constants
rather than importing them, deliberately -- each file should state the guardrail
it reports -- and `frontier_3547.py` has always carried a comment saying a test
keeps them in sync. This is that test.

It matters because the constants are what make two studies comparable. #3547's
central H2 claim compares a 5.7% incidence against a 1.0% one measured by a
different script; if one of them drifted to a different threshold, the
comparison would be silently meaningless rather than loudly broken.

Copies are discovered by scanning, not listed, so a *new* restatement of the
rule is covered the moment it is written.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

RULE_NAMES = ("WARM_T", "DEEP_COST", "DEEP_EXCESS")
CALIBRATION = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "calibration"
#: The file the others must agree with.
OWNER = "analyze_spikes.py"


def _literal(node: ast.AST) -> float | None:
    """The numeric value of *node*, seeing through the `os.environ.get` default.

    `analyze_spikes.py` writes the rule as an env-overridable default --
    ``int(os.environ.get("SPIKE_WARM_T", "20"))`` -- so the constant that has to
    match is the string literal inside, not the call.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Call):
        # int(...) / float(...) wrapping os.environ.get(NAME, DEFAULT)
        if node.args:
            inner = node.args[0]
            if isinstance(inner, ast.Call) and len(inner.args) >= 2:
                default = inner.args[1]
                if isinstance(default, ast.Constant) and isinstance(default.value, (int, float, str)):
                    return float(default.value)
            return _literal(inner)
    return None


def _rule_in(path: Path) -> dict[str, float]:
    """The three rule constants defined at module level in *path*, if any."""
    found: dict[str, float] = {}
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        # `WARM_T, DEEP_COST, DEEP_EXCESS = 20, 0.25, 0.20`
        target = node.targets[0]
        if isinstance(target, ast.Tuple) and isinstance(node.value, ast.Tuple):
            for name_node, val_node in zip(target.elts, node.value.elts):
                if isinstance(name_node, ast.Name) and name_node.id in RULE_NAMES:
                    v = _literal(val_node)
                    if v is not None:
                        found[name_node.id] = v
        # `WARM_T = int(os.environ.get("SPIKE_WARM_T", "20"))`
        elif isinstance(target, ast.Name) and target.id in RULE_NAMES:
            v = _literal(node.value)
            if v is not None:
                found[target.id] = v
    return found


def _copies() -> dict[str, dict[str, float]]:
    return {
        p.name: rule for p in sorted(CALIBRATION.glob("*.py")) if (rule := _rule_in(p)) and set(rule) == set(RULE_NAMES)
    }


@pytest.mark.skipif(not CALIBRATION.is_dir(), reason="calibration scripts not present")
def test_owner_states_the_rule() -> None:
    """The source of truth parses, so a drifted copy fails loudly rather than silently."""
    owner = _rule_in(CALIBRATION / OWNER)
    assert set(owner) == set(RULE_NAMES), f"{OWNER} no longer states the rule as parsed: {owner}"


@pytest.mark.skipif(not CALIBRATION.is_dir(), reason="calibration scripts not present")
def test_frontier_3547_spike_rule_matches_analyze_spikes() -> None:
    """Every script restating the deep-spike rule agrees with `analyze_spikes.py`."""
    copies = _copies()
    assert OWNER in copies, f"{OWNER} must define {RULE_NAMES}"
    owner = copies[OWNER]

    # The #3547 analyzers are the reason this test exists; assert they are
    # actually among the files scanned, so a rename cannot quietly empty it out.
    assert "frontier_3547.py" in copies, "frontier_3547.py no longer states the rule"

    disagree = {name: rule for name, rule in copies.items() if rule != owner}
    assert not disagree, (
        f"deep-spike rule drifted from {OWNER} ({owner}). "
        f"Offending copies: {disagree}. An incidence measured under a different "
        f"threshold is not comparable to one measured under this rule."
    )
