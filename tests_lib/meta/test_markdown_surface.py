"""The markdown-only fast path's premise: which tests can see a repo doc.

``run-tests.sh`` narrows itself when a branch changes nothing but tracked
markdown — every cheap gate still runs, but pytest is cut to
``tests_shared.markdown_surface.MARKDOWN_TEST_SURFACE``.  That is sound only
while the surface really is every test that can observe a doc, and the failure
mode if it stops being true is silent: a doc-reading test added to, say,
``tests/sorting/`` would simply stop running on doc changes, and the run would
still print ``RUN PASSED``.

So the premise is checked rather than trusted.  A test reaches a doc in one of
two shapes, and each gets a rule:

``DIRECT``
    the test names the file itself (``REPO_ROOT / "docs" / "API.md"``).  Caught
    by scanning code string literals for tracked markdown paths.

``INDIRECT``
    the test imports something that carries the path, and never names a doc at
    all.  This is the shape that actually bit: ``tests/core/test_achievements.py``
    reads four repo docs through ``vtsearch.achievements_catalog``, and no scan
    of its own literals would ever have found it.  Caught by registering the
    first-party modules that embed doc paths and banning non-surface tests from
    referencing them.

The registry is itself checked for completeness, so it cannot quietly fall
behind: a first-party module that starts naming a tracked doc fails this gate
until it is registered, and registration is what arms the INDIRECT rule.

What the scan cannot see, and does not claim to: a doc path assembled from
separate path segments (``ROOT / "docs" / "user" / "USER_GUIDE.md"``) is only
matched on its final segment, so a *new* module reading a doc that way is
caught only when that segment is a repo-root markdown file.  The registry is
curated for exactly that reason; these rules are a net under it, not a proof.
"""

from __future__ import annotations

import ast
import importlib.util
import re
import sys
from pathlib import Path

import pytest

from tests_shared.markdown_surface import DOC_READING_SOURCES, MARKDOWN_TEST_SURFACE

REPO_ROOT = Path(__file__).resolve().parents[2]

# First-party Python that ships or tools the repo. Test trees are scanned
# separately (they are the subject of the INDIRECT rule, not its input).
SOURCE_ROOTS = ("vtsearch/", "vtscore/", "scripts/", "app.py")
TEST_ROOTS = ("tests/", "tests_lib/")

# A bare filename like "README.md" is ambiguous — vtscore/detectors/
# portable_bundle.py writes one *into a zip*, which is not a repo read. Require
# corroborating evidence that the file addresses the repo from its root before
# treating a bare name as a doc read.
_REPO_ROOT_ANCHOR = re.compile(r"\bREPO_ROOT\b|parents\[\d+\]|^\s*ROOT\s*=", re.MULTILINE)


def _load_docs_gate():
    """`scripts/check-docs.py`, for its tracked-file enumeration."""
    spec = importlib.util.spec_from_file_location("_docs_gate_for_surface", REPO_ROOT / "scripts" / "check-docs.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_GATE = _load_docs_gate()
TRACKED: list[str] = _GATE.tracked_files()
TRACKED_MD = {p for p in TRACKED if p.endswith(".md")}
ROOT_MD = {p for p in TRACKED_MD if "/" not in p}


def _code_strings(tree: ast.AST) -> list[str]:
    """Every string constant except docstrings.

    Docstrings are excluded because prose cites paths constantly — a test that
    merely *mentions* `scripts/check-extension-docs.py` in its module docstring
    is not reading a doc, and flagging it would train people to route around
    this gate.  Comments never reach the AST at all.
    """
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            first = node.body[0] if node.body else None
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                docstrings.add(id(first.value))
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings
    ]


def _doc_paths_named(path: Path, text: str) -> set[str]:
    """Tracked markdown paths this file names in code (see the module docstring)."""
    try:
        tree = ast.parse(text)
    except SyntaxError:  # pragma: no cover - the repo does not carry unparseable Python
        return set()
    anchored = bool(_REPO_ROOT_ANCHOR.search(text))
    found = set()
    for literal in _code_strings(tree):
        if "/" in literal and literal in TRACKED_MD:
            found.add(literal)
        elif anchored and literal in ROOT_MD:
            found.add(literal)
    return found


def _python_under(prefixes: tuple[str, ...]) -> list[str]:
    return [p for p in TRACKED if p.endswith(".py") and p.startswith(prefixes)]


def _in_surface(rel: str) -> bool:
    return any(rel == entry or rel.startswith(entry.rstrip("/") + "/") for entry in MARKDOWN_TEST_SURFACE)


def _reference_tokens(source: str) -> set[str]:
    """How a test could name `source`: its repo path, or its dotted module path."""
    tokens = {source}
    if not source.startswith("scripts/"):
        tokens.add(source.removesuffix(".py").replace("/", "."))
    return tokens


class TestSurfaceIsWellFormed:
    def test_every_surface_entry_exists(self):
        for entry in MARKDOWN_TEST_SURFACE:
            assert (REPO_ROOT / entry).exists(), (
                f"MARKDOWN_TEST_SURFACE names {entry}, which no longer exists. "
                f"run-tests.sh hands this list straight to pytest, so a stale "
                f"entry fails every markdown-only run."
            )

    def test_every_registered_source_exists(self):
        for source in DOC_READING_SOURCES:
            assert (REPO_ROOT / source).exists(), (
                f"DOC_READING_SOURCES names {source}, which no longer exists; drop the entry."
            )

    def test_run_tests_reads_the_surface(self):
        """The runner must take its selection from here, not from a copy."""
        text = (REPO_ROOT / "run-tests.sh").read_text(encoding="utf-8")
        assert "tests_shared.markdown_surface" in text and "MARKDOWN_TEST_SURFACE" in text, (
            "run-tests.sh no longer reads MARKDOWN_TEST_SURFACE from "
            "tests_shared/markdown_surface.py. A second copy of the list is how "
            "this gate stops describing what actually runs."
        )


class TestRegistryIsComplete:
    """A first-party module that names a repo doc must be registered."""

    def test_no_unregistered_doc_reader(self):
        offenders = {}
        for rel in _python_under(SOURCE_ROOTS):
            if rel in DOC_READING_SOURCES:
                continue
            named = _doc_paths_named(REPO_ROOT / rel, (REPO_ROOT / rel).read_text(encoding="utf-8", errors="ignore"))
            if named:
                offenders[rel] = sorted(named)
        assert not offenders, (
            "These files name a tracked repo doc but are not in DOC_READING_SOURCES:\n"
            + "\n".join(f"  {k}: {', '.join(v)}" for k, v in offenders.items())
            + "\n\nAdd them to tests_shared/markdown_surface.py. Any test that "
            "references one then has to sit in MARKDOWN_TEST_SURFACE, which is "
            "the point: that test can be broken by a markdown-only change."
        )


class TestNoDocReaderOutsideTheSurface:
    """The rules that make the fast path's pytest selection true."""

    def test_no_direct_doc_read_outside_the_surface(self):
        offenders = {}
        for rel in _python_under(TEST_ROOTS):
            if _in_surface(rel):
                continue
            named = _doc_paths_named(REPO_ROOT / rel, (REPO_ROOT / rel).read_text(encoding="utf-8", errors="ignore"))
            if named:
                offenders[rel] = sorted(named)
        assert not offenders, (
            "These tests read a repo doc but are outside MARKDOWN_TEST_SURFACE, so a "
            "markdown-only run would not run them:\n"
            + "\n".join(f"  {k}: {', '.join(v)}" for k, v in offenders.items())
            + "\n\nEither move the test into tests_lib/meta/, or add its path to "
            "MARKDOWN_TEST_SURFACE in tests_shared/markdown_surface.py."
        )

    def test_no_indirect_doc_read_outside_the_surface(self):
        tokens = {source: _reference_tokens(source) for source in DOC_READING_SOURCES}
        offenders = {}
        for rel in _python_under(TEST_ROOTS):
            if _in_surface(rel):
                continue
            text = (REPO_ROOT / rel).read_text(encoding="utf-8", errors="ignore")
            try:
                tree = ast.parse(text)
            except SyntaxError:  # pragma: no cover
                continue
            referenced = set()
            for literal in _code_strings(tree):
                for source, toks in tokens.items():
                    if any(tok in literal for tok in toks):
                        referenced.add(source)
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module] + [f"{node.module}.{alias.name}" for alias in node.names]
                for name in names:
                    for source, toks in tokens.items():
                        if any(name == tok or name.startswith(tok + ".") for tok in toks):
                            referenced.add(source)
            if referenced:
                offenders[rel] = sorted(referenced)
        assert not offenders, (
            "These tests reach a repo doc through a registered module but sit outside "
            "MARKDOWN_TEST_SURFACE, so a markdown-only run would not run them:\n"
            + "\n".join(f"  {k}: via {', '.join(v)}" for k, v in offenders.items())
            + "\n\nAdd the test to MARKDOWN_TEST_SURFACE, or stop reaching the doc."
        )


class TestTheRulesActuallyFire:
    """A gate nobody has seen fail is a gate nobody knows is wired up."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ('P = "docs/API.md"', {"docs/API.md"}),
            ('REPO_ROOT = x\nP = "README.md"', {"README.md"}),
            # A bare name with nothing addressing the repo root: a zip entry,
            # not a doc read (vtscore/detectors/portable_bundle.py).
            ('with zf: zf.write("README.md")', set()),
            # Prose is not a read.
            ('"""See docs/API.md for the reference."""', set()),
            # An output file that happens to end in .md.
            ('OUT = "REPORT.md"', set()),
        ],
    )
    def test_direct_rule(self, text, expected):
        assert _doc_paths_named(REPO_ROOT / "__synthetic__.py", text) == expected

    def test_reference_tokens_cover_both_shapes(self):
        assert _reference_tokens("vtsearch/achievements_catalog.py") == {
            "vtsearch/achievements_catalog.py",
            "vtsearch.achievements_catalog",
        }
        # Scripts are loaded by path, never imported by a dotted name.
        assert _reference_tokens("scripts/check-docs.py") == {"scripts/check-docs.py"}
