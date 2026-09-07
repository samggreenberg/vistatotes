"""The markdown-only fast path: which tests can observe a repo markdown file.

``run-tests.sh`` prunes itself when a branch changes nothing but tracked
markdown.  Almost nothing in this repo can see a ``.md`` file: pyright
excludes markdown, pip-audit and the frontend lanes cannot read one, and of
the ~1600 Python tests only a handful ever open a doc.  Paying ~3.5 minutes
to prove that a plan file's prose still lints is a bad trade, so a
markdown-only run keeps every cheap gate in stage 1 (they are the ones that
*do* read markdown -- check-docs, codespell, the doc-inventory and
screenshot-wiring snapshots, the deck preflight) and narrows pytest to the
tests below.

The exemption is only sound while ``MARKDOWN_TEST_SURFACE`` really is every
test that can observe a doc.  That premise is mechanically checkable, so
``tests_lib/meta/test_markdown_surface.py`` checks it rather than trusting
it -- the failure mode is otherwise silent, because a doc-reading test added
to, say, ``tests/sorting/`` simply stops running on doc changes and nothing
anywhere says so.

Both constants are read by the runner (via ``python -c``) and by that gate,
so the two cannot disagree.
"""

from __future__ import annotations

# The pytest selection a markdown-only run uses, as paths rather than markers:
# the surface is one folder plus one file, and no marker expression names that.
#
#   tests_lib/meta              the repo's own tooling tests -- the docs gate,
#                               the extension-docs gate, the USER_GUIDE anchor
#                               checker.  Every deliberate reader of a repo doc
#                               lives here.
#   tests/core/test_achievements.py
#                               the one outlier, and the reason this file
#                               exists: the Readme Reader achievement pins a
#                               phrase inside README.md, docs/user/USER_GUIDE.md,
#                               docs/CLI.md and docs/API.md, so editing any of
#                               them can fail a test in `core`.  It reaches the
#                               docs through vtsearch.achievements_catalog, so
#                               no scan of the test file's own literals would
#                               ever have found it.
MARKDOWN_TEST_SURFACE: tuple[str, ...] = (
    "tests_lib/meta",
    "tests/core/test_achievements.py",
)

# First-party Python that embeds a tracked repo markdown path, and so can hand
# a doc to a test that never names one itself.  Registration is not paperwork:
# the gate rejects an unregistered module carrying a doc path (so the list
# cannot silently fall behind), and requires every test that *references* a
# registered module to sit in the surface above.
#
# The four scripts are stage-1 gates, which a markdown-only run keeps in full;
# they are listed because their self-tests reach the docs through them.
DOC_READING_SOURCES: tuple[str, ...] = (
    "vtsearch/achievements_catalog.py",
    "scripts/check-calibration-index.py",
    "scripts/check-docs.py",
    "scripts/check-extension-docs.py",
    "scripts/screenshots/wiring-check.py",
)
