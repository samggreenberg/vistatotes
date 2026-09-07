"""The embed batch size has to reach the provenance sidecar (#3683).

``VTSEARCH_EMBED_BATCH_SIZE`` is a build parameter that changes the vectors.
Rebuilding ``siglip2_l`` at 31 instead of its configured 32 -- same images, same
node, everything else equal -- moved 27 of 7,746 vectors by up to 1.6e-04,
because the batched GEMM's reduction order is not independent of what an image
was batched with. That is 400x the same-node rebuild floor, and larger than the
fp16 difference #3143 measured and rejected, so a cell built at one size and a
cell built at another are not the same cell -- and until #3683 the sidecar said
nothing about which one you were holding.

The plumbing is what these tests pin, because the value is only *legible* for
the length of one ``with`` block. ``build_pile._embed_batch_size`` sets the env
var for the embed pass and pops it afterwards, so a later reader -- provenance
is written afterwards -- asking the environment gets the shipped default of 32
back and records a size the pass never ran at. That is the same failure the
``aten_cpu_capability_requested`` comment warns about: a recorded pin that never
took is worse than no record, because it reads as evidence.

Source-level, so it costs nothing and needs neither a GPU nor the 100 GB of
sources a real build reads. Importing ``build_pile`` would run
``pile_config.setup_env()``, which edits ``sys.meta_path`` process-wide; parsing
is what lets this live beside the other cheap pile checks.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_PILE = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "pile"
_PROVENANCE = _PILE / "pilebuild" / "provenance.py"
_REPORT = _PILE / "pilebuild" / "provenance_report.py"
_BUILD = _PILE / "build_pile.py"

KEY = "embed_batch_size"


def _func(path: Path, name: str) -> ast.FunctionDef:
    """The named top-level function, or a failure that says which file lacked it."""
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    pytest.fail(f"{path.name} no longer defines {name}()")


def _params(fn: ast.FunctionDef) -> list[str]:
    a = fn.args
    return [p.arg for p in (*a.posonlyargs, *a.args, *a.kwonlyargs)]


def _calls(fn: ast.FunctionDef, name: str) -> list[ast.Call]:
    return [n for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == name]


def test_device_record_writes_the_key_from_its_argument():
    """The sidecar carries the size, and carries the one it was *handed*.

    Reading the environment here instead would answer with the default, because
    the build pops the env var before provenance is written.
    """
    fn = _func(_PROVENANCE, "_device_record")
    assert KEY in _params(fn), f"_device_record() must take {KEY}; it takes {_params(fn)}"

    values = [
        v
        for node in ast.walk(fn)
        if isinstance(node, ast.Dict)
        for k, v in zip(node.keys, node.values, strict=True)
        if isinstance(k, ast.Constant) and k.value == KEY
    ]
    assert values, f"_device_record() writes no {KEY!r} key into the record"
    assert all(isinstance(v, ast.Name) and v.id == KEY for v in values), (
        f"_device_record() must record the {KEY} it was passed, not one it "
        "re-derives: the env var is gone by the time provenance is written, so "
        "any local re-read records the shipped default rather than the size the "
        "embed pass actually ran at."
    )


def test_write_provenance_threads_the_size_into_the_device_record():
    fn = _func(_PROVENANCE, "write_provenance")
    assert KEY in _params(fn), f"write_provenance() must take {KEY}; it takes {_params(fn)}"

    calls = _calls(fn, "_device_record")
    assert calls, "write_provenance() no longer builds a device record"
    passed = {
        arg.id
        for call in calls
        for arg in [*call.args, *(kw.value for kw in call.keywords)]
        if isinstance(arg, ast.Name)
    }
    assert KEY in passed, f"write_provenance() does not pass {KEY} to _device_record()"


def test_the_size_is_read_inside_the_window_where_the_env_var_is_set():
    """``effective_embed_batch_size`` belongs to the context manager, not the caller.

    ``_embed_batch_size`` is the only scope in which ``VTSEARCH_EMBED_BATCH_SIZE``
    holds the value the build chose; ``build_cell`` reading it afterwards would
    get 32 back for every embedder whose table entry is not 32.
    """
    cm = _func(_BUILD, "_embed_batch_size")
    assert _calls(cm, "effective_embed_batch_size"), (
        "_embed_batch_size() must resolve the size itself -- it owns the only "
        "window in which VTSEARCH_EMBED_BATCH_SIZE is set."
    )
    assert all(isinstance(n.value, ast.Call) for n in ast.walk(cm) if isinstance(n, ast.Yield) and n.value), (
        "every yield in _embed_batch_size() must hand back the resolved size"
    )

    build = _func(_BUILD, "build_cell")
    assert not _calls(build, "effective_embed_batch_size"), (
        "build_cell() must not resolve the batch size itself: outside the "
        "context manager the env var is unset, so it would record the default."
    )


def test_build_cell_passes_the_yielded_size_to_write_provenance():
    """The `with ... as` target, and nothing else, is what reaches the sidecar."""
    build = _func(_BUILD, "build_cell")
    targets = {
        item.optional_vars.id
        for node in ast.walk(build)
        if isinstance(node, ast.With)
        for item in node.items
        if isinstance(item.context_expr, ast.Call)
        and isinstance(item.context_expr.func, ast.Name)
        and item.context_expr.func.id == "_embed_batch_size"
        and isinstance(item.optional_vars, ast.Name)
    }
    assert targets, "build_cell() must bind the size _embed_batch_size() yields (`with ... as size:`)"

    calls = _calls(build, "write_provenance")
    assert calls, "build_cell() no longer writes provenance"
    handed = {kw.value.id for call in calls for kw in call.keywords if kw.arg == KEY and isinstance(kw.value, ast.Name)}
    assert handed & targets, (
        f"build_cell() must pass the yielded size as {KEY}=; it passes "
        f"{handed or 'nothing'} while the context manager binds {targets}."
    )


def test_provenance_report_shows_the_size():
    """``--provenance`` is where a human compares two cells; the size has to be in it."""
    src = _REPORT.read_text()
    assert f'"{KEY}"' in src, (
        f"provenance_report.py never reads {KEY} out of the sidecar, so "
        "`build_pile.py --provenance` cannot show what a cell was batched at."
    )
