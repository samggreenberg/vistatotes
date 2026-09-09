"""One module per ``DATASETS[ds]["kind"]``, registered here.

Each loader module owns **both** halves of its dataset:

* ``load(dataset, medias, embedder_name)`` -- build the cell's medias.
* ``check(dataset) -> str`` -- what a rebuild of it would read, without
  embedding anything. Raises ``SystemExit`` naming the missing source, and
  returns the one-line "ok" description otherwise.

The two live together on purpose. They used to be two ``kind`` switches in two
functions a thousand lines apart, and the drift that arrangement invites is
exactly #3299: the canary checked ``COCO_IMAGES`` while the builder opened
``val2017.zip`` inline, and reported ``coco_val`` REBUILD-BROKEN against a
staging area that was entirely intact. A canary that names a different path than
the build is not a canary -- so a new kind now adds one module carrying both,
and a kind with no module fails loudly at dispatch rather than silently
defaulting to the demo path.
"""

from __future__ import annotations

from types import ModuleType

from pilebuild.loaders import coco, demo, vg_band, vg_scale, vg_scale_any, vg_scale_deep

#: ``DATASETS[ds]["kind"]`` -> the module that owns it.
LOADERS: dict[str, ModuleType] = {
    "coco": coco,
    "demo": demo,
    "vg_band": vg_band,
    "vg_scale": vg_scale,
    "vg_scale_any": vg_scale_any,
    "vg_scale_deep": vg_scale_deep,
}


def loader_for(dataset: str, kind: str) -> ModuleType:
    """The module owning *kind*, or ``SystemExit`` naming what needs teaching.

    Unknown kinds fail rather than falling through to the demo loader. The old
    ``else: _load_demo(...)`` branch meant a typo in ``DATASETS`` built a
    plausible-looking cell out of the wrong source.
    """
    try:
        return LOADERS[kind]
    except KeyError:
        raise SystemExit(
            f"{dataset}: unknown kind {kind!r}; add a pilebuild/loaders/ module and register it "
            f"(known: {', '.join(sorted(LOADERS))})"
        ) from None
