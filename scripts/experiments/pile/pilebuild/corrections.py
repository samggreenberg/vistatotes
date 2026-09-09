"""Human verdicts on ``(image, class)`` pairs, and the one box-space crossing.

Everything in this module exists to make **one** conversion happen exactly once.
A correction's boxes arrive normalised (they come from the app's ``region_box``);
every other box in the scale build is in pixels. Converting here, on the way in,
is what keeps the rest of the loader in a single space -- and #3281 is what the
other arrangement costs: a normalised box merged unconverted is normalised a
second time by the region write, which divides it by ~500 and parks it on the
frame origin, with the band derived from the same corrupted box so that nothing
downstream can see the disagreement.
"""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path

import pile_config as pc


def dropped_rows(old: list[dict], new: list[dict]) -> dict[str, int]:
    """``{source: n}`` for pairs the old file has and the new one does not.

    A regeneration is supposed to be a function of the inputs, so a *smaller*
    result means the inputs are not the ones that produced what is on disk.
    Measured on the live file: the defaults in this script reproduce 488 of its
    640 rows, and no invocation anyone could reconstruct reproduces all of them
    -- 379 rows come from verdict files, triage flags and adjudications nobody
    recorded. Overwriting on those terms is a silent deletion of human work,
    which is why :func:`main` refuses rather than warns.
    """
    have = {(int(r["image_id"]), r["class"]) for r in new}
    lost: dict[str, int] = {}
    for r in old:
        if (int(r["image_id"]), r["class"]) not in have:
            lost[r.get("source", "unknown")] = lost.get(r.get("source", "unknown"), 0) + 1
    return lost


def write_json_locked(path: Path, payload: object, indent: int = 1) -> None:
    """Write *payload* to *path* atomically, under an exclusive lock.

    Both halves matter for a file several sessions share on one pile (#3729).

    **Atomic**, because the old spelling was ``path.write_text(...)``: it
    truncates first, so a reader that arrives mid-write gets a half file and a
    writer that dies mid-write leaves one. The verdicts are the least
    reproducible thing here and were being rewritten in place with no landing
    strip.

    **Locked**, because two sessions regenerating this file at once is not
    hypothetical -- the pile is shared, and `corrections.json` is rebuilt from
    the verdict files by whoever ingests a slate. Without the lock the later
    writer silently wins with whatever inputs *it* could see; with it, the two
    runs serialise and the second reads the first's output. The lock is held on
    a sidecar rather than on the file itself so that the ``os.replace`` below
    cannot pull the locked inode out from under a waiter.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(path.suffix + ".lock")
    with lock.open("w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=indent) + "\n")
        os.replace(tmp, path)


def load_corrections() -> dict[tuple[int, str], dict]:
    """``{(image_id, class): verdict}`` from the corrections file, if any.

    Verdicts, not corrections: a row exists for every reviewed ``(image, class)``
    pair whether or not the human disagreed, so review *coverage* is knowable.
    Without that, "no bus here" and "nobody looked" are the same absence, and
    every rate computed afterwards is biased by an unknown amount.

    Written by ``ingest_slate.py``; absent until the first review lands, which
    is why this returns empty rather than failing.

    **Boxes here are NORMALISED, unlike VG's and COCO's** -- they come from the
    app's ``region_box``. The space is validated on the way in and converted to
    pixels once, by :func:`correction_boxes_px`, so that everything downstream
    of this function is in one space. See ``pile_config.CORRECTION_BOX_SPACE``.
    """
    path = Path(os.environ.get("VTS_CORRECTIONS", pc.PILE / "corrections.json"))
    if not path.exists():
        return {}
    rows = json.loads(path.read_text())
    out: dict[tuple[int, str], dict] = {}
    for r in rows:
        assert_correction_box_space(r, path)
        out[(int(r["image_id"]), r["class"])] = r
    return out


def assert_correction_box_space(row: dict, path: Path) -> None:
    """Refuse a correction row whose boxes are not in the declared space.

    The space is a *declaration*, not a guess: a pixel-space box handed to the
    normalised path is undetectable once it has been divided by (W, H) -- the
    result is a small, plausible, entirely wrong box, which is #3281 in the
    other direction. Checking it here costs one comparison per row and is the
    only point at which the two spaces are still distinguishable.
    """
    space = row.get("box_space", pc.CORRECTION_BOX_SPACE)
    where = f"{path.name}: image {row.get('image_id')} / {row.get('class')!r}"
    if space != pc.CORRECTION_BOX_SPACE:
        raise SystemExit(f"{where}: box_space {space!r}, expected {pc.CORRECTION_BOX_SPACE!r}")
    for b in row.get("boxes") or []:
        if len(b) != 4:
            raise SystemExit(f"{where}: box {b} is not [x0, y0, x1, y1]")
        if not all(-1e-6 <= float(v) <= 1.0 + 1e-6 for v in b):
            raise SystemExit(
                f"{where}: box {b} has a coordinate outside [0, 1], so it is in PIXEL space "
                f"while the file declares {pc.CORRECTION_BOX_SPACE!r}"
            )
        if float(b[2]) <= float(b[0]) or float(b[3]) <= float(b[1]):
            raise SystemExit(f"{where}: box {b} is degenerate (x1 <= x0 or y1 <= y0)")


def correction_boxes_px(row: dict, W: int, H: int) -> list[list[float]]:
    """A verdict's boxes in the pixel space of ``(W, H)``.

    ``(W, H)`` is the space the image's *other* boxes were measured in -- the
    COCO original for an anchored image, the VG copy otherwise -- because that
    is what the region write later divides by. Scaling up here and dividing down
    there is an exact round trip, so the stored box is the reviewer's box to the
    last bit rather than merely close to it.
    """
    return [[float(b[0]) * W, float(b[1]) * H, float(b[2]) * W, float(b[3]) * H] for b in (row.get("boxes") or [])]
