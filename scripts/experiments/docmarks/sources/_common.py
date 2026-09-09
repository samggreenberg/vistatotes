"""Shared record types and pure helpers for the DocMarks source adapters.

Every adapter splits the same way:

* ``fetch_*`` touches the network and the filesystem.  Not unit-tested; it is
  exercised for real on the GRID and guarded by ``--probe``.
* everything else is a pure function of bytes already on disk, and *is* unit
  tested against small fixtures.

That split is the reason the corpus builder can be developed and verified in an
environment that cannot reach Kaggle or hold a 3 GB archive.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, NamedTuple, Optional, Sequence

# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Mark:
    """One ground-truth mark on one page.

    ``class_id`` is ``None`` until the identity-clustering pass runs: SPODS and
    StaVer ship *where* a mark is without shipping *which* mark it is.
    """

    kind: str  # "logo" | "stamp" | "signature" | "text" | "icon"
    box: tuple[int, int, int, int]  # x, y, w, h in page pixels
    class_id: Optional[str] = None
    #: "gt" (shipped by the source), "clustered" (derived, needs audit),
    #: "weak" (metadata-implied, unverified) or "synthetic" (true by construction).
    provenance: str = "gt"

    def area(self) -> int:
        return self.box[2] * self.box[3]

    def longest_side(self) -> int:
        return max(self.box[2], self.box[3])


@dataclass
class Page:
    """One page image plus everything known about it."""

    page_id: str  # globally unique, "<source>/<local id>"
    source: str  # spods | staver | tobacco800 | ucsf | synth
    path: str  # relative to the corpus root
    width: int
    height: int
    marks: list[Mark] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["marks"] = [{**asdict(m), "box": list(m.box)} for m in self.marks]
        return d

    @staticmethod
    def from_json(d: dict[str, Any]) -> "Page":
        marks = [
            Mark(
                kind=m["kind"],
                box=tuple(m["box"]),  # type: ignore[arg-type]
                class_id=m.get("class_id"),
                provenance=m.get("provenance", "gt"),
            )
            for m in d.get("marks", [])
        ]
        return Page(
            page_id=d["page_id"],
            source=d["source"],
            path=d["path"],
            width=d["width"],
            height=d["height"],
            marks=marks,
            meta=d.get("meta", {}),
        )


def write_manifest(pages: Iterable[Page], path: Path) -> int:
    """Write ``corpus.jsonl``.  Returns the number of records written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for page in pages:
            fh.write(json.dumps(page.to_json(), sort_keys=True) + "\n")
            n += 1
    tmp.replace(path)
    return n


def read_manifest(path: Path) -> Iterator[Page]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield Page.from_json(json.loads(line))


# --------------------------------------------------------------------------
# Deterministic sampling
# --------------------------------------------------------------------------


def stable_rank(key: str, salt: str) -> float:
    """A stable pseudo-random rank in ``[0, 1)`` for *key*.

    Used to pick which distractors land in which tier.  It must be a pure
    function of the id, never of iteration order or of how many pages happened
    to be fetched — otherwise a tier reshuffles every time the corpus grows, and
    two studies that both say "tier s" are not comparable.
    """
    h = hashlib.sha256(f"{salt}\x00{key}".encode()).digest()
    return int.from_bytes(h[:8], "big") / float(1 << 64)


def spread(items: Sequence[Any], limit: int) -> list[Any]:
    """Up to *limit* of *items*, evenly spaced across the whole sequence.

    Every pass that samples a class hits the same trap: page ids sort by source
    and number, so the head of the list is whatever the scanner did first, and a
    class whose later instances drifted (a re-inked stamp, a second printing, a
    second mark that only appears late) looks homogeneous for no better reason
    than alphabetical order.

    Spaced by index rather than by a ``[::step]`` stride, because the stride
    degenerates exactly where it is needed most: ``step = n // limit`` is 1 for
    any class between ``limit`` and ``2 * limit`` instances, so a 27-instance
    class sampled at 24 silently gets its first 24 — the head sample the stride
    was reached for to avoid (#3610).  Spacing the indices over ``n - 1``
    instead always reaches the tail.
    """
    n = len(items)
    if limit <= 0 or n == 0:
        return []
    if n <= limit:
        return list(items)
    if limit == 1:
        return [items[0]]
    return [items[round(i * (n - 1) / (limit - 1))] for i in range(limit)]


# --------------------------------------------------------------------------
# Masks -> boxes
# --------------------------------------------------------------------------
#
# The order of operations here is load-bearing, and getting it wrong is silent.
#
# A mark's mask is *not* one connected component.  A rubber stamp breaks into a
# ring, the text inside it, and a dozen broken arcs where the ink did not take;
# a script stamp breaks into one component per pen stroke.  Individually those
# strokes are thin and tiny.  So an area floor applied to *raw* components
# deletes eleven fragments of the dozen as "speckle", the merge that exists to
# reassemble them never sees them, and what survives is the one or two chunkiest
# pieces — which then become their own "classes".  That is how ``stamp_00129_1``
# came to be 38 instances of the word "New" (issue #3361).
#
# The correct order is therefore: **decompose, merge, then filter.**  The floor
# asks "does this assembled group carry enough ink to be a mark?", which is the
# question it was always meant to ask, rather than "is this individual pen
# stroke big enough to be a mark?", which has no useful answer.
#
# The one thing that must still be dropped *before* the merge is true single-
# pixel scan noise: it carries no evidence either way, but it can extend a box
# or bridge a gap between two marks that should stay apart.  That is
# ``speckle_px`` — an absolute, deliberately tiny floor, not a page fraction.


#: Filled-pixel count at or below which a raw component is scan noise rather
#: than evidence.  Dropped *before* the merge; see the note above for why this
#: is the only pre-merge filter.
SPECKLE_PX = 4

#: Merge gap as a fraction of the page's longest side, with an absolute floor.
#: A gap in pixels does not travel between a 950 px thumbnail and a 3,500 px
#: scan, and the distance between a broken stamp's fragments scales with the
#: stamp, which scales with the page.  See :func:`merge_gap_for_page`.
#:
#: **0.035 is read off the data, not chosen.**  Swept over all 1,088 SPODS pages
#: (A4 at 300 DPI, ~2,476x3,480), the per-page mark count falls as the gap grows
#: and then stops dead: every kind reaches exactly one mark per page that has
#: one, and from **gap 90 px through gap 300 px the output is identical** — same
#: counts, same median (428 px), same p90, same largest box (758 px, 3.5% of the
#: page).  Nothing changes over a 3.3x range because on these pages there is
#: nothing else within 300 px of a mark to merge with, which is as clean a
#: plateau as a parameter ever gets.  Below 90 the merge under-runs and splits
#: real stamps: at 60 the three-line "RTI Unit" stamp on ``spods/00837`` breaks
#: across its 70 px line spacing, and at 20 the ``spods/00129`` stamp splits into
#: the fragments that became the "New" class.  0.035 lands at 122 px on a SPODS
#: page — 1.35x inside the plateau's lower edge, and kept nearer that edge than
#: the middle because other sources (StaVer) do carry two stamps on one page,
#: where an over-large gap would weld them.
#:
#: **One value, not one per kind**, because the sweep says so: the three SPODS
#: kinds settle at different points (logo from 42 px, signature from 60, stamp
#: from 90) but they settle into the *same* plateau, so any gap at or above the
#: stamp's requirement is simultaneously correct for all three.  Splitting it per
#: kind would be three numbers where the data supports one.
MERGE_GAP_FRAC = 0.035
MERGE_GAP_MIN_PX = 6


class Component(NamedTuple):
    """One connected component of a mask: where it is, and how much ink it is.

    ``ink`` is the filled-pixel count, which is *not* the box area — a ring is
    mostly hole.  Keeping the two apart is what lets the area floor run after
    the merge: a merged group's ink is the sum of its parts' ink, while its box
    area would count the paper between them.
    """

    box: tuple[int, int, int, int]  # x, y, w, h
    ink: int


def merge_gap_for_page(width: int, height: int) -> int:
    """The merge gap to use on a page of this size, in pixels.

    Scale-relative rather than fixed: the fragments of a broken stamp sit a
    fixed fraction of the stamp apart, and a stamp is a fixed fraction of the
    page.  ``MERGE_GAP_MIN_PX`` keeps a thumbnail-sized page from merging
    nothing at all.
    """
    return max(MERGE_GAP_MIN_PX, round(MERGE_GAP_FRAC * max(width, height)))


def mask_components(
    mask: Any,
    *,
    polarity: str = "auto",
    speckle_px: int = SPECKLE_PX,
) -> list[Component]:
    """Every connected component of a binary *mask*, unfiltered but de-speckled.

    This is the raw decomposition: no page-fraction floor, no merging.  Callers
    that want *marks* rather than components want :func:`mask_to_boxes`, which
    applies both in the right order.

    **Polarity is detected, not assumed.**  SPODS ships 1-bit masks with the
    mark in *black* on white paper, so a naive "non-zero is foreground" reads
    99.8% of every page as one enormous mark — which is not a crash, it is 1,088
    page-sized boxes that cluster into a single class and look superficially
    like a working corpus. Taking the minority phase as foreground is safe
    because a ground-truth mask marks a *mark*: on real SPODS pages the marked
    fraction runs 0.2–1.1%, and any mask where most of the page is "on" is
    inverted by definition of the task.

    Pass ``polarity="light"`` or ``"dark"`` to force it when a source's masks are
    genuinely dense (a text mask on a very full page can approach half, though
    none observed comes close).
    """
    import numpy as np

    arr = np.asarray(mask)
    if arr.ndim == 3:  # RGB(A) mask -> any channel lit
        arr = arr[..., :3].max(axis=2)

    threshold = arr.max() / 2 if arr.max() > 1 else 0
    lit = arr > threshold
    if polarity == "auto":
        foreground = lit if lit.mean() <= 0.5 else ~lit
    elif polarity == "light":
        foreground = lit
    elif polarity == "dark":
        foreground = ~lit
    else:
        raise ValueError(f"unknown polarity {polarity!r} (expected auto|light|dark)")

    binary = foreground.astype("uint8")
    if not binary.any():
        return []

    comps: list[Component] = []
    try:
        import cv2

        n, _labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for i in range(1, n):  # 0 is background
            x, y, w, h, ink = (int(v) for v in stats[i])
            if ink > speckle_px:
                comps.append(Component((x, y, w, h), ink))
    except ImportError:
        from scipy import ndimage  # type: ignore[import-untyped]

        labels, n = ndimage.label(binary)
        for i, sl in enumerate(ndimage.find_objects(labels), start=1):
            if sl is None:
                continue
            ys, xs = sl
            ink = int((labels[sl] == i).sum())
            if ink > speckle_px:
                box = (int(xs.start), int(ys.start), int(xs.stop - xs.start), int(ys.stop - ys.start))
                comps.append(Component(box, ink))

    comps.sort(key=lambda c: (c.box[1], c.box[0]))
    return comps


def mask_to_boxes(
    mask: Any,
    min_area_frac: float = 0.0002,
    *,
    polarity: str = "auto",
    merge_gap: int = 0,
    speckle_px: int = SPECKLE_PX,
) -> list[tuple[int, int, int, int]]:
    """Marks of a binary *mask* as ``(x, y, w, h)`` boxes.

    SPODS and StaVer both ship per-category pixel masks rather than boxes, so
    this is the first step for either.

    Components within *merge_gap* pixels of each other are merged **first**, and
    the *min_area_frac* floor is then applied to each merged group's total ink.
    Do not reverse that (see the note at the top of this section); pass
    ``merge_gap=0`` only for a mask whose components are genuinely separate
    marks.  :func:`merge_gap_for_page` derives the gap from the page size.
    """
    import numpy as np

    # Convert once and hand the array on: these masks are 8-megapixel scans, and
    # a second np.asarray(PIL image) here costs as much as the decomposition.
    arr = np.asarray(mask)
    comps = mask_components(arr, polarity=polarity, speckle_px=speckle_px)
    if not comps:
        return []

    height, width = arr.shape[:2]
    min_ink = max(1.0, min_area_frac * float(width * height))

    merged = merge_components(comps, gap=merge_gap) if merge_gap > 0 else comps
    return [c.box for c in merged if c.ink >= min_ink]


def merge_components(components: Sequence[Component], *, gap: int = 6) -> list[Component]:
    """Merge components that touch or nearly touch, summing their ink.

    A rubber stamp's mask usually breaks into a dozen components — the ring, the
    text inside it, a broken arc where the ink did not take.  Left unmerged,
    every fragment becomes its own "mark" and the class inventory is nonsense.
    """
    remaining = list(components)
    out: list[Component] = []
    while remaining:
        (x, y, w, h), ink = remaining.pop()
        changed = True
        while changed:
            changed = False
            for other in list(remaining):
                (ox, oy, ow, oh), oink = other
                if x - gap < ox + ow and ox - gap < x + w and y - gap < oy + oh and oy - gap < y + h:
                    nx, ny = min(x, ox), min(y, oy)
                    x, y, w, h = nx, ny, max(x + w, ox + ow) - nx, max(y + h, oy + oh) - ny
                    ink += oink
                    remaining.remove(other)
                    changed = True
        out.append(Component((x, y, w, h), ink))
    out.sort(key=lambda c: (c.box[1], c.box[0]))
    return out


def merge_overlapping(boxes: list[tuple[int, int, int, int]], gap: int = 6) -> list[tuple[int, int, int, int]]:
    """:func:`merge_components` for callers that have boxes but no ink counts."""
    return [c.box for c in merge_components([Component(b, 0) for b in boxes], gap=gap)]


def reject_oversize(
    boxes: Sequence[tuple[int, int, int, int]],
    width: int,
    height: int,
    max_area_frac: float,
) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]]:
    """Split *boxes* into ``(kept, rejected)`` on the fraction of page they cover.

    A mark is a thing *on* a page, not the page.  A box covering half the sheet
    is a mask artefact — most often a ruled table, whose borders weld the whole
    grid into one connected component — and admitting it as a mark puts a
    page-sized crop into the class inventory.  Rejections are returned rather
    than dropped so the caller can report them: an unexplained missing mark is
    worse than a noisy one.
    """
    page_area = float(width * height)
    kept, rejected = [], []
    for box in boxes:
        (kept if page_area and box[2] * box[3] < max_area_frac * page_area else rejected).append(box)
    return kept, rejected


# --------------------------------------------------------------------------
# Fetch helpers
# --------------------------------------------------------------------------


class FetchError(RuntimeError):
    """A source could not be fetched, with an actionable reason."""


class RateLimited(FetchError):
    """The server asked us to slow down.

    Separated from :class:`FetchError` because it is the one failure that is
    about *us* rather than about the document: a 403 on a restricted PDF is
    permanent and must be skipped, while a 429 means back off and the document
    is still there.  Conflating them would have the fetcher discard documents it
    was merely asking for too quickly.
    """


def require_kaggle_credentials(slug: str) -> None:
    """Raise :class:`FetchError` unless a Kaggle credential is in place.

    The Kaggle CLI reads ``~/.kaggle/kaggle.json``, ``~/.kaggle/access_token``
    or the ``KAGGLE_USERNAME`` / ``KAGGLE_KEY`` environment pair.  Checking here
    means a missing token is reported with the setup instruction, rather than as
    a 403 halfway through a grid job.

    ``access_token`` is in that list because it is what Kaggle's own "Create New
    Token" now writes, and what ``kagglesdk`` reads natively (its
    ``kaggle_creds.py`` / ``kaggle_oauth.py``).  Accepting only ``kaggle.json``
    made a *working* credential look like a missing one: on the GRID the probe
    reported both Kaggle sources BLOCKED while ``kaggle datasets download``
    succeeded from the very same shell, against the very same token.  Since the
    whole point of this gate is to fail fast rather than 403 mid-job, a false
    BLOCKED is the one way it can be worse than having no gate at all.
    """
    has_env = bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
    kaggle_dir = Path.home() / ".kaggle"
    has_file = (kaggle_dir / "kaggle.json").exists() or (kaggle_dir / "access_token").exists()
    if not (has_env or has_file):
        raise FetchError(
            f"Kaggle credentials not found, needed for '{slug}'. "
            "Put a token at ~/.kaggle/kaggle.json or ~/.kaggle/access_token "
            "(Kaggle > Settings > Create New Token writes one of the two), "
            "or export KAGGLE_USERNAME and KAGGLE_KEY."
        )


def kaggle_probe(slug: str) -> None:
    """Check that *slug* is reachable with the credential in place, fetching nothing.

    This is the reachability half of :func:`kaggle_download`, and it exists so
    that ``build_corpus.py --probe`` costs seconds rather than gigabytes: it
    lists the dataset's files (a metadata call) instead of pulling the bundle.
    A missing token, a revoked token and a slug that has been renamed or taken
    down all surface here, which is every Kaggle failure mode the real fetch has.

    **The CLI's exit code is not enough on its own.** ``kaggle datasets files``
    catches API errors itself, prints them, and still exits 0 — so a 403 would
    read as a pass. Success is therefore recognised positively: a CSV listing
    with a ``name`` column and at least one row. Anything else is a failure and
    the raw output is quoted back.
    """
    require_kaggle_credentials(slug)

    cmd = ["kaggle", "datasets", "files", "-d", slug, "--csv"]
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)  # noqa: S603
    except FileNotFoundError as exc:
        raise FetchError("The 'kaggle' CLI is not installed (pip install kaggle).") from exc
    except subprocess.CalledProcessError as exc:
        raise FetchError(f"kaggle metadata call for '{slug}' failed: {exc.stderr.strip()[:400]}") from exc

    # FIND the header rather than assuming it is the first line.  Kaggle CLI
    # 2.2.4 prints a pagination preamble ahead of the CSV, and the CSV itself is
    # CRLF, so the naive read of this output is wrong twice over:
    #
    #     Next Page Token = CfDJ8ImuQD4OY2pEnVW2WQ-kgndQdHqu9wY-...
    #     name,size,creationDate\r
    #     ground-truth-maps/.../stampDS-00001-gt.png,9151,2018-04-11 ...\r
    #
    # Taking row 0 as the header made every reachable dataset report as
    # unreachable -- `staver BLOCKED` and `tobacco800 BLOCKED` against a token
    # that had just downloaded 32.8 MB from the same shell.  Since this probe
    # exists so a missing token is caught before a queue slot is burned, a false
    # negative here costs exactly what the probe was written to save.
    rows = [line.strip() for line in (proc.stdout or "").splitlines() if line.strip()]
    header_at = next(
        (i for i, row in enumerate(rows) if "name" in [cell.strip() for cell in row.lower().split(",")]),
        None,
    )
    if header_at is None or len(rows) - header_at < 2:
        detail = " ".join((proc.stdout or "").split())[:400] or "(no output)"
        raise FetchError(f"kaggle could not list '{slug}': {detail}")


def kaggle_download(slug: str, dest: Path, *, unzip: bool = True) -> Path:
    """Download a Kaggle dataset *slug* (``owner/name``) into *dest*.

    Uses the Kaggle CLI; see :func:`require_kaggle_credentials` for how the
    token is found.  Use :func:`kaggle_probe` when you only want to know whether
    the source is reachable — this one transfers the whole bundle.
    """
    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.iterdir()):
        return dest

    require_kaggle_credentials(slug)

    cmd = ["kaggle", "datasets", "download", "-d", slug, "-p", str(dest)]
    if unzip:
        cmd.append("--unzip")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)  # noqa: S603
    except FileNotFoundError as exc:
        raise FetchError("The 'kaggle' CLI is not installed (pip install kaggle).") from exc
    except subprocess.CalledProcessError as exc:
        raise FetchError(f"kaggle download of '{slug}' failed: {exc.stderr.strip()[:400]}") from exc
    return dest


def http_download(url: str, dest: Path, *, chunk: int = 1 << 20, session: Any = None) -> Path:
    """Stream *url* to *dest*, resuming a partial file and writing atomically.

    Pass *session* to reuse one connection across a long pull.  Measured on the
    UCSF endpoint it is worth only ~1.09x (307ms -> 281ms per PDF, so the cost
    is the archive generating and sending the file, not the TLS handshake), but
    it is free and it is strictly *less* load on a shared public service than
    re-handshaking once per document across 216,000 of them.
    """
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest

    tmp = dest.with_suffix(dest.suffix + ".part")
    have = tmp.stat().st_size if tmp.exists() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    get = session.get if session is not None else requests.get
    with get(url, headers=headers, stream=True, timeout=(20, 120)) as resp:
        if resp.status_code in (429, 503, 509):
            raise RateLimited(f"{url} returned HTTP {resp.status_code}")
        if resp.status_code not in (200, 206):
            raise FetchError(f"{url} returned HTTP {resp.status_code}")
        mode = "ab" if have and resp.status_code == 206 else "wb"
        with tmp.open(mode) as fh:
            for block in resp.iter_content(chunk_size=chunk):
                fh.write(block)
    tmp.replace(dest)
    return dest


def extract_rar(archive: Path, dest: Path) -> Path:
    """Unpack a RAR archive, trying the tools most likely to exist on a cluster.

    SPODS ships as RAR4, which Python cannot read from the standard library.
    ``bsdtar`` (libarchive) handles it and is far more commonly installed on a
    compute node than ``unrar`` is.
    """
    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.iterdir()):
        return dest

    for cmd in (
        ["bsdtar", "-x", "-f", str(archive), "-C", str(dest)],
        ["7z", "x", f"-o{dest}", "-y", str(archive)],
        ["unar", "-o", str(dest), str(archive)],
        ["unrar", "x", "-y", str(archive), str(dest) + "/"],
    ):
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)  # noqa: S603
            return dest
        except FileNotFoundError:
            continue
        except subprocess.CalledProcessError as exc:
            raise FetchError(f"{cmd[0]} failed on {archive.name}: {exc.stderr.strip()[:300]}") from exc
    raise FetchError(
        f"No RAR extractor found for {archive.name}. Install one of: bsdtar (libarchive), 7z (p7zip), unar, unrar."
    )


def extract_zip(archive: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.iterdir()):
        return dest
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            # Reject absolute paths and traversal before writing anything.
            name = Path(member.filename)
            if name.is_absolute() or ".." in name.parts:
                raise FetchError(f"unsafe path in {archive.name}: {member.filename}")
        zf.extractall(dest)  # noqa: S202 - paths validated above
    return dest
