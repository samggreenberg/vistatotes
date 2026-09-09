"""Turn located marks into identity classes — and be honest that it's a guess.

SPODS and StaVer both ship *where* every logo and stamp is, and neither ships
*which* one it is.  A previous VTSearch study reported "64 logo/stamp classes"
for SPODS with names like ``logo_14``; those identities were derived, not read
off the dataset, and nothing verified them.  Since class identity is the entire
ground truth of an instance-retrieval benchmark, a derived clustering that
nobody checked is not a benchmark — it is a hypothesis with error bars nobody
measured.

So this module does three things in order:

1. crop every mark and describe it (``phash`` by default, ``siglip`` when the
   pile's models are available);
2. single-linkage agglomerate under a distance threshold into candidate classes;
3. emit the material a human needs to confirm or correct the result, and mark
   every derived ``class_id`` with ``provenance="clustered"`` so downstream code
   can tell a verified identity from a guessed one.

Single linkage is chosen deliberately over a centroid method: instances of one
stamp vary continuously with ink coverage and scan quality, so a chain of near
neighbours is the right shape, and the failure mode it does have — two classes
bridged by one ambiguous crop — is exactly what a human reviewing the largest
clusters will spot immediately.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from sources._common import Mark, Page

#: Side length the crop is normalised to before hashing.
PHASH_RESIZE = 64

#: Low-frequency DCT block kept.  16x16 coefficients -> a 256-bit hash.
#:
#: This was 8x8 (64 bits), and 64 bits could not tell two round stamps apart.
#: Measured on the real corpus: a red book stamp and a blue elephant stamp,
#: both circular with a heavy ring border, landed in one class of 32 and no
#: threshold separated them — a single pair at Hamming 2/64 bridged the set, so
#: it was one group at 0.04 and 21 fragments at 0.03.
#:
#: The mechanism is a frequency argument.  A stamp's border ring is a big,
#: smooth, *low*-frequency structure, and its interior — the part that says
#: which stamp this is — is higher-frequency detail.  An 8x8 block keeps almost
#: nothing but the ring, so the descriptor encodes "is a round stamp" rather
#: than "which round stamp".  Keeping 16x16 pushes the balance the other way.
PHASH_BLOCK = 16

#: Fraction of the crop's half-diagonal over which the radial taper acts.  The
#: border is also the part of a mark that varies least between *different*
#: marks, so damping it costs little discrimination and removes a lot of shared
#: signal.  Applied on top of the bigger DCT block, not instead of it.
PHASH_TAPER = 0.22


@dataclass(frozen=True)
class MarkRef:
    """A pointer to one mark inside the page list, plus its crop descriptor."""

    page_index: int
    mark_index: int
    page_id: str
    kind: str
    box: tuple[int, int, int, int]


# --------------------------------------------------------------------------
# Descriptors
# --------------------------------------------------------------------------


def _dct2(block: np.ndarray) -> np.ndarray:
    """2-D DCT-II via matrix multiply.

    Written out rather than pulled from scipy so the clustering pass has the
    same dependency footprint as the rest of the builder (numpy + Pillow) and
    runs identically on a login node and a compute node.
    """
    n = block.shape[0]
    k = np.arange(n)
    basis = np.cos(np.pi * (2 * k[:, None] + 1) * k[None, :] / (2 * n))
    basis[0, :] = basis[0, :] / np.sqrt(2)
    return basis @ block @ basis.T


def _radial_taper(arr: np.ndarray, width: float = PHASH_TAPER) -> np.ndarray:
    """Fade the crop's outer annulus toward its own mean.

    A stamp's border ring is the single strongest thing in the image and the
    *least* informative: near enough every round stamp has one, so it is shared
    signal that crowds out the interior.  Fading toward the mean rather than to
    white avoids replacing one strong edge with another.

    The taper is radial because the marks that need separating are round, and
    it is soft because a hard mask would itself be a circular edge — the exact
    artefact being removed.
    """
    n = arr.shape[0]
    axis = (np.arange(n) - (n - 1) / 2) / ((n - 1) / 2)
    radius = np.hypot(*np.meshgrid(axis, axis, indexing="ij"))
    # 1 in the middle, falling smoothly to 0 at the corners of the inscribed
    # circle and beyond.
    weight = np.clip((1.0 - radius) / max(width, 1e-6), 0.0, 1.0)
    weight = weight * weight * (3 - 2 * weight)  # smoothstep
    return arr.mean() + (arr - arr.mean()) * weight


def phash(image: Any) -> np.ndarray:
    """A perceptual hash of *image* as a boolean vector, `PHASH_BLOCK`² bits.

    Scale-invariant by construction (everything is resized to a fixed square),
    which is what we want: the same stamp appears at whatever size the scanner
    and the page layout produced.  Aspect ratio is deliberately discarded here
    and reintroduced as a separate gate in :func:`distance_matrix`, because two
    marks with very different aspect ratios are not the same mark however
    similar their normalised pixels look.

    Greyscale on purpose, so ink colour never splits a class.  Confirmed on the
    real corpus, where the same elephant stamp appears in blue on 26 pages and
    red on one, and lands in a single group.
    """
    grey = image.convert("L").resize((PHASH_RESIZE, PHASH_RESIZE))
    arr = _radial_taper(np.asarray(grey, dtype=np.float64))
    coeffs = _dct2(arr)[:PHASH_BLOCK, :PHASH_BLOCK]
    flat = coeffs.flatten()
    # Drop the DC term before taking the median: it encodes overall brightness,
    # which on a scan is the paper, not the mark.
    median = np.median(flat[1:])
    return flat > median


def crop_mark(page_image: Any, box: tuple[int, int, int, int], *, pad_frac: float = 0.04) -> Any:
    """Crop *box* out of *page_image* with a little context padding."""
    x, y, w, h = box
    pad = int(round(max(w, h) * pad_frac))
    left = max(0, x - pad)
    top = max(0, y - pad)
    right = min(page_image.width, x + w + pad)
    bottom = min(page_image.height, y + h + pad)
    return page_image.crop((left, top, right, bottom))


def describe_marks(
    pages: Sequence[Page],
    refs: Sequence[MarkRef],
    *,
    backend: str = "phash",
) -> np.ndarray:
    """Descriptor matrix for *refs*, one row per mark.

    ``phash`` returns a boolean matrix compared by normalised Hamming distance;
    ``siglip`` returns L2-normalised float vectors compared by cosine distance.
    """
    from PIL import Image

    if backend == "phash":
        rows = []
        cache: dict[str, Any] = {}
        for ref in refs:
            page = pages[ref.page_index]
            if page.path not in cache:
                cache.clear()  # one page open at a time; pages are large
                cache[page.path] = Image.open(page.path).convert("L")
            rows.append(phash(crop_mark(cache[page.path], ref.box)))
        return np.array(rows, dtype=bool)

    if backend == "siglip":
        from vtscore.media.image.embedder_siglip import ImageSiglipEmbedder  # noqa: PLC0415

        embedder = ImageSiglipEmbedder()
        crops = []
        cache_path: Optional[str] = None
        cache_img: Any = None
        for ref in refs:
            page = pages[ref.page_index]
            if page.path != cache_path:
                cache_path, cache_img = page.path, Image.open(page.path).convert("RGB")
            crops.append(crop_mark(cache_img, ref.box))
        # `embed_pil_image` is the embedder's only in-memory entry point, and it
        # returns None for a crop the model declines.  A zero row keeps the
        # matrix aligned with *refs* -- dropping the row would silently shift
        # every later label onto the wrong mark -- and normalises to a vector at
        # distance 1.0 from everything, so that mark simply clusters alone.
        rows = [embedder.embed_pil_image(crop) for crop in crops]
        dim = embedder.embedding_dim
        vecs = np.asarray([row if row is not None else np.zeros(dim) for row in rows], dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.clip(norms, 1e-8, None)

    raise ValueError(f"unknown cluster backend {backend!r} (expected 'phash' or 'siglip')")


def distance_matrix(
    desc: np.ndarray,
    refs: Sequence[MarkRef],
    *,
    backend: str = "phash",
    max_aspect_ratio: float = 2.0,
) -> np.ndarray:
    """Pairwise distances, with mismatched aspect ratios forced apart.

    The aspect gate is what stops a round rubber stamp and a wide letterhead
    banner from merging just because both are dark ink on white paper at 32x32.
    """
    if backend == "phash":
        bits = desc.astype(np.uint8)
        # Hamming distance via matrix algebra: |a XOR b| = a.(1-b) + (1-a).b
        inv = 1 - bits
        dist = (bits @ inv.T + inv @ bits.T).astype(np.float64) / desc.shape[1]
    else:
        dist = 1.0 - (desc @ desc.T)

    aspects = np.array([max(b.box[2], 1) / max(b.box[3], 1) for b in refs], dtype=np.float64)
    ratio = np.maximum(aspects[:, None] / aspects[None, :], aspects[None, :] / aspects[:, None])
    dist = np.where(ratio > max_aspect_ratio, 1.0, dist)

    np.fill_diagonal(dist, 0.0)
    return dist


# --------------------------------------------------------------------------
# Clustering
# --------------------------------------------------------------------------


def single_linkage(
    dist: np.ndarray,
    threshold: float,
    *,
    cannot_link: Optional[Sequence[tuple[int, int]]] = None,
    must_link: Optional[Sequence[tuple[int, int]]] = None,
) -> list[int]:
    """Union-find single-linkage clustering.  Returns a label per row.

    Two kinds of human decision override the distances, and both are honoured
    before the threshold gets a say:

    * *must_link* — pairs adjudicated as the **same mark**.  Joined whatever
      their distance, because a clustering that splits one mark in two is a
      thing a person can see and the descriptor cannot.
    * *cannot_link* — pairs adjudicated as **different marks**.  Never joined,
      however similar they look, and the constraint propagates: once ``a`` and
      ``b`` are apart, nothing may reunite their groups through a third crop.

    The asymmetry between the two is the whole operating strategy.  Run the
    threshold **strict**, so the partition over-splits: a split is one visible
    mistake that a person fixes with one merge, while a bad merge is invisible
    contamination that quietly makes a class mean two things.  These two lists
    are how that repair survives — they are replayed on every re-cluster, so
    the work is done once rather than re-done whenever a number moves.

    Threshold merges are applied in increasing distance order, so the result
    does not depend on iteration order once constraints start blocking merges.
    """
    n = dist.shape[0]
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    # Groups that must stay apart, tracked by representative and kept current
    # as unions happen.
    forbidden: set[tuple[int, int]] = set()

    def forbid(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            forbidden.add((min(ra, rb), max(ra, rb)))

    def blocked(ra: int, rb: int) -> bool:
        return (min(ra, rb), max(ra, rb)) in forbidden

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        keep, gone = min(ra, rb), max(ra, rb)
        parent[gone] = keep
        nonlocal forbidden
        if forbidden:
            forbidden = {
                (
                    min(keep if x == gone else x, keep if y == gone else y),
                    max(keep if x == gone else x, keep if y == gone else y),
                )
                for x, y in forbidden
            }

    # Human merges first, so a subsequent separation is checked against the
    # groups the person actually meant rather than against raw rows.
    for a, b in must_link or ():
        union(a, b)

    for a, b in cannot_link or ():
        if find(a) == find(b):
            raise ValueError(
                f"rows {a} and {b} are adjudicated both same and different — "
                "one of the two verdicts is wrong, and guessing which would "
                "silently pick a side"
            )
        forbid(a, b)

    candidates = sorted(
        ((dist[i, j], i, j) for i in range(n) for j in range(i + 1, n) if dist[i, j] <= threshold),
        key=lambda t: (t[0], t[1], t[2]),
    )
    for _d, i, j in candidates:
        ri, rj = find(i), find(j)
        if ri == rj or blocked(ri, rj):
            continue
        keep, gone = min(ri, rj), max(ri, rj)
        parent[gone] = keep
        # Re-key every constraint that referenced the absorbed group.
        if forbidden:
            rekeyed = set()
            for x, y in forbidden:
                nx = keep if x == gone else x
                ny = keep if y == gone else y
                rekeyed.add((min(nx, ny), max(nx, ny)))
            forbidden = rekeyed

    # Relabel roots to dense 0..k-1 in first-appearance order, so the labelling
    # is a pure function of the distance matrix and not of dict iteration.
    remap: dict[int, int] = {}
    labels = []
    for i in range(n):
        root = find(i)
        if root not in remap:
            remap[root] = len(remap)
        labels.append(remap[root])
    return labels


def assign_class_ids(
    pages: list[Page],
    refs: Sequence[MarkRef],
    labels: Sequence[int],
    *,
    source: str,
    provenance: str = "clustered",
) -> dict[str, list[MarkRef]]:
    """Write clustered ``class_id``\\ s back onto the pages' marks.

    Class ids are derived from the cluster's *smallest page id*, not from its
    index, so adding pages to the corpus cannot silently renumber existing
    classes and invalidate a previous run's audit verdicts.
    """
    members: dict[int, list[MarkRef]] = {}
    for ref, label in zip(refs, labels):
        members.setdefault(label, []).append(ref)

    out: dict[str, list[MarkRef]] = {}
    for label, group in members.items():
        anchor = min(group, key=lambda r: (r.page_id, r.mark_index))
        kind = group[0].kind
        class_id = f"{source}/{kind}_{anchor.page_id.split('/')[-1]}_{anchor.mark_index}"
        out[class_id] = group
        for ref in group:
            mark = pages[ref.page_index].marks[ref.mark_index]
            pages[ref.page_index].marks[ref.mark_index] = Mark(
                kind=mark.kind,
                box=mark.box,
                class_id=class_id,
                provenance=provenance,
            )
    return out


def resolve_pairs(
    refs: Sequence[MarkRef],
    pairs: Sequence[tuple[str, str]],
) -> list[tuple[int, int]]:
    """Turn adjudicated ``(page_id, page_id)`` pairs into row-index pairs.

    Adjudications are stored against **page ids**, not row indices or class
    ids, because those are the only identifiers that survive a re-cluster.  A
    row index changes whenever the corpus grows; a class id changes whenever
    the threshold moves — and a human decision about two marks must outlive
    both, or every re-run quietly discards the annotation it was supposed to be
    built on.

    A pair naming a page that is no longer in the corpus is skipped rather than
    raising: pages come and go with tier budgets, and a dropped page is not a
    reason to refuse to build.
    """
    rows_by_page: dict[str, list[int]] = {}
    for index, ref in enumerate(refs):
        rows_by_page.setdefault(ref.page_id, []).append(index)

    out: list[tuple[int, int]] = []
    for left, right in pairs:
        for a in rows_by_page.get(left, ()):
            for b in rows_by_page.get(right, ()):
                if a != b:
                    out.append((a, b))
    return out


def collect_refs(pages: Sequence[Page], *, kinds: Iterable[str], source: str) -> list[MarkRef]:
    """Every unlabelled mark of the given *kinds* from *source*'s pages."""
    wanted = set(kinds)
    refs = []
    for pi, page in enumerate(pages):
        if page.source != source:
            continue
        for mi, mark in enumerate(page.marks):
            if mark.kind in wanted and mark.class_id is None:
                refs.append(MarkRef(pi, mi, page.page_id, mark.kind, mark.box))
    return refs


def cluster_source(
    pages: list[Page],
    source: str,
    *,
    kinds: Iterable[str] = ("logo", "stamp"),
    backend: str = "phash",
    threshold: float = 0.18,
    same: Optional[Sequence[tuple[str, str]]] = None,
    different: Optional[Sequence[tuple[str, str]]] = None,
    provenance: str = "clustered",
) -> dict[str, Any]:
    """Cluster one source's marks in place.  Returns a summary for the report.

    *same* and *different* are adjudicated ``(page_id, page_id)`` pairs. They
    survive re-clustering: a human decision about two marks must not be undone
    by someone later nudging the threshold.
    """
    refs = collect_refs(pages, kinds=kinds, source=source)
    if not refs:
        return {"source": source, "marks": 0, "classes": 0, "backend": backend, "threshold": threshold}

    desc = describe_marks(pages, refs, backend=backend)
    dist = distance_matrix(desc, refs, backend=backend)
    must_link = resolve_pairs(refs, same or ())
    cannot_link = resolve_pairs(refs, different or ())
    labels = single_linkage(dist, threshold, cannot_link=cannot_link, must_link=must_link)
    classes = assign_class_ids(pages, refs, labels, source=source, provenance=provenance)

    sizes = sorted((len(v) for v in classes.values()), reverse=True)
    return {
        "source": source,
        "marks": len(refs),
        "classes": len(classes),
        "backend": backend,
        "threshold": threshold,
        "merges_applied": len(must_link),
        "separations_applied": len(cannot_link),
        "largest_clusters": sizes[:10],
        "singletons": sum(1 for s in sizes if s == 1),
    }


def write_cluster_report(summaries: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# Adjudicated same/different decisions
# --------------------------------------------------------------------------
#
# The corpus stores *both* directions of the ground truth, because a benchmark
# needs both: same-mark pairs say what the detector must find, different-mark
# pairs say what it must keep apart.  Clustering can only ever propose the
# first; the second has to be adjudicated and then enforced, which is what
# `cannot_link` in single_linkage does with these.


def load_adjudications(path: Path) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """``(same, different)`` page-id pairs a human has ruled on.

    Both directions live in one file because they are one decision procedure
    seen from two sides, and because a pair appearing in both is a conflict
    that has to be catchable — which it is not if they sit in separate stores
    that nothing reads together.
    """
    if not path.exists():
        return [], []
    payload = json.loads(path.read_text(encoding="utf-8"))

    def rows(key: str) -> list[tuple[str, str]]:
        return [(r["left_page_id"], r["right_page_id"]) for r in payload.get(key, [])]

    return rows("same"), rows("different")


def save_adjudications(
    same: Sequence[dict[str, Any]],
    different: Sequence[dict[str, Any]],
    path: Path,
) -> None:
    """Write both directions, deduplicated and ordered, refusing a conflict.

    Pairs are stored unordered-within-pair (sorted) so that ruling on
    ``(a, b)`` and later ``(b, a)`` records one decision rather than two.
    """

    def canon(rows: Sequence[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
        out: dict[tuple[str, str], dict[str, Any]] = {}
        for row in rows:
            key = tuple(sorted((row["left_page_id"], row["right_page_id"])))
            merged = dict(row)
            merged["left_page_id"], merged["right_page_id"] = key
            out[key] = merged  # type: ignore[index]
        return out

    same_map, diff_map = canon(same), canon(different)
    clash = sorted(set(same_map) & set(diff_map))
    if clash:
        raise ValueError(
            f"{len(clash)} pair(s) adjudicated both same and different, e.g. {clash[0]} — "
            "resolve the verdicts before writing; storing both would let whichever "
            "is applied last silently win"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "same": [same_map[k] for k in sorted(same_map)],
        "different": [diff_map[k] for k in sorted(diff_map)],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
