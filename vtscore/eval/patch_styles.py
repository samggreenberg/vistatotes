"""Detection-style abstraction for the Max-Patch experiment.

The voting-iterations harness (:mod:`vtscore.eval.voting_iterations`) can run
each simulated detector under a named **detection style** - the bundle of rules
that decides (a) which vector a Good vote trains on, (b) which vector(s) a Bad
vote trains on, (c) how a trained MLP scores an image at inference, and (d) how
a cropped exemplar seeds the startup sort.  The styles are:

* ``whole_image`` - the classic single-vector pipeline (SigLIP et al.): every
  vote and every score uses the image-level embedding; region boxes are
  ignored.  The baseline arm.

* ``max_patch`` - **the production patch pipeline**: a Good region-vote trains
  on the **single raw patch** closest to the voted box
  (:func:`vtscore.media.patch_embed.nearest_patch_to_box`), a Bad vote floods
  the full-image vector + **every raw patch** of the image as negatives, and an
  image scores by max-pooling the MLP over the full-image vector plus all
  ``H x W`` raw patch vectors.  No region tree is consulted at any point.

* ``max_patch_hac`` / ``max_patch_pca_hac`` - the raw-patch-leaf HAC hybrids.
  These build a per-image binary tree whose leaves are the raw patches
  (:func:`build_patch_hac_tree`), snap a Good vote to the best-matching node,
  and flood / max-pool over every node.  They lost the Max-Patch study at the
  operating point despite ranking best, and the calibration study pinned that
  on calibration rather than geometry - the open "max-pool-aware calibration"
  follow-up in ``docs/plans/calibration-experiment.md`` is why they are still
  here.

The ``max_hac`` arm (the pre-#2886 production pipeline: K-means-pooled HAC
leaves, snap-to-node Good votes, CLS+leaf floods) is **gone**.  It lost the
study - ``docs/experiments/2026-07-29-max-patch/REPORT.md`` - and production no longer
carries the tree it delegated to, so the arm could only have been kept alive by
re-implementing the very code the study told us to delete.  Its numbers live in
the report.

Each style also maps a *query vector* (e.g. the full-image embedding of a
cropped exemplar) to per-image similarities for the Autopilot seed phase:
whole-image cosine, max-over-patches cosine, and max-over-tree-nodes cosine
respectively.

**Every vector a style can train a vote on must also be a row that style
scores over.**  ``max_patch`` originally scored raw patches only, so a *boxless*
Good vote - which falls back to the image-level vector - trained on a vector
that was never scored; the classifier then separated "full-image-like" from
"raw-patch-like" (every Bad vote floods raw patches as negatives) and the
calibrated threshold landed in a gap the production score distribution never
reaches - perfect ranking, zero FPR, catastrophic FNR.  The full-image row in
:meth:`MaxPatchStyle.score_rows` (and its matching negative in
:meth:`MaxPatchStyle.bad_vecs`) closes that hole.  The tree styles get the same
property from their CLS node.

Styles are **stateful per run**: :func:`resolve_style` returns a fresh instance
whose flattened score matrices are memoised per media-id set, so repeated
per-step scoring of the same test/sim split doesn't rebuild a multi-hundred-
thousand-row matrix 150 times.  Do not share one instance across datasets.

This is experiment-tier code.  ``max_patch`` mirrors the production vote/score
geometry in :mod:`vtscore.detectors.training` +
:mod:`vtscore.embedding.matrix`; the HAC node type and box-snap rule below are
experiment-only and live here rather than in :mod:`vtscore.media.patch_embed`,
which is now tree-free.
"""

from __future__ import annotations

import os

from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    import numpy as np
    import torch.nn as nn

from vtscore.embedding.media_vectors import media_embedding

#: Rows per forward-pass chunk when scoring a flattened patch matrix.  Patch
#: matrices are stored float16 (the pickle dtype) and upcast chunk-wise, so
#: peak float32 memory stays bounded regardless of dataset size.
_SCORE_CHUNK_ROWS = 65_536


def _unit(vec: "np.ndarray") -> "np.ndarray":
    import numpy as np  # noqa: PLC0415

    v = np.asarray(vec, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def _forward_sigmoid_chunked(model: "nn.Sequential", matrix: "np.ndarray") -> "np.ndarray":
    """Run ``sigmoid(model(matrix))`` in chunks; accepts a float16 or float32 matrix."""
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    device = next(model.parameters()).device
    out = np.empty(matrix.shape[0], dtype=np.float64)
    with torch.no_grad():
        for start in range(0, matrix.shape[0], _SCORE_CHUNK_ROWS):
            chunk = torch.from_numpy(np.ascontiguousarray(matrix[start : start + _SCORE_CHUNK_ROWS]))
            chunk = chunk.to(device=device, dtype=torch.float32)
            out[start : start + chunk.shape[0]] = torch.sigmoid(model(chunk)).squeeze(1).cpu().numpy()
    return out


def _segment_max(flat: "np.ndarray", seg_starts: "np.ndarray") -> "np.ndarray":
    import numpy as np  # noqa: PLC0415

    return np.maximum.reduceat(flat, seg_starts)


# ---------------------------------------------------------------------------
# Experiment-tier region tree (raw-patch-leaf HAC)
# ---------------------------------------------------------------------------


@dataclass
class RegionVector:
    """One node of an experiment-tier per-image region tree.

    Production is tree-free (#2886), so this type - and the tree builder and
    box-snap rule below - live here rather than in
    :mod:`vtscore.media.patch_embed`.  Only :func:`build_patch_hac_tree` and
    the two ``max_patch_hac`` styles construct them.

    The flat node list follows the convention: index 0 is the CLS whole-image
    node (``children = None``), then the raw-patch leaves (``children = None``),
    then the internal merge nodes whose ``children`` index earlier entries.
    """

    box: tuple[float, float, float, float]
    """Normalised image coordinates ``(x0, y0, x1, y1)``, each in ``[0, 1]``."""

    vec: "np.ndarray"
    """L2-normalised vector for this region, shape ``(D,)``."""

    children: Optional[tuple[int, int]] = None
    """Indices of the two children when this is an internal merge node."""


def snap_box_to_region(
    regions: list[RegionVector],
    box: tuple[float, float, float, float],
) -> "Optional[np.ndarray]":
    """Snap a user-drawn *box* to the tree node it best matches.

    Returns the L2-normalised (float32) vector of the node with the highest box
    IoU against *box* - i.e. one of the very candidates the style max-pools over
    at inference, so a Good region-vote trains in scoring geometry.

    When every IoU is zero (a degenerate zero-area drawn box), falls back to the
    node whose centroid is nearest the drawn box's centre.  The CLS whole-image
    node overlaps any in-bounds box, so a whole-image box collapses to the CLS
    vector (an image-level Good vote).  Returns ``None`` when *regions* is
    empty.
    """
    import numpy as np  # noqa: PLC0415

    if not regions:
        return None

    x0, y0, x1, y1 = (float(v) for v in box)
    dx0, dx1 = min(x0, x1), max(x0, x1)
    dy0, dy1 = min(y0, y1), max(y0, y1)
    d_area = max(0.0, dx1 - dx0) * max(0.0, dy1 - dy0)

    best_idx = 0
    best_iou = -1.0
    for idx, r in enumerate(regions):
        rx0, ry0, rx1, ry1 = r.box
        inter_w = max(0.0, min(dx1, rx1) - max(dx0, rx0))
        inter_h = max(0.0, min(dy1, ry1) - max(dy0, ry0))
        inter = inter_w * inter_h
        r_area = max(0.0, rx1 - rx0) * max(0.0, ry1 - ry0)
        union = d_area + r_area - inter
        iou = inter / union if union > 0.0 else 0.0
        if iou > best_iou:
            best_iou = iou
            best_idx = idx

    if best_iou <= 0.0:
        dcx, dcy = 0.5 * (dx0 + dx1), 0.5 * (dy0 + dy1)
        best_idx = min(
            range(len(regions)),
            key=lambda i: (
                (0.5 * (regions[i].box[0] + regions[i].box[2]) - dcx) ** 2
                + (0.5 * (regions[i].box[1] + regions[i].box[3]) - dcy) ** 2
            ),
        )

    # Node vectors are stored L2-normalised, but re-normalise defensively: a
    # float16 round-trip can drift the norm off 1.0 on upcast.
    return _unit(np.asarray(regions[best_idx].vec, dtype=np.float32))


def _fit_pca_projector(
    patch_grid: "np.ndarray",
    n_components: int,
) -> "Optional[Callable[[np.ndarray], np.ndarray]]":
    """Fit a per-image PCA on the patch grid and return a vec→reduced-vec projector.

    Fits :class:`sklearn.decomposition.PCA` on *this image's* flattened patch
    vectors ``(H*W, D)``.  The returned callable projects any original-space
    ``(D,)`` vector - or an ``(N, D)`` batch, in one ``pca.transform`` call -
    into the fitted ``k``-dim space and L2-normalises it, so it can be dropped
    into the cosine half of a HAC merge affinity to decide tree *topology* in a
    denoised space while the stored node vectors stay full-dim.
    ``n_components`` is clamped to ``min(n_components, H*W, D)``; returns
    ``None`` (caller falls back to the full-dim cosine) when ``k < 1``.
    """
    import numpy as np  # noqa: PLC0415

    height, width, dim = patch_grid.shape
    n_samples = height * width
    k = min(int(n_components), n_samples, dim)
    if k < 1:
        return None
    from sklearn.decomposition import PCA  # noqa: PLC0415

    matrix = patch_grid.reshape(n_samples, dim).astype(np.float32, copy=False)
    pca = PCA(n_components=k)
    pca.fit(matrix)

    def project(vec: "np.ndarray") -> "np.ndarray":
        arr = np.asarray(vec, dtype=np.float32)
        if arr.ndim == 1:
            return _unit(pca.transform(arr[None, :])[0])
        reduced = pca.transform(arr)
        norms = np.linalg.norm(reduced, axis=1, keepdims=True)
        return reduced / np.where(norms > 1e-12, norms, 1.0)

    return project


class WholeImageStyle:
    """Single-vector baseline: votes and scores use the image-level embedding."""

    name = "whole_image"

    def good_vec(self, media: dict[str, Any], box: Optional[tuple[float, float, float, float]]) -> "np.ndarray":
        return media_embedding(media)

    def bad_vecs(self, media: dict[str, Any]) -> list["np.ndarray"]:
        return [media_embedding(media)]

    def score_rows(self, media: dict[str, Any]) -> "np.ndarray":
        import numpy as np  # noqa: PLC0415

        return np.asarray(media_embedding(media), dtype=np.float32)[None, :]

    def score_media(self, model: "nn.Sequential", clips_dict: dict[int, dict[str, Any]]) -> dict[int, float]:
        import numpy as np  # noqa: PLC0415

        ids = sorted(clips_dict)
        if not ids:
            return {}
        matrix = np.stack([np.asarray(media_embedding(clips_dict[cid]), dtype=np.float32) for cid in ids])
        scores = _forward_sigmoid_chunked(model, matrix)
        return {cid: float(s) for cid, s in zip(ids, scores, strict=True)}

    def node_scores(
        self, model: "nn.Sequential", clips_dict: dict[int, dict[str, Any]]
    ) -> tuple[list[int], "np.ndarray", "np.ndarray"]:
        """``(ids, flat, seg_starts)`` — one node per image (the whole-image vector).

        The single-vector analogue of :meth:`_FlattenedStyle.node_scores`, so the
        calibration study can pool every style uniformly.  Each segment holds
        exactly one node, so ``seg_starts`` is ``0..N-1`` and ``flat`` is the
        per-image sigmoid.
        """
        import numpy as np  # noqa: PLC0415

        ids = sorted(clips_dict)
        if not ids:
            return [], np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int64)
        matrix = np.stack([np.asarray(media_embedding(clips_dict[cid]), dtype=np.float32) for cid in ids])
        flat = _forward_sigmoid_chunked(model, matrix)
        seg_starts = np.arange(len(ids), dtype=np.int64)
        return ids, flat, seg_starts

    def exemplar_sims(self, clips_dict: dict[int, dict[str, Any]], query_vec: "np.ndarray") -> dict[int, float]:
        import numpy as np  # noqa: PLC0415

        q = _unit(query_vec)
        ids = sorted(clips_dict)
        if not ids:
            return {}
        matrix = np.stack([_unit(media_embedding(clips_dict[cid])) for cid in ids])
        cos = matrix @ q
        return {cid: float(c) for cid, c in zip(ids, cos, strict=True)}


class _FlattenedStyle:
    """Shared max-pool machinery for the two patch styles.

    Subclasses provide :meth:`_rows_for_media` - the per-image stack of
    candidate vectors an image is max-pooled over (region-tree nodes for
    tree nodes for the HAC hybrids, raw patches for ``max_patch``).  The flattened
    ``(rows, seg_starts, ids)`` arrays are memoised per media-id set: region
    and patch vectors never change during a run, only the MLP weights do.

    The memo is **LRU-bounded**, because two of its callers hand it an id set
    that changes every step - the shrinking unlabeled pool, and the growing
    labelset the Smart indicator is measured on - so an unbounded dict grows a
    full flattened matrix per step.  A handful of entries is all any caller
    needs live at once (the stable simulation set, plus whichever moving set is
    being scored), and every miss costs only a re-flatten.
    """

    name = "abstract"

    #: How many flattened id sets stay memoised.  Sized for the sets one
    #: simulation step touches - the sim set, the pool, the labelset - with room
    #: to spare, not for a run's whole history.
    _MATRIX_CACHE_SIZE = 4

    def __init__(self) -> None:
        self._matrix_cache: OrderedDict[frozenset[int], tuple[list[int], Any, Any]] = OrderedDict()

    def _rows_for_media(self, media: dict[str, Any]) -> "np.ndarray":
        raise NotImplementedError  # pragma: no cover - abstract hook

    def score_rows(self, media: dict[str, Any]) -> "np.ndarray":
        """The rows this style max-pools over when scoring *media* at inference.

        Public counterpart of :meth:`_rows_for_media`, upcast to float32.  The
        calibrator uses it to collapse each vote's bag in **inference**
        geometry rather than in the geometry it happened to train on - see
        :func:`vtscore.training.thresholds.compute_fold_orderings`.
        """
        import numpy as np  # noqa: PLC0415

        return np.asarray(self._rows_for_media(media), dtype=np.float32)

    def _flattened(self, clips_dict: dict[int, dict[str, Any]]) -> tuple[list[int], "np.ndarray", "np.ndarray"]:
        import numpy as np  # noqa: PLC0415

        key = frozenset(clips_dict)
        cached = self._matrix_cache.get(key)
        if cached is not None:
            self._matrix_cache.move_to_end(key)
            return cached
        ids = sorted(clips_dict)
        blocks = [self._rows_for_media(clips_dict[cid]) for cid in ids]
        seg_starts = np.zeros(len(blocks), dtype=np.int64)
        np.cumsum([b.shape[0] for b in blocks[:-1]], out=seg_starts[1:])
        # Keep the flattened stack float16 (the pickle dtype) so a large
        # patch dataset doesn't double its memory here; the scorer upcasts
        # chunk-wise.
        matrix = np.concatenate(blocks, axis=0).astype(np.float16, copy=False)
        result = (ids, matrix, seg_starts)
        self._matrix_cache[key] = result
        while len(self._matrix_cache) > self._MATRIX_CACHE_SIZE:
            self._matrix_cache.popitem(last=False)
        return result

    def score_media(self, model: "nn.Sequential", clips_dict: dict[int, dict[str, Any]]) -> dict[int, float]:
        if not clips_dict:
            return {}
        ids, matrix, seg_starts = self._flattened(clips_dict)
        flat = _forward_sigmoid_chunked(model, matrix)
        pooled = _segment_max(flat, seg_starts)
        return {cid: float(s) for cid, s in zip(ids, pooled, strict=True)}

    def node_scores(
        self, model: "nn.Sequential", clips_dict: dict[int, dict[str, Any]]
    ) -> tuple[list[int], "np.ndarray", "np.ndarray"]:
        """``(ids, flat, seg_starts)`` — the per-node sigmoid scores before pooling.

        Exposes the raw ``flat`` vector and per-image segment boundaries that
        :meth:`score_media` collapses with a segment max, so the calibration
        study (issue #2781) can re-pool the same model's node scores under
        alternative rules (top-k mean, extreme-value ``pnorm``) from one forward
        pass.  ``ids[i]``'s nodes are ``flat[seg_starts[i]:seg_starts[i+1]]``.
        """
        import numpy as np  # noqa: PLC0415

        if not clips_dict:
            return [], np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int64)
        ids, matrix, seg_starts = self._flattened(clips_dict)
        flat = np.asarray(_forward_sigmoid_chunked(model, matrix), dtype=np.float64)
        return ids, flat, seg_starts

    def exemplar_sims(self, clips_dict: dict[int, dict[str, Any]], query_vec: "np.ndarray") -> dict[int, float]:
        import numpy as np  # noqa: PLC0415

        if not clips_dict:
            return {}
        q = _unit(query_vec)
        ids, matrix, seg_starts = self._flattened(clips_dict)
        flat = matrix.astype(np.float32, copy=False) @ q
        pooled = _segment_max(flat.astype(np.float64, copy=False), seg_starts)
        return {cid: float(s) for cid, s in zip(ids, pooled, strict=True)}


class MaxPatchStyle(_FlattenedStyle):
    """The production pipeline: nearest patch / all-patch flood / patch max-pool.

    Delegates every geometry decision to the production helpers so the harness
    and the live detector cannot drift: :func:`~vtscore.detectors.training.pool_box_from_media`
    for the Good vote and :func:`~vtscore.embedding.matrix.media_score_rows` for
    the flood / scoring stack.
    """

    name = "max_patch"

    def good_vec(self, media: dict[str, Any], box: Optional[tuple[float, float, float, float]]) -> "np.ndarray":
        from vtscore.detectors.training import pool_box_from_media  # noqa: PLC0415

        pooled = pool_box_from_media(media, box)
        # Image-level Good vote (or a grid-less media): the CLS/full-image
        # vector - the only image-level representative available.
        return pooled if pooled is not None else media_embedding(media)

    def bad_vecs(self, media: dict[str, Any]) -> list["np.ndarray"]:
        """The full-image vector plus every raw patch, as negatives.

        The full-image row is load-bearing: a Bad vote asserts that *no* row of
        this image should score high, and :meth:`_rows_for_media` max-pools the
        full-image row at inference.  Leaving it out would hand every image an
        un-suppressed scoring row.
        """
        from vtscore.detectors.training import bad_negative_vecs  # noqa: PLC0415

        return bad_negative_vecs(media)

    def _rows_for_media(self, media: dict[str, Any]) -> "np.ndarray":
        """The full-image vector stacked above every raw patch.

        Row 0 is the image-level (CLS) vector; without it a boxless Good vote
        (:meth:`good_vec` with ``box=None``) would train on a vector this
        scorer never evaluates.  See the module docstring.
        """
        import numpy as np  # noqa: PLC0415

        from vtscore.embedding.matrix import media_score_rows  # noqa: PLC0415

        rows = media_score_rows(media, dtype=np.float16)
        if rows is None:  # pragma: no cover - a media with no vector at all
            raise ValueError(f"media {media.get('id')!r} has no scoring rows")
        return rows


def build_patch_hac_tree(
    patch_grid: "np.ndarray",
    cls_vec: "Optional[np.ndarray]" = None,
    *,
    alpha: float = 0.5,
    pca_dims: "Optional[int]" = None,
) -> list:
    """Binary HAC tree with the **raw patches as leaves** - the MaxPatchHAC tree.

    Where the pre-#2886 production tree K-means-pooled patches into ~12 leaves
    *before* merging, this keeps every one
    of the ``H*W`` raw patches as its own leaf and agglomeratively merges them
    (blended cosine + spatial distance, average linkage) into progressively
    larger region nodes.  The tree therefore carries candidates at every scale
    from a single patch (which wins on small targets, like ``max_patch``) up to
    the whole image (which wins on large targets) at only ~2x
    the node count of the raw patches (``2*H*W - 1`` tree nodes + the CLS node).

    Returns a :class:`RegionVector` list: index 0 is the CLS whole-image
    node (when *cls_vec* is given), then the raw-patch leaves, then the internal
    merge nodes whose ``children`` index earlier entries in the list.  Internal
    node vectors are the L2-normalised **uniform** mean of their member patches
    (the experiment carries no per-patch saliency, so - unlike production -
    every patch counts equally).
    """
    import numpy as np  # noqa: PLC0415
    from scipy.cluster.hierarchy import linkage  # noqa: PLC0415
    from scipy.spatial.distance import squareform  # noqa: PLC0415

    grid = np.asarray(patch_grid, dtype=np.float32)
    height, width, dim = grid.shape
    n = height * width
    patches = grid.reshape(n, dim)
    norms = np.linalg.norm(patches, axis=1, keepdims=True)
    patches = patches / np.where(norms > 1e-12, norms, 1.0)

    rows, cols = np.divmod(np.arange(n), width)
    leaf_boxes = [
        (c / width, r / height, (c + 1) / width, (r + 1) / height) for r, c in zip(rows.tolist(), cols.tolist())
    ]

    if n > 1:
        # Optional PCA on the merge *order* only: decide affinities on cosines
        # in a per-image PCA-reduced space (denoised); stored node vecs stay
        # full-dim, so scoring is unchanged. pca_dims=None is the raw path.
        sim = patches
        if pca_dims:
            project = _fit_pca_projector(grid, int(pca_dims))
            if project is not None:
                sim = project(patches)  # one batched transform, not n per-vector calls
        cos_d = np.clip((1.0 - sim @ sim.T) * 0.5, 0.0, 1.0)
        centers = np.stack([(cols + 0.5) / width, (rows + 0.5) / height], axis=1).astype(np.float32)
        spatial = np.sqrt(((centers[:, None, :] - centers[None, :, :]) ** 2).sum(-1)) / np.sqrt(2.0)
        # *alpha* mixes two terms that share a nominal [0, 1] range but not a
        # realised one: *spatial* fills its range on every image by
        # construction, while *cos_d* only spans whatever the embedder's
        # patch-to-patch cosines happen to span.  A concentrated space keeps
        # cos_d in a narrow band near 0 and the merge order comes out mostly
        # spatial; a less concentrated one gives the cosine term more say at
        # the same *alpha*.  So the **effective** alpha is per-embedder, and
        # `max_patch_hac` results are not alpha-comparable across embedders
        # (#3347; the within-image patch cosine spread was not itself measured
        # by #3329, which read media-level vectors).  Whitening cos_d per image
        # would fix it, and would change the arm's definition — so it is a
        # caveat here rather than an edit.
        blended = alpha * cos_d + (1.0 - alpha) * spatial
        np.fill_diagonal(blended, 0.0)
        linkage_matrix = linkage(squareform(blended, checks=False), method="average")
    else:
        linkage_matrix = np.empty((0, 4))

    sums = [patches[i].copy() for i in range(n)]
    boxes = list(leaf_boxes)
    nodes = [RegionVector(box=leaf_boxes[i], vec=patches[i], children=None) for i in range(n)]
    for merge in linkage_matrix:
        a, b = int(merge[0]), int(merge[1])
        total = sums[a] + sums[b]
        sums.append(total)
        vec = total / max(float(np.linalg.norm(total)), 1e-12)
        ax0, ay0, ax1, ay1 = boxes[a]
        bx0, by0, bx1, by1 = boxes[b]
        box = (min(ax0, bx0), min(ay0, by0), max(ax1, bx1), max(ay1, by1))
        boxes.append(box)
        nodes.append(
            RegionVector(
                box=box,
                vec=vec.astype(np.float32),
                children=(a, b),
            )
        )

    if cls_vec is None:
        return nodes
    full = RegionVector(
        box=(0.0, 0.0, 1.0, 1.0),
        vec=_unit(np.asarray(cls_vec, dtype=np.float32)),
        children=None,
    )
    out = [full]
    for node in nodes:
        if node.children is None:
            out.append(node)
        else:
            ci, cj = node.children
            out.append(RegionVector(box=node.box, vec=node.vec, children=(ci + 1, cj + 1)))
    return out


class MaxPatchHacStyle(_FlattenedStyle):
    """Raw-patch-leaf HAC tree: multi-scale snap / all-node flood / all-node max-pool.

    The hybrid under test.  It builds a HAC tree whose leaves are the raw
    patches and merges them up a binary tree (:func:`build_patch_hac_tree`), so
    the tree carries candidates at every scale.  A Good region-vote **snaps to
    the tree node whose box best matches** the drawn box (multi-scale, like
    over a raw-patch-leaved tree); a Bad vote floods **every
    tree node** as a negative - symmetric with inference, which max-pools the
    MLP over every node; an image scores by max-pooling over all nodes.  The
    per-image tree is memoised per media id (it depends only on the frozen
    ``patch_grid``), so the 150-step trajectory builds each tree once.
    """

    name = "max_patch_hac"

    def __init__(self) -> None:
        super().__init__()
        self._tree_cache: "dict[int, list]" = {}

    def _tree(self, media: dict[str, Any]) -> list:
        mid = int(media.get("id", id(media)))
        cached = self._tree_cache.get(mid)
        if cached is not None:
            return cached
        import numpy as np  # noqa: PLC0415

        grid = media.get("patch_grid")
        if grid is None:
            tree = [
                RegionVector(
                    box=(0.0, 0.0, 1.0, 1.0),
                    vec=_unit(np.asarray(media_embedding(media), dtype=np.float32)),
                    children=None,
                )
            ]
        else:
            tree = build_patch_hac_tree(np.asarray(grid), media_embedding(media))
        self._tree_cache[mid] = tree
        return tree

    def good_vec(self, media: dict[str, Any], box: Optional[tuple[float, float, float, float]]) -> "np.ndarray":
        if box is not None and media.get("patch_grid") is not None:
            snapped = snap_box_to_region(self._tree(media), box)
            if snapped is not None:
                return snapped
        return media_embedding(media)

    def bad_vecs(self, media: dict[str, Any]) -> list["np.ndarray"]:
        import numpy as np  # noqa: PLC0415

        return [np.asarray(node.vec, dtype=np.float32) for node in self._tree(media)]

    def _rows_for_media(self, media: dict[str, Any]) -> "np.ndarray":
        import numpy as np  # noqa: PLC0415

        return np.stack([np.asarray(node.vec, dtype=np.float16) for node in self._tree(media)])


class MaxPatchPcaHacStyle(MaxPatchHacStyle):
    """MaxPatchHAC with a PCA-denoised merge order.

    Identical to :class:`MaxPatchHacStyle` except the raw-patch HAC tree's merge
    *order* is decided on cosines in a per-image PCA space (``pca_dims``
    components) rather than the full 768-dim patch space — the option ported
    from the HAC-tree-improvements branch.  Only the tree *topology* changes;
    every stored node vector stays full-dim, so the scoring / vote / flood
    machinery is exactly :class:`MaxPatchHacStyle`'s.  ``MAXPATCH_PCA_DIMS``
    (default 32) sets the reduced dimensionality.
    """

    name = "max_patch_pca_hac"
    pca_dims = int(os.environ.get("MAXPATCH_PCA_DIMS", "32"))

    def _tree(self, media: dict[str, Any]) -> list:
        mid = int(media.get("id", id(media)))
        cached = self._tree_cache.get(mid)
        if cached is not None:
            return cached
        import numpy as np  # noqa: PLC0415

        grid = media.get("patch_grid")
        if grid is None:
            tree = [
                RegionVector(
                    box=(0.0, 0.0, 1.0, 1.0),
                    vec=_unit(np.asarray(media_embedding(media), dtype=np.float32)),
                    children=None,
                )
            ]
        else:
            tree = build_patch_hac_tree(np.asarray(grid), media_embedding(media), pca_dims=self.pca_dims)
        self._tree_cache[mid] = tree
        return tree


#: Style-name registry.  Values are *classes*; :func:`resolve_style` returns a
#: fresh instance so per-run matrix memoisation never leaks across datasets.
STYLES: dict[str, type] = {
    WholeImageStyle.name: WholeImageStyle,
    MaxPatchStyle.name: MaxPatchStyle,
    MaxPatchHacStyle.name: MaxPatchHacStyle,
    MaxPatchPcaHacStyle.name: MaxPatchPcaHacStyle,
}


def resolve_style(name: str) -> Any:
    """Return a fresh style instance for *name*; raise ``KeyError`` on a typo."""
    try:
        cls = STYLES[name]
    except KeyError:
        raise KeyError(f"Unknown detection style {name!r}; available: {', '.join(sorted(STYLES))}") from None
    return cls()
