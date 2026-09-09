#!/usr/bin/env python
"""Re-run the DocMarks audit's similarity questions on a semantic embedder.

    python siglip_audit.py --embed                      # GPU: cache one vector per crop
    python siglip_audit.py --analyze                    # CPU: pairs, splits, query check
    python make_audit_slate.py --task merge --descriptor siglip2_l   # CPU: re-order the slate

``phash`` is the right descriptor for *clustering* marks: it is cheap, it runs
on 200k pages, and the corpus was built with it.  It is the wrong descriptor for
the questions the **audit** asks, and #3600 measured how wrong -- on corpus v2
the one literal duplicate on the slate (the ``DY.Secretary`` stamp, since
merged) ranked **83rd of 120** in the near-pair appendix, behind 82 pairs of
stamps nobody would confuse, while two internally-mixed classes took 37% of the
appendix between them.  The cause is not a threshold: a perceptual hash of a
blue rubber stamp on white paper measures ink layout, and two different stamps
of the same size in the same typeface have nearly the same ink layout.  The same
failure is already on record for UCSF letterhead bands.

So this module asks the audit's three similarity questions with ``siglip2_l``
instead, and asks them of the *instances* rather than of one exemplar:

``pairs``
    Which classes are nearest each other, ranked by the cosine distance between
    class **centroids**.  This is the near-pair appendix, re-derived.  A centroid
    is what makes it robust: the phash ranking read a single query crop, so one
    unrepresentative exemplar moved a whole class (#3599).

``splits``
    Which classes hold more than one mark, by clustering each class's own
    instances.  ``--task cluster`` already renders every instance for a human to
    look at; this proposes *where* the boundary falls, so the answer is a
    confirmation rather than a search.  The threshold is not invented: it is
    swept, and the sweep is recorded beside the answer.  One class-level flag is
    read off it -- ``mixed``, for a class whose two most distant instances sit
    further apart than the loosest threshold in the sweep.

``query check``
    Does a class's query crop retrieve its own class, and how much of it?  The
    eval searches with that crop, so these are the questions the eval will ask.
    The first is stated as the **rank of the class's own centroid** among all
    centroids, which is a property of the retrieval rather than of a distance
    whose scale nobody knows.  The phash version of that screen does not work --
    #3599 records it scoring the one confirmed-wrong crop second-best of 60 --
    and a screen that ranks a known defect near the top of the healthy pile is
    worse than none.  The second is **how many of the class's own instances the
    crop reaches** before the nearest other class's centroid, which is what a
    rank cannot see: #3610's five-mark StaVer class ranks 0 at distance 0.078
    because its query crop is a good instance of the largest of its five marks.

Both of those are screens for a *crop*.  ``mixed`` is the screen for a *class*,
and the two are not substitutes: #3610 is the case where a class holds five
marks and every crop-level number about it is healthy.

Vectors are cached to ``audit/siglip/vectors.npz`` so that only ``--embed``
needs a card: the slate render, the analysis and every re-render afterwards read
the cache and stay on the ``cpu`` partition, which is where the audit's own
launcher runs them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import docmarks_config as cfg  # noqa: E402
from sources._common import Page, read_manifest, spread  # noqa: E402

#: Where the cache lives, relative to the corpus root.
VECTORS = Path("audit") / "siglip" / "vectors.npz"
ITEMS = Path("audit") / "siglip" / "items.json"


# --------------------------------------------------------------------------- #
# embedding
# --------------------------------------------------------------------------- #


def _crop_bytes(page: Page, box: tuple[int, int, int, int], pad_frac: float = 0.08) -> bytes:
    """One mark, cropped and PNG-encoded, ready to hand to the embedder."""
    import io

    from PIL import Image

    x, y, w, h = box
    pad = int(round(max(w, h) * pad_frac))
    with Image.open(page.path) as im:
        crop = im.convert("RGB").crop(
            (max(0, x - pad), max(0, y - pad), min(im.width, x + w + pad), min(im.height, y + h + pad))
        )
    buf = io.BytesIO()
    crop.save(buf, format="PNG")
    return buf.getvalue()


def collect_items(pages: Sequence[Page], classes: dict[str, Any], *, max_per_class: int) -> list[dict[str, Any]]:
    """Every instance of every class, plus each class's query crop.

    Capped per class because the cost of a class is linear in its instances and
    the marginal instance stops moving a centroid quickly; the cap is recorded
    in the items file so a later reader knows a centroid is over a sample.

    The sample is **spread** over the class rather than taken off its head.  A
    head sample makes every split proposal a statement about a class's first
    ``max_per_class`` page ids instead of about the class: the two classes #3610
    was filed over are 27 and 30 instances against a cap of 24, and between them
    five marks live only in the tail the head sample never reaches.
    """
    by_id = {p.page_id: p for p in pages}
    items: list[dict[str, Any]] = []
    for class_id, meta in sorted(classes.items()):
        for page_id in spread(meta.get("page_ids", []), max_per_class):
            page = by_id.get(page_id)
            if page is None:
                continue
            for mark in page.marks:
                if mark.class_id == class_id and mark.area() > 0:
                    items.append({"class_id": class_id, "page_id": page_id, "kind": "instance", "box": list(mark.box)})
                    break
        query = meta.get("query_crop")
        if query and Path(query).exists():
            items.append(
                {"class_id": class_id, "page_id": meta.get("query_page_id", ""), "kind": "query", "path": query}
            )
    return items


def embed_items(items: Sequence[dict[str, Any]], pages: Sequence[Page], embedder: str) -> np.ndarray:
    """Embed each item's pixels, returning one L2-normalised row per item."""
    from vtscore.datasets.stages.embedding import embed_missing
    from vtscore.embedding.media_vectors import media_embedding

    by_id = {p.page_id: p for p in pages}
    medias: dict[int, dict[str, Any]] = {}
    for index, item in enumerate(items):
        if item["kind"] == "query":
            raw = Path(item["path"]).read_bytes()
            name = Path(item["path"]).name
        else:
            page = by_id[item["page_id"]]
            raw = _crop_bytes(page, tuple(item["box"]))
            name = f"{item['page_id'].replace('/', '__')}.png"
        medias[index] = {
            "id": index,
            "media_type": "image",
            "embedder": embedder,
            "duration": 0,
            "file_size": len(raw),
            "md5": "",
            "embeddings": {},
            "media_bytes": raw,
            "media_string": None,
            "filename": name,
            "category": item["class_id"],
            "categories": [item["class_id"]],
            "regions": [],
            "origin": {"importer": "docmarks_audit", "params": {"embedder": embedder}},
            "origin_name": f"{item['class_id']}:{item['kind']}:{item['page_id']}",
        }

    embed_missing(medias, embedder)

    vecs = []
    for index in range(len(items)):
        vec = media_embedding(medias[index], embedder)
        if vec is None:
            raise SystemExit(f"embedder {embedder!r} returned no vector for item {index} ({items[index]})")
        vecs.append(np.asarray(vec, dtype=np.float32))
    matrix = np.vstack(vecs)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


# --------------------------------------------------------------------------- #
# analysis
# --------------------------------------------------------------------------- #


def instances_by_class(items: Sequence[dict[str, Any]]) -> dict[str, list[int]]:
    """``{class_id: [row index, ...]}`` over the *instance* items only.

    Query crops are excluded everywhere they would otherwise be counted as
    members: a centroid a query crop helped build is a centroid partly compared
    with itself, and an instance count that includes the query overstates what
    a search would have to find.
    """
    by_class: dict[str, list[int]] = {}
    for i, item in enumerate(items):
        if item["kind"] == "instance":
            by_class.setdefault(item["class_id"], []).append(i)
    return by_class


def class_centroids(items: Sequence[dict[str, Any]], vecs: np.ndarray) -> tuple[list[str], np.ndarray]:
    """One L2-normalised centroid per class, over its *instance* vectors."""
    order: list[str] = []
    rows: list[np.ndarray] = []
    by_class = instances_by_class(items)
    for class_id in sorted(by_class):
        mean = vecs[by_class[class_id]].mean(axis=0)
        rows.append(mean / max(float(np.linalg.norm(mean)), 1e-12))
        order.append(class_id)
    return order, np.vstack(rows) if rows else np.zeros((0, vecs.shape[1]), dtype=np.float32)


def cosine_distance(matrix: np.ndarray) -> np.ndarray:
    """Pairwise cosine distance for L2-normalised rows, diagonal zeroed."""
    dist = 1.0 - matrix @ matrix.T
    np.fill_diagonal(dist, 0.0)
    return np.clip(dist, 0.0, 2.0)


def split_class(vectors: np.ndarray, threshold: float) -> np.ndarray:
    """Average-linkage sub-groups of one class's instances, at *threshold*.

    Average linkage rather than single: single linkage's failure mode is exactly
    the one this pass exists to catch -- one ambiguous crop bridging two marks
    into a chain -- and using it here would reproduce the defect while claiming
    to detect it.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    if len(vectors) < 2:
        return np.ones(len(vectors), dtype=int)
    condensed = squareform(cosine_distance(vectors), checks=False)
    return fcluster(linkage(condensed, method="average"), t=threshold, criterion="distance")


def split_report(
    items: Sequence[dict[str, Any]],
    vecs: np.ndarray,
    thresholds: Sequence[float],
    *,
    mixed_at: float = cfg.AUDIT_MIXED_MAX_WITHIN,
) -> list[dict[str, Any]]:
    """For every class: how it breaks up, at each threshold in the sweep.

    Reported as a sweep rather than a verdict because the operating point is a
    property of this corpus and this embedder, and picking one silently is how
    ``CLUSTER_THRESHOLD``'s 0.16 outlived the decomposition it was measured on.

    One flag *is* drawn from the sweep, because #3610 needed a screen and not
    just a table: a class is ``mixed`` when its two most distant instances sit
    further apart than ``mixed_at`` -- by default the loosest threshold in the
    sweep, i.e. wider than any cut that would still be called one mark.  That is
    the question the ``--task cluster`` sheets exist to adjudicate, and it is
    the one the query-crop rank cannot ask: the rank scores a *crop*, and a
    five-mark class whose query crop is a good instance of its largest mark
    scores in the healthiest tier while being the second most mixed class in the
    corpus.
    """
    by_class = instances_by_class(items)

    rows = []
    for class_id in sorted(by_class):
        idx = by_class[class_id]
        sub = vecs[idx]
        within = cosine_distance(sub)
        sweep = {}
        for t in thresholds:
            labels = split_class(sub, t)
            sizes = sorted((int((labels == k).sum()) for k in set(labels.tolist())), reverse=True)
            sweep[f"{t:.2f}"] = sizes
        max_within = float(within.max())
        rows.append(
            {
                "class_id": class_id,
                "n": len(idx),
                "median_within": round(float(np.median(within[np.triu_indices(len(idx), k=1)])), 3)
                if len(idx) > 1
                else 0.0,
                "max_within": round(max_within, 3),
                "mixed": bool(max_within >= mixed_at),
                "mixed_at": mixed_at,
                "sweep": sweep,
            }
        )
    rows.sort(key=lambda r: -r["max_within"])
    return rows


def query_check(
    items: Sequence[dict[str, Any]], vecs: np.ndarray, order: Sequence[str], centroids: np.ndarray
) -> list[dict[str, Any]]:
    """Does each class's query crop retrieve its own class -- and how much of it?

    Two numbers, because the eval asks two things of one crop.

    ``rank_of_own_class``
        The **rank** of the class's own centroid among all centroids, because
        that is what the eval does with the crop.  A distance alone cannot say
        whether 0.30 is fine, and #3599 is the case where a distance said fine.

    ``own_instances_reached``
        How many of the class's own instances are closer to the query crop than
        the nearest **other** class's centroid is.  The rank is a statement
        about one crop against 59 classes; this is a statement about that crop
        against the class it is supposed to stand for, and it is what catches a
        query crop that represents only part of its class.  #3610's five-mark
        StaVer class is exactly that: rank 0, distance 0.078, healthiest tier --
        and a crop of the 16-strong routing box says nothing about the other
        eleven instances.

        The cut-off is not invented either.  It is the distance to the nearest
        foreign centroid, i.e. the point past which a retrieval is picking up
        another class anyway, so "reached" means "would come back before the
        confusion starts".
    """
    position = {class_id: i for i, class_id in enumerate(order)}
    by_class = instances_by_class(items)
    rows = []
    for i, item in enumerate(items):
        if item["kind"] != "query" or item["class_id"] not in position:
            continue
        sims = centroids @ vecs[i]
        ranking = np.argsort(-sims)
        own = position[item["class_id"]]
        rank = int(np.where(ranking == own)[0][0])
        others = [int(j) for j in ranking if int(j) != own]
        nearest_other = others[0] if others else None

        idx = by_class.get(item["class_id"], [])
        own_sims = vecs[idx] @ vecs[i] if idx else np.zeros(0, dtype=np.float32)
        reached = int((own_sims >= sims[nearest_other]).sum()) if nearest_other is not None else len(idx)

        rows.append(
            {
                "class_id": item["class_id"],
                "query_page_id": item["page_id"],
                "rank_of_own_class": rank,
                "distance_to_own": round(float(1.0 - sims[own]), 3),
                "nearest_class": order[int(ranking[0])],
                "distance_to_nearest": round(float(1.0 - sims[int(ranking[0])]), 3),
                "own_instances": len(idx),
                "own_instances_reached": reached,
                "nearest_other_class": order[nearest_other] if nearest_other is not None else "",
                "distance_to_nearest_other": round(float(1.0 - sims[nearest_other]), 3)
                if nearest_other is not None
                else 0.0,
            }
        )
    # Worst first, by both screens at once: a crop that does not retrieve its own
    # class, then one that retrieves only part of it.
    rows.sort(
        key=lambda r: (
            -r["rank_of_own_class"],
            r["own_instances_reached"] / max(1, r["own_instances"]),
            -r["distance_to_own"],
        )
    )
    return rows


def pair_report(order: Sequence[str], centroids: np.ndarray, top: int) -> list[dict[str, Any]]:
    """The closest class pairs by centroid distance, nearest first."""
    dist = cosine_distance(centroids)
    n = len(order)
    pairs = [(float(dist[i, j]), i, j) for i in range(n) for j in range(i + 1, n)]
    pairs.sort(key=lambda t: (t[0], t[1], t[2]))
    return [
        {"rank": r, "left": order[i], "right": order[j], "distance": round(d, 4)}
        for r, (d, i, j) in enumerate(pairs[:top])
    ]


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #


def load_cache(corpus: Path) -> tuple[list[dict[str, Any]], np.ndarray]:
    vec_path, item_path = corpus / VECTORS, corpus / ITEMS
    if not vec_path.exists() or not item_path.exists():
        raise SystemExit(f"no vector cache at {vec_path} — run siglip_audit.py --embed first")
    items = json.loads(item_path.read_text(encoding="utf-8"))["items"]
    return items, np.load(vec_path)["vecs"]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=cfg.OUT)
    ap.add_argument("--embed", action="store_true", help="GPU: embed every crop and cache the vectors")
    ap.add_argument("--analyze", action="store_true", help="CPU: pairs, split proposals and the query check")
    ap.add_argument("--embedder", default=cfg.AUDIT_EMBEDDER)
    ap.add_argument("--max-per-class", type=int, default=cfg.AUDIT_MAX_PER_CLASS)
    ap.add_argument("--top-pairs", type=int, default=cfg.MERGE_SLATE_NEAR_PAIRS)
    args = ap.parse_args(argv)

    if not args.embed and not args.analyze:
        ap.error("nothing to do: pass --embed, --analyze, or both")

    pages = list(read_manifest(args.corpus / "corpus.jsonl"))
    classes = json.loads((args.corpus / "classes.json").read_text(encoding="utf-8"))
    out = args.corpus / "audit" / "siglip"
    out.mkdir(parents=True, exist_ok=True)

    if args.embed:
        items = collect_items(pages, classes, max_per_class=args.max_per_class)
        print(f"embedding {len(items)} crop(s) from {len(classes)} class(es) with {args.embedder}")
        vecs = embed_items(items, pages, args.embedder)
        np.savez_compressed(args.corpus / VECTORS, vecs=vecs)
        (args.corpus / ITEMS).write_text(
            json.dumps(
                {"embedder": args.embedder, "max_per_class": args.max_per_class, "items": items},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"  wrote {args.corpus / VECTORS}  {vecs.shape}")

    if args.analyze:
        items, vecs = load_cache(args.corpus)
        order, centroids = class_centroids(items, vecs)
        pairs = pair_report(order, centroids, args.top_pairs)
        splits = split_report(items, vecs, cfg.AUDIT_SPLIT_SWEEP)
        queries = query_check(items, vecs, order, centroids)

        (out / "pairs.json").write_text(json.dumps(pairs, indent=2) + "\n", encoding="utf-8")
        (out / "splits.json").write_text(json.dumps(splits, indent=2) + "\n", encoding="utf-8")
        (out / "query_check.json").write_text(json.dumps(queries, indent=2) + "\n", encoding="utf-8")

        print(
            f"pairs:   {len(pairs)} closest of {len(order)} classes, "
            f"{pairs[0]['distance']:.3f}–{pairs[-1]['distance']:.3f}"
        )
        print("  " + "\n  ".join(f"{p['distance']:.3f}  {p['left']}  vs  {p['right']}" for p in pairs[:5]))
        misses = [q for q in queries if q["rank_of_own_class"] > 0]
        print(f"query:   {len(misses)} of {len(queries)} query crop(s) do not retrieve their own class first")
        for q in misses[:5]:
            print(f"  rank {q['rank_of_own_class']:2d}  {q['class_id']}  (nearest: {q['nearest_class']})")

        # The second half of the query screen: a crop can retrieve its own class
        # first and still stand for only part of it (#3610).
        partial = [
            q for q in queries if q["rank_of_own_class"] == 0 and q["own_instances_reached"] < q["own_instances"]
        ]
        print(f"reach:   {len(partial)} query crop(s) rank their class first but reach only part of it")
        for q in partial[:8]:
            print(
                f"  {q['own_instances_reached']:3d}/{q['own_instances']:<3d} {q['class_id']:44s}"
                f" (cut at {q['distance_to_nearest_other']:.3f}, {q['nearest_other_class']})"
            )

        mixed = [s for s in splits if s["mixed"]]
        print(f"mixed:   {len(mixed)} of {len(splits)} class(es) spread wider than {cfg.AUDIT_MIXED_MAX_WITHIN:.2f}")
        for s in splits[:8]:
            flag = "MIXED" if s["mixed"] else "     "
            print(f"  {flag} {s['max_within']:.3f}  {s['class_id']:44s} n={s['n']:3d}  sweep={s['sweep']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
