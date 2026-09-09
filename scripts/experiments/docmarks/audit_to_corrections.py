#!/usr/bin/env python
"""Fold filled-in audit verdicts back into the corpus.

    python audit_to_corrections.py --task merge --apply        # the merge slate
    python audit_to_corrections.py --task membership --apply
    python audit_to_corrections.py --task cluster --apply
    python audit_to_corrections.py --task confusable --apply
    python audit_to_corrections.py --task letterhead          # dry run (default)

Without ``--apply`` it prints what it would change and touches nothing.

Verdicts are additive and idempotent: they are recorded in ``classes.json``
under each class's ``audit`` block, and re-running with the same verdict file is
a no-op.  Nothing is ever deleted — a class judged ``generic`` keeps all its
instances and simply stops being part of the headline stratum, so both numbers
stay available and the decision stays visible.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import docmarks_config as cfg  # noqa: E402
from sources._common import Mark, Page, read_manifest, write_manifest  # noqa: E402


def load_verdicts(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"no verdict file at {path} — run make_audit_slate.py first")
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                row = json.loads(line)
                if str(row.get("verdict", "")).strip():
                    out.append(row)
    return out


def apply_cluster(
    pages: list[Page], classes: dict[str, Any], verdicts: list[dict[str, Any]]
) -> tuple[list[str], list[str], list[str]]:
    """``ok`` / ``split`` / ``merge_into:<id>`` / ``drop`` on derived classes.

    Returns ``(changes, problems, resplit)``; *resplit* names classes the
    reviewer judged over-merged, which the caller re-clusters at a tighter
    threshold.
    """
    changes: list[str] = []
    problems: list[str] = []
    resplit: list[str] = []

    for row in verdicts:
        class_id = row["class_id"]
        verdict = str(row["verdict"]).strip()
        meta = classes.get(class_id)
        if meta is None:
            problems.append(f"{class_id}: not in classes.json")
            continue

        if verdict == "ok":
            meta["audit"]["cluster_ok"] = True
            changes.append(f"{class_id}: confirmed")
        elif verdict == "split":
            # A contact sheet says "this holds more than one mark"; it cannot
            # say which crop belongs to which. Rather than leave the class dead,
            # re-cluster *only its own instances* at a tighter threshold and
            # re-sheet the pieces. That converges: each round either resolves
            # into confirmable classes or gets split again, and no other class
            # is disturbed.
            meta["audit"]["cluster_ok"] = False
            meta["audit"]["notes"] = (row.get("notes") or "over-merged; re-cluster this class alone").strip()
            resplit.append(class_id)
            changes.append(f"{class_id}: queued for re-clustering at a tighter threshold")
        elif verdict.startswith("merge_into:"):
            target = verdict.split(":", 1)[1].strip()
            if target not in classes:
                problems.append(f"{class_id}: merge target {target!r} does not exist")
                continue
            for page in pages:
                for i, mark in enumerate(page.marks):
                    if mark.class_id == class_id:
                        page.marks[i] = Mark(mark.kind, mark.box, target, mark.provenance)
            classes[target]["n_instances"] += meta["n_instances"]
            classes[target]["page_ids"] = sorted(set(classes[target]["page_ids"]) | set(meta["page_ids"]))
            classes[target]["audit"]["cluster_ok"] = True
            classes.pop(class_id)
            changes.append(f"{class_id}: merged into {target}")
        elif verdict == "drop":
            for page in pages:
                page.marks = [m for m in page.marks if m.class_id != class_id]
            classes.pop(class_id)
            changes.append(f"{class_id}: dropped")
        else:
            problems.append(f"{class_id}: unrecognised verdict {verdict!r}")

    return changes, problems, resplit


def apply_membership(
    pages: list[Page], classes: dict[str, Any], verdicts: list[dict[str, Any]]
) -> tuple[list[str], list[str]]:
    """Remove hand-rejected instances and mark the class fully verified.

    A rejected crop loses its ``class_id`` but keeps its box and stays on its
    page.  It is not deleted, for two reasons: the page remains a *known*
    negative for this class — same scanner, same paper, verified clean, which is
    the hardest and most useful kind of negative — and the mark itself is still
    a real mark that a later roster may want.

    Setting ``membership_verified`` is the point of the pass.  Before it, a
    class is a clustering proposal; after it, every positive in the eval has
    been looked at, so a miss is the detector's fault and not possibly the
    label's.
    """
    changes: list[str] = []
    problems: list[str] = []

    for row in verdicts:
        class_id = row["class_id"]
        meta = classes.get(class_id)
        if meta is None:
            problems.append(f"{class_id}: not in classes.json")
            continue

        raw = str(row["verdict"]).strip().lower()
        page_ids: list[str] = row.get("page_ids", [])
        if raw == "ok":
            rejected_idx: list[int] = []
        else:
            try:
                rejected_idx = sorted({int(tok) for tok in raw.replace(" ", "").split(",") if tok})
            except ValueError:
                problems.append(f"{class_id}: verdict must be 'ok' or comma-separated indices, got {raw!r}")
                continue
        out_of_range = [i for i in rejected_idx if not 0 <= i < len(page_ids)]
        if out_of_range:
            problems.append(f"{class_id}: index/indices {out_of_range} are outside 0..{len(page_ids) - 1}")
            continue

        dropped = {page_ids[i] for i in rejected_idx}
        if dropped:
            for page in pages:
                if page.page_id in dropped:
                    page.marks = [
                        Mark(m.kind, m.box, None, m.provenance) if m.class_id == class_id else m for m in page.marks
                    ]
            meta["page_ids"] = [p for p in meta["page_ids"] if p not in dropped]
            meta["n_instances"] = len(meta["page_ids"])

        meta["audit"]["membership_verified"] = True
        meta["audit"]["rejected_page_ids"] = sorted(dropped)
        changes.append(f"{class_id}: verified, {len(dropped)} rejected, {meta['n_instances']} instance(s) remain")
    return changes, problems


def apply_distinctive(classes: dict[str, Any], verdicts: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    changes: list[str] = []
    problems: list[str] = []
    for row in verdicts:
        class_id = row["class_id"]
        verdict = str(row["verdict"]).strip().lower()
        meta = classes.get(class_id)
        if meta is None:
            problems.append(f"{class_id}: not in classes.json")
            continue
        if verdict not in ("distinctive", "generic"):
            problems.append(f"{class_id}: unrecognised verdict {verdict!r}")
            continue
        meta["audit"]["distinctive"] = verdict == "distinctive"
        changes.append(f"{class_id}: {verdict}")
    return changes, problems


def apply_confusable(
    pages: list[Page],
    classes: dict[str, Any],
    verdicts: list[dict[str, Any]],
) -> tuple[list[str], list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    """``same`` merges the two classes; ``different`` separates them for good.

    Both verdicts are recorded as page-id pairs and replayed on every future
    re-cluster, so an afternoon of merging is not undone the next time a
    threshold moves.

    The two are deliberately not symmetric in cost, which is the whole reason
    the threshold runs strict: a split shows up here as one obvious pair to
    merge, while a bad merge never shows up at all.

    A ``different`` verdict is the only way the corpus can state "these must be
    told apart", and it is stored against **page ids** rather than class ids so
    it survives every future re-cluster: class ids move when a threshold moves,
    page ids do not. Without that, each rebuild would silently discard the
    adjudication it was supposed to be built on.
    """
    changes: list[str] = []
    problems: list[str] = []
    separations: list[dict[str, Any]] = []
    merges: list[dict[str, Any]] = []
    #: A class merged away is gone from `classes`, but a later verdict may
    #: still name it; follow the chain rather than reporting a missing class.
    moved: dict[str, str] = {}

    def resolve(cid: str) -> str:
        seen = set()
        while cid in moved and cid not in seen:
            seen.add(cid)
            cid = moved[cid]
        return cid

    for row in verdicts:
        left, right = resolve(row["left_class_id"]), resolve(row["right_class_id"])
        verdict = str(row["verdict"]).strip().lower()
        if left == right:
            changes.append(f"{row['left_class_id']} / {row['right_class_id']}: already one class")
            continue
        lmeta, rmeta = classes.get(left), classes.get(right)
        if lmeta is None or rmeta is None:
            problems.append(f"{left} / {right}: one of the pair is not in classes.json")
            continue

        if verdict == "same":
            # Merge into the larger class, so the surviving id is the one whose
            # instances dominate it and the name keeps meaning what it meant.
            keep, gone = (left, right) if lmeta["n_instances"] >= rmeta["n_instances"] else (right, left)
            kmeta, gmeta = classes[keep], classes[gone]
            merges.append(
                {
                    "left_page_id": kmeta["page_ids"][0],
                    "right_page_id": gmeta["page_ids"][0],
                    "kept_class_id": keep,
                    "merged_class_id": gone,
                    "note": row.get("notes", ""),
                }
            )
            for page in pages:
                for i, mark in enumerate(page.marks):
                    if mark.class_id == gone:
                        page.marks[i] = Mark(mark.kind, mark.box, keep, mark.provenance)
            kmeta["page_ids"] = sorted(set(kmeta["page_ids"]) | set(gmeta["page_ids"]))
            kmeta["n_instances"] = len(kmeta["page_ids"])
            classes.pop(gone)
            moved[gone] = keep
            changes.append(f"{gone} merged into {keep} ({kmeta['n_instances']} instances)")
        elif verdict == "different":
            lmeta.setdefault("distinct_from", [])
            rmeta.setdefault("distinct_from", [])
            if right not in lmeta["distinct_from"]:
                lmeta["distinct_from"].append(right)
            if left not in rmeta["distinct_from"]:
                rmeta["distinct_from"].append(left)
            # One representative page per side is enough to pin the constraint,
            # and keeps the store small; the cannot-link propagates to the whole
            # group through union-find.
            separations.append(
                {
                    "left_page_id": lmeta["page_ids"][0],
                    "right_page_id": rmeta["page_ids"][0],
                    "left_class_id": left,
                    "right_class_id": right,
                    "note": row.get("notes", ""),
                }
            )
            changes.append(f"{left} != {right}: separation recorded")
        else:
            problems.append(f"{left} / {right}: unrecognised verdict {verdict!r} (expected same|different)")

    return changes, problems, separations, merges


# --------------------------------------------------------------------------
# The merge slate
# --------------------------------------------------------------------------
#
# `merges.txt` is a partition, not a stream of verdicts, and that is the whole
# reason the slate is usable: a reviewer states the few groups that are one mark
# and says nothing about the thousand pairs that obviously are not.  Everything
# below turns that statement back into the same/different pairs `apply_confusable`
# already knows how to record, so the merge slate adds an input format and not a
# second code path through the ground truth.

REVIEWED_ALL = "REVIEWED-ALL"


def parse_merge_groups(text: str, n_classes: int) -> tuple[list[dict[str, Any]], bool, list[str]]:
    """Parse a filled-in ``merges.txt`` into ``(groups, reviewed_all, problems)``.

    A group is a set of slate indices the reviewer says are one mark.  Groups
    that share an index are **unioned** rather than refused: "3 8" and "8 12" are
    two observations of one equivalence class, and treating that as a
    contradiction would punish a reviewer for writing down the same truth twice.
    Sameness is transitive; the file is allowed to be redundant about it.

    What is refused is anything that cannot be resolved into indices at all --
    an out-of-range number, a one-element group, a token that is not a number --
    because each of those is a typo whose silent interpretation would write a
    wrong permanent merge.
    """
    groups: list[dict[str, Any]] = []
    problems: list[str] = []
    reviewed_all = False

    for lineno, raw in enumerate(text.splitlines(), start=1):
        body, _, note = raw.partition("#")
        body = body.strip()
        if not body:
            continue
        if body.upper() == REVIEWED_ALL:
            reviewed_all = True
            continue

        indices: list[int] = []
        bad = False
        for token in body.replace(",", " ").split():
            try:
                idx = int(token)
            except ValueError:
                problems.append(f"line {lineno}: {token!r} is not a slate index")
                bad = True
                continue
            if not 0 <= idx < n_classes:
                problems.append(f"line {lineno}: index {idx} is outside the slate (0..{n_classes - 1})")
                bad = True
                continue
            indices.append(idx)
        if bad:
            continue
        if len(set(indices)) < 2:
            problems.append(f"line {lineno}: {body!r} names fewer than two distinct classes — a group needs a pair")
            continue
        groups.append({"indices": sorted(set(indices)), "note": note.strip(), "line": lineno})

    return _union_groups(groups), reviewed_all, problems


def _union_groups(groups: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fold groups that share an index into one, keeping every note."""
    parent: dict[int, int] = {}

    def find(i: int) -> int:
        parent.setdefault(i, i)
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[min(ra, rb)] = parent[max(ra, rb)] = min(ra, rb)

    for group in groups:
        head = group["indices"][0]
        for idx in group["indices"][1:]:
            union(head, idx)

    merged: dict[int, dict[str, Any]] = {}
    for group in groups:
        root = find(group["indices"][0])
        slot = merged.setdefault(root, {"indices": set(), "notes": [], "lines": []})
        slot["indices"].update(group["indices"])
        slot["lines"].append(group["line"])
        if group["note"]:
            slot["notes"].append(group["note"])

    return [
        {"indices": sorted(v["indices"]), "note": "; ".join(v["notes"]), "lines": sorted(v["lines"])}
        for _root, v in sorted(merged.items())
    ]


def merge_verdicts(
    index: dict[str, Any],
    groups: Sequence[dict[str, Any]],
    reviewed_all: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Compile the partition into ``confusable``-shaped verdict rows.

    Order matters and is not cosmetic: every ``same`` row is emitted before every
    ``different`` row, because :func:`apply_confusable` merges as it goes and
    follows the resulting chain when a later row names a class that has since
    been absorbed.  Emitting a separation first would pin it against a class id
    that is about to stop existing.

    ``reviewed_all`` is what licenses the separations, and it licenses exactly
    the appendix pairs -- the ones that got their own side-by-side sheet.  Pairs
    that appear nowhere but the far end of the distance ranking stay
    unadjudicated, because nobody looked at them; recording them would put a
    decision no human made into the file every future re-cluster is bound by,
    which is precisely the failure the separations exist to prevent.
    """
    by_index = {int(row["index"]): row["class_id"] for row in index.get("classes", [])}
    problems: list[str] = []
    rows: list[dict[str, Any]] = []

    #: Every index mapped to the class it will *be* once the merges are applied
    #: -- its group's lowest member, or itself. Two uses, and the first is a
    #: correctness gate rather than tidiness: a near pair inside a group must
    #: not also be separated, because `save_adjudications` refuses a pair ruled
    #: both ways and would abort the whole apply. The second is deduplication:
    #: once 3 and 8 are one class, the appendix pairs (3, 12) and (8, 12) are
    #: one statement about one pair of classes, and emitting both prints a
    #: contradiction-shaped log for a decision that was made once.
    root: dict[int, int] = {}
    for group in groups:
        for idx in group["indices"]:
            root[idx] = group["indices"][0]

    for group in groups:
        indices = group["indices"]
        missing = [i for i in indices if i not in by_index]
        if missing:
            problems.append(f"group {indices}: index {missing} is not on the slate")
            continue
        # A star from the first member, not the full clique: sameness is
        # transitive and `apply_confusable` merges the classes outright, so n-1
        # rows state the whole group and n(n-1)/2 would restate it.
        head = indices[0]
        for other in indices[1:]:
            rows.append(
                {
                    "left_class_id": by_index[head],
                    "right_class_id": by_index[other],
                    "verdict": "same",
                    "notes": group.get("note", ""),
                }
            )

    if reviewed_all:
        seen: set[frozenset[int]] = set()
        for pair in index.get("near_pairs", []):
            li, ri = int(pair["left_index"]), int(pair["right_index"])
            if li not in by_index or ri not in by_index:
                problems.append(f"near pair [{li}]/[{ri}]: not on the slate")
                continue
            key = frozenset((root.get(li, li), root.get(ri, ri)))
            if len(key) == 1:
                continue  # merged by the reviewer; not a separation
            if key in seen:
                continue  # a nearer appendix pair already separated these two classes
            seen.add(key)
            rows.append(
                {
                    "left_class_id": by_index[li],
                    "right_class_id": by_index[ri],
                    "verdict": "different",
                    "notes": f"slate REVIEWED-ALL; appendix rank {pair.get('rank')} at d={pair.get('distance')}",
                }
            )

    return rows, problems


def load_merge_answer(audit_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Read ``index.json`` + ``merges.txt`` and compile them into verdict rows."""
    index_path, answer_path = audit_dir / "index.json", audit_dir / "merges.txt"
    if not index_path.exists():
        raise SystemExit(f"no slate at {index_path} — run make_audit_slate.py --task merge first")
    if not answer_path.exists():
        raise SystemExit(f"no answer file at {answer_path} — the slate should have written a template")

    index = json.loads(index_path.read_text(encoding="utf-8"))
    groups, reviewed_all, problems = parse_merge_groups(
        answer_path.read_text(encoding="utf-8"), len(index.get("classes", []))
    )
    rows, more = merge_verdicts(index, groups, reviewed_all)
    n_same = sum(1 for r in rows if r["verdict"] == "same")
    print(f"  slate: {len(groups)} merge group(s) -> {n_same} same, {len(rows) - n_same} different")
    if not reviewed_all:
        print(f"  slate: no {REVIEWED_ALL} line — recording merges only, no separations")
    return rows, problems + more


def apply_letterhead(classes: dict[str, Any], verdicts: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    changes: list[str] = []
    problems: list[str] = []
    for row in verdicts:
        author = row.get("author", "?")
        try:
            hits = int(str(row["verdict"]).strip())
        except ValueError:
            problems.append(f"{author}: verdict must be the count of bands carrying a printed mark")
            continue
        sampled = int(row.get("sampled", 0)) or 1
        yield_frac = hits / sampled
        flag = "" if yield_frac >= 0.5 else "  <- under half; this pool may not be worth clustering"
        changes.append(f"{author}: candidate yield {yield_frac:.2f} ({hits}/{sampled}){flag}")
    return changes, problems


def resplit_classes(
    pages: list[Page],
    classes: dict[str, Any],
    class_ids: Sequence[str],
    *,
    backend: str,
    threshold: float,
    corpus: Path,
    min_mark_px: int = cfg.MIN_MARK_PX,
    factor: float = 0.5,
) -> list[str]:
    """Re-cluster each over-merged class alone, at a tighter threshold.

    Only that class's own instances are touched, so re-splitting one class can
    never disturb another's already-confirmed membership.  The resulting pieces
    come back as fresh candidate classes for the next ``cluster`` sheet.

    The pieces are **registered in** ``classes``, not merely written onto the
    marks.  ``assign_class_ids`` relabels the manifest, but the slate, the
    roster, the embedder and the report all read ``classes.json``; the only
    other thing that writes it is ``build_corpus.py``, which rebuilds from the
    sources and would discard this split along with every other audit verdict.
    Popping the parent without adding its pieces therefore does not defer the
    decision to the next sheet -- it deletes the class from everything
    downstream while leaving its marks pointing at ids nothing knows.
    """
    from build_corpus import admit_classes, write_query_crops
    from cluster_marks import assign_class_ids, describe_marks, distance_matrix, single_linkage

    notes: list[str] = []
    tighter = threshold * factor
    for class_id in class_ids:
        meta = classes.get(class_id)
        if meta is None:
            continue
        refs = _refs_for_class(pages, class_id)
        if len(refs) < 2:
            continue
        desc = describe_marks(pages, refs, backend=backend)
        dist = distance_matrix(desc, refs, backend=backend)
        labels = single_linkage(dist, tighter)
        source = class_id.split("/", 1)[0]
        provenance = "clustered_band" if meta.get("located_by") == "band" else "clustered"
        pieces = assign_class_ids(pages, refs, labels, source=source, provenance=provenance)
        classes.pop(class_id, None)

        # `min_instances=1`: the reviewer said this class holds more than one
        # mark, so every piece is a finding.  Sizing them out here would drop
        # the small ones silently; the roster is where a piece too small to
        # search gets excluded, and it can only exclude what it can see.
        inventory = {cid: [(r.page_index, r.mark_index) for r in group] for cid, group in pieces.items()}
        fresh, rejected = admit_classes(pages, inventory, min_instances=1, min_mark_px=min_mark_px, roster=None)
        for cid, fresh_meta in fresh.items():
            fresh_meta["audit"]["notes"] = f"re-clustered out of {class_id} at {tighter:.3f}"
            classes[cid] = fresh_meta
        write_query_crops(pages, inventory, fresh, corpus / "queries")
        note = f"{class_id}: re-clustered at {tighter:.3f} into {len(pieces)} piece(s)"
        if rejected:
            note += f", {len(rejected)} not admitted ({'; '.join(sorted(rejected.values()))})"
        notes.append(note)
    return notes


def _refs_for_class(pages: list[Page], class_id: str) -> list[Any]:
    from cluster_marks import MarkRef

    return [
        MarkRef(pi, mi, page.page_id, mark.kind, mark.box)
        for pi, page in enumerate(pages)
        for mi, mark in enumerate(page.marks)
        if mark.class_id == class_id
    ]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--task",
        required=True,
        choices=("merge", "membership", "cluster", "confusable", "distinctive", "letterhead"),
    )
    ap.add_argument("--corpus", type=Path, default=cfg.OUT)
    ap.add_argument("--apply", action="store_true", help="write the changes (default is a dry run)")
    ap.add_argument("--cluster-backend", default=cfg.CLUSTER_BACKEND, choices=("phash", "siglip"))
    ap.add_argument("--cluster-threshold", type=float, default=cfg.CLUSTER_THRESHOLD)
    ap.add_argument("--min-mark-px", type=int, default=cfg.MIN_MARK_PX)
    args = ap.parse_args(argv)

    classes_path = args.corpus / "classes.json"
    manifest_path = args.corpus / "corpus.jsonl"
    adjudications_path = args.corpus / "adjudications.json"
    classes = json.loads(classes_path.read_text(encoding="utf-8"))
    audit_dir = args.corpus / "audit" / args.task
    slate_problems: list[str] = []
    if args.task == "merge":
        # The slate is an input *format*, not a second way of recording ground
        # truth: it compiles to the same same/different rows the pairwise pass
        # produces and goes through the same applier.
        verdicts, slate_problems = load_merge_answer(audit_dir)
    else:
        verdicts = load_verdicts(audit_dir / "verdicts.jsonl")

    mutates_pages = args.task in ("cluster", "membership", "confusable", "merge")
    pages = list(read_manifest(manifest_path)) if mutates_pages else []
    new_separations: list[dict[str, Any]] = []
    new_merges: list[dict[str, Any]] = []
    resplit: list[str] = []

    if args.task == "membership":
        changes, problems = apply_membership(pages, classes, verdicts)
    elif args.task == "cluster":
        changes, problems, resplit = apply_cluster(pages, classes, verdicts)
    elif args.task in ("confusable", "merge"):
        changes, problems, new_separations, new_merges = apply_confusable(pages, classes, verdicts)
        problems += slate_problems
        if args.task == "merge":
            reviewed = any(str(r.get("notes", "")).startswith("slate REVIEWED-ALL") for r in verdicts)
            if reviewed and not problems:
                for meta in classes.values():
                    meta.setdefault("audit", {})["partition_reviewed"] = True
                changes.append(f"{len(classes)} class(es) marked partition_reviewed")
    elif args.task == "distinctive":
        changes, problems = apply_distinctive(classes, verdicts)
    else:
        changes, problems = apply_letterhead(classes, verdicts)

    for c in changes:
        print(f"  {c}")
    for p in problems:
        print(f"  PROBLEM: {p}")
    print(f"\n{len(changes)} change(s), {len(problems)} problem(s) from {len(verdicts)} filled verdict(s)")

    if not args.apply:
        print("dry run — pass --apply to write")
        return 1 if problems else 0

    if resplit:
        for note in resplit_classes(
            pages,
            classes,
            resplit,
            backend=args.cluster_backend,
            threshold=args.cluster_threshold,
            corpus=args.corpus,
            min_mark_px=args.min_mark_px,
        ):
            print(f"  {note}")
        print("  re-run make_audit_slate.py --task cluster to review the new pieces")

    if new_separations or new_merges:
        from cluster_marks import load_adjudications, save_adjudications

        old_same, old_diff = load_adjudications(adjudications_path)
        save_adjudications(
            [{"left_page_id": a, "right_page_id": b} for a, b in old_same] + new_merges,
            [{"left_page_id": a, "right_page_id": b} for a, b in old_diff] + new_separations,
            adjudications_path,
        )
        print(f"  wrote {len(new_merges)} merge(s) and {len(new_separations)} separation(s) to {adjudications_path}")

    classes_path.write_text(json.dumps(classes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if mutates_pages:
        write_manifest(pages, manifest_path)
    print(f"wrote {classes_path}" + (f" and {manifest_path}" if mutates_pages else ""))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
