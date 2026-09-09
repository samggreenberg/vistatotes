"""Re-issue one class's disputed verdicts after a definition change.

A class whose meaning differs between the halves of a dataset is not noisy, it
is two classes wearing one name. `book` was exactly that: COCO has no magazine
class, so its annotators put magazines in `book`, while the human pass applied
the narrower English reading -- leaving 21 verdicts on COCO's definition and 49
on a different one.

Fixing that is not a re-annotation of the class. Only the verdicts the rule
change can *flip* need revisiting: the reviewer's `absent` votes, since a
positive stays positive under a widened definition. Everything else stands.

The rule travels in the dataset name (`book incl magazines`) because a reviewer
cannot see a manifest while voting, and an unstated convention is what produced
the split in the first place. The wording lives in
`pile_config.SCALE_CLASS_RULES` rather than in whoever runs this, so a
re-review is issued under the corrected rule instead of the one that produced
the disputed verdicts (#3612).

Usage::

    python make_definition_reslate.py --class book
    python make_definition_reslate.py --class book --name "book incl magazines"
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from pathlib import Path

import pile_config as pc

pc.setup_env()


def log(msg: str) -> None:
    print(f"[reslate] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    base = pc.PILE.parent / "vgscale-3156"
    ap.add_argument("--class", dest="klass", default="book")
    ap.add_argument(
        "--name",
        default="",
        help="dataset name carrying the rule (default: the class's SCALE_CLASS_RULES name + ' reviewed')",
    )
    ap.add_argument(
        "--verdicts",
        default="/exp/sgreenberg/vgscale-3156-labelsets/verdicts_final_20260824.json,"
        "/exp/sgreenberg/vgscale-3156-labelsets/verdicts_audit_20260825.json",
    )
    ap.add_argument("--fresh", type=int, default=20, help="extra ranked negatives never reviewed")
    ap.add_argument("--out", default=str(base / "slates_redef"))
    ap.add_argument("--seed", type=int, default=20260825)
    args = ap.parse_args()

    from vtscore.embedding import embed_text_query  # noqa: PLC0415
    from vtscore.embedding.media_vectors import media_embedding  # noqa: PLC0415

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "calibration"))
    from _cells_io import load_medias  # noqa: PLC0415

    from build_pile import _vg_image_paths  # noqa: PLC0415

    cls = args.klass
    # The corrected wording, not whichever one the last slate happened to use:
    # a re-review issued under the rule that produced the disputed verdicts
    # would just reproduce them (#3612).
    # `reviewed` keeps this pass's detector distinct from the first pass's:
    # `ingest_slate.py` keys a manifest row by (image, class, detector), so two
    # slates of one class sharing a name would overwrite each other's rows.
    name = args.name or pc.review_name(cls, "reviewed")
    rule = pc.SCALE_CLASS_RULES.get(cls)
    if rule:
        log(f"rule: {rule.test}")

    verdicts = []
    for p in args.verdicts.split(","):
        if Path(p).exists():
            verdicts += json.loads(Path(p).read_text())
    # Only `absent` can flip when a definition widens.
    disputed = {v["image_id"]: v for v in verdicts if v["class"] == cls and v["human"] == "absent"}
    log(f"{len(disputed)} prior 'absent' verdicts on {cls!r} to revisit")

    medias = load_medias(pc.EMBEDDINGS / "vg_scale__siglip.pkl")
    paths = _vg_image_paths()
    rng = random.Random(args.seed)

    # Fresh candidates: ranked negatives nobody has ruled on, since a narrow
    # definition also suppresses what a reviewer bothers to look at.
    reviewed = {v["image_id"] for v in verdicts if v["class"] == cls}
    negatives = [i for i, m in medias.items() if not m.get("categories") and i not in reviewed]
    tvec = embed_text_query(cls, "image", embedder_name="siglip")
    if tvec is None:
        raise SystemExit("no text tower for siglip")
    import numpy as np  # noqa: PLC0415

    tv = np.asarray(tvec, dtype=np.float32)
    tv /= np.linalg.norm(tv) + 1e-12
    scored = []
    for i in negatives:
        v = np.asarray(media_embedding(medias[i]), dtype=np.float32)
        n = np.linalg.norm(v)
        if n:
            scored.append((float(v @ tv / n), i))
    scored.sort(reverse=True)
    fresh = [i for _s, i in scored[: args.fresh]]

    out = Path(args.out) / name.replace(" ", "_")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    rows = []
    for iid in list(disputed) + fresh:
        src = paths.get(iid)
        if src is None:
            continue
        (out / f"{iid}.jpg").write_bytes(src.read_bytes())
        rows.append(
            {
                "image_id": iid,
                "class": cls,
                "stratum": "redef" if iid in disputed else "redef_fresh",
                "cell": "",
                "text_score": 0.0,
                "reference": "absent",
                "exhaustive": "no",
                "n_boxes": 0,
                "detector": name,
            }
        )
    rng.shuffle(rows)
    with (out / "manifest.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    log(f"{len(rows)} images -> {out}  (dataset/detector name: {name!r})")
    print(f"\nimport path: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
