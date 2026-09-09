"""Are the images an ambiguous name withholds the pool's HARD negatives?

:data:`pile_config.SCALE_VG_AMBIGUOUS` is priced by ``name_evidence.py`` as a
count: ``1 / precision`` images leave the shared negative pool per contaminated
negative removed. That treats every withheld image as interchangeable, and they
are not. A name's *precision* is the share of what it withholds that really does
hold the class; the remaining ``1 - precision`` are true negatives being thrown
away, and **which** true negatives decides whether the table is repairing the
pool or leaking it.

The two ends of the range, both already in the shipped table:

* `bike` for `bicycle` -- 47% precision. Nearly half of what it withholds is a
  bicycle wrongly serving as its own class's negative. Withholding repairs.
* `sign` for `stop sign` -- 7.9%. Twelve of every thirteen withheld images are
  true negatives, and every one of them contains an object of the class's
  immediate SUPERORDINATE category. Those are the negatives a stop-sign detector
  exists to be discriminated from; withholding them does not repair the pool, it
  removes the part of it that makes the class measurable (#3635).

The definitional argument above is strong but it is still an argument. This
measures it: rank the drawn negative pool by the class's own text query -- the
same tower ``make_audit_slate.py`` uses to find its `boundary` stratum -- and ask
where the withheld images sit. If they concentrate at the top, the name is
withholding exactly the images that are hard to tell from the class.

Reports the withheld share of each top-k prefix against their share of the whole
pool. A ratio of 1.0 means the name withholds indiscriminately and only the count
matters; well above 1.0 means the count is the wrong instrument.

Usage::

    python withheld_difficulty.py --class "stop sign" --names sign,signs --out hard.json
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pile_config as pc

pc.setup_env()

VG_ROOT = pc.DEMO_CACHE / "visual_genome"


def log(msg: str) -> None:
    print(f"[hard] {msg}", flush=True)


def load_cell(path: Path) -> dict:
    with path.open("rb") as fh:
        obj = pickle.load(fh)  # noqa: S301 - our own artefact
    return obj["medias"] if isinstance(obj, dict) and "medias" in obj else obj


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--class", dest="cls", required=True, help="a class in SCALE_CLASSES")
    ap.add_argument("--names", required=True, help="comma-separated VG names the proposal would withhold")
    ap.add_argument("--cell", default=str(pc.EMBEDDINGS / "vg_scale__siglip.pkl"))
    ap.add_argument("--embedder", default="siglip")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.cls not in pc.SCALE_CLASSES:
        raise SystemExit(f"{args.cls!r} is not in SCALE_CLASSES")
    names = {n.strip().lower() for n in args.names.split(",") if n.strip()}

    from vtscore.embedding import embed_text_query  # noqa: PLC0415
    from vtscore.embedding.media_vectors import media_embedding  # noqa: PLC0415

    medias = load_cell(Path(args.cell))
    log(f"loaded {len(medias)} medias from {Path(args.cell).name}")

    # The drawn shared pool: a media with no `categories` is a negative for every
    # cell. Spares carry an empty `evaluable_categories` and are NOT in the pool,
    # so they are excluded here too -- the question is about the images a cell
    # actually scores against.
    pool = [i for i, m in medias.items() if not m.get("categories") and m.get("evaluable_categories")]
    log(f"{len(pool)} images in the drawn negative pool")

    log(f"loading VG objects.json ({(VG_ROOT / 'objects.json').stat().st_size / 1e6:.0f} MB)")
    with (VG_ROOT / "objects.json").open() as fh:
        records = json.load(fh)
    pool_set = set(pool)
    withheld: set[int] = set()
    for rec in records:
        iid = int(rec["image_id"])
        if iid not in pool_set:
            continue
        for obj in rec.get("objects") or []:
            nm = (obj.get("names") or [None])[0]
            if nm and str(nm).strip().lower() in names:
                withheld.add(iid)
                break
    log(f"{len(withheld)} of them carry one of the {len(names)} names ({100.0 * len(withheld) / len(pool):.1f}%)")

    ids = sorted(pool)
    mat = np.stack([np.asarray(media_embedding(medias[i]), dtype=np.float32) for i in ids])
    mat /= np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    tvec = embed_text_query(args.cls, "image", embedder_name=args.embedder)
    if tvec is None:
        raise SystemExit(f"no text tower for embedder {args.embedder!r}")
    tvec = np.asarray(tvec, dtype=np.float32)
    tvec /= np.linalg.norm(tvec) + 1e-12
    scores = mat @ tvec

    order = np.argsort(-scores)
    ranked = [ids[j] for j in order]
    base = len(withheld) / len(ranked)

    rows = []
    print("\n" + "=" * 78)
    print(f"WITHHELD DIFFICULTY -- pool ranked by the `{args.cls}` text query ({args.embedder})")
    print(f"base rate: {100 * base:.1f}% of the {len(ranked)} drawn negatives carry one of these names")
    print("=" * 78)
    print("%-10s %10s %10s %10s" % ("top-k", "withheld", "share", "vs base"))
    for k in (50, 100, 250, 500, 1000, 2000, len(ranked)):
        if k > len(ranked):
            continue
        w = sum(1 for i in ranked[:k] if i in withheld)
        share = w / k
        rows.append({"k": k, "withheld": w, "share": share, "lift": share / base if base else 0.0})
        print("%-10d %10d %9.1f%% %9.2fx" % (k, w, 100 * share, share / base if base else 0.0))

    wr = [n for n, i in enumerate(ranked) if i in withheld]
    rr = [n for n, i in enumerate(ranked) if i not in withheld]
    med_w = float(np.median(wr)) / len(ranked) if wr else float("nan")
    med_r = float(np.median(rr)) / len(ranked) if rr else float("nan")
    print(
        f"\nmedian percentile rank: withheld {100 * med_w:.0f}%, retained {100 * med_r:.0f}% (0% = most {args.cls}-like)"
    )

    report = {
        "class": args.cls,
        "names": sorted(names),
        "embedder": args.embedder,
        "pool": len(ranked),
        "withheld": len(withheld),
        "base_rate": base,
        "prefixes": rows,
        "median_pct_rank_withheld": med_w,
        "median_pct_rank_retained": med_r,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
        log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
