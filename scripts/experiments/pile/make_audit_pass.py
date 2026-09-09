"""Build the reviewer's audit slate: the model's flags, plus the sample that bounds them.

A triage pass is only as good as its measured recall, and recall cannot be
measured from the flags themselves -- they are chosen to be positive. So the
audit has two parts, recorded separately and never pooled:

* ``flag`` -- every image the triage called a hidden positive. Reviewing these
  turns a suspicion into a correction, and their disagreement rate is the
  triage's *precision*.
* ``audit`` -- a uniform sample of the negatives the triage did **not** flag.
  This is the only part that can estimate what the pass missed; without it the
  residual error rate after triage is unknown, not small.

The two are shuffled together and named by image id, so the reviewer cannot tell
which is which and the audit stratum stays unbiased.

Usage::

    python make_audit_pass.py --classes bus,dog --audit-per-class 25
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path

import pile_config as pc

pc.setup_env()


def log(msg: str) -> None:
    print(f"[audit] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    base = pc.PILE.parent / "vgscale-3156"
    ap.add_argument("--triage", default=str(base / "tri_flags_all.json"))
    ap.add_argument("--sheets", default=str(base / "sheets_neg"))
    ap.add_argument("--slates", default=str(base / "slates"))
    ap.add_argument("--out", default=str(base / "slates_audit"))
    ap.add_argument("--classes", default="")
    ap.add_argument("--audit-per-class", type=int, default=20, help="unflagged negatives sampled per class")
    ap.add_argument("--seed", type=int, default=20260824)
    args = ap.parse_args()

    from build_pile import _vg_image_paths  # noqa: PLC0415

    paths = _vg_image_paths()
    flags = json.loads(Path(args.triage).read_text())
    want = [c.strip() for c in args.classes.split(",") if c.strip()] or sorted(flags)
    rng = random.Random(args.seed)
    out_root = Path(args.out)
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)

    index = []
    for cls in want:
        folder = cls.replace(" ", "_")
        idx_path = Path(args.sheets) / folder / "index.json"
        if not idx_path.exists():
            log(f"  {cls}: no sheet index, skipping")
            continue
        idx = json.loads(idx_path.read_text())
        by_tile = {(r["sheet"], r["tile"]): r for r in idx}
        flagged: dict[int, str] = {}
        for kind in ("definite", "maybe"):
            for sheet, tile in flags[cls][kind]:
                r = by_tile.get((sheet, tile))
                if r:
                    flagged[r["image_id"]] = kind

        unflagged = [r["image_id"] for r in idx if r["image_id"] not in flagged]
        sample = rng.sample(unflagged, min(args.audit_per_class, len(unflagged)))

        rows = []
        cdir = out_root / folder
        cdir.mkdir(parents=True, exist_ok=True)
        for iid, kind in sorted(flagged.items()):
            rows.append({"image_id": iid, "class": cls, "stratum": "flag", "triage": kind})
        for iid in sorted(sample):
            rows.append({"image_id": iid, "class": cls, "stratum": "audit", "triage": "none"})
        rng.shuffle(rows)
        written = []
        for r in rows:
            src = paths.get(r["image_id"])
            if src is None:
                continue
            (cdir / f"{r['image_id']}.jpg").write_bytes(src.read_bytes())
            written.append(
                {
                    **r,
                    "cell": "",
                    "text_score": 0.0,
                    "reference": "absent",  # every one of these is a current negative
                    "exhaustive": "no",
                    "n_boxes": 0,
                    "detector": pc.review_name(cls, "audit"),
                }
            )
        with (cdir / "manifest.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(written[0]))
            w.writeheader()
            w.writerows(written)
        n_flag = sum(1 for r in written if r["stratum"] == "flag")
        index.append({"class": cls, "dir": str(cdir), "n": len(written), "detector": pc.review_name(cls, "audit")})
        log(f"  {cls:<12}{len(written):4d} images  ({n_flag} flags + {len(written) - n_flag} audit)")

    (out_root / "slates.json").write_text(json.dumps(index, indent=1) + "\n")
    total = sum(e["n"] for e in index)
    print(f"\n{total} images across {len(index)} classes under {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
