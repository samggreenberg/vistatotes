#!/usr/bin/env python3
"""Open-vocabulary detection over the queue, for the over-flagged review (#3720).

Why a detector rather than the VLM that was piloted: this pass needs a **box** to
put in front of the reviewer and a **score** to sort and cut on. The VLM gave
names only, and a yes/no answer has no dial -- the operating point was wherever
the model happened to sit (precision 0.76, recall 0.71 for `open`). A detector
returns a score per box, so the cut becomes a choice, and the reviewer's own
economics decide it: clicking Bad is cheap and drawing a box is not, so we buy
recall with precision deliberately rather than accepting what we are given.

Everything above a deliberately very low floor is written out, so the threshold
can be swept offline against COCO truth without paying for the GPU again.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

FLOOR = 0.03  # keep almost everything; the real cut is chosen later, offline


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="google/owlv2-base-patch16-ensemble")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    sys.path.insert(0, "scripts/experiments/pile")
    import pile_config as pc  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415
    from transformers import Owlv2ForObjectDetection, Owlv2Processor  # noqa: PLC0415

    classes = list(pc.SCALE_CLASSES)
    queries = [f"a photo of a {c}" for c in classes]

    rows = [json.loads(x) for x in Path(args.sample).read_text().splitlines() if x.strip()]
    if args.limit:
        rows = rows[: args.limit]
    print(f"[owl] {len(rows)} images, {len(classes)} queries; loading {args.model}", flush=True)

    proc = Owlv2Processor.from_pretrained(args.model)
    model = Owlv2ForObjectDetection.from_pretrained(args.model).to("cuda").eval()

    out = Path(args.out).open("w")
    t0 = time.time()
    done = 0
    for i in range(0, len(rows), args.batch):
        chunk = rows[i : i + args.batch]
        imgs, keep = [], []
        for r in chunk:
            try:
                imgs.append(Image.open(r["path"]).convert("RGB"))
                keep.append(r)
            except Exception as exc:  # noqa: BLE001 - a bad file must not sink the run
                out.write(json.dumps({**r, "error": str(exc)}) + "\n")
        if not imgs:
            continue
        inputs = proc(text=[queries] * len(imgs), images=imgs, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            outputs = model(**inputs)
        sizes = torch.tensor([[im.height, im.width] for im in imgs], device="cuda")
        results = proc.post_process_grounded_object_detection(outputs=outputs, target_sizes=sizes, threshold=FLOOR)
        for r, im, res in zip(keep, imgs, results):
            dets = []
            for score, label, box in zip(res["scores"], res["labels"], res["boxes"]):
                x0, y0, x1, y1 = (float(v) for v in box)
                dets.append(
                    {
                        "cls": classes[int(label)],
                        "score": round(float(score), 4),
                        # normalised, matching how the pile stores regions
                        "box": [
                            round(x0 / im.width, 5),
                            round(y0 / im.height, 5),
                            round(x1 / im.width, 5),
                            round(y1 / im.height, 5),
                        ],
                    }
                )
            dets.sort(key=lambda d: -d["score"])
            out.write(json.dumps({**r, "dets": dets}) + "\n")
        out.flush()
        done += len(keep)
        if (i // args.batch) % 10 == 0:
            rate = done / (time.time() - t0)
            print(
                f"[owl] {done}/{len(rows)}  {rate:.1f} img/s  eta {(len(rows) - done) / max(rate, 1e-9) / 60:.1f} min",
                flush=True,
            )
    out.close()
    print(f"[owl] done in {(time.time() - t0) / 60:.1f} min -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
