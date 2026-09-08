#!/usr/bin/env python3
"""Qwen2.5-VL grounding over the queue (#3720).

Open-vocabulary *naming with boxes*, because that is the framing the pilot
actually measured as better: handing the model our 25 rules cost recall badly
(0.56 vs 0.71 overall; `vase` 0.11 vs 0.79), so the model is asked only to see
and name, and our own table decides what its words mean.

**Coordinates.** Qwen2.5-VL emits absolute pixel coordinates in the *resized*
frame the vision tower actually saw, not the original image, and the resize is
`smart_resize`'s doing rather than a plain scale. Getting this wrong yields boxes
that look plausible and sit in the wrong place, so the resized size is read back
off `image_grid_thw` (patch units x 14) and every box is divided by it -- and
anything that still lands outside the frame is counted and reported rather than
silently clipped.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

PROMPT = (
    "Detect every distinct physical object in this image. For each one output "
    "its bounding box and a short common-noun label. Include small and partly "
    "hidden objects. Return ONLY a JSON array of "
    '{"bbox_2d": [x1, y1, x2, y2], "label": "..."} and nothing else.'
)


def parse_dets(text: str) -> list[dict]:
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        got = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    out = []
    for d in got:
        if not isinstance(d, dict):
            continue
        box, lab = d.get("bbox_2d"), d.get("label")
        if isinstance(box, list) and len(box) == 4 and isinstance(lab, str):
            try:
                out.append({"box_px": [float(v) for v in box], "label": lab.strip().lower()})
            except (TypeError, ValueError):
                continue
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    args = ap.parse_args()

    import torch  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration  # noqa: PLC0415

    rows = [json.loads(x) for x in Path(args.sample).read_text().splitlines() if x.strip()]
    if args.limit:
        rows = rows[: args.limit]
    print(f"[qwen] {len(rows)} images; loading {args.model}", flush=True)
    proc = AutoProcessor.from_pretrained(args.model)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda").eval()

    out = Path(args.out).open("w")
    t0, done, oob = time.time(), 0, 0
    for i, r in enumerate(rows, 1):
        try:
            img = Image.open(r["path"]).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            out.write(json.dumps({**r, "error": str(exc)}) + "\n")
            continue
        msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": PROMPT}]}]
        inputs = proc.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to("cuda")
        with torch.inference_mode():
            gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        reply = proc.decode(gen[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)

        # the frame the vision tower saw: grid is in 14px patch units
        grid = inputs["image_grid_thw"][0].tolist()
        rh, rw = grid[1] * 14, grid[2] * 14
        dets = []
        for d in parse_dets(reply):
            x0, y0, x1, y1 = d["box_px"]
            nb = [x0 / rw, y0 / rh, x1 / rw, y1 / rh]
            if not all(-0.02 <= v <= 1.02 for v in nb) or nb[2] <= nb[0] or nb[3] <= nb[1]:
                oob += 1
                continue
            dets.append({"label": d["label"], "box": [round(min(max(v, 0.0), 1.0), 5) for v in nb]})
        out.write(json.dumps({**r, "resized": [rw, rh], "dets": dets, "raw": reply[:400]}) + "\n")
        out.flush()
        done += 1
        if i % 25 == 0:
            rate = done / (time.time() - t0)
            print(
                f"[qwen] {i}/{len(rows)}  {rate:.2f} img/s  "
                f"eta {(len(rows) - i) / max(rate, 1e-9) / 60:.1f} min  oob={oob}",
                flush=True,
            )
    out.close()
    print(f"[qwen] done in {(time.time() - t0) / 60:.1f} min, {oob} boxes out of frame -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
