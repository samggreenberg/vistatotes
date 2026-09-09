#!/usr/bin/env python3
"""Score a local VLM against COCO truth on the anchored half (#3720).

Two framings are run over the *same* images, because they fail differently and
the pilot exists to find out which failure we would rather have:

* ``ruled`` -- hand the model our 25 classes together with the wording of
  `SCALE_CLASS_RULES`, and ask which are present. This measures the thing we
  would actually deploy, but it makes the model both perceive AND adjudicate,
  so a miss cannot be attributed.
* ``open`` -- ask only what objects it can see, in its own words, and map those
  names onto our classes afterwards with our own rulebook. Perception and
  definition come apart: a name we fail to map is a *vocabulary* gap we can fix
  without re-running the model, and a rule change (`bench`, 2026-09-08) re-maps
  rather than re-infers.

The point of separating them is that a definitional disagreement is not a model
failure. "Is a magazine a Book" has no answer the model could have guessed --
COCO says yes, most people say no, and our rule says yes because COCO has no
magazine class. Blaming that on the VLM would understate it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, "scripts/experiments/pile")
import pile_config as pc  # noqa: E402

RULED_HEAD = (
    "You are labelling images for an object-detection dataset. "
    "Below are the object classes, each with the exact rule that defines it. "
    "The rules override your own intuitions -- follow them literally even where "
    "they disagree with ordinary usage.\n\n"
)
RULED_TAIL = (
    "\nLook at the image. Return ONLY a JSON array of the class names from the "
    "list above that are present, following the rules exactly. Return [] if none "
    "are present. No prose."
)
OPEN_PROMPT = (
    "List every distinct physical object you can clearly see in this image. "
    'Use ordinary common nouns, singular (for example: "car", "coffee mug", '
    '"potted plant"). Do not interpret or group them; just name what is there. '
    "Return ONLY a JSON array of strings. No prose."
)


def ruled_prompt() -> str:
    lines = []
    for cls in pc.SCALE_CLASSES:
        rule = pc.SCALE_CLASS_RULES.get(cls)
        test = getattr(rule, "test", "") or ""
        lines.append(f"- {cls}: {test}" if test else f"- {cls}")
    return RULED_HEAD + "\n".join(lines) + RULED_TAIL


def parse_json_array(text: str) -> list[str]:
    """Pull the first JSON array out of a reply, tolerating fences and prose."""
    m = re.search(r"\[.*?\]", text, re.S)
    if not m:
        return []
    try:
        got = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return [str(x).strip().lower() for x in got if isinstance(x, (str, int, float))]


def main() -> int:
    pc.setup_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="google/gemma-4-12B-it")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    import torch  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415
    from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: PLC0415

    rows = [json.loads(x) for x in Path(args.sample).read_text().splitlines() if x.strip()]
    if args.limit:
        rows = rows[: args.limit]
    print(f"[vlm] {len(rows)} images; loading {args.model} ...", flush=True)

    t0 = time.time()
    proc = AutoProcessor.from_pretrained(args.model)
    # No `device_map` here: it needs `accelerate`, which this venv does not have.
    # A 12B in bf16 is ~24 GB and the card is 48 GB, so a plain single-device
    # load is both sufficient and one less dependency.
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16)
    model = model.to("cuda").eval()
    print(f"[vlm] loaded in {time.time() - t0:.0f}s", flush=True)

    prompts = {"ruled": ruled_prompt(), "open": OPEN_PROMPT}
    out = Path(args.out).open("w")
    t0 = time.time()
    for i, r in enumerate(rows, 1):
        try:
            img = Image.open(r["path"]).convert("RGB")
        except Exception as exc:  # noqa: BLE001 - one bad file must not sink the run
            out.write(json.dumps({**r, "error": str(exc)}) + "\n")
            continue
        rec = {**r}
        for mode, prompt in prompts.items():
            msgs = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            inputs = proc.apply_chat_template(
                msgs,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)
            with torch.inference_mode():
                gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            reply = proc.decode(gen[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
            rec[f"{mode}_raw"] = reply
            rec[f"{mode}_parsed"] = parse_json_array(reply)
        out.write(json.dumps(rec) + "\n")
        out.flush()
        if i % 20 == 0:
            rate = i / (time.time() - t0)
            print(
                f"[vlm] {i}/{len(rows)}  {rate:.2f} img/s  eta {(len(rows) - i) / max(rate, 1e-6) / 60:.1f} min",
                flush=True,
            )
    out.close()
    print(f"[vlm] done in {(time.time() - t0) / 60:.1f} min -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
