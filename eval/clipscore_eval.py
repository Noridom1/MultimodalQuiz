"""Compute CLIP-based similarity (CLIPScore) between images and texts.

Expect a JSONL `--pairs` file where each line is {"id":..., "text":..., "image_path":...}
"""
import argparse
import json
import os
from typing import List
from tqdm import tqdm
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from eval.utils import read_jsonl, ensure_dir


def load_clip(device: str = None):
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    if device:
        model.to(device)
    return model, processor


def compute_score(items: List[dict], device: str = None) -> List[dict]:
    model, proc = load_clip(device)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    out = []
    for it in tqdm(items, desc="Computing CLIPScore"):
        t = it.get("text", "")
        img_path = it.get("image_path")
        if not img_path or not os.path.exists(img_path):
            out.append({"id": it.get("id"), "score": None})
            continue
        image = Image.open(img_path).convert("RGB")
        inputs = proc(text=[t], images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds
            # normalize
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
            sim = (image_embeds * text_embeds).sum(dim=-1).cpu().item()
        out.append({"id": it.get("id"), "score": float(sim)})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", required=True, help="JSONL file with text+image_path")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default=None)
    args = p.parse_args()
    items = read_jsonl(args.pairs)
    res = compute_score(items, device=args.device)
    ensure_dir(args.out)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    avg = np.mean([r["score"] for r in res if r["score"] is not None]) if res else None
    print(f"Saved {len(res)} scores to {args.out}; avg={avg}")


if __name__ == "__main__":
    main()
