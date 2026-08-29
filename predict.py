"""
predict.py — REQUIRED DELIVERABLE
=================================
Takes an image directory and writes a JSON file with, for every image, its
path and `pred` = the model's confidence (0..1) that the image is AI-generated.

Output format (exactly what the submission asks for):
    [
      {"image_path": "test/img001.jpg", "pred": 0.97},
      {"image_path": "test/img002.jpg", "pred": 0.03},
      ...
    ]

Run:
    python predict.py --model checkpoints/detector.pt \
                      --image_dir path/to/test_images \
                      --out predictions.json

The checkpoint stores which CLIP backbone was used, so you don't pass it here.
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from train import CLIPDetector, IMG_EXTS   # reuse the exact model definition


def load_model(ckpt_path, device):
    """Rebuild the frozen-CLIP + head model and load trained head weights."""
    ck = torch.load(ckpt_path, map_location=device)
    model = CLIPDetector(
        backbone=ck["backbone"],
        pretrained=ck["pretrained"],
        head_hidden=ck.get("head_hidden", 512),
    ).to(device)
    model.head.load_state_dict(ck["head"])
    model.eval()
    return model


def list_images(image_dir):
    paths = []
    for p in sorted(Path(image_dir).rglob("*")):
        if p.suffix.lower() in IMG_EXTS:
            paths.append(p)
    return paths


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints/detector.pt")
    ap.add_argument("--image_dir", required=True,
                    help="directory of images to score (searched recursively)")
    ap.add_argument("--out", default="predictions.json")
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, device)
    preprocess = model.preprocess

    paths = list_images(args.image_dir)
    if not paths:
        raise SystemExit(f"No images found under {args.image_dir}")
    print(f"Scoring {len(paths)} images on {device} ...")

    results = []
    batch_tensors, batch_paths = [], []

    def flush():
        if not batch_tensors:
            return
        x = torch.stack(batch_tensors).to(device)
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(x)
        probs = torch.sigmoid(logits).float().cpu().tolist()
        if isinstance(probs, float):        # single-item batch
            probs = [probs]
        for p, prob in zip(batch_paths, probs):
            results.append({"image_path": str(p), "pred": round(float(prob), 6)})
        batch_tensors.clear()
        batch_paths.clear()

    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
        except Exception as e:
            print(f"  [warn] could not read {p}: {e} -> pred=0.5")
            results.append({"image_path": str(p), "pred": 0.5})
            continue
        batch_tensors.append(preprocess(img))
        batch_paths.append(p)
        if len(batch_tensors) >= args.batch_size:
            flush()
    flush()

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    n_ai = sum(1 for r in results if r["pred"] > 0.5)
    print(f"Wrote {len(results)} predictions to {args.out}")
    print(f"  flagged AI-generated (pred>0.5): {n_ai}   authentic: {len(results)-n_ai}")


if __name__ == "__main__":
    main()
