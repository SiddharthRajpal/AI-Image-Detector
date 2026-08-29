"""
run.py — the main entry point for the AI-image detector.
=========================================================
Point it at an image (or a folder of images) and it will, for each image,
print a readable verdict and confidence, and also save a JSON results file.

Examples:
    python run.py --input path/to/test_images
    python run.py --input one_photo.jpg
    python run.py --input path/to/images --threshold 0.7 --out results.json

Output (printed):
    photo1.jpg    ->  AI-GENERATED   97.3% confident
    photo2.jpg    ->  AUTHENTIC      92.1% confident
    ...
    Summary: 12 images | 5 AI-generated | 7 authentic

Output (saved JSON, the required deliverable format):
    [ {"image_path": "...", "pred": 0.973}, ... ]
    where `pred` = probability the image is AI-generated (0..1).
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from train import IMG_EXTS
from predict import load_model, list_images


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description="Detect AI-generated images.")
    ap.add_argument("--input", required=True,
                    help="an image file OR a directory of images (searched recursively)")
    ap.add_argument("--model", default="checkpoints/detector.pt")
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="pred >= threshold => flagged AI-generated")
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    # resolve input to a list of image paths
    inp = Path(args.input)
    if inp.is_file():
        paths = [inp]
    elif inp.is_dir():
        paths = list_images(inp)
    else:
        raise SystemExit(f"Input not found: {inp}")
    if not paths:
        raise SystemExit(f"No images found at {inp}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading detector on {device} ...")
    model = load_model(args.model, device)
    preprocess = model.preprocess
    print(f"Scoring {len(paths)} image(s)  (threshold {args.threshold})\n")

    results = []
    batch_tensors, batch_paths = [], []

    def flush():
        if not batch_tensors:
            return
        x = torch.stack(batch_tensors).to(device)
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(x)
        probs = torch.sigmoid(logits).float().cpu().tolist()
        if isinstance(probs, float):
            probs = [probs]
        for p, prob in zip(batch_paths, probs):
            is_ai = prob >= args.threshold
            verdict = "AI-GENERATED" if is_ai else "AUTHENTIC   "
            conf = prob if is_ai else (1 - prob)
            name = Path(p).name
            print(f"  {name[:40]:<40} ->  {verdict}  {conf*100:5.1f}% confident")
            results.append({"image_path": str(p), "pred": round(float(prob), 6)})
        batch_tensors.clear()
        batch_paths.clear()

    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
        except Exception as e:
            print(f"  {Path(p).name[:40]:<40} ->  [unreadable: {e}]")
            results.append({"image_path": str(p), "pred": 0.5})
            continue
        batch_tensors.append(preprocess(img))
        batch_paths.append(p)
        if len(batch_tensors) >= args.batch_size:
            flush()
    flush()

    n_ai = sum(1 for r in results if r["pred"] >= args.threshold)
    print(f"\nSummary: {len(results)} image(s) | {n_ai} AI-generated | "
          f"{len(results) - n_ai} authentic")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved JSON results to {args.out}")


if __name__ == "__main__":
    main()