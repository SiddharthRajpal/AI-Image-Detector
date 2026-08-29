"""
evaluate.py — REQUIRED DELIVERABLE (Robustness Evaluation Summary)
==================================================================
Scores a LABELLED test set (real/ and fake/ folders) under clean conditions
and under each benchmark transform at each severity, then prints a
clean-vs-transformed comparison table and saves it to CSV.

This is the table that shows whether the detector actually survives real-world
post-processing — the core claim of the project.

Run:
    python evaluate.py --model checkpoints/detector.pt \
                       --real_dir data/val_real --fake_dir data/val_fake \
                       --out robustness.csv

Tip: point --real_dir / --fake_dir at a HELD-OUT slice you did NOT train on.
"""

import argparse
import csv
import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from torchvision.transforms import functional as TF
import torch
from sklearn.metrics import roc_auc_score, accuracy_score

from train import IMG_EXTS
from predict import load_model


# --------------------------------------------------------------------------- #
# Deterministic transforms matching the benchmark table exactly
# --------------------------------------------------------------------------- #
def t_identity(img):
    return img


def t_jpeg(q):
    def f(img):
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=q)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    return f


def t_blur(sigma):
    def f(img):
        return img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return f


def t_resize(scale):
    def f(img):
        w, h = img.size
        small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                           Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)
    return f


def t_noise(sigma):
    def f(img):
        arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        arr = np.clip(arr + np.random.randn(*arr.shape).astype(np.float32) * sigma,
                      0.0, 1.0)
        return Image.fromarray((arr * 255).astype(np.uint8))
    return f


def t_jitter(img):
    img = TF.adjust_brightness(img, 1.2)
    img = TF.adjust_contrast(img, 1.2)
    img = TF.adjust_saturation(img, 1.2)
    return img


def t_crop(frac):
    def f(img):
        w, h = img.size
        cw, ch = int(w * frac), int(h * frac)
        left, top = (w - cw) // 2, (h - ch) // 2
        return img.crop((left, top, left + cw, top + ch))
    return f


# Ordered so the table reads top-to-bottom clean -> harsher
CONDITIONS = [
    ("clean",       t_identity),
    ("jpeg_q90",    t_jpeg(90)),
    ("jpeg_q70",    t_jpeg(70)),
    ("jpeg_q50",    t_jpeg(50)),
    ("jpeg_q30",    t_jpeg(30)),
    ("blur_0.5",    t_blur(0.5)),
    ("blur_1.0",    t_blur(1.0)),
    ("blur_2.0",    t_blur(2.0)),
    ("resize_0.5",  t_resize(0.5)),
    ("resize_0.25", t_resize(0.25)),
    ("noise_0.02",  t_noise(0.02)),
    ("noise_0.05",  t_noise(0.05)),
    ("noise_0.10",  t_noise(0.10)),
    ("jitter_20%",  t_jitter),
    ("crop_80%",    t_crop(0.8)),
]


def gather(dirs, label, cap):
    samples = []
    for d in dirs:
        for p in sorted(Path(d).rglob("*")):
            if p.suffix.lower() in IMG_EXTS:
                samples.append((str(p), label))
    return samples[:cap]


@torch.no_grad()
def score_condition(model, preprocess, samples, transform, device, batch_size):
    """Apply `transform` to every image, return (accuracy, auroc)."""
    ys, ps = [], []
    bt, by = [], []

    def flush():
        if not bt:
            return
        x = torch.stack(bt).to(device)
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(x)
        probs = torch.sigmoid(logits).float().cpu().numpy().reshape(-1)
        ps.extend(probs.tolist())
        ys.extend(by)
        bt.clear()
        by.clear()

    for path, label in samples:
        try:
            img = Image.open(path).convert("RGB")
            img = transform(img)
            bt.append(preprocess(img))
            by.append(label)
        except Exception:
            continue
        if len(bt) >= batch_size:
            flush()
    flush()

    y = np.array(ys)
    p = np.array(ps)
    acc = accuracy_score(y, (p > 0.5).astype(int))
    try:
        auc = roc_auc_score(y, p)
    except ValueError:
        auc = float("nan")
    return acc, auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints/detector.pt")
    ap.add_argument("--real_dir", nargs="+", required=True)
    ap.add_argument("--fake_dir", nargs="+", required=True)
    ap.add_argument("--out", default="robustness.csv")
    ap.add_argument("--max_per_class", type=int, default=1000,
                    help="cap images per class to keep the sweep fast")
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, device)
    preprocess = model.preprocess

    samples = (gather(args.real_dir, 0, args.max_per_class) +
               gather(args.fake_dir, 1, args.max_per_class))
    if not samples:
        raise SystemExit("No eval images found.")
    n_real = sum(1 for _, l in samples if l == 0)
    print(f"Evaluating on {n_real} real + {len(samples)-n_real} fake images, "
          f"{len(CONDITIONS)} conditions, on {device}\n")

    rows = []
    clean_auc = None
    header = f"{'condition':<14}{'accuracy':>10}{'AUROC':>9}{'ΔAUROC':>9}"
    print(header)
    print("-" * len(header))
    for name, fn in CONDITIONS:
        acc, auc = score_condition(model, preprocess, samples, fn, device,
                                   args.batch_size)
        if name == "clean":
            clean_auc = auc
        delta = "" if name == "clean" else f"{auc - clean_auc:+.3f}"
        print(f"{name:<14}{acc:>10.3f}{auc:>9.3f}{delta:>9}")
        rows.append({"condition": name, "accuracy": round(acc, 4),
                     "auroc": round(auc, 4),
                     "delta_auroc": "" if name == "clean" else round(auc - clean_auc, 4)})

    # robustness summary: average over the transformed (non-clean) conditions
    trans = [r for r in rows if r["condition"] != "clean"]
    mean_acc = np.mean([r["accuracy"] for r in trans])
    mean_auc = np.mean([r["auroc"] for r in trans])
    print("-" * len(header))
    print(f"{'MEAN(transf)':<14}{mean_acc:>10.3f}{mean_auc:>9.3f}"
          f"{mean_auc - clean_auc:>+9.3f}")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["condition", "accuracy", "auroc", "delta_auroc"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved table to {args.out}")


if __name__ == "__main__":
    main()
