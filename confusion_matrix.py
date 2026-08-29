"""
confusion_matrix.py — cross-source confusion matrix for the detector
====================================================================
Scores a labelled test set, applies a decision threshold, and reports the
2x2 confusion matrix (TN / FP / FN / TP) plus precision, recall, F1, and the
false-positive / false-negative rates. Also saves a labelled PNG for slides.

Convention:  POSITIVE class = AI-generated (label 1).
             Real = label 0, Fake/AI = label 1.

Recommended (honest, cross-source) run — COCO reals + SID fakes:
    python confusion_matrix.py \
        --model checkpoints/detector.pt \
        --real_dir "C:\\Users\\siddh\\fiftyone\\coco-2017\\validation\\data" \
        --fake_dir val/fake \
        --threshold 0.5 \
        --out_png confusion_matrix.png

Because the reals come from COCO (a source not used in training) and the fakes
from SID, this measures true cross-source performance, not in-distribution.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from train import IMG_EXTS
from predict import load_model


def gather(dirs, label, cap):
    out = []
    for d in dirs:
        for p in sorted(Path(d).rglob("*")):
            if p.suffix.lower() in IMG_EXTS:
                out.append((str(p), label))
    return out[:cap]


@torch.no_grad()
def score(model, device, samples, batch_size):
    preprocess = model.preprocess
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
            bt.append(preprocess(img))
            by.append(label)
        except Exception:
            continue
        if len(bt) >= batch_size:
            flush()
    flush()
    return np.array(ys), np.array(ps)


def save_png(tn, fp, fn, tp, threshold, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  [note] matplotlib not available ({e}); skipping PNG. "
              f"pip install matplotlib to enable it.")
        return

    cm = np.array([[tn, fp], [fn, tp]])
    labels = ["Real", "AI-generated"]
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=[f"Pred {l}" for l in labels])
    ax.set_yticks([0, 1], labels=[f"Actual {l}" for l in labels])
    ax.set_title(f"Confusion Matrix (threshold = {threshold})\ncross-source: COCO reals + SID fakes")
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            c = cm[i, j]
            ax.text(j, i, f"{c}\n({c/total*100:.1f}%)", ha="center", va="center",
                    color="white" if c > cm.max() / 2 else "black", fontsize=13)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints/detector.pt")
    ap.add_argument("--real_dir", nargs="+", required=True)
    ap.add_argument("--fake_dir", nargs="+", required=True)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="pred >= threshold => predicted AI-generated")
    ap.add_argument("--max_per_class", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--out_png", default="confusion_matrix.png")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, device)

    samples = (gather(args.real_dir, 0, args.max_per_class) +
               gather(args.fake_dir, 1, args.max_per_class))
    if not samples:
        raise SystemExit("No images found.")
    n_real = sum(1 for _, l in samples if l == 0)
    print(f"Scoring {n_real} real + {len(samples)-n_real} fake on {device} "
          f"(threshold {args.threshold}) ...\n")

    y, p = score(model, device, samples, args.batch_size)
    pred = (p >= args.threshold).astype(int)

    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    total = tp + tn + fp + fn

    # derived metrics (positive = AI)
    acc = (tp + tn) / total
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")   # = detection rate for fakes
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")   # reals wrongly flagged
    fnr = fn / (fn + tp) if (fn + tp) else float("nan")   # fakes missed

    print("Confusion matrix (positive = AI-generated)")
    print("                    Pred Real    Pred AI")
    print(f"  Actual Real   {tn:>10}   {fp:>8}   <- {fp} false positives")
    print(f"  Actual AI     {fn:>10}   {tp:>8}   <- {fn} false negatives")
    print()
    print(f"  Accuracy            : {acc:.4f}")
    print(f"  Precision (AI)      : {prec:.4f}   (of images flagged AI, how many really are)")
    print(f"  Recall / TPR (AI)   : {rec:.4f}   (of real fakes, how many we caught)")
    print(f"  F1 (AI)             : {f1:.4f}")
    print(f"  False-positive rate : {fpr:.4f}   (reals wrongly flagged AI)")
    print(f"  False-negative rate : {fnr:.4f}   (fakes missed)")

    save_png(tn, fp, fn, tp, args.threshold, args.out_png)


if __name__ == "__main__":
    main()