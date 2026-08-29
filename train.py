"""
train.py — Robust AI-generated image detector
================================================
Frozen CLIP backbone (feature extractor)  +  small trainable MLP head
+  transform-matched augmentation applied in PIXEL space before CLIP.

Design rationale (maps to the hackathon judging criteria):
  * Frozen CLIP probe  -> generalizes across unseen generators (DALL-E, diffusion,
    GANs) far better than a CNN trained from scratch, and stays well under the
    2B-parameter limit while training in minutes on a local GPU.
  * Augmentation that mirrors the benchmark transforms (JPEG / blur / resize /
    noise / color jitter / crop) -> the single biggest robustness lever. We teach
    invariance to exactly the corruptions the hidden benchmark will apply.
  * The SAME augmentation is applied to BOTH classes, so JPEG-quality / resolution
    cannot become a shortcut the model exploits instead of real AIGC cues.

Expected data layout (any of these work — point --real_dirs / --fake_dirs at them):
    data/train/real/*        <- authentic images        (label 0)
    data/train/fake/*        <- AI-generated images      (label 1)
  You can pass MULTIPLE directories to each flag to mix datasets (CIFAKE, SID_Set,
  WildFake). *** Do NOT include COCO val2017 or DALL-E Advanced — that is the
  forbidden benchmark set. ***

Run:
    pip install -r requirements.txt
    python train.py --real_dirs data/train/real --fake_dirs data/train/fake \
                    --backbone ViT-B-16 --pretrained openai --epochs 8

Outputs:
    checkpoints/detector.pt   <- head weights + config, loaded by the inference script
"""

import argparse
import io
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ColorJitter
import open_clip
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# --------------------------------------------------------------------------- #
# 1.  Robustness augmentation — the transform family from the problem statement
# --------------------------------------------------------------------------- #
class RobustnessAugment:
    """Applies the benchmark transform family in pixel space, on a PIL image,
    BEFORE CLIP preprocessing. Applied to real AND fake images identically so
    that compression/resolution carry no class signal."""

    def __init__(self, strength=0.9):
        self.p = strength
        self.jitter = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)

    def __call__(self, img: Image.Image) -> Image.Image:
        # --- color jitter (filter apps, auto-enhance) ---
        if random.random() < 0.5:
            img = self.jitter(img)

        # --- center/random crop (profile-pic cropping, framing) ---
        if random.random() < 0.4:
            w, h = img.size
            cw, ch = int(w * 0.8), int(h * 0.8)
            left = random.randint(0, w - cw)
            top = random.randint(0, h - ch)
            img = img.crop((left, top, left + cw, top + ch))

        # --- resize down then up (thumbnail generation) ---
        if random.random() < 0.5:
            scale = random.choice([0.25, 0.5, 0.75])
            w, h = img.size
            small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                               Image.BILINEAR)
            img = small.resize((w, h), Image.BILINEAR)

        # --- gaussian blur (out of focus) ---
        if random.random() < 0.5:
            sigma = random.uniform(0.0, 2.0)
            if sigma > 0.05:
                img = img.filter(ImageFilter.GaussianBlur(radius=sigma))

        # --- gaussian noise (low-light sensor) — done in [0,1] numpy space ---
        if random.random() < 0.5:
            sigma = random.choice([0.02, 0.05, 0.10])
            arr = np.asarray(img, dtype=np.float32) / 255.0
            arr = arr + np.random.randn(*arr.shape).astype(np.float32) * sigma
            arr = np.clip(arr, 0.0, 1.0)
            img = Image.fromarray((arr * 255).astype(np.uint8))

        # --- JPEG recompression (social re-encode) — applied last & often ---
        if random.random() < self.p:
            q = random.randint(30, 95)
            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="JPEG", quality=q)
            buf.seek(0)
            img = Image.open(buf).convert("RGB")

        return img.convert("RGB")


# --------------------------------------------------------------------------- #
# 2.  Dataset
# --------------------------------------------------------------------------- #
class ImageDataset(Dataset):
    def __init__(self, samples, preprocess, augment=None):
        self.samples = samples            # list of (path, label)
        self.preprocess = preprocess      # CLIP resize+centercrop+normalize
        self.augment = augment            # RobustnessAugment or None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            # corrupt file -> return a black image, it will be near-uninformative
            img = Image.new("RGB", (224, 224))
        if self.augment is not None:
            img = self.augment(img)
        x = self.preprocess(img)
        return x, torch.tensor(label, dtype=torch.float32)


def gather(dirs, label):
    samples = []
    for d in dirs:
        d = Path(d)
        if not d.exists():
            print(f"  [warn] directory not found: {d}")
            continue
        for p in d.rglob("*"):
            if p.suffix.lower() in IMG_EXTS:
                samples.append((str(p), label))
    return samples


# --------------------------------------------------------------------------- #
# 3.  Model — frozen CLIP + MLP head
# --------------------------------------------------------------------------- #
class CLIPDetector(nn.Module):
    def __init__(self, backbone="ViT-B-16", pretrained="openai",
                 head_hidden=512, dropout=0.5):
        super().__init__()
        self.clip, _, self.preprocess = open_clip.create_model_and_transforms(
            backbone, pretrained=pretrained)
        for p in self.clip.parameters():
            p.requires_grad = False
        self.clip.eval()
        feat_dim = self.clip.visual.output_dim
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )
        self.backbone_name = backbone
        self.pretrained = pretrained

    @torch.no_grad()
    def encode(self, x):
        feats = self.clip.encode_image(x)
        return feats.float()

    def forward(self, x):
        feats = self.encode(x)          # frozen backbone, no grad
        return self.head(feats).squeeze(-1)   # logits


# --------------------------------------------------------------------------- #
# 4.  Eval helper
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(x)
        probs = torch.sigmoid(logits).float().cpu().numpy()
        ps.append(probs)
        ys.append(y.numpy())
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    acc = accuracy_score(y, (p > 0.5).astype(int))
    try:
        auc = roc_auc_score(y, p)
    except ValueError:
        auc = float("nan")
    return acc, auc


# --------------------------------------------------------------------------- #
# 5.  Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_dirs", nargs="+", required=True,
                    help="one or more directories of authentic images (label 0)")
    ap.add_argument("--fake_dirs", nargs="+", required=True,
                    help="one or more directories of AI-generated images (label 1)")
    ap.add_argument("--backbone", default="ViT-B-16",
                    help="open_clip backbone, e.g. ViT-B-16 or ViT-L-14")
    ap.add_argument("--pretrained", default="openai",
                    help="e.g. openai, laion2b_s34b_b88k (B-16), laion2b_s32b_b82k (L-14)")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--aug_strength", type=float, default=0.9)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="checkpoints/detector.pt")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- model (built first so we can reuse its CLIP preprocess) ----
    model = CLIPDetector(args.backbone, args.pretrained).to(device)
    preprocess = model.preprocess
    print(f"Backbone: {args.backbone}/{args.pretrained} "
          f"({sum(p.numel() for p in model.clip.parameters())/1e6:.0f}M frozen params)")

    # ---- data ----
    real = gather(args.real_dirs, 0)
    fake = gather(args.fake_dirs, 1)
    print(f"Found {len(real)} real  +  {len(fake)} fake  = {len(real)+len(fake)} images")
    if not real or not fake:
        raise SystemExit("Need images in BOTH real and fake dirs.")

    samples = real + fake
    random.shuffle(samples)
    n_val = int(len(samples) * args.val_frac)
    val_samples, train_samples = samples[:n_val], samples[n_val:]

    aug = RobustnessAugment(strength=args.aug_strength)
    train_ds = ImageDataset(train_samples, preprocess, augment=aug)
    val_clean = ImageDataset(val_samples, preprocess, augment=None)   # clean metric
    val_robust = ImageDataset(val_samples, preprocess, augment=aug)   # robust metric

    dl_kw = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                 pin_memory=(device.type == "cuda"))
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **dl_kw)
    clean_loader = DataLoader(val_clean, shuffle=False, **dl_kw)
    robust_loader = DataLoader(val_robust, shuffle=False, **dl_kw)

    # ---- class balance for the loss ----
    n_pos = sum(l for _, l in train_samples)
    n_neg = len(train_samples) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optim = torch.optim.AdamW(model.head.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_auc = -1.0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.head.train()
        running = 0.0
        for x, y in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}"):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            running += loss.item() * x.size(0)
        sched.step()

        clean_acc, clean_auc = evaluate(model, clean_loader, device)
        rob_acc, rob_auc = evaluate(model, robust_loader, device)
        print(f"  loss {running/len(train_ds):.4f} | "
              f"clean acc {clean_acc:.3f} auc {clean_auc:.3f} | "
              f"robust acc {rob_acc:.3f} auc {rob_auc:.3f}")

        # select on ROBUST auc — that's what the benchmark rewards
        if rob_auc > best_auc:
            best_auc = rob_auc
            torch.save({
                "head": model.head.state_dict(),
                "backbone": args.backbone,
                "pretrained": args.pretrained,
                "head_hidden": 512,
                "robust_auc": rob_auc,
                "clean_auc": clean_auc,
            }, args.out)
            print(f"  ** saved {args.out} (robust auc {rob_auc:.3f})")

    print(f"\nDone. Best robust AUROC = {best_auc:.3f} -> {args.out}")


if __name__ == "__main__":
    main()
