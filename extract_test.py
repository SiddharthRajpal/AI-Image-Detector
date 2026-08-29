"""
extract_test.py — pull a real/fake TEST set from any HF image dataset,
safely handling different label conventions.

THE PROBLEM THIS SOLVES: different datasets label classes oppositely.
  - Your model / SID_Set:      0 = real, 1 = fake (AI)
  - Parveshiiii/AI-vs-Real:    0 = AI,   1 = real   <-- OPPOSITE!
Getting this backwards makes a good model look broken (0% recall). So this
script auto-detects the label meaning by name, PRINTS it for you to confirm,
and always writes into OUR convention:  <out_root>/real  and  <out_root>/fake

Usage:
    # 1) INSPECT first (no --confirm) — shows schema + how it will map labels:
    python extract_test.py --dataset Parveshiiii/AI-vs-Real --split train

    # 2) If the printed mapping looks right, add --confirm to actually extract:
    python extract_test.py --dataset Parveshiiii/AI-vs-Real --split train \
                           --n_per_class 1000 --out_root test_set --confirm

    # If auto-detection can't tell, force it with --ai_label (the integer that
    # means AI-generated in THAT dataset), e.g. --ai_label 0 for AI-vs-Real.

Then test:
    python confusion_matrix.py --model checkpoints/detector.pt \
        --real_dir test_set/real --fake_dir test_set/fake --threshold 0.5
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from datasets import load_dataset
from datasets.features import ClassLabel, Image as HFImage
from tqdm import tqdm

AI_WORDS = {"ai", "fake", "synthetic", "generated", "artificial", "gan", "diffusion", "deepfake"}
REAL_WORDS = {"real", "authentic", "human", "natural", "genuine", "camera", "photo"}


def load(dataset, config, split, stream=True):
    kw = dict(split=split, streaming=stream, verification_mode="no_checks")
    if config:
        kw["name"] = config
    try:
        return load_dataset(dataset, **kw)
    except Exception:
        return load_dataset(dataset, trust_remote_code=True, **kw)


def find_columns(features, image_col, label_col):
    """Locate the image and label columns, auto-detecting when not given."""
    if image_col is None:
        for k, v in features.items():
            if isinstance(v, HFImage):
                image_col = k
                break
        if image_col is None:
            for cand in ("image", "img", "images"):
                if cand in features:
                    image_col = cand
                    break
    if label_col is None:
        for k, v in features.items():
            if isinstance(v, ClassLabel):
                label_col = k
                break
        if label_col is None:
            for cand in ("label", "labels", "label_A", "class"):
                if cand in features:
                    label_col = cand
                    break
    return image_col, label_col


def build_label_map(features, label_col, ai_label):
    """Return dict {raw_label_value -> 'real'|'fake'} and a human explanation."""
    feat = features.get(label_col)
    # Case 1: user forced which integer means AI
    if ai_label is not None:
        return None, ai_label, "forced by --ai_label"
    # Case 2: ClassLabel with names -> map by keyword
    if isinstance(feat, ClassLabel) and feat.names:
        mapping = {}
        explain = []
        for idx, name in enumerate(feat.names):
            low = name.lower()
            if any(w in low for w in AI_WORDS):
                mapping[idx] = "fake"
            elif any(w in low for w in REAL_WORDS):
                mapping[idx] = "real"
            else:
                mapping[idx] = "?"
            explain.append(f"    label {idx} = '{name}' -> {mapping[idx]}")
        return mapping, None, "\n".join(explain)
    # Case 3: plain int, no names -> we can't know
    return None, None, "UNKNOWN (plain integer labels, no names)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="HF dataset id")
    ap.add_argument("--config", default=None, help="dataset config/name if it has one")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n_per_class", type=int, default=1000)
    ap.add_argument("--out_root", default="test_set")
    ap.add_argument("--image_col", default=None)
    ap.add_argument("--label_col", default=None)
    ap.add_argument("--ai_label", type=int, default=None,
                    help="the integer label that means AI-generated in THIS dataset")
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--confirm", action="store_true",
                    help="actually extract; without it, only inspect + show mapping")
    ap.add_argument("--no_stream", action="store_true",
                    help="download the split instead of streaming (use if streaming stalls / yields nothing)")
    args = ap.parse_args()

    ds = load(args.dataset, args.config, args.split, stream=not args.no_stream)
    features = ds.features
    print(f"\nDataset: {args.dataset}  split={args.split}")
    print("Columns / features:")
    for k, v in features.items():
        print(f"    {k}: {type(v).__name__}"
              + (f"  names={v.names}" if isinstance(v, ClassLabel) else ""))

    image_col, label_col = find_columns(features, args.image_col, args.label_col)
    print(f"\nUsing image column: {image_col}")
    print(f"Using label column: {label_col}")
    if image_col is None or label_col is None:
        raise SystemExit("Could not auto-find image/label columns. "
                         "Pass --image_col and --label_col explicitly.")

    mapping, ai_label, explain = build_label_map(features, label_col, args.ai_label)
    print("\nLabel mapping (how each label becomes real/fake):")
    if ai_label is not None:
        print(f"    label {ai_label} -> fake ;  everything else -> real   ({explain})")
    else:
        print(explain)

    # show a few real samples' labels
    print("\nFirst few examples' labels:")
    for i, ex in enumerate(ds):
        print(f"    example {i}: {label_col} = {ex[label_col]}")
        if i >= 4:
            break

    def to_class(raw):
        if ai_label is not None:
            return "fake" if raw == ai_label else "real"
        return mapping.get(raw, "?")

    if not args.confirm:
        print("\n[inspect only] If the mapping above is correct, re-run with --confirm "
              "(and --n_per_class / --out_root). If a label shows '?', pass --ai_label N.")
        return

    if any(to_class(k) == "?" for k in (mapping or {ai_label: None})):
        raise SystemExit("Mapping has unknown '?' labels. Re-run with --ai_label N to force it.")

    out = Path(args.out_root)
    (out / "real").mkdir(parents=True, exist_ok=True)
    (out / "fake").mkdir(parents=True, exist_ok=True)
    counts = {"real": 0, "fake": 0}
    target = args.n_per_class
    bar = tqdm(total=target * 2, desc="saved")

    for ex in ds:
        cls = to_class(ex[label_col])
        if cls not in ("real", "fake") or counts[cls] >= target:
            if counts["real"] >= target and counts["fake"] >= target:
                break
            continue
        img = ex[image_col]
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(out / cls / f"{cls}_{counts[cls]:06d}.jpg", format="JPEG",
                 quality=args.quality)
        counts[cls] += 1
        bar.update(1)
    bar.close()

    print(f"\nDone. real={counts['real']}  fake={counts['fake']}")
    print(f"Saved to {out/'real'} and {out/'fake'}")
    print("Now run:")
    print(f"  python confusion_matrix.py --model checkpoints/detector.pt "
          f"--real_dir {out/'real'} --fake_dir {out/'fake'} --threshold 0.5")


if __name__ == "__main__":
    main()