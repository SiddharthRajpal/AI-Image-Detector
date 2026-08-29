"""
extract_sid.py — pull a balanced subset of SID_Set onto disk WITHOUT
downloading all 140GB.

SID_Set is stored as Parquet shards. Each row has an `image` and a `label`:
    0 = real (OpenImages photos)
    1 = fully synthetic (AI-generated)
    2 = tampered (real photo with an edited region)  <- SKIPPED

We stream the dataset (shards are fetched on the fly and we stop early once
we have enough), decode each image, and write it into the folder layout that
train.py expects:

    <out_root>/real/   <- label 0
    <out_root>/fake/   <- label 1

Run:
    huggingface-cli login          # do this first for full download speed
    python extract_sid.py --n_per_class 8000 --out_root data

Then train:
    python train.py --real_dirs data/real --fake_dirs data/fake --epochs 10
"""

import argparse
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

# label -> (folder, human name).  Label 2 (tampered) is intentionally absent.
LABEL_MAP = {0: "real", 1: "fake"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_per_class", type=int, default=8000,
                    help="how many images to save for EACH of real / fake")
    ap.add_argument("--out_root", default="data",
                    help="creates <out_root>/real and <out_root>/fake")
    ap.add_argument("--split", default="train", choices=["train", "validation"])
    ap.add_argument("--quality", type=int, default=95,
                    help="JPEG quality for saved images")
    args = ap.parse_args()

    out = Path(args.out_root)
    for name in LABEL_MAP.values():
        (out / name).mkdir(parents=True, exist_ok=True)

    print(f"Streaming SID_Set [{args.split}] — target {args.n_per_class} per class")
    print("(first shard takes a moment to start; this does NOT download all 140GB)")

    ds = load_dataset("saberzl/SID_Set", split=args.split, streaming=True)

    counts = {0: 0, 1: 0}
    target = args.n_per_class
    seen = 0
    bar = tqdm(total=target * 2, desc="saved")

    for ex in ds:
        seen += 1
        label = ex["label"]
        if label not in LABEL_MAP:          # skip tampered (2)
            continue
        if counts[label] >= target:         # this class is full
            if counts[0] >= target and counts[1] >= target:
                break
            continue

        img = ex["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")

        img_id = ex.get("img_id") or f"{LABEL_MAP[label]}_{counts[label]:06d}"
        # sanitize id for a filename
        safe = "".join(c for c in str(img_id) if c.isalnum() or c in "-_")
        path = out / LABEL_MAP[label] / f"{safe}.jpg"
        try:
            img.save(path, format="JPEG", quality=args.quality)
            counts[label] += 1
            bar.update(1)
        except Exception as e:
            print(f"  [skip] {img_id}: {e}")

        if seen % 2000 == 0:
            bar.set_postfix(real=counts[0], fake=counts[1], scanned=seen)

    bar.close()
    print(f"\nDone. real={counts[0]}  fake={counts[1]}  (scanned {seen} rows)")
    print(f"Saved to {out/'real'} and {out/'fake'}")
    print("Now run:")
    print(f"  python train.py --real_dirs {out/'real'} --fake_dirs {out/'fake'} --epochs 10")


if __name__ == "__main__":
    main()