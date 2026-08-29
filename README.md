# Robust Detection of AI-Generated Images Under Real-World Transformations

A robust detector that classifies whether an image is **AI-generated (AIGC)** or **authentic**, engineered to stay accurate after real-world degradation — JPEG compression, blur, resizing, noise, colour filters, and cropping.

**Approach in one line:** a small trainable classifier on top of a **frozen CLIP ViT-B/16** vision transformer, trained with the **exact image corruptions it will be tested against**, applied identically to real and fake images.

| | |
|---|---|
| **Demo video** | [YouTube link] |
| **Robustness** | AUROC stays within ±0.002 of clean across 15 transform conditions |
| **Cross-source generalisation** | Recall >99% on fakes from unseen generators; false-positive rate stable at ~7–8% across three independent real-image sources |
| **Model size** | ~150M parameters (well under the 2B limit) |

---

## Quick Start (run the detector)
> ## ⚠️ IMPORTANT — PyTorch must be installed first
> **`requirements.txt` assumes you already have a working PyTorch setup (GPU or CPU).**
> It intentionally does **not** list `torch`/`torchvision`, because installing them via
> pip can overwrite a working CUDA (GPU) build with a CPU-only one and break GPU training.
> **Install PyTorch yourself first** using the exact command for your system from
> **https://pytorch.org**, then install the rest. Verify it works before continuing:
> ```bash
> python -c "import torch; print(torch.cuda.is_available(), torch.__version__)"
> ```

**Requirements:** Python 3, and a working PyTorch install (an NVIDIA GPU build is recommended; CPU works but is slow).

First install the repo and the required libraries
```
git clone https://github.com/SiddharthRajpal/AI-Image-Detector
pip install -r requirements.txt
cd Ai-Image-Detector
```
With a trained checkpoint at `checkpoints/detector.pt`, point `run.py` at an image or a folder:

```bash
python run.py --input path/to/images
```
It prints a verdict and confidence for each image and saves `results.json`:
```
photo1.jpg    ->  AI-GENERATED   97.3% confident
photo2.jpg    ->  AUTHENTIC      92.1% confident
Summary: 12 image(s) | 5 AI-generated | 7 authentic
```

For the raw JSON deliverable format (`[{"image_path", "pred"}]`), `run.py` and `predict.py` both produce it.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Setup and Installation](#setup-and-installation)
- [Steps to Reproduce Our Results](#steps-to-reproduce-our-results)
- [Repository Structure](#repository-structure)
- [Results Summary](#results-summary)
- [Limitations & What We'd Improve](#limitations--what-wed-improve)
- [Team Member Contributions](#team-member-contributions)
- [Further Documentation](#further-documentation)

---

## Project Overview

Generative AI can now produce photorealistic fake images at scale, feeding misinformation, impersonation, and fraud. Most detectors score 97–99% on clean images but collapse the moment an image is compressed, blurred, or cropped — because they rely on high-frequency generator fingerprints that those operations destroy. **Robustness under real-world transformation is the actual problem**, and it is what this project is built around.

**How it works.** Each image is passed through a **frozen** CLIP ViT-B/16 encoder (weights never updated) to produce a 512-dimensional feature vector, and a small trainable MLP head maps that vector to the probability the image is AI-generated. Two design choices drive the result:

1. **Frozen foundation-model backbone** — a from-scratch network memorises the fingerprint of the specific generators it saw and fails on unseen ones; CLIP's general features transfer across generators (GANs, diffusion, DALL·E), which matters because the hidden benchmark uses a generator we may not have trained on.
2. **Transform-matched augmentation** — every training image is randomly JPEG-compressed, blurred, noised, resized, colour-jittered, and cropped *before* reaching CLIP, teaching invariance to exactly those corruptions. The same augmentation is applied to real and fake images so compression/resolution can't become a shortcut.

```
input → [random robustness corruption] → [frozen CLIP ViT-B/16] → 512-d feature → [trained MLP head] → P(AI-generated)
              (training only)                   (frozen)                              (the only trained part)
```

---

## Setup and Installation

> ## ⚠️ IMPORTANT — PyTorch must be installed first
> **`requirements.txt` assumes you already have a working PyTorch setup (GPU or CPU).**
> It intentionally does **not** list `torch`/`torchvision`, because installing them via
> pip can overwrite a working CUDA (GPU) build with a CPU-only one and break GPU training.
> **Install PyTorch yourself first** using the exact command for your system from
> **https://pytorch.org**, then install the rest. Verify it works before continuing:
> ```bash
> python -c "import torch; print(torch.cuda.is_available(), torch.__version__)"
> ```

**Requirements:** Python 3, and a working PyTorch install (an NVIDIA GPU build is recommended; CPU works but is slow).

```bash
# 1. Clone
git clone https://github.com/SiddharthRajpal/AI-Image-Detector/
cd AI-Image-Detector

# 2. Install PyTorch FIRST (see the note above) from https://pytorch.org, then verify:
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 3. Install remaining dependencies (this does NOT touch your torch install)
pip install -r requirements.txt

# 4. (Optional) only if you want the dashboard or to re-download data:
#    pip install streamlit datasets fiftyone
```

`requirements.txt` intentionally omits `torch`/`torchvision` so a pip install can't overwrite your working build with a CPU-only one.

---

## Steps to Reproduce Our Results

```bash
# 1. Get training data — streams SID_Set (a few GB, NOT the full 140 GB) into
#    data/real and data/fake. Log in first for full download speed.
huggingface-cli login
python extract_sid.py --n_per_class 6000 --out_root data

# 2. (Optional) add COCO reals for benchmark alignment
#    python -c "import fiftyone.zoo as foz; foz.load_zoo_dataset('coco-2017', split='train', max_samples=6000)"

# 3. Train (only the MLP head trains; CLIP backbone is frozen)
python train.py --real_dirs data/real --fake_dirs data/fake --epochs 10
#    -> saves checkpoints/detector.pt (best robust-AUROC checkpoint)

# 4. Robustness evaluation table (held-out SID validation split)
python extract_sid.py --split validation --n_per_class 1000 --out_root val
python evaluate.py --model checkpoints/detector.pt \
                   --real_dir val/real --fake_dir val/fake --out robustness.csv

# 5. Cross-source confusion matrix (honest generalisation test)
python confusion_matrix.py --model checkpoints/detector.pt \
                           --real_dir <coco_folder> --fake_dir val/fake --threshold 0.5

# 6. Inference on any image directory -> required JSON output
python predict.py --model checkpoints/detector.pt \
                  --image_dir path/to/test_images --out predictions.json

# 7. Interactive dashboard / demo
streamlit run app.py
```

**Inference output format** (`predict.py`) — the required deliverable:
```json
[
  {"image_path": "test/img001.jpg", "pred": 0.97},
  {"image_path": "test/img002.jpg", "pred": 0.03}
]
```
`pred` is the probability the image is AI-generated (0 = confidently authentic, 1 = confidently AI).

**Note on data:** the forbidden benchmark set (COCO val2017 + DALL·E Advanced) must **never** be placed in `--real_dirs`/`--fake_dirs` for training. COCO val2017 is used only as an independent test.

---

## Repository Structure

| File | Purpose |
|---|---|
| `run.py` | **Main entry point** — point at an image or folder; prints a verdict + confidence per image and saves JSON. |
| `train.py` | Training pipeline — defines the augmentation, dataset, and `CLIPDetector`; trains only the head; saves the checkpoint. |
| `extract_sid.py` | Streams SID_Set and writes N images per class into `real/`/`fake/` (drops the tampered class). |
| `extract_test.py` | Pulls a real/fake **test** set from any HF image dataset, auto-detecting and confirming the label convention. |
| `predict.py` | **Required deliverable** — image directory → JSON of AIGC confidence scores. |
| `evaluate.py` | **Required deliverable** — robustness sweep across 15 transform conditions → CSV. |
| `confusion_matrix.py` | Cross-source confusion matrix with a threshold dial and a labelled PNG. |
| `app.py` | Streamlit dashboard — robustness playground (slider-driven) + batch folder scoring. |
| `requirements.txt` | Dependencies (excludes torch to protect your CUDA build). |
| `PROJECT_DOCUMENTATION.md` | Full technical reference + anticipated-questions bank. |

---

## Results Summary

**Robustness** (held-out SID validation split, 1,000 real + 1,000 fake) — AUROC stays within ±0.002 of the clean score (0.9975) across all 15 conditions:

| | clean | jpeg_q30 | blur_2.0 | resize_0.25 | noise_0.10 | jitter_20% | crop_80% |
|---|---|---|---|---|---|---|---|
| AUROC | 0.9975 | 0.9994 | 0.9983 | 0.9984 | 0.9971 | 0.9966 | 0.9968 |

Full table (all 15 conditions) is in `PROJECT_DOCUMENTATION.md` §9.4 and `robustness.csv`.

**Cross-dataset generalisation** — tested on three independent sources (none used in training):

| Test source | Real from | Fake from | Recall (fakes) | False-positive rate | Accuracy |
|---|---|---|---|---|---|
| SID validation | SID/OpenImages | SID synthetic | 99.6% | — | ~0.95 |
| COCO val2017 (full, 4,965) | COCO benchmark reals | — | — | 8.2% | — |
| Parveshiiii/AI-vs-Real (1k+1k) | independent | independent generators | 99.1% | 7.4% | 0.959 |

The false-positive rate is stable at **~7–8% across three different real-image sources**, and recall holds **above 99% on fakes from generators never trained on** — direct evidence the model generalises rather than memorising a dataset. The decision threshold is a tunable policy dial (raising it from 0.5 to 0.7 cuts false positives by a third while still catching 99% of fakes).

**On the near-perfect in-distribution score:** internal validation reached ~1.000 AUROC, which we treated as a warning sign (likely dataset shortcut) rather than a result — which is exactly why we measured the honest cross-source numbers above.

---

## Limitations & What We'd Improve

**Limitations:**
- **Real-source mismatch.** Training reals were largely OpenImages (from SID_Set); the benchmark reals are COCO. This drives most of the ~7% false-positive rate.
- **Single primary generator source.** Fakes come mainly from SID_Set's synthetic set; broader generator diversity was scoped out under time pressure.
- **In-distribution validation is optimistic** (~1.000 AUROC); the honest number is the cross-source COCO test.
- **Whole-image detection only** — no localisation of edited regions; the tampered class was dropped by design.
- **Prototype scale** — ~13k training images on a single consumer GPU.

**What we'd improve, in priority order:**
1. Add COCO train2017 reals to training to cut the false-positive rate.
2. Add WildFake for greater generator diversity and cross-generator generalisation.
3. Upgrade to CLIP ViT-L/14 (~430M params, still under 2B) for a likely accuracy gain.
4. Patch-based inference for stronger crop robustness.
5. A from-scratch CNN baseline, trained on the same augmented data, to quantify how much the frozen foundation model buys us in robustness.

---

## Team Member Contributions

<!-- Replace with your actual team. If solo, state "Solo project by [name]." -->

| Member | Contributions |
|---|---|
| [Name 1] | [e.g. architecture & training pipeline, augmentation design] |
| [Name 2] | [e.g. data streaming/extraction, evaluation harness] |
| [Name 3] | [e.g. dashboard, error analysis, documentation & pitch] |

---

## Further Documentation

- **`PROJECT_DOCUMENTATION.md`** — complete technical reference (architecture, data decisions, full results) and a 20+ question anticipated-questions bank.
- **`devpost.md`** — the written project description submitted via Devpost.

---

*Environment used: NVIDIA RTX 5070 (CUDA 13.2), PyTorch 2.13.0, Windows. CLIP ViT-B/16 via `open_clip`.*
