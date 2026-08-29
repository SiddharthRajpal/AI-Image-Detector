
import io
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from torchvision.transforms import functional as TF
import streamlit as st

from train import CLIPDetector, IMG_EXTS   # noqa: F401  (CLIPDetector used via load_model)
from predict import load_model, list_images

st.set_page_config(page_title="AI Image Detector", layout="wide")


# --------------------------------------------------------------------------- #
# Model (cached so it loads once)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading detector…")
def get_model(ckpt_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(ckpt_path, device)
    return model, device


def predict_prob(model, device, img):
    x = model.preprocess(img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logit = model(x)
    return float(torch.sigmoid(logit).item())


# --------------------------------------------------------------------------- #
# Benchmark transforms (deterministic)
# --------------------------------------------------------------------------- #
def apply_pipeline(img, jpeg_q, blur_s, noise_s, resize_scale, jitter_pct, crop_frac):
    img = img.convert("RGB")
    if jitter_pct > 0:
        f = 1 + jitter_pct / 100.0
        img = TF.adjust_brightness(img, f)
        img = TF.adjust_contrast(img, f)
        img = TF.adjust_saturation(img, f)
    if crop_frac < 1.0:
        w, h = img.size
        cw, ch = int(w * crop_frac), int(h * crop_frac)
        left, top = (w - cw) // 2, (h - ch) // 2
        img = img.crop((left, top, left + cw, top + ch))
    if resize_scale < 1.0:
        w, h = img.size
        small = img.resize((max(1, int(w * resize_scale)), max(1, int(h * resize_scale))),
                           Image.BILINEAR)
        img = small.resize((w, h), Image.BILINEAR)
    if blur_s > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_s))
    if noise_s > 0:
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = np.clip(arr + np.random.randn(*arr.shape).astype(np.float32) * noise_s, 0, 1)
        img = Image.fromarray((arr * 255).astype(np.uint8))
    if jpeg_q < 100:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=jpeg_q)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
    return img


def verdict(prob, threshold):
    if prob >= threshold:
        st.error(f"### 🤖 AI-generated  ·  {prob*100:.1f}% confidence")
    else:
        st.success(f"### 📷 Authentic  ·  {(1-prob)*100:.1f}% confidence")
    st.progress(prob, text=f"P(AI-generated) = {prob:.3f}")


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
st.sidebar.title("⚙️ Settings")
ckpt = st.sidebar.text_input("Checkpoint path", "checkpoints/detector.pt")
threshold = st.sidebar.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01,
                              help="pred ≥ threshold → flagged AI-generated")

try:
    model, device = get_model(ckpt)
    st.sidebar.success(f"Model loaded on {device}")
except Exception as e:
    st.sidebar.error(f"Could not load model:\n{e}")
    st.stop()


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
tab1, tab2 = st.tabs(["🔍 Single image + robustness", "📁 Batch folder"])

with tab1:
    st.header("Test one image")
    up = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp", "bmp"])
    if up:
        original = Image.open(up).convert("RGB")
        st.subheader("Robustness playground")
        st.caption("Drag the sliders to simulate real-world degradation. "
                   "A robust detector's verdict should barely move.")

        c = st.columns(6)
        jpeg_q = c[0].slider("JPEG quality", 10, 100, 100, 5)
        blur_s = c[1].slider("Blur σ", 0.0, 3.0, 0.0, 0.1)
        noise_s = c[2].slider("Noise σ", 0.0, 0.15, 0.0, 0.01)
        resize_scale = c[3].slider("Resize ×", 0.1, 1.0, 1.0, 0.05)
        jitter_pct = c[4].slider("Color jitter %", 0, 40, 0, 5)
        crop_frac = c[5].slider("Center crop", 0.4, 1.0, 1.0, 0.05)

        transformed = apply_pipeline(original, jpeg_q, blur_s, noise_s,
                                     resize_scale, jitter_pct, crop_frac)

        p_orig = predict_prob(model, device, original)
        p_trans = predict_prob(model, device, transformed)

        left, right = st.columns(2)
        with left:
            st.markdown("**Original**")
            st.image(original, use_container_width=True)
            verdict(p_orig, threshold)
        with right:
            st.markdown("**After transforms**")
            st.image(transformed, use_container_width=True)
            verdict(p_trans, threshold)

        drift = abs(p_trans - p_orig)
        st.metric("Score drift under transforms", f"{drift:.3f}",
                  help="How much P(AI) moved. Small = robust.")
        if drift < 0.1:
            st.success("✅ Verdict is stable under these transforms — robust.")
        elif drift < 0.25:
            st.warning("⚠️ Moderate drift — holding up but sensitive here.")
        else:
            st.error("❌ Large drift — this transform is a weak point.")

with tab2:
    st.header("Score a folder")
    folder = st.text_input("Folder path (searched recursively)")
    max_n = st.number_input("Max images", 10, 100000, 500, 10)
    if st.button("Run") and folder:
        paths = list_images(folder)[: int(max_n)]
        if not paths:
            st.warning("No images found there.")
        else:
            prog = st.progress(0.0, text="Scoring…")
            rows = []
            for i, p in enumerate(paths):
                try:
                    img = Image.open(p).convert("RGB")
                    prob = predict_prob(model, device, img)
                    rows.append({"image_path": str(p), "pred": round(prob, 4),
                                 "verdict": "AI" if prob >= threshold else "Real"})
                except Exception:
                    pass
                if i % 5 == 0:
                    prog.progress((i + 1) / len(paths), text=f"Scoring… {i+1}/{len(paths)}")
            prog.empty()

            preds = np.array([r["pred"] for r in rows])
            n_ai = int((preds >= threshold).sum())
            m1, m2, m3 = st.columns(3)
            m1.metric("Images", len(rows))
            m2.metric("Flagged AI", n_ai)
            m3.metric("Flagged Real", len(rows) - n_ai)

            st.subheader("Score distribution")
            hist, edges = np.histogram(preds, bins=20, range=(0, 1))
            st.bar_chart({"count": hist})

            st.subheader("Predictions")
            st.dataframe(sorted(rows, key=lambda r: -r["pred"]),
                         use_container_width=True, hide_index=True)