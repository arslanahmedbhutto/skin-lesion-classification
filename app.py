"""Streamlit demo: upload a dermatoscopic image and see the CNN's prediction. Run with: streamlit run app.py"""
import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from src.skin_lesion import CODES, MODEL_PATH, NAMES, load_stats, preprocess, prepare_image

SAMPLES_DIR = Path(__file__).parent / "samples"
RESULTS = json.loads((Path(__file__).parent / "model" / "results.json").read_text())

st.set_page_config(page_title="Skin Lesion Classifier", page_icon="🔬", layout="centered")


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    from tensorflow import keras
    return keras.models.load_model(MODEL_PATH), load_stats()


def predict(img: Image.Image):
    model, stats = load_model()
    x = preprocess(prepare_image(img)[None, ...], stats)
    return model.predict(x, verbose=0)[0]


# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## 🔬 Skin Lesion Classifier")
    st.caption("CNN trained on the HAM10000 dermatoscopy dataset (DermaMNIST, 28x28)")
    st.metric("Test accuracy", f"{RESULTS['test_accuracy']:.1%}")
    st.metric("Balanced accuracy", f"{RESULTS['balanced_accuracy']:.1%}",
              help="Average recall over the 7 classes, so rare classes count as much as common ones.")
    st.markdown("#### Lesion types")
    for code, name in zip(CODES, NAMES):
        st.markdown(f"**{code}**: {name}")

# ---------- Main ----------
st.title("Skin Lesion Classifier")
st.write("Upload a dermatoscopic image of a skin lesion, or pick a sample from the test set.")
st.warning("**For education only.** This model is not a medical device and must not be used for diagnosis.", icon="⚠️")

tab_upload, tab_sample = st.tabs(["📤 Upload image", "🖼️ Sample images"])
image, true_label = None, None
with tab_upload:
    file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if file:
        image = Image.open(file)
with tab_sample:
    samples = sorted(SAMPLES_DIR.glob("*.png"))
    if samples:
        choice = st.selectbox("Sample image", samples, format_func=lambda p: p.stem.replace("_", " "))
        if not file:
            image = Image.open(choice)
            true_label = choice.stem.split("_")[0]

if image is not None:
    probs = predict(image)
    order = np.argsort(probs)[::-1]
    top = order[0]

    col_img, col_res = st.columns([1, 1.4])
    with col_img:
        st.image(image.convert("RGB").resize((260, 260), Image.BICUBIC), caption="Input image")
    with col_res:
        st.markdown(f"### {NAMES[top]}")
        st.markdown(f"Predicted class **{CODES[top]}** with **{probs[top]:.1%}** confidence")
        if true_label:
            correct = true_label == CODES[top]
            st.markdown(f"True label: **{true_label}** {'✅' if correct else '❌'}")
        st.markdown("##### Top 3 predictions")
        for i in order[:3]:
            st.progress(float(probs[i]), text=f"{CODES[i]} - {NAMES[i]}: {probs[i]:.1%}")
