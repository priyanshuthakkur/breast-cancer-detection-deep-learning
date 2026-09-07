"""Streamlit demo: upload a mammogram crop, get a benign/malignant prediction.

Run with:  streamlit run app/streamlit_app.py

This is a portfolio demo, not a diagnostic tool — see the disclaimer in the
app and in the README. The model classifies pre-cropped lesion patches (the
CBIS-DDSM "ROI" crops), not full mammogram scans.
"""

import sys
from pathlib import Path

import streamlit as st
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.predict import load_trained_model, predict_image  # noqa: E402

st.set_page_config(page_title="Breast Cancer Mammogram Classifier", page_icon="🩺")

st.title("Mammogram Mass Classifier")
st.caption("ResNet-50 transfer-learning model trained on CBIS-DDSM · MSc dissertation project")

st.warning(
    "**Not a diagnostic tool.** This is a portfolio demo of a research model "
    "(64.8% validation accuracy / 65.8% F1 — see README for full results and "
    "limitations). It must never be used for real clinical decisions.",
    icon="⚠️",
)


@st.cache_resource
def get_model_cached():
    return load_trained_model()


model, device, weights_loaded = get_model_cached()

if not weights_loaded:
    st.error(
        "No trained checkpoint found at `models/best_model.pth`. Run "
        "`python run_train.py` first, or drop a trained checkpoint into "
        "`models/` — see README > Getting started.",
    )

uploaded = st.file_uploader("Upload a mammogram lesion crop (JPEG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = Image.open(uploaded)
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded image", use_container_width=True)

    if weights_loaded:
        with st.spinner("Classifying..."):
            result = predict_image(model, device, image)

        with col2:
            st.metric("Prediction", result["label"], f"{result['confidence']:.1%} confidence")
            st.bar_chart(result["probabilities"])
