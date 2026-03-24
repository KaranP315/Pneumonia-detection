import os
import sys
import numpy as np
from PIL import Image

import streamlit as st

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.predict import predict, LABELS
from src.preprocess import IMG_SIZE
from src.gradcam import make_gradcam_heatmap, overlay_heatmap
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess

DEFAULT_MODEL = os.path.join(PROJECT_ROOT, "models", "efficientnet_transfer.h5")


@st.cache_resource
def load_model_cached(path: str):
    from tensorflow.keras.models import load_model
    return load_model(path)


def is_efficientnet_model(path: str) -> bool:
    return "efficientnet" in os.path.basename(path).lower()


# ─── Page setup ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pneumonia Detector",
    layout="centered",
)

st.title("Pneumonia Detection from Chest X-Rays")
st.markdown(
    "Upload a chest X-ray image and the model will classify it as "
    "**Normal** or **Pneumonia**, with a confidence score and a "
    "Grad-CAM heatmap highlighting the relevant regions."
)

st.divider()

model_path = DEFAULT_MODEL

# ─── File uploader ────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Choose a chest X-ray image",
    type=["jpg", "jpeg", "png"],
)

if uploaded is not None:
    # Display the uploaded image
    pil_img = Image.open(uploaded).convert("RGB")
    st.image(pil_img, caption="Uploaded X-ray", use_container_width=True)

    # Load the model
    if not os.path.isfile(model_path):
        st.error(
            f"Model file not found at `{model_path}`. "
            "Train a model first (`python -m src.train`) or update the path."
        )
        st.stop()

    model = load_model_cached(model_path)

    # Preprocess (must match what the model was trained on)
    img_resized = pil_img.resize(IMG_SIZE)
    img_raw = np.array(img_resized, dtype=np.uint8)      # keep for display
    img_float = np.array(img_resized, dtype=np.float32)

    if is_efficientnet_model(model_path):
        img_preprocessed = effnet_preprocess(img_float.copy())
    else:
        img_preprocessed = img_float / 255.0

    img_batch = np.expand_dims(img_preprocessed, axis=0)

    # Predict
    with st.spinner("Running inference…"):
        label, confidence = predict(model, img_batch)

    # ── Results ────────────────────────────────────────────────────────
    st.subheader("Prediction")

    col1, col2 = st.columns(2)
    with col1:
        colour = "🟢" if label == "Normal" else "🔴"
        st.metric(label="Diagnosis", value=f"{colour} {label}")
    with col2:
        st.metric(label="Confidence", value=f"{confidence:.1%}")

    st.progress(confidence)

    # ── Grad-CAM ───────────────────────────────────────────────────────
    st.subheader("Grad-CAM Heatmap")
    st.caption(
        "The heatmap highlights areas the model focused on. "
        "Bright regions contributed most to the prediction."
    )

    with st.spinner("Generating heatmap…"):
        heatmap = make_gradcam_heatmap(model, img_batch)
        overlay = overlay_heatmap(heatmap, img_raw)

    cam_col1, cam_col2 = st.columns(2)
    with cam_col1:
        st.image(img_raw, caption="Original", use_container_width=True)
    with cam_col2:
        st.image(overlay, caption="Grad-CAM Overlay", use_container_width=True)

else:
    st.info("Upload an X-ray image")

st.divider()
st.caption("This tool provides predictions for informational purposes ")
