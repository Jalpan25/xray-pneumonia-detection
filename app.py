# app.py
import os
from pathlib import Path
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np

# Optional: use gdown for Google Drive downloads (installed via requirements)
try:
    import gdown
except Exception:
    gdown = None

MODEL_FILENAME = "best_resnet50.h5"   # local filename we will use
IMG_SIZE = (224, 224)

st.set_page_config(page_title="Pneumonia Detector", layout="centered")

# ==========================
# 🔹 Fix for "Unknown layer: Cast"
# ==========================
class CastLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        return tf.cast(inputs, tf.float32)

# ==========================
# 🔹 Download helper
# ==========================
@st.cache_resource
def download_model_from_gdrive(file_id: str, dest: str):
    if gdown is None:
        raise RuntimeError("gdown not installed. Add gdown to requirements or upload model into repo.")
    url = f"https://drive.google.com/uc?id={file_id}"
    st.info("Downloading model from Google Drive (first run, may take a minute)...")
    gdown.download(url, dest, quiet=False)
    return dest

# ==========================
# 🔹 Load model
# ==========================
@st.cache_resource
def load_model():
    if Path(MODEL_FILENAME).exists():
        model = keras.models.load_model(
            MODEL_FILENAME,
            compile=False,
            custom_objects={"Cast": CastLayer}  # 👈 Fix applied here
        )
        return model

    file_id = os.environ.get("MODEL_GDRIVE_ID") or os.environ.get("MODEL_URL")
    if file_id:
        if "drive.google.com" in file_id and gdown is not None:
            dest = MODEL_FILENAME
            gdown.download(file_id, dest, quiet=False)
            model = keras.models.load_model(
                MODEL_FILENAME,
                compile=False,
                custom_objects={"Cast": CastLayer}
            )
            return model
        if gdown is None:
            raise RuntimeError("Model not found locally and gdown is not installed.")
        download_model_from_gdrive(file_id, MODEL_FILENAME)
        model = keras.models.load_model(
            MODEL_FILENAME,
            compile=False,
            custom_objects={"Cast": CastLayer}
        )
        return model

    raise FileNotFoundError(f"{MODEL_FILENAME} not found locally and MODEL_GDRIVE_ID / MODEL_URL not set.")

# Load model (cached)
try:
    model = load_model()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

# ==========================
# 🔹 Streamlit UI
# ==========================
st.title("🩺 Pneumonia Detector (Chest X-ray)")
st.write("Upload a chest X-ray image (jpg/png). This demo is for educational purposes only — not medical advice.")

uploaded = st.file_uploader("Upload X-ray image", type=["jpg","jpeg","png"])
threshold = st.sidebar.slider("Decision threshold (for PNEUMONIA)", 0.1, 0.9, 0.5, 0.01)

def preprocess_image(pil_img: Image.Image):
    img = pil_img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

if uploaded is not None:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded image", use_column_width=True)
    x = preprocess_image(img)
    with st.spinner("Predicting..."):
        preds = model.predict(x)
    prob = float(preds[0][0])
    label = "PNEUMONIA" if prob >= threshold else "NORMAL"
    confidence = prob if label == "PNEUMONIA" else 1 - prob

    st.markdown(f"## Result: **{label}**")
    st.write(f"Confidence: **{confidence:.2f}**")
    st.caption(f"Raw model sigmoid output (prob for PNEUMONIA): {prob:.4f}")
else:
    st.info("Upload an X-ray to start.")

st.markdown("---")
st.caption("Disclaimer: This model is a demo and not a medical diagnostic tool.")
