import os
from pathlib import Path
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np

# ==========================
# 🔹 Fix for "Unknown layer: Cast"
# ==========================
class CastLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, inputs):
        return tf.cast(inputs, tf.float32)

# ==========================
# 🔹 Streamlit config
# ==========================
st.set_page_config(page_title="Pneumonia Detector (3 Models)", layout="centered")
st.title("🩺 Pneumonia Detector – Model Comparison")
st.write("Upload a chest X-ray image to compare predictions from **DenseNet121**, **MobileNetV2**, and **ResNet50**.\
         \n⚠️ This is for educational purposes only — not medical advice.")

# ==========================
# 🔹 Model files
# ==========================
MODEL_FILES = {
    "DenseNet121": "best_densenet121.h5",
    "MobileNetV2": "best_mobilenetv2.h5",
    "ResNet50": "best_resnet50.h5"
}
IMG_SIZE = (224, 224)

# ==========================
# 🔹 Load models (cached)
# ==========================
@st.cache_resource
def load_all_models():
    models = {}
    for name, filename in MODEL_FILES.items():
        if Path(filename).exists():
            models[name] = keras.models.load_model(
                filename, compile=False, custom_objects={"Cast": CastLayer}
            )
        else:
            st.error(f"❌ Model file `{filename}` not found.")
    return models

try:
    models = load_all_models()
    st.success("✅ All models loaded successfully!")
except Exception as e:
    st.error(f"Could not load models: {e}")
    st.stop()

# ==========================
# 🔹 Preprocess image
# ==========================
def preprocess_image(pil_img: Image.Image):
    img = pil_img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

# ==========================
# 🔹 Upload image
# ==========================
uploaded = st.file_uploader("Upload X-ray image", type=["jpg","jpeg","png"])
threshold = st.sidebar.slider("Decision threshold (for PNEUMONIA)", 0.1, 0.9, 0.5, 0.01)

if uploaded is not None:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded image", use_column_width=True)
    x = preprocess_image(img)

    st.markdown("### 🔍 Predictions from all models")
    results = []

    for name, model in models.items():
        with st.spinner(f"Predicting with {name}..."):
            preds = model.predict(x)
            prob = float(preds[0][0])
            label = "PNEUMONIA" if prob >= threshold else "NORMAL"
            confidence = prob if label == "PNEUMONIA" else 1 - prob
            results.append((name, label, confidence, prob))

    # Display results in a table
    st.table({
        "Model": [r[0] for r in results],
        "Prediction": [r[1] for r in results],
        "Confidence": [f"{r[2]:.2f}" for r in results],
        "Raw Prob (Pneumonia)": [f"{r[3]:.4f}" for r in results],
    })

else:
    st.info("📤 Upload an X-ray to start.")

st.markdown("---")
st.caption("Disclaimer: This model is a demo and not a medical diagnostic tool.")
