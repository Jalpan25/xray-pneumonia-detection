import os
from pathlib import Path
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import pandas as pd

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
st.set_page_config(page_title="Advanced Pneumonia Detector", layout="wide")
st.title("🩺 Advanced Pneumonia Detection - Multi-Model Comparison")
st.markdown("""
**Compare predictions from multiple deep learning models** for pneumonia detection in chest X-rays.
⚠️ **Disclaimer**: This is for educational and research purposes only — not for medical diagnosis.
""")

# ==========================
# 🔹 Model files configuration
# ==========================
MODEL_FILES = {
    "DenseNet121": {"file": "best_densenet121.h5", "color": "#FF6B6B", "description": "Dense connectivity between layers"},
    "MobileNetV2": {"file": "best_mobilenetv2.h5", "color": "#4ECDC4", "description": "Lightweight mobile-optimized architecture"},
    "ResNet50": {"file": "best_resnet50.h5", "color": "#45B7D1", "description": "Residual connections for deeper networks"},
}

IMG_SIZE = (224, 224)

# ==========================
# 🔹 Load models (cached)
# ==========================
@st.cache_resource
def load_all_models():
    models = {}
    model_info = {}
    
    for name, config in MODEL_FILES.items():
        filename = config["file"]
        if Path(filename).exists():
            try:
                models[name] = keras.models.load_model(
                    filename, compile=False, custom_objects={"Cast": CastLayer}
                )
                model_info[name] = {
                    "loaded": True,
                    "color": config["color"],
                    "description": config["description"]
                }
                st.success(f"✅ {name} loaded successfully!")
            except Exception as e:
                st.warning(f"⚠️ Failed to load {name}: {str(e)}")
                model_info[name] = {"loaded": False, "error": str(e)}
        else:
            st.info(f"📁 Model file `{filename}` not found for {name}")
            model_info[name] = {"loaded": False, "error": "File not found"}
    
    return models, model_info

# ==========================
# 🔹 Preprocess image
# ==========================
def preprocess_image(pil_img: Image.Image):
    """Preprocess image for model prediction"""
    img = pil_img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

# ==========================
# 🔹 Calculate ensemble prediction
# ==========================
def calculate_ensemble(results, method="average"):
    """Calculate ensemble prediction from multiple models"""
    if not results:
        return None, None
    
    probs = [r["raw_prob"] for r in results]
    
    if method == "average":
        ensemble_prob = np.mean(probs)
    elif method == "weighted":
        # Weight by confidence (you can modify weights based on model performance)
        weights = [0.35, 0.35, 0.3][:len(probs)]  # Adjust based on your models
        ensemble_prob = np.average(probs, weights=weights)
    elif method == "voting":
        votes = [1 if p >= 0.5 else 0 for p in probs]
        ensemble_prob = np.mean(votes)
    
    return ensemble_prob, "PNEUMONIA" if ensemble_prob >= 0.5 else "NORMAL"

# ==========================
# 🔹 Sidebar configuration
# ==========================
st.sidebar.header("🎛️ Configuration")
threshold = st.sidebar.slider("Decision Threshold", 0.1, 0.9, 0.5, 0.01)
ensemble_method = st.sidebar.selectbox("Ensemble Method", ["average", "weighted", "voting"])
show_details = st.sidebar.checkbox("Show Model Details", True)

# ==========================
# 🔹 Load models
# ==========================
try:
    models, model_info = load_all_models()
    loaded_models = {k: v for k, v in models.items()}
    
    if not loaded_models:
        st.error("❌ No models could be loaded. Please check your model files.")
        st.stop()
    
    st.sidebar.success(f"🎯 {len(loaded_models)} models loaded")
    
except Exception as e:
    st.error(f"Could not load models: {e}")
    st.stop()

# ==========================
# 🔹 Model information display
# ==========================
if show_details:
    st.header("📋 Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Loaded Models")
        for name, info in model_info.items():
            if info.get("loaded"):
                st.markdown(f"✅ **{name}**: {info.get('description', 'N/A')}")
            else:
                st.markdown(f"❌ **{name}**: {info.get('error', 'Unknown error')}")
    
    with col2:
        st.subheader("Model Architecture Comparison")
        arch_data = {
            "Model": ["DenseNet121", "MobileNetV2", "ResNet50"],
            "Parameters (M)": [8.1, 3.4, 25.6],
            "Depth": [121, 154, 50]
        }
        st.dataframe(pd.DataFrame(arch_data))

# ==========================
# 🔹 Main prediction interface
# ==========================
st.header("📤 Upload X-ray Image")

uploaded = st.file_uploader(
    "Choose a chest X-ray image", 
    type=["jpg", "jpeg", "png", "bmp"],
    help="Upload a clear chest X-ray image for pneumonia detection"
)

if uploaded is not None:
    # Display uploaded image
    col1, col2 = st.columns([1, 2])
    
    with col1:
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded X-ray", use_column_width=True)
        st.info(f"Image size: {img.size}")
    
    with col2:
        st.subheader("🔍 Processing Results")
        
        # Preprocess image
        x = preprocess_image(img)
        results = []
        
        # Get predictions from all loaded models
        progress_bar = st.progress(0)
        
        for i, (name, model) in enumerate(loaded_models.items()):
            with st.spinner(f"Predicting with {name}..."):
                preds = model.predict(x, verbose=0)
                prob = float(preds[0][0])
                label = "PNEUMONIA" if prob >= threshold else "NORMAL"
                confidence = prob if label == "PNEUMONIA" else 1 - prob
                
                results.append({
                    "model": name,
                    "prediction": label,
                    "confidence": confidence,
                    "raw_prob": prob,
                    "color": model_info[name]["color"]
                })
            
            progress_bar.progress((i + 1) / len(loaded_models))
        
        progress_bar.empty()
    
    # ==========================
    # 🔹 Results visualization
    # ==========================
    st.header("📊 Prediction Results")
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Summary", "📈 Visualizations", "🎯 Ensemble", "📝 Details"])
    
    with tab1:
        # Summary table
        df_results = pd.DataFrame([
            {
                "Model": r["model"],
                "Prediction": r["prediction"],
                "Confidence": f"{r['confidence']:.2%}",
                "Pneumonia Probability": f"{r['raw_prob']:.4f}",
                "Status": "🔴 PNEUMONIA" if r["prediction"] == "PNEUMONIA" else "🟢 NORMAL"
            }
            for r in results
        ])
        
        st.dataframe(df_results, use_container_width=True)
        
        # Quick stats
        col1, col2, col3, col4 = st.columns(4)
        pneumonia_count = sum(1 for r in results if r["prediction"] == "PNEUMONIA")
        normal_count = len(results) - pneumonia_count
        avg_confidence = np.mean([r["confidence"] for r in results])
        
        col1.metric("Total Models", len(results))
        col2.metric("Pneumonia Predictions", pneumonia_count)
        col3.metric("Normal Predictions", normal_count)
        col4.metric("Average Confidence", f"{avg_confidence:.2%}")
    
    with tab2:
        # Simple visualizations using Streamlit components
        st.subheader("📊 Model Probability Comparison")
        
        # Create a simple bar chart using Streamlit
        prob_data = pd.DataFrame({
            'Model': [r["model"] for r in results],
            'Pneumonia_Probability': [r["raw_prob"] for r in results]
        })
        
        st.bar_chart(prob_data.set_index('Model'))
        
        # Add threshold line information
        st.info(f"🎯 Decision Threshold: {threshold:.2f}")
        
        # Show individual model probabilities
        col1, col2, col3 = st.columns(3)
        
        for i, r in enumerate(results):
            with [col1, col2, col3][i]:
                st.metric(
                    r["model"], 
                    f"{r['raw_prob']:.3f}",
                    delta=f"{r['raw_prob'] - threshold:.3f}",
                    delta_color="inverse"
                )
    
    with tab3:
        # Ensemble predictions
        ensemble_prob, ensemble_pred = calculate_ensemble(results, ensemble_method)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Ensemble Prediction")
            st.metric(
                "Ensemble Result", 
                ensemble_pred,
                delta=f"{ensemble_prob:.2%} probability"
            )
            
            st.info(f"**Method**: {ensemble_method.title()}")
            
            # Agreement analysis
            unanimous = len(set(r["prediction"] for r in results)) == 1
            majority_pred = max(set(r["prediction"] for r in results), 
                              key=lambda x: sum(1 for r in results if r["prediction"] == x))
            
            st.write(f"**Agreement**: {'Unanimous' if unanimous else f'Majority: {majority_pred}'}")
        
        with col2:
            st.subheader("📊 Ensemble vs Individual")
            
            # Create comparison data
            individual_probs = [r["raw_prob"] for r in results]
            comparison_data = pd.DataFrame({
                'Source': ["Ensemble"] + [r["model"] for r in results],
                'Probability': [ensemble_prob] + individual_probs
            })
            
            st.bar_chart(comparison_data.set_index('Source'))
            
            # Show numeric values
            st.write("**Probability Values:**")
            for i, row in comparison_data.iterrows():
                st.write(f"- {row['Source']}: {row['Probability']:.3f}")
    
    with tab4:
        # Detailed analysis
        st.subheader("📝 Detailed Analysis")
        
        for r in results:
            with st.expander(f"{r['model']} - {r['prediction']}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Prediction**: {r['prediction']}")
                    st.write(f"**Raw Probability**: {r['raw_prob']:.4f}")
                    st.write(f"**Confidence**: {r['confidence']:.2%}")
                    
                    # Threshold comparison
                    if r['raw_prob'] >= threshold:
                        st.error(f"🔴 Above threshold ({threshold})")
                    else:
                        st.success(f"🟢 Below threshold ({threshold})")
                
                with col2:
                    st.write(f"**Probability Breakdown**:")
                    
                    # Visual probability using progress bar
                    st.write(f"Pneumonia: {r['raw_prob']:.1%}")
                    st.progress(r['raw_prob'])
                    
                    st.write(f"Normal: {1-r['raw_prob']:.1%}")
                    st.progress(1-r['raw_prob'])
                    
                    # Model characteristics
                    st.write(f"**Architecture**: {model_info[r['model']]['description']}")

else:
    st.info("📤 Upload a chest X-ray image to start the analysis")
    
    # Show example predictions or model capabilities
    st.header("🎯 Model Capabilities")
    
    capabilities_data = {
        "Feature": [
            "Multi-model Comparison",
            "Ensemble Predictions",
            "Confidence Analysis",
            "Interactive Visualizations",
            "Detailed Metrics"
        ],
        "Description": [
            "Compare predictions from multiple CNN architectures",
            "Combine model predictions using different ensemble methods",
            "Analyze prediction confidence and agreement between models",
            "Interactive charts and visualizations for better understanding",
            "Comprehensive metrics including probability scores and confidence levels"
        ]
    }
    
    st.dataframe(pd.DataFrame(capabilities_data), use_container_width=True)

# ==========================
# 🔹 Footer
# ==========================
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
<p><strong>⚠️ Medical Disclaimer</strong></p>
<p>This application is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of qualified healthcare providers.</p>
</div>
""", unsafe_allow_html=True)